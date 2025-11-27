"""
Manifold Alignment Model

Orthogonal Transform + Contrastive Alignment 기반의 멀티모달 manifold alignment
각 모달리티(Text, Image, Layout)를 공통 absolute semantic space로 매핑

기존 파이프라인에서 추출된 h_text, h_image, h_layout을 입력으로 받아
alignment 및 downstream task 수행
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable


class OrthoLinear(nn.Module):
    """
    Orthogonal Linear Transform

    Weight를 orthogonal로 초기화하고, ortho_reg()로 정규화 term 제공
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Bias 없이 선형 변환을 두어 순수한 직교 변환이 되도록 구성
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # Orthogonal initialization
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim] 또는 [B, L, in_dim]
        Returns:
            [B, out_dim] 또는 [B, L, out_dim]
        """
        if x.dim() == 2:
            return self.linear(x)
        elif x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.view(B * L, D)
            out_flat = self.linear(x_flat)
            return out_flat.view(B, L, -1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

    def ortho_reg(self) -> torch.Tensor:
        """
        Orthogonal regularization: ||WᵀW - I||²

        Returns:
            MSE between WᵀW and identity matrix
        """
        W = self.linear.weight  # [out_dim, in_dim]
        WT_W = W.T @ W  # [in_dim, in_dim]
        I = torch.eye(WT_W.size(0), device=W.device, dtype=W.dtype)
        return F.mse_loss(WT_W, I)


class LinearProjector(nn.Module):
    """Simple Linear + ReLU projector (non-orthogonal baseline)."""

    def __init__(self, in_dim: int, out_dim: int, use_activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.use_activation = use_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            out = self.linear(x)
        elif x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.view(B * L, D)
            out_flat = self.linear(x_flat)
            out = out_flat.view(B, L, -1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        return F.relu(out) if self.use_activation else out

    def ortho_reg(self) -> torch.Tensor:
        device = self.linear.weight.device
        return torch.tensor(0.0, device=device)


class AttentionPooler(nn.Module):
    """Learnable attention pooling over region sequences."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq: [B, L, D], mask: [B, L]
        logits = self.proj(seq).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        weights = weights / denom
        return torch.sum(seq * weights.unsqueeze(-1), dim=1)


class ManifoldAlignModel(nn.Module):
    """
    Manifold Alignment Model

    기존 파이프라인에서 추출된 세 모달리티 embedding (h_text, h_image, h_layout)을
    orthogonal transform으로 공통 absolute semantic space로 매핑
    """

    def __init__(
        self,
        text_dim: int = 3584,
        image_dim: int = 3584,
        layout_dim: int = 3584,
        z_dim: int = 512,
        num_labels: int = None,
        projector_cls: Optional[Callable[[int, int], nn.Module]] = None,
        pooling_mode: str = "mean"
    ):
        """
        Args:
            text_dim: h_text 차원 (기존 파이프라인 출력)
            image_dim: h_image 차원 (기존 파이프라인 출력)
            layout_dim: h_layout 차원 (기존 파이프라인 출력)
            z_dim: Absolute semantic space 차원
            num_labels: 분류 클래스 수 (None이면 classifier 없음)
            projector_cls: modality projector로 사용할 모듈
        """
        super().__init__()

        self.z_dim = z_dim
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode

        projector_cls = projector_cls or OrthoLinear

        # modality → absolute space projectors
        self.T_text = projector_cls(text_dim, z_dim)
        self.T_image = projector_cls(image_dim, z_dim)
        self.T_layout = projector_cls(layout_dim, z_dim)

        if self.pooling_mode == "attention":
            self.text_pooler = AttentionPooler(text_dim)
            self.image_pooler = AttentionPooler(image_dim)
            self.layout_pooler = AttentionPooler(layout_dim)
        else:
            self.text_pooler = None
            self.image_pooler = None
            self.layout_pooler = None

        # Classifier: fused representation -> labels (optional)
        if num_labels is not None:
            self.classifier = nn.Sequential(
                nn.Linear(z_dim * 3, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, num_labels)
            )
        else:
            self.classifier = None

    def forward(
        self,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        h_layout: torch.Tensor,
        text_seq: Optional[torch.Tensor] = None,
        text_seq_mask: Optional[torch.Tensor] = None,
        image_seq: Optional[torch.Tensor] = None,
        image_seq_mask: Optional[torch.Tensor] = None,
        layout_seq: Optional[torch.Tensor] = None,
        layout_seq_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass

        Args:
            h_text: [B, text_dim] - 기존 파이프라인에서 추출된 text embedding
            h_image: [B, image_dim] - 기존 파이프라인에서 추출된 image embedding
            h_layout: [B, layout_dim] - 기존 파이프라인에서 추출된 layout embedding

        Returns:
            dict with:
                - "z_text": [B, z_dim]
                - "z_image": [B, z_dim]
                - "z_layout": [B, z_dim]
                - "logits": [B, num_labels] (if classifier exists)
        """
        poolers = {
            "text": self.text_pooler,
            "image": self.image_pooler,
            "layout": self.layout_pooler,
        }

        h_text = self._pool_modal(
            h_text, text_seq, text_seq_mask,
            pooler=poolers["text"] if self.pooling_mode == "attention" else None
        )
        h_image = self._pool_modal(
            h_image, image_seq, image_seq_mask,
            pooler=poolers["image"] if self.pooling_mode == "attention" else None
        )
        h_layout = self._pool_modal(
            h_layout, layout_seq, layout_seq_mask,
            pooler=poolers["layout"] if self.pooling_mode == "attention" else None
        )

        # Orthogonal transform으로 absolute space로 매핑
        z_t = self.T_text(h_text)    # [B, z_dim]
        z_i = self.T_image(h_image)  # [B, z_dim]
        z_l = self.T_layout(h_layout)  # [B, z_dim]

        result = {
            "z_text": z_t,
            "z_image": z_i,
            "z_layout": z_l,
        }

        # Classification (optional)
        if self.classifier is not None:
            z_cat = torch.cat([z_t, z_i, z_l], dim=-1)  # [B, 3*z_dim]
            logits = self.classifier(z_cat)  # [B, num_labels]
            result["logits"] = logits

        return result

    def ortho_reg(self) -> torch.Tensor:
        """
        모든 OrthoLinear의 orthogonal regularization 평균
        """
        regs = [
            self.T_text.ortho_reg(),
            self.T_image.ortho_reg(),
            self.T_layout.ortho_reg()
        ]

        total = sum(regs)
        return total / max(len(regs), 1)

    def _pool_modal(
        self,
        base: torch.Tensor,
        seq: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        pooler: Optional[AttentionPooler]
    ) -> torch.Tensor:
        if seq is None or mask is None:
            return base

        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if pooler is not None:
            pooled = pooler(seq, mask)
        else:
            pooled = self._masked_mean(seq, mask)

        valid = mask.any(dim=1, keepdim=True)
        if base.dim() == 1:
            base = base.unsqueeze(0)
        pooled = torch.where(valid, pooled, base)
        return pooled.squeeze(0) if pooled.size(0) == 1 else pooled

    def _masked_mean(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        summed = (seq * weights).sum(dim=1)
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1e-6)
        mean = summed / denom
        zero_rows = (mask.sum(dim=1, keepdim=True) == 0)
        if zero_rows.any():
            mean = mean.masked_fill(zero_rows, 0.0)
        return mean


class ConcatBaselineModel(nn.Module):
    """Baseline: concat anchor embeddings without projectors."""

    def __init__(
        self,
        text_dim: int = 3584,
        image_dim: int = 3584,
        layout_dim: int = 3584,
        z_dim: int = 512,
        num_labels: int = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        concat_dim = text_dim + image_dim + layout_dim

        if num_labels is not None:
            self.classifier = nn.Sequential(
                nn.Linear(concat_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, num_labels)
            )
        else:
            self.classifier = None

    def forward(
        self,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        h_layout: torch.Tensor,
        **kwargs
    ) -> Dict:
        result = {
            "z_text": h_text,
            "z_image": h_image,
            "z_layout": h_layout,
        }

        if self.classifier is not None:
            z_cat = torch.cat([h_text, h_image, h_layout], dim=-1)
            result["logits"] = self.classifier(z_cat)

        return result

    def ortho_reg(self) -> torch.Tensor:
        return self._zero


def contrastive_alignment(
    z_text: torch.Tensor,
    z_image: torch.Tensor,
    z_layout: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
    image_mask: Optional[torch.Tensor] = None,
    layout_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Contrastive Alignment Loss (InfoNCE style)

    같은 인덱스의 샘플은 positive pair, 나머지는 negative

    Args:
        z_text: [B, z_dim]
        z_image: [B, z_dim]
        z_layout: [B, z_dim]
        temperature: softmax temperature

    Returns:
        Average contrastive loss across all modality pairs
    """
    def info_nce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss between two modalities"""
        B = a.size(0)

        # L2 normalize
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        # Similarity matrix
        sim = a @ b.T / temperature  # [B, B]

        # Target: diagonal (same index = positive pair)
        target = torch.arange(B, device=a.device)

        # Symmetric loss
        loss_ab = F.cross_entropy(sim, target)
        loss_ba = F.cross_entropy(sim.T, target)

        return (loss_ab + loss_ba) / 2.0

    def pair_loss(a, b, mask_a, mask_b) -> Optional[torch.Tensor]:
        valid = mask_a & mask_b
        if valid.sum().item() < 2:
            return None
        return info_nce(a[valid], b[valid])

    device = z_layout.device
    B = z_layout.size(0)
    if text_mask is None:
        text_mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        text_mask = text_mask.to(device)
    if image_mask is None:
        image_mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        image_mask = image_mask.to(device)
    if layout_mask is None:
        layout_mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        layout_mask = layout_mask.to(device)

    losses = []
    for (a, b, ma, mb) in [
        (z_text, z_image, text_mask, image_mask),
        (z_text, z_layout, text_mask, layout_mask),
        (z_image, z_layout, image_mask, layout_mask),
    ]:
        loss = pair_loss(a, b, ma, mb)
        if loss is not None:
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)

    return sum(losses) / len(losses)


def region_contrastive_alignment(
    layout_seq: torch.Tensor,
    layout_seq_mask: torch.Tensor,
    text_seq: Optional[torch.Tensor] = None,
    text_seq_mask: Optional[torch.Tensor] = None,
    text_to_layout: Optional[torch.Tensor] = None,
    image_seq: Optional[torch.Tensor] = None,
    image_seq_mask: Optional[torch.Tensor] = None,
    image_to_layout: Optional[torch.Tensor] = None,
    temperature: float = 0.1
) -> torch.Tensor:
    """Region-level InfoNCE using layout indices as anchors."""

    losses = []

    if text_seq is not None and text_seq_mask is not None and text_to_layout is not None:
        loss = _region_info_nce(
            text_seq, text_seq_mask, text_to_layout,
            layout_seq, layout_seq_mask, temperature
        )
        if loss is not None:
            losses.append(loss)

    if image_seq is not None and image_seq_mask is not None and image_to_layout is not None:
        loss = _region_info_nce(
            image_seq, image_seq_mask, image_to_layout,
            layout_seq, layout_seq_mask, temperature
        )
        if loss is not None:
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=layout_seq.device)

    return sum(losses) / len(losses)


def _region_info_nce(
    query_seq: torch.Tensor,
    query_mask: torch.Tensor,
    mapping: torch.Tensor,
    key_seq: torch.Tensor,
    key_mask: torch.Tensor,
    temperature: float
) -> Optional[torch.Tensor]:
    device = query_seq.device
    B, Lq, D = query_seq.shape
    Lk = key_seq.shape[1]

    valid = query_mask & (mapping >= 0)
    if valid.sum().item() == 0:
        return None

    query_flat = query_seq[valid]
    batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, Lq)
    batch_flat = batch_ids[valid]
    mapping_flat = mapping[valid]

    key_flat = key_seq.view(B * Lk, D)
    key_mask_flat = key_mask.view(B * Lk)

    sim = query_flat @ key_flat.T / temperature
    sim[:, ~key_mask_flat] = -1e9

    targets = batch_flat * Lk + mapping_flat
    loss = F.cross_entropy(sim, targets)
    return loss


def train_epoch(
    model: ManifoldAlignModel,
    dataloader,
    optimizer,
    lambda_align: float = 1.0,
    lambda_ortho: float = 1e-3,
    lambda_region: float = 0.0,
    device: str = "cuda"
) -> float:
    """
    Single training epoch

    dataloader는 다음 형태의 batch를 반환해야 함:
        - "h_text": [B, text_dim]
        - "h_image": [B, image_dim]
        - "h_layout": [B, layout_dim]
        - (선택) "*_seq", "*_seq_mask", "*_to_layout"
        - "labels": [B] (optional, for classification)

    Args:
        model: ManifoldAlignModel instance
        dataloader: PyTorch DataLoader
        optimizer: Optimizer
        lambda_align: Weight for contrastive alignment loss
        lambda_ortho: Weight for orthogonal regularization
        lambda_region: Weight for region-level alignment loss
        device: Device to use

    Returns:
        Average total loss for the epoch
    """
    model.train()
    total_loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move batch to device
        h_text = batch["h_text"].to(device)
        h_image = batch["h_image"].to(device)
        h_layout = batch["h_layout"].to(device)
        text_seq = batch.get("text_seq")
        if text_seq is not None:
            text_seq = text_seq.to(device)
        text_seq_mask = batch.get("text_seq_mask")
        if text_seq_mask is not None:
            text_seq_mask = text_seq_mask.to(device)
        image_seq = batch.get("image_seq")
        if image_seq is not None:
            image_seq = image_seq.to(device)
        image_seq_mask = batch.get("image_seq_mask")
        if image_seq_mask is not None:
            image_seq_mask = image_seq_mask.to(device)
        layout_seq = batch.get("layout_seq")
        if layout_seq is not None:
            layout_seq = layout_seq.to(device)
        layout_seq_mask = batch.get("layout_seq_mask")
        if layout_seq_mask is not None:
            layout_seq_mask = layout_seq_mask.to(device)
        text_to_layout = batch.get("text_to_layout")
        if text_to_layout is not None:
            text_to_layout = text_to_layout.to(device)
        image_to_layout = batch.get("image_to_layout")
        if image_to_layout is not None:
            image_to_layout = image_to_layout.to(device)
        text_mask = batch.get("text_mask")
        image_mask = batch.get("image_mask")
        layout_mask = batch.get("layout_mask")
        if text_mask is not None:
            text_mask = text_mask.to(device)
        if image_mask is not None:
            image_mask = image_mask.to(device)
        if layout_mask is not None:
            layout_mask = layout_mask.to(device)

        # Forward pass
        out = model(
            h_text,
            h_image,
            h_layout,
            text_seq=text_seq,
            text_seq_mask=text_seq_mask,
            image_seq=image_seq,
            image_seq_mask=image_seq_mask,
            layout_seq=layout_seq,
            layout_seq_mask=layout_seq_mask
        )

        # Contrastive alignment loss
        align_loss = contrastive_alignment(
            out["z_text"],
            out["z_image"],
            out["z_layout"],
            text_mask=text_mask,
            image_mask=image_mask,
            layout_mask=layout_mask
        )

        # Orthogonal regularization
        ortho_loss = model.ortho_reg()

        # Region-level contrastive loss
        region_loss = torch.tensor(0.0, device=device)
        if lambda_region > 0.0 and layout_seq is not None and layout_seq_mask is not None:
            region_loss = region_contrastive_alignment(
                layout_seq=layout_seq,
                layout_seq_mask=layout_seq_mask,
                text_seq=text_seq,
                text_seq_mask=text_seq_mask,
                text_to_layout=text_to_layout,
                image_seq=image_seq,
                image_seq_mask=image_seq_mask,
                image_to_layout=image_to_layout
            )

        # Total loss
        total_loss = (
            lambda_align * align_loss +
            lambda_ortho * ortho_loss +
            lambda_region * region_loss
        )

        # Task loss (if labels provided)
        if "labels" in batch and model.classifier is not None:
            labels = batch["labels"].to(device)
            task_loss = F.cross_entropy(out["logits"], labels)
            total_loss = total_loss + task_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        num_batches += 1

    return total_loss_sum / max(num_batches, 1)
