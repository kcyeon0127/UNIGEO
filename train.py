"""
UniDoc Training Script

기존 파이프라인으로 embedding 추출 후 ManifoldAlignModel 학습
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any
import json
from tqdm import tqdm

from unidoc.detection import DetectionPipeline, collate_embeddings
from unidoc.model import (
    ManifoldAlignModel,
    contrastive_alignment,
    region_contrastive_alignment,
    train_epoch,
)


class EmbeddingDataset(Dataset):
    """문서별 region embedding을 학습 입력으로 변환."""

    def __init__(
        self,
        embeddings_list: List[dict],
        labels: Optional[List[int]] = None,
        hidden_dim: int = 3584,
        pooling_mode: str = "mean",
        max_text_regions: int = 32,
        max_image_regions: int = 32,
        max_layout_regions: int = 128,
    ):
        self.embeddings_list = embeddings_list
        self.labels = labels
        self.hidden_dim = hidden_dim
        self.pooling_mode = pooling_mode
        self.max_text_regions = max_text_regions
        self.max_image_regions = max_image_regions
        self.max_layout_regions = max_layout_regions
        self.trunc_stats = {"text": 0, "image": 0, "layout": 0}
        self._trunc_warned = False

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        emb = self.embeddings_list[idx]

        layout_seq, layout_seq_mask = self._prepare_sequence(
            emb.get("h_layout"), self.max_layout_regions, stat_key="layout"
        )
        text_seq, text_seq_mask = self._prepare_sequence(
            emb.get("h_text"), self.max_text_regions, stat_key="text"
        )
        image_seq, image_seq_mask = self._prepare_sequence(
            emb.get("h_image"), self.max_image_regions, stat_key="image"
        )

        h_layout = self._aggregate(layout_seq, layout_seq_mask)
        h_text = self._aggregate(text_seq, text_seq_mask)
        h_image = self._aggregate(image_seq, image_seq_mask)

        text_exists = text_seq_mask.any().item()
        image_exists = image_seq_mask.any().item()

        item = {
            "h_text": h_text,
            "h_image": h_image,
            "h_layout": h_layout,
            "text_mask": torch.tensor(text_exists, dtype=torch.bool),
            "image_mask": torch.tensor(image_exists, dtype=torch.bool),
            "layout_mask": torch.tensor(True, dtype=torch.bool),
            "text_seq": text_seq,
            "text_seq_mask": text_seq_mask,
            "text_to_layout": self._prepare_indices(
                emb.get("text_indices"), self.max_text_regions
            ),
            "image_seq": image_seq,
            "image_seq_mask": image_seq_mask,
            "image_to_layout": self._prepare_indices(
                emb.get("image_indices"), self.max_image_regions
            ),
            "layout_seq": layout_seq,
            "layout_seq_mask": layout_seq_mask,
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    def _prepare_sequence(
        self,
        tensor: Optional[torch.Tensor],
        max_len: int,
        stat_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.zeros(max_len, self.hidden_dim, dtype=torch.float32)
        mask = torch.zeros(max_len, dtype=torch.bool)

        if tensor is None:
            return seq, mask

        tensor = tensor.detach().cpu().float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        orig_len = tensor.size(0)
        length = min(orig_len, max_len)
        if length > 0:
            seq[:length] = tensor[:length]
            mask[:length] = True

        if stat_key and orig_len > max_len:
            self.trunc_stats[stat_key] += 1
            if not self._trunc_warned:
                print(f"[EmbeddingDataset] Truncated {stat_key} sequences encountered.")
                self._trunc_warned = True

        return seq, mask

    def _aggregate(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            weights = mask.float()
            summed = (seq * weights.unsqueeze(-1)).sum(dim=0)
            denom = weights.sum().clamp(min=1e-6)
            return (summed / denom).float()
        return torch.zeros(self.hidden_dim, dtype=torch.float32)

    def _prepare_indices(
        self,
        indices: Optional[List[int]],
        max_len: int
    ) -> torch.Tensor:
        idx_tensor = torch.full((max_len,), -1, dtype=torch.long)
        if not indices:
            return idx_tensor
        tensor = torch.tensor(indices, dtype=torch.long)
        length = min(tensor.size(0), max_len)
        idx_tensor[:length] = tensor[:length]
        return idx_tensor


def extract_embeddings_from_pdfs(
    pdf_paths: List[Path],
    pipeline: DetectionPipeline,
    pages: Optional[List[int]] = None,
    show_progress: bool = False,
    progress_desc: str = "processing"
) -> List[dict]:
    """
    PDF 리스트에서 embedding 추출

    Args:
        pdf_paths: PDF 파일 경로 리스트
        pipeline: DetectionPipeline 인스턴스
        pages: 처리할 페이지 (None이면 첫 페이지만)

    Returns:
        collate_embeddings() 출력 리스트
    """
    all_embeddings = []

    iterator = pdf_paths
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(pdf_paths, desc=progress_desc)

    for pdf_path in iterator:
        try:
            page_embeddings = pipeline.process_pdf(pdf_path, pages=pages)
            for page_emb in page_embeddings:
                if page_emb:  # 빈 페이지 스킵
                    collated = collate_embeddings(page_emb)
                    if collated is not None:
                        all_embeddings.append(collated)
        except Exception as e:
            import traceback
            print(f"Error processing {pdf_path}: {e}")
            traceback.print_exc()
            continue

    return all_embeddings


def train(
    train_pdf_paths: Optional[List[Path]] = None,
    train_embeddings_cache: Optional[List[str]] = None,
    train_labels: Optional[List[int]] = None,
    val_pdf_paths: Optional[List[Path]] = None,
    val_embeddings_cache: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    hidden_dim: int = 3584,
    z_dim: int = 512,
    num_labels: Optional[int] = None,
    batch_size: int = 8,
    num_epochs: int = 10,
    lr: float = 1e-4,
    lambda_align: float = 1.0,
    lambda_ortho: float = 5e-4,
    lambda_region: float = 0.01,
    pooling_mode: str = "attention",
    max_text_regions: int = 64,
    max_image_regions: int = 64,
    max_layout_regions: int = 256,
    device: str = "cuda",
    save_path: Optional[Path] = None,
    model_cls: Callable[..., ManifoldAlignModel] = ManifoldAlignModel,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> ManifoldAlignModel:
    """
    전체 학습 파이프라인

    Args:
        train_pdf_paths: 학습 PDF 경로 리스트
        train_labels: 학습 라벨 (optional)
        val_pdf_paths: 검증 PDF 경로 리스트 (optional)
        val_labels: 검증 라벨 (optional)
        hidden_dim: 기존 파이프라인 embedding 차원
        z_dim: Absolute space 차원
        num_labels: 분류 클래스 수 (None이면 alignment만)
        batch_size: 배치 크기
        num_epochs: 에폭 수
        lr: learning rate
        lambda_align: alignment loss 가중치
        lambda_ortho: orthogonal regularization 가중치
        device: 디바이스
        save_path: 모델 저장 경로
        model_cls: 학습에 사용할 모델 클래스
        model_kwargs: 모델 생성 시 추가 인자

    Returns:
        학습된 ManifoldAlignModel
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Pipeline 초기화
    train_embeddings = []
    val_embeddings = []

    if train_embeddings_cache:
        for cache_path in train_embeddings_cache:
            cache_path = Path(cache_path)
            cache_data = torch.load(cache_path)
            if isinstance(cache_data, dict) and "embeddings" in cache_data:
                train_embeddings.extend(cache_data["embeddings"])
                if "labels" in cache_data and cache_data["labels"] is not None:
                    if train_labels is None:
                        train_labels = list(cache_data["labels"])
                    else:
                        train_labels.extend(cache_data["labels"])
            else:
                train_embeddings.extend(cache_data)
        print(f"Loaded {len(train_embeddings)} cached training samples")
    elif train_pdf_paths:
        print("Initializing pipeline...")
        pipeline = DetectionPipeline()
        print(f"Extracting embeddings from {len(train_pdf_paths)} training PDFs...")
        train_embeddings = extract_embeddings_from_pdfs(train_pdf_paths, pipeline)
        print(f"Extracted {len(train_embeddings)} training samples")
    else:
        raise ValueError("Either train_pdf_paths or train_embeddings_cache must be provided")

    if val_embeddings_cache:
        for cache_path in val_embeddings_cache:
            cache_path = Path(cache_path)
            cache_data = torch.load(cache_path)
            if isinstance(cache_data, dict) and "embeddings" in cache_data:
                val_embeddings.extend(cache_data["embeddings"])
                if "labels" in cache_data and cache_data["labels"] is not None:
                    if val_labels is None:
                        val_labels = cache_data["labels"]
                    else:
                        val_labels.extend(cache_data["labels"])
            else:
                val_embeddings.extend(cache_data)
        print(f"Loaded {len(val_embeddings)} cached validation samples")
    elif val_pdf_paths:
        if 'pipeline' not in locals():
            pipeline = DetectionPipeline()
        print(f"Extracting embeddings from {len(val_pdf_paths)} validation PDFs...")
        val_embeddings = extract_embeddings_from_pdfs(val_pdf_paths, pipeline)
        print(f"Extracted {len(val_embeddings)} validation samples")
    else:
        val_embeddings = None

    # 3. Dataset & DataLoader
    train_dataset = EmbeddingDataset(
        train_embeddings,
        train_labels,
        hidden_dim=hidden_dim,
        pooling_mode=pooling_mode,
        max_text_regions=max_text_regions,
        max_image_regions=max_image_regions,
        max_layout_regions=max_layout_regions
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    if val_embeddings:
        val_dataset = EmbeddingDataset(
            val_embeddings,
            val_labels,
            hidden_dim=hidden_dim,
            pooling_mode=pooling_mode,
            max_text_regions=max_text_regions,
            max_image_regions=max_image_regions,
            max_layout_regions=max_layout_regions
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    else:
        val_loader = None

    if model_kwargs is None:
        model_kwargs = {}

    model_config = {
        "text_dim": hidden_dim,
        "image_dim": hidden_dim,
        "layout_dim": hidden_dim,
        "z_dim": z_dim,
        "num_labels": num_labels,
        "pooling_mode": pooling_mode,
    }
    model_config.update(model_kwargs)

    # 4. Model 초기화
    print(f"Initializing {model_cls.__name__}...")
    model = model_cls(**model_config)
    model.to(device)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 6. Training loop
    print(f"Starting training for {num_epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            lambda_align=lambda_align,
            lambda_ortho=lambda_ortho,
            lambda_region=lambda_region,
            device=str(device)
        )

        # Validation
        val_loss = None
        if val_loader:
            val_loss = evaluate(
                model,
                val_loader,
                lambda_align,
                lambda_ortho,
                lambda_region,
                device
            )

        # Logging
        log_msg = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        print(log_msg)

        # Save best model
        current_loss = val_loss if val_loss is not None else train_loss
        if current_loss < best_loss and save_path:
            best_loss = current_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
            }, save_path)
            print(f"  Saved best model to {save_path}")

    print("Training completed!")
    if isinstance(train_dataset, EmbeddingDataset):
        print("Train truncation stats:", train_dataset.trunc_stats)
    if val_loader and val_dataset is not None:
        print("Val truncation stats:", val_dataset.trunc_stats)
    return model


def evaluate(
    model: ManifoldAlignModel,
    dataloader: DataLoader,
    lambda_align: float = 1.0,
    lambda_ortho: float = 1e-3,
    lambda_region: float = 0.0,
    device: str = "cuda"
) -> float:
    """
    모델 평가

    Args:
        model: ManifoldAlignModel
        dataloader: 평가용 DataLoader
        lambda_align: alignment loss 가중치
        lambda_ortho: orthogonal regularization 가중치
        device: 디바이스

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Eval", leave=False)
        for batch in iterator:
            h_text = batch["h_text"].to(device)
            h_image = batch["h_image"].to(device)
            h_layout = batch["h_layout"].to(device)
            text_seq = batch.get("text_seq")
            text_seq_mask = batch.get("text_seq_mask")
            image_seq = batch.get("image_seq")
            image_seq_mask = batch.get("image_seq_mask")
            layout_seq = batch.get("layout_seq")
            layout_seq_mask = batch.get("layout_seq_mask")
            text_to_layout = batch.get("text_to_layout")
            image_to_layout = batch.get("image_to_layout")
            text_mask = batch.get("text_mask").to(device)
            image_mask = batch.get("image_mask").to(device)
            layout_mask = batch.get("layout_mask")
            layout_mask = layout_mask.to(device) if layout_mask is not None else torch.ones_like(text_mask)

            def _move(tensor):
                return tensor.to(device) if tensor is not None else None

            text_seq = _move(text_seq)
            text_seq_mask = _move(text_seq_mask)
            image_seq = _move(image_seq)
            image_seq_mask = _move(image_seq_mask)
            layout_seq = _move(layout_seq)
            layout_seq_mask = _move(layout_seq_mask)
            text_to_layout = _move(text_to_layout)
            image_to_layout = _move(image_to_layout)

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

            align_loss = contrastive_alignment(
                out["z_text"],
                out["z_image"],
                out["z_layout"],
                text_mask=text_mask,
                image_mask=image_mask,
                layout_mask=layout_mask
            )

            ortho_loss = model.ortho_reg()
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

            loss = (
                lambda_align * align_loss +
                lambda_ortho * ortho_loss +
                lambda_region * region_loss
            )

            task_loss = torch.tensor(0.0, device=device)
            if "labels" in batch and model.classifier is not None:
                labels = batch["labels"].to(device)
                task_loss = nn.functional.cross_entropy(out["logits"], labels)
                loss = loss + task_loss

            total_loss += loss.item()
            num_batches += 1

            iterator.set_postfix(
                align=align_loss.item(),
                region=region_loss.item() if lambda_region > 0 else 0.0,
                ortho=ortho_loss.item(),
                task=task_loss.item() if ("labels" in batch and model.classifier is not None) else 0.0
            )

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train UniDoc ManifoldAlignModel")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing training PDF files")
    parser.add_argument("--val_dir", type=str, default=None, help="Directory containing validation PDF files")
    parser.add_argument("--cache_train", type=str, nargs='*', default=None, help="Cached training embeddings (.pt)")
    parser.add_argument("--cache_val", type=str, nargs='*', default=None, help="Cached validation embeddings (.pt)")
    parser.add_argument("--z_dim", type=int, default=512, help="Absolute space dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="model.pt", help="Model save path")
    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "attention"], help="Pooling strategy for region embeddings")
    parser.add_argument("--max_text_regions", type=int, default=32, help="Max text regions per document")
    parser.add_argument("--max_image_regions", type=int, default=32, help="Max image regions per document")
    parser.add_argument("--max_layout_regions", type=int, default=128, help="Max layout regions per document")
    parser.add_argument("--lambda_region", type=float, default=0.0, help="Region-level contrastive weight")

    args = parser.parse_args()

    train_pdf_paths = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_pdf_paths = list(data_dir.glob("**/*.pdf"))
        print(f"Found {len(train_pdf_paths)} training PDF files")

    val_pdf_paths = None
    if args.val_dir:
        val_dir = Path(args.val_dir)
        val_pdf_paths = list(val_dir.glob("**/*.pdf"))
        print(f"Found {len(val_pdf_paths)} validation PDF files")

    if not train_pdf_paths and not args.cache_train:
        raise ValueError("Provide either --data_dir or --cache_train for training data")

    model = train(
        train_pdf_paths=train_pdf_paths,
        train_embeddings_cache=args.cache_train,
        val_pdf_paths=val_pdf_paths,
        val_embeddings_cache=args.cache_val,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=Path(args.save_path),
        pooling_mode=args.pooling_mode,
        max_text_regions=args.max_text_regions,
        max_image_regions=args.max_image_regions,
        max_layout_regions=args.max_layout_regions,
        lambda_region=args.lambda_region
    )
