"""
Layout Embedding: LayoutLMv3의 pretrained 2D position embedding 직접 활용

LayoutLMv3의 x_position_embeddings, y_position_embeddings를 사용하여
문서 구조로 이미 학습된 layout representation 추출
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
from enum import IntEnum


class RegionTypeId(IntEnum):
    """RegionType을 정수 ID로 매핑"""
    PARAGRAPH = 0
    TITLE = 1
    HEADING = 2
    TABLE = 3
    FIGURE = 4
    CAPTION = 5
    HEADER = 6
    FOOTER = 7
    PAGE_NUMBER = 8
    LIST = 9
    FOOTNOTE = 10
    UNKNOWN = 11


class LayoutLMv3LayoutEmbedding(nn.Module):
    """
    LayoutLMv3의 pretrained 2D position embedding을 직접 사용

    - bbox 좌표 (x0, y0, x1, y1)를 LayoutLMv3의 학습된 position embedding으로 변환
    - Full encoder 대신 position embedding만 추출하여 효율적
    - 문서 구조로 이미 pretrained된 representation
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        hidden_size: int = 3584,
        use_projection: bool = True,
        freeze_embeddings: bool = True,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: LayoutLMv3 모델 이름
            hidden_size: 최종 출력 차원 (Qwen2 호환용)
            use_projection: position embedding 출력을 hidden_size로 projection할지
            freeze_embeddings: position embedding weights를 freeze할지
            device: 디바이스
        """
        super().__init__()

        from transformers import LayoutLMv3Model

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.use_projection = use_projection

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # LayoutLMv3 로드 (position embedding만 사용)
        layoutlm = LayoutLMv3Model.from_pretrained(model_name)

        # LayoutLMv3의 coordinate embeddings 추출
        # x, y 좌표 각각 0-1023 범위의 embedding
        self.x_position_embeddings = layoutlm.embeddings.x_position_embeddings
        self.y_position_embeddings = layoutlm.embeddings.y_position_embeddings
        self.h_position_embeddings = layoutlm.embeddings.h_position_embeddings
        self.w_position_embeddings = layoutlm.embeddings.w_position_embeddings

        # LayoutLMv3 coordinate embedding dim (base: 128)
        self.coord_embedding_dim = self.x_position_embeddings.embedding_dim

        # Total embedding dim: x0, y0, x1, y1, w, h = 6 * 128 = 768 (for base)
        self.layoutlm_embedding_size = self.coord_embedding_dim * 6

        # Freeze embeddings if specified
        if freeze_embeddings:
            for emb in [self.x_position_embeddings, self.y_position_embeddings,
                       self.h_position_embeddings, self.w_position_embeddings]:
                for param in emb.parameters():
                    param.requires_grad = False

        # Projection layer (768 -> 3584)
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.layoutlm_embedding_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
        else:
            self.projection = None

        # 원본 모델 삭제 (메모리 절약)
        del layoutlm

        self.to(self.device)

    def _normalize_bbox_to_1024(
        self,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float
    ) -> Tuple[int, int, int, int, int, int]:
        """
        bbox를 LayoutLMv3 형식으로 변환 (0-1023 scale)

        Args:
            bbox: (x0, y0, x1, y1) 원본 좌표
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            (x0, y0, x1, y1, w, h) 0-1023 scale로 정규화
        """
        x0, y0, x1, y1 = bbox

        # Normalize to 0-1023 (LayoutLMv3 embedding table size)
        x0_norm = int((x0 / page_width) * 1023)
        y0_norm = int((y0 / page_height) * 1023)
        x1_norm = int((x1 / page_width) * 1023)
        y1_norm = int((y1 / page_height) * 1023)

        # Clamp to valid range
        x0_norm = max(0, min(x0_norm, 1023))
        y0_norm = max(0, min(y0_norm, 1023))
        x1_norm = max(0, min(x1_norm, 1023))
        y1_norm = max(0, min(y1_norm, 1023))

        # Width and height
        w_norm = max(0, min(x1_norm - x0_norm, 1023))
        h_norm = max(0, min(y1_norm - y0_norm, 1023))

        return x0_norm, y0_norm, x1_norm, y1_norm, w_norm, h_norm

    def forward(
        self,
        bbox: Tuple[float, float, float, float],
        page_id: int = 0,
        reading_order: int = 0,
        region_type: int = 0,
        page_width: float = 1.0,
        page_height: float = 1.0
    ) -> torch.Tensor:
        """
        단일 Region의 layout embedding 계산

        Args:
            bbox: (x0, y0, x1, y1) 원본 좌표
            page_id: 페이지 번호 (현재 미사용, 호환성용)
            reading_order: 읽기 순서 (현재 미사용, 호환성용)
            region_type: RegionTypeId 값 (현재 미사용, 호환성용)
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            h_layout: (hidden_size,) layout embedding
        """
        # bbox 정규화
        x0, y0, x1, y1, w, h = self._normalize_bbox_to_1024(bbox, page_width, page_height)

        # Position embedding 추출
        x0_emb = self.x_position_embeddings(torch.tensor(x0, device=self.device))
        y0_emb = self.y_position_embeddings(torch.tensor(y0, device=self.device))
        x1_emb = self.x_position_embeddings(torch.tensor(x1, device=self.device))
        y1_emb = self.y_position_embeddings(torch.tensor(y1, device=self.device))
        w_emb = self.w_position_embeddings(torch.tensor(w, device=self.device))
        h_emb = self.h_position_embeddings(torch.tensor(h, device=self.device))

        # Concatenate all position embeddings
        h_layout = torch.cat([x0_emb, y0_emb, x1_emb, y1_emb, w_emb, h_emb], dim=-1)

        # Projection if needed
        if self.projection is not None:
            h_layout = self.projection(h_layout)

        return h_layout

    def forward_batch(
        self,
        bboxes: torch.Tensor,
        page_ids: torch.Tensor = None,
        reading_orders: torch.Tensor = None,
        region_types: torch.Tensor = None,
        page_width: float = 1.0,
        page_height: float = 1.0
    ) -> torch.Tensor:
        """
        배치 처리를 위한 forward

        Args:
            bboxes: (batch_size, 4) - bbox coordinates
            page_ids: (batch_size,) - 미사용, 호환성용
            reading_orders: (batch_size,) - 미사용, 호환성용
            region_types: (batch_size,) - 미사용, 호환성용
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            h_layout: (batch_size, hidden_size) layout embeddings
        """
        batch_size = bboxes.shape[0]

        # 각 bbox를 정규화
        coords = []
        for i in range(batch_size):
            bbox = bboxes[i].tolist()
            x0, y0, x1, y1, w, h = self._normalize_bbox_to_1024(bbox, page_width, page_height)
            coords.append((x0, y0, x1, y1, w, h))

        # Tensor로 변환
        x0_t = torch.tensor([c[0] for c in coords], device=self.device)
        y0_t = torch.tensor([c[1] for c in coords], device=self.device)
        x1_t = torch.tensor([c[2] for c in coords], device=self.device)
        y1_t = torch.tensor([c[3] for c in coords], device=self.device)
        w_t = torch.tensor([c[4] for c in coords], device=self.device)
        h_t = torch.tensor([c[5] for c in coords], device=self.device)

        # Position embeddings
        x0_emb = self.x_position_embeddings(x0_t)  # (batch, coord_dim)
        y0_emb = self.y_position_embeddings(y0_t)
        x1_emb = self.x_position_embeddings(x1_t)
        y1_emb = self.y_position_embeddings(y1_t)
        w_emb = self.w_position_embeddings(w_t)
        h_emb = self.h_position_embeddings(h_t)

        # Concatenate
        h_layout = torch.cat([x0_emb, y0_emb, x1_emb, y1_emb, w_emb, h_emb], dim=-1)

        # Projection if needed
        if self.projection is not None:
            h_layout = self.projection(h_layout)

        return h_layout


# Alias for backward compatibility
LayoutEmbedding = LayoutLMv3LayoutEmbedding


def region_type_to_id(region_type_str: str) -> int:
    """RegionType enum string을 ID로 변환"""
    mapping = {
        "paragraph": RegionTypeId.PARAGRAPH,
        "text": RegionTypeId.PARAGRAPH,  # Text -> Paragraph로 매핑
        "title": RegionTypeId.TITLE,
        "heading": RegionTypeId.HEADING,
        "table": RegionTypeId.TABLE,
        "figure": RegionTypeId.FIGURE,
        "caption": RegionTypeId.CAPTION,
        "header": RegionTypeId.HEADER,
        "footer": RegionTypeId.FOOTER,
        "page_number": RegionTypeId.PAGE_NUMBER,
        "list": RegionTypeId.LIST,
        "footnote": RegionTypeId.FOOTNOTE,
        "unknown": RegionTypeId.UNKNOWN,
    }
    return mapping.get(region_type_str.lower(), RegionTypeId.UNKNOWN)
