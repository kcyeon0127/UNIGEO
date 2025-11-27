"""
Region Tokenizer: Region을 세 모달리티 embedding으로 변환

입력: List[DetectedRegion] (from LayoutDetector)
출력: List[RegionEmbedding] (hᵀ, hᴵ, hᴸ)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from PIL import Image
import math

from ..detection.layout_detector import DetectedRegion, RegionLabel
from .layout_embedding import LayoutEmbedding, region_type_to_id
from .text_embedding import TextEmbedding, TextEmbeddingLite
from .visual_embedding import VisualEmbedding, VisualEmbeddingLite


@dataclass
class RegionEmbedding:
    """Region의 세 모달리티 embedding"""
    region_id: int
    region_type: RegionLabel

    # 세 모달리티 embedding
    h_text: Optional[torch.Tensor]    # (hidden_size,) or None
    h_image: Optional[torch.Tensor]   # (hidden_size,) or None
    h_layout: torch.Tensor            # (hidden_size,) - 항상 존재

    # 메타데이터
    has_text: bool
    has_image: bool
    bbox: Tuple[float, float, float, float]
    reading_order: int

    def to_dict(self):
        """Dictionary로 변환"""
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "h_text_shape": self.h_text.shape if self.h_text is not None else None,
            "h_image_shape": self.h_image.shape if self.h_image is not None else None,
            "h_layout_shape": self.h_layout.shape,
            "has_text": self.has_text,
            "has_image": self.has_image,
            "bbox": self.bbox,
            "reading_order": self.reading_order
        }


class RegionTokenizer(nn.Module):
    """
    Region을 세 모달리티 embedding으로 변환

    각 Region에 대해:
    - hᵀ (text): 텍스트가 있는 경우 text embedding
    - hᴵ (image): FIGURE/TABLE인 경우 visual embedding
    - hᴸ (layout): 항상 layout embedding
    """

    # Visual embedding을 생성할 region type
    VISUAL_REGION_TYPES = {RegionLabel.FIGURE, RegionLabel.TABLE}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        hidden_size: int = 3584,
        text_mode: str = "token_only",
        visual_pooling: str = "mean",
        use_lite: bool = False,
        device: Optional[str] = None,
        load_model: bool = True
    ):
        """
        Args:
            model_name: Qwen2-VL 모델 이름
            hidden_size: embedding 차원
            text_mode: TextEmbedding 모드 ("token_only", "shallow", "deep")
            visual_pooling: VisualEmbedding pooling 방식
            use_lite: True면 경량 버전 사용 (테스트용)
            device: 디바이스
            load_model: True면 모델 로드
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.use_lite = use_lite

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 세 encoder 초기화
        if use_lite:
            # 테스트용 경량 버전
            self.text_encoder = TextEmbeddingLite(hidden_size=hidden_size)
            self.visual_encoder = VisualEmbeddingLite(hidden_size=hidden_size)
        else:
            # Qwen2-VL 기반
            self.text_encoder = TextEmbedding(
                model_name=model_name,
                mode=text_mode,
                load_model=load_model
            )
            self.visual_encoder = VisualEmbedding(
                model_name=model_name,
                pooling=visual_pooling,
                load_model=load_model
            )

        # Layout encoder (항상 사용)
        self.layout_encoder = LayoutEmbedding(hidden_size=hidden_size)
        self.layout_encoder.to(self.device)

    def tokenize(
        self,
        regions: List[DetectedRegion],
        page_image: Optional[Image.Image] = None,
        page_width: float = 612.0,
        page_height: float = 792.0,
        page_id: int = 0
    ) -> List[RegionEmbedding]:
        """
        Region 리스트를 embedding으로 변환

        Args:
            regions: LayoutDetector 출력
            page_image: 렌더링된 PDF 페이지 이미지 (visual embedding용)
            page_width: 페이지 너비 (PDF 좌표계)
            page_height: 페이지 높이 (PDF 좌표계)
            page_id: 페이지 번호

        Returns:
            List[RegionEmbedding]: 각 region의 세 모달리티 embedding
        """
        results = []

        for idx, region in enumerate(regions):
            region_emb = self.tokenize_single(
                region=region,
                reading_order=idx,
                page_image=page_image,
                page_width=page_width,
                page_height=page_height,
                page_id=page_id
            )
            results.append(region_emb)

        return results

    def tokenize_single(
        self,
        region: DetectedRegion,
        reading_order: int = 0,
        page_image: Optional[Image.Image] = None,
        page_width: float = 612.0,
        page_height: float = 792.0,
        page_id: int = 0
    ) -> RegionEmbedding:
        """
        단일 Region을 embedding으로 변환

        Args:
            region: 단일 DetectedRegion
            reading_order: 읽기 순서
            page_image: 렌더링된 페이지 이미지
            page_width, page_height: 페이지 크기
            page_id: 페이지 번호

        Returns:
            RegionEmbedding
        """
        # 1. Text embedding (텍스트가 있으면)
        h_text = None
        has_text = region.text and region.text.strip()
        if has_text:
            h_text = self.text_encoder.encode(region.text)

        # 2. Visual embedding (FIGURE/TABLE만)
        h_image = None
        if region.label in self.VISUAL_REGION_TYPES and page_image is not None:
            cropped = self._crop_region(
                page_image, region.bbox, page_width, page_height
            )
            if cropped is not None:
                h_image = self.visual_encoder.encode(cropped)

        # 3. Layout embedding (항상)
        region_type_id = region_type_to_id(region.label.value)
        h_layout = self.layout_encoder(
            bbox=region.bbox,
            page_id=page_id,
            reading_order=reading_order,
            region_type=region_type_id,
            page_width=page_width,
            page_height=page_height
        )

        return RegionEmbedding(
            region_id=region.region_id,
            region_type=region.label,
            h_text=h_text,
            h_image=h_image,
            h_layout=h_layout,
            has_text=h_text is not None,
            has_image=h_image is not None,
            bbox=region.bbox,
            reading_order=reading_order
        )

    def _crop_region(
        self,
        page_image: Image.Image,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float
    ) -> Optional[Image.Image]:
        """
        페이지 이미지에서 region 영역 crop

        Args:
            page_image: 렌더링된 페이지 이미지
            bbox: (x0, y0, x1, y1) PDF 좌표
            page_width, page_height: PDF 페이지 크기

        Returns:
            crop된 이미지 또는 None
        """
        try:
            # 이미지 크기
            img_width, img_height = page_image.size

            # PDF 좌표 → 이미지 좌표 변환
            scale_x = img_width / page_width
            scale_y = img_height / page_height

            x0, y0, x1, y1 = bbox
            img_x0 = int(x0 * scale_x)
            img_y0 = int(y0 * scale_y)
            img_x1 = int(x1 * scale_x)
            img_y1 = int(y1 * scale_y)

            # Clamp to image bounds
            img_x0 = max(0, min(img_x0, img_width - 1))
            img_y0 = max(0, min(img_y0, img_height - 1))
            img_x1 = max(img_x0 + 1, min(img_x1, img_width))
            img_y1 = max(img_y0 + 1, min(img_y1, img_height))

            # 너무 작은 영역은 스킵
            if img_x1 - img_x0 < 10 or img_y1 - img_y0 < 10:
                return None

            # Crop
            cropped = page_image.crop((img_x0, img_y0, img_x1, img_y1))

            return cropped

        except Exception as e:
            print(f"Warning: Failed to crop region: {e}")
            return None

    def forward(
        self,
        regions: List[DetectedRegion],
        page_image: Optional[Image.Image] = None,
        page_width: float = 612.0,
        page_height: float = 792.0,
        page_id: int = 0
    ) -> List[RegionEmbedding]:
        """Forward pass (nn.Module 인터페이스)"""
        return self.tokenize(regions, page_image, page_width, page_height, page_id)


class DocumentTokenizer:
    """
    전체 문서를 tokenize

    여러 페이지의 Region들을 일괄 처리
    """

    def __init__(
        self,
        region_tokenizer: Optional[RegionTokenizer] = None,
        **kwargs
    ):
        """
        Args:
            region_tokenizer: RegionTokenizer 인스턴스 (없으면 새로 생성)
            **kwargs: RegionTokenizer 생성 인자
        """
        if region_tokenizer is not None:
            self.tokenizer = region_tokenizer
        else:
            self.tokenizer = RegionTokenizer(**kwargs)

    def tokenize_document(
        self,
        pages_regions: List[List[DetectedRegion]],
        page_images: List[Image.Image],
        page_sizes: List[Tuple[float, float]]
    ) -> List[List[RegionEmbedding]]:
        """
        전체 문서의 모든 페이지를 tokenize

        Args:
            pages_regions: 각 페이지의 Region 리스트들
            page_images: 각 페이지의 렌더링된 이미지
            page_sizes: 각 페이지의 (width, height)

        Returns:
            List[List[RegionEmbedding]]: 페이지별 RegionEmbedding 리스트
        """
        all_embeddings = []

        for page_id, (regions, image, (width, height)) in enumerate(
            zip(pages_regions, page_images, page_sizes)
        ):
            page_embeddings = self.tokenizer.tokenize(
                regions=regions,
                page_image=image,
                page_width=width,
                page_height=height,
                page_id=page_id
            )
            all_embeddings.append(page_embeddings)

        return all_embeddings

    def flatten_embeddings(
        self,
        pages_embeddings: List[List[RegionEmbedding]]
    ) -> List[RegionEmbedding]:
        """페이지별 embedding을 flat list로 변환"""
        return [emb for page_embs in pages_embeddings for emb in page_embs]


def collate_region_embeddings(
    embeddings: List[RegionEmbedding],
    device: Optional[torch.device] = None
) -> dict:
    """
    RegionEmbedding 리스트를 배치 텐서로 변환

    Args:
        embeddings: RegionEmbedding 리스트
        device: 출력 디바이스

    Returns:
        dict with:
        - h_text: (num_text_regions, hidden_size) or None
        - h_image: (num_image_regions, hidden_size) or None
        - h_layout: (num_regions, hidden_size)
        - text_mask: (num_regions,) bool - text가 있는 region
        - image_mask: (num_regions,) bool - image가 있는 region
        - region_types: (num_regions,) int
        - reading_orders: (num_regions,) int
    """
    if not embeddings:
        return None

    if device is None:
        device = embeddings[0].h_layout.device

    num_regions = len(embeddings)
    hidden_size = embeddings[0].h_layout.shape[0]

    # Layout embeddings (항상 존재)
    h_layout = torch.stack([e.h_layout for e in embeddings], dim=0).to(device)

    # Text embeddings (있는 것만)
    text_indices = [i for i, e in enumerate(embeddings) if e.has_text]
    if text_indices:
        h_text = torch.stack([embeddings[i].h_text for i in text_indices], dim=0).to(device)
    else:
        h_text = None

    # Image embeddings (있는 것만)
    image_indices = [i for i, e in enumerate(embeddings) if e.has_image]
    if image_indices:
        h_image = torch.stack([embeddings[i].h_image for i in image_indices], dim=0).to(device)
    else:
        h_image = None

    # Masks
    text_mask = torch.tensor([e.has_text for e in embeddings], dtype=torch.bool, device=device)
    image_mask = torch.tensor([e.has_image for e in embeddings], dtype=torch.bool, device=device)

    # Metadata
    region_types = torch.tensor(
        [region_type_to_id(e.region_type.value) for e in embeddings],
        dtype=torch.long, device=device
    )
    reading_orders = torch.tensor(
        [e.reading_order for e in embeddings],
        dtype=torch.long, device=device
    )

    return {
        "h_text": h_text,
        "h_image": h_image,
        "h_layout": h_layout,
        "text_mask": text_mask,
        "image_mask": image_mask,
        "text_indices": text_indices,
        "image_indices": image_indices,
        "region_types": region_types,
        "reading_orders": reading_orders,
        "num_regions": num_regions
    }
