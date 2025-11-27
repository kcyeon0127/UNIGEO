"""
Detection Pipeline: Layout Detection + OCR + Tokenization 통합

Vision 기반 파이프라인 (어떤 PDF든 처리 가능)
"""

import torch
from PIL import Image
from pdf2image import convert_from_path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path

from .layout_detector import LayoutDetector, DetectedRegion, RegionLabel
from .ocr import OCREngine
from ..tokenization.layout_embedding import LayoutEmbedding, region_type_to_id
from ..tokenization.text_embedding import TextEmbedding, TextEmbeddingLite
from ..tokenization.visual_embedding import VisualEmbedding, VisualEmbeddingLite


@dataclass
class RegionEmbedding:
    """Region의 세 모달리티 embedding (Detection 기반)"""
    region_id: int
    label: RegionLabel

    # 세 모달리티 embedding
    h_text: Optional[torch.Tensor]    # (hidden_size,) or None
    h_image: Optional[torch.Tensor]   # (hidden_size,) or None
    h_layout: torch.Tensor            # (hidden_size,) - 항상 존재

    # 원본 데이터
    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def has_text(self) -> bool:
        return self.h_text is not None

    @property
    def has_image(self) -> bool:
        return self.h_image is not None


class DetectionPipeline:
    """
    Vision 기반 전체 파이프라인

    PDF → 이미지 → Layout Detection → OCR → Embedding
    """

    # RegionLabel → RegionType ID 매핑
    LABEL_TO_TYPE_ID = {
        RegionLabel.TEXT: 0,      # PARAGRAPH
        RegionLabel.TITLE: 1,     # TITLE
        RegionLabel.LIST: 9,      # LIST
        RegionLabel.TABLE: 3,     # TABLE
        RegionLabel.FIGURE: 4,    # FIGURE
        RegionLabel.UNKNOWN: 11,  # UNKNOWN
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        hidden_size: int = 3584,
        ocr_languages: List[str] = ['en'],
        detection_confidence: float = 0.3,
        device: Optional[str] = None,
        load_model: bool = True
    ):
        """
        Args:
            model_name: Qwen2-VL 모델 이름
            hidden_size: embedding 차원
            ocr_languages: OCR 언어 리스트
            detection_confidence: detection 최소 confidence
            use_lite: 경량 encoder 사용 (테스트용)
            device: 디바이스
            load_model: 모델 로드 여부
        """
        self.hidden_size = hidden_size
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Layout Detector
        print("Initializing Layout Detector...")
        self.layout_detector = LayoutDetector(
            confidence_threshold=detection_confidence
        )

        # OCR Engine
        print("Initializing OCR Engine...")
        self.ocr_engine = OCREngine(
            languages=ocr_languages,
            gpu=torch.cuda.is_available()
        )

        # Encoders
        print("Initializing Encoders...")
        self.text_encoder = TextEmbedding(
            model_name=model_name,
            load_model=load_model
        )
        self.visual_encoder = VisualEmbedding(
            model_name=model_name,
            load_model=load_model
        )

        self.layout_encoder = LayoutEmbedding(hidden_size=hidden_size)
        self.layout_encoder.to(self.device)

        print("Pipeline initialized!")

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        dpi: int = 150
    ) -> List[List[RegionEmbedding]]:
        """
        PDF 전체 처리

        Args:
            pdf_path: PDF 파일 경로
            pages: 처리할 페이지 번호 (1-indexed), None이면 전체
            dpi: 렌더링 DPI

        Returns:
            List[List[RegionEmbedding]]: 페이지별 region embeddings
        """
        pdf_path = Path(pdf_path)

        # PDF → 이미지
        if pages:
            page_images = []
            for p in pages:
                imgs = convert_from_path(pdf_path, first_page=p, last_page=p, dpi=dpi)
                page_images.extend(imgs)
        else:
            page_images = convert_from_path(pdf_path, dpi=dpi)

        # 각 페이지 처리
        all_embeddings = []
        for page_idx, page_image in enumerate(page_images):
            page_id = pages[page_idx] - 1 if pages else page_idx
            embeddings = self.process_page(page_image, page_id=page_id)
            all_embeddings.append(embeddings)

        return all_embeddings

    def process_page(
        self,
        page_image: Image.Image,
        page_id: int = 0
    ) -> List[RegionEmbedding]:
        """
        단일 페이지 처리

        Args:
            page_image: 페이지 이미지
            page_id: 페이지 번호 (0-indexed)

        Returns:
            List[RegionEmbedding]: region embeddings
        """
        page_width, page_height = page_image.size

        # 1. Layout Detection
        detected_regions = self.layout_detector.detect(page_image, extract_figures=True)

        # 2. OCR (텍스트 영역)
        detected_regions = self.ocr_engine.process_regions(detected_regions, page_image)

        # 3. Embedding 생성
        embeddings = []
        for region in detected_regions:
            emb = self._create_embedding(
                region=region,
                page_image=page_image,
                page_id=page_id,
                page_width=page_width,
                page_height=page_height
            )
            embeddings.append(emb)

        return embeddings

    def _create_embedding(
        self,
        region: DetectedRegion,
        page_image: Image.Image,
        page_id: int,
        page_width: float,
        page_height: float
    ) -> RegionEmbedding:
        """
        DetectedRegion → RegionEmbedding 변환
        """
        # 1. Text embedding
        h_text = None
        if region.text and region.text.strip():
            h_text = self.text_encoder.encode(region.text)
        else:
            region.text = ""

        # 2. Visual embedding (Figure/Table만)
        h_image = None
        if region.label in [RegionLabel.FIGURE, RegionLabel.TABLE]:
            try:
                if region.image is not None:
                    h_image = self.visual_encoder.encode(region.image)
                else:
                    cropped = self._crop_region(page_image, region.bbox)
                    if cropped:
                        h_image = self.visual_encoder.encode(cropped)
            except Exception as e:
                print(f"[VisualEmbedding] Skipping region {region.region_id} due to error: {e}")
                h_image = None

        # 3. Layout embedding (항상)
        region_type_id = self.LABEL_TO_TYPE_ID.get(region.label, 11)
        h_layout = self.layout_encoder(
            bbox=region.bbox,
            page_id=page_id,
            reading_order=region.region_id,
            region_type=region_type_id,
            page_width=page_width,
            page_height=page_height
        )

        return RegionEmbedding(
            region_id=region.region_id,
            label=region.label,
            h_text=h_text,
            h_image=h_image,
            h_layout=h_layout,
            text=region.text,
            bbox=region.bbox,
            confidence=region.confidence
        )

    def _crop_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Image.Image]:
        """Region 영역 crop"""
        try:
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.width, int(x2))
            y2 = min(image.height, int(y2))

            if x2 - x1 < 10 or y2 - y1 < 10:
                return None

            return image.crop((x1, y1, x2, y2))
        except Exception:
            return None


def collate_embeddings(
    embeddings: List[RegionEmbedding],
    device: Optional[torch.device] = None
) -> dict:
    """
    RegionEmbedding 리스트를 배치 텐서로 변환
    """
    if not embeddings:
        return None

    if device is None:
        device = embeddings[0].h_layout.device

    # Layout embeddings (항상 존재)
    h_layout = torch.stack([e.h_layout for e in embeddings], dim=0).to(device)

    # Text embeddings (있는 것만)
    text_indices = [i for i, e in enumerate(embeddings) if e.has_text]
    h_text = None
    if text_indices:
        h_text = torch.stack([embeddings[i].h_text for i in text_indices], dim=0).to(device)

    # Image embeddings (있는 것만)
    image_indices = [i for i, e in enumerate(embeddings) if e.has_image]
    h_image = None
    if image_indices:
        h_image = torch.stack([embeddings[i].h_image for i in image_indices], dim=0).to(device)

    # Masks
    text_mask = torch.tensor([e.has_text for e in embeddings], dtype=torch.bool, device=device)
    image_mask = torch.tensor([e.has_image for e in embeddings], dtype=torch.bool, device=device)

    return {
        "h_text": h_text,
        "h_image": h_image,
        "h_layout": h_layout,
        "text_mask": text_mask,
        "image_mask": image_mask,
        "text_indices": text_indices,
        "image_indices": image_indices,
        "num_regions": len(embeddings)
    }
