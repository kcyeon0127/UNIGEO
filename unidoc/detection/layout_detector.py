"""
Layout Detector: Vision-based document layout detection

LayoutParser + EfficientDet (PubLayNet) 사용
어떤 PDF든 (digital / scanned) 처리 가능
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import layoutparser as lp


class RegionLabel(Enum):
    """Region 타입 라벨 (PubLayNet 기준)"""
    TEXT = "Text"
    TITLE = "Title"
    LIST = "List"
    TABLE = "Table"
    FIGURE = "Figure"
    UNKNOWN = "Unknown"


@dataclass
class DetectedRegion:
    """Detection 결과로 나온 Region"""
    region_id: int
    label: RegionLabel
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float

    # OCR 결과 (나중에 채워짐)
    text: str = ""

    # 이미지 crop (Figure/Table용)
    image: Optional[Image.Image] = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


class LayoutDetector:
    """
    LayoutParser 기반 Layout Detection

    PubLayNet 모델 사용:
    - Text, Title, List, Table, Figure 탐지
    """

    # 모델 설정
    AVAILABLE_MODELS = {
        "efficientdet_d0": "lp://PubLayNet/tf_efficientdet_d0/config",
        "efficientdet_d1": "lp://PubLayNet/tf_efficientdet_d1/config",
    }

    # PubLayNet 라벨 매핑
    LABEL_MAP = {
        "Text": RegionLabel.TEXT,
        "Title": RegionLabel.TITLE,
        "List": RegionLabel.LIST,
        "Table": RegionLabel.TABLE,
        "Figure": RegionLabel.FIGURE,
    }

    def __init__(
        self,
        model_name: str = "efficientdet_d0",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: 사용할 모델 ("efficientdet_d0" or "efficientdet_d1")
            confidence_threshold: 최소 confidence score
            device: "cuda" or "cpu" (None이면 자동 선택)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # 모델 로드
        model_config = self.AVAILABLE_MODELS.get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")

        print(f"Loading LayoutParser model: {model_name}")
        self.model = lp.EfficientDetLayoutModel(model_config)
        print("Model loaded successfully")

    def detect(
        self,
        image: Image.Image,
        extract_figures: bool = True
    ) -> List[DetectedRegion]:
        """
        이미지에서 layout regions 탐지

        Args:
            image: PIL Image (페이지 이미지)
            extract_figures: True면 Figure/Table 영역을 crop해서 저장

        Returns:
            List[DetectedRegion]: 탐지된 regions
        """
        # PIL Image → numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Layout detection
        layout = self.model.detect(image_array)

        # 결과 변환
        regions = []
        for i, block in enumerate(layout):
            # Confidence 필터링
            if block.score < self.confidence_threshold:
                continue

            # 라벨 변환
            label = self.LABEL_MAP.get(block.type, RegionLabel.UNKNOWN)

            # bbox 추출
            bbox = (
                float(block.block.x_1),
                float(block.block.y_1),
                float(block.block.x_2),
                float(block.block.y_2)
            )

            region = DetectedRegion(
                region_id=i,
                label=label,
                bbox=bbox,
                confidence=float(block.score)
            )

            # Figure/Table이면 이미지 crop
            if extract_figures and label in [RegionLabel.FIGURE, RegionLabel.TABLE]:
                region.image = self._crop_region(image, bbox)

            regions.append(region)

        # Reading order 정렬 (위→아래, 왼쪽→오른쪽)
        regions = self._sort_reading_order(regions)

        # region_id 재할당
        for i, region in enumerate(regions):
            region.region_id = i

        return regions

    def detect_batch(
        self,
        images: List[Image.Image],
        extract_figures: bool = True
    ) -> List[List[DetectedRegion]]:
        """
        여러 이미지 배치 처리
        """
        results = []
        for image in images:
            regions = self.detect(image, extract_figures)
            results.append(regions)
        return results

    def _crop_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float]
    ) -> Image.Image:
        """Region 영역 crop"""
        x1, y1, x2, y2 = bbox

        # 이미지 경계 체크
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.width, int(x2))
        y2 = min(image.height, int(y2))

        return image.crop((x1, y1, x2, y2))

    def _sort_reading_order(
        self,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """
        Reading order로 정렬

        간단한 휴리스틱:
        1. y 좌표로 정렬 (위에서 아래)
        2. 같은 줄이면 x 좌표로 정렬 (왼쪽에서 오른쪽)
        """
        if not regions:
            return regions

        # 평균 높이 계산 (같은 줄 판단용)
        avg_height = sum(r.height for r in regions) / len(regions)
        line_threshold = avg_height * 0.5

        # y 좌표 기준으로 그룹핑
        sorted_regions = sorted(regions, key=lambda r: r.bbox[1])

        lines = []
        current_line = [sorted_regions[0]]
        current_y = sorted_regions[0].bbox[1]

        for region in sorted_regions[1:]:
            if abs(region.bbox[1] - current_y) < line_threshold:
                # 같은 줄
                current_line.append(region)
            else:
                # 새 줄
                lines.append(current_line)
                current_line = [region]
                current_y = region.bbox[1]

        lines.append(current_line)

        # 각 줄 내에서 x 좌표로 정렬
        result = []
        for line in lines:
            line_sorted = sorted(line, key=lambda r: r.bbox[0])
            result.extend(line_sorted)

        return result

    def filter_by_label(
        self,
        regions: List[DetectedRegion],
        labels: List[RegionLabel]
    ) -> List[DetectedRegion]:
        """특정 라벨만 필터링"""
        return [r for r in regions if r.label in labels]

    def get_text_regions(
        self,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """Text 영역만 반환 (Text, Title, List)"""
        text_labels = [RegionLabel.TEXT, RegionLabel.TITLE, RegionLabel.LIST]
        return self.filter_by_label(regions, text_labels)

    def get_visual_regions(
        self,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """Visual 영역만 반환 (Figure, Table)"""
        visual_labels = [RegionLabel.FIGURE, RegionLabel.TABLE]
        return self.filter_by_label(regions, visual_labels)
