"""
OCR Engine: EasyOCR 기반 텍스트 추출

DetectedRegion의 텍스트 영역에서 텍스트 추출
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import easyocr

from .layout_detector import DetectedRegion, RegionLabel


class OCREngine:
    """
    EasyOCR 기반 OCR 엔진

    Text/Title/List 영역에서 텍스트 추출
    """

    def __init__(
        self,
        languages: List[str] = ['en'],
        gpu: bool = True,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            languages: 인식할 언어 리스트 (예: ['en'], ['en', 'ko'])
            gpu: GPU 사용 여부
            confidence_threshold: 최소 confidence score
        """
        self.languages = languages
        self.confidence_threshold = confidence_threshold

        print(f"Loading EasyOCR with languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR loaded successfully")

    def extract_text(
        self,
        image: Image.Image,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> str:
        """
        이미지에서 텍스트 추출

        Args:
            image: PIL Image
            bbox: crop할 영역 (None이면 전체 이미지)

        Returns:
            추출된 텍스트
        """
        # bbox가 있으면 crop
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.width, int(x2))
            y2 = min(image.height, int(y2))
            image = image.crop((x1, y1, x2, y2))

        # PIL Image → numpy array
        if image.width == 0 or image.height == 0:
            return ""

        image_array = np.array(image)

        # OCR 실행
        results = self.reader.readtext(image_array)

        # 결과 합치기 (reading order 순)
        texts = []
        for (box, text, conf) in results:
            if conf >= self.confidence_threshold:
                texts.append(text)

        return " ".join(texts)

    def extract_text_with_boxes(
        self,
        image: Image.Image,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> List[Dict]:
        """
        이미지에서 텍스트 + bbox 추출

        Returns:
            List of {text, bbox, confidence}
        """
        # bbox가 있으면 crop
        offset_x, offset_y = 0, 0
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            offset_x, offset_y = int(x1), int(y1)
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.width, int(x2))
            y2 = min(image.height, int(y2))
            image = image.crop((x1, y1, x2, y2))

        # PIL Image → numpy array
        if image.width == 0 or image.height == 0:
            return []

        image_array = np.array(image)

        # OCR 실행
        results = self.reader.readtext(image_array)

        # 결과 변환
        output = []
        for (box, text, conf) in results:
            if conf >= self.confidence_threshold:
                # box는 4개 점의 리스트: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                text_bbox = (
                    min(xs) + offset_x,
                    min(ys) + offset_y,
                    max(xs) + offset_x,
                    max(ys) + offset_y
                )
                output.append({
                    "text": text,
                    "bbox": text_bbox,
                    "confidence": conf
                })

        return output

    def process_regions(
        self,
        regions: List[DetectedRegion],
        page_image: Image.Image,
        text_labels: Optional[List[RegionLabel]] = None
    ) -> List[DetectedRegion]:
        """
        DetectedRegion 리스트에 텍스트 추가

        Args:
            regions: DetectedRegion 리스트
            page_image: 전체 페이지 이미지
            text_labels: OCR할 라벨들 (None이면 Text, Title, List)

        Returns:
            텍스트가 추가된 regions
        """
        if text_labels is None:
            text_labels = [RegionLabel.TEXT, RegionLabel.TITLE, RegionLabel.LIST]

        for region in regions:
            if region.label in text_labels:
                # 해당 영역에서 OCR
                region.text = self.extract_text(page_image, region.bbox)

        return regions

    def process_region(
        self,
        region: DetectedRegion,
        page_image: Image.Image
    ) -> DetectedRegion:
        """단일 region에 텍스트 추가"""
        if region.label in [RegionLabel.TEXT, RegionLabel.TITLE, RegionLabel.LIST]:
            region.text = self.extract_text(page_image, region.bbox)
        return region


class OCRResult:
    """OCR 결과를 담는 컨테이너"""

    def __init__(
        self,
        text: str,
        boxes: List[Dict],
        confidence: float
    ):
        self.text = text
        self.boxes = boxes
        self.confidence = confidence

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __str__(self) -> str:
        return self.text
