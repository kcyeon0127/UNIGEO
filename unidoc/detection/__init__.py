"""
UniDoc Detection Module

Vision-based layout detection + OCR for any PDF (digital or scanned)
"""

from .layout_detector import LayoutDetector, DetectedRegion, RegionLabel
from .ocr import OCREngine
from .pipeline import DetectionPipeline, RegionEmbedding, collate_embeddings

__all__ = [
    "LayoutDetector",
    "DetectedRegion",
    "RegionLabel",
    "OCREngine",
    "DetectionPipeline",
    "RegionEmbedding",
    "collate_embeddings",
]
