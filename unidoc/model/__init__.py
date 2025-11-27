"""
UniDoc Model Module

Manifold Alignment 기반 멀티모달 문서 이해 모델
"""

from .manifold_align import (
    OrthoLinear,
    LinearProjector,
    ManifoldAlignModel,
    ConcatBaselineModel,
    contrastive_alignment,
    region_contrastive_alignment,
    train_epoch,
)

__all__ = [
    "OrthoLinear",
    "LinearProjector",
    "ManifoldAlignModel",
    "ConcatBaselineModel",
    "contrastive_alignment",
    "region_contrastive_alignment",
    "train_epoch",
]
