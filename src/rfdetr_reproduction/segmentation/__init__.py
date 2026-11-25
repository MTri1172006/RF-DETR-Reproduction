"""Optional RF-DETR-Seg instance segmentation head extension (Section 9)."""

from rfdetr_reproduction.segmentation.seg_head import (
    LightweightRFDETRSegHead,
    run_segmentation_smoke_test,
)

__all__ = ["LightweightRFDETRSegHead", "run_segmentation_smoke_test"]
