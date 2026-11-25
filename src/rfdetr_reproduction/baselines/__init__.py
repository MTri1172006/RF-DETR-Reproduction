"""Fair YOLO baseline dataset conversion, training, and test-set comparison (Section 8)."""

from rfdetr_reproduction.baselines.compare import run_fair_baseline_comparison
from rfdetr_reproduction.baselines.train_yolo import (
    evaluate_yolo_on_coco_split,
    train_yolo_baseline,
)
from rfdetr_reproduction.baselines.yolo_dataset import export_coco_to_yolo

__all__ = [
    "export_coco_to_yolo",
    "train_yolo_baseline",
    "evaluate_yolo_on_coco_split",
    "run_fair_baseline_comparison",
]
