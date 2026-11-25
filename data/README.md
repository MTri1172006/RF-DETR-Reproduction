# Data Directory

Datasets downloaded and converted by `scripts/prepare_dataset.py` are stored here:

- `voc_raw/` – Raw PASCAL VOC 2007 (`trainval` and `test`) downloaded via `torchvision.datasets.VOCDetection`.
- `voc2007_rfdetr/` – COCO-formatted `train/`, `valid/`, and `test/` splits (`_annotations.coco.json` + images) with `difficult=1` objects filtered out and deterministic 90/10 train/valid split (`SEED=42`).
- `voc2007_yolo_fair/` – YOLO-formatted dataset exported from `voc2007_rfdetr/` for fair YOLOv8n / YOLO11n baseline comparison.

All data files are excluded from Git via `.gitignore`.
