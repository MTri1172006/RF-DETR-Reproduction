"""Configuration for fair YOLO baseline comparisons (YOLOv8n, YOLO11n)."""

from configs.nas_config import EFFECTIVE_BATCH, NAS_EPOCHS, SEED

BASELINE_CONFIG = {
    "compare_res": 384,
    "epochs": NAS_EPOCHS,
    "batch_size": EFFECTIVE_BATCH,
    "seed": SEED,
    "models": [
        ("YOLOv8n", "yolov8n.pt"),
        ("YOLO11n", "yolo11n.pt"),
    ],
    "score_thresh": 0.01,
    "latency_warmup": 10,
    "latency_repeats": 40,
    "latency_buffer_sec": 0.2,
}
