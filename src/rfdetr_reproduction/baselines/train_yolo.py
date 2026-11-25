"""Train and evaluate Ultralytics YOLO baselines under the exact VOC2007 COCOeval protocol (Section 8)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def train_yolo_baseline(
    weights: str,
    yolo_yaml: Path,
    project_dir: Path,
    model_tag: str,
    epochs: int = 20,
    imgsz: int = 384,
    batch_size: int = 16,
    seed: int = 42,
) -> Path:
    """Train a YOLO baseline model and return path to `best.pt`."""
    from ultralytics import YOLO

    y_model = YOLO(weights)
    y_model.train(
        data=str(yolo_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        seed=seed,
        project=str(project_dir),
        name=model_tag,
        exist_ok=True,
        verbose=False,
    )
    return Path(project_dir) / model_tag / "weights" / "best.pt"


def evaluate_yolo_on_coco_split(
    yolo_model,
    split_dir: Path,
    imgsz: int = 384,
    warmup: int = 10,
    repeats_lat: int = 40,
    buffer_sec: float = 0.2,
    score_thresh: float = 0.01,
) -> dict[str, float]:
    """Evaluate a trained YOLO model on a COCO-formatted split using pycocotools + 200ms buffer latency."""
    split_dir = Path(split_dir)
    coco_gt_split = COCO(str(split_dir / "_annotations.coco.json"))
    img_ids = sorted(coco_gt_split.getImgIds())
    is_cuda = torch.cuda.is_available()

    sample_path = str(split_dir / coco_gt_split.loadImgs(img_ids[0])[0]["file_name"])
    for _ in range(warmup):
        _ = yolo_model.predict(sample_path, imgsz=imgsz, verbose=False)

    times_ms = []
    for _ in range(repeats_lat):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = yolo_model.predict(sample_path, imgsz=imgsz, verbose=False)
        if is_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if buffer_sec > 0:
            time.sleep(buffer_sec)

    coco_results = []
    for img_id in img_ids:
        info = coco_gt_split.loadImgs(img_id)[0]
        res = yolo_model.predict(
            str(split_dir / info["file_name"]), imgsz=imgsz, conf=score_thresh, verbose=False
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for box, sc, cls_idx in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()
                coco_results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cls_idx) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(sc),
                    }
                )

    if not coco_results:
        return {"map": 0.0, "map50": 0.0, "latency_median_ms": float(np.median(times_ms))}

    coco_dt = coco_gt_split.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt_split, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "map": float(coco_eval.stats[0]),
        "map50": float(coco_eval.stats[1]),
        "latency_median_ms": float(np.median(times_ms)),
    }
