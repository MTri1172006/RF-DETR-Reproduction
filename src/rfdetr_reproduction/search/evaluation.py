"""Subnet latency benchmarking (200ms buffer protocol) and COCO mAP evaluation (Section 6)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval

from rfdetr_reproduction.data.dataset import SymmetricVOCDetectionDataset
from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter, get_nested_tensor_helpers


def benchmark_subnet_latency(
    adapter: RFDETRNASAdapter,
    cfg: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int = 15,
    repeats: int = 40,
    buffer_sec: float = 0.2,
) -> dict[str, float]:
    """Measure PyTorch Eager latency with 200ms thermal-cooldown buffer between runs."""
    adapter.eval()
    adapter.activate_subnet(cfg, mode="inference")
    res = int(cfg["resolution"])
    sample_x = torch.randn(1, 3, res, res, device=device)
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=is_cuda):
        for _ in range(warmup):
            _ = adapter(sample_x)
        if is_cuda:
            torch.cuda.synchronize(device)

        times_ms = []
        for _ in range(repeats):
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = adapter(sample_x)
            if is_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
            if buffer_sec > 0:
                time.sleep(buffer_sec)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3) if is_cuda else 0.0
    return {
        "latency_median_ms": float(np.median(times_ms)),
        "latency_mean_ms": float(np.mean(times_ms)),
        "latency_p90_ms": float(np.percentile(times_ms, 90)),
        "latency_std_ms": float(np.std(times_ms)),
        "peak_vram_gb": round(float(peak_vram), 3),
    }


@torch.no_grad()
def evaluate_subnet_coco(
    adapter: RFDETRNASAdapter,
    cfg: dict[str, Any],
    dataset: SymmetricVOCDetectionDataset,
    postprocessors: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    max_images: int | None = None,
    score_thresh: float = 0.01,
) -> dict[str, float]:
    """Evaluate a subnet on a COCO-formatted split using `pycocotools.COCOeval`."""
    _, nested_tensor_from_list = get_nested_tensor_helpers()
    adapter.eval()
    adapter.activate_subnet(cfg, mode="inference")
    res = int(cfg["resolution"])
    postproc = postprocessors["bbox"]
    coco_gt = dataset.coco
    is_cuda = device.type == "cuda"

    coco_results = []
    eval_indices = (
        list(range(len(dataset)))
        if max_images is None
        else list(range(min(max_images, len(dataset))))
    )

    for i in eval_indices:
        img_t, tgt = dataset[i]
        img_id = int(tgt["image_id"].item())
        orig_size = tgt["orig_size"].unsqueeze(0).to(device)

        img_resized = F.interpolate(
            img_t.unsqueeze(0), size=(res, res), mode="bilinear", align_corners=False
        )
        samples = nested_tensor_from_list([img_resized.squeeze(0)]).to(device)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=is_cuda):
            outputs = adapter(samples)

        out_fp32 = {
            "pred_logits": outputs["pred_logits"].float(),
            "pred_boxes": outputs["pred_boxes"].float(),
        }
        preds = postproc(out_fp32, orig_size)[0]
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        boxes = preds["boxes"].cpu().numpy()

        for box, sc, lbl in zip(boxes, scores, labels):
            if sc < score_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            coco_results.append(
                {
                    "image_id": img_id,
                    "category_id": int(lbl) + 1,
                    "bbox": [x1, y1, w, h],
                    "score": float(sc),
                }
            )

    if not coco_results:
        return {"map": 0.0, "map50": 0.0, "ap_s": 0.0, "ap_m": 0.0, "ap_l": 0.0}

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if max_images is not None:
        coco_eval.params.imgIds = [dataset.img_ids[i] for i in eval_indices]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map": float(coco_eval.stats[0]),
        "map50": float(coco_eval.stats[1]),
        "ap_s": float(coco_eval.stats[3]),
        "ap_m": float(coco_eval.stats[4]),
        "ap_l": float(coco_eval.stats[5]),
    }
