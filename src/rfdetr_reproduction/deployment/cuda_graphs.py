"""PyTorch CUDA Graphs latency benchmark for a selected RF-DETR subnet (Section 7)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter
from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


def benchmark_cuda_graphs(
    adapter: RFDETRNASAdapter,
    cfg: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int = 15,
    repeats: int = 40,
    buffer_sec: float = 0.2,
) -> dict[str, Any]:
    """Capture and replay a PyTorch CUDA Graph for static-shape subnet inference."""
    if device.type != "cuda":
        logger.warning("[CUDA Graphs] Skipping because device is %s (requires CUDA).", device)
        return {"available": False, "reason": "non_cuda_device"}

    adapter.eval()
    adapter.activate_subnet(cfg, mode="inference")
    res = int(cfg["resolution"])
    static_x = torch.randn(1, 3, res, res, device=device)

    try:
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            for _ in range(warmup):
                _ = adapter(static_x)
        torch.cuda.current_stream(device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            static_out = adapter(static_x)

        times_ms = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            if buffer_sec > 0:
                time.sleep(buffer_sec)

        del static_out
        return {
            "available": True,
            "latency_median_ms": float(np.median(times_ms)),
            "latency_mean_ms": float(np.mean(times_ms)),
            "latency_p90_ms": float(np.percentile(times_ms, 90)),
            "latency_std_ms": float(np.std(times_ms)),
        }
    except Exception as exc:
        logger.warning("[CUDA Graphs] Capture failed: %s", exc)
        return {"available": False, "reason": str(exc)}
