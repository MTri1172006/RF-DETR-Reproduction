"""GPU Power Throttling check (0ms vs 200ms buffer) and full deployment pipeline (Section 7)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.experiment_config import DEPLOYMENT_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.deployment.cuda_graphs import benchmark_cuda_graphs
from rfdetr_reproduction.deployment.onnx_export import (
    export_subnet_to_onnx,
    select_best_pareto_subnet,
)
from rfdetr_reproduction.deployment.tensorrt import build_and_benchmark_tensorrt_fp16
from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter, build_official_rfdetr_supernet
from rfdetr_reproduction.training.checkpoint import load_ema_weights
from rfdetr_reproduction.training.train_utils import resolve_device_and_amp
from rfdetr_reproduction.utils.backup import BackupManager, save_json
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths
from rfdetr_reproduction.utils.plotting import plt, save_figure

logger = get_logger(__name__)


def compare_buffering_throttling(
    adapter: RFDETRNASAdapter,
    cfg: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int = 15,
    num_runs: int = 60,
) -> dict[str, list[float]]:
    """Compare unbuffered (0ms) vs paper-protocol buffered (200ms) latency."""
    adapter.eval()
    adapter.activate_subnet(cfg, mode="inference")
    res = int(cfg["resolution"])
    x = torch.randn(1, 3, res, res, device=device)
    is_cuda = device.type == "cuda"

    results_buf: dict[str, list[float]] = {}
    for label, buf in [("0ms_no_buffer", 0.0), ("200ms_paper_buffer", 0.2)]:
        times: list[float] = []
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=is_cuda):
            for _ in range(warmup):
                _ = adapter(x)
            if is_cuda:
                torch.cuda.synchronize(device)
            for _ in range(num_runs):
                if is_cuda:
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = adapter(x)
                if is_cuda:
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1000.0)
                if buf > 0:
                    time.sleep(buf)
        results_buf[label] = times
        logger.info(
            "[%-18s] median=%.2f ms | mean=%.2f ms | std=%.2f ms | p90=%.2f ms",
            label,
            np.median(times),
            np.mean(times),
            np.std(times),
            np.percentile(times, 90),
        )
    return results_buf


def plot_throttling_comparison(throttle_res: dict[str, list[float]], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        throttle_res["0ms_no_buffer"],
        label="0ms buffer (unbuffered)",
        alpha=0.8,
        color="#d62728",
    )
    ax.plot(
        throttle_res["200ms_paper_buffer"],
        label="200ms buffer (paper protocol)",
        alpha=0.85,
        color="#2ca02c",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("GPU Power Throttling Check: 0ms vs 200ms Buffer (Table 1 Protocol)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return save_figure(fig, output_path, dpi=300)


def run_deployment_pipeline(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    deploy_config: dict[str, Any] = DEPLOYMENT_CONFIG,
    checkpoint_path: Path | None = None,
    pareto_csv: Path | None = None,
    device_str: str | None = None,
    backup_manager: BackupManager | None = None,
) -> dict[str, Any]:
    """Run ONNX export, TensorRT FP16 build, CUDA Graphs benchmark, and throttling check."""
    paths.ensure_dirs()
    device, amp_dtype, _ = resolve_device_and_amp(device_str)

    ckpt_path = checkpoint_path or (
        paths.nas_ema_ckpt if paths.nas_ema_ckpt.exists() else paths.nas_latest_ckpt
    )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"EMA/NAS checkpoint not found at {ckpt_path}. Run scripts/train_nas.py first."
        )

    official_model, _, _, _ = build_official_rfdetr_supernet(
        dataset_root=paths.dataset_root,
        num_classes=NUM_CLASSES,
        device=device,
        paper_config=paper_config,
    )
    load_ema_weights(Path(ckpt_path), official_model, device=device)
    ema_nas_model = RFDETRNASAdapter(official_model, paper_config=paper_config).to(device).eval()

    resolved_pareto = pareto_csv or (paths.grid_search_output_dir / "pareto_frontier.csv")
    best_cfg = select_best_pareto_subnet(
        resolved_pareto, fallback_cfg=deploy_config["fallback_subnet_cfg"]
    )
    logger.info("Selected Pareto subnet for deployment benchmark: %s", best_cfg)

    onnx_path = paths.deployment_output_dir / "rfdetr_pareto_subnet_opset17.onnx"
    trt_engine_path = paths.deployment_output_dir / "rfdetr_pareto_subnet_fp16.engine"
    exported_onnx = export_subnet_to_onnx(
        ema_nas_model,
        best_cfg,
        onnx_path=onnx_path,
        device=device,
        opset_version=int(deploy_config["opset_version"]),
    )
    trt_stats = build_and_benchmark_tensorrt_fp16(
        onnx_path=onnx_path,
        trt_engine_path=trt_engine_path,
        warmup_ms=int(deploy_config["trt_warmup_ms"]),
        iterations=int(deploy_config["trt_iterations"]),
        avg_runs=int(deploy_config["trt_avg_runs"]),
        use_cuda_graph=True,
    )
    cuda_graph_stats = benchmark_cuda_graphs(
        ema_nas_model,
        best_cfg,
        device=device,
        amp_dtype=amp_dtype,
        warmup=int(deploy_config["cuda_graph_warmup"]),
        repeats=int(deploy_config["cuda_graph_repeats"]),
    )
    throttle_res = compare_buffering_throttling(
        ema_nas_model,
        best_cfg,
        device=device,
        amp_dtype=amp_dtype,
        warmup=int(deploy_config["throttling_warmup"]),
        num_runs=int(deploy_config["throttling_num_runs"]),
    )
    throttle_png = plot_throttling_comparison(
        throttle_res, paths.deployment_output_dir / "throttling_comparison.png"
    )

    summary = {
        "best_subnet_cfg": best_cfg,
        "onnx_exported": exported_onnx is not None,
        "onnx_path": str(exported_onnx) if exported_onnx else None,
        "tensorrt": trt_stats,
        "cuda_graphs": cuda_graph_stats,
        "throttling_summary": {
            k: {
                "median_ms": float(np.median(v)),
                "mean_ms": float(np.mean(v)),
                "std_ms": float(np.std(v)),
                "p90_ms": float(np.percentile(v, 90)),
            }
            for k, v in throttle_res.items()
        },
    }
    summary_json = save_json(summary, paths.deployment_output_dir / "deployment_summary.json")

    if backup_manager is not None:
        sync_list = [summary_json, throttle_png]
        if exported_onnx is not None:
            sync_list.append(exported_onnx)
        if trt_engine_path.exists():
            sync_list.append(trt_engine_path)
        backup_manager.sync_files(sync_list, subdir="deployment")
    return summary
