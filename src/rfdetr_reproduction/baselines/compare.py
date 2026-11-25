"""Fair head-to-head comparison between YOLOv8n/YOLO11n and RF-DETR-NAS on VOC2007 test (Section 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from configs.baseline_config import BASELINE_CONFIG
from configs.experiment_config import DEPLOYMENT_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.baselines.train_yolo import (
    evaluate_yolo_on_coco_split,
    train_yolo_baseline,
)
from rfdetr_reproduction.baselines.yolo_dataset import export_coco_to_yolo
from rfdetr_reproduction.data.dataset import SymmetricVOCDetectionDataset
from rfdetr_reproduction.deployment.onnx_export import select_best_pareto_subnet
from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter, build_official_rfdetr_supernet
from rfdetr_reproduction.search.evaluation import benchmark_subnet_latency, evaluate_subnet_coco
from rfdetr_reproduction.training.checkpoint import load_ema_weights
from rfdetr_reproduction.training.train_utils import resolve_device_and_amp
from rfdetr_reproduction.utils.backup import BackupManager, save_dataframe
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths

logger = get_logger(__name__)


def run_fair_baseline_comparison(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    baseline_config: dict[str, Any] = BASELINE_CONFIG,
    checkpoint_path: Path | None = None,
    pareto_csv: Path | None = None,
    epochs: int | None = None,
    device_str: str | None = None,
    backup_manager: BackupManager | None = None,
) -> pd.DataFrame:
    """Train YOLOv8n/YOLO11n and evaluate alongside best RF-DETR-NAS subnet on VOC2007 test."""
    from ultralytics import YOLO

    paths.ensure_dirs()
    device, amp_dtype, _ = resolve_device_and_amp(device_str)
    compare_res = int(baseline_config["compare_res"])
    total_epochs = int(epochs if epochs is not None else baseline_config["epochs"])

    yolo_yaml = export_coco_to_yolo(paths.dataset_root, paths.yolo_dataset_dir)
    yolo_runs_dir = paths.baselines_output_dir / "yolo_runs"

    baseline_rows: list[dict[str, Any]] = []
    for model_tag, weights in baseline_config["models"]:
        best_pt = train_yolo_baseline(
            weights=weights,
            yolo_yaml=yolo_yaml,
            project_dir=yolo_runs_dir,
            model_tag=model_tag,
            epochs=total_epochs,
            imgsz=compare_res,
            batch_size=int(baseline_config["batch_size"]),
            seed=int(baseline_config["seed"]),
        )
        y_eval = YOLO(str(best_pt))
        metrics_test = evaluate_yolo_on_coco_split(
            y_eval,
            split_dir=paths.dataset_root / "test",
            imgsz=compare_res,
            warmup=int(baseline_config["latency_warmup"]),
            repeats_lat=int(baseline_config["latency_repeats"]),
            buffer_sec=float(baseline_config["latency_buffer_sec"]),
            score_thresh=float(baseline_config["score_thresh"]),
        )
        baseline_rows.append({"model": model_tag, "resolution": compare_res, **metrics_test})
        logger.info("[%s on TEST] %s", model_tag, metrics_test)

    ckpt_path = checkpoint_path or (
        paths.nas_ema_ckpt if paths.nas_ema_ckpt.exists() else paths.nas_latest_ckpt
    )
    if Path(ckpt_path).exists():
        official_model, _, postprocessors, _ = build_official_rfdetr_supernet(
            dataset_root=paths.dataset_root,
            num_classes=NUM_CLASSES,
            device=device,
            paper_config=paper_config,
        )
        load_ema_weights(Path(ckpt_path), official_model, device=device)
        ema_nas_model = RFDETRNASAdapter(official_model, paper_config=paper_config).to(device).eval()

        resolved_pareto = pareto_csv or (paths.grid_search_output_dir / "pareto_frontier.csv")
        best_cfg = select_best_pareto_subnet(
            resolved_pareto, fallback_cfg=DEPLOYMENT_CONFIG["fallback_subnet_cfg"]
        )
        test_ds = SymmetricVOCDetectionDataset(paths.dataset_root / "test", is_train=False)
        rf_test_map = evaluate_subnet_coco(
            ema_nas_model,
            best_cfg,
            dataset=test_ds,
            postprocessors=postprocessors,
            device=device,
            amp_dtype=amp_dtype,
            score_thresh=float(baseline_config["score_thresh"]),
        )
        rf_test_lat = benchmark_subnet_latency(
            ema_nas_model,
            best_cfg,
            device=device,
            amp_dtype=amp_dtype,
            warmup=int(baseline_config["latency_warmup"]),
            repeats=int(baseline_config["latency_repeats"]),
            buffer_sec=float(baseline_config["latency_buffer_sec"]),
        )
        baseline_rows.append(
            {
                "model": f"RF-DETR-NAS (r{best_cfg['resolution']}/d{best_cfg['decoder_layers']})",
                "resolution": best_cfg["resolution"],
                "map": rf_test_map["map"],
                "map50": rf_test_map["map50"],
                "latency_median_ms": rf_test_lat["latency_median_ms"],
            }
        )
    else:
        logger.warning("RF-DETR checkpoint not found at %s; skipping RF-DETR row.", ckpt_path)

    comp_df = pd.DataFrame(baseline_rows)
    out_csv = save_dataframe(comp_df, paths.baselines_output_dir / "fair_baseline_comparison_test.csv")
    logger.info("Fair baseline comparison:\n%s", comp_df.to_string(index=False))

    if backup_manager is not None:
        backup_manager.sync_files(
            [out_csv, yolo_runs_dir],
            subdir="baselines",
            exts=(".csv", ".png", ".jpg", ".yaml", ".pt"),
        )
    return comp_df
