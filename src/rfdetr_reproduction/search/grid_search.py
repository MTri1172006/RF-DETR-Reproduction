"""Post-Training Grid Search runner on Validation Set (Section 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from configs.experiment_config import GRID_SEARCH_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.data.dataset import SymmetricVOCDetectionDataset
from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter, build_official_rfdetr_supernet
from rfdetr_reproduction.nas.search_space import (
    build_infer_search_space_8ps,
    build_representative_infer_configs,
)
from rfdetr_reproduction.search.evaluation import benchmark_subnet_latency, evaluate_subnet_coco
from rfdetr_reproduction.search.pareto import pareto_frontier, plot_pareto_frontier
from rfdetr_reproduction.training.checkpoint import load_ema_weights
from rfdetr_reproduction.training.train_utils import resolve_device_and_amp
from rfdetr_reproduction.utils.backup import BackupManager, save_dataframe
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths

logger = get_logger(__name__)


def run_grid_search_experiment(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    nas_config: dict[str, Any] = NAS_CONFIG,
    grid_config: dict[str, Any] = GRID_SEARCH_CONFIG,
    mode: str | None = None,
    checkpoint_path: Path | None = None,
    max_images: int | None = None,
    device_str: str | None = None,
    backup_manager: BackupManager | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate validated inference subnets using EMA supernet weights and extract Pareto frontier."""
    paths.ensure_dirs()
    device, amp_dtype, is_cuda = resolve_device_and_amp(device_str)
    search_mode = mode or str(grid_config["mode"])

    ckpt_path = checkpoint_path or (
        paths.nas_ema_ckpt if paths.nas_ema_ckpt.exists() else paths.nas_latest_ckpt
    )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"EMA/NAS checkpoint not found at {ckpt_path}. Run scripts/train_nas.py first."
        )

    official_model, _, postprocessors, _ = build_official_rfdetr_supernet(
        dataset_root=paths.dataset_root,
        num_classes=NUM_CLASSES,
        device=device,
        paper_config=paper_config,
    )
    load_ema_weights(Path(ckpt_path), official_model, device=device)
    ema_nas_model = RFDETRNASAdapter(official_model, paper_config=paper_config).to(device).eval()

    valid_ds = SymmetricVOCDetectionDataset(paths.dataset_root / "valid", is_train=False)

    if search_mode == "full":
        _, configs_to_eval, _ = build_infer_search_space_8ps(
            paper_config=paper_config,
            min_grid_size=int(nas_config["min_grid_size"]),
        )
    else:
        configs_to_eval = build_representative_infer_configs(
            paper_config=paper_config,
            min_grid_size=int(nas_config["min_grid_size"]),
        )

    logger.info("Grid search mode=%s -> Evaluating %d validated subnets...", search_mode, len(configs_to_eval))

    results_csv = paths.grid_search_output_dir / "nas_validation_results.csv"
    pareto_csv = paths.grid_search_output_dir / "pareto_frontier.csv"
    pareto_png = paths.grid_search_output_dir / "nas_pareto_frontier.png"

    done_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if results_csv.exists():
        prev_df = pd.read_csv(results_csv)
        rows = prev_df.to_dict("records")
        done_ids = set(prev_df["config_id"].astype(str).tolist())
        logger.info("Resuming grid search with %d completed configs.", len(done_ids))

    eval_max_images = max_images if max_images is not None else grid_config["max_images"]
    for cfg in tqdm(configs_to_eval, desc="Post-Training NAS Grid Search"):
        if cfg["config_id"] in done_ids:
            continue
        row: dict[str, Any] = {
            "config_id": cfg["config_id"],
            "resolution": cfg["resolution"],
            "patch_size": cfg["patch_size"],
            "num_windows": cfg["num_windows"],
            "decoder_layers": cfg["decoder_layers"],
            "num_queries": cfg["num_queries"],
            "valid": False,
            "error": "",
        }
        try:
            lat_stats = benchmark_subnet_latency(
                ema_nas_model,
                cfg,
                device=device,
                amp_dtype=amp_dtype,
                warmup=int(grid_config["latency_warmup"]),
                repeats=int(grid_config["latency_repeats"]),
                buffer_sec=float(grid_config["latency_buffer_sec"]),
            )
            map_stats = evaluate_subnet_coco(
                ema_nas_model,
                cfg,
                dataset=valid_ds,
                postprocessors=postprocessors,
                device=device,
                amp_dtype=amp_dtype,
                max_images=eval_max_images,
                score_thresh=float(grid_config["score_thresh"]),
            )
            row.update(lat_stats)
            row.update(map_stats)
            row["valid"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:180]
            logger.warning("Subnet %s failed -> %s", cfg["config_id"], row["error"])
            if is_cuda:
                torch.cuda.empty_cache()

        rows.append(row)
        save_dataframe(pd.DataFrame(rows), results_csv)

    results_df = pd.read_csv(results_csv)
    frontier_df = pareto_frontier(results_df, accuracy_col="map", latency_col="latency_median_ms")
    save_dataframe(frontier_df, pareto_csv)
    plot_pareto_frontier(results_df, frontier_df, pareto_png)

    logger.info(
        "Saved %d measurements -> %s | Found %d Pareto-optimal subnets -> %s",
        len(results_df),
        results_csv,
        len(frontier_df),
        pareto_csv,
    )
    if backup_manager is not None:
        backup_manager.sync_files([results_csv, pareto_csv, pareto_png], subdir="grid_search")
    return results_df, frontier_df
