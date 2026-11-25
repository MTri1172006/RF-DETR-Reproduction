"""Hypothesis 1: Per-architecture convergence curves and T90 analysis (Section 5d)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from configs.experiment_config import HYPO1_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.data.dataset import build_dataloader
from rfdetr_reproduction.nas.adapter import (
    RFDETRNASAdapter,
    build_official_rfdetr_supernet,
    ensure_coco_pretrain_checkpoint,
    load_coco_pretrain_into_supernet,
)
from rfdetr_reproduction.nas.sampling import get_train_cfg
from rfdetr_reproduction.nas.search_space import build_train_search_space
from rfdetr_reproduction.training.checkpoint import load_hypo1_state, save_hypo1_state
from rfdetr_reproduction.training.evaluator import evaluate_subnet_val_loss
from rfdetr_reproduction.training.train_utils import build_paper_optimizer, resolve_device_and_amp
from rfdetr_reproduction.training.trainer import train_one_nas_epoch
from rfdetr_reproduction.utils.backup import BackupManager, save_dataframe
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths
from rfdetr_reproduction.utils.plotting import plt, save_figure
from rfdetr_reproduction.utils.seed import seed_everything

logger = get_logger(__name__)


def compute_hypo1_t90_table(
    df: pd.DataFrame,
    probes: Sequence[tuple[str, dict[str, Any]]],
    total_epochs: int,
) -> pd.DataFrame:
    """Compute T90 convergence epoch and second-half drop percentage for each probe subnet."""
    mid_ep = total_epochs // 2
    rows = []
    ep = df["epoch"].to_numpy()
    for name, _ in probes:
        losses = df[name].to_numpy()
        total_drop = losses[0] - losses[-1]
        if total_drop > 1e-6:
            prog = (losses[0] - losses) / total_drop
            t90 = int(ep[np.argmax(prog >= 0.9)])
        else:
            t90 = np.nan
        i_mid = int(np.argmin(np.abs(ep - mid_ep)))
        late_drop = losses[i_mid] - losses[-1]
        rows.append(
            {
                "probe": name,
                "L0": losses[0],
                "L_end": losses[-1],
                "total_drop": total_drop,
                "T90_epoch": t90,
                f"drop_ep{int(ep[i_mid])}_to_{int(ep[-1])}": late_drop,
                "late_drop_pct_of_total": 100.0 * late_drop / total_drop if total_drop > 1e-6 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_hypothesis1_experiment(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    nas_config: dict[str, Any] = NAS_CONFIG,
    hypo1_config: dict[str, Any] = HYPO1_CONFIG,
    epochs: int | None = None,
    eval_every: int | None = None,
    resume: bool = True,
    device_str: str | None = None,
    max_val_batches: int | None = None,
    backup_manager: BackupManager | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Hypothesis 1 experiment using shared trainer, evaluator, and atomic checkpointing."""
    paths.ensure_dirs()
    seed_everything(int(hypo1_config["seed"]))
    device, amp_dtype, is_cuda = resolve_device_and_amp(device_str)

    total_epochs = int(epochs if epochs is not None else hypo1_config["epochs"])
    eval_interval = int(eval_every if eval_every is not None else hypo1_config["eval_every"])
    micro_batch = int(nas_config["micro_batch"])
    accum_steps = int(nas_config["accum_steps"])

    _, valid_train_configs, _ = build_train_search_space(
        paper_config=paper_config,
        min_grid_size=int(nas_config["min_grid_size"]),
        max_tokens=int(nas_config["train_max_tokens_l4"]),
    )
    probes = [
        (name, get_train_cfg(valid_train_configs, res, ps, nw))
        for name, (res, ps, nw) in hypo1_config["probes"]
    ]

    _, train_loader = build_dataloader(
        paths.dataset_root / "train",
        batch_size=micro_batch,
        is_train=True,
        scale_range=tuple(paper_config["multi_scale_factor_range"]),
        num_workers=int(nas_config["num_workers"]),
        pin_memory=is_cuda,
    )
    _, val_loader = build_dataloader(
        paths.dataset_root / "valid",
        batch_size=micro_batch,
        is_train=False,
        num_workers=int(nas_config["num_workers"]),
        pin_memory=is_cuda,
    )

    h1_raw_model, criterion, _, _ = build_official_rfdetr_supernet(
        dataset_root=paths.dataset_root,
        num_classes=NUM_CLASSES,
        device=device,
        paper_config=paper_config,
    )
    if not (resume and paths.hypo1_state_path.exists()):
        pretrain = ensure_coco_pretrain_checkpoint(paths.coco_pretrain_path)
        load_coco_pretrain_into_supernet(h1_raw_model, pretrain)

    h1_adapter = RFDETRNASAdapter(h1_raw_model, paper_config=paper_config).to(device)
    h1_opt = build_paper_optimizer(
        h1_adapter,
        base_lr=float(paper_config["lr"]),
        layer_decay=float(paper_config["layerwise_lr_decay"]),
        weight_decay=float(paper_config["weight_decay"]),
    )
    h1_scaler = torch.amp.GradScaler(device.type, enabled=(is_cuda and amp_dtype == torch.float16))

    h1_logs: list[dict[str, Any]] = []
    h1_start_ep = 1
    conv_csv = paths.diagnostics_output_dir / "hypo1_convergence.csv"

    def log_probes(ep: int) -> None:
        row: dict[str, Any] = {"epoch": ep}
        for name, cfg in probes:
            row[name] = evaluate_subnet_val_loss(
                h1_adapter,
                criterion,
                cfg,
                val_loader,
                device=device,
                amp_dtype=amp_dtype,
                max_batches=max_val_batches,
            )
        h1_logs.append(row)
        save_dataframe(pd.DataFrame(h1_logs), conv_csv)
        logger.info(
            "[Hypo 1 | Epoch %02d] %s",
            ep,
            " | ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "epoch"),
        )

    if resume and paths.hypo1_state_path.exists():
        h1_start_ep, h1_logs = load_hypo1_state(
            paths.hypo1_state_path, h1_adapter.model, h1_opt, h1_scaler
        )
    else:
        log_probes(0)

    for ep in range(h1_start_ep, total_epochs + 1):
        train_one_nas_epoch(
            nas_model=h1_adapter,
            criterion=criterion,
            optimizer=h1_opt,
            train_loader=train_loader,
            valid_train_configs=valid_train_configs,
            device=device,
            amp_dtype=amp_dtype,
            scaler=h1_scaler,
            accum_steps=accum_steps,
            grad_clip_norm=float(paper_config["grad_clip_norm"]),
            epoch_idx=ep,
            total_epochs=total_epochs,
            ema_tracker=None,
        )
        if ep % eval_interval == 0 or ep == total_epochs:
            log_probes(ep)

        save_hypo1_state(
            paths.hypo1_state_path,
            epoch=ep,
            model=h1_adapter.model,
            optimizer=h1_opt,
            scaler=h1_scaler,
            logs=h1_logs,
        )
        if is_cuda:
            torch.cuda.empty_cache()

    df_h1 = pd.DataFrame(h1_logs).sort_values("epoch").reset_index(drop=True)
    h1_summary = compute_hypo1_t90_table(df_h1, probes, total_epochs)
    summary_csv = paths.diagnostics_output_dir / "hypo1_t90_summary.csv"
    save_dataframe(h1_summary, summary_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, _ in probes:
        axes[0].plot(df_h1["epoch"], df_h1[name], marker="o", label=name)
        l0, l_end = df_h1[name].iloc[0], df_h1[name].iloc[-1]
        denom = max(1e-6, l0 - l_end)
        progress = 100.0 * (l0 - df_h1[name]) / denom
        axes[1].plot(df_h1["epoch"], progress, marker="o", label=name)

    axes[0].set_title("Hypo 1: Validation Loss by Epoch per Architecture")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Loss")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].axhline(90, color="black", linestyle="--", alpha=0.5, label="90% Convergence (T90)")
    axes[1].set_title("Hypo 1: Normalized Convergence Progress (% total drop)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Convergence Progress (%)")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(fontsize=8)

    fig_path = save_figure(fig, paths.diagnostics_output_dir / "hypo1_convergence.png", dpi=200)
    logger.info("Hypo 1 T90 Summary:\n%s", h1_summary.round(4).to_string(index=False))

    if backup_manager is not None:
        backup_manager.sync_files([conv_csv, summary_csv, fig_path], subdir="diagnostics")
    return df_h1, h1_summary
