"""Hypothesis 2: Weight Drift / Interference on small subnets when oversampling large subnet A (Section 5e)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from configs.experiment_config import HYPO2_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.data.dataset import build_dataloader
from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter, build_official_rfdetr_supernet
from rfdetr_reproduction.nas.sampling import get_train_cfg, sample_biased_subnet
from rfdetr_reproduction.nas.search_space import build_train_search_space
from rfdetr_reproduction.training.evaluator import evaluate_subnet_val_loss
from rfdetr_reproduction.training.train_utils import build_paper_optimizer, resolve_device_and_amp
from rfdetr_reproduction.training.trainer import train_one_nas_epoch
from rfdetr_reproduction.utils.backup import BackupManager, save_dataframe
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths
from rfdetr_reproduction.utils.plotting import plt, save_figure
from rfdetr_reproduction.utils.seed import seed_everything

logger = get_logger(__name__)


def run_hypothesis2_experiment(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    nas_config: dict[str, Any] = NAS_CONFIG,
    hypo2_config: dict[str, Any] = HYPO2_CONFIG,
    checkpoint_path: Path | None = None,
    extra_epochs: int | None = None,
    device_str: str | None = None,
    max_val_batches: int | None = None,
    backup_manager: BackupManager | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Continue training from `nas_supernet_latest.pth` under varying p(A) and measure weight drift."""
    paths.ensure_dirs()
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else paths.nas_latest_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Base NAS checkpoint not found at {ckpt_path}. Run scripts/train_nas.py first."
        )

    device, amp_dtype, is_cuda = resolve_device_and_amp(device_str)
    n_extra = int(extra_epochs if extra_epochs is not None else hypo2_config["extra_epochs"])
    probs = [float(p) for p in hypo2_config["probs"]]
    micro_batch = int(nas_config["micro_batch"])
    accum_steps = int(nas_config["accum_steps"])

    _, valid_train_configs, _ = build_train_search_space(
        paper_config=paper_config,
        min_grid_size=int(nas_config["min_grid_size"]),
        max_tokens=int(nas_config["train_max_tokens_l4"]),
    )
    target_big_a = get_train_cfg(valid_train_configs, *hypo2_config["target_big_a"])
    other_configs = [c for c in valid_train_configs if c["config_id"] != target_big_a["config_id"]]
    watch_list = [
        (name, get_train_cfg(valid_train_configs, *spec)) for name, spec in hypo2_config["watch"]
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

    official_model, criterion, _, _ = build_official_rfdetr_supernet(
        dataset_root=paths.dataset_root,
        num_classes=NUM_CLASSES,
        device=device,
        paper_config=paper_config,
    )
    nas_model = RFDETRNASAdapter(official_model, paper_config=paper_config).to(device)

    base_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    base_ep = int(base_ckpt["epoch"])
    if base_ep != int(hypo2_config["expected_base_epoch"]):
        logger.warning(
            "Checkpoint is at epoch %d (expected %d).",
            base_ep,
            hypo2_config["expected_base_epoch"],
        )
    if base_ckpt.get("optimizer") is None:
        logger.warning("Checkpoint has no optimizer state; AdamW moments will restart.")

    h2_csv = paths.diagnostics_output_dir / "hypo2_interference.csv"
    h2_rows: list[dict[str, Any]] = []
    done_probs: set[float] = set()
    if h2_csv.exists():
        prev = pd.read_csv(h2_csv)
        for p_val, grp in prev.groupby("p_big"):
            if grp["extra_epoch"].max() >= n_extra:
                done_probs.add(round(float(p_val), 4))
                h2_rows.extend(grp.to_dict("records"))
        if done_probs:
            logger.info("[Hypo 2] Skipping completed branches: %s", sorted(done_probs))

    try:
        for p_big in probs:
            if round(p_big, 4) in done_probs:
                continue
            logger.info("=== [Hypo 2] Branch p(A=640/p16/w2) = %.2f from Epoch %d ===", p_big, base_ep)
            seed_everything(int(hypo2_config["seed"]))

            nas_model.model.load_state_dict(base_ckpt["model"])
            nas_model.restore_supernet()
            h2_opt = build_paper_optimizer(
                nas_model,
                base_lr=float(paper_config["lr"]),
                layer_decay=float(paper_config["layerwise_lr_decay"]),
                weight_decay=float(paper_config["weight_decay"]),
            )
            if base_ckpt.get("optimizer") is not None:
                h2_opt.load_state_dict(base_ckpt["optimizer"])
            h2_scaler = torch.amp.GradScaler(
                device.type, enabled=(is_cuda and amp_dtype == torch.float16)
            )

            branch_rows: list[dict[str, Any]] = []
            base_losses: dict[str, float] = {}
            row0: dict[str, Any] = {"p_big": p_big, "extra_epoch": 0, "frac_steps_A": np.nan}
            for name, cfg in watch_list:
                base_losses[name] = evaluate_subnet_val_loss(
                    nas_model, criterion, cfg, val_loader, device, amp_dtype, max_val_batches
                )
                row0[name] = base_losses[name]
                row0[f"delta_{name}"] = 0.0
            branch_rows.append(row0)

            for ex_ep in range(1, n_extra + 1):
                step_counts = {"total": 0, "target_a": 0}

                def _sampler() -> dict[str, Any]:
                    cfg = sample_biased_subnet(
                        valid_train_configs, target_big_a, other_configs, p_big
                    )
                    step_counts["total"] += 1
                    step_counts["target_a"] += int(cfg["config_id"] == target_big_a["config_id"])
                    return cfg

                train_one_nas_epoch(
                    nas_model=nas_model,
                    criterion=criterion,
                    optimizer=h2_opt,
                    train_loader=train_loader,
                    valid_train_configs=valid_train_configs,
                    device=device,
                    amp_dtype=amp_dtype,
                    scaler=h2_scaler,
                    accum_steps=accum_steps,
                    grad_clip_norm=float(paper_config["grad_clip_norm"]),
                    epoch_idx=ex_ep,
                    total_epochs=n_extra,
                    ema_tracker=None,
                    subnet_sampler=_sampler,
                )

                row: dict[str, Any] = {
                    "p_big": p_big,
                    "extra_epoch": ex_ep,
                    "frac_steps_A": step_counts["target_a"] / max(1, step_counts["total"]),
                }
                for name, cfg in watch_list:
                    val_l = evaluate_subnet_val_loss(
                        nas_model, criterion, cfg, val_loader, device, amp_dtype, max_val_batches
                    )
                    row[name] = val_l
                    row[f"delta_{name}"] = val_l - base_losses[name]
                branch_rows.append(row)
                logger.info(
                    "  [pA=%.2f | +%d ep | A-steps=%.2f] %s",
                    p_big,
                    ex_ep,
                    row["frac_steps_A"],
                    " | ".join(
                        f"{n}: {row[n]:.3f} ({row[f'delta_{n}']:+.3f})" for n, _ in watch_list
                    ),
                )

            h2_rows.extend(branch_rows)
            save_dataframe(pd.DataFrame(h2_rows), h2_csv)
            del h2_opt
            if is_cuda:
                torch.cuda.empty_cache()
    finally:
        nas_model.model.load_state_dict(base_ckpt["model"])
        nas_model.restore_supernet()

    df_h2 = pd.DataFrame(h2_rows)
    final = df_h2[df_h2["extra_epoch"] == n_extra].set_index("p_big")
    delta_cols = [f"delta_{n}" for n, _ in watch_list]
    h2_summary = final[["frac_steps_A"] + delta_cols].copy()
    if 0.0 in h2_summary.index:
        for col in delta_cols:
            h2_summary[f"{col}_vs_uniform"] = h2_summary[col] - h2_summary.loc[0.0, col]
    summary_csv = paths.diagnostics_output_dir / "hypo2_summary.csv"
    save_dataframe(h2_summary, summary_csv, index=True)

    fig, axes = plt.subplots(1, len(watch_list), figsize=(16, 4.0), sharey=False)
    for ax, (name, _) in zip(axes, watch_list):
        for p_big in probs:
            sub = df_h2[df_h2["p_big"] == p_big].sort_values("extra_epoch")
            ax.plot(sub["extra_epoch"], sub[f"delta_{name}"], marker="o", label=f"p(A)={p_big:.2f}")
        ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Extra Epochs (from Ep {base_ep})")
        ax.set_ylabel("Delta Val Loss (+ = worse)")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Hypo 2: Validation Loss Drift (Delta L) when Oversampling Large Subnet A (640/p16/w2)",
        fontweight="bold",
    )
    fig_path = save_figure(fig, paths.diagnostics_output_dir / "hypo2_interference.png", dpi=200)
    logger.info("Hypo 2 Summary:\n%s", h2_summary.round(4).to_string())

    if backup_manager is not None:
        backup_manager.sync_files([h2_csv, summary_csv, fig_path], subdir="diagnostics")
    return df_h2, h2_summary
