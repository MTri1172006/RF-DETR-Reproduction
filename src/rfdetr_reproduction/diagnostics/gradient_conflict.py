"""Gradient Conflict Diagnosis across Weight-Sharing NAS Subnets (Section 5b)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from configs.experiment_config import GRADIENT_CONFLICT_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.data.dataset import SymmetricVOCDetectionDataset, nas_collate_fn
from rfdetr_reproduction.nas.adapter import (
    RFDETRNASAdapter,
    build_official_rfdetr_supernet,
    ensure_coco_pretrain_checkpoint,
    load_coco_pretrain_into_supernet,
)
from rfdetr_reproduction.nas.search_space import build_train_search_space
from rfdetr_reproduction.training.checkpoint import load_ema_weights, load_nas_checkpoint
from rfdetr_reproduction.training.train_utils import (
    compute_weighted_loss,
    prepare_batch_for_resolution,
    resolve_device_and_amp,
)
from rfdetr_reproduction.utils.backup import BackupManager, save_dataframe
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths
from rfdetr_reproduction.utils.plotting import plt, save_figure

logger = get_logger(__name__)


def _gc_pick_cfgs(
    candidates: Sequence[tuple[int, int, int]],
    valid_train_configs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lut = {(c["resolution"], c["patch_size"], c["num_windows"]): c for c in valid_train_configs}
    picked = [lut[t] for t in candidates if t in lut]
    skipped = [t for t in candidates if t not in lut]
    if skipped:
        logger.warning("[GC] Skipped invalid configs per validator: %s", skipped)
    return picked


def _gc_param_groups(adapter: RFDETRNASAdapter) -> dict[str, list[nn.Parameter]]:
    patch_ids = {id(p) for p in adapter.patch_conv.parameters()}
    groups: dict[str, list[nn.Parameter]] = {
        "patch_embed": [],
        "backbone": [],
        "decoder": [],
        "other_head": [],
    }
    for name, param in adapter.model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in patch_ids:
            groups["patch_embed"].append(param)
        elif name.startswith("backbone"):
            groups["backbone"].append(param)
        elif name.startswith("transformer.decoder"):
            groups["decoder"].append(param)
        else:
            groups["other_head"].append(param)
    return {g: ps for g, ps in groups.items() if ps}


def _gc_flat_grads(
    adapter: RFDETRNASAdapter,
    criterion: nn.Module,
    cfg: dict[str, Any],
    img_list: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    groups: dict[str, list[nn.Parameter]],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], float]:
    adapter.zero_grad(set_to_none=True)
    adapter.activate_subnet(cfg, mode="train")
    samples, targets_gpu = prepare_batch_for_resolution(
        img_list, targets, int(cfg["resolution"]), device
    )

    is_cuda = device.type == "cuda"
    gc_scale = 1024.0 if (is_cuda and amp_dtype == torch.float16) else 1.0
    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=is_cuda):
        outputs = adapter(samples, targets_gpu)
        loss = compute_weighted_loss(criterion, outputs, targets_gpu)
    (loss * gc_scale).backward()

    flats = {}
    for g, ps in groups.items():
        flats[g] = (
            torch.cat(
                [
                    (
                        p.grad.detach().float().flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=p.device)
                    )
                    for p in ps
                ]
            )
            / gc_scale
        )
    adapter.zero_grad(set_to_none=True)
    return flats, float(loss.detach())


def run_gradient_conflict_diagnosis(
    adapter: RFDETRNASAdapter,
    criterion: nn.Module,
    train_ds: Dataset,
    valid_train_configs: Sequence[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
    output_dir: Path,
    candidates: Sequence[tuple[int, int, int]] = GRADIENT_CONFLICT_CONFIG["candidates"],
    num_batches: int = GRADIENT_CONFLICT_CONFIG["num_batches"],
    batch_size: int = GRADIENT_CONFLICT_CONFIG["batch_size"],
    seed: int = GRADIENT_CONFLICT_CONFIG["seed"],
    tag: str = "ep0",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute pairwise gradient cosine similarities across subnets and save CSVs + heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfgs = _gc_pick_cfgs(candidates, valid_train_configs)
    assert len(cfgs) >= 2, "Need at least 2 valid subnets to diagnose gradient conflict."
    labels = [f"r{c['resolution']}/p{c['patch_size']}/w{c['num_windows']}" for c in cfgs]
    k_cfgs = len(cfgs)

    was_training = adapter.training
    saved_grads = [
        (p, p.grad.detach().clone() if p.grad is not None else None) for p in adapter.parameters()
    ]
    py_state, torch_state = random.getstate(), torch.get_rng_state()

    adapter.train()
    criterion.train()
    adapter.restore_supernet()
    groups = _gc_param_groups(adapter)
    logger.info("[GC] #params/group: %s", {g: sum(p.numel() for p in ps) for g, ps in groups.items()})

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=nas_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    it = iter(loader)

    cos_sum = {g: torch.zeros(k_cfgs, k_cfgs) for g in groups}
    norm_sum = {g: torch.zeros(k_cfgs) for g in groups}
    self_cos: dict[str, list[torch.Tensor]] = {g: [] for g in groups}
    prev_unit: dict[str, torch.Tensor] | None = None
    losses = torch.zeros(k_cfgs)

    try:
        for _ in tqdm(range(num_batches), desc="Gradient-conflict diagnosis"):
            img_list, targets = next(it)
            per_cfg = []
            for i, cfg in enumerate(cfgs):
                f, loss_val = _gc_flat_grads(
                    adapter, criterion, cfg, img_list, targets, groups, device, amp_dtype
                )
                per_cfg.append(f)
                losses[i] += loss_val
            cur_unit = {}
            for g in groups:
                grad_mat = torch.stack([f[g] for f in per_cfg])
                unit_mat = F.normalize(grad_mat, dim=1)
                cos_sum[g] += (unit_mat @ unit_mat.T).cpu()
                norm_sum[g] += grad_mat.norm(dim=1).cpu()
                cur_unit[g] = unit_mat
                if prev_unit is not None:
                    self_cos[g].append((unit_mat * prev_unit[g]).sum(dim=1).cpu())
            prev_unit = cur_unit
            del per_cfg
    finally:
        for param, gsv in saved_grads:
            param.grad = gsv
        adapter.restore_supernet()
        adapter.train(was_training)
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = []
    mats: dict[str, pd.DataFrame] = {}
    off_mask = ~torch.eye(k_cfgs, dtype=torch.bool)
    for g in groups:
        cos_mat = cos_sum[g] / num_batches
        base = (
            torch.stack(self_cos[g]).mean(0)
            if self_cos[g]
            else torch.full((k_cfgs,), float("nan"))
        )
        diag_mat = cos_mat.clone()
        diag_mat[torch.arange(k_cfgs), torch.arange(k_cfgs)] = base
        mats[g] = pd.DataFrame(diag_mat.numpy(), index=labels, columns=labels)
        off = cos_mat[off_mask]
        norms = norm_sum[g] / num_batches
        summary_rows.append(
            {
                "group": g,
                "mean_cos_between_subnets": off.mean().item(),
                "min_cos_between_subnets": off.min().item(),
                "pct_pairs_negative": 100.0 * (off < 0).float().mean().item(),
                "mean_cos_anchor_vs_others": cos_mat[0, 1:].mean().item(),
                "batch_noise_baseline(self_cos)": base.nanmean().item(),
                "grad_norm_cv": (norms.std() / norms.mean().clamp_min(1e-12)).item(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    loss_df = pd.DataFrame({"subnet": labels, "mean_loss": (losses / num_batches).numpy()})

    save_dataframe(summary_df, output_dir / f"gradient_conflict_summary_{tag}.csv")
    save_dataframe(loss_df, output_dir / f"gradient_conflict_losses_{tag}.csv")
    for g, mat in mats.items():
        save_dataframe(mat, output_dir / f"gradient_conflict_cos_{g}_{tag}.csv", index=True)

    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.8))
    axes = np.atleast_1d(axes)
    im = None
    for ax, (g, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat.values, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(k_cfgs))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(k_cfgs))
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(k_cfgs):
            for j in range(k_cfgs):
                ax.text(
                    j,
                    i,
                    f"{mat.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold" if i == j else "normal",
                )
        ax.set_title(f"{g}\n(diagonal = batch-noise baseline)", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="cosine similarity")
    fig.suptitle(
        f"Gradient Cosine Similarity across Subnets (Weight-Sharing NAS) [{tag}]",
        fontweight="bold",
    )
    save_figure(fig, output_dir / f"gradient_conflict_heatmap_{tag}.png", dpi=200, tight=False)
    logger.info("Gradient conflict summary [%s]:\n%s", tag, summary_df.round(4).to_string(index=False))
    return summary_df, mats, loss_df


def run_gradient_conflict_experiment(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    nas_config: dict[str, Any] = NAS_CONFIG,
    gc_config: dict[str, Any] = GRADIENT_CONFLICT_CONFIG,
    checkpoint_path: Path | None = None,
    use_ema: bool = False,
    tag: str | None = None,
    device_str: str | None = None,
    backup_manager: BackupManager | None = None,
):
    """Standalone entry function to run Gradient Conflict diagnosis from a checkpoint or COCO init."""
    paths.ensure_dirs()
    device, amp_dtype, _ = resolve_device_and_amp(device_str)
    _, valid_train_configs, _ = build_train_search_space(
        paper_config=paper_config,
        min_grid_size=int(nas_config["min_grid_size"]),
        max_tokens=int(nas_config["train_max_tokens_l4"]),
    )

    official_model, criterion, _, _ = build_official_rfdetr_supernet(
        dataset_root=paths.dataset_root,
        num_classes=NUM_CLASSES,
        device=device,
        paper_config=paper_config,
    )

    ckpt_to_load = checkpoint_path or (paths.nas_ema_ckpt if use_ema else paths.nas_latest_ckpt)
    resolved_tag = tag
    if ckpt_to_load is not None and Path(ckpt_to_load).exists():
        if use_ema:
            raw = load_ema_weights(Path(ckpt_to_load), official_model, device=device)
            resolved_tag = resolved_tag or f"ema_ep{raw.get('epoch', 0)}"
        else:
            start_ep, _, _ = load_nas_checkpoint(
                Path(ckpt_to_load), model=official_model, device=device, restore_rng=False
            )
            resolved_tag = resolved_tag or f"ep{start_ep - 1}"
    else:
        pretrain = ensure_coco_pretrain_checkpoint(paths.coco_pretrain_path)
        load_coco_pretrain_into_supernet(official_model, pretrain)
        resolved_tag = resolved_tag or "ep0"

    adapter = RFDETRNASAdapter(official_model, paper_config=paper_config).to(device)
    train_ds = SymmetricVOCDetectionDataset(
        paths.dataset_root / "train",
        is_train=True,
        scale_range=tuple(paper_config["multi_scale_factor_range"]),
    )

    summary_df, mats, loss_df = run_gradient_conflict_diagnosis(
        adapter=adapter,
        criterion=criterion,
        train_ds=train_ds,
        valid_train_configs=valid_train_configs,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=paths.diagnostics_output_dir,
        candidates=gc_config["candidates"],
        num_batches=int(gc_config["num_batches"]),
        batch_size=int(gc_config["batch_size"]),
        seed=int(gc_config["seed"]),
        tag=resolved_tag,
    )
    if backup_manager is not None:
        backup_manager.sync_files(
            list(paths.diagnostics_output_dir.glob(f"gradient_conflict_*_{resolved_tag}.*")),
            subdir="diagnostics",
        )
    return summary_df, mats, loss_df
