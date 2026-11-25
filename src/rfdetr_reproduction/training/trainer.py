"""Weight-Sharing NAS training loop (Section 5)."""

from __future__ import annotations

import datetime
import time
from typing import Any, Callable, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.nas_config import NAS_CONFIG
from configs.paper_config import NUM_CLASSES, PAPER
from rfdetr_reproduction.data.dataset import build_dataloader
from rfdetr_reproduction.environment.provenance import collect_provenance
from rfdetr_reproduction.nas.adapter import (
    RFDETRNASAdapter,
    build_official_rfdetr_supernet,
    ensure_coco_pretrain_checkpoint,
    load_coco_pretrain_into_supernet,
)
from rfdetr_reproduction.nas.sampling import sample_uniform_subnet
from rfdetr_reproduction.nas.search_space import build_train_search_space
from rfdetr_reproduction.training.checkpoint import load_nas_checkpoint, save_nas_checkpoints
from rfdetr_reproduction.training.train_utils import (
    SupernetEMA,
    build_paper_optimizer,
    compute_weighted_loss,
    prepare_batch_for_resolution,
    resolve_device_and_amp,
)
from rfdetr_reproduction.utils.backup import BackupManager, save_dataframe, save_json
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.paths import ProjectPaths
from rfdetr_reproduction.utils.seed import seed_everything

logger = get_logger(__name__)


def train_one_nas_epoch(
    nas_model: RFDETRNASAdapter,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_train_configs: Sequence[dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    accum_steps: int,
    grad_clip_norm: float,
    epoch_idx: int,
    total_epochs: int,
    ema_tracker: SupernetEMA | None = None,
    subnet_sampler: Callable[[], dict[str, Any]] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one Weight-Sharing NAS epoch with gradient accumulation and per-step subnet sampling."""
    nas_model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)

    sampler = subnet_sampler or (lambda: sample_uniform_subnet(valid_train_configs))
    total_loss = 0.0
    n_steps = 0
    sampled_hist: list[tuple[int, int, int]] = []
    t_start = time.time()
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    current_cfg = sampler()
    total_loader_steps = len(train_loader) if max_steps is None else min(len(train_loader), max_steps)

    for step, (img_list, targets) in enumerate(
        tqdm(train_loader, total=total_loader_steps, desc=f"NAS Epoch {epoch_idx}/{total_epochs}")
    ):
        if max_steps is not None and step >= max_steps:
            break
        if step % accum_steps == 0:
            current_cfg = sampler()
            sampled_hist.append(
                (current_cfg["resolution"], current_cfg["patch_size"], current_cfg["num_windows"])
            )

        nas_model.activate_subnet(current_cfg, mode="train")
        samples, targets_gpu = prepare_batch_for_resolution(
            img_list, targets, int(current_cfg["resolution"]), device
        )

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=is_cuda):
            outputs = nas_model(samples, targets_gpu)
            loss = compute_weighted_loss(criterion, outputs, targets_gpu)
            loss_scaled = loss / accum_steps

        if scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        total_loss += float(loss.detach().item())
        n_steps += 1

        if (step + 1) % accum_steps == 0 or (step + 1) == total_loader_steps:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(nas_model.parameters(), max_norm=grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_tracker is not None:
                ema_tracker.update(nas_model.model)

        del outputs, loss, loss_scaled

    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if is_cuda else 0.0
    )
    elapsed = time.time() - t_start
    return {
        "epoch": epoch_idx,
        "train_loss": total_loss / max(1, n_steps),
        "unique_nas_configs_sampled": len(set(sampled_hist)),
        "peak_vram_gb": round(peak_vram, 2),
        "elapsed_sec": round(elapsed, 1),
    }


def run_nas_training(
    paths: ProjectPaths,
    paper_config: dict[str, Any] = PAPER,
    nas_config: dict[str, Any] = NAS_CONFIG,
    epochs: int | None = None,
    resume: bool = True,
    use_coco_pretrain: bool | None = None,
    device_str: str | None = None,
    max_steps_per_epoch: int | None = None,
    backup_manager: BackupManager | None = None,
) -> list[dict[str, Any]]:
    """End-to-end Weight-Sharing NAS supernet training with atomic checkpointing."""
    paths.ensure_dirs()
    seed = int(nas_config["seed"])
    seed_everything(seed)

    total_epochs = int(epochs if epochs is not None else nas_config["nas_epochs"])
    micro_batch = int(nas_config["micro_batch"])
    accum_steps = int(nas_config["accum_steps"])
    load_pretrain = (
        bool(nas_config["use_coco_pretrain"]) if use_coco_pretrain is None else use_coco_pretrain
    )

    device, amp_dtype, is_cuda = resolve_device_and_amp(device_str)
    provenance = collect_provenance(seed=seed, require_cuda=False)

    _, valid_train_configs, _ = build_train_search_space(
        paper_config=paper_config,
        min_grid_size=int(nas_config["min_grid_size"]),
        max_tokens=int(nas_config["train_max_tokens_l4"]),
    )
    assert len(valid_train_configs) > 0, "No valid training NAS configs!"

    official_model, criterion, _, _ = build_official_rfdetr_supernet(
        dataset_root=paths.dataset_root,
        num_classes=NUM_CLASSES,
        device=device,
        paper_config=paper_config,
    )
    if load_pretrain and not (resume and paths.nas_latest_ckpt.exists()):
        pretrain_ckpt = ensure_coco_pretrain_checkpoint(paths.coco_pretrain_path)
        load_coco_pretrain_into_supernet(official_model, pretrain_ckpt)

    nas_model = RFDETRNASAdapter(official_model, paper_config=paper_config).to(device)
    optimizer = build_paper_optimizer(
        nas_model,
        base_lr=float(paper_config["lr"]),
        layer_decay=float(paper_config["layerwise_lr_decay"]),
        weight_decay=float(paper_config["weight_decay"]),
    )
    ema_tracker = SupernetEMA(nas_model.model, decay=float(paper_config["ema_decay"]))
    scaler = torch.amp.GradScaler(device.type, enabled=(is_cuda and amp_dtype == torch.float16))

    _, train_loader = build_dataloader(
        split_dir=paths.dataset_root / "train",
        batch_size=micro_batch,
        is_train=True,
        scale_range=tuple(paper_config["multi_scale_factor_range"]),
        num_workers=int(nas_config["num_workers"]),
        pin_memory=is_cuda,
    )

    start_epoch = 1
    train_history: list[dict[str, Any]] = []
    if resume and paths.nas_latest_ckpt.exists():
        start_epoch, train_history, _ = load_nas_checkpoint(
            paths.nas_latest_ckpt,
            model=nas_model.model,
            ema_tracker=ema_tracker,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
    else:
        logger.info("Starting NAS supernet training from epoch 1.")

    save_json(
        {
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "provenance": provenance,
            "paper_config": paper_config,
            "nas_config": nas_config,
            "epochs": total_epochs,
            "use_coco_pretrain": load_pretrain,
        },
        paths.nas_output_dir / "run_config.json",
    )

    history_csv = paths.nas_output_dir / "nas_train_history.csv"
    for ep in range(start_epoch, total_epochs + 1):
        stats = train_one_nas_epoch(
            nas_model=nas_model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_train_configs=valid_train_configs,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            accum_steps=accum_steps,
            grad_clip_norm=float(paper_config["grad_clip_norm"]),
            epoch_idx=ep,
            total_epochs=total_epochs,
            ema_tracker=ema_tracker,
            max_steps=max_steps_per_epoch,
        )
        train_history.append(stats)
        logger.info("Epoch %d stats: %s", ep, stats)

        save_nas_checkpoints(
            latest_path=paths.nas_latest_ckpt,
            ema_only_path=paths.nas_ema_ckpt,
            epoch=ep,
            model=nas_model.model,
            ema_tracker=ema_tracker,
            optimizer=optimizer,
            scaler=scaler,
            history=train_history,
            provenance=provenance,
        )
        save_dataframe(pd.DataFrame(train_history), history_csv)
        if is_cuda:
            torch.cuda.empty_cache()

    if backup_manager is not None:
        backup_manager.sync_files(
            [paths.nas_latest_ckpt, paths.nas_ema_ckpt, history_csv],
            subdir="training",
        )
    return train_history
