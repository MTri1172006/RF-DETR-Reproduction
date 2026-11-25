"""Centralized, atomic checkpoint management (independent of Google Drive).

Manages:
  - `checkpoints/nas/nas_supernet_latest.pth`: full resumable state (model, ema_model,
    ema_updates, optimizer, scaler, epoch, history, rng_state, provenance)
  - `checkpoints/nas/nas_supernet_ema_only.pth`: lightweight EMA weights for Grid Search & Export
  - `checkpoints/diagnostics/hypo1_state.pth`: dedicated Hypothesis 1 resume state
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rfdetr_reproduction.training.train_utils import SupernetEMA
from rfdetr_reproduction.utils.logging import get_logger
from rfdetr_reproduction.utils.seed import get_rng_state, set_rng_state

logger = get_logger(__name__)


def atomic_torch_save(payload: dict[str, Any], target_path: Path) -> Path:
    """Write checkpoint to `<path>.tmp` and atomically swap via `os.replace`."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)
    return target_path


def save_nas_checkpoints(
    latest_path: Path,
    ema_only_path: Path,
    epoch: int,
    model: nn.Module,
    ema_tracker: SupernetEMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    history: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Atomically save both full resumable checkpoint and lightweight EMA-only checkpoint."""
    latest_payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema_model": ema_tracker.ema.state_dict(),
        "ema_updates": ema_tracker.updates,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
        "history": history,
        "rng_state": get_rng_state(),
        "provenance": provenance,
    }
    ema_payload = {
        "epoch": epoch,
        "ema_model": ema_tracker.ema.state_dict(),
        "ema_updates": ema_tracker.updates,
        "provenance": provenance,
    }
    atomic_torch_save(latest_payload, latest_path)
    atomic_torch_save(ema_payload, ema_only_path)
    logger.info(
        "Saved atomic checkpoints at epoch %d -> %s (%.1f MB) & %s (%.1f MB)",
        epoch,
        latest_path.name,
        latest_path.stat().st_size / 1e6,
        ema_only_path.name,
        ema_only_path.stat().st_size / 1e6,
    )
    return latest_path, ema_only_path


def load_nas_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    ema_tracker: SupernetEMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """Load full NAS checkpoint and return `(start_epoch, history, raw_ckpt)`."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema_tracker is not None and "ema_model" in ckpt:
        ema_tracker.ema.load_state_dict(ckpt["ema_model"])
        if "ema_updates" in ckpt:
            ema_tracker.updates = int(ckpt["ema_updates"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng_state") is not None:
        set_rng_state(ckpt["rng_state"])

    history = list(ckpt.get("history", []))
    start_epoch = int(ckpt["epoch"]) + 1
    logger.info("Loaded NAS checkpoint from %s (epoch=%d)", ckpt_path, ckpt["epoch"])
    return start_epoch, history, ckpt


def load_ema_weights(
    ckpt_path: Path,
    target_model: nn.Module,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load EMA weights from either `nas_supernet_ema_only.pth` or `nas_supernet_latest.pth`."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_model", ckpt.get("model", ckpt))
    target_model.load_state_dict(state_dict)
    logger.info("Loaded EMA weights from %s (epoch=%s)", ckpt_path, ckpt.get("epoch", "unknown"))
    return ckpt


def save_hypo1_state(
    state_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    logs: list[dict[str, Any]],
) -> Path:
    """Atomically save Hypothesis 1 training & probe state."""
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
        "logs": logs,
        "rng_state": get_rng_state(),
    }
    return atomic_torch_save(payload, state_path)


def load_hypo1_state(
    state_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Load Hypothesis 1 state and restore RNG for exact resume."""
    st = torch.load(str(state_path), map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    optimizer.load_state_dict(st["optimizer"])
    if scaler is not None and st.get("scaler") is not None and scaler.is_enabled():
        scaler.load_state_dict(st["scaler"])
    if "rng_state" in st:
        set_rng_state(st["rng_state"])
    elif "py_rng" in st:
        import random

        random.setstate(st["py_rng"])
    logs = list(st.get("logs", []))
    start_epoch = int(st["epoch"]) + 1
    logger.info("Resumed Hypo 1 from epoch %d (%d logged points)", st["epoch"], len(logs))
    return start_epoch, logs
