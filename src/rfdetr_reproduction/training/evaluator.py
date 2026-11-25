"""Shared validation-loss evaluator for fixed subnets (Section 5c, used by Hypo 1 & Hypo 2)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter
from rfdetr_reproduction.training.train_utils import (
    compute_weighted_loss,
    prepare_batch_for_resolution,
)


@torch.no_grad()
def evaluate_subnet_val_loss(
    adapter: RFDETRNASAdapter,
    criterion: nn.Module,
    cfg: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> float:
    """Compute average official `SetCriterion` loss of a single subnet on a fixed validation loader."""
    was_training = adapter.training
    crit_was_training = criterion.training
    adapter.eval()
    criterion.eval()
    # Keep 6 decoder layers + 300 queries ("train" mode) so all subnets share the same loss scale
    adapter.activate_subnet(cfg, mode="train")

    res = int(cfg["resolution"])
    total_loss = 0.0
    n_batches = 0
    is_cuda = device.type == "cuda"

    try:
        for b_idx, (img_list, targets) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            samples, targets_gpu = prepare_batch_for_resolution(img_list, targets, res, device)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=is_cuda):
                outputs = adapter(samples, targets_gpu)
                loss = compute_weighted_loss(criterion, outputs, targets_gpu)
            total_loss += float(loss.item())
            n_batches += 1
    finally:
        adapter.restore_supernet()
        adapter.train(was_training)
        criterion.train(crit_was_training)

    return total_loss / max(1, n_batches)
