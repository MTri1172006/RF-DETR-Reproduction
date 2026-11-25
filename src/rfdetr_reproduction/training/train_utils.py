"""Training utilities: paper AdamW with layerwise LR decay, EMA, AMP, and batch resizing."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter, get_nested_tensor_helpers


def resolve_device_and_amp(device_str: str | None = None) -> tuple[torch.device, torch.dtype, bool]:
    """Resolve torch device, enable TF32/cuDNN benchmark on CUDA, and pick AMP dtype."""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        amp_dtype = torch.float32
    return device, amp_dtype, is_cuda


def build_paper_optimizer(
    nas_adapter: RFDETRNASAdapter,
    base_lr: float = 1e-4,
    layer_decay: float = 0.8,
    weight_decay: float = 1e-4,
    num_layers: int = 12,
) -> torch.optim.AdamW:
    """AdamW with Layer-wise LR Decay = 0.8 for 12-layer DINOv2-S backbone (Appendix A, scheduler-free)."""
    param_groups: dict[tuple[str, float, float], dict[str, Any]] = {}

    for name, param in nas_adapter.model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            layer_id = 0
            if "embeddings" in name or "patch_embed" in name or "pos_embed" in name:
                layer_id = 0
            else:
                parts = name.split(".")
                for i, token in enumerate(parts):
                    if (
                        token in ("layer", "layers", "blocks")
                        and i + 1 < len(parts)
                        and parts[i + 1].isdigit()
                    ):
                        layer_id = int(parts[i + 1]) + 1
                        break
                else:
                    layer_id = num_layers
            scale = layer_decay ** (num_layers + 1 - layer_id)
            lr = base_lr * scale
            group_key = f"backbone_l{layer_id}"
        else:
            lr = base_lr
            group_key = "detector_head"

        wd = 0.0 if (param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()) else weight_decay
        gk = (group_key, lr, wd)
        if gk not in param_groups:
            param_groups[gk] = {"params": [], "lr": lr, "weight_decay": wd, "name": group_key}
        param_groups[gk]["params"].append(param)

    return torch.optim.AdamW(list(param_groups.values()))


class SupernetEMA:
    """Exponential Moving Average tracker for the RF-DETR supernet."""

    def __init__(self, model: nn.Module, decay: float = 0.993) -> None:
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)


def prepare_batch_for_resolution(
    img_list: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    resolution: int,
    device: torch.device,
):
    """Resize a batch of images to `resolution x resolution` and pack into NestedTensor."""
    _, nested_tensor_from_list = get_nested_tensor_helpers()
    resized_list = [
        F.interpolate(
            im.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False
        ).squeeze(0)
        for im in img_list
    ]
    samples = nested_tensor_from_list(resized_list).to(device)
    targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets_gpu


def compute_weighted_loss(criterion: nn.Module, outputs: dict, targets: list[dict]) -> torch.Tensor:
    """Compute official weighted DETR loss from `SetCriterion`."""
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
