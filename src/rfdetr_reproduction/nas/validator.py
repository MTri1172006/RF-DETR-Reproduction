"""Shared NAS Search Space Validator (Section 3).

Used uniformly across training, diagnostics, grid search, subnet evaluation, and deployment.
"""

from __future__ import annotations

from typing import Any

from configs.paper_config import PAPER


def validate_nas_config(
    cfg: dict[str, Any],
    mode: str = "inference",
    min_grid_size: int = 8,
    max_tokens: int | None = None,
    require_even_grid: bool = False,
    paper_config: dict[str, Any] = PAPER,
) -> tuple[bool, str]:
    """Validate an RF-DETR NAS subnet configuration.

    Returns:
        `(is_valid, reason)` where `reason == "ok"` when valid.
    """
    res = int(cfg["resolution"])
    ps = int(cfg["patch_size"])
    nw = int(cfg["num_windows"])
    dl = int(cfg["decoder_layers"])
    nq = int(cfg["num_queries"])

    if res <= 0 or ps <= 0 or nw <= 0:
        return False, "non_positive_dimension"
    if res % ps != 0:
        return False, "resolution_not_divisible_by_patch_size"

    grid_size = res // ps
    if grid_size < min_grid_size:
        return False, f"grid_too_small(<{min_grid_size})"
    if grid_size % nw != 0:
        return False, "token_grid_not_divisible_by_num_windows"
    if require_even_grid and (grid_size % 2 != 0):
        return False, "odd_grid_size_for_multiscale_projector"

    num_tokens = grid_size * grid_size
    if max_tokens is not None and num_tokens > max_tokens:
        return False, f"exceeds_token_budget(>{max_tokens})"

    if mode == "train":
        if dl != paper_config["train_decoder_layers"]:
            return False, "train_decoder_layers_must_be_6"
        if nq != paper_config["train_queries"]:
            return False, "train_queries_must_be_300"
    else:
        if dl not in paper_config["inference_decoder_layers"]:
            return False, "invalid_inference_decoder_layers"
        if nq not in paper_config["inference_queries"]:
            return False, "invalid_inference_queries"

    return True, "ok"
