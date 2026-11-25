"""Subnet sampling helpers for Weight-Sharing NAS training and diagnostics."""

from __future__ import annotations

import random
from typing import Any, Sequence


def get_train_cfg(
    valid_train_configs: Sequence[dict[str, Any]],
    res: int,
    ps: int,
    nw: int,
) -> dict[str, Any]:
    """Look up a specific (resolution, patch_size, num_windows) config in `valid_train_configs`."""
    for cfg in valid_train_configs:
        if (
            int(cfg["resolution"]) == res
            and int(cfg["patch_size"]) == ps
            and int(cfg["num_windows"]) == nw
        ):
            return cfg
    raise ValueError(f"Config {(res, ps, nw)} is not present in valid_train_configs!")


def sample_uniform_subnet(
    valid_train_configs: Sequence[dict[str, Any]],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Uniformly sample one subnet configuration."""
    chooser = rng.choice if rng is not None else random.choice
    return chooser(valid_train_configs)


def sample_biased_subnet(
    valid_train_configs: Sequence[dict[str, Any]],
    target_cfg: dict[str, Any],
    other_configs: Sequence[dict[str, Any]],
    p_target: float,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Sample `target_cfg` with probability `p_target`, otherwise uniform from `other_configs`.

    When `p_target == 0.0`, falls back to uniform sampling across `valid_train_configs`
    (matching Hypothesis 2 baseline protocol).
    """
    chooser = rng.choice if rng is not None else random.choice
    rand_fn = rng.random if rng is not None else random.random
    if p_target == 0.0:
        return chooser(valid_train_configs)
    if rand_fn() < p_target:
        return target_cfg
    return chooser(other_configs)
