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


class AdaptiveNASSampler:
    """Supports 3 Weight-Sharing NAS sampling strategies:

    - `uniform`: baseline uniform sampling across all valid train configs.
    - `pruned_anchor`: Conflict-Guided Pruning (drops p=32, w=4) + 2-phase Anchor-Budgeted
      allocation (Phase 1 uniform on pruned space; Phase 2 ~68% core Pareto / ~32% small anchors).
    - `sandwich`: Sandwich Micro-Batching across 4 micro-steps per optimizer step
      (slot 0 = large Pareto anchor, slot 1 = small anchor, slots 2-3 = pruned/core).
    """

    VALID_STRATEGIES = ("uniform", "pruned_anchor", "sandwich")

    def __init__(
        self,
        valid_configs: Sequence[dict[str, Any]],
        strategy: str = "uniform",
        pruned_patch_sizes: tuple[int, ...] = (12, 14, 16),
        pruned_num_windows: tuple[int, ...] = (1, 2),
        anchor_resolutions: tuple[int, ...] = (320, 384, 448),
        core_resolutions: tuple[int, ...] = (512, 576, 640, 704),
        phase1_epochs: int = 8,
        core_prob: float = 0.68,
        seed: int = 42,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown sampling strategy '{strategy}'. Expected one of {self.VALID_STRATEGIES}."
            )
        self.all_configs = list(valid_configs)
        self.strategy = strategy
        self.phase1_epochs = int(phase1_epochs)
        self.core_prob = float(core_prob)
        self.rng = random.Random(seed)

        # 1. Conflict-Guided Pruning (drop p=32 & w=4)
        self.pruned_configs = [
            c
            for c in self.all_configs
            if int(c["patch_size"]) in pruned_patch_sizes
            and int(c["num_windows"]) in pruned_num_windows
        ] or self.all_configs

        # 2. Anchor pool (small resolutions) & Core Pareto pool (mid-large, p16)
        self.anchor_configs = [
            c for c in self.pruned_configs if int(c["resolution"]) in anchor_resolutions
        ] or self.pruned_configs

        self.core_configs = [
            c
            for c in self.pruned_configs
            if int(c["resolution"]) in core_resolutions and int(c["patch_size"]) == 16
        ] or self.pruned_configs

        # 3. Extremes for Sandwich Micro-Batching
        self.large_anchors = [
            c for c in self.core_configs if int(c["resolution"]) >= 640
        ] or self.core_configs
        self.small_anchors = [
            c
            for c in self.anchor_configs
            if int(c["patch_size"]) == 16 and int(c["resolution"]) <= 384
        ] or self.anchor_configs

    @classmethod
    def from_config(
        cls,
        valid_configs: Sequence[dict[str, Any]],
        nas_config: dict[str, Any],
        strategy: str | None = None,
        seed: int | None = None,
    ) -> "AdaptiveNASSampler":
        return cls(
            valid_configs=valid_configs,
            strategy=strategy or str(nas_config.get("sampling_strategy", "uniform")),
            pruned_patch_sizes=tuple(nas_config.get("pruned_patch_sizes", (12, 14, 16))),
            pruned_num_windows=tuple(nas_config.get("pruned_num_windows", (1, 2))),
            anchor_resolutions=tuple(nas_config.get("anchor_resolutions", (320, 384, 448))),
            core_resolutions=tuple(nas_config.get("core_resolutions", (512, 576, 640, 704))),
            phase1_epochs=int(nas_config.get("phase1_epochs", 8)),
            core_prob=float(nas_config.get("core_pareto_prob", 0.68)),
            seed=int(seed if seed is not None else nas_config.get("seed", 42)),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "all_configs": len(self.all_configs),
            "pruned_configs": len(self.pruned_configs),
            "anchor_configs": len(self.anchor_configs),
            "core_configs": len(self.core_configs),
            "large_anchors": len(self.large_anchors),
            "small_anchors": len(self.small_anchors),
            "phase1_epochs": self.phase1_epochs,
            "core_prob": self.core_prob,
        }

    def sample(self, epoch: int, micro_step_idx: int = 0) -> dict[str, Any]:
        if self.strategy == "uniform":
            return self.rng.choice(self.all_configs)

        if self.strategy == "pruned_anchor":
            if epoch <= self.phase1_epochs:
                return self.rng.choice(self.pruned_configs)
            if self.rng.random() < self.core_prob:
                return self.rng.choice(self.core_configs)
            return self.rng.choice(self.anchor_configs)

        if self.strategy == "sandwich":
            slot = micro_step_idx % 4
            if slot == 0:
                return self.rng.choice(self.large_anchors)
            if slot == 1:
                return self.rng.choice(self.small_anchors)
            if epoch <= self.phase1_epochs:
                return self.rng.choice(self.pruned_configs)
            return self.rng.choice(self.core_configs)

        raise ValueError(f"Unknown sampling strategy: {self.strategy}")
