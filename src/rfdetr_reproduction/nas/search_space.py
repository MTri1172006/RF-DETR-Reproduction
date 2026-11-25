"""Validated RF-DETR NAS search space builders (Section 3 & Section 6)."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

from configs.nas_config import (
    MIN_GRID_SIZE,
    REPRESENTATIVE_INFER_GRID,
    TRAIN_MAX_TOKENS_L4,
    build_nas_infer_grid_7ps,
    build_nas_infer_grid_8ps,
    build_nas_train_grid,
)
from configs.paper_config import PAPER
from rfdetr_reproduction.nas.validator import validate_nas_config


def build_validated_search_space(
    grid_dict: dict[str, list[Any]],
    mode: str = "inference",
    min_grid_size: int = MIN_GRID_SIZE,
    max_tokens: int | None = None,
    require_even_grid: bool = False,
    paper_config: dict[str, Any] = PAPER,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Expand a Cartesian grid and filter it through `validate_nas_config`."""
    keys = list(grid_dict.keys())
    raw_configs = [dict(zip(keys, vals)) for vals in product(*grid_dict.values())]
    valid_configs: list[dict[str, Any]] = []
    rejection_counts: Counter = Counter()

    for cfg in raw_configs:
        ok, reason = validate_nas_config(
            cfg,
            mode=mode,
            min_grid_size=min_grid_size,
            max_tokens=max_tokens,
            require_even_grid=require_even_grid,
            paper_config=paper_config,
        )
        if ok:
            cfg_copy = dict(cfg)
            cfg_copy["config_id"] = f"{mode}_{len(valid_configs):04d}"
            cfg_copy["grid_size"] = cfg["resolution"] // cfg["patch_size"]
            cfg_copy["num_tokens"] = cfg_copy["grid_size"] ** 2
            valid_configs.append(cfg_copy)
        else:
            rejection_counts[reason] += 1

    return raw_configs, valid_configs, rejection_counts


def build_train_search_space(
    paper_config: dict[str, Any] = PAPER,
    min_grid_size: int = MIN_GRID_SIZE,
    max_tokens: int | None = TRAIN_MAX_TOKENS_L4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Build validated training search space (111 valid configs for L4 token budget <= 5776)."""
    grid = build_nas_train_grid(paper_config)
    return build_validated_search_space(
        grid,
        mode="train",
        min_grid_size=min_grid_size,
        max_tokens=max_tokens,
        paper_config=paper_config,
    )


def build_infer_search_space_8ps(
    paper_config: dict[str, Any] = PAPER,
    min_grid_size: int = MIN_GRID_SIZE,
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Build validated inference search space with all 8 listed patch sizes (3,948 valid)."""
    grid = build_nas_infer_grid_8ps(paper_config)
    return build_validated_search_space(
        grid,
        mode="inference",
        min_grid_size=min_grid_size,
        max_tokens=max_tokens,
        paper_config=paper_config,
    )


def build_infer_search_space_7ps(
    paper_config: dict[str, Any] = PAPER,
    min_grid_size: int = MIN_GRID_SIZE,
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Build validated inference search space with 7 training patch sizes (6,468 raw / 3,780 valid)."""
    grid = build_nas_infer_grid_7ps(paper_config)
    return build_validated_search_space(
        grid,
        mode="inference",
        min_grid_size=min_grid_size,
        max_tokens=max_tokens,
        paper_config=paper_config,
    )


def build_representative_infer_configs(
    paper_config: dict[str, Any] = PAPER,
    min_grid_size: int = MIN_GRID_SIZE,
) -> list[dict[str, Any]]:
    """Build the representative inference subset used in Section 6 Grid Search."""
    _, rep_valid, _ = build_validated_search_space(
        REPRESENTATIVE_INFER_GRID,
        mode="inference",
        min_grid_size=min_grid_size,
        paper_config=paper_config,
    )
    return [
        c
        for c in rep_valid
        if (c["resolution"] == 384 and c["patch_size"] == 16 and c["num_windows"] == 2)
        or (c["decoder_layers"] in (2, 6) and c["num_queries"] == 300)
    ]
