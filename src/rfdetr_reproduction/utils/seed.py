"""Deterministic RNG seeding and state capture/restore helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU & CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states for deterministic resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG states captured by `get_rng_state`."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state and state["torch"] is not None:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass
