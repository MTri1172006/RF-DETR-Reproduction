"""Headless Matplotlib plotting helpers (always Agg backend, savefig only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200, tight: bool = True) -> Path:
    """Save a matplotlib figure to disk and close it in headless Agg mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
