"""Pareto frontier filtering and headless plotting (Section 6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfdetr_reproduction.utils.plotting import plt, save_figure


def pareto_frontier(
    df: pd.DataFrame,
    accuracy_col: str = "map",
    latency_col: str = "latency_median_ms",
) -> pd.DataFrame:
    """Return non-dominated subnets sorted by ascending latency."""
    valid_df = df[df["valid"] == True].dropna(subset=[accuracy_col, latency_col]).copy()  # noqa: E712
    keep = []
    for idx, row in valid_df.iterrows():
        dominated = (
            (valid_df[accuracy_col] >= row[accuracy_col])
            & (valid_df[latency_col] <= row[latency_col])
            & ((valid_df[accuracy_col] > row[accuracy_col]) | (valid_df[latency_col] < row[latency_col]))
        ).any()
        if not dominated:
            keep.append(idx)
    return valid_df.loc[keep].sort_values(latency_col)


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot evaluated subnets and Pareto frontier in headless mode and save to disk."""
    valid_df = results_df[results_df["valid"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        valid_df["latency_median_ms"],
        valid_df["map"] * 100.0,
        alpha=0.45,
        s=45,
        color="#7f7f7f",
        label="Evaluated Subnets",
    )
    ax.plot(
        frontier_df["latency_median_ms"],
        frontier_df["map"] * 100.0,
        marker="o",
        linewidth=2.5,
        color="#2ca02c",
        label="RF-DETR Pareto Frontier (Ours)",
    )
    for _, row in frontier_df.iterrows():
        lbl = (
            f"r{int(row['resolution'])}/p{int(row['patch_size'])}/"
            f"d{int(row['decoder_layers'])}/q{int(row['num_queries'])}"
        )
        ax.annotate(
            lbl,
            (row["latency_median_ms"] + 0.15, row["map"] * 100.0 + 0.25),
            fontsize=8,
            color="#1b611b",
        )

    ax.set_title(
        "RF-DETR Post-Training Weight-Sharing NAS: Pareto Frontier on VOC2007 Valid",
        fontweight="bold",
    )
    ax.set_xlabel("PyTorch Eager Latency (ms, median with 200ms buffer)")
    ax.set_ylabel("mAP@50:95 (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    return save_figure(fig, output_path, dpi=300)
