"""ONNX Opset 17 export for a selected Pareto RF-DETR subnet (Section 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from configs.experiment_config import DEPLOYMENT_CONFIG
from rfdetr_reproduction.nas.adapter import RFDETRNASAdapter
from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


class RFDETROnnxWrapper(nn.Module):
    """Wrap an activated NAS subnet to export ONNX with tuple output `(pred_logits, pred_boxes)`."""

    def __init__(self, adapter: RFDETRNASAdapter, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.adapter = adapter
        self.cfg = dict(cfg)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.adapter.activate_subnet(self.cfg, mode="inference")
        out = self.adapter(images)
        return out["pred_logits"], out["pred_boxes"]


def select_best_pareto_subnet(
    pareto_csv: Path | None,
    fallback_cfg: dict[str, Any] = DEPLOYMENT_CONFIG["fallback_subnet_cfg"],
) -> dict[str, Any]:
    """Select the highest-accuracy Pareto subnet from `pareto_frontier.csv`, or fall back."""
    if pareto_csv is not None and Path(pareto_csv).exists():
        frontier_df = pd.read_csv(pareto_csv)
        if len(frontier_df) > 0:
            best_row = frontier_df.iloc[-1]
            return {
                "resolution": int(best_row["resolution"]),
                "patch_size": int(best_row["patch_size"]),
                "num_windows": int(best_row["num_windows"]),
                "decoder_layers": int(best_row["decoder_layers"]),
                "num_queries": int(best_row["num_queries"]),
            }
    return dict(fallback_cfg)


def export_subnet_to_onnx(
    adapter: RFDETRNASAdapter,
    cfg: dict[str, Any],
    onnx_path: Path,
    device: torch.device,
    opset_version: int = 17,
) -> Path | None:
    """Export the chosen subnet to ONNX Opset 17."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    res = int(cfg["resolution"])

    wrapper = RFDETROnnxWrapper(adapter, cfg).eval().to(device)
    dummy_input = torch.randn(1, 3, res, res, device=device)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["pred_logits", "pred_boxes"],
            )
        logger.info(
            "[ONNX] Exported -> %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6
        )
        return onnx_path
    except Exception as exc:
        logger.warning("[ONNX] Export failed: %s", exc)
        return None
