"""Dynamic path management driven by CLI arguments or environment variables.

Replaces hardcoded notebook working/drive paths with configurable `--data-dir`,
`--ckpt-dir`, and `--output-dir` flags (or `RFDETR_*` environment variables).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from configs.paper_config import COCO_PRETRAIN_NAME


def _default_project_root() -> Path:
    return Path.cwd()


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path
    ckpt_dir: Path
    output_dir: Path

    @classmethod
    def from_env_or_args(
        cls,
        data_dir: str | Path | None = None,
        ckpt_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> "ProjectPaths":
        root = _default_project_root()
        resolved_data = Path(
            data_dir or os.environ.get("RFDETR_DATA_DIR") or (root / "data")
        ).expanduser().resolve()
        resolved_ckpt = Path(
            ckpt_dir or os.environ.get("RFDETR_CKPT_DIR") or (root / "checkpoints")
        ).expanduser().resolve()
        resolved_out = Path(
            output_dir or os.environ.get("RFDETR_OUTPUT_DIR") or (root / "outputs")
        ).expanduser().resolve()
        return cls(data_dir=resolved_data, ckpt_dir=resolved_ckpt, output_dir=resolved_out)

    # --- Data paths ---
    @property
    def voc_raw_dir(self) -> Path:
        return self.data_dir / "voc_raw"

    @property
    def dataset_root(self) -> Path:
        return self.data_dir / "voc2007_rfdetr"

    @property
    def yolo_dataset_dir(self) -> Path:
        return self.data_dir / "voc2007_yolo_fair"

    # --- Checkpoint paths ---
    @property
    def pretrained_dir(self) -> Path:
        return self.ckpt_dir / "pretrained"

    @property
    def coco_pretrain_path(self) -> Path:
        return self.pretrained_dir / COCO_PRETRAIN_NAME

    @property
    def nas_ckpt_dir(self) -> Path:
        return self.ckpt_dir / "nas"

    @property
    def nas_latest_ckpt(self) -> Path:
        return self.nas_ckpt_dir / "nas_supernet_latest.pth"

    @property
    def nas_ema_ckpt(self) -> Path:
        return self.nas_ckpt_dir / "nas_supernet_ema_only.pth"

    @property
    def diagnostics_ckpt_dir(self) -> Path:
        return self.ckpt_dir / "diagnostics"

    @property
    def hypo1_state_path(self) -> Path:
        return self.diagnostics_ckpt_dir / "hypo1_state.pth"

    # --- Output paths ---
    @property
    def provenance_path(self) -> Path:
        return self.output_dir / "provenance.json"

    @property
    def nas_output_dir(self) -> Path:
        return self.output_dir / "nas"

    @property
    def diagnostics_output_dir(self) -> Path:
        return self.output_dir / "diagnostics"

    @property
    def grid_search_output_dir(self) -> Path:
        return self.output_dir / "grid_search"

    @property
    def deployment_output_dir(self) -> Path:
        return self.output_dir / "deployment"

    @property
    def baselines_output_dir(self) -> Path:
        return self.output_dir / "baselines"

    def ensure_dirs(self) -> "ProjectPaths":
        for d in [
            self.data_dir,
            self.ckpt_dir,
            self.pretrained_dir,
            self.nas_ckpt_dir,
            self.diagnostics_ckpt_dir,
            self.output_dir,
            self.nas_output_dir,
            self.diagnostics_output_dir,
            self.grid_search_output_dir,
            self.deployment_output_dir,
            self.baselines_output_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        return self


def add_path_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register standard `--data-dir`, `--ckpt-dir`, and `--output-dir` CLI flags."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root directory for datasets (or set RFDETR_DATA_DIR).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Root directory for checkpoints (or set RFDETR_CKPT_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for experiment outputs (or set RFDETR_OUTPUT_DIR).",
    )
    return parser
