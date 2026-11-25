"""Thin CLI entry point: run Hypothesis 2 weight drift / interference experiment (Section 5e)."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment_config import HYPO2_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.diagnostics.hypothesis2 import run_hypothesis2_experiment
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hypothesis 2 weight drift experiment.")
    add_path_args(parser)
    parser.add_argument("--ckpt", type=Path, default=None, help="Base NAS checkpoint path.")
    parser.add_argument("--extra-epochs", type=int, default=HYPO2_CONFIG["extra_epochs"])
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_hypothesis2_experiment(
        paths=paths,
        paper_config=PAPER,
        nas_config=NAS_CONFIG,
        hypo2_config=HYPO2_CONFIG,
        checkpoint_path=args.ckpt,
        extra_epochs=args.extra_epochs,
        device_str=args.device,
        max_val_batches=args.max_val_batches,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
