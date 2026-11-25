"""Thin CLI entry point: train and fairly evaluate YOLOv8n/YOLO11n vs RF-DETR-NAS (Section 8)."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.baseline_config import BASELINE_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.baselines.compare import run_fair_baseline_comparison
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair YOLO baseline comparison on VOC2007 test.")
    add_path_args(parser)
    parser.add_argument("--ckpt", type=Path, default=None, help="EMA/NAS checkpoint path.")
    parser.add_argument("--pareto-csv", type=Path, default=None, help="Path to pareto_frontier.csv.")
    parser.add_argument("--epochs", type=int, default=BASELINE_CONFIG["epochs"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_fair_baseline_comparison(
        paths=paths,
        paper_config=PAPER,
        baseline_config=BASELINE_CONFIG,
        checkpoint_path=args.ckpt,
        pareto_csv=args.pareto_csv,
        epochs=args.epochs,
        device_str=args.device,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
