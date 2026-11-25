"""Thin CLI entry point: run Post-Training NAS Grid Search & Pareto Frontier (Section 6)."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment_config import GRID_SEARCH_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.search.grid_search import run_grid_search_experiment
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Post-Training NAS Grid Search & Pareto.")
    add_path_args(parser)
    parser.add_argument(
        "--mode",
        choices=["representative", "full"],
        default=GRID_SEARCH_CONFIG["mode"],
    )
    parser.add_argument("--ckpt", type=Path, default=None, help="EMA/NAS checkpoint path.")
    parser.add_argument("--max-images", type=int, default=None, help="Cap valid images (smoke test).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_grid_search_experiment(
        paths=paths,
        paper_config=PAPER,
        nas_config=NAS_CONFIG,
        grid_config=GRID_SEARCH_CONFIG,
        mode=args.mode,
        checkpoint_path=args.ckpt,
        max_images=args.max_images,
        device_str=args.device,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
