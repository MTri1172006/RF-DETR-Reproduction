"""Thin CLI entry point: run Hypothesis 1 per-architecture convergence experiment (Section 5d)."""

from __future__ import annotations

import argparse

from configs.experiment_config import HYPO1_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.diagnostics.hypothesis1 import run_hypothesis1_experiment
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hypothesis 1 convergence curve experiment.")
    add_path_args(parser)
    parser.add_argument("--epochs", type=int, default=HYPO1_CONFIG["epochs"])
    parser.add_argument("--eval-every", type=int, default=HYPO1_CONFIG["eval_every"])
    parser.add_argument("--no-resume", action="store_true", help="Restart Hypo 1 from epoch 0.")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_hypothesis1_experiment(
        paths=paths,
        paper_config=PAPER,
        nas_config=NAS_CONFIG,
        hypo1_config=HYPO1_CONFIG,
        epochs=args.epochs,
        eval_every=args.eval_every,
        resume=not args.no_resume,
        device_str=args.device,
        max_val_batches=args.max_val_batches,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
