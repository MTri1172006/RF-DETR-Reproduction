"""Thin CLI entry point: run Gradient Conflict diagnosis across NAS subnets (Section 5b)."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment_config import GRADIENT_CONFLICT_CONFIG
from configs.nas_config import NAS_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.diagnostics.gradient_conflict import run_gradient_conflict_experiment
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gradient Conflict diagnosis across subnets.")
    add_path_args(parser)
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional checkpoint path.")
    parser.add_argument("--use-ema", action="store_true", help="Evaluate EMA weights instead of train weights.")
    parser.add_argument("--tag", type=str, default=None, help="Output filename tag (e.g., ep0, ep20).")
    parser.add_argument("--num-batches", type=int, default=GRADIENT_CONFLICT_CONFIG["num_batches"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    gc_cfg = {**GRADIENT_CONFLICT_CONFIG, "num_batches": args.num_batches}
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_gradient_conflict_experiment(
        paths=paths,
        paper_config=PAPER,
        nas_config=NAS_CONFIG,
        gc_config=gc_cfg,
        checkpoint_path=args.ckpt,
        use_ema=args.use_ema,
        tag=args.tag,
        device_str=args.device,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
