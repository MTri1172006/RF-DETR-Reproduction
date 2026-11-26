"""Thin CLI entry point: train Weight-Sharing NAS RF-DETR supernet."""

from __future__ import annotations

import argparse

from configs.nas_config import NAS_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.training.trainer import run_nas_training
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR Weight-Sharing NAS supernet.")
    add_path_args(parser)
    parser.add_argument("--epochs", type=int, default=NAS_CONFIG["nas_epochs"])
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default=NAS_CONFIG["sampling_strategy"],
        choices=["uniform", "pruned_anchor", "sandwich"],
        help="NAS supernet sampling strategy (default: uniform).",
    )
    parser.add_argument(
        "--exp-tag",
        "--exp-name",
        dest="exp_tag",
        type=str,
        default=None,
        help="Optional experiment suffix for checkpoints & logs (e.g., pab_nas, sandwich_nas).",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming.")
    parser.add_argument("--no-pretrain", action="store_true", help="Disable COCO pretrain init.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. cuda, cpu).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap per epoch (smoke test).")
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_nas_training(
        paths=paths,
        paper_config=PAPER,
        nas_config=NAS_CONFIG,
        epochs=args.epochs,
        resume=not args.no_resume,
        use_coco_pretrain=not args.no_pretrain,
        sampling_strategy=args.sampling_strategy,
        exp_tag=args.exp_tag,
        device_str=args.device,
        max_steps_per_epoch=args.max_steps,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
