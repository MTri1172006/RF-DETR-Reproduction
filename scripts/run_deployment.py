"""Thin CLI entry point: ONNX export, TensorRT FP16, CUDA Graphs, and Power Throttling (Section 7)."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment_config import DEPLOYMENT_CONFIG
from configs.paper_config import PAPER
from rfdetr_reproduction.deployment.throttling import run_deployment_pipeline
from rfdetr_reproduction.utils.backup import BackupManager
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ONNX/TensorRT and benchmark throttling.")
    add_path_args(parser)
    parser.add_argument("--ckpt", type=Path, default=None, help="EMA/NAS checkpoint path.")
    parser.add_argument("--pareto-csv", type=Path, default=None, help="Path to pareto_frontier.csv.")
    parser.add_argument("--num-runs", type=int, default=DEPLOYMENT_CONFIG["throttling_num_runs"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backup-backend", choices=["local", "hf", "wandb", "s3"], default="local")
    parser.add_argument("--backup-dest", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    deploy_cfg = {**DEPLOYMENT_CONFIG, "throttling_num_runs": args.num_runs}
    backup = BackupManager(backend=args.backup_backend, destination=args.backup_dest)
    run_deployment_pipeline(
        paths=paths,
        paper_config=PAPER,
        deploy_config=deploy_cfg,
        checkpoint_path=args.ckpt,
        pareto_csv=args.pareto_csv,
        device_str=args.device,
        backup_manager=backup,
    )


if __name__ == "__main__":
    main()
