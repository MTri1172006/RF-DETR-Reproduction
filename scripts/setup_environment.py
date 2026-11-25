"""Thin CLI entry point: verify environment, configure CUDA backend, and save provenance.json."""

from __future__ import annotations

import argparse
import json

from configs.nas_config import SEED
from rfdetr_reproduction.environment.provenance import record_provenance
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args
from rfdetr_reproduction.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup environment and record provenance.")
    add_path_args(parser)
    parser.add_argument("--seed", type=int, default=SEED, help="Global random seed.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail immediately if CUDA GPU is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    seed_everything(args.seed)
    prov = record_provenance(
        output_path=paths.provenance_path,
        seed=args.seed,
        require_cuda=args.require_cuda,
    )
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
