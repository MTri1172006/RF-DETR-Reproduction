"""Thin CLI entry point: download VOC2007, convert to COCO JSON, and validate zero-overlap splits."""

from __future__ import annotations

import argparse

from configs.nas_config import NAS_CONFIG
from rfdetr_reproduction.data.validation import validate_all_splits
from rfdetr_reproduction.data.voc_to_coco import prepare_voc2007_coco
from rfdetr_reproduction.utils.paths import ProjectPaths, add_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate VOC2007 COCO dataset.")
    add_path_args(parser)
    parser.add_argument("--seed", type=int, default=NAS_CONFIG["seed"])
    parser.add_argument("--valid-ratio", type=float, default=NAS_CONFIG["valid_split_ratio"])
    parser.add_argument("--no-download", action="store_true", help="Skip torchvision download.")
    parser.add_argument("--overwrite", action="store_true", help="Re-export cached COCO JSON files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_env_or_args(args.data_dir, args.ckpt_dir, args.output_dir).ensure_dirs()
    prepare_voc2007_coco(
        raw_root=paths.voc_raw_dir,
        dataset_root=paths.dataset_root,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        download=not args.no_download,
        overwrite=args.overwrite,
    )
    validate_all_splits(paths.dataset_root)


if __name__ == "__main__":
    main()
