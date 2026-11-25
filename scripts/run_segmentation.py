"""Thin CLI entry point: gated RF-DETR-Seg instance segmentation head check (Section 9)."""

from __future__ import annotations

import argparse

from rfdetr_reproduction.segmentation.seg_head import run_segmentation_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LightweightRFDETRSegHead extension.")
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable gated RF-DETR-Seg head smoke test (disabled by default for bbox-only VOC2007).",
    )
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--active-layers", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_segmentation_smoke_test(
        enabled=args.enable,
        active_layers=args.active_layers,
        num_queries=args.num_queries,
        input_resolution=args.resolution,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
