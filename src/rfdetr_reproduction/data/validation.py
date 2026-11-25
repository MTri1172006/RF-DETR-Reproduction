"""Dataset integrity and zero-overlap validation across train/valid/test splits."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


def validate_split(dataset_root: Path, split: str, sample_images_to_check: int = 5) -> tuple[int, int, set[str]]:
    """Validate annotation integrity, positive boxes, and image dimensions for a split."""
    folder = Path(dataset_root) / split
    ann_path = folder / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_ids = {x["id"] for x in data["images"]}
    file_names = {x["file_name"] for x in data["images"]}
    if len(image_ids) != len(data["images"]):
        raise AssertionError(f"Duplicate image_ids in split '{split}'")

    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            raise AssertionError(f"Annotation references unknown image_id {ann['image_id']} in '{split}'")
        _, _, w, h = ann["bbox"]
        if not (w > 0 and h > 0):
            raise AssertionError(f"Non-positive bbox ({w}, {h}) in '{split}'")

    for im in data["images"][:sample_images_to_check]:
        with Image.open(folder / im["file_name"]) as img:
            if img.size != (im["width"], im["height"]):
                raise AssertionError(
                    f"Image size mismatch for {im['file_name']}: file={img.size} vs json={(im['width'], im['height'])}"
                )

    return len(data["images"]), len(data["annotations"]), file_names


def validate_all_splits(
    dataset_root: Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    sample_images_to_check: int = 5,
) -> dict[str, tuple[int, int]]:
    """Validate all splits and assert zero filename overlap between train/valid/test."""
    split_stats: dict[str, tuple[int, int]] = {}
    split_files: dict[str, set[str]] = {}

    for sp in splits:
        n_im, n_ann, fnames = validate_split(
            dataset_root, sp, sample_images_to_check=sample_images_to_check
        )
        split_stats[sp] = (n_im, n_ann)
        split_files[sp] = fnames
        logger.info("Validated %-5s: %d images, %d annotations", sp, n_im, n_ann)

    if "train" in split_files and "valid" in split_files:
        assert split_files["train"].isdisjoint(split_files["valid"]), "Overlap between train and valid!"
    if "train" in split_files and "test" in split_files:
        assert split_files["train"].isdisjoint(split_files["test"]), "Overlap between train and test!"
    if "valid" in split_files and "test" in split_files:
        assert split_files["valid"].isdisjoint(split_files["test"]), "Overlap between valid and test!"

    logger.info("Zero-overlap check PASSED across %s.", " / ".join(splits))
    return split_stats
