"""Unit tests for VOC->COCO conversion, difficult=1 filtering, and zero-overlap validation."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfdetr_reproduction.data.validation import validate_all_splits
from rfdetr_reproduction.data.voc_to_coco import (
    export_split_from_pairs,
    split_trainval_indices,
)


def _create_synthetic_voc_sample(
    root: Path,
    stem: str,
    width: int,
    height: int,
    objects: list[tuple[str, int, tuple[int, int, int, int]]],
) -> tuple[Path, Path]:
    img_path = root / f"{stem}.jpg"
    xml_path = root / f"{stem}.xml"
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(img_path)

    obj_xml = []
    for name, difficult, (xmin, ymin, xmax, ymax) in objects:
        obj_xml.append(
            f"<object><name>{name}</name><difficult>{difficult}</difficult>"
            f"<bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
            f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>"
        )
    xml_content = (
        f"<annotation><size><width>{width}</width><height>{height}</height></size>"
        + "".join(obj_xml)
        + "</annotation>"
    )
    xml_path.write_text(xml_content, encoding="utf-8")
    return img_path, xml_path


class TestVOCDatasetConversionAndValidation(unittest.TestCase):
    def test_deterministic_split_and_zero_overlap(self):
        train_idx, valid_idx = split_trainval_indices(5011, seed=42, valid_ratio=0.1)
        self.assertEqual(len(valid_idx), 501)
        self.assertEqual(len(train_idx), 4510)
        self.assertTrue(set(train_idx).isdisjoint(set(valid_idx)))

        # Re-running with same seed must match identically
        train_idx2, valid_idx2 = split_trainval_indices(5011, seed=42, valid_ratio=0.1)
        self.assertEqual(train_idx, train_idx2)
        self.assertEqual(valid_idx, valid_idx2)

    def test_difficult_filtering_and_split_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            dataset_root = tmp / "voc2007_rfdetr"

            # Create 4 synthetic samples: 2 train, 1 valid, 1 test
            # Include difficult=1 objects and unknown class to verify filtering
            p1 = _create_synthetic_voc_sample(
                raw_dir,
                "000001",
                200,
                150,
                [
                    ("person", 0, (10, 20, 100, 120)),
                    ("car", 1, (5, 5, 50, 50)),  # difficult=1 -> must be filtered out
                    ("unknown_cls", 0, (10, 10, 40, 40)),  # unknown -> must be filtered out
                ],
            )
            p2 = _create_synthetic_voc_sample(
                raw_dir,
                "000002",
                300,
                200,
                [
                    ("dog", 0, (20, 30, 150, 180)),
                    ("cat", 0, (160, 40, 250, 150)),
                    ("bird", 1, (10, 10, 30, 30)),  # difficult=1 -> filtered out
                ],
            )
            p3 = _create_synthetic_voc_sample(
                raw_dir,
                "000003",
                250,
                250,
                [("bicycle", 0, (15, 25, 180, 200))],
            )
            p4 = _create_synthetic_voc_sample(
                raw_dir,
                "000004",
                320,
                240,
                [
                    ("bus", 0, (1, 1, 320, 240)),
                    ("train", 1, (50, 50, 100, 100)),  # difficult=1 -> filtered out
                ],
            )

            export_split_from_pairs([p1, p2], dataset_root, "train")
            export_split_from_pairs([p3], dataset_root, "valid")
            export_split_from_pairs([p4], dataset_root, "test")

            stats = validate_all_splits(dataset_root)
            self.assertEqual(stats["train"], (2, 3))  # 1 from p1 + 2 from p2
            self.assertEqual(stats["valid"], (1, 1))
            self.assertEqual(stats["test"], (1, 1))

            # Verify bbox coordinate conversion (VOC 1-indexed xmin/ymin -> 0-indexed)
            train_json = json.loads(
                (dataset_root / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
            )
            first_bbox = train_json["annotations"][0]["bbox"]
            self.assertEqual(first_bbox, [9.0, 19.0, 91.0, 101.0])

    def test_overlap_detection_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            dataset_root = tmp / "voc2007_rfdetr"

            shared = _create_synthetic_voc_sample(
                raw_dir, "000099", 100, 100, [("person", 0, (10, 10, 50, 50))]
            )
            export_split_from_pairs([shared], dataset_root, "train")
            export_split_from_pairs([shared], dataset_root, "valid")
            export_split_from_pairs(
                [_create_synthetic_voc_sample(raw_dir, "000100", 100, 100, [("car", 0, (5, 5, 40, 40))])],
                dataset_root,
                "test",
            )

            with self.assertRaises(AssertionError):
                validate_all_splits(dataset_root)


if __name__ == "__main__":
    unittest.main()
