"""Convert PASCAL VOC 2007 XML annotations to COCO JSON format (Section 2)."""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from configs.nas_config import SEED
from configs.paper_config import VOC_CAT_ID, VOC_CLASSES
from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


def split_trainval_indices(
    num_samples: int,
    seed: int = SEED,
    valid_ratio: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Deterministic 90/10 train/valid split matching the notebook exactly."""
    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, round(valid_ratio * len(indices)))
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]
    return train_indices, valid_indices


def parse_voc_xml_objects(
    xml_path: Path,
    cat_id_map: dict[str, int] = VOC_CAT_ID,
) -> tuple[int, int, list[dict]]:
    """Parse a VOC XML annotation file, filtering `difficult == 1` and invalid boxes."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects: list[dict] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name", "") or "").lower().strip()
        difficult = int(obj.findtext("difficult", "0"))
        if name not in cat_id_map or difficult == 1:
            continue
        bndbox = obj.find("bndbox")
        x1 = max(0.0, float(bndbox.findtext("xmin")) - 1.0)
        y1 = max(0.0, float(bndbox.findtext("ymin")) - 1.0)
        x2 = min(float(width), float(bndbox.findtext("xmax")))
        y2 = min(float(height), float(bndbox.findtext("ymax")))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0.0 or bh <= 0.0:
            continue
        objects.append(
            {
                "category_id": cat_id_map[name],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            }
        )
    return width, height, objects


def export_split_from_pairs(
    image_xml_pairs: Sequence[tuple[Path, Path]],
    dataset_root: Path,
    split_name: str,
    classes: Sequence[str] = VOC_CLASSES,
    cat_id_map: dict[str, int] = VOC_CAT_ID,
    overwrite: bool = False,
) -> Path:
    """Export a list of (image_path, xml_path) pairs into COCO JSON under `dataset_root/split_name`."""
    split_dir = Path(dataset_root) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    ann_file = split_dir / "_annotations.coco.json"

    if ann_file.exists() and not overwrite:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "[Cached] %s: images=%d, annotations=%d",
            split_name,
            len(data["images"]),
            len(data["annotations"]),
        )
        return ann_file

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cat_id_map[name], "name": name, "supercategory": "object"} for name in classes
        ],
    }

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(enumerate(image_xml_pairs), total=len(image_xml_pairs), desc=f"Export {split_name}")
    except ImportError:
        iterator = enumerate(image_xml_pairs)

    ann_id = 1
    for new_id, (img_path, xml_path) in iterator:
        img_path = Path(img_path)
        xml_path = Path(xml_path)
        dst_name = img_path.name
        dst_path = split_dir / dst_name
        if not dst_path.exists():
            shutil.copy2(img_path, dst_path)

        width, height, objects = parse_voc_xml_objects(xml_path, cat_id_map=cat_id_map)
        coco["images"].append(
            {"id": new_id, "file_name": dst_name, "width": width, "height": height}
        )
        for obj in objects:
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": new_id,
                    **obj,
                }
            )
            ann_id += 1

    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    logger.info(
        "[Exported] %s: images=%d, annotations=%d",
        split_name,
        len(coco["images"]),
        len(coco["annotations"]),
    )
    return ann_file


def prepare_voc2007_coco(
    raw_root: Path,
    dataset_root: Path,
    seed: int = SEED,
    valid_ratio: float = 0.1,
    download: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download PASCAL VOC 2007 and export train / valid / test COCO splits."""
    from torchvision.datasets import VOCDetection

    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    voc_trainval = VOCDetection(root=str(raw_root), year="2007", image_set="trainval", download=download)
    voc_test = VOCDetection(root=str(raw_root), year="2007", image_set="test", download=download)

    train_indices, valid_indices = split_trainval_indices(
        len(voc_trainval), seed=seed, valid_ratio=valid_ratio
    )

    train_pairs = [(Path(voc_trainval.images[i]), Path(voc_trainval.annotations[i])) for i in train_indices]
    valid_pairs = [(Path(voc_trainval.images[i]), Path(voc_trainval.annotations[i])) for i in valid_indices]
    test_pairs = [
        (Path(voc_test.images[i]), Path(voc_test.annotations[i])) for i in range(len(voc_test))
    ]

    return {
        "train": export_split_from_pairs(train_pairs, dataset_root, "train", overwrite=overwrite),
        "valid": export_split_from_pairs(valid_pairs, dataset_root, "valid", overwrite=overwrite),
        "test": export_split_from_pairs(test_pairs, dataset_root, "test", overwrite=overwrite),
    }
