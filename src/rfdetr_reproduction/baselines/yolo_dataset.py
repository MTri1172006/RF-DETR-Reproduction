"""Convert COCO-formatted VOC2007 splits into YOLO format for fair baseline comparison (Section 8)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

from configs.paper_config import VOC_CLASSES


def export_coco_to_yolo(
    coco_root: Path,
    yolo_root: Path,
    classes: Sequence[str] = VOC_CLASSES,
) -> Path:
    """Export `train`, `valid`, and `test` COCO splits to YOLO images/labels + YAML."""
    coco_root = Path(coco_root)
    yolo_root = Path(yolo_root)

    for split in ["train", "valid", "test"]:
        src_dir = coco_root / split
        img_out = yolo_root / "images" / split
        lbl_out = yolo_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        with open(src_dir / "_annotations.coco.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        anns_by_img: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for im in data["images"]:
            dst_img = img_out / im["file_name"]
            if not dst_img.exists():
                shutil.copy2(src_dir / im["file_name"], dst_img)
            w_im, h_im = float(im["width"]), float(im["height"])
            lbl_path = lbl_out / (Path(im["file_name"]).stem + ".txt")
            lines = []
            for ann in anns_by_img.get(im["id"], []):
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2.0) / w_im
                cy = (y + bh / 2.0) / h_im
                nw = bw / w_im
                nh = bh / h_im
                cls_id = int(ann["category_id"]) - 1
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            lbl_path.write_text("\n".join(lines), encoding="utf-8")

    yaml_text = (
        f"path: {yolo_root}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "names:\n"
        + "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
        + "\n"
    )
    yaml_path = yolo_root / "voc2007_fair.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path
