"""PyTorch Dataset and DataLoader utilities for COCO-formatted VOC2007."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from rfdetr_reproduction.data.transforms import apply_symmetric_detection_transforms


class SymmetricVOCDetectionDataset(Dataset):
    """Reads COCO-formatted VOC2007 splits with symmetric paper augmentations."""

    def __init__(
        self,
        split_dir: Path,
        is_train: bool = True,
        scale_range: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        self.split_dir = Path(split_dir)
        self.is_train = is_train
        self.scale_range = scale_range
        self.coco = COCO(str(self.split_dir / "_annotations.coco.json"))
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.split_dir / info["file_name"]).convert("RGB")
        orig_w, orig_h = img.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes_xyxy = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 1.0 and h > 1.0:
                boxes_xyxy.append([x, y, x + w, y + h])
                labels.append(int(ann["category_id"]) - 1)

        img_t = TF.to_tensor(img)
        boxes = (
            torch.tensor(boxes_xyxy, dtype=torch.float32)
            if boxes_xyxy
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_t = (
            torch.tensor(labels, dtype=torch.long)
            if labels
            else torch.zeros((0,), dtype=torch.long)
        )

        img_t, boxes_cxcywh, labels_t, height, width = apply_symmetric_detection_transforms(
            img_t,
            boxes,
            labels_t,
            is_train=self.is_train,
            scale_range=self.scale_range,
        )

        target = {
            "boxes": boxes_cxcywh,
            "labels": labels_t,
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([height, width]),
        }
        return img_t, target


def nas_collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def build_dataloader(
    split_dir: Path,
    batch_size: int,
    is_train: bool,
    scale_range: tuple[float, float] = (0.5, 1.5),
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool | None = None,
    generator: torch.Generator | None = None,
) -> tuple[SymmetricVOCDetectionDataset, DataLoader]:
    dataset = SymmetricVOCDetectionDataset(
        split_dir=split_dir,
        is_train=is_train,
        scale_range=scale_range,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=nas_collate_fn,
        drop_last=is_train if drop_last is None else drop_last,
        generator=generator,
    )
    return dataset, loader
