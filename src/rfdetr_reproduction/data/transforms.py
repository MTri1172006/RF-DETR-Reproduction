"""Symmetric detection augmentations following RF-DETR paper protocol.

- Multi-scale factor range (0.5, 1.5)
- HorizontalFlip (p=0.5)
- RandomCrop (p=0.5)
- ImageNet normalization
- NO VerticalFlip or MixUp
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF

from configs.paper_config import IMAGENET_MEAN, IMAGENET_STD


def apply_symmetric_detection_transforms(
    img_t: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    is_train: bool = True,
    scale_range: tuple[float, float] = (0.5, 1.5),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Apply paper-exact train/eval transforms and return normalized tensor + cxcywh boxes."""
    _, height, width = img_t.shape

    if is_train:
        scale = random.uniform(*scale_range)
        new_h = max(128, int(height * scale))
        new_w = max(128, int(width * scale))
        img_t = TF.resize(img_t, [new_h, new_w], antialias=True)
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= new_w / width
            boxes[:, [1, 3]] *= new_h / height
        height, width = new_h, new_w

        if random.random() < 0.5:
            img_t = TF.hflip(img_t)
            if len(boxes) > 0:
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2

        if random.random() < 0.5 and height > 160 and width > 160:
            ch = random.randint(int(0.7 * height), height)
            cw = random.randint(int(0.7 * width), width)
            top = random.randint(0, height - ch)
            left = random.randint(0, width - cw)
            img_t = TF.crop(img_t, top, left, ch, cw)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clamp(0, cw)
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clamp(0, ch)
                keep = (boxes[:, 2] - boxes[:, 0] > 2.0) & (boxes[:, 3] - boxes[:, 1] > 2.0)
                boxes = boxes[keep]
                labels = labels[keep]
            height, width = ch, cw

    img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if len(boxes) > 0:
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / width
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / height
        bw = (boxes[:, 2] - boxes[:, 0]) / width
        bh = (boxes[:, 3] - boxes[:, 1]) / height
        boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=-1).clamp(0.0, 1.0)
    else:
        boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)

    return img_t, boxes_cxcywh, labels, height, width
