"""Paper hyperparameters (RF-DETR Appendix A) and VOC2007 dataset constants."""

PAPER = {
    "lr": 1e-4,
    "effective_batch_size": 128,
    "weight_decay": 1e-4,  # Repo default (Appendix A does not override)
    "ema": True,
    "ema_decay": 0.993,
    "lr_warmup": False,
    "lr_scheduler": None,
    "grad_clip_norm": 0.1,
    "layerwise_lr_decay": 0.8,
    "window_block_indexes": [0, 1, 3, 4, 6, 7, 9, 10],
    "multi_scale_factor_range": (0.5, 1.5),
    "train_resolutions": [320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960],
    "train_patch_sizes": [8, 10, 12, 16, 20, 24, 32],
    "train_decoder_layers": 6,
    "train_windows": [1, 2, 4],
    "train_queries": 300,
    "inference_decoder_layers": [0, 1, 2, 3, 4, 5, 6],
    "inference_queries": [50, 100, 200, 300],
    "inference_patch_sizes_as_listed": [8, 10, 12, 14, 16, 20, 24, 32],
}

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

NUM_CLASSES = len(VOC_CLASSES)
VOC_CAT_ID = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}

# Mapping VOC class name -> COCO category_id (rf-detr trains COCO with class index = COCO category_id)
VOC_TO_COCO_ID = {
    "aeroplane": 5,
    "bicycle": 2,
    "bird": 16,
    "boat": 9,
    "bottle": 44,
    "bus": 6,
    "car": 3,
    "cat": 17,
    "chair": 62,
    "cow": 21,
    "diningtable": 67,
    "dog": 18,
    "horse": 19,
    "motorbike": 4,
    "person": 1,
    "pottedplant": 64,
    "sheep": 20,
    "sofa": 63,
    "train": 7,
    "tvmonitor": 72,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

COCO_PRETRAIN_NAME = "rf-detr-base.pth"
COCO_PRETRAIN_FALLBACK_URLS = [
    "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
]
RFDETR_PINNED_COMMIT = "382607f3d7ecbf513b991a8ee9f1a3abe3a340f0"
