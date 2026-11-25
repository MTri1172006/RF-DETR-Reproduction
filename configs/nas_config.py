"""Weight-Sharing NAS training and search-space configuration."""

from configs.paper_config import PAPER

SEED = 42
MICRO_BATCH = 4
# Paper uses effective batch 128 on COCO/O365. On VOC2007 (~4.5k train images),
# 16 matches the official rf-detr fine-tune default (4 x 4) and YOLO baseline (~282 steps/epoch).
EFFECTIVE_BATCH = 16
ACCUM_STEPS = EFFECTIVE_BATCH // MICRO_BATCH
assert MICRO_BATCH * ACCUM_STEPS == EFFECTIVE_BATCH

NAS_EPOCHS = 20
USE_COCO_PRETRAIN = True
NUM_WORKERS = 2

# Token budget on NVIDIA L4 24GB with AMP bfloat16 to avoid OOM at 960/p8 (14,400 tokens)
TRAIN_MAX_TOKENS_L4 = 5776
MIN_GRID_SIZE = 8

NAS_CONFIG = {
    "seed": SEED,
    "micro_batch": MICRO_BATCH,
    "effective_batch": EFFECTIVE_BATCH,
    "accum_steps": ACCUM_STEPS,
    "nas_epochs": NAS_EPOCHS,
    "use_coco_pretrain": USE_COCO_PRETRAIN,
    "num_workers": NUM_WORKERS,
    "train_max_tokens_l4": TRAIN_MAX_TOKENS_L4,
    "min_grid_size": MIN_GRID_SIZE,
    "valid_split_ratio": 0.1,
}


def build_nas_train_grid(paper: dict = PAPER) -> dict:
    return {
        "resolution": list(paper["train_resolutions"]),
        "patch_size": list(paper["train_patch_sizes"]),
        "decoder_layers": [paper["train_decoder_layers"]],
        "num_windows": list(paper["train_windows"]),
        "num_queries": [paper["train_queries"]],
    }


def build_nas_infer_grid_8ps(paper: dict = PAPER) -> dict:
    return {
        "resolution": list(paper["train_resolutions"]),
        "patch_size": list(paper["inference_patch_sizes_as_listed"]),
        "decoder_layers": list(paper["inference_decoder_layers"]),
        "num_windows": list(paper["train_windows"]),
        "num_queries": list(paper["inference_queries"]),
    }


def build_nas_infer_grid_7ps(paper: dict = PAPER) -> dict:
    grid = build_nas_infer_grid_8ps(paper)
    grid["patch_size"] = list(paper["train_patch_sizes"])
    return grid


REPRESENTATIVE_INFER_GRID = {
    "resolution": [320, 384, 448, 512, 640],
    "patch_size": [14, 16],
    "num_windows": [2, 4],
    "decoder_layers": [0, 1, 2, 4, 6],
    "num_queries": [50, 100, 300],
}
