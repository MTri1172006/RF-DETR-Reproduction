"""Experiment configurations for Gradient Conflict, Hypo 1, Hypo 2, Grid Search, and Deployment."""

from configs.nas_config import MICRO_BATCH, NAS_EPOCHS, SEED

GRADIENT_CONFLICT_CONFIG = {
    "num_batches": 4,
    "batch_size": MICRO_BATCH,
    "seed": 0,
    # (resolution, patch_size, num_windows): anchor + single-axis variations
    "candidates": [
        (512, 16, 2),  # anchor
        (512, 8, 2),
        (512, 32, 2),  # patch size -> FlexiViT PI-resize stress
        (512, 16, 1),
        (512, 16, 4),  # num_windows -> attention windowing stress
        (320, 16, 2),
        (896, 16, 2),  # resolution -> positional embedding stress
    ],
}

HYPO1_CONFIG = {
    "epochs": NAS_EPOCHS,
    "eval_every": 2,
    "seed": SEED,
    # (probe_name, (resolution, patch_size, num_windows))
    "probes": [
        ("small_320_p16_w2", (320, 16, 2)),
        ("nano_384_p16_w2", (384, 16, 2)),
        ("mid_512_p16_w2", (512, 16, 2)),
        ("large_640_p16_w2", (640, 16, 2)),
        ("fine_512_p8_w2", (512, 8, 2)),
        ("coarse_512_p32_w2", (512, 32, 2)),
    ],
}

HYPO2_CONFIG = {
    "extra_epochs": 5,
    "probs": [0.0, 0.30, 0.70, 1.00],  # 0.0 = uniform baseline; 1.0 = train big A only
    "seed": SEED,
    "expected_base_epoch": NAS_EPOCHS,
    "target_big_a": (640, 16, 2),
    "watch": [
        ("big_A_640_p16_w2", (640, 16, 2)),
        ("mid_512_p16_w2", (512, 16, 2)),
        ("small_384_p16_w2", (384, 16, 2)),
        ("tiny_320_p16_w2", (320, 16, 2)),
    ],
}

GRID_SEARCH_CONFIG = {
    "mode": "representative",  # "representative" | "full"
    "latency_warmup": 10,
    "latency_repeats": 30,
    "latency_buffer_sec": 0.2,
    "score_thresh": 0.01,
    "max_images": None,
}

DEPLOYMENT_CONFIG = {
    "opset_version": 17,
    "fallback_subnet_cfg": {
        "resolution": 384,
        "patch_size": 16,
        "num_windows": 2,
        "decoder_layers": 2,
        "num_queries": 300,
    },
    "trt_warmup_ms": 2000,
    "trt_iterations": 200,
    "trt_avg_runs": 200,
    "throttling_num_runs": 60,
    "throttling_warmup": 15,
    "cuda_graph_warmup": 15,
    "cuda_graph_repeats": 40,
}
