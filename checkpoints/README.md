# Checkpoints Directory

Model weights and training states managed by `rfdetr_reproduction.training.checkpoint`:

- `pretrained/rf-detr-base.pth` – Official COCO-pretrained RF-DETR Base weights.
- `nas/nas_supernet_latest.pth` – Full resumable checkpoint (`model`, `ema_model`, `ema_updates`, `optimizer`, `scaler`, `epoch`, `history`, `rng_state`, `provenance`) saved atomically (`.pth.tmp` -> `os.replace`).
- `nas/nas_supernet_ema_only.pth` – Lightweight EMA-only checkpoint for Post-Training Grid Search, Pareto analysis, and ONNX/TensorRT export.
- `diagnostics/hypo1_state.pth` – Dedicated resumable training state for Hypothesis 1 convergence experiments.

All `.pth` / `.pt` files are excluded from Git via `.gitignore`.
