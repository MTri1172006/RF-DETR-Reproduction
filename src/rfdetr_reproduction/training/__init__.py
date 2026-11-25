"""Training loop, optimizer/EMA helpers, atomic checkpointing, and shared val_loss evaluator."""

from rfdetr_reproduction.training.checkpoint import (
    atomic_torch_save,
    load_ema_weights,
    load_hypo1_state,
    load_nas_checkpoint,
    save_hypo1_state,
    save_nas_checkpoints,
)
from rfdetr_reproduction.training.evaluator import evaluate_subnet_val_loss
from rfdetr_reproduction.training.train_utils import (
    SupernetEMA,
    build_paper_optimizer,
    compute_weighted_loss,
    prepare_batch_for_resolution,
    resolve_device_and_amp,
)
from rfdetr_reproduction.training.trainer import run_nas_training, train_one_nas_epoch

__all__ = [
    "SupernetEMA",
    "build_paper_optimizer",
    "compute_weighted_loss",
    "prepare_batch_for_resolution",
    "resolve_device_and_amp",
    "atomic_torch_save",
    "save_nas_checkpoints",
    "load_nas_checkpoint",
    "load_ema_weights",
    "save_hypo1_state",
    "load_hypo1_state",
    "evaluate_subnet_val_loss",
    "train_one_nas_epoch",
    "run_nas_training",
]
