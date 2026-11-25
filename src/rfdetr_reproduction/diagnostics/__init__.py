"""Diagnostic experiments: Gradient Conflict (5b), Hypothesis 1 (5d), and Hypothesis 2 (5e)."""

from rfdetr_reproduction.diagnostics.gradient_conflict import (
    run_gradient_conflict_diagnosis,
    run_gradient_conflict_experiment,
)
from rfdetr_reproduction.diagnostics.hypothesis1 import (
    compute_hypo1_t90_table,
    run_hypothesis1_experiment,
)
from rfdetr_reproduction.diagnostics.hypothesis2 import run_hypothesis2_experiment

__all__ = [
    "run_gradient_conflict_diagnosis",
    "run_gradient_conflict_experiment",
    "compute_hypo1_t90_table",
    "run_hypothesis1_experiment",
    "run_hypothesis2_experiment",
]
