"""Deployment export and benchmarking: ONNX, TensorRT FP16, CUDA Graphs, and Power Throttling."""

from rfdetr_reproduction.deployment.cuda_graphs import benchmark_cuda_graphs
from rfdetr_reproduction.deployment.onnx_export import (
    RFDETROnnxWrapper,
    export_subnet_to_onnx,
    select_best_pareto_subnet,
)
from rfdetr_reproduction.deployment.tensorrt import build_and_benchmark_tensorrt_fp16
from rfdetr_reproduction.deployment.throttling import (
    compare_buffering_throttling,
    plot_throttling_comparison,
    run_deployment_pipeline,
)

__all__ = [
    "RFDETROnnxWrapper",
    "select_best_pareto_subnet",
    "export_subnet_to_onnx",
    "build_and_benchmark_tensorrt_fp16",
    "benchmark_cuda_graphs",
    "compare_buffering_throttling",
    "plot_throttling_comparison",
    "run_deployment_pipeline",
]
