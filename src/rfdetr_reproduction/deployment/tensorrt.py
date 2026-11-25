"""TensorRT FP16 + CUDA Graphs engine build and benchmark via `trtexec` (Section 7)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


def build_and_benchmark_tensorrt_fp16(
    onnx_path: Path,
    trt_engine_path: Path,
    warmup_ms: int = 2000,
    iterations: int = 200,
    avg_runs: int = 200,
    use_cuda_graph: bool = True,
) -> dict[str, Any]:
    """Invoke `trtexec` to build an FP16 TensorRT engine and benchmark latency."""
    trtexec_bin = shutil.which("trtexec")
    onnx_path = Path(onnx_path)
    trt_engine_path = Path(trt_engine_path)
    trt_engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not trtexec_bin or not onnx_path.exists():
        msg = (
            "`trtexec` not found in PATH or ONNX file missing. "
            f"Manual command: trtexec --onnx={onnx_path} --saveEngine={trt_engine_path} "
            f"--fp16 --useCudaGraph --iterations={iterations}"
        )
        logger.warning("[TensorRT] %s", msg)
        return {"available": False, "returncode": None, "message": msg}

    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_engine_path}",
        "--fp16",
        f"--warmUp={warmup_ms}",
        f"--iterations={iterations}",
        f"--avgRuns={avg_runs}",
    ]
    if use_cuda_graph:
        cmd.append("--useCudaGraph")

    logger.info("Running TensorRT FP16 benchmark: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("trtexec exited with code %d: %s", proc.returncode, proc.stderr[-1000:])
    return {
        "available": True,
        "returncode": proc.returncode,
        "engine_path": str(trt_engine_path),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-1000:],
    }
