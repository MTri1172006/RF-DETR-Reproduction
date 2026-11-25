"""Collect hardware, software, and repository provenance into outputs/provenance.json."""

from __future__ import annotations

import datetime
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any

from configs.nas_config import SEED
from configs.paper_config import RFDETR_PINNED_COMMIT
from rfdetr_reproduction.utils.backup import save_json
from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)

TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "rfdetr",
    "pycocotools",
    "numpy",
    "pandas",
    "matplotlib",
    "Pillow",
    "onnx",
    "onnxruntime-gpu",
    "ultralytics",
]


def _run_git_cmd(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def configure_cuda_backend(require_cuda: bool = False) -> dict[str, Any]:
    """Enable TF32 and cuDNN benchmark (optimized for NVIDIA L4 / Ada Lovelace)."""
    info: dict[str, Any] = {
        "cuda_available": False,
        "pytorch": None,
        "cuda": None,
        "gpu": None,
        "vram_gb": None,
        "bf16_supported": False,
    }
    try:
        import torch

        info["pytorch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        elif require_cuda:
            raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is False.")
    except ImportError as exc:
        if require_cuda:
            raise RuntimeError("PyTorch is not installed.") from exc
    return info


def collect_provenance(
    seed: int = SEED,
    repo_dir: Path | None = None,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Gather runtime, package, git, and GPU provenance metadata."""
    cuda_info = configure_cuda_backend(require_cuda=require_cuda)
    pkg_versions: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[pkg] = None

    git_sha = _run_git_cmd(["rev-parse", "HEAD"], cwd=repo_dir)
    git_branch = _run_git_cmd(["branch", "--show-current"], cwd=repo_dir)

    return {
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": seed,
        "git_branch": git_branch,
        "git_commit": git_sha,
        "rfdetr_pinned_commit": RFDETR_PINNED_COMMIT,
        "pytorch": cuda_info["pytorch"],
        "cuda": cuda_info["cuda"],
        "cuda_available": cuda_info["cuda_available"],
        "gpu": cuda_info["gpu"],
        "vram_gb": cuda_info["vram_gb"],
        "bf16_supported": cuda_info["bf16_supported"],
        "packages": pkg_versions,
    }


def record_provenance(
    output_path: Path,
    seed: int = SEED,
    repo_dir: Path | None = None,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Collect provenance and persist it to `output_path`."""
    prov = collect_provenance(seed=seed, repo_dir=repo_dir, require_cuda=require_cuda)
    save_json(prov, output_path)
    logger.info("Recorded environment provenance -> %s", output_path)
    return prov
