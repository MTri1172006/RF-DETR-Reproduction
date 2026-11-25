"""Storage-agnostic backup and artifact persistence helpers.

Replaces Google Drive-specific `backup_to_drive` with configurable backends:
  - `local`: store directly under `checkpoints/` & `outputs/` (optional local mirror dir)
  - `hf`: upload artifacts to a Hugging Face Hub repository
  - `wandb`: log artifacts to Weights & Biases
  - `s3`: upload artifacts to an S3 / Cloudflare R2 bucket via boto3
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    try:
        import torch

        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    return str(obj)


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


class BackupManager:
    """Backend-agnostic artifact synchronization manager."""

    def __init__(
        self,
        backend: str = "local",
        destination: str | Path | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.backend = backend.lower()
        self.destination = destination
        self.endpoint_url = endpoint_url
        if self.backend not in {"local", "hf", "wandb", "s3"}:
            raise ValueError(f"Unsupported backup backend: {backend}")

    def sync_files(
        self,
        paths: Iterable[Path],
        subdir: str = "",
        exts: tuple[str, ...] | None = None,
    ) -> list[str]:
        files: list[Path] = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and (exts is None or f.suffix.lower() in exts):
                        files.append(f)

        if not files:
            return []

        if self.backend == "local":
            return self._sync_local(files, subdir)
        if self.backend == "hf":
            return self._sync_hf(files, subdir)
        if self.backend == "wandb":
            return self._sync_wandb(files, subdir)
        if self.backend == "s3":
            return self._sync_s3(files, subdir)
        return []

    def _sync_local(self, files: list[Path], subdir: str) -> list[str]:
        if self.destination is None:
            # Files are already written directly to checkpoints/ and outputs/
            return [str(f) for f in files]
        dst_root = Path(self.destination) / subdir
        dst_root.mkdir(parents=True, exist_ok=True)
        synced = []
        for f in files:
            dst = dst_root / f.name
            shutil.copy2(f, dst)
            synced.append(str(dst))
        logger.info("Backed up %d file(s) locally to %s", len(synced), dst_root)
        return synced

    def _sync_hf(self, files: list[Path], subdir: str) -> list[str]:
        if not self.destination:
            raise ValueError("HF backup backend requires destination='<owner>/<repo>'.")
        from huggingface_hub import HfApi

        api = HfApi()
        synced = []
        for f in files:
            path_in_repo = f"{subdir.strip('/')}/{f.name}".lstrip("/")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=path_in_repo,
                repo_id=str(self.destination),
                repo_type="model",
            )
            synced.append(path_in_repo)
        logger.info("Uploaded %d file(s) to HF Hub repo %s", len(synced), self.destination)
        return synced

    def _sync_wandb(self, files: list[Path], subdir: str) -> list[str]:
        import wandb

        artifact_name = (subdir or "rfdetr-artifacts").replace("/", "_")
        artifact = wandb.Artifact(name=artifact_name, type="experiment_output")
        for f in files:
            artifact.add_file(str(f))
        if wandb.run is None:
            wandb.init(project=str(self.destination or "rfdetr-reproduction"))
        wandb.log_artifact(artifact)
        logger.info("Logged %d file(s) to W&B artifact %s", len(files), artifact_name)
        return [str(f) for f in files]

    def _sync_s3(self, files: list[Path], subdir: str) -> list[str]:
        if not self.destination:
            raise ValueError("S3/R2 backup backend requires destination='<bucket>' or '<bucket>/<prefix>'.")
        import boto3

        dest_str = str(self.destination).removeprefix("s3://")
        parts = dest_str.split("/", 1)
        bucket = parts[0]
        prefix = parts[1].strip("/") if len(parts) > 1 else ""

        client = boto3.client("s3", endpoint_url=self.endpoint_url)
        synced = []
        for f in files:
            key = "/".join(p for p in [prefix, subdir.strip("/"), f.name] if p)
            client.upload_file(str(f), bucket, key)
            synced.append(f"s3://{bucket}/{key}")
        logger.info("Uploaded %d file(s) to S3/R2 bucket %s", len(synced), bucket)
        return synced
