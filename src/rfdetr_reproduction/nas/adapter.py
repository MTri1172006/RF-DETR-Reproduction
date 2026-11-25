"""Weight-Sharing NAS Adapter on Official RF-DETR (Section 4).

Centralizes:
  - Dynamic symbol discovery in `rfdetr`
  - FlexiViT Pseudo-Inverse (PI) patch embedding resize
  - Official `LWDETR` + `DINOv2-S` + `SetCriterion` instantiation
  - `RFDETRNASAdapter` (dynamic resolution, patch_size, num_windows, decoder_layers, num_queries)
  - Official COCO pretrain checkpoint download & VOC class-head remapping
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.paper_config import (
    COCO_PRETRAIN_FALLBACK_URLS,
    COCO_PRETRAIN_NAME,
    NUM_CLASSES,
    PAPER,
    VOC_CLASSES,
    VOC_TO_COCO_ID,
)
from rfdetr_reproduction.nas.validator import validate_nas_config
from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


def find_rfdetr_symbol(symbol_name: str, preferred_modules: Sequence[str] = ()) -> Any:
    """Locate a symbol in the installed `rfdetr` package without hardcoding internal paths."""
    import rfdetr

    for mod_name in preferred_modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, symbol_name):
                return getattr(mod, symbol_name)
        except Exception:
            pass

    for _, mod_name, _ in pkgutil.walk_packages(rfdetr.__path__, prefix="rfdetr."):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, symbol_name):
                return getattr(mod, symbol_name)
        except Exception:
            continue
    return None


def get_nested_tensor_helpers():
    nested_tensor_cls = find_rfdetr_symbol(
        "NestedTensor", ["rfdetr.util.misc", "rfdetr.utilities.tensors"]
    )
    nested_tensor_from_list = find_rfdetr_symbol(
        "nested_tensor_from_tensor_list",
        ["rfdetr.utilities.tensors", "rfdetr.utilities.misc", "rfdetr.utilities"],
    )
    if nested_tensor_from_list is None:
        raise ImportError("Could not locate `nested_tensor_from_tensor_list` in `rfdetr`.")
    return nested_tensor_cls, nested_tensor_from_list


# ==============================================================================
# FlexiViT Pseudo-Inverse (PI) Patch Embedding Resize
# ==============================================================================
def _build_bilinear_resize_matrix(old_size: int, new_size: int, device, dtype) -> torch.Tensor:
    """Build linear matrix B such that vec(bilinear(x)) = B @ vec(x)."""
    eye = torch.eye(old_size * old_size, device=device, dtype=torch.float32)
    eye_2d = eye.view(old_size * old_size, 1, old_size, old_size)
    resized = F.interpolate(eye_2d, size=(new_size, new_size), mode="bilinear", align_corners=False)
    mat_b = resized.view(old_size * old_size, new_size * new_size).T  # [new^2, old^2]
    return mat_b.to(dtype=dtype)


_PI_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def pi_resize_patch_embed(weight: torch.Tensor, target_patch_size: int) -> torch.Tensor:
    """FlexiViT PI-resize for Conv2d patch_embed: [C_out, C_in, P, P] -> [C_out, C_in, P_new, P_new]."""
    c_out, c_in, old_h, old_w = weight.shape
    assert old_h == old_w, "Patch embedding must be square"
    if old_h == target_patch_size:
        return weight

    key = (old_h, target_patch_size, str(weight.device))
    if key not in _PI_CACHE:
        mat_b = _build_bilinear_resize_matrix(target_patch_size, old_h, weight.device, torch.float32)
        pinv_bt = torch.linalg.pinv(mat_b.T)
        _PI_CACHE[key] = pinv_bt

    pinv_bt = _PI_CACHE[key].to(device=weight.device, dtype=weight.dtype)
    w_flat = weight.reshape(c_out * c_in, old_h * old_w)
    w_new = w_flat @ pinv_bt
    return w_new.reshape(c_out, c_in, target_patch_size, target_patch_size)


# ==============================================================================
# Build Official RF-DETR Supernet
# ==============================================================================
def build_official_rfdetr_supernet(
    dataset_root: Path,
    num_classes: int = NUM_CLASSES,
    device: str | torch.device = "cuda",
    paper_config: dict[str, Any] = PAPER,
):
    """Instantiate official LWDETR + DINOv2-S + SetCriterion from `roboflow/rf-detr`."""
    from rfdetr.config import RFDETRBaseConfig, TrainConfig

    mc = RFDETRBaseConfig(
        encoder="dinov2_windowed_small",
        num_classes=num_classes,
        resolution=560,
        patch_size=14,
        num_windows=4,
        dec_layers=paper_config["train_decoder_layers"],
        num_queries=paper_config["train_queries"],
        pretrain_weights=None,
        out_feature_indexes=[2, 5, 8, 11],
        projector_scale=["P4"],
        group_detr=1,
        two_stage=True,
        lite_refpoint_refine=True,
        bbox_reparam=True,
        layer_norm=True,
        positional_encoding_size=37,
        device=str(device),
    )
    tc = TrainConfig(dataset_dir=str(dataset_root))

    rfdetr_model_module = find_rfdetr_symbol(
        "RFDETRModelModule",
        ["rfdetr.training.module_model", "rfdetr.training"],
    )
    if rfdetr_model_module is not None:
        pl_mod = rfdetr_model_module(mc, tc)
        model = pl_mod.model
        criterion = pl_mod.criterion
        postprocessors = getattr(pl_mod, "postprocess", getattr(pl_mod, "postprocessors", None))
        if not isinstance(postprocessors, dict):
            postprocessors = {"bbox": postprocessors}
        model.to(device)
        criterion.to(device)
        return model, criterion, postprocessors, mc

    build_model = find_rfdetr_symbol("build_model", ["rfdetr.models.lwdetr", "rfdetr.models"])
    build_criterion_and_postprocessors = find_rfdetr_symbol(
        "build_criterion_and_postprocessors",
        ["rfdetr.models.lwdetr", "rfdetr.models"],
    )
    defaults = {
        "vit_encoder_num_layers": 12,
        "pretrained_encoder": None,
        "window_block_indexes": list(paper_config["window_block_indexes"]),
        "hidden_dim": 256,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "drop_path": 0.0,
        "aux_loss": True,
        "ia_bce_loss": True,
        "cls_loss_coef": 1.0,
        "bbox_loss_coef": 5.0,
        "giou_loss_coef": 2.0,
        "focal_alpha": 0.25,
        "set_cost_class": 2.0,
        "set_cost_bbox": 5.0,
        "set_cost_giou": 2.0,
        "num_select": paper_config["train_queries"],
        "segmentation_head": False,
        "rms_norm": False,
        "backbone_lora": False,
        "force_no_pretrain": False,
        "gradient_checkpointing": False,
        "load_dinov2_weights": True,
        "sum_group_losses": False,
        "use_varifocal_loss": False,
        "use_position_supervised_loss": False,
    }
    merged = {**defaults, **tc.model_dump(), **mc.model_dump()}
    merged["window_block_indexes"] = list(paper_config["window_block_indexes"])
    merged["aux_loss"] = True
    args = SimpleNamespace(**merged)

    model = build_model(args)
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    if not isinstance(postprocessors, dict):
        postprocessors = {"bbox": postprocessors}

    model.to(device)
    criterion.to(device)
    return model, criterion, postprocessors, args


# ==============================================================================
# Dynamic Subnet Activation Wrapper (RFDETRNASAdapter)
# ==============================================================================
class _IdentityDecoderLayer(nn.Module):
    def forward(self, tgt, *args, **kwargs):
        return tgt


class RFDETRNASAdapter(nn.Module):
    """True Weight-Sharing NAS wrapper over official LWDETR without weight duplication."""

    def __init__(self, official_model: nn.Module, paper_config: dict[str, Any] = PAPER) -> None:
        super().__init__()
        self.model = official_model
        self.paper_config = paper_config
        self.active_cfg: dict[str, Any] | None = None
        self._nested_tensor_cls = find_rfdetr_symbol(
            "NestedTensor", ["rfdetr.util.misc", "rfdetr.utilities.tensors"]
        )

        self.patch_conv = self._find_patch_embed_conv()
        self.base_patch_size = self.patch_conv.kernel_size[0]
        self.window_holders = self._find_window_holders()
        self.decoder = self.model.transformer.decoder
        self.orig_decoder_layers = self.decoder.layers
        self.max_decoder_layers = len(self.orig_decoder_layers)
        self.orig_num_queries = getattr(self.model, "num_queries", 300)

        self.orig_refpoint_weight = getattr(self.model, "refpoint_embed", None)
        self.orig_query_feat_weight = getattr(self.model, "query_feat", None)

        self._patch_hook_active_size = self.base_patch_size
        self._install_flexivit_patch_forward()

        logger.info(
            "[RFDETRNASAdapter] Ready | base_patch=%d | max_dec_layers=%d | "
            "max_queries=%d | window_holders=%d",
            self.base_patch_size,
            self.max_decoder_layers,
            self.orig_num_queries,
            len(self.window_holders),
        )

    def _find_patch_embed_conv(self) -> nn.Conv2d:
        for _, mod in self.model.backbone.named_modules():
            if (
                isinstance(mod, nn.Conv2d)
                and mod.in_channels == 3
                and mod.kernel_size[0] == mod.kernel_size[1]
                and mod.kernel_size[0] in (8, 10, 12, 14, 16, 20, 24, 32)
            ):
                return mod
        raise RuntimeError("Could not locate Conv2d patch_embed in official RF-DETR backbone.")

    def _find_window_holders(self) -> list[Any]:
        holders: list[Any] = []
        for _, mod in self.model.backbone.named_modules():
            if hasattr(mod, "num_windows"):
                holders.append(mod)
            if hasattr(mod, "config") and hasattr(mod.config, "num_windows"):
                holders.append(mod.config)
        return holders

    def _install_flexivit_patch_forward(self) -> None:
        conv = self.patch_conv
        adapter_ref = self

        def dynamic_conv_forward(x: torch.Tensor) -> torch.Tensor:
            target_ps = adapter_ref._patch_hook_active_size
            w = pi_resize_patch_embed(conv.weight, target_ps)
            return F.conv2d(x, w, conv.bias, stride=(target_ps, target_ps), padding=0)

        conv.forward = dynamic_conv_forward

    def activate_subnet(self, cfg: dict[str, Any], mode: str = "inference") -> None:
        ok, reason = validate_nas_config(cfg, mode=mode, paper_config=self.paper_config)
        if not ok:
            raise ValueError(f"Invalid NAS config {cfg}: {reason}")
        self.active_cfg = dict(cfg)

        ps = int(cfg["patch_size"])
        self._patch_hook_active_size = ps
        for mod in self.model.backbone.modules():
            if hasattr(mod, "patch_size") and isinstance(getattr(mod, "patch_size"), int):
                mod.patch_size = ps
            if hasattr(mod, "config") and hasattr(mod.config, "patch_size"):
                mod.config.patch_size = ps

        nw = int(cfg["num_windows"])
        for holder in self.window_holders:
            holder.num_windows = nw

        dl = int(cfg["decoder_layers"])
        if dl == 0:
            self.decoder.layers = nn.ModuleList([_IdentityDecoderLayer()])
        else:
            self.decoder.layers = self.orig_decoder_layers[:dl]

        nq = int(cfg["num_queries"])
        self.model.num_queries = nq
        if hasattr(self.model.transformer, "two_stage_num_proposals"):
            self.model.transformer.two_stage_num_proposals = nq
        if hasattr(self.model.transformer, "num_queries"):
            self.model.transformer.num_queries = nq

    def restore_supernet(self) -> None:
        self.activate_subnet(
            {
                "resolution": 560,
                "patch_size": self.base_patch_size,
                "num_windows": 4,
                "decoder_layers": self.max_decoder_layers,
                "num_queries": self.orig_num_queries,
            },
            mode="inference",
        )

    def _resize_samples(self, samples, target_res: int):
        nt_cls = self._nested_tensor_cls
        if nt_cls is not None and isinstance(samples, nt_cls):
            tensors, mask = samples.decompose()
            if tensors.shape[-2] != target_res or tensors.shape[-1] != target_res:
                tensors = F.interpolate(
                    tensors, size=(target_res, target_res), mode="bilinear", align_corners=False
                )
                mask = (
                    F.interpolate(
                        mask.unsqueeze(1).float(), size=(target_res, target_res), mode="nearest"
                    )
                    .squeeze(1)
                    .bool()
                )
            return nt_cls(tensors, mask)
        if isinstance(samples, torch.Tensor):
            if samples.shape[-2] != target_res or samples.shape[-1] != target_res:
                samples = F.interpolate(
                    samples, size=(target_res, target_res), mode="bilinear", align_corners=False
                )
            return samples
        return samples

    def forward(self, samples, targets=None):
        assert self.active_cfg is not None, "Call activate_subnet(cfg) before forward()!"
        target_res = int(self.active_cfg["resolution"])
        dl = int(self.active_cfg["decoder_layers"])
        nq = int(self.active_cfg["num_queries"])

        samples = self._resize_samples(samples, target_res)

        saved_ref = None
        saved_qf = None
        if nq < self.orig_num_queries:
            if self.orig_refpoint_weight is not None and hasattr(self.orig_refpoint_weight, "weight"):
                saved_ref = self.orig_refpoint_weight.weight
                self.orig_refpoint_weight.weight = nn.Parameter(
                    saved_ref[:nq], requires_grad=saved_ref.requires_grad
                )
            if self.orig_query_feat_weight is not None and hasattr(self.orig_query_feat_weight, "weight"):
                saved_qf = self.orig_query_feat_weight.weight
                self.orig_query_feat_weight.weight = nn.Parameter(
                    saved_qf[:nq], requires_grad=saved_qf.requires_grad
                )

        try:
            outputs = self.model(samples, targets)
        finally:
            if saved_ref is not None:
                self.orig_refpoint_weight.weight = saved_ref
            if saved_qf is not None:
                self.orig_query_feat_weight.weight = saved_qf

        if dl == 0 and isinstance(outputs, dict) and "enc_outputs" in outputs:
            outputs = {
                "pred_logits": outputs["enc_outputs"]["pred_logits"][:, :nq],
                "pred_boxes": outputs["enc_outputs"]["pred_boxes"][:, :nq],
            }
        return outputs


# ==============================================================================
# Official COCO Pretrain Checkpoint Download & Remapping
# ==============================================================================
def ensure_coco_pretrain_checkpoint(
    ckpt_path: Path,
    pretrain_name: str = COCO_PRETRAIN_NAME,
    fallback_urls: Sequence[str] = COCO_PRETRAIN_FALLBACK_URLS,
) -> Path:
    """Download `rf-detr-base.pth` into `checkpoints/pretrained/` if not already cached."""
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists() and ckpt_path.stat().st_size > 1e6:
        return ckpt_path

    urls: list[str] = []
    hosted = find_rfdetr_symbol(
        "HOSTED_MODELS",
        ["rfdetr.main", "rfdetr.assets.model_weights", "rfdetr.util.files"],
    )
    if isinstance(hosted, dict) and pretrain_name in hosted:
        val = hosted[pretrain_name]
        if isinstance(val, str):
            urls.append(val)
        elif isinstance(val, dict) and "url" in val:
            urls.append(val["url"])
        elif hasattr(val, "url"):
            urls.append(val.url)
    urls += [u for u in fallback_urls if u not in urls]

    tmp = ckpt_path.with_suffix(".part")
    for url in urls:
        try:
            logger.info("Downloading %s from %s ...", pretrain_name, url)
            torch.hub.download_url_to_file(url, str(tmp), progress=True)
            tmp.replace(ckpt_path)
            return ckpt_path
        except Exception as exc:
            logger.warning("Download failed (%s): %s", url, exc)
    raise RuntimeError(
        f"Failed to download {pretrain_name}. Download manually and place at {ckpt_path}."
    )


def load_coco_pretrain_into_supernet(
    model: nn.Module,
    ckpt_path: Path,
    classes: Sequence[str] = VOC_CLASSES,
    voc_to_coco_id: dict[str, int] = VOC_TO_COCO_ID,
) -> Counter:
    """Load COCO-pretrained weights into the 6-layer VOC2007 supernet."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    tgt = model.state_dict()
    if not any(k in tgt for k in src) and any(k.startswith("model.") for k in src):
        src = {k[len("model.") :]: v for k, v in src.items() if k.startswith("model.")}

    dec_pat = re.compile(r"^(.*decoder\.layers\.)(\d+)(\..*)$")
    src_dec_ids = {int(m.group(2)) for k in src for m in [dec_pat.match(k)] if m}
    n_src_dec = (max(src_dec_ids) + 1) if src_dec_ids else 0
    cls_rows = torch.tensor([voc_to_coco_id[c] for c in classes], dtype=torch.long)

    new_sd: dict[str, torch.Tensor] = {}
    stats: Counter = Counter()
    for k, v_t in tgt.items():
        v = src.get(k)
        if v is None and n_src_dec > 0:
            m = dec_pat.match(k)
            if m and int(m.group(2)) >= n_src_dec:
                v = src.get(f"{m.group(1)}{int(m.group(2)) % n_src_dec}{m.group(3)}")
                if v is not None and v.shape == v_t.shape:
                    new_sd[k] = v.clone()
                    stats["decoder_layer_duplicated"] += 1
                    continue
        if v is None:
            stats["missing_in_ckpt"] += 1
            continue
        if v.shape == v_t.shape:
            new_sd[k] = v
            stats["exact"] += 1
        elif (
            "class_embed" in k
            and v.shape[1:] == v_t.shape[1:]
            and v.shape[0] > int(cls_rows.max())
        ):
            w = v_t.detach().cpu().clone()
            w[: len(classes)] = v[cls_rows].to(w.dtype)
            new_sd[k] = w
            stats["class_head_remapped"] += 1
        elif (
            ("refpoint_embed" in k or "query_feat" in k)
            and v.shape[1:] == v_t.shape[1:]
            and v.shape[0] > v_t.shape[0]
        ):
            new_sd[k] = v[: v_t.shape[0]]
            stats["query_sliced"] += 1
        else:
            stats["shape_mismatch_skipped"] += 1
            logger.debug("[skip] %s: ckpt%s vs model%s", k, tuple(v.shape), tuple(v_t.shape))

    model.load_state_dict(new_sd, strict=False)
    loaded = len(new_sd)
    logger.info(
        "[COCO pretrain] src_decoder_layers=%d | loaded %d/%d tensors | %s",
        n_src_dec,
        loaded,
        len(tgt),
        dict(stats),
    )
    if loaded < 0.8 * len(tgt):
        logger.warning("Loaded < 80%% of parameters — verify checkpoint keys against model!")
    return stats
