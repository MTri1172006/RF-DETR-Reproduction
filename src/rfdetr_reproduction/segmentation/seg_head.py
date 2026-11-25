"""Lightweight RF-DETR-Seg Instance Segmentation Head (Section 9 & Appendix L)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr_reproduction.utils.logging import get_logger

logger = get_logger(__name__)


class LightweightRFDETRSegHead(nn.Module):
    """Lightweight instance segmentation head per Section 3 & Appendix L of RF-DETR.

    - Bilinearly upsamples encoder feature map to 1/4 of input resolution (`res // 4`).
    - Predicts mask logits per active decoder layer via dot product with projected queries.
    """

    def __init__(self, hidden_dim: int = 256, num_decoder_layers: int = 6) -> None:
        super().__init__()
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.query_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(
        self,
        enc_feat_2d: torch.Tensor,
        decoder_hs_list: list[torch.Tensor],
        input_resolution: int,
    ) -> list[torch.Tensor]:
        target_mask_size = max(16, input_resolution // 4)
        up_feat = F.interpolate(
            enc_feat_2d,
            size=(target_mask_size, target_mask_size),
            mode="bilinear",
            align_corners=False,
        )
        pixel_embed = self.pixel_proj(up_feat)

        pred_masks_per_layer = []
        for idx, hs in enumerate(decoder_hs_list):
            q_embed = self.query_mlps[idx](hs)
            masks = torch.einsum("bqc,bchw->bqhw", q_embed, pixel_embed)
            pred_masks_per_layer.append(masks)
        return pred_masks_per_layer


def run_segmentation_smoke_test(
    enabled: bool = False,
    hidden_dim: int = 256,
    num_decoder_layers: int = 6,
    active_layers: int = 3,
    num_queries: int = 100,
    input_resolution: int = 384,
    device_str: str | None = None,
) -> list[tuple[int, ...]] | None:
    """Run gated shape verification for `LightweightRFDETRSegHead`."""
    if not enabled:
        logger.info(
            "RF-DETR-Seg extension is gated (pass --enable to run). "
            "VOC2007 detection split has bounding boxes only; enable on COCO-Seg or SAM2 masks."
        )
        return None

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    seg_head = LightweightRFDETRSegHead(
        hidden_dim=hidden_dim, num_decoder_layers=num_decoder_layers
    ).to(device)
    with torch.no_grad():
        dummy_enc = torch.randn(1, hidden_dim, 24, 24, device=device)
        dummy_hs = [torch.randn(1, num_queries, hidden_dim, device=device) for _ in range(active_layers)]
        masks_out = seg_head(dummy_enc, dummy_hs, input_resolution=input_resolution)
    shapes = [tuple(m.shape) for m in masks_out]
    logger.info(
        "[RF-DETR-Seg Head OK] Active layers=%d, Mask shape=%s (expected %dx%d)",
        len(shapes),
        shapes[-1],
        input_resolution // 4,
        input_resolution // 4,
    )
    return shapes
