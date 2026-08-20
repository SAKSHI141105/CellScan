"""Ported verbatim from
research/mammography_generalization/src/models/lesion_guided_model.py —
see backbone.py's module docstring for why this is a mirror, not an import.

Shared backbone, two heads: classification (malignant/benign) and a U-Net
decoder producing a lesion segmentation mask. The mask output isn't
consumed by CellScan's API at all (no ground-truth mask to compare against
for an arbitrary uploaded image) — it's kept in the forward pass anyway
because it's the thing that makes the *classification* head's features
lesion-aware in the first place; dropping it here would mean loading a
checkpoint trained with it into a graph that doesn't have it, which breaks
state_dict compatibility with research-pipeline checkpoints.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mammography.backbone import build_backbone


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LesionGuidedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 1,
        num_classes: int = 1,
        decoder_channels: list[int] = (256, 128, 64, 32),
    ):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained, in_channels)
        pyramid_channels = self.backbone.out_channels

        n_decoder_stages = len(decoder_channels)
        if n_decoder_stages >= len(pyramid_channels):
            raise ValueError(
                f"decoder_channels has {n_decoder_stages} stages but the backbone only "
                f"produces a {len(pyramid_channels)}-level pyramid."
            )

        used_levels = pyramid_channels[-(n_decoder_stages + 1):]
        deepest = used_levels[-1]
        skips = list(reversed(used_levels[:-1]))

        self.decoder_blocks = nn.ModuleList()
        in_ch = deepest
        for skip_ch, out_ch in zip(skips, decoder_channels):
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.mask_head = nn.Conv2d(in_ch, 1, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(deepest, num_classes),
        )
        self._n_decoder_stages = n_decoder_stages

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        pyramid = self.backbone(x)
        used = pyramid[-(self._n_decoder_stages + 1):]
        deepest = used[-1]
        skips = list(reversed(used[:-1]))

        class_logits = self.classifier(self.pool(deepest))

        d = deepest
        for block, skip in zip(self.decoder_blocks, skips):
            d = block(d, skip)
        mask_logits = self.mask_head(d)
        mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)

        return {"class_logits": class_logits, "mask_logits": mask_logits}
