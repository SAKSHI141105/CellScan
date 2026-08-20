"""Ported verbatim from research/mammography_generalization/src/models/backbone.py.

Deliberately duplicated rather than imported: both this package and the
research pipeline define a top-level `src` package, so importing across
directories would collide inside one Python process. Keeping the class
definitions structurally identical (same layers, same names) means a
checkpoint trained via the research pipeline's train.py loads here
unmodified — this is an inference-serving mirror of that architecture, not
a divergent one. If you change the architecture here, change it there too,
or checkpoints stop being interchangeable.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ImagenetBackbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, in_chans=3 if in_channels == 1 else in_channels
        )
        self.feature_info = self.backbone.feature_info
        self.out_channels = [f["num_chs"] for f in self.feature_info.info]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.in_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def build_backbone(name: str, pretrained: bool, in_channels: int) -> ImagenetBackbone:
    return ImagenetBackbone(name=name, pretrained=pretrained, in_channels=in_channels)
