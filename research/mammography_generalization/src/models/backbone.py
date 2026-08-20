"""Backbone factory — one function, swappable architecture via `timm`.

`features_only=True` is the load-bearing choice here: it gets us the
multi-scale feature pyramid (stride 4/8/16/32 stages) that the lesion-guided
decoder needs for skip connections, from any CNN timm knows about, without
hand-writing a forward-hook per architecture.

TODO: this cleanly swaps resnet50 <-> efficientnet_b0 today (both are true
multi-stage CNNs). Swapping in a ViT backbone later will need a different
decoder path — plain ViTs don't have a spatial feature pyramid, you'd want
something like `vit_small_patch16_224.augreg_in21k` with an FPN adapter, or
a hierarchical ViT (Swin) which timm *does* expose via features_only. Not
implemented — flagging so it's not a surprise when someone tries it.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ImagenetBackbone(nn.Module):
    """Wraps a timm feature-pyramid backbone, replicating single-channel
    mammography input to 3 channels so ImageNet pretrained weights apply —
    same trick as CellScan's image_cnn.py transfer-learning path, same
    reasoning: retraining a 1-channel stem from scratch throws away the
    pretrained low-level edge/texture filters for no benefit at this dataset
    scale.
    """

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
