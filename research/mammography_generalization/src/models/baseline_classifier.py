"""Control-condition model — same backbone as the lesion-guided variant,
classification head only, no segmentation supervision anywhere in the
graph. This is what "does auxiliary lesion supervision actually help
cross-dataset generalization" gets measured against.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbone import build_backbone


class BaselineClassifier(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained, in_channels)
        deepest_channels = self.backbone.out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(deepest_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features[-1])
        return self.classifier(pooled)  # raw logits — caller applies sigmoid/BCEWithLogits
