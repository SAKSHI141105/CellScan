"""Loss functions. The one thing worth reading closely here is how the mask
loss handles samples with no lesion annotation — get this wrong and you
either silently train on garbage zero-masks (teaching the model "no lesion
mask" is a valid target for annotated-but-boundary-ambiguous cases) or crash
on an empty batch when nothing in it happens to have a mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice  # per-sample, caller reduces


class LesionGuidanceLoss(nn.Module):
    """Dice + BCE on the mask, computed only over samples where has_mask=True.
    Returns a scalar tensor that's exactly 0 (with grad) when no sample in
    the batch has a mask, rather than NaN from averaging over zero elements.
    """

    def __init__(self, mode: str = "dice_bce"):
        super().__init__()
        if mode not in {"dice", "bce", "dice_bce"}:
            raise ValueError(f"unknown lesion_loss_type: {mode}")
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, mask_logits: torch.Tensor, mask_targets: torch.Tensor, has_mask: torch.Tensor) -> torch.Tensor:
        if not has_mask.any():
            return mask_logits.sum() * 0.0  # keeps it in the autograd graph, contributes nothing

        logits = mask_logits[has_mask]
        targets = mask_targets[has_mask]

        loss = torch.zeros((), device=mask_logits.device)
        if self.mode in ("dice", "dice_bce"):
            loss = loss + dice_loss(logits, targets).mean()
        if self.mode in ("bce", "dice_bce"):
            loss = loss + self.bce(logits, targets).flatten(1).mean(dim=1).mean()
        return loss


class JointLoss(nn.Module):
    """Total = classification BCE + lambda * lesion guidance loss.
    lambda=0 (see configs/baseline_cbis.yaml) makes this behave identically
    to a plain classification loss without needing a separate code path —
    the decoder head still runs and costs compute, but the config is what
    actually defines "baseline" here, not a different model class.
    """

    def __init__(self, lesion_loss_weight: float = 0.5, lesion_loss_type: str = "dice_bce"):
        super().__init__()
        self.lesion_loss_weight = lesion_loss_weight
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.lesion_loss = LesionGuidanceLoss(mode=lesion_loss_type)

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        class_loss = self.classification_loss(outputs["class_logits"].squeeze(-1), batch["label"])

        if self.lesion_loss_weight > 0 and "mask_logits" in outputs:
            lesion_loss = self.lesion_loss(outputs["mask_logits"], batch["mask"], batch["has_mask"])
        else:
            lesion_loss = torch.zeros((), device=class_loss.device)

        total = class_loss + self.lesion_loss_weight * lesion_loss
        return {"total": total, "classification": class_loss, "lesion_guidance": lesion_loss}
