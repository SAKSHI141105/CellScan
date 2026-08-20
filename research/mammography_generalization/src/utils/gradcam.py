"""Grad-CAM for the classification head, plus an alignment metric against
ground-truth lesion masks. The actual research question this project asks —
"does lesion supervision make the backbone attend to the lesion, not scanner
artifacts" — isn't answerable from validation AUC alone; two models can hit
the same AUC while one keys off a lesion's actual boundary and the other
keys off a view-marker or compression-paddle artifact that happens to
correlate with the label in one dataset. This is the check for that.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Hooks the last conv layer of the backbone's deepest pyramid stage.
    Works for any of this project's model classes since both expose
    `model.backbone.backbone` (the underlying timm feature extractor) —
    grabbing its last stage's last conv module generically rather than
    hardcoding a layer name per architecture.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """image: (1, C, H, W). Returns an (H, W) heatmap normalized to [0, 1]."""
        self.model.zero_grad()
        outputs = self.model(image)
        logits = outputs["class_logits"] if isinstance(outputs, dict) else outputs
        logits[:, 0].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0].cpu().numpy()

        cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
        cam -= cam.min()
        max_val = cam.max()
        return cam / max_val if max_val > 0 else cam


def find_last_conv(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("no Conv2d layer found in the given module")
    return last


def mask_alignment_iou(heatmap: np.ndarray, ground_truth_mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between the thresholded Grad-CAM heatmap and the ground-truth
    lesion mask — the metric src/evaluation/evaluate_cross_dataset.py uses
    to compare "attention alignment" between baseline and lesion-guided
    checkpoints, not just raw classification accuracy.
    """
    heat_binary = (heatmap >= threshold).astype(np.uint8)
    gt_binary = (ground_truth_mask > 0.5).astype(np.uint8)
    intersection = np.logical_and(heat_binary, gt_binary).sum()
    union = np.logical_or(heat_binary, gt_binary).sum()
    return float(intersection / union) if union > 0 else 0.0


def overlay_heatmap(image_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(np.uint8(255 * image_gray), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heatmap_color, alpha, base_bgr, 1 - alpha, 0)
