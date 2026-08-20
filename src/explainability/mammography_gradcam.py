"""PyTorch Grad-CAM for the mammography model — ported from
research/mammography_generalization/src/utils/gradcam.py (see
src/models/mammography/backbone.py for why this is a mirror, not an
import). Hooks the last conv layer of the backbone's deepest pyramid
stage so the heatmap highlights whatever the classifier actually keyed off
of, mapped back onto the preprocessed grayscale mammogram.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
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


def overlay_heatmap(image_gray_float: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """image_gray_float: (H, W) in [0, 1]. Returns a BGR uint8 overlay,
    same convention as the histopathology Grad-CAM (explainability/gradcam.py)
    so the frontend can render both the same way.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(np.uint8(255 * np.clip(image_gray_float, 0, 1)), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heatmap_color, alpha, base_bgr, 1 - alpha, 0)


def explain_prediction(model: nn.Module, preprocessed_img: np.ndarray) -> np.ndarray:
    """preprocessed_img: (H, W) float32 in [0, 1]. Returns a BGR overlay."""
    target_layer = find_last_conv(model.backbone.backbone)
    cam = GradCAM(model, target_layer)

    tensor = torch.from_numpy(preprocessed_img).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    heatmap = cam(tensor)
    return overlay_heatmap(preprocessed_img, heatmap)
