"""Grad-CAM, implemented directly against tf.GradientTape rather than pulling
in tf-explain — it's ~40 lines and gives us full control over which layer we
hook, which matters since the custom CNN and each transfer backbone name
their last conv layer differently.
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

# last conv layer per architecture — needed because we can't just grep for
# "the last Conv2D", ResNet/EfficientNet/VGG all name theirs differently
LAST_CONV_LAYER = {
    "custom_cnn": "last_conv",
    "transfer_resnet50": "conv5_block3_out",
    "transfer_efficientnetb0": "top_conv",
    "transfer_vgg16": "block5_conv3",
}


def compute_gradcam(model: tf.keras.Model, image_batch: np.ndarray, layer_name: str) -> np.ndarray:
    """image_batch: shape (1, H, W, C), already preprocessed the same way as training.
    Returns a (H, W) heatmap normalized to [0, 1].
    """
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """original_gray: (H, W) float32 in [0,1]. Returns a BGR uint8 image with the
    heatmap overlaid, resized to match the original.
    """
    h, w = original_gray.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(np.uint8(255 * original_gray), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heatmap_color, alpha, base_bgr, 1 - alpha, 0)


def explain_prediction(model: tf.keras.Model, model_key: str, preprocessed_img: np.ndarray) -> np.ndarray:
    """preprocessed_img: (H, W) or (H, W, 1) float32 in [0,1]. Returns the overlay image."""
    layer_name = LAST_CONV_LAYER.get(model_key)
    if layer_name is None:
        raise ValueError(f"no known last-conv layer for {model_key}")

    img_for_model = preprocessed_img if preprocessed_img.ndim == 3 else preprocessed_img[..., np.newaxis]
    batch = img_for_model[np.newaxis, ...]

    heatmap = compute_gradcam(model, batch, layer_name)
    gray_for_overlay = preprocessed_img[..., 0] if preprocessed_img.ndim == 3 else preprocessed_img
    return overlay_heatmap(gray_for_overlay, heatmap)
