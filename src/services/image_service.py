"""Image prediction + Grad-CAM, framework-agnostic (see tabular_service.py
for why — this is the twin of that module for the image pipeline).
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CFG = load_config()
IMAGE_MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_dir"] / "image"

_cache: dict = {}


def load_image_model(prefer: str = "transfer"):
    """Returns (model, model_key) or (None, None) if nothing's been trained yet."""
    if "model" in _cache:
        return _cache["model"]

    import tensorflow as tf

    backbone = CFG["image"]["transfer_backbone"]
    candidates = [
        (IMAGE_MODELS_DIR / f"transfer_{backbone}_final.h5", f"transfer_{backbone}"),
        (IMAGE_MODELS_DIR / "custom_cnn_final.h5", "custom_cnn"),
    ]
    if prefer == "custom":
        candidates.reverse()

    for path, key in candidates:
        if path.exists():
            result = (tf.keras.models.load_model(path), key)
            _cache["model"] = result
            logger.info("Loaded image model: %s", key)
            return result

    _cache["model"] = (None, None)
    return None, None


def gray_float_to_png_base64(img_float: np.ndarray) -> str:
    """img_float: (H, W) or (H, W, 1) in [0, 1] -> base64-encoded PNG string."""
    img = img_float[..., 0] if img_float.ndim == 3 else img_float
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img_uint8)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def bgr_to_png_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def predict(preprocessed_img: np.ndarray) -> dict | None:
    from src.explainability.gradcam import explain_prediction

    model, model_key = load_image_model()
    if model is None:
        return None

    batch = preprocessed_img[np.newaxis, ..., np.newaxis] if preprocessed_img.ndim == 2 else preprocessed_img[np.newaxis, ...]
    proba = float(model.predict(batch, verbose=0)[0, 0])
    predicted_class = "Malignant" if proba >= 0.5 else "Benign"
    overlay = explain_prediction(model, model_key, preprocessed_img)

    return {
        "predicted_class": predicted_class,
        "probability_malignant": proba,
        "preprocessed_png_base64": gray_float_to_png_base64(preprocessed_img),
        "gradcam_png_base64": bgr_to_png_base64(overlay),
        "model_key": model_key,
    }
