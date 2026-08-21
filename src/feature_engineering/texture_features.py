"""Classical texture/edge features — the non-deep half of the image feature
extraction step. These feed the clustering models and give us an interpretable
fallback when we want to sanity-check what the CNN is picking up on.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]


def glcm_features(img_uint8: np.ndarray) -> dict:
    """img_uint8 must be single-channel, 0-255. Averages each property across all
    distance/angle combos — keeps the feature vector small (6 numbers) instead of
    exploding to 6 * len(distances) * len(angles).
    """
    glcm = graycomatrix(
        img_uint8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=256, symmetric=True, normed=True
    )
    feats = {}
    for prop in GLCM_PROPS:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    return feats


def edge_density_features(img_uint8: np.ndarray) -> dict:
    """Canny edge density + Sobel gradient magnitude stats. Malignant nuclei tend
    to have denser, more irregular boundaries than benign tissue — this is a
    crude proxy for that before the CNN gets involved.
    """
    edges = cv2.Canny(img_uint8, 100, 200)
    edge_density = float(np.mean(edges > 0))

    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)

    return {
        "canny_edge_density": edge_density,
        "sobel_mean": float(grad_mag.mean()),
        "sobel_std": float(grad_mag.std()),
    }


def extract_classical_features(img_float: np.ndarray) -> dict:
    """img_float is the normalized [0,1] output of preprocess_single; we scale it
    back to uint8 here since GLCM/Canny expect discrete intensity levels.
    """
    img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    if img_uint8.ndim == 3:
        img_uint8 = img_uint8[..., 0]
    feats = {}
    feats.update(glcm_features(img_uint8))
    feats.update(edge_density_features(img_uint8))
    return feats


def batch_extract(images: np.ndarray) -> "pd.DataFrame":
    import pandas as pd

    rows = [extract_classical_features(img) for img in images]
    return pd.DataFrame(rows)


def pixel_intensity_histogram(img_float: np.ndarray, n_bins: int = 32) -> dict:
    """Real per-image pixel-intensity distribution (post-CLAHE/denoise) — not
    a fabricated "benign vs. malignant baseline" overlay. We don't have a
    real trained-on dataset in this repo to compute honest class-conditional
    reference distributions from, so this deliberately stays a single-image
    histogram rather than pretending to compare against reference curves.
    """
    img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    counts, edges = np.histogram(img_uint8, bins=n_bins, range=(0, 255))
    bin_centers = ((edges[:-1] + edges[1:]) / 2).round(1)
    return {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "mean_intensity": round(float(img_uint8.mean()), 2),
        "std_intensity": round(float(img_uint8.std()), 2),
    }
