import numpy as np

from src.data_preprocessing.image_preprocessing import apply_clahe, denoise, normalize, resize, to_grayscale
from src.feature_engineering.texture_features import extract_classical_features


def _fake_bgr_image(size=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_to_grayscale_collapses_channels():
    img = _fake_bgr_image()
    gray = to_grayscale(img)
    assert gray.ndim == 2
    assert gray.shape == (64, 64)


def test_to_grayscale_is_idempotent_on_already_gray_input():
    gray = _fake_bgr_image()[..., 0]
    assert to_grayscale(gray).shape == gray.shape


def test_resize_produces_target_dimensions():
    gray = _fake_bgr_image()[..., 0]
    resized = resize(gray, 224)
    assert resized.shape == (224, 224)


def test_clahe_output_stays_in_valid_range():
    gray = _fake_bgr_image()[..., 0]
    enhanced = apply_clahe(gray)
    assert enhanced.dtype == np.uint8
    assert enhanced.min() >= 0 and enhanced.max() <= 255


def test_denoise_gaussian_and_median_both_run():
    gray = _fake_bgr_image()[..., 0]
    assert denoise(gray, "gaussian", 3).shape == gray.shape
    assert denoise(gray, "median", 3).shape == gray.shape


def test_normalize_scales_to_unit_interval():
    gray = _fake_bgr_image()[..., 0]
    normed = normalize(gray)
    assert normed.dtype == np.float32
    assert normed.min() >= 0.0 and normed.max() <= 1.0


def test_classical_features_are_finite():
    gray_float = normalize(_fake_bgr_image()[..., 0])
    feats = extract_classical_features(gray_float)
    assert all(np.isfinite(v) for v in feats.values())
    assert "glcm_contrast" in feats
    assert "canny_edge_density" in feats
