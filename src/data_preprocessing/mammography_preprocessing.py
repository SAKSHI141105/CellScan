"""Mammography preprocessing — the thing that makes "drag any raw mammogram
in" actually work regardless of what format it showed up in. Reuses the
CLAHE/denoise/normalize primitives from image_preprocessing.py (the
histopathology pipeline) since those steps are format-agnostic; what's new
here is getting from "arbitrary uploaded bytes" to "a clean 8-bit grayscale
array" in the first place — DICOM's VOI LUT + MONOCHROME1 inversion and
16-bit TIFF both need handling that plain PNG/JPG never does.
"""
from __future__ import annotations

import io

import cv2
import numpy as np

from src.data_preprocessing.image_preprocessing import apply_clahe, denoise, normalize
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import pydicom

    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    pydicom = None

DICOM_MAGIC_OFFSET = 128
DICOM_MAGIC = b"DICM"


def _looks_like_dicom(raw_bytes: bytes, filename: str) -> bool:
    if filename.lower().endswith((".dcm", ".dicom")):
        return True
    # DICOM files carry a 128-byte preamble then the literal bytes "DICM" —
    # checking this instead of trusting the extension catches mammogram
    # exports that get renamed/re-extensioned along the way
    return len(raw_bytes) > 132 and raw_bytes[DICOM_MAGIC_OFFSET:DICOM_MAGIC_OFFSET + 4] == DICOM_MAGIC


def _load_dicom_pixels(raw_bytes: bytes) -> np.ndarray:
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM mammograms — pip install pydicom")

    dcm = pydicom.dcmread(io.BytesIO(raw_bytes))
    pixels = apply_voi_lut(dcm.pixel_array, dcm).astype(np.float32)

    if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    pixels -= pixels.min()
    max_val = pixels.max()
    if max_val > 0:
        pixels /= max_val
    return (pixels * 255).astype(np.uint8)


def _load_raster_pixels(raw_bytes: bytes) -> np.ndarray:
    """Handles RGB PNG/JPG, 1-channel grayscale, and 16-bit TIF uniformly.
    IMREAD_UNCHANGED preserves bit depth/channel count so 16-bit TIFs don't
    get silently clipped to 8-bit before we've had a chance to rescale them.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode image — not a readable PNG/JPG/TIF")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)

    if img.dtype != np.uint8:
        # covers 16-bit TIFF and anything else outside 8-bit range —
        # rescale to the image's own min/max rather than assuming a fixed
        # bit depth, since 12-bit-in-16-bit-container mammography scans are
        # common and a blind /65535 would leave them looking almost black
        img = img.astype(np.float32)
        img -= img.min()
        max_val = img.max()
        img = (img / max_val * 255).astype(np.uint8) if max_val > 0 else img.astype(np.uint8)

    return img


def load_pixels_auto(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Entry point: raw uploaded bytes -> 8-bit single-channel array,
    regardless of whether they came in as DICOM, RGB, or 16-bit TIFF.
    """
    if _looks_like_dicom(raw_bytes, filename):
        return _load_dicom_pixels(raw_bytes)
    return _load_raster_pixels(raw_bytes)


def preprocess_mammogram(
    gray_uint8: np.ndarray,
    img_size: int = 512,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple[int, int] = (8, 8),
    denoise_method: str = "median",
    denoise_kernel: int = 3,
) -> np.ndarray:
    """gray_uint8 -> resized, CLAHE-enhanced, denoised, [0,1]-normalized
    float32 array. Same chain as the histopathology pipeline
    (image_preprocessing.preprocess_single), just parameterized for
    mammography's larger default resolution — a 224x224 crop throws away
    exactly the fine calcification detail this whole project cares about,
    so 512 is the default here, not 224.
    """
    resized = cv2.resize(gray_uint8, (img_size, img_size), interpolation=cv2.INTER_AREA)
    enhanced = apply_clahe(resized, clahe_clip_limit, clahe_tile_grid)
    cleaned = denoise(enhanced, denoise_method, denoise_kernel)
    return normalize(cleaned)


def preprocess_upload(raw_bytes: bytes, filename: str, cfg: dict) -> np.ndarray:
    gray = load_pixels_auto(raw_bytes, filename)
    return preprocess_mammogram(
        gray,
        img_size=cfg["img_size"],
        clahe_clip_limit=cfg["clahe_clip_limit"],
        clahe_tile_grid=tuple(cfg["clahe_tile_grid"]),
        denoise_method=cfg["denoise_method"],
        denoise_kernel=cfg["denoise_kernel"],
    )
