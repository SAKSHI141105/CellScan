"""Mammography dataset loader — the part of this pipeline that actually
differs meaningfully between institutions' scanner output, which is the
whole reason this project exists.

Handles two source formats transparently:
  - raw DICOM (.dcm): VOI LUT applied, MONOCHROME1 inverted to MONOCHROME2
    convention so pixel intensity means the same thing across scanners
  - pre-extracted raster (.png/.jpg): already normalized, loaded as-is

Not every sample has a lesion mask (CBIS-DDSM's benign-without-callback
subset, for instance) — those rows get a zero mask and has_mask=False, and
the loss function (src/training/losses.py) skips the Dice/BCE mask term for
them rather than penalizing the model for "missing" a lesion in an image
that was never annotated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import pydicom

    try:
        from pydicom.pixels import apply_voi_lut  # pydicom >= 3.0
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut  # pydicom < 3.0
except ImportError:  # pydicom is only needed for the .dcm path
    pydicom = None


@dataclass
class MammoSample:
    image: torch.Tensor   # (1, H, W) float32, normalized to [0, 1]
    mask: torch.Tensor    # (1, H, W) float32, {0, 1}
    has_mask: bool
    label: torch.Tensor   # scalar float32, {0, 1}
    patient_id: str


def _load_dicom_pixels(path: Path) -> np.ndarray:
    if pydicom is None:
        raise ImportError("pydicom is required to load .dcm files — pip install pydicom")

    dcm = pydicom.dcmread(str(path))
    pixels = apply_voi_lut(dcm.pixel_array, dcm)

    # MONOCHROME1 means "0 = white, max = black" — the opposite convention
    # from MONOCHROME2 and from every non-DICOM raster format. Some CBIS-DDSM
    # source scans came through this way; leaving it uninverted would train
    # the model on visually negated images for a subset of samples.
    if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    pixels = pixels.astype(np.float32)
    pixels -= pixels.min()
    max_val = pixels.max()
    if max_val > 0:
        pixels /= max_val
    return pixels


def _load_raster_pixels(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image at {path}")
    return img.astype(np.float32) / 255.0


def load_grayscale(path: Path, apply_voi_lut_flag: bool, invert_monochrome1: bool) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        if not apply_voi_lut_flag:
            # rare case: caller explicitly wants raw pixel values, e.g. for
            # debugging a scanner-specific windowing issue
            dcm = pydicom.dcmread(str(path))
            pixels = dcm.pixel_array.astype(np.float32)
            if invert_monochrome1 and getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixels = pixels.max() - pixels
            pixels -= pixels.min()
            m = pixels.max()
            return pixels / m if m > 0 else pixels
        return _load_dicom_pixels(path)
    return _load_raster_pixels(path)


class MammographyDataset(Dataset):
    """CSV-driven — expects columns [image_path, mask_path, label, patient_id].
    mask_path may be empty/NaN for unmasked samples. Paths are resolved
    relative to `image_root`/`mask_root` if not already absolute.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        mask_root: str | Path,
        image_size: int = 512,
        apply_voi_lut_flag: bool = True,
        invert_monochrome1: bool = True,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)
        required = {"image_path", "label", "patient_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_size = image_size
        self.apply_voi_lut_flag = apply_voi_lut_flag
        self.invert_monochrome1 = invert_monochrome1
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, root: Path, value) -> Path | None:
        if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
            return None
        p = Path(value)
        return p if p.is_absolute() else root / p

    def __getitem__(self, idx: int) -> MammoSample:
        row = self.df.iloc[idx]

        image_path = self._resolve(self.image_root, row["image_path"])
        image = load_grayscale(image_path, self.apply_voi_lut_flag, self.invert_monochrome1)

        mask_path = self._resolve(self.mask_root, row.get("mask_path"))
        has_mask = mask_path is not None and mask_path.exists()
        if has_mask:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros_like(image, dtype=np.float32)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(image).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)

        return MammoSample(
            image=image_t,
            mask=mask_t,
            has_mask=has_mask,
            label=torch.tensor(float(row["label"]), dtype=torch.float32),
            patient_id=str(row["patient_id"]),
        )


def collate_mammo(batch: list[MammoSample]) -> dict:
    """Default collate can't handle the dataclass directly, and we want
    has_mask to stay a plain bool list (per-sample masking in the loss)
    rather than getting stacked into a tensor.
    """
    return {
        "image": torch.stack([b.image for b in batch]),
        "mask": torch.stack([b.mask for b in batch]),
        "has_mask": torch.tensor([b.has_mask for b in batch], dtype=torch.bool),
        "label": torch.stack([b.label for b in batch]),
        "patient_id": [b.patient_id for b in batch],
    }
