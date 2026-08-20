"""Exercises the actual file-loading path with synthetic PNGs on disk —
model tests use random tensors and never touch dataset.py at all, so this
is the only place the resize/mask-alignment/has_mask logic gets checked.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.dataset import MammographyDataset, collate_mammo


@pytest.fixture
def synthetic_dataset(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.default_rng(0)

    # row 0: has a mask
    img0 = (rng.random((300, 200)) * 255).astype(np.uint8)
    cv2.imwrite(str(image_dir / "sample0.png"), img0)
    mask0 = np.zeros((300, 200), dtype=np.uint8)
    mask0[100:150, 80:120] = 255
    cv2.imwrite(str(mask_dir / "sample0_mask.png"), mask0)

    # row 1: no mask (blank/benign-without-callback style)
    img1 = (rng.random((300, 200)) * 255).astype(np.uint8)
    cv2.imwrite(str(image_dir / "sample1.png"), img1)

    df = pd.DataFrame({
        "image_path": ["sample0.png", "sample1.png"],
        "mask_path": ["sample0_mask.png", ""],
        "label": [1, 0],
        "patient_id": ["P0", "P1"],
    })
    csv_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)

    return MammographyDataset(
        csv_path=csv_path, image_root=image_dir, mask_root=mask_dir,
        image_size=128, apply_voi_lut_flag=False, invert_monochrome1=False,
    )


def test_resizes_image_and_mask_to_configured_size(synthetic_dataset):
    sample = synthetic_dataset[0]
    assert sample.image.shape == (1, 128, 128)
    assert sample.mask.shape == (1, 128, 128)


def test_has_mask_flag_reflects_actual_file_presence(synthetic_dataset):
    assert synthetic_dataset[0].has_mask is True
    assert synthetic_dataset[1].has_mask is False


def test_missing_mask_returns_zero_mask_not_garbage(synthetic_dataset):
    sample = synthetic_dataset[1]
    assert sample.mask.sum().item() == 0.0


def test_mask_values_stay_binary_after_resize(synthetic_dataset):
    sample = synthetic_dataset[0]
    unique_vals = set(sample.mask.unique().tolist())
    assert unique_vals <= {0.0, 1.0}


def test_image_normalized_to_unit_range(synthetic_dataset):
    sample = synthetic_dataset[0]
    assert sample.image.min() >= 0.0
    assert sample.image.max() <= 1.0


def test_collate_batches_correctly(synthetic_dataset):
    batch = collate_mammo([synthetic_dataset[0], synthetic_dataset[1]])
    assert batch["image"].shape == (2, 1, 128, 128)
    assert batch["has_mask"].tolist() == [True, False]
    assert batch["label"].tolist() == [1.0, 0.0]
