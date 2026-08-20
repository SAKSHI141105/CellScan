"""Builds an image model with ImageNet-pretrained weights only — no
histopathology fine-tuning at all — and saves it where image_service.py
looks for a trained checkpoint, so the Upload Image page has something to
run end-to-end while you're waiting on the real dataset download/training
run.

This is NOT a diagnostic model. Its predictions are effectively noise —
ImageNet features run through an untrained classification head initialized
at random. It exists purely so the UI, Grad-CAM plumbing, and report export
can be exercised without a trained checkpoint. The API tags every response
built from this file with "is_demo": true, and the dashboard shows a
persistent warning banner whenever it's in use — don't strip that labeling
if you're editing this.

    python scripts/generate_demo_weights.py
"""
from __future__ import annotations

from src.models import image_cnn
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

DEMO_FILENAME = "demo_untrained_transfer_{backbone}.h5"


def main():
    cfg = load_config()
    img_cfg = cfg["image"]
    backbone = img_cfg["transfer_backbone"]

    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "image"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / DEMO_FILENAME.format(backbone=backbone)

    logger.warning(
        "Building a DEMO model (%s, ImageNet weights only, zero histopathology training). "
        "This is for exercising the UI, not for anything resembling a real prediction.",
        backbone,
    )
    model = image_cnn.build_transfer_model(backbone, img_cfg["img_size"], img_cfg["fine_tune_last_n_layers"])
    model.save(out_path)

    logger.info("Saved demo weights to %s", out_path)
    logger.info("Restart the API (or just re-run `python run.py`) to pick it up.")


if __name__ == "__main__":
    main()
