"""Builds demo checkpoints for whichever image-based pipelines don't have a
real trained model yet — histopathology (TensorFlow/Keras) and mammography
(PyTorch) — so their Upload pages work end to end without waiting on a
dataset download/training run first.

Neither of these is a diagnostic model. Both carry ImageNet-pretrained
weights only, run through an untrained (or, for mammography, freshly
initialized) head. Predictions from them are effectively noise. They exist
purely so the UI, Grad-CAM plumbing, and report export can be exercised.
Every response built from one of these checkpoints is tagged "is_demo":
true by the corresponding service module, and the dashboard shows a
persistent warning wherever it's in use — don't strip that labeling if
you're editing this.

    python scripts/generate_demo_weights.py                # both
    python scripts/generate_demo_weights.py --target histopathology
    python scripts/generate_demo_weights.py --target mammography
"""
from __future__ import annotations

import argparse

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

HISTOPATHOLOGY_DEMO_FILENAME = "demo_untrained_transfer_{backbone}.h5"
MAMMOGRAPHY_DEMO_FILENAME = "demo_lesion_guided_{backbone}.pt"


def generate_histopathology_demo(cfg: dict):
    from src.models import image_cnn

    img_cfg = cfg["image"]
    backbone = img_cfg["transfer_backbone"]

    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "image"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / HISTOPATHOLOGY_DEMO_FILENAME.format(backbone=backbone)

    logger.warning("Building histopathology DEMO model (%s, ImageNet weights only).", backbone)
    model = image_cnn.build_transfer_model(backbone, img_cfg["img_size"], img_cfg["fine_tune_last_n_layers"])
    model.save(out_path)
    logger.info("Saved histopathology demo weights to %s", out_path)


def generate_mammography_demo(cfg: dict):
    import torch

    from src.models.mammography.lesion_guided_model import LesionGuidedModel

    mammo_cfg = cfg["mammography"]
    backbone = mammo_cfg["backbone"]

    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "mammography"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / MAMMOGRAPHY_DEMO_FILENAME.format(backbone=backbone)

    logger.warning("Building mammography DEMO model (%s, ImageNet weights only, no lesion-guided training).", backbone)
    model = LesionGuidedModel(
        backbone_name=backbone,
        pretrained=True,
        in_channels=1,
        num_classes=1,
        decoder_channels=mammo_cfg["decoder_channels"],
    )
    torch.save({"model_state": model.state_dict(), "is_demo": True, "config": {"model": mammo_cfg}}, out_path)
    logger.info("Saved mammography demo weights to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["all", "histopathology", "mammography"], default="all")
    args = parser.parse_args()

    cfg = load_config()
    if args.target in ("all", "histopathology"):
        generate_histopathology_demo(cfg)
    if args.target in ("all", "mammography"):
        generate_mammography_demo(cfg)

    logger.info("Restart the API (or just re-run `python run.py`) to pick these up.")


if __name__ == "__main__":
    main()
