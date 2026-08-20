"""Zero-shot cross-dataset evaluation — load a checkpoint trained on one
dataset (default: CBIS-DDSM), run it unmodified against another (default:
VinDr-Mammo) with no fine-tuning. This is the actual experiment: a model
that only hit high in-distribution accuracy by fitting CBIS-specific scanner
characteristics should show a bigger AUC drop here than one whose backbone
was pushed toward lesion-level features.

    python -m src.evaluation.evaluate_cross_dataset \
        --checkpoint outputs/lesion_guided_resnet50_cbis/checkpoints/best.pt \
        --target-csv data/processed/vindr_mammo/test.csv \
        --target-image-root data/processed/vindr_mammo/images \
        --target-mask-root data/processed/vindr_mammo/masks
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import MammographyDataset, collate_mammo
from src.training.trainer import build_model
from src.utils.gradcam import GradCAM, find_last_conv, mask_alignment_iou
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["config"]


@torch.no_grad()
def run_predictions(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        logits = outputs["class_logits"] if isinstance(outputs, dict) else outputs
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_attention_alignment(model, dataset: MammographyDataset, device: str, n_samples: int = 50) -> float | None:
    """Averages Grad-CAM/mask IoU over samples that actually have a mask —
    skipped entirely (returns None) if the target dataset has no masks at
    all, which is a legitimate state for a pure zero-shot classification
    benchmark that doesn't ship lesion annotations.
    """
    target_layer = find_last_conv(model.backbone.backbone)
    cam = GradCAM(model, target_layer)

    ious = []
    checked = 0
    for i in range(len(dataset)):
        if checked >= n_samples:
            break
        sample = dataset[i]
        if not sample.has_mask:
            continue
        image = sample.image.unsqueeze(0).to(device)
        heatmap = cam(image)
        iou = mask_alignment_iou(heatmap, sample.mask[0].numpy())
        ious.append(iou)
        checked += 1

    return float(np.mean(ious)) if ious else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--target-image-root", required=True)
    parser.add_argument("--target-mask-root", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, _ = load_checkpoint(args.checkpoint, args.device)

    dataset = MammographyDataset(
        csv_path=args.target_csv,
        image_root=args.target_image_root,
        mask_root=args.target_mask_root,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mammo)

    probs, labels = run_predictions(model, loader, args.device)
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    alignment_iou = compute_attention_alignment(model, dataset, args.device)

    results = {
        "checkpoint": args.checkpoint,
        "target_dataset": args.target_csv,
        "n_samples": len(dataset),
        "auc": round(auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "gradcam_mask_iou": round(alignment_iou, 4) if alignment_iou is not None else None,
    }
    logger.info("cross-dataset zero-shot results: %s", json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
