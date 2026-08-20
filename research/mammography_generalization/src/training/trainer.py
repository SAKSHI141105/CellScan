"""Training loop. One `Trainer` class handles both the baseline and
lesion-guided runs — which model class gets built and whether the lesion
loss actually contributes is entirely a function of the config
(model.decoder_channels present vs absent, training.lesion_loss_weight),
not a fork in this file. Keeps the two experiment configs honest: if they
diverge anywhere except the auxiliary-loss knobs, this code doesn't care and
won't paper over it.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.baseline_classifier import BaselineClassifier
from src.models.lesion_guided_model import LesionGuidedModel
from src.training.losses import JointLoss
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    if "decoder_channels" in model_cfg:
        return LesionGuidedModel(
            backbone_name=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            in_channels=model_cfg["in_channels"],
            num_classes=model_cfg["num_classes"],
            decoder_channels=model_cfg["decoder_channels"],
        )
    return BaselineClassifier(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
    )


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    if train_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    if train_cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=0.9, weight_decay=train_cfg["weight_decay"])
    raise ValueError(f"unknown optimizer: {train_cfg['optimizer']}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    train_cfg = cfg["training"]
    if train_cfg.get("lr_scheduler") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    return None


class Trainer:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.criterion = JointLoss(
            lesion_loss_weight=cfg["training"]["lesion_loss_weight"],
            lesion_loss_type=cfg["training"].get("lesion_loss_type", "dice_bce"),
        )

        self.checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _move_batch(self, batch: dict) -> dict:
        return {
            "image": batch["image"].to(self.device),
            "mask": batch["mask"].to(self.device),
            "has_mask": batch["has_mask"].to(self.device),
            "label": batch["label"].to(self.device),
        }

    def _forward(self, batch: dict) -> dict:
        outputs = self.model(batch["image"])
        if isinstance(outputs, torch.Tensor):
            outputs = {"class_logits": outputs}
        return outputs

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "classification": 0.0, "lesion_guidance": 0.0}
        n_batches = 0

        for raw_batch in tqdm(loader, desc="train", leave=False):
            batch = self._move_batch(raw_batch)
            self.optimizer.zero_grad()
            outputs = self._forward(batch)
            losses = self.criterion(outputs, batch)
            losses["total"].backward()
            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals = {"total": 0.0, "classification": 0.0, "lesion_guidance": 0.0}
        n_batches = 0

        for raw_batch in tqdm(loader, desc="val", leave=False):
            batch = self._move_batch(raw_batch)
            outputs = self._forward(batch)
            losses = self.criterion(outputs, batch)
            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs = self.cfg["training"]["epochs"]
        patience = self.cfg["training"]["early_stopping_patience"]

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            logger.info(
                "epoch %d/%d (%.1fs) | train loss=%.4f (cls=%.4f, lesion=%.4f) | val loss=%.4f (cls=%.4f, lesion=%.4f)",
                epoch, epochs, time.time() - t0,
                train_metrics["total"], train_metrics["classification"], train_metrics["lesion_guidance"],
                val_metrics["total"], val_metrics["classification"], val_metrics["lesion_guidance"],
            )

            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= patience:
                    logger.info("early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                    break

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        path = self.checkpoint_dir / ("best.pt" if is_best else f"epoch_{epoch}.pt")
        torch.save(
            {"epoch": epoch, "model_state": self.model.state_dict(), "metrics": metrics, "config": self.cfg},
            path,
        )
        logger.info("saved checkpoint: %s", path)
