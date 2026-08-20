"""CLI entry point.

    python train.py --config configs/baseline_cbis.yaml
    python train.py --config configs/lesion_guided_cbis.yaml
"""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import MammographyDataset, collate_mammo
from src.training.trainer import Trainer
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]

    common_kwargs = dict(
        image_root=data_cfg["image_root"],
        mask_root=data_cfg["mask_root"],
        image_size=data_cfg["image_size"],
        apply_voi_lut_flag=data_cfg["apply_voi_lut"],
        invert_monochrome1=data_cfg["invert_monochrome1"],
    )
    train_set = MammographyDataset(csv_path=data_cfg["train_csv"], **common_kwargs)
    val_set = MammographyDataset(csv_path=data_cfg["val_csv"], **common_kwargs)

    train_loader = DataLoader(
        train_set, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"], collate_fn=collate_mammo, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], collate_fn=collate_mammo,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["training"]["seed"])

    logger.info("experiment: %s", cfg["experiment_name"])
    train_loader, val_loader = build_loaders(cfg)

    trainer = Trainer(cfg)
    logger.info("model params: %.1fM", sum(p.numel() for p in trainer.model.parameters()) / 1e6)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
