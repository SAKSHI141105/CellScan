"""One place to configure logging so every module just does `logger = get_logger(__name__)`."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config

_CONFIGURED = False


def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    cfg = load_config().get("logging", {})
    level = getattr(logging, cfg.get("level", "INFO"))
    log_dir = PROJECT_ROOT / cfg.get("log_dir", "logs")
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "cellscan.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
