"""Loads config/config.yaml once and hands back a plain dict.

Kept deliberately dumb — no schema validation library, no singleton class.
We only have one config file and it's small enough that a dataclass wrapper
would be more code than it saves.
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative: str) -> Path:
    """Config stores paths relative to project root; this makes them absolute."""
    p = PROJECT_ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(parents=True, exist_ok=True)
    return p
