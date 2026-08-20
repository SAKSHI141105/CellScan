"""One command to get every image-based pipeline (histopathology,
mammography) into a testable state — thin wrapper around
generate_demo_weights.py so `python scripts/setup_demo_environment.py` is a
memorable single entry point for "I just cloned this, let me try it".

Doesn't duplicate that script's logic — see generate_demo_weights.py for
what actually gets built and why every checkpoint it produces is loudly
tagged as a demo, not a real model.

    python scripts/setup_demo_environment.py
"""
from __future__ import annotations

from scripts.generate_demo_weights import main as generate_demo_weights
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Setting up demo environment: histopathology + mammography demo weights, tabular fallback needs no setup.")
    generate_demo_weights()
    logger.info(
        "Demo environment ready. Run `python run.py` and open http://localhost:5173 — "
        "Clinical Data, Upload Image, and Mammography all work immediately."
    )


if __name__ == "__main__":
    main()
