"""Logging configuration for Vektori."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure structured logging for Vektori."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("vektori").setLevel(log_level)
