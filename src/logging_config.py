"""Application logging configuration."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure concise console logging without exposing sensitive values."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

