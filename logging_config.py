"""Structured logging setup for CLI and library use."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", quiet: bool = False) -> None:
    lvl = logging.WARNING if quiet else getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
