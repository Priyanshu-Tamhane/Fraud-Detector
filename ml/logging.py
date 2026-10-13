"""Structured logging configuration for FraudShield."""

import logging
import sys

from rich.logging import RichHandler


def setup_logger(name: str = "fraudshield", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger with rich console formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            tracebacks_suppress=[sys],
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
