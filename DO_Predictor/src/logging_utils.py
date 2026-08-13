"""Logging helpers for research experiments."""

from __future__ import annotations

import logging
from pathlib import Path

from config.research_config import LOG_FORMAT


def setup_experiment_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger
