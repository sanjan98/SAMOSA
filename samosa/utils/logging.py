"""Centralized logging utilities for the SAMOSA package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class SamosaLogger:
    """Factory class for configured SAMOSA loggers."""

    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    @classmethod
    def get_logger(
        cls,
        name: str = "samosa",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        propagate: bool = False,
    ) -> logging.Logger:
        """Return a logger with consistent SAMOSA formatting/handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = propagate

        if not logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(cls._formatter)
            logger.addHandler(stream_handler)

        if log_file is not None:
            log_path = Path(log_file)
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
                for handler in logger.handlers
            )
            if not has_file_handler:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(cls._formatter)
                logger.addHandler(file_handler)

        return logger
