# logging_config.py — Structured logging for FX MLBacktester
#
# Replaces raw print() and log_print() calls with Python's logging module.
# Supports the same LOG_MODE levels (COMPACT, DEBUG, QUIET) for backward compat.
#
# Usage:
#   from logging_config import get_logger
#   log = get_logger(__name__)
#   log.info("Training started")
#   log.debug("Verbose details")

"""
Centralized logging configuration.

Drop-in replacement for the ad-hoc log_print() / print() pattern.
Maintains backward compatibility with LOG_MODE env variable.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Optional


# ---------------------------------------------------------------------------
# LOG_MODE backward-compatible mapping
# ---------------------------------------------------------------------------
_LOG_MODE_MAP = {
    "QUIET": logging.WARNING + 10,    # suppress almost everything
    "COMPACT": logging.INFO,           # normal operation
    "DEBUG": logging.DEBUG,            # verbose
}

_DEFAULT_LEVEL = logging.INFO


def _resolve_level() -> int:
    """Resolve log level from LOG_MODE env var."""
    mode = os.getenv("LOG_MODE", "COMPACT").upper().strip()
    return _LOG_MODE_MAP.get(mode, _DEFAULT_LEVEL)


# ---------------------------------------------------------------------------
# Custom formatter
# ---------------------------------------------------------------------------
class _CompactFormatter(logging.Formatter):
    """
    Compact format:  [LEVEL] message
    For DEBUG level: includes timestamp and module.
    """

    _compact = logging.Formatter("[%(levelname)s] %(message)s")
    _debug = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno <= logging.DEBUG:
            return self._debug.format(record)
        return self._compact.format(record)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
_configured: bool = False


def _ensure_root_configured() -> None:
    """Configure the root 'mlbacktester' logger once."""
    global _configured
    if _configured:
        return
    _configured = True

    level = _resolve_level()
    logger = logging.getLogger("mlbacktester")
    logger.setLevel(level)

    # Only add handler if none exist (prevents duplicate handlers in notebooks)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_CompactFormatter())
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger under the 'mlbacktester' namespace.

    Parameters
    ----------
    name : str, optional
        Sub-module name (e.g. 'pipeline.data_loader').
        If None, returns the root 'mlbacktester' logger.

    Returns
    -------
    logging.Logger
    """
    _ensure_root_configured()
    if name:
        return logging.getLogger(f"mlbacktester.{name}")
    return logging.getLogger("mlbacktester")


# ---------------------------------------------------------------------------
# Backward-compatible log_print() shim
# ---------------------------------------------------------------------------
def log_print(msg: str, level: str = "COMPACT") -> None:
    """
    Backward-compatible shim for the existing log_print() calls.

    Maps the old level strings to Python logging levels:
        - "COMPACT"  → INFO
        - "DEBUG"    → DEBUG
        - "QUIET"    → WARNING

    This function is intentionally a thin wrapper so the existing
    codebase doesn't need to change all call sites at once.
    """
    _level_map = {
        "COMPACT": logging.INFO,
        "DEBUG": logging.DEBUG,
        "QUIET": logging.WARNING,
    }
    _ensure_root_configured()
    logger = logging.getLogger("mlbacktester")
    py_level = _level_map.get(level.upper().strip(), logging.INFO)
    logger.log(py_level, msg)


# ---------------------------------------------------------------------------
# Suppress noisy third-party loggers
# ---------------------------------------------------------------------------
def suppress_noisy_loggers() -> None:
    """Suppress verbose third-party loggers (optuna, PIL, matplotlib, etc.)."""
    for name in (
        "optuna",
        "PIL",
        "matplotlib",
        "urllib3",
        "filelock",
        "tensorflow",
        "absl",
        "numba",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# Auto-suppress on import
suppress_noisy_loggers()