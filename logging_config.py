# logging_config.py -- Structured logging for FX MLBacktester
#
# Supports two output formats controlled by LOG_FORMAT env var:
#   LOG_FORMAT=text  (default) -- human-readable: [LEVEL] message
#   LOG_FORMAT=json           -- machine-parseable: {"ts","level","msg","module"}
#
# LOG_MODE controls verbosity: COMPACT (default), DEBUG, QUIET.
#
# Usage:
#   from logging_config import get_logger
#   log = get_logger(__name__)
#   log.info("Training started")

"""
Centralized logging configuration.

Drop-in replacement for the ad-hoc log_print() / print() pattern.
Supports human-readable text (default) and structured JSON output
for machine parsing (e.g. Electron frontend progress tracking).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional


# ---------------------------------------------------------------------------
# LOG_MODE backward-compatible mapping
# ---------------------------------------------------------------------------
_LOG_MODE_MAP = {
    "QUIET": logging.WARNING + 10,
    "COMPACT": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_DEFAULT_LEVEL = logging.INFO


def _resolve_level() -> int:
    mode = os.getenv("LOG_MODE", "COMPACT").upper().strip()
    return _LOG_MODE_MAP.get(mode, _DEFAULT_LEVEL)


def _is_json_mode() -> bool:
    return os.getenv("LOG_FORMAT", "text").lower().strip() in ("json", "1", "true")


# ---------------------------------------------------------------------------
# Custom handlers
# ---------------------------------------------------------------------------
class _SafeStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            try:
                self.stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                safe = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                self.stream.write(safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class _CompactFormatter(logging.Formatter):
    _compact = logging.Formatter("[%(levelname)s] %(message)s")
    _debug = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno <= logging.DEBUG:
            return self._debug.format(record)
        return self._compact.format(record)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.name,
        }
        if record.exc_info and record.exc_info[0]:
            entry["exc_type"] = record.exc_info[0].__name__
            entry["exc_msg"] = str(record.exc_info[1])
        try:
            return json.dumps(entry, ensure_ascii=True)
        except (TypeError, ValueError):
            entry["msg"] = str(record.getMessage())
            return json.dumps(entry, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
_configured: bool = False


def _ensure_root_configured() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level = _resolve_level()
    logger = logging.getLogger("mlbacktester")
    logger.setLevel(level)

    if not logger.handlers:
        handler = _SafeStreamHandler(sys.stdout)
        handler.setLevel(level)
        if _is_json_mode():
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(_CompactFormatter())
        logger.addHandler(handler)

    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _ensure_root_configured()
    if name:
        return logging.getLogger(f"mlbacktester.{name}")
    return logging.getLogger("mlbacktester")


# ---------------------------------------------------------------------------
# Backward-compatible log_print() shim
# ---------------------------------------------------------------------------
_LEVEL_MAP = {
    "COMPACT": logging.INFO,
    "DEBUG": logging.DEBUG,
    "QUIET": logging.WARNING,
}


def log_print(msg: str, level: str = "COMPACT") -> None:
    _ensure_root_configured()
    logger = logging.getLogger("mlbacktester")
    py_level = _LEVEL_MAP.get(level.upper().strip(), logging.INFO)
    try:
        logger.log(py_level, msg)
    except UnicodeEncodeError:
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        logger.log(py_level, safe_msg)


# ---------------------------------------------------------------------------
# Structured event helper (for Electron progress tracking)
# ---------------------------------------------------------------------------
def emit_event(event_type: str, **kwargs) -> None:
    """Emit a structured JSON event for machine parsing.

    Always outputs JSON (even in text mode) so the Electron parser
    can reliably detect event lines by the ``"evt"`` key.
    In text mode the JSON is prefixed with [EVT] for grep-ability.
    """
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "evt": event_type}
    entry.update(kwargs)
    try:
        line = json.dumps(entry, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        entry = {"ts": entry["ts"], "evt": event_type, "raw": str(kwargs)}
        line = json.dumps(entry, ensure_ascii=True)
    if _is_json_mode():
        sys.stdout.write(line + "\n")
    else:
        sys.stdout.write("[EVT] " + line + "\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Suppress noisy third-party loggers
# ---------------------------------------------------------------------------
def suppress_noisy_loggers() -> None:
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


suppress_noisy_loggers()