"""Shared FastAPI dependencies with thread-safe singleton pattern."""
from __future__ import annotations

import threading

from api.config import Settings, settings
from pipeline.data.data_sqlite import DataStore

_lock = threading.Lock()
_store: DataStore | None = None


def get_settings() -> Settings:
    return settings


def get_data_store() -> DataStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = DataStore(settings.db_full_path)
    return _store
