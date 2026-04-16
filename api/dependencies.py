"""Shared FastAPI dependencies."""
from __future__ import annotations

from functools import lru_cache

from api.config import Settings, settings
from pipeline.data_sqlite import DataStore


@lru_cache
def get_settings() -> Settings:
    return settings


@lru_cache
def get_data_store() -> DataStore:
    return DataStore(settings.db_full_path)
