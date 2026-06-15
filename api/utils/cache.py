"""Pluggable response cache for the FastAPI sidecar.

The default backend is SQLite-backed file storage so that multiple Uvicorn
workers on the same node share a coherent cache without requiring Redis.
A memory backend is available for single-process / desktop use, and a
Redis backend stub is defined for future horizontal scaling.

Cached values are normalized to JSON-serializable dicts so that any backend
can store them and FastAPI's ``response_model`` can re-validate on the way out.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Response
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _get_settings() -> Any:
    """Lazy import to avoid circular dependencies during module load."""
    try:
        from api.config import settings
        return settings
    except Exception:
        return None


def _resolve_backend() -> "CacheBackend":
    """Instantiate the configured cache backend."""
    settings = _get_settings()
    backend = (
        os.environ.get("KODA_CACHE_BACKEND", "")
        or (settings.cache_backend if settings else "sqlite")
    ).lower()

    if backend == "memory":
        return MemoryCacheBackend()

    if backend == "redis":
        redis_url = os.environ.get("REDIS_URL", "")
        if settings and not redis_url:
            redis_url = settings.redis_url
        return RedisCacheBackend(redis_url)

    db_path = os.environ.get("KODA_CACHE_DB_PATH", "")
    if not db_path and settings:
        db_path = settings.cache_db_full_path
    if not db_path:
        db_path = "data/cache.db"
    return SqliteCacheBackend(db_path)


_backend: Optional[CacheBackend] = None
_backend_lock = threading.Lock()


def get_backend() -> CacheBackend:
    """Return the singleton cache backend instance."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _resolve_backend()
    return _backend


def reset_backend() -> None:
    """Close and reset the singleton backend (useful in tests)."""
    global _backend
    with _backend_lock:
        if _backend is not None:
            try:
                _backend.close()
            except Exception:
                pass
            _backend = None


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts for JSON storage."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _serialize(value: Any) -> str:
    return json.dumps(_jsonable(value), default=str, ensure_ascii=False)


def _deserialize(value: str) -> Any:
    return json.loads(value)


def _key_str(key: Any) -> str:
    """Stable string representation of a cache key."""
    return json.dumps(key, default=str, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class CacheBackend(ABC):
    @abstractmethod
    def get(self, namespace: str, key: Any) -> Any | None:
        ...

    @abstractmethod
    def set(self, namespace: str, key: Any, value: Any, ttl: int) -> None:
        ...

    @abstractmethod
    def delete(self, namespace: str, key: Any) -> None:
        ...

    @abstractmethod
    def clear(self, namespace: str) -> None:
        ...

    def close(self) -> None:
        """Optional lifecycle hook called on shutdown."""
        pass


class MemoryCacheBackend(CacheBackend):
    """Per-process in-memory cache. Fast, but not shared across workers."""

    def __init__(self) -> None:
        self._data: Dict[tuple[str, str], tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def _purge_expired(self, namespace: str) -> None:
        now = self._now()
        expired = [
            k for k, (v, exp) in self._data.items()
            if k[0] == namespace and exp <= now
        ]
        for k in expired:
            del self._data[k]

    def get(self, namespace: str, key: Any) -> Any | None:
        key_str = _key_str(key)
        with self._lock:
            self._purge_expired(namespace)
            entry = self._data.get((namespace, key_str))
            if entry is None:
                return None
            value, expires = entry
            if expires <= self._now():
                del self._data[(namespace, key_str)]
                return None
            return value

    def set(self, namespace: str, key: Any, value: Any, ttl: int) -> None:
        key_str = _key_str(key)
        with self._lock:
            self._data[(namespace, key_str)] = (value, self._now() + ttl)

    def delete(self, namespace: str, key: Any) -> None:
        key_str = _key_str(key)
        with self._lock:
            self._data.pop((namespace, key_str), None)

    def clear(self, namespace: str) -> None:
        with self._lock:
            for k in list(self._data.keys()):
                if k[0] == namespace:
                    del self._data[k]


class SqliteCacheBackend(CacheBackend):
    """File-backed cache using a separate SQLite database.

    This backend is shared across all workers on the same node and persists
    across restarts. It uses WAL mode to minimize reader/writer contention.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-16000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    namespace TEXT NOT NULL,
                    key       TEXT NOT NULL,
                    value     TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_cache_expires "
                "ON api_cache(expires_at)"
            )

    def _now(self) -> float:
        return time.time()

    def _cleanup_expired(self, conn: sqlite3.Connection, namespace: str) -> None:
        conn.execute(
            "DELETE FROM api_cache WHERE namespace = ? AND expires_at <= ?",
            (namespace, self._now()),
        )

    def get(self, namespace: str, key: Any) -> Any | None:
        key_str = _key_str(key)
        with self._connect() as conn:
            self._cleanup_expired(conn, namespace)
            cur = conn.execute(
                "SELECT value FROM api_cache WHERE namespace = ? AND key = ?",
                (namespace, key_str),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _deserialize(row[0])

    def set(self, namespace: str, key: Any, value: Any, ttl: int) -> None:
        key_str = _key_str(key)
        expires_at = self._now() + ttl
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (namespace, key, value, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace, key_str, _serialize(value), expires_at),
            )

    def delete(self, namespace: str, key: Any) -> None:
        key_str = _key_str(key)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM api_cache WHERE namespace = ? AND key = ?",
                (namespace, key_str),
            )

    def clear(self, namespace: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM api_cache WHERE namespace = ?",
                (namespace,),
            )

    def close(self) -> None:
        pass


class RedisCacheBackend(CacheBackend):
    """Stub for future horizontal scaling.

    Defining the class completes the backend abstraction, but the methods are
    intentionally not implemented until multi-node scaling is required.
    """

    def __init__(self, url: str) -> None:
        self.url = url

    def get(self, namespace: str, key: Any) -> Any | None:
        raise NotImplementedError("Redis cache backend is not yet implemented")

    def set(self, namespace: str, key: Any, value: Any, ttl: int) -> None:
        raise NotImplementedError("Redis cache backend is not yet implemented")

    def delete(self, namespace: str, key: Any) -> None:
        raise NotImplementedError("Redis cache backend is not yet implemented")

    def clear(self, namespace: str) -> None:
        raise NotImplementedError("Redis cache backend is not yet implemented")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clear_cache(name: str) -> None:
    """Clear an entire namespace (e.g., after a mutation)."""
    get_backend().clear(name)


def _default_key(args: Dict[str, Any]) -> tuple:
    """Build a cache key from bound arguments, ignoring injected objects."""
    return tuple(
        (k, v)
        for k, v in args.items()
        if not isinstance(v, (Response,))
    )


def cached(
    cache_name: str,
    *,
    ttl: int,
    key_func: Optional[Callable[[Dict[str, Any]], Any]] = None,
):
    """Decorator that caches a FastAPI endpoint result.

    Parameters
    ----------
    cache_name : str
        Logical cache bucket / namespace; use ``clear_cache(name)`` to invalidate.
    ttl : int
        TTL in seconds (also emitted as ``Cache-Control: private, max-age=<ttl>``).
    key_func : callable, optional
        Custom cache-key builder receiving the bound arguments dict.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        has_response = "response" in sig.parameters

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key = key_func(bound.arguments) if key_func else _default_key(bound.arguments)

            backend = get_backend()
            cached_value = backend.get(cache_name, key)
            if cached_value is not None:
                hit = True
            else:
                cached_value = _jsonable(func(*args, **kwargs))
                backend.set(cache_name, key, cached_value, ttl)
                hit = False

            response = bound.arguments.get("response") if has_response else None
            if isinstance(response, Response):
                response.headers["Cache-Control"] = f"private, max-age={ttl}"
                response.headers["X-Cache"] = "HIT" if hit else "MISS"
            return cached_value

        # Preserve FastAPI's dependency-injection signature.
        wrapper.__signature__ = sig  # type: ignore[attr-defined]
        wrapper._cache_name = cache_name  # type: ignore[attr-defined]
        return wrapper

    return decorator
