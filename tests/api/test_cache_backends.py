"""Tests for the pluggable response cache backends."""
from __future__ import annotations

import os
import tempfile
import time
import uuid

import pytest
from fastapi import Response
from pydantic import BaseModel

from api.utils.cache import (
    MemoryCacheBackend,
    RedisCacheBackend,
    SqliteCacheBackend,
    cached,
    clear_cache,
    get_backend,
    reset_backend,
)


class _SampleModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def sqlite_backend(tmp_path):
    db = tmp_path / "cache.db"
    backend = SqliteCacheBackend(str(db))
    yield backend
    backend.close()


@pytest.fixture
def memory_backend():
    return MemoryCacheBackend()


class TestMemoryCacheBackend:
    def test_set_get(self, memory_backend):
        memory_backend.set("ns", "k", {"a": 1}, ttl=60)
        assert memory_backend.get("ns", "k") == {"a": 1}

    def test_missing_returns_none(self, memory_backend):
        assert memory_backend.get("ns", "missing") is None

    def test_ttl_expiration(self, memory_backend):
        memory_backend.set("ns", "k", {"a": 1}, ttl=0)
        time.sleep(0.01)
        assert memory_backend.get("ns", "k") is None

    def test_clear_namespace(self, memory_backend):
        memory_backend.set("a", "k", 1, ttl=60)
        memory_backend.set("b", "k", 2, ttl=60)
        memory_backend.clear("a")
        assert memory_backend.get("a", "k") is None
        assert memory_backend.get("b", "k") == 2


class TestSqliteCacheBackend:
    def test_set_get(self, sqlite_backend):
        sqlite_backend.set("ns", "k", {"a": 1}, ttl=60)
        assert sqlite_backend.get("ns", "k") == {"a": 1}

    def test_missing_returns_none(self, sqlite_backend):
        assert sqlite_backend.get("ns", "missing") is None

    def test_ttl_expiration(self, sqlite_backend):
        sqlite_backend.set("ns", "k", {"a": 1}, ttl=0)
        time.sleep(0.01)
        assert sqlite_backend.get("ns", "k") is None

    def test_clear_namespace(self, sqlite_backend):
        sqlite_backend.set("a", "k", 1, ttl=60)
        sqlite_backend.set("b", "k", 2, ttl=60)
        sqlite_backend.clear("a")
        assert sqlite_backend.get("a", "k") is None
        assert sqlite_backend.get("b", "k") == 2

    def test_pydantic_roundtrip(self, sqlite_backend):
        sqlite_backend.set("ns", "k", _SampleModel(name="x", value=42), ttl=60)
        result = sqlite_backend.get("ns", "k")
        assert result == {"name": "x", "value": 42}

    def test_shared_between_instances(self, tmp_path):
        db = tmp_path / "shared.db"
        a = SqliteCacheBackend(str(db))
        a.set("ns", "k", {"shared": True}, ttl=60)
        b = SqliteCacheBackend(str(db))
        assert b.get("ns", "k") == {"shared": True}


class TestRedisCacheBackend:
    def test_stub_raises(self):
        backend = RedisCacheBackend("redis://localhost")
        with pytest.raises(NotImplementedError):
            backend.get("ns", "k")
        with pytest.raises(NotImplementedError):
            backend.set("ns", "k", 1, ttl=1)
        with pytest.raises(NotImplementedError):
            backend.delete("ns", "k")
        with pytest.raises(NotImplementedError):
            backend.clear("ns")


class TestCacheDecorator:
    def test_decorator_uses_backend_and_sets_headers(self, sqlite_backend):
        reset_backend()
        # Monkey-patch singleton for this test
        from api.utils import cache as cache_mod
        cache_mod._backend = sqlite_backend

        calls = []

        @cached("test_ns", ttl=60)
        def endpoint(response, x: int):
            calls.append(x)
            return _SampleModel(name="result", value=x)

        r1 = Response()
        out1 = endpoint(r1, x=5)
        assert out1 == {"name": "result", "value": 5}
        assert r1.headers["X-Cache"] == "MISS"

        r2 = Response()
        out2 = endpoint(r2, x=5)
        assert out2 == {"name": "result", "value": 5}
        assert r2.headers["X-Cache"] == "HIT"
        assert len(calls) == 1

        clear_cache("test_ns")
        assert sqlite_backend.get("test_ns", (("x", 5),)) is None

        reset_backend()


class TestDefaultBackend:
    def test_default_is_sqlite(self):
        reset_backend()
        backend = get_backend()
        assert isinstance(backend, SqliteCacheBackend)
        reset_backend()

    def test_memory_override_via_env(self, monkeypatch):
        reset_backend()
        monkeypatch.setenv("KODA_CACHE_BACKEND", "memory")
        backend = get_backend()
        assert isinstance(backend, MemoryCacheBackend)
        reset_backend()
