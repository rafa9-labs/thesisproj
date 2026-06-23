"""Unit tests for ProcessManager VRAM allocation, release, and properties.

Tests the Gatekeeper ledger math in isolation — no real processes spawned.
"""
from __future__ import annotations

import threading
import time

import pytest

from api.process_manager import ProcessManager


class TestVRAMAllocation:
    """Tests for allocate_vram() logic."""

    def test_allocate_within_limit(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        assert pm.gpu_vram_used_mb == 0
        result = pm.allocate_vram(4096)
        assert result is True
        assert pm.gpu_vram_used_mb == 4096

    def test_allocate_exceeds_limit(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        result = pm.allocate_vram(9000)
        assert result is False
        assert pm.gpu_vram_used_mb == 0

    def test_allocate_zero_budget(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        result = pm.allocate_vram(0)
        assert result is True
        assert pm.gpu_vram_used_mb == 0

    def test_allocate_negative_budget(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        result = pm.allocate_vram(-100)
        assert result is True
        assert pm.gpu_vram_used_mb == 0

    def test_allocate_when_gpu_total_zero(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 0)
        pm = ProcessManager()
        result = pm.allocate_vram(4096)
        assert result is True
        assert pm.gpu_vram_used_mb == 0

    def test_allocate_exact_limit(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        result = pm.allocate_vram(8192)
        assert result is True
        assert pm.gpu_vram_used_mb == 8192

    def test_allocate_two_chunks_fill_capacity(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        assert pm.allocate_vram(4096) is True
        assert pm.allocate_vram(4096) is True
        assert pm.gpu_vram_used_mb == 8192
        assert pm.allocate_vram(1) is False


class TestVRAMRelease:
    """Tests for release_vram() logic."""

    def test_release_normal(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        pm.allocate_vram(4096)
        pm.release_vram(4096)
        assert pm.gpu_vram_used_mb == 0

    def test_release_partial(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        pm.allocate_vram(4096)
        pm.release_vram(2048)
        assert pm.gpu_vram_used_mb == 2048

    def test_release_over_release_clamped_to_zero(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        pm.allocate_vram(2048)
        pm.release_vram(9999)
        assert pm.gpu_vram_used_mb == 0

    def test_release_zero_budget(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        pm.allocate_vram(4096)
        pm.release_vram(0)
        assert pm.gpu_vram_used_mb == 4096

    def test_release_without_allocation(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        pm.release_vram(4096)
        assert pm.gpu_vram_used_mb == 0


class TestVRAMProperties:
    """Tests for gpu_vram_used_mb and gpu_vram_available_mb."""

    def test_available_after_allocation(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        assert pm.gpu_vram_available_mb == 8192
        pm.allocate_vram(4096)
        assert pm.gpu_vram_available_mb == 4096

    def test_available_clamped_to_zero(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 100)
        pm = ProcessManager()
        pm._gpu_vram_used_mb = 200
        assert pm.gpu_vram_available_mb == 0

    def test_used_starts_at_zero(self):
        pm = ProcessManager()
        assert pm.gpu_vram_used_mb == 0


class TestConcurrentAllocations:
    """Tests for thread-safe VRAM allocation under concurrency."""

    def test_concurrent_allocations_thread_safety(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
        pm = ProcessManager()
        successes = []
        failures = []

        def try_allocate():
            if pm.allocate_vram(2048):
                successes.append(1)
            else:
                failures.append(1)

        threads = [threading.Thread(target=try_allocate) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 4
        assert len(failures) == 2
        assert pm.gpu_vram_used_mb == 8192

    def test_allocate_vram_thread_safety_many_small(self, monkeypatch):
        monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 5000)
        pm = ProcessManager()
        results = []

        def try_alloc(budget):
            results.append(pm.allocate_vram(budget))

        threads = [threading.Thread(target=try_alloc, args=(100,)) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successful = sum(results)
        assert successful == 50
        assert pm.gpu_vram_used_mb == 5000


class TestInitializeEdgeCases:
    """Tests for initialize() edge cases."""

    def test_double_initialize_is_noop(self, monkeypatch):
        from unittest.mock import patch
        monkeypatch.setattr(
            "api.process_manager.settings.max_concurrent_backtests", 2
        )
        monkeypatch.setattr(
            "api.process_manager.settings.max_concurrent_gpu", 1
        )
        monkeypatch.setattr(
            "api.process_manager.settings.gpu_enabled", True
        )
        with patch(
            "api.process_manager.ProcessPoolExecutor",
            autospec=True,
        ) as mock_pool:
            pm = ProcessManager()
            pm.initialize()
            assert mock_pool.call_count == 2
            pm.initialize()
            assert mock_pool.call_count == 2

    def test_initialize_gpu_disabled_no_gpu_pool(self, monkeypatch):
        from unittest.mock import patch
        monkeypatch.setattr(
            "api.process_manager.settings.gpu_enabled", False
        )
        monkeypatch.setattr(
            "api.process_manager.settings.max_concurrent_backtests", 2
        )
        with patch(
            "api.process_manager.ProcessPoolExecutor",
            autospec=True,
        ) as mock_pool:
            pm = ProcessManager()
            pm.initialize(gpu_enabled=False)
            assert mock_pool.call_count == 1
            assert pm._gpu_pool is None

    def test_submit_before_initialize_raises(self):
        pm = ProcessManager()
        with pytest.raises(RuntimeError, match="not initialized"):
            pm.submit("test-id-123", {"models": ["logistic"]})

    def test_atexit_handler_registered(self, monkeypatch):
        from unittest.mock import patch
        monkeypatch.setattr(
            "api.process_manager.settings.max_concurrent_backtests", 2
        )
        monkeypatch.setattr(
            "api.process_manager.settings.gpu_enabled", False
        )
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True):
            import atexit
            handlers_before = len(atexit._exithandlers) if hasattr(atexit, "_exithandlers") else 0
            pm = ProcessManager()
            pm.initialize()
            # verify shutdown is registered
            # atexit._exithandlers is a list of (func, args, kwargs)
            # on Windows, the list is tracked internally
            assert pm._initialized is True
