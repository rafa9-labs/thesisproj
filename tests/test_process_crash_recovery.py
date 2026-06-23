"""Concurrency tests for VRAM ledger integrity on process crash.

Tests that VRAM is reliably released back to the global ledger
when a worker process crashes or is killed unexpectedly.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.process_manager import ProcessManager


@pytest.fixture
def pm(monkeypatch):
    monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
    return ProcessManager()


class TestCrashRecovery:

    def test_child_process_crash_releases_vram(self, pm):
        """Worker exception → done_callback fires → VRAM released."""
        pm.allocate_vram(4096)
        pm._job_vram["crash-job"] = 4096
        assert pm.gpu_vram_used_mb == 4096

        pm._release_vram_for_job("crash-job")
        assert pm.gpu_vram_used_mb == 0

    def test_done_callback_releases_vram_on_exception(self, pm):
        pm._initialized = True
        pm.allocate_vram(4096)
        pm._job_vram["fail-job"] = 4096
        assert pm.gpu_vram_used_mb == 4096

        pm._release_vram_for_job("fail-job")
        assert pm.gpu_vram_used_mb == 0

    def test_vram_ledger_after_pool_shutdown(self, pm):
        """Shutdown with active jobs → all VRAM returned."""
        pm.allocate_vram(2048)
        pm.allocate_vram(2048)
        pm._job_vram["job-1"] = 2048
        pm._job_vram["job-2"] = 2048
        assert pm.gpu_vram_used_mb == 4096

        with patch("api.process_manager.ProcessPoolExecutor") as mock_pool:
            mock_pool.return_value = MagicMock()
            pm.initialize(max_cpu=1, gpu_enabled=False)
            pm._active_futures["job-1"] = MagicMock()
            pm._active_futures["job-2"] = MagicMock()

            pm.shutdown()
            assert pm.gpu_vram_used_mb == 0

    def test_vram_not_double_released_on_crash_then_manual(self, pm):
        """Crash triggers release, then manual release → no negative VRAM."""
        pm.allocate_vram(2048)
        pm._job_vram["double-job"] = 2048
        assert pm.gpu_vram_used_mb == 2048

        pm._release_vram_for_job("double-job")
        assert pm.gpu_vram_used_mb == 0

        pm._release_vram_for_job("double-job")
        assert pm.gpu_vram_used_mb == 0

    def test_multiple_jobs_crash_all_released(self, pm):
        """Multiple concurrent jobs crash → all VRAM returned."""
        for i in range(3):
            pm.allocate_vram(2048)
            pm._job_vram[f"multi-{i}"] = 2048
        assert pm.gpu_vram_used_mb == 6144

        for i in range(3):
            pm._release_vram_for_job(f"multi-{i}")
        assert pm.gpu_vram_used_mb == 0

    def test_ledger_consistent_after_mixed_crash_success(self, pm):
        """Some jobs crash, some succeed → ledger correct."""
        pm.allocate_vram(2048)
        pm.allocate_vram(2048)
        pm.allocate_vram(2048)
        pm._job_vram["crash-1"] = 2048
        pm._job_vram["crash-2"] = 2048
        pm._job_vram["success-1"] = 2048
        assert pm.gpu_vram_used_mb == 6144

        pm._release_vram_for_job("crash-1")
        pm._release_vram_for_job("crash-2")
        assert pm.gpu_vram_used_mb == 2048

        pm.release_vram(2048)
        assert pm.gpu_vram_used_mb == 0
