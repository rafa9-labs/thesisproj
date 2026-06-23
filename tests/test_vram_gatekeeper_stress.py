"""Concurrency stress tests for VRAM Gatekeeper under load.

Tests thread-safe allocation, release cycles, and mixed CPU/GPU submissions.
"""
from __future__ import annotations

import threading
import time

import pytest

from api.process_manager import ProcessManager


@pytest.fixture
def pm(monkeypatch):
    monkeypatch.setattr("api.process_manager.settings.gpu_total_vram_mb", 8192)
    return ProcessManager()


class TestGatekeeperUnderLoad:

    def test_queue_5_gpu_jobs_with_2_slots(self, pm):
        """5 concurrent GPU allocs of 4096MB on 8192 total → exactly 2 succeed."""
        successes = []
        failures = []

        def try_alloc():
            if pm.allocate_vram(4096):
                successes.append(1)
            else:
                failures.append(1)

        threads = [threading.Thread(target=try_alloc) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 2
        assert len(failures) == 3
        assert pm.gpu_vram_used_mb == 8192

    def test_vram_released_between_submissions(self, pm):
        """Submit job, complete, then next job can allocate."""
        assert pm.allocate_vram(4096) is True
        assert pm.gpu_vram_used_mb == 4096
        pm.release_vram(4096)
        assert pm.allocate_vram(4096) is True
        assert pm.gpu_vram_used_mb == 4096

    def test_vram_ledger_consistent_after_100_rapid_cycles(self, pm):
        """Allocate+release 100 times → ledger stays consistent."""
        for _ in range(100):
            assert pm.allocate_vram(1024) is True
            pm.release_vram(1024)
        assert pm.gpu_vram_used_mb == 0

    def test_mixed_cpu_gpu_submissions(self, pm):
        """CPU allocs always succeed (no VRAM gate), GPU allocs are gated."""
        cpu_successes = []
        gpu_successes = []
        gpu_failures = []

        def cpu_alloc():
            cpu_successes.append(1)

        def gpu_alloc():
            if pm.allocate_vram(4096):
                gpu_successes.append(1)
            else:
                gpu_failures.append(1)

        threads = []
        threads += [threading.Thread(target=cpu_alloc) for _ in range(3)]
        threads += [threading.Thread(target=gpu_alloc) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(cpu_successes) == 3
        assert len(gpu_successes) == 2
        assert len(gpu_failures) == 1

    def test_rapid_force_stop_releases_vram(self, pm):
        """force-stop triggers release_vram_for_job."""
        pm.allocate_vram(4096)
        pm._job_vram["job-a"] = 4096
        assert pm.gpu_vram_used_mb == 4096
        pm._release_vram_for_job("job-a")
        assert pm.gpu_vram_used_mb == 0

    def test_vram_ledger_never_negative(self, pm):
        """Double-release never produces negative used VRAM."""
        pm.allocate_vram(4096)
        pm.release_vram(8192)
        assert pm.gpu_vram_used_mb == 0
        pm.release_vram(1)
        assert pm.gpu_vram_used_mb == 0
