"""Integration tests for ProcessManager lifecycle — init, shutdown, routing.

Tests pool creation, GPU/CPU routing, and active state tracking.
Uses mocked ProcessPoolExecutor to avoid spawning real processes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.process_manager import ProcessManager


@pytest.fixture
def pm():
    return ProcessManager()


class TestInitialize:

    def test_creates_cpu_pool(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True) as mock_pool:
            pm.initialize(max_cpu=3, gpu_enabled=False)
            assert pm._cpu_pool is not None
            assert mock_pool.call_count == 1

    def test_creates_gpu_pool_when_enabled(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True) as mock_pool:
            pm.initialize(max_cpu=2, max_gpu=1, gpu_enabled=True)
            assert pm._gpu_pool is not None
            assert mock_pool.call_count == 2

    def test_no_gpu_pool_when_disabled(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True):
            pm.initialize(max_cpu=2, gpu_enabled=False)
            assert pm._gpu_pool is None

    def test_no_gpu_pool_when_size_zero(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True):
            pm.initialize(max_cpu=2, max_gpu=0, gpu_enabled=True)
            assert pm._gpu_pool is None


class TestShutdown:

    def test_cleans_up_pools(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True):
            pm.initialize(max_cpu=2, gpu_enabled=False)
            pm.shutdown()
            assert pm._cpu_pool is None
            assert pm._gpu_pool is None
            assert pm._initialized is False

    def test_signals_cancellation_for_active_jobs(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor", autospec=True):
            pm.initialize(max_cpu=2, gpu_enabled=False)
            mock_future = MagicMock()
            mock_future.done.return_value = True
            pm._active_futures["job-a"] = mock_future
            pm._active_futures["job-b"] = mock_future
            pm.shutdown()
            assert pm._cancel_events is None


class TestRouting:

    def test_gpu_models_routed_to_gpu_pool(self, pm):
        mock_cpu_exec = MagicMock()
        mock_gpu_exec = MagicMock()

        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_cls.side_effect = [mock_cpu_exec, mock_gpu_exec]
            pm.initialize(max_cpu=1, max_gpu=1, gpu_enabled=True)
            pm.submit("gpu-job", {"models": ["lstm"]})
            mock_gpu_exec.submit.assert_called_once()
            mock_cpu_exec.submit.assert_not_called()

    def test_cpu_models_routed_to_cpu_pool(self, pm):
        mock_cpu_exec = MagicMock()
        mock_gpu_exec = MagicMock()

        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_cls.side_effect = [mock_cpu_exec, mock_gpu_exec]
            pm.initialize(max_cpu=1, max_gpu=1, gpu_enabled=True)
            pm.submit("cpu-job", {"models": ["logistic"]})
            mock_cpu_exec.submit.assert_called_once()
            mock_gpu_exec.submit.assert_not_called()

    def test_gpu_models_fallback_to_cpu_when_gpu_disabled(self, pm):
        mock_cpu = MagicMock()

        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_cls.return_value = mock_cpu
            pm.initialize(max_cpu=1, gpu_enabled=False)
            pm.submit("gpu-fallback", {"models": ["lstm"]})
            mock_cpu.submit.assert_called_once()


class TestActiveState:

    def test_active_count(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_cls.return_value = MagicMock()
            pm.initialize(max_cpu=1, gpu_enabled=False)
            pm.submit("job-1", {"models": ["logistic"]})
            pm.submit("job-2", {"models": ["xgboost"]})
            assert pm.active_count == 2

    def test_active_job_ids(self, pm):
        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_cls.return_value = MagicMock()
            pm.initialize(max_cpu=1, gpu_enabled=False)
            pm.submit("job-a", {"models": ["logistic"]})
            ids = pm.active_job_ids
            assert "job-a" in ids
