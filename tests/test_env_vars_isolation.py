"""Integration tests for per-process environment variable isolation.

Tests that env_vars are applied in the worker WITHOUT mutating parent os.environ.
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from api.process_manager import _run_backtest_in_worker


def _mock_backtest_impl(job_id, config):
    return {"status": "ok"}


class TestEnvVarsAppliedInWorker:

    def test_mlb_threads_applied(self, monkeypatch):
        monkeypatch.setattr("api.tasks._run_backtest_impl", _mock_backtest_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", MagicMock())
        monkeypatch.setattr("os.chdir", MagicMock())

        _run_backtest_in_worker("test-job", {}, {"MLB_THREADS": "2"})
        assert os.environ.get("MLB_THREADS") == "2"

    def test_cuda_vram_limit_applied(self, monkeypatch):
        monkeypatch.setattr("api.tasks._run_backtest_impl", _mock_backtest_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", MagicMock())
        monkeypatch.setattr("os.chdir", MagicMock())

        _run_backtest_in_worker("test-job", {}, {"CUDA_VRAM_LIMIT_MB": "4096"})
        assert os.environ.get("CUDA_VRAM_LIMIT_MB") == "4096"

    def test_multiple_env_vars_applied(self, monkeypatch):
        monkeypatch.setattr("api.tasks._run_backtest_impl", _mock_backtest_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", MagicMock())
        monkeypatch.setattr("os.chdir", MagicMock())

        _run_backtest_in_worker("test-job", {}, {
            "MLB_THREADS": "2",
            "CUDA_VRAM_LIMIT_MB": "4096",
            "BLAS_THREADS_PER_TRIAL": "2",
        })
        assert os.environ.get("MLB_THREADS") == "2"
        assert os.environ.get("CUDA_VRAM_LIMIT_MB") == "4096"
        assert os.environ.get("BLAS_THREADS_PER_TRIAL") == "2"

    def test_empty_env_vars_noop(self, monkeypatch):
        monkeypatch.setattr("api.tasks._run_backtest_impl", _mock_backtest_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", MagicMock())
        monkeypatch.setattr("os.chdir", MagicMock())

        try:
            _run_backtest_in_worker("test-job", {}, {})
        except Exception:
            pytest.fail("empty env_vars should not crash")


class TestParentEnvNotMutated:

    def test_parent_env_unchanged_by_submit(self):
        from api.process_manager import ProcessManager

        pm = ProcessManager()
        env_before = {k: os.environ.get(k) for k in ["MLB_THREADS", "CUDA_VRAM_LIMIT_MB"]}

        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_pool = MagicMock()
            mock_pool._max_workers = 1
            mock_cls.return_value = mock_pool
            pm.initialize(max_cpu=1, gpu_enabled=False)
            pm.submit_or_queue("test-id", {"models": ["logistic"]},
                               env_vars={"MLB_THREADS": "4"})

        for k in env_before:
            assert os.environ.get(k) == env_before[k]

    def test_no_cuda_vram_in_parent_after_gpu_submit(self):
        from api.process_manager import ProcessManager

        pm = ProcessManager()
        parent_has_cuda = "CUDA_VRAM_LIMIT_MB" in os.environ

        with patch("api.process_manager.ProcessPoolExecutor") as mock_cls:
            mock_pool = MagicMock()
            mock_pool._max_workers = 1
            mock_cls.return_value = mock_pool
            pm.initialize(max_cpu=1, gpu_enabled=False)
            pm.submit_or_queue("test-gpu", {"models": ["lstm"]},
                               env_vars={"CUDA_VRAM_LIMIT_MB": "4096"})

        assert ("CUDA_VRAM_LIMIT_MB" in os.environ) == parent_has_cuda
