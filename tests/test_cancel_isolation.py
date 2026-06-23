"""Concurrency tests for scoped cancellation isolation.

Tests that cancelling one job does not kill other concurrent jobs.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from api.services import JobManager
from api.process_manager import ProcessManager


class TestCancelIsolation:

    def test_force_stop_only_target_job(self, tmp_path):
        from pipeline.data_sqlite import DataStore

        db_path = str(tmp_path / "cancel_iso.db")
        store = DataStore(db_path)
        jm = JobManager(store)

        job_a = f"cancel-a-{uuid.uuid4().hex[:6]}"
        job_b = f"cancel-b-{uuid.uuid4().hex[:6]}"
        jm.create_job(job_a, "backtest", {"pair": "EURUSD"})
        jm.create_job(job_b, "backtest", {"pair": "EURUSD"})
        jm.update_status(job_a, "running")
        jm.update_status(job_b, "running")

        jm.force_stop_job(job_a)

        job_a_data = jm.get_job(job_a)
        job_b_data = jm.get_job(job_b)
        assert job_a_data["status"] == "failed"
        assert job_b_data["status"] == "running"

    def test_force_stop_nonexistent_no_side_effects(self, tmp_path):
        from pipeline.data_sqlite import DataStore

        db_path = str(tmp_path / "cancel_sidefx.db")
        store = DataStore(db_path)
        jm = JobManager(store)

        job_b = f"sidefx-b-{uuid.uuid4().hex[:6]}"
        jm.create_job(job_b, "backtest", {"pair": "EURUSD"})
        jm.update_status(job_b, "running")

        result = jm.force_stop_job(f"bogus-{uuid.uuid4().hex[:6]}")
        assert result is False

        job_b_data = jm.get_job(job_b)
        assert job_b_data["status"] == "running"

    def test_process_manager_cancel_scoped(self, monkeypatch):
        pm = ProcessManager()
        monkeypatch.setattr(pm, "_cancel_events", {"job-a": False, "job-b": False})
        pm._initialized = True

        pm.request_cancellation("job-a")
        assert pm._cancel_events["job-a"] is True
        assert pm._cancel_events["job-b"] is False

    def test_cancel_events_not_cross_contaminated(self, monkeypatch):
        pm = ProcessManager()
        monkeypatch.setattr(pm, "_cancel_events", {"job-a": False, "job-b": False, "job-c": False})
        pm._initialized = True

        pm.request_cancellation("job-a")
        pm.request_cancellation("job-c")

        assert pm._cancel_events["job-a"] is True
        assert pm._cancel_events["job-b"] is False
        assert pm._cancel_events["job-c"] is True

    def test_done_callback_only_cleans_own_job(self, monkeypatch):
        pm = ProcessManager()
        pm._initialized = True
        pm._cancel_events = {"job-a": False, "job-b": False}
        with patch("api.process_manager.ProcessPoolExecutor") as mock_pool:
            mock_pool.return_value = MagicMock()
            pm._cpu_pool = MagicMock()
            pm.submit("job-a", {"models": ["logistic"]})
            pm.submit("job-b", {"models": ["xgboost"]})

            future_a = pm._active_futures.get("job-a")
            if future_a:
                future_a.set_result(MagicMock())

            assert "job-b" in pm._active_futures

    def test_force_stop_only_active_affected(self, tmp_path):
        from pipeline.data_sqlite import DataStore

        db_path = str(tmp_path / "only_active.db")
        store = DataStore(db_path)
        jm = JobManager(store)

        job_pending = f"pending-{uuid.uuid4().hex[:6]}"
        job_running = f"running-{uuid.uuid4().hex[:6]}"
        job_completed = f"completed-{uuid.uuid4().hex[:6]}"

        jm.create_job(job_pending, "backtest", {"pair": "EURUSD"})
        jm.create_job(job_running, "backtest", {"pair": "EURUSD"})
        jm.create_job(job_completed, "backtest", {"pair": "EURUSD"})
        jm.update_status(job_running, "running")
        jm.update_status(job_completed, "completed")

        jm.force_stop_job(job_running)
        jm.force_stop_job(job_pending)

        assert jm.get_job(job_pending)["status"] == "failed"
        assert jm.get_job(job_running)["status"] == "failed"
        assert jm.get_job(job_completed)["status"] == "completed"

    def test_simultaneous_force_stop_two_jobs(self, tmp_path):
        from pipeline.data_sqlite import DataStore
        import threading

        db_path = str(tmp_path / "dual_stop.db")
        store = DataStore(db_path)
        jm = JobManager(store)

        job_a = f"dual-a-{uuid.uuid4().hex[:6]}"
        job_b = f"dual-b-{uuid.uuid4().hex[:6]}"
        jm.create_job(job_a, "backtest", {"pair": "EURUSD"})
        jm.create_job(job_b, "backtest", {"pair": "EURUSD"})
        jm.update_status(job_a, "running")
        jm.update_status(job_b, "running")

        def stop_a():
            jm.force_stop_job(job_a)
        def stop_b():
            jm.force_stop_job(job_b)

        ta = threading.Thread(target=stop_a)
        tb = threading.Thread(target=stop_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert jm.get_job(job_a)["status"] == "failed"
        assert jm.get_job(job_b)["status"] == "failed"
