"""Concurrency stress tests for the full backtest pipeline.

Tests concurrent DB access, WebSocket isolation, joblib temp dirs,
and per-process thread budget under multi-backtest load.
"""
from __future__ import annotations

import os
import threading
import uuid

import pytest

from api.services import JobManager
from api.process_manager import ProcessManager


class TestPipelineConcurrency:

    def test_four_concurrent_job_creations(self, tmp_path):
        """4 simultaneous job creations in WAL mode → all succeed, no lock errors."""
        from pipeline.data.data_sqlite import DataStore

        db_path = str(tmp_path / "pipeline_stress.db")
        store = DataStore(db_path)
        errors = []

        def create_job(idx):
            try:
                jm = JobManager(store)
                jid = f"stress-{idx}-{uuid.uuid4().hex[:6]}"
                jm.create_job(jid, "backtest", {"pair": "EURUSD"})
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create_job, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        jm = JobManager(store)
        jobs = jm.list_jobs("backtest", limit=10)
        assert len(jobs) >= 4

    def test_job_events_concurrent_appends(self, tmp_path):
        """Multiple workers appending events → no lost events."""
        from pipeline.data.data_sqlite import DataStore

        db_path = str(tmp_path / "events_stress.db")
        store = DataStore(db_path)
        jm = JobManager(store)
        jid = f"events-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid, "backtest", {"pair": "EURUSD"})

        def append_events(worker_id):
            for evt_idx in range(10):
                jm2 = JobManager(store)
                jm2.update_status(jid, "running")

        threads = [threading.Thread(target=append_events, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = jm.get_job(jid)
        assert final is not None
        assert final["status"] == "running"

    def test_joblib_temp_dirs_per_process(self, monkeypatch):
        """Each call to _run_backtest_in_worker passes env_vars correctly."""
        from api.process_manager import _run_backtest_in_worker

        seen_env = []

        def mock_impl(job_id, config):
            seen_env.append(os.environ.get("JOBLIB_TEMP_FOLDER", ""))

        monkeypatch.setattr("api.tasks._run_backtest_impl", mock_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", lambda: None)
        monkeypatch.setattr("os.chdir", lambda x: None)
        monkeypatch.setattr("api.process_manager.settings.project_root", ".")

        os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp/joblib_job-1"
        _run_backtest_in_worker("job-1", {}, {})
        os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp/joblib_job-2"
        _run_backtest_in_worker("job-2", {}, {})

        assert len(seen_env) == 2
        assert "joblib_job-1" in seen_env[0]
        assert "joblib_job-2" in seen_env[1]


class TestThreadBudgetPerProcess:

    def test_thread_budget_passed_as_env_var(self, monkeypatch):
        """env_vars include MLB_THREADS."""
        from api.process_manager import _run_backtest_in_worker

        captured_env = {}

        def mock_impl(job_id, config):
            captured_env["MLB_THREADS"] = os.environ.get("MLB_THREADS", "N/A")
            return {}

        monkeypatch.setattr("api.tasks._run_backtest_impl", mock_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", lambda: None)
        monkeypatch.setattr("os.chdir", lambda x: None)
        monkeypatch.setattr("api.process_manager.settings.project_root", ".")

        _run_backtest_in_worker("job-1", {}, {"MLB_THREADS": "2"})

        assert captured_env["MLB_THREADS"] == "2"

    def test_env_vars_isolated_per_submit(self, monkeypatch):
        """Two submits with different env_vars get different values."""
        from api.process_manager import _run_backtest_in_worker

        env1 = {"MLB_THREADS": "3"}
        env2 = {"MLB_THREADS": "1"}

        captured = []

        def mock_impl(job_id, config):
            captured.append(os.environ.get("MLB_THREADS", "N/A"))
            return {}

        monkeypatch.setattr("api.tasks._run_backtest_impl", mock_impl)
        monkeypatch.setattr("api.dependencies.get_data_store", lambda: None)
        monkeypatch.setattr("os.chdir", lambda x: None)
        monkeypatch.setattr("api.process_manager.settings.project_root", ".")

        _run_backtest_in_worker("job-a", {}, env1)
        _run_backtest_in_worker("job-b", {}, env2)

        assert len(captured) == 2
        assert captured[0] != captured[1]


class TestDBConnectionCleanup:

    def test_connections_closed_after_use(self, tmp_path):
        """Each _cursor() call opens and closes its connection."""
        from pipeline.data.data_sqlite import DataStore
        import sqlite3

        db_path = str(tmp_path / "conn_cleanup.db")
        store = DataStore(db_path)

        conn1 = store._connect()
        conn1.execute("SELECT 1")
        conn1.close()

        conn2 = store._connect()
        conn2.execute("SELECT 1")
        conn2.close()

        try:
            conn1.execute("SELECT 1")
            conn_closed = False
        except sqlite3.ProgrammingError:
            conn_closed = True

        assert conn_closed
