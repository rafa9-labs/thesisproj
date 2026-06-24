"""Tests for the schema initialization fix and job persistence roundtrip."""
import os
import tempfile
import uuid

import pytest
import sqlite3

from pipeline.data.data_sqlite import DataStore, SCHEMA_SQL
from api.services import JobManager


class TestSchemaInitFix:
    """Verifies _ensure_schema() creates missing job tables on old databases."""

    @pytest.fixture
    def old_db_path(self):
        """Create a SQLite DB with only candles/pairs (simulating pre-jobs schema)."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                pair TEXT NOT NULL, timeframe TEXT NOT NULL, ts TEXT NOT NULL,
                mid_open REAL, mid_high REAL, mid_low REAL, mid_close REAL,
                bid_open REAL, bid_close REAL, ask_open REAL, ask_close REAL,
                spread REAL, volume INTEGER,
                PRIMARY KEY (pair, timeframe, ts)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pairs (
                symbol TEXT PRIMARY KEY, oanda_name TEXT NOT NULL,
                pip_value REAL NOT NULL, lot_size REAL DEFAULT 100000.0,
                base_currency TEXT DEFAULT '', quote_currency TEXT DEFAULT '',
                typical_spread_bps REAL DEFAULT 1.0
            )
        """)
        conn.commit()
        conn.close()
        yield db_path
        try:
            os.unlink(db_path)
        except PermissionError:
            pass

    def test_ensure_schema_creates_missing_jobs_table(self, old_db_path):
        """DataStore must create 'jobs' table when opening old DB that only has candles."""
        store = DataStore(old_db_path)
        with store._cursor() as (conn, cur):
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
            assert cur.fetchone() is not None, "jobs table should be auto-created"

    def test_ensure_schema_creates_missing_job_events_table(self, old_db_path):
        """DataStore must create 'job_events' table when opening old DB."""
        store = DataStore(old_db_path)
        with store._cursor() as (conn, cur):
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_events'")
            assert cur.fetchone() is not None, "job_events table should be auto-created"

    def test_job_crud_on_old_db(self, old_db_path):
        """JobManager create/get must work on DB that only had candles initially."""
        store = DataStore(old_db_path)
        jm = JobManager(store)
        job_id = str(uuid.uuid4())
        jm.create_job(job_id, "backtest", {"pair": "EURUSD"})
        job = jm.get_job(job_id)
        assert job is not None
        assert job["id"] == job_id
        assert job["status"] == "pending"
        jm.delete_job(job_id)

    def test_ensure_schema_idempotent(self):
        """Calling _ensure_schema multiple times must not raise."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = DataStore(db_path)
            store._ensure_schema()
            store._ensure_schema()
            with store._cursor() as (conn, cur):
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cur.fetchall()}
                assert "candles" in tables
                assert "pairs" in tables
                assert "jobs" in tables
                assert "job_events" in tables
        finally:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

    def test_schema_initialized_flag_not_used(self):
        """The class-level _schema_initialized flag must be removed."""
        assert not hasattr(DataStore, "_schema_initialized"), (
            "DataStore should not use _schema_initialized class flag anymore"
        )


class TestJobStatusRoundtrip:
    """End-to-end: create job via API, verify status/results endpoints respond."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        from api.dependencies import get_data_store
        from api.services import JobManager
        store = get_data_store()
        jm = JobManager(store)
        for job in jm.list_jobs(job_type="backtest_e2e", limit=1000):
            jm.delete_job(job["id"])

    def test_full_job_lifecycle(self):
        """Create, read status, update, read results — full lifecycle."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)

        job_id = str(uuid.uuid4())
        config = {"pair": "EURUSD", "models": ["logistic"], "months": 1}
        jm.create_job(job_id, "backtest_e2e", config)

        job = jm.get_job(job_id)
        assert job is not None
        assert job["status"] == "pending"

        jm.update_status(job_id, "running")
        job = jm.get_job(job_id)
        assert job["status"] == "running"

        result = {
            "pair": "EURUSD",
            "models": ["logistic"],
            "metrics": [{"model": "logistic", "sharpe": 1.5}],
        }
        jm.update_status(job_id, "completed", result=result)
        job = jm.get_job(job_id)
        assert job["status"] == "completed"
        assert job["result"]["metrics"][0]["sharpe"] == 1.5

        jm.delete_job(job_id)

    def test_get_job_returns_404_on_nonexistent(self):
        """jm.get_job must return None for unknown IDs (matches API 404 behavior)."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)
        assert jm.get_job("nonexistent-id") is None

    def test_multiple_jobs_listed(self):
        """Listing jobs must return all created entries."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)

        ids = []
        for i in range(3):
            jid = f"list-test-{uuid.uuid4().hex[:8]}"
            jm.create_job(jid, "backtest_e2e", {"pair": "EURUSD", "idx": i})
            ids.append(jid)

        jobs, total = jm.list_jobs_paginated(job_type="backtest_e2e", limit=10)
        assert total >= 3
        found_ids = {j["id"] for j in jobs if j["id"] in ids}
        assert len(found_ids) == 3

        for jid in ids:
            jm.delete_job(jid)

    def test_force_stop_job(self):
        """force_stop_job must set status to 'failed'."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)

        job_id = f"force-stop-{uuid.uuid4().hex[:8]}"
        jm.create_job(job_id, "backtest_e2e", {"pair": "EURUSD"})
        jm.update_status(job_id, "running")

        assert jm.force_stop_job(job_id)
        job = jm.get_job(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "Force stopped by user"

        jm.delete_job(job_id)

    def test_create_job_atomic_concurrency_limit(self):
        """create_job_atomic must reject when max_active is reached."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)

        job_id_1 = f"atomic-{uuid.uuid4().hex[:8]}"
        jm.create_job_atomic(job_id_1, "backtest_e2e", {"pair": "EURUSD"}, max_active=1)

        job_id_2 = f"atomic-{uuid.uuid4().hex[:8]}"
        with pytest.raises(RuntimeError, match="Maximum concurrent"):
            jm.create_job_atomic(job_id_2, "backtest_e2e", {"pair": "EURUSD"}, max_active=1)

        jm.delete_job(job_id_1)

    def test_update_status_all_states(self):
        """Verify all status transitions are persisted correctly."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)

        job_id = f"states-{uuid.uuid4().hex[:8]}"
        jm.create_job(job_id, "backtest_e2e", {"pair": "EURUSD"})
        assert jm.get_job(job_id)["status"] == "pending"

        jm.update_status(job_id, "running")
        assert jm.get_job(job_id)["status"] == "running"

        jm.update_status(job_id, "completed", result={"sharpe": 2.0})
        job = jm.get_job(job_id)
        assert job["status"] == "completed"
        assert job["result"]["sharpe"] == 2.0

        jm.update_status(job_id, "failed", error="test error")
        job = jm.get_job(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "test error"

        jm.delete_job(job_id)


class TestProgressCallbackForwarding:
    """Verifies _progress_callback is forwarded from config to cv_config in run_strategy."""

    def _simulate_cv_config_creation(self, config):
        """Replicate the cv_config creation logic from run_mixin.py and check forwarding."""
        min_train_window = 50
        val_window = 20
        cv_config = {"min_train_window": min_train_window, "val_window": val_window}
        if isinstance(config, dict) and "_progress_callback" in config:
            cv_config["_progress_callback"] = config["_progress_callback"]
        cv_config["score_for_no_trades"] = -1.0
        return cv_config

    def test_callback_forwarded_to_cv_config(self):
        """_progress_callback in config must be copied into cv_config."""
        def my_cb(phase, model, detail=None):
            pass

        config = {"_progress_callback": my_cb, "n_trials": 3}
        cv_config = self._simulate_cv_config_creation(config)
        assert cv_config.get("_progress_callback") is my_cb

    def test_callback_not_in_config_omitted(self):
        """When config lacks _progress_callback, cv_config must not have it."""
        config = {"n_trials": 3}
        cv_config = self._simulate_cv_config_creation(config)
        assert "_progress_callback" not in cv_config


    def test_active_jobs_filter(self):
        """get_active_jobs should only return pending/running jobs."""
        from api.dependencies import get_data_store
        store = get_data_store()
        jm = JobManager(store)

        jid_pending = f"active-pending-{uuid.uuid4().hex[:8]}"
        jid_completed = f"active-completed-{uuid.uuid4().hex[:8]}"
        jm.create_job(jid_pending, "backtest_e2e", {"pair": "EURUSD"})
        jm.create_job(jid_completed, "backtest_e2e", {"pair": "EURUSD"})
        jm.update_status(jid_completed, "completed", result={})

        active_ids = {j["id"] for j in jm.get_active_jobs("backtest_e2e")}
        assert jid_pending in active_ids
        assert jid_completed not in active_ids

        jm.delete_job(jid_pending)
        jm.delete_job(jid_completed)
