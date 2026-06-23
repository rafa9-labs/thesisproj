"""Integration tests for SQLite WAL-mode concurrency safety.

Tests concurrent reads, writes, and atomic job creation under multi-threaded load.
"""
from __future__ import annotations

import uuid
import threading

import pytest

from api.services import JobManager
from pipeline.data_sqlite import DataStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "concurrent.db")
    ds = DataStore(db_path)
    yield ds


@pytest.fixture
def jm(store):
    return JobManager(store)


class TestWALConcurrentWrites:

    def test_concurrent_writers_no_errors(self, store):
        rows_per_thread = 10
        num_threads = 5

        def insert_rows(thread_id):
            jm = JobManager(store)
            for i in range(rows_per_thread):
                jid = f"concurrent-{thread_id}-{i}-{uuid.uuid4().hex[:6]}"
                jm.create_job(jid, "backtest", {"pair": "EURUSD"})

        threads = [threading.Thread(target=insert_rows, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        jm = JobManager(store)
        all_jobs = jm.list_jobs("backtest", limit=100)
        assert len(all_jobs) == num_threads * rows_per_thread

    def test_concurrent_readers_during_write(self, store):
        write_done = threading.Event()
        results = {"reads_ok": 0, "reader_errors": 0}

        def writer():
            jm = JobManager(store)
            for i in range(500):
                jm.create_job(f"wr-{i}-{uuid.uuid4().hex[:6]}", "backtest", {"pair": "EURUSD"})
            write_done.set()

        def reader():
            jm = JobManager(store)
            try:
                while not write_done.is_set():
                    jm.list_jobs("backtest", limit=10)
                    results["reads_ok"] += 1
            except Exception:
                results["reader_errors"] += 1

        writers = [threading.Thread(target=writer)]
        readers = [threading.Thread(target=reader) for _ in range(3)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert results["reader_errors"] == 0
        assert results["reads_ok"] > 0

    def test_concurrent_create_job_atomic_race(self, store):
        max_active = 2
        successes = []
        failures = []

        def try_create(idx):
            jm = JobManager(store)
            try:
                jm.create_job_atomic(
                    f"race-{idx}-{uuid.uuid4().hex[:6]}",
                    "backtest_race2",
                    {"pair": "EURUSD"},
                    max_active=max_active,
                )
                successes.append(1)
            except RuntimeError:
                failures.append(1)

        threads = [threading.Thread(target=try_create, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) >= max_active
        assert len(failures) > 0


class TestWALPragmaSettings:

    def test_wal_mode_verified(self, store):
        conn = store._connect()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

    def test_synchronous_normal(self, store):
        conn = store._connect()
        cur = conn.cursor()
        cur.execute("PRAGMA synchronous")
        val = cur.fetchone()[0]
        conn.close()
        assert val == 1

    def test_cache_size_set(self, store):
        conn = store._connect()
        cur = conn.cursor()
        cur.execute("PRAGMA cache_size")
        val = cur.fetchone()[0]
        conn.close()
        assert val == -64000


class TestCursorTransactionSafety:

    def test_commit_persists_row(self, store):
        jm = JobManager(store)
        jid = f"commit-persist-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid, "backtest", {"pair": "EURUSD"})
        assert jm.get_job(jid) is not None

    def test_rollback_does_not_persist(self, store):
        jm = JobManager(store)
        jid = f"rollback-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid, "backtest", {"pair": "EURUSD"})

        try:
            with store._cursor() as (conn, cur):
                cur.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
                raise ValueError("simulated mid-transaction failure")
        except ValueError:
            pass

        job = jm.get_job(jid)
        assert job is not None
        assert job["status"] == "pending"


class TestStaleCleanup:

    def test_stale_jobs_excluded_from_active_count(self, store, jm):
        jid_stale = f"stale-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid_stale, "backtest", {"pair": "EURUSD"})

        with store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET updated_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
                (jid_stale,),
            )

        jid_new = f"fresh-{uuid.uuid4().hex[:6]}"
        jm.create_job_atomic(jid_new, "backtest_stale", {"pair": "EURUSD"}, max_active=1)

        stale_job = jm.get_job(jid_stale)
        assert stale_job["status"] == "failed"
        assert "orphaned" in stale_job.get("error", "").lower()

    def test_stale_cleanup_transactional(self, store, jm):
        jid_s = f"stale-tx-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid_s, "backtest_tx", {"pair": "EURUSD"})

        with store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET updated_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
                (jid_s,),
            )

        cleaned = jm._cleanup_stale_jobs()
        assert cleaned >= 1

        active = jm.get_active_jobs("backtest_tx")
        assert len(active) == 0
