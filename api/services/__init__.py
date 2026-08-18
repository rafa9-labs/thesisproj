"""Job manager -- tracks backtest and download jobs in SQLite."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pipeline.data.data_sqlite import DataStore

_STALE_TIMEOUT_MINUTES = 90

# Serializes the check-and-insert sequence in create_job_atomic. Each
# connection reads its own SQLite snapshot, so without this lock the
# max_active limit can be violated by concurrent submissions.
_ATOMIC_CREATE_LOCK = threading.Lock()


class JobManager:
    def __init__(self, store: DataStore):
        self.store = store

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _stale_threshold(self) -> str:
        return (datetime.now(timezone.utc) - timedelta(minutes=_STALE_TIMEOUT_MINUTES)).isoformat()

    def _cleanup_stale_jobs(self) -> int:
        """Mark stale pending/running jobs as failed. Returns count cleaned.
        
        Before marking 'running' jobs as failed, checks Celery task state
        to avoid killing jobs whose worker is still active but hasn't
        updated updated_at recently (e.g. during long HPO runs).
        """
        threshold = self._stale_threshold()
        now = self._now()
        # Fetch stale jobs to inspect task state
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "SELECT id, status, task_id FROM jobs "
                "WHERE status IN ('pending', 'running', 'queued') AND updated_at < ?",
                (threshold,),
            )
            stale_rows = cur.fetchall()
        stale_ids = []
        for row in stale_rows:
            job_id, status, task_id = row
            if status == "running" and task_id:
                try:
                    from celery.result import AsyncResult
                    ar = AsyncResult(task_id)
                    if ar.state in ("STARTED", "RETRY"):
                        continue
                except Exception:
                    pass
            stale_ids.append(job_id)
        if not stale_ids:
            return 0
        count = 0
        with self.store._cursor() as (conn, cur):
            for jid in stale_ids:
                cur.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ? AND status IN ('pending', 'running', 'queued')",
                    ("failed", f"Job orphaned -- no update for {_STALE_TIMEOUT_MINUTES}+ min", now, jid),
                )
                count += cur.rowcount
            conn.commit()
        return count

    def create_job(self, job_id: str, job_type: str, config: Dict[str, Any]) -> Dict:
        """Backward-compatible wrapper that bypasses the concurrency limit."""
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "INSERT INTO jobs (id, type, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, job_type, "pending", json.dumps(config), now, now),
            )
        return {"id": job_id, "type": job_type, "status": "pending", "created_at": now}

    def create_job_atomic(self, job_id: str, job_type: str, config: Dict[str, Any], max_active: int = 1) -> Dict:
        now = self._now()
        with _ATOMIC_CREATE_LOCK:
            with self.store._cursor() as (conn, cur):
                self._cleanup_stale_jobs()
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE type = ? AND status IN ('pending', 'running', 'queued')",
                    (job_type,),
                )
                active_count = cur.fetchone()[0]
                if max_active > 0 and active_count >= max_active:
                    raise RuntimeError("Maximum concurrent backtest jobs reached")
                # Full-config dedup: only reject if an identical job is already active/queued
                config_json = json.dumps(config, sort_keys=True)
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE type = ? AND status IN ('pending', 'running', 'queued') "
                    "AND config = ?",
                    (job_type, config_json),
                )
                dup_count = cur.fetchone()[0]
                if dup_count > 0:
                    raise RuntimeError(
                        "An identical backtest is already running or queued. "
                        "Wait for it to complete or change a parameter before resubmitting."
                    )
                cur.execute(
                    "INSERT INTO jobs (id, type, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (job_id, job_type, "pending", json.dumps(config), now, now),
                )
        return {"id": job_id, "type": job_type, "status": "pending", "created_at": now}

    def get_active_jobs(self, job_type: Optional[str] = None) -> List[Dict]:
        self._cleanup_stale_jobs()
        sql = "SELECT * FROM jobs WHERE status IN ('pending', 'running', 'queued')"
        params: list = []
        if job_type:
            sql += " AND type = ?"
            params.append(job_type)
        sql += " ORDER BY created_at DESC"
        with self.store._cursor() as (conn, cur):
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            if r.get("config"):
                r["config"] = json.loads(r["config"])
            if r.get("result"):
                r["result"] = json.loads(r["result"])
            if r.get("study_meta"):
                r["study_meta"] = json.loads(r["study_meta"])
        return rows

    def force_stop_job(self, job_id: str) -> bool:
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ? AND status IN ('pending', 'running', 'queued')",
                ("failed", "Force stopped by user", now, job_id),
            )
            return cur.rowcount > 0

    def clear_pending_queue(self) -> int:
        """Mark all pending/running jobs as failed. Returns count updated."""
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE status IN ('pending', 'running', 'queued')",
                ("failed", "Queue cleared by cancellation or restart", now),
            )
            count = cur.rowcount
            conn.commit()
            return count

    def touch_job(self, job_id: str) -> bool:
        """Update updated_at to prevent stale-timeout kills during long execution."""
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            return cur.rowcount > 0

    def set_task_id(self, job_id: str, task_id: str) -> bool:
        """Store the Celery task ID for cross-worker revocation."""
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET task_id = ?, updated_at = ? WHERE id = ?",
                (task_id, now, job_id),
            )
            return cur.rowcount > 0

    def get_task_id(self, job_id: str) -> Optional[str]:
        """Retrieve the stored Celery task ID."""
        with self.store._cursor() as (conn, cur):
            cur.execute("SELECT task_id FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else None

    def ensure_job_exists(self, job_id: str, job_type: str = "backtest", config: Optional[Dict] = None) -> bool:
        """Insert a job row if it doesn't exist (e.g. after a race-condition delete).
        Returns True if inserted, False if already existed."""
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute("SELECT COUNT(1) FROM jobs WHERE id = ?", (job_id,))
            if cur.fetchone()[0] > 0:
                return False
            cur.execute(
                "INSERT INTO jobs (id, type, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, job_type, "running", json.dumps(config) if config else "{}", now, now),
            )
            return True

    def update_status(self, job_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None):
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(result) if result else None, error, now, job_id),
            )

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self.store._cursor() as (conn, cur):
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            d = dict(zip(cols, row))
            if d.get("config"):
                d["config"] = json.loads(d["config"])
            if d.get("result"):
                d["result"] = json.loads(d["result"])
            if d.get("study_meta"):
                d["study_meta"] = json.loads(d["study_meta"])
            return d

    def list_jobs(self, job_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        sql = "SELECT * FROM jobs"
        params: list = []
        if job_type:
            sql += " WHERE type = ?"
            params.append(job_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.store._cursor() as (conn, cur):
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            if r.get("config"):
                r["config"] = json.loads(r["config"])
            if r.get("result"):
                r["result"] = json.loads(r["result"])
            if r.get("study_meta"):
                r["study_meta"] = json.loads(r["study_meta"])
        return rows

    def list_jobs_paginated(self, job_type: Optional[str] = None, limit: int = 50, offset: int = 0) -> tuple[List[Dict], int]:
        count_sql = "SELECT COUNT(*) FROM jobs"
        params_count: list = []
        if job_type:
            count_sql += " WHERE type = ?"
            params_count.append(job_type)
        with self.store._cursor() as (conn, cur):
            cur.execute(count_sql, params_count)
            total = cur.fetchone()[0]

        sql = "SELECT * FROM jobs"
        params: list = []
        if job_type:
            sql += " WHERE type = ?"
            params.append(job_type)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.store._cursor() as (conn, cur):
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            if r.get("config"):
                r["config"] = json.loads(r["config"])
            if r.get("result"):
                r["result"] = json.loads(r["result"])
            if r.get("study_meta"):
                r["study_meta"] = json.loads(r["study_meta"])
        return rows, total

    def delete_job(self, job_id: str) -> bool:
        with self.store._cursor() as (conn, cur):
            cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def update_study_meta(self, job_id: str, meta: dict) -> bool:
        now = self._now()
        meta["saved_at"] = now
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET study_meta = ?, updated_at = ? WHERE id = ?",
                (json.dumps(meta), now, job_id),
            )
            return cur.rowcount > 0

    def get_study_meta(self, job_id: str) -> Optional[dict]:
        with self.store._cursor() as (conn, cur):
            cur.execute("SELECT study_meta FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return None
