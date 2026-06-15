"""Job manager -- tracks backtest and download jobs in SQLite."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pipeline.data_sqlite import DataStore


class JobManager:
    def __init__(self, store: DataStore):
        self.store = store

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_job(self, job_id: str, job_type: str, config: Dict[str, Any]) -> Dict:
        """Backward-compatible wrapper that bypasses the concurrency limit."""
        now = self._now()
        with self.store._write_cursor() as (conn, cur):
            cur.execute(
                "INSERT INTO jobs (id, type, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, job_type, "pending", json.dumps(config), now, now),
            )
        return {"id": job_id, "type": job_type, "status": "pending", "created_at": now}

    def create_job_atomic(self, job_id: str, job_type: str, config: Dict[str, Any], max_active: int = 1) -> Dict:
        now = self._now()
        with self.store._write_cursor() as (conn, cur):
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE type = ? AND status IN ('pending', 'running')",
                (job_type,),
            )
            active_count = cur.fetchone()[0]
            if active_count >= max_active:
                raise RuntimeError("Maximum concurrent backtest jobs reached")
            cur.execute(
                "INSERT INTO jobs (id, type, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, job_type, "pending", json.dumps(config), now, now),
            )
        return {"id": job_id, "type": job_type, "status": "pending", "created_at": now}

    def get_active_jobs(self, job_type: Optional[str] = None) -> List[Dict]:
        sql = "SELECT * FROM jobs WHERE status IN ('pending', 'running')"
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
        with self.store._write_cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ? AND status IN ('pending', 'running')",
                ("failed", "Force stopped by user", now, job_id),
            )
            return cur.rowcount > 0

    def clear_pending_queue(self) -> int:
        """Mark all pending/running jobs as failed. Returns count updated."""
        now = self._now()
        with self.store._write_cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE status IN ('pending', 'running')",
                ("failed", "Queue cleared by cancellation or restart", now),
            )
            count = cur.rowcount
            conn.commit()
            return count

    def update_status(self, job_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None):
        now = self._now()
        with self.store._write_cursor() as (conn, cur):
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
        with self.store._write_cursor() as (conn, cur):
            cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def update_study_meta(self, job_id: str, meta: dict) -> bool:
        now = self._now()
        meta["saved_at"] = now
        with self.store._write_cursor() as (conn, cur):
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
