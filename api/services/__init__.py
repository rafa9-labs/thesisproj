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
        now = self._now()
        with self.store._cursor() as (conn, cur):
            cur.execute(
                "INSERT INTO jobs (id, type, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, job_type, "pending", json.dumps(config), now, now),
            )
        return {"id": job_id, "type": job_type, "status": "pending", "created_at": now}

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
        return rows

    def delete_job(self, job_id: str) -> bool:
        with self.store._cursor() as (conn, cur):
            cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0
