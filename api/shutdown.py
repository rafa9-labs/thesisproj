"""Graceful shutdown handlers for KodaQuant backend.

Called from FastAPI lifespan shutdown and/or Electron cleanup.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def wal_checkpoint(db_path: str) -> None:
    """Force WAL checkpoint to compact the write-ahead log into the main DB."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception:
        pass


def wal_checkpoint_periodic(db_path: str) -> None:
    """Lightweight checkpoint (PASSIVE) suitable for periodic calls."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()
    except Exception:
        pass


def mark_stale_jobs_failed(db_path: str) -> int:
    """On server restart, transition stale jobs.

    - 'running' jobs with an active Celery task are LEFT as 'running'
      (the worker survived the restart, or a different worker picked it up).
    - 'running' jobs where the Celery task is dead/superseded are reset
      to 'pending' so they can be re-queued.
    - 'pending' jobs stay 'pending' -- they were never picked up.

    Returns number of jobs updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    total = 0
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, task_id FROM jobs WHERE status = 'running'"
        )
        running = cur.fetchall()
        for row in running:
            job_id = row["id"]
            task_id = row["task_id"]
            reset = True
            if task_id:
                try:
                    from celery.result import AsyncResult
                    ar = AsyncResult(task_id)
                    if ar.state in ("STARTED", "RETRY", "PENDING"):
                        reset = False
                except Exception:
                    pass
            if reset:
                conn.execute(
                    "UPDATE jobs SET status = 'pending', "
                    "error = 'Server restarted while running -- reset to pending', "
                    "updated_at = ? WHERE id = ?",
                    (now, job_id),
                )
                total += 1
        conn.commit()
        conn.close()
        return total
    except Exception:
        return total


def mark_running_jobs_interrupted(db_path: str) -> int:
    """On shutdown, mark running jobs as interrupted.

    These will be reset to 'pending' on next startup so Celery can re-pick
    them up. Returns number of jobs updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.execute(
            "UPDATE jobs SET error = 'Server shutdown interrupted backtest', updated_at = ? WHERE status = 'running'",
            (now,),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def shutdown_deep_pool() -> None:
    """Shut down the DeepMixin ProcessPoolExecutor if it exists."""
    try:
        from pipeline.backtester.deep_mixin import DeepMixin
        DeepMixin._shutdown_deep_pool()
    except Exception:
        pass


def startup_cleanup(db_path: str) -> int:
    """Run all startup cleanup: prune stale jobs, reap orphaned processes."""
    from api.process_cleanup import cleanup_all, reap_orphaned_processes
    reap_orphaned_processes()
    cleanup_all()
    return mark_stale_jobs_failed(db_path)


def shutdown_cleanup(db_path: str) -> None:
    """Run all shutdown cleanup: kill pools, checkpoint WAL, mark interrupted jobs."""
    from api.process_cleanup import cleanup_all
    cleanup_all()
    shutdown_deep_pool()
    mark_running_jobs_interrupted(db_path)
    wal_checkpoint(db_path)
