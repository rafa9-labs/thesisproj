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
    """Mark all pending/running jobs as failed on server restart.

    Returns number of jobs updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed', error = 'Server restarted before completion', updated_at = ? WHERE status IN ('pending', 'running')",
            (now,),
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def mark_running_jobs_interrupted(db_path: str) -> int:
    """Mark running jobs as failed because the server is shutting down.

    Returns number of jobs updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed', error = 'Server shutdown interrupted backtest', updated_at = ? WHERE status = 'running'",
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
