"""
Cleanup script — kill stale processes, reset DB jobs, flush Redis.

Usage:
    python scripts/cleanup.py
"""
import os
import sqlite3
import subprocess
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)


def kill_workers():
    """Kill running Celery workers and their child Python processes."""
    print("[cleanup] Killing Celery + child Python processes...")
    subprocess.run(
        ["taskkill", "/F", "/IM", "celery.exe"],
        capture_output=True,
    )
    subprocess.run(
        ["powershell", "-Command",
         "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -eq '' } | Where-Object { (Get-Process -Id $_.Id).Path -like '*celery*' } | Stop-Process -Force"],
        capture_output=True,
    )
    print("  Done.")


def reset_db():
    """Mark all running/pending jobs as failed."""
    db_path = os.path.join(_project_root, "data", "forex.db")
    if not os.path.exists(db_path):
        print(f"[cleanup] DB not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE jobs SET status='failed', error='Worker terminated (cleanup)' WHERE status IN ('running', 'pending')"
    )
    count = cur.rowcount
    conn.commit()
    conn.close()
    print(f"[cleanup] Marked {count} stale DB jobs as failed.")


def flush_redis():
    """Flush all Redis keys/queues."""
    try:
        import redis
        r = redis.from_url("redis://localhost:6379/0", socket_connect_timeout=2)
        r.ping()
        r.flushall()
        print("[cleanup] Redis flushed.")
    except Exception as e:
        print(f"[cleanup] Redis not available (ok): {e}")


if __name__ == "__main__":
    print("=== KodaQuant Cleanup ===\n")
    kill_workers()
    reset_db()
    flush_redis()
    print("\nDone. Start fresh:\n  celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery\n  uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload\n  cd frontend; npm run dev")
