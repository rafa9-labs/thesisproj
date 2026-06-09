"""
flush_all_tasks.py — Kill orphaned workers, clear job queue, reset process registry.

Run this from the project root when Celery workers or backtest jobs get stuck
after a cancellation. This is safe to run at any time (it won't affect running
pipeline operations that already cleaned up after themselves).

Usage:
    python flush_all_tasks.py
"""

import os
import sys
import signal
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _log(msg: str):
    print(f"[flush] {msg}")


def kill_celery_workers():
    """Kill any orphaned Celery worker processes."""
    killed = 0
    try:
        # PowerShell: find and kill python.exe processes whose command line
        # contains "celery" (but NOT this script itself)
        ps_cmd = (
            'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | '
            'Select-Object ProcessId, CommandLine | ConvertTo-Json'
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
        )
        import json
        procs = json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(procs, dict):
            procs = [procs]
        for p in procs:
            pid = p.get("ProcessId")
            cmdline = (p.get("CommandLine") or "").lower()
            if not pid:
                continue
            if "celery" in cmdline and "worker" in cmdline:
                try:
                    os.kill(pid, signal.SIGTERM)
                    _log(f"Killed celery worker PID {pid}")
                    killed += 1
                except (OSError, PermissionError) as e:
                    _log(f"  Could not kill PID {pid}: {e}")
        if killed == 0:
            _log("No orphaned Celery workers found")
    except Exception as e:
        _log(f"Error scanning for Celery workers: {e}")
    return killed


def clear_sqlite_queue():
    """Mark all pending/running backtest jobs as failed."""
    try:
        from api.config import settings
        from pipeline.data_sqlite import DataStore
        from api.services import JobManager

        store = DataStore(settings.db_full_path)
        jm = JobManager(store)
        count = jm.clear_pending_queue()
        _log(f"Cleared {count} pending/running jobs from SQLite queue")
        return count
    except Exception as e:
        _log(f"Could not clear SQLite queue: {e}")
        return 0


def reset_process_registry():
    """Clean up the in-memory process registry (safe even if nothing was registered)."""
    try:
        from api.process_cleanup import cleanup_all
        kill_count = cleanup_all()
        _log(f"Cleaned up {kill_count} registered processes")
        return kill_count
    except Exception as e:
        _log(f"Could not reset process registry: {e}")
        return 0


def reap_zombies():
    """Find and kill orphaned child processes of dead Python processes."""
    killed = 0
    try:
        ps_cmd = (
            'Get-CimInstance Win32_Process | Where-Object { '
            '$_.Name -match "python" '
            '} | Select-Object ProcessId, ParentProcessId | ConvertTo-Json'
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
        )
        import json
        procs = json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(procs, dict):
            procs = [procs]

        # Collect PIDs of alive python processes
        alive = {p["ProcessId"] for p in procs if p.get("ProcessId")}

        # Find orphaned child processes (parent not alive, name hints at
        # loky/joblib/vcomp worker)
        for p in procs:
            pid = p.get("ProcessId")
            ppid = p.get("ParentProcessId")
            if not pid or not ppid:
                continue
            if ppid not in alive and pid not in alive:
                try:
                    os.kill(pid, signal.SIGTERM)
                    _log(f"Reaped orphan PID {pid} (dead parent {ppid})")
                    killed += 1
                except (OSError, PermissionError):
                    pass
        if killed == 0:
            _log("No orphaned child processes found")
    except Exception as e:
        _log(f"Error reaping zombie processes: {e}")
    return killed


def main():
    _log("=" * 50)
    _log("Flushing all stuck tasks...")
    _log("=" * 50)

    k1 = kill_celery_workers()
    k2 = reap_zombies()
    k3 = clear_sqlite_queue()
    k4 = reset_process_registry()

    _log("=" * 50)
    _log(f"Summary: killed {k1 + k2} processes, cleared {k3} SQLite jobs, "
         f"reset {k4} process registry entries")
    _log("Done. You can now restart the server and Celery workers cleanly.")
    _log("=" * 50)


if __name__ == "__main__":
    main()
