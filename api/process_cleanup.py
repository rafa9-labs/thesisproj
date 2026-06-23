"""Cross-platform process registry and cleanup for KodaQuant.

Tracks child processes spawned during backtest / full-cycle jobs, enables
graceful termination on cancellation, and reaps orphaned processes on startup.
"""
from __future__ import annotations

import ctypes
import threading
import time
from typing import Any, Dict, List, Optional

import psutil

_job_processes: Dict[str, List[int]] = {}
_bt_refs: Dict[str, Any] = {}
_deep_pools: Dict[str, Any] = {}
_job_threads: Dict[str, threading.Thread] = {}
_job_cancellation_events: Dict[str, threading.Event] = {}


def register_job_process(job_id: str, pid: int) -> None:
    if job_id not in _job_processes:
        _job_processes[job_id] = []
    if pid not in _job_processes[job_id]:
        _job_processes[job_id].append(pid)


def register_backtester(job_id: str, bt) -> None:
    _bt_refs[job_id] = bt


def register_deep_pool(job_id: str, pool) -> None:
    _deep_pools[job_id] = pool


def register_job_thread(job_id: str, thread: threading.Thread) -> None:
    _job_threads[job_id] = thread


def register_cancellation_event(job_id: str, event: threading.Event) -> None:
    _job_cancellation_events[job_id] = event


def force_stop_job(job_id: str) -> int:
    """Aggressively stop a job: signal cancellation, interrupt thread, kill process tree."""
    killed = 0

    # 1. Set cancellation event
    evt = _job_cancellation_events.pop(job_id, None)
    if evt is not None:
        evt.set()

    # 2. Interrupt the thread via async exception
    thread = _job_threads.pop(job_id, None)
    if thread is not None and thread.is_alive():
        try:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident),
                ctypes.py_object(SystemExit),
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), None)
        except Exception:
            pass
        thread.join(timeout=3)

    # 3. Kill child processes
    killed += cleanup_job(job_id)

    return killed


def _terminate_process(pid: int, timeout: float = 3.0) -> bool:
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return False


def _kill_process_tree(pid: int) -> int:
    killed = 0
    try:
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)
        for child in children:
            try:
                _terminate_process(child.pid)
                killed += 1
            except Exception:
                pass
        _terminate_process(pid)
        killed += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return killed


def cleanup_job(job_id: str) -> int:
    killed = 0
    pids = _job_processes.pop(job_id, [])
    for pid in pids:
        try:
            killed += _kill_process_tree(pid)
        except Exception:
            pass

    bt = _bt_refs.pop(job_id, None)
    if bt is not None:
        try:
            bt.free(release_data=True)
        except Exception:
            pass

    pool = _deep_pools.pop(job_id, None)
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    # Also shut down the class-level DeepMixin pool if this job owns it
    try:
        from pipeline.backtester.deep_mixin import DeepMixin
        if DeepMixin._deep_pool_job_id == job_id:
            DeepMixin._shutdown_deep_pool()
    except Exception:
        pass

    # Kill any lingering joblib/loky workers
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
    except Exception:
        pass

    return killed


def cleanup_all() -> int:
    killed = 0
    for job_id in list(_job_processes.keys()):
        killed += cleanup_job(job_id)
    _job_processes.clear()
    _bt_refs.clear()
    _deep_pools.clear()
    _job_threads.clear()
    _job_cancellation_events.clear()
    return killed


def reap_orphaned_processes() -> int:
    killed = 0
    target_keywords = {"python", "mlb", "mlbacktester", "optuna"}
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            info = proc.info
            if info["ppid"] != 1:
                continue
            cmdline = " ".join(info["cmdline"] or [])
            is_ml_process = any(
                kw.lower() in cmdline.lower() for kw in target_keywords
            ) or "joblib" in cmdline.lower() or "loky" in cmdline.lower()
            if not is_ml_process:
                continue
            if info["pid"] == psutil.Process().pid:
                continue
            _terminate_process(info["pid"])
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed
