"""Process pool manager for concurrent backtest execution.

Gatekeeper & Dispatcher architecture:
- GPU VRAM ledger tracks per-job memory budgets.
- Jobs declare vram_budget_mb; the gate checks remaining VRAM before approval.
- Per-process env_vars (CUDA_VRAM_LIMIT_MB, MLB_THREADS) pass via args.
- Parent os.environ is NEVER mutated.
- VRAM released in done_callback regardless of success/crash.
"""
from __future__ import annotations

import atexit
import os
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager, get_context
from typing import Any, Dict, List, Optional

from api.config import settings

GPU_MODELS = {"lstm", "cnn", "transformer"}


class ProcessManager:
    """Manages CPU pool, GPU pool, and VRAM ledger for backtest execution."""

    def __init__(self):
        self._cpu_pool: Optional[ProcessPoolExecutor] = None
        self._gpu_pool: Optional[ProcessPoolExecutor] = None
        self._manager: Optional[Manager] = None
        self._cancel_events: Any = None
        self._active_futures: Dict[str, Future] = {}
        self._initialized = False

        self._vram_lock = threading.Lock()
        self._gpu_vram_used_mb: int = 0

        self._job_vram: Dict[str, int] = {}

    @property
    def gpu_vram_used_mb(self) -> int:
        return self._gpu_vram_used_mb

    @property
    def gpu_vram_available_mb(self) -> int:
        return max(0, settings.gpu_total_vram_mb - self._gpu_vram_used_mb)

    def allocate_vram(self, budget_mb: int) -> bool:
        if budget_mb <= 0:
            return True
        if settings.gpu_total_vram_mb <= 0:
            return True
        with self._vram_lock:
            if self._gpu_vram_used_mb + budget_mb > settings.gpu_total_vram_mb:
                return False
            self._gpu_vram_used_mb += budget_mb
            return True

    def release_vram(self, budget_mb: int) -> None:
        if budget_mb <= 0:
            return
        with self._vram_lock:
            self._gpu_vram_used_mb = max(0, self._gpu_vram_used_mb - budget_mb)

    def _release_vram_for_job(self, job_id: str) -> None:
        budget = self._job_vram.pop(job_id, 0)
        self.release_vram(budget)

    def initialize(
        self,
        max_cpu: int | None = None,
        max_gpu: int | None = None,
        gpu_enabled: bool | None = None,
    ) -> None:
        if self._initialized:
            return

        cpu_size = max_cpu if max_cpu is not None else settings.max_concurrent_backtests
        gpu_size = max_gpu if max_gpu is not None else settings.max_concurrent_gpu
        gpu_on = gpu_enabled if gpu_enabled is not None else settings.gpu_enabled

        self._manager = Manager()
        self._cancel_events = self._manager.dict()

        self._cpu_pool = ProcessPoolExecutor(
            max_workers=cpu_size,
            mp_context=get_context("spawn"),
            initializer=_worker_initializer,
            initargs=(self._cancel_events,),
        )
        print(f"[ProcessManager] CPU pool: {cpu_size} workers", flush=True)

        if gpu_on and gpu_size > 0:
            self._gpu_pool = ProcessPoolExecutor(
                max_workers=gpu_size,
                mp_context=get_context("spawn"),
                initializer=_worker_initializer,
                initargs=(self._cancel_events,),
            )
            print(f"[ProcessManager] GPU pool: {gpu_size} workers", flush=True)
        else:
            print("[ProcessManager] GPU pool: disabled", flush=True)

        atexit.register(self.shutdown)
        self._initialized = True
        print(
            f"[ProcessManager] VRAM ledger: {settings.gpu_total_vram_mb} MB total, "
            f"{self.gpu_vram_available_mb} MB available",
            flush=True,
        )

    def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
        if not self._initialized:
            return

        had_active = False
        for job_id in list(self._active_futures.keys()):
            if self._cancel_events is not None:
                self._cancel_events[job_id] = True
                had_active = True

        if had_active:
            print(
                "[ProcessManager] Signalled cancellation, waiting for clean exit...",
                flush=True,
            )

        _ESCALATE_S = 3.0 if wait else 0.0
        if _ESCALATE_S > 0 and had_active:
            deadline = time.monotonic() + _ESCALATE_S
            while time.monotonic() < deadline:
                remaining = any(
                    not f.done() for f in list(self._active_futures.values())
                )
                if not remaining:
                    print("[ProcessManager] All jobs stopped cleanly.", flush=True)
                    break
                time.sleep(0.25)

        if wait or cancel_futures:
            if self._cpu_pool:
                self._cpu_pool.shutdown(wait=wait, cancel_futures=cancel_futures)
                self._cpu_pool = None

            if self._gpu_pool:
                self._gpu_pool.shutdown(wait=wait, cancel_futures=cancel_futures)
                self._gpu_pool = None

        if self._manager:
            self._manager.shutdown()
            self._manager = None
            self._cancel_events = None

        self._active_futures.clear()
        self._job_vram.clear()
        self._gpu_vram_used_mb = 0
        self._initialized = False
        print("[ProcessManager] Shutdown complete", flush=True)

    def submit(
        self,
        job_id: str,
        config: Dict[str, Any],
        env_vars: Optional[Dict[str, str]] = None,
        vram_budget_mb: int = 0,
    ) -> Future:
        if not self._initialized:
            raise RuntimeError("ProcessManager not initialized")

        models: List[str] = config.get("models", [])
        is_gpu = any(m.lower() in GPU_MODELS for m in models)

        if is_gpu and self._gpu_pool is not None:
            pool = self._gpu_pool
            queue_name = "gpu"
        elif is_gpu and self._gpu_pool is None:
            pool = self._cpu_pool
            queue_name = "cpu (gpu disabled)"
        else:
            pool = self._cpu_pool
            queue_name = "cpu"

        self._cancel_events[job_id] = False

        if vram_budget_mb > 0:
            self._job_vram[job_id] = vram_budget_mb

        future = pool.submit(
            _run_backtest_in_worker, job_id, config, env_vars or {}
        )
        self._active_futures[job_id] = future

        def _done_callback(f: Future):
            self._active_futures.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            self._release_vram_for_job(job_id)
            if f.exception():
                print(
                    f"[ProcessManager] Job {job_id[:8]} failed: {f.exception()}",
                    flush=True,
                )

        future.add_done_callback(_done_callback)
        print(
            f"[ProcessManager] Job {job_id[:8]} submitted to {queue_name} pool "
            f"(models={models}, vram={vram_budget_mb} MB, env={list(env_vars or {})})",
            flush=True,
        )
        return future

    def request_cancellation(self, job_id: str) -> bool:
        if self._cancel_events is None:
            return False
        if job_id not in self._cancel_events:
            return False
        self._cancel_events[job_id] = True
        return True

    @property
    def active_count(self) -> int:
        return len(self._active_futures)

    @property
    def active_job_ids(self) -> List[str]:
        return list(self._active_futures.keys())


_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager


_cancel_dict: Any = None


def _worker_initializer(cancel_dict: Any):
    global _cancel_dict
    _cancel_dict = cancel_dict


def is_cancelled(job_id: str) -> bool:
    global _cancel_dict
    if _cancel_dict is None:
        return False
    try:
        return bool(_cancel_dict.get(job_id, False))
    except Exception:
        return False


def _run_backtest_in_worker(
    job_id: str,
    config: Dict[str, Any],
    env_vars: Dict[str, str],
):
    """Entry point executed inside the worker process.

    env_vars are applied to the child's os.environ BEFORE any imports,
    so CUDA_VRAM_LIMIT_MB and MLB_THREADS are set before TF/BLAS init.
    """
    for k, v in env_vars.items():
        os.environ[k] = str(v)

    if env_vars:
        try:
            from threadpoolctl import threadpool_limits
            tb = env_vars.get("MLB_THREADS")
            if tb:
                threadpool_limits(limits=int(tb))
        except Exception:
            pass

    project_root = os.environ.get("FX_PROJECT_ROOT", settings.project_root)
    if project_root not in __import__("sys").path:
        __import__("sys").path.insert(0, project_root)
    os.chdir(project_root)

    from api.dependencies import get_data_store
    get_data_store()

    from api.tasks import _run_backtest_impl as _impl
    return _impl(job_id, config)
