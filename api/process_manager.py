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
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager, get_context
from typing import Any, Dict, List, Optional

from api.config import settings

GPU_MODELS = {"lstm", "cnn", "transformer", "dqn"}


class _PendingQueue:
    """FIFO per-pool queue. cpu/gpu queues are independent."""

    def __init__(self):
        self._cpu_queue: deque = deque()
        self._gpu_queue: deque = deque()
        self._lock = threading.Lock()

    def enqueue(self, kind: str, job_id: str, config, env_vars, vram_budget_mb: int) -> None:
        with self._lock:
            q = self._gpu_queue if kind == "gpu" else self._cpu_queue
            q.append((job_id, config, env_vars, vram_budget_mb))

    def pop(self, kind: str):
        with self._lock:
            q = self._gpu_queue if kind == "gpu" else self._cpu_queue
            return q.popleft() if q else None

    def push_front(self, kind: str, item) -> None:
        with self._lock:
            q = self._gpu_queue if kind == "gpu" else self._cpu_queue
            q.appendleft(item)

    def cpu_len(self) -> int:
        with self._lock:
            return len(self._cpu_queue)

    def gpu_len(self) -> int:
        with self._lock:
            return len(self._gpu_queue)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cpu_queue) + len(self._gpu_queue)

    def clear(self) -> None:
        with self._lock:
            self._cpu_queue.clear()
            self._gpu_queue.clear()


class ProcessManager:
    """Manages CPU pool, GPU pool, and VRAM ledger for backtest execution."""

    def __init__(self):
        self._cpu_pool: Optional[ProcessPoolExecutor] = None
        self._gpu_pool: Optional[ProcessPoolExecutor] = None
        self._manager: Optional[Manager] = None
        self._cancel_events: Any = None
        self._active_futures: Dict[str, Future] = {}
        self._job_pool: Dict[str, str] = {}
        self._initialized = False

        self._vram_lock = threading.Lock()
        self._gpu_vram_used_mb: int = 0

        self._job_vram: Dict[str, int] = {}
        self._pending_vram: Dict[str, int] = {}
        self._pending = _PendingQueue()

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
        self._job_pool.clear()
        self._job_vram.clear()
        self._pending_vram.clear()
        self._pending.clear()
        self._gpu_vram_used_mb = 0
        self._initialized = False
        print("[ProcessManager] Shutdown complete", flush=True)

    def submit_or_queue(
        self,
        job_id: str,
        config: Dict[str, Any],
        env_vars: Optional[Dict[str, str]] = None,
        vram_budget_mb: int = 0,
    ) -> str:
        """Submit job to pool or enqueue if full. Returns 'dispatched' or 'queued'."""
        if not self._initialized:
            raise RuntimeError("ProcessManager not initialized")

        models: List[str] = config.get("models", [])
        is_gpu = any(m.lower() in GPU_MODELS for m in models)
        pool_name = "gpu" if is_gpu and self._gpu_pool is not None else "cpu"
        pool = self._gpu_pool if pool_name == "gpu" else self._cpu_pool
        total_vram = settings.gpu_total_vram_mb

        if is_gpu and pool_name == "gpu" and vram_budget_mb > 0 and total_vram > 0:
            if vram_budget_mb > total_vram:
                raise RuntimeError(
                    f"Requested VRAM ({vram_budget_mb} MB) exceeds total system VRAM "
                    f"({total_vram} MB)"
                )
            if not self.allocate_vram(vram_budget_mb):
                self._pending_vram[job_id] = vram_budget_mb
                self._pending.enqueue(pool_name, job_id, config, env_vars, vram_budget_mb)
                print(
                    f"[ProcessManager] Job {job_id[:8]} queued (VRAM full, "
                    f"need {vram_budget_mb} MB, available {self.gpu_vram_available_mb} MB)",
                    flush=True,
                )
                return "queued"

        pool_size = pool._max_workers
        active_in_pool = self.active_gpu_count() if pool_name == "gpu" else self.active_cpu_count()
        if active_in_pool >= pool_size:
            self._pending.enqueue(pool_name, job_id, config, env_vars, vram_budget_mb)
            print(
                f"[ProcessManager] Job {job_id[:8]} queued ({pool_name} pool full, "
                f"{active_in_pool}/{pool_size} active)",
                flush=True,
            )
            return "queued"

        self._dispatch_to_pool(job_id, config, env_vars or {}, vram_budget_mb, pool_name)
        return "dispatched"

    def _dispatch_to_pool(
        self,
        job_id: str,
        config: Dict[str, Any],
        env_vars: Dict[str, str],
        vram_budget_mb: int,
        pool_name: str,
    ) -> None:
        pool = self._gpu_pool if pool_name == "gpu" else self._cpu_pool
        queue_name = pool_name if pool_name == "gpu" else "cpu"

        self._cancel_events[job_id] = False

        if vram_budget_mb > 0:
            self._job_vram[job_id] = vram_budget_mb

        future = pool.submit(
            _run_backtest_in_worker, job_id, config, env_vars
        )
        self._active_futures[job_id] = future
        self._job_pool[job_id] = pool_name

        def _done_callback(f: Future):
            self._active_futures.pop(job_id, None)
            self._job_pool.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            self._release_vram_for_job(job_id)
            if f.exception():
                print(
                    f"[ProcessManager] Job {job_id[:8]} failed: {f.exception()}",
                    flush=True,
                )
            self._pump_queue(pool_name)

        future.add_done_callback(_done_callback)
        models: List[str] = config.get("models", [])
        print(
            f"[ProcessManager] Job {job_id[:8]} dispatched to {queue_name} pool "
            f"(models={models}, vram={vram_budget_mb} MB, env={list(env_vars)})",
            flush=True,
        )

    def _pump_queue(self, pool_name: str) -> None:
        """Pull next job from pending queue and dispatch if a slot is free."""
        pool = self._gpu_pool if pool_name == "gpu" else self._cpu_pool
        if pool is None:
            return
        pool_size = pool._max_workers
        while True:
            active = self.active_gpu_count() if pool_name == "gpu" else self.active_cpu_count()
            if active >= pool_size:
                break
            job = self._pending.pop(pool_name)
            if job is None:
                break
            job_id, config, env_vars, vram_budget_mb = job
            pending_vram = self._pending_vram.pop(job_id, 0)
            if pending_vram > 0 and not self.allocate_vram(pending_vram):
                self._pending_vram[job_id] = pending_vram
                self._pending.push_front(pool_name, (job_id, config, env_vars, vram_budget_mb))
                break
            if pending_vram > 0:
                vram_budget_mb = pending_vram
            try:
                self._dispatch_to_pool(job_id, config, env_vars or {}, vram_budget_mb, pool_name)
            except Exception:
                self._pending.push_front(pool_name, (job_id, config, env_vars, vram_budget_mb))
                if pending_vram > 0:
                    self._pending_vram[job_id] = pending_vram
                break

    def resize_pools(self, cpu_size: int, gpu_size: int, gpu_enabled: bool) -> None:
        """Gracefully swap out pools with new sizes. Running jobs finish on old pools."""
        if not self._initialized:
            self.initialize(max_cpu=cpu_size, max_gpu=gpu_size, gpu_enabled=gpu_enabled)
            return

        if self._cpu_pool is not None and self._cpu_pool._max_workers != cpu_size:
            old = self._cpu_pool
            self._cpu_pool = ProcessPoolExecutor(
                max_workers=cpu_size,
                mp_context=get_context("spawn"),
                initializer=_worker_initializer,
                initargs=(self._cancel_events,),
            )
            old.shutdown(wait=False, cancel_futures=False)
            print(f"[ProcessManager] CPU pool resized: {cpu_size} workers", flush=True)

        if gpu_enabled and gpu_size > 0:
            if self._gpu_pool is None or self._gpu_pool._max_workers != gpu_size:
                old_gpu = self._gpu_pool
                self._gpu_pool = ProcessPoolExecutor(
                    max_workers=gpu_size,
                    mp_context=get_context("spawn"),
                    initializer=_worker_initializer,
                    initargs=(self._cancel_events,),
                )
                if old_gpu is not None:
                    old_gpu.shutdown(wait=False, cancel_futures=False)
                print(f"[ProcessManager] GPU pool resized: {gpu_size} workers", flush=True)
        else:
            if self._gpu_pool is not None:
                self._gpu_pool.shutdown(wait=False, cancel_futures=False)
                self._gpu_pool = None
                print("[ProcessManager] GPU pool disabled", flush=True)

        self._pump_queue("cpu")
        self._pump_queue("gpu")

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

    def active_cpu_count(self) -> int:
        return sum(1 for jid in self._active_futures if self._job_pool.get(jid) == "cpu")

    def active_gpu_count(self) -> int:
        return sum(1 for jid in self._active_futures if self._job_pool.get(jid) == "gpu")

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

    tb = env_vars.get("MLB_THREADS", "2")
    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if _key not in os.environ:
            os.environ[_key] = tb

    project_root = os.environ.get("FX_PROJECT_ROOT", settings.project_root)
    if project_root not in __import__("sys").path:
        __import__("sys").path.insert(0, project_root)
    os.chdir(project_root)

    from api.dependencies import get_data_store
    get_data_store()

    from api.tasks import _run_backtest_impl as _impl
    return _impl(job_id, config)
