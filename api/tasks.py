"""Celery application and task definitions.

In desktop mode (FX_APP_MODE=desktop), Celery is unavailable so backtests
run synchronously in-process. The task functions still exist as wrappers
but fall back to direct execution when the broker is unreachable.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from api.config import settings
from pipeline.data.data_sqlite import DataStore

HYPERPARAM_ALIASES: Dict[str, Dict[str, str]] = {
    "logistic": {
        "C": "logit_C",
        "solver": "logit_solver",
        "penalty": "logit_penalty",
        "max_iter": "logit_max_iter",
        "tol": "logit_tol",
        "class_weight": "logit_class_weight",
    },
    "xgboost": {
        "n_estimators": "xgb_n_estimators",
        "max_depth": "xgb_max_depth",
        "learning_rate": "xgb_learning_rate",
        "subsample": "xgb_subsample",
        "colsample_bytree": "xgb_colsample_bytree",
    },
    "svm": {
        "C": "svm_C",
        "gamma": "svm_gamma",
        "kernel": "svm_kernel",
        "class_weight": "svm_class_weight",
    },
    "random_forest": {
        "n_estimators": "rf_n_estimators",
        "max_depth": "rf_max_depth",
        "min_samples_leaf": "rf_min_samples_leaf",
        "max_features": "rf_max_features",
    },
    "decision_tree": {
        "max_depth": "dt_max_depth",
        "min_samples_leaf": "dt_min_samples_leaf",
        "max_features": "dt_max_features",
        "ccp_alpha": "dt_ccp_alpha",
    },
    "lstm": {
        "units": "lstm_units",
        "num_layers": "lstm_num_layers",
        "dropout_rate": "lstm_dropout_rate",
        "learning_rate": "lstm_learning_rate",
    },
    "cnn": {
        "filters": "cnn_filters",
        "kernel_size": "cnn_kernel_size",
        "learning_rate": "cnn_learning_rate",
    },
    "transformer": {
        "d_model": "transformer_d_model",
        "num_heads": "transformer_num_heads",
        "dropout_rate": "transformer_dropout_rate",
        "learning_rate": "transformer_learning_rate",
    },
}


def _convert_model_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Convert model__param keys (e.g. 'logistic__C') to internal names (e.g. 'logit_C')."""
    converted: Dict[str, Any] = {}
    consumed: set = set()
    for key, value in overrides.items():
        if "__" not in key:
            continue
        parts = key.split("__", 1)
        if len(parts) != 2:
            continue
        model, param = parts
        aliases = HYPERPARAM_ALIASES.get(model, {})
        internal_key = aliases.get(param, f"{model}_{param}")
        converted[internal_key] = value
        consumed.add(key)
    result = {k: v for k, v in overrides.items() if k not in consumed}
    result.update(converted)
    return result
from api.schemas.backtest import HPO_TRIAL_MAPS
from logging_config import emit_event, log_print

IS_DESKTOP = os.environ.get("FX_APP_MODE", "") == "desktop"

_TRIAL_COUNTS_FALLBACK = {
    "logistic":       {"random": 3, "bayes": 3},
    "svm":            {"random": 3, "bayes": 3},
    "decision_tree":  {"random": 3, "bayes": 3},
    "random_forest":  {"random": 3, "bayes": 3},
    "xgboost":        {"random": 3, "bayes": 3},
    "lstm":           {"random": 3, "bayes": 3},
    "cnn":            {"random": 3, "bayes": 3},
    "transformer":    {"random": 3, "bayes": 3},
    "ensemble_adaptive_regime":     {"random": 3, "bayes": 3},
    "ensemble_cnn_lstm_xgboost":   {"random": 3, "bayes": 3},
    "dqn":            {"random": 3, "bayes": 3},
}


def _get_trial_counts(hpo_intensity: str | None, model: str) -> Dict[str, int]:
    if hpo_intensity and hpo_intensity in HPO_TRIAL_MAPS:
        model_map = HPO_TRIAL_MAPS[hpo_intensity]
        if model in model_map:
            return model_map[model]
    return _TRIAL_COUNTS_FALLBACK.get(model, {"random": 3, "bayes": 3})

CV_BLOCKS_DEFAULT = 5


def _compute_total_work(models: list[str], months: int, hpo_intensity: str | None = None, cv_blocks: int = CV_BLOCKS_DEFAULT, period_unit: str = "months", n_trials_override: int | None = None) -> int:
    from config import convert_month_count_to_periods
    n_periods = convert_month_count_to_periods(months, period_unit)
    total = 0
    for m in models:
        if n_trials_override is not None and n_trials_override >= 0:
            n_trials = n_trials_override
        else:
            tc = _get_trial_counts(hpo_intensity, m)
            n_trials = tc.get("random", 3) + tc.get("bayes", 3)
        hpo_work = n_trials * cv_blocks
        sim_work = n_periods
        total += hpo_work + sim_work
    return max(total, 1)


def _make_progress_callback(job_id: str, total_work: int, jm):
    """Factory for progress callback closures. Each job gets its own state."""
    job_start_time = time.time()
    completed_work = 0
    _last_heartbeat = 0
    _completion_times = []

    def _progress_cb(phase: str, model: str, detail: dict | None = None):
        nonlocal completed_work, _last_heartbeat, _completion_times
        _check_force_stopped(job_id)
        data = {"model": model, "phase": phase}
        if detail:
            data.update(detail)
        if phase == "hpo_trial":
            cv_b = detail.get("cv_blocks", CV_BLOCKS_DEFAULT) if detail else CV_BLOCKS_DEFAULT
            completed_work += cv_b
        elif phase == "month":
            completed_work += 1
        elif phase == "period":
            completed_work += 1
        pct = min(round((completed_work / total_work) * 100, 1), 100) if total_work > 0 else 0
        data["completed_work"] = completed_work
        data["total_work"] = total_work
        data["progress_pct"] = pct

        # ETA computation via rolling window of work-unit timestamps
        _completion_times.append((completed_work, time.time()))
        if len(_completion_times) > 20:
            _completion_times = _completion_times[-20:]
        elapsed = time.time() - job_start_time
        data["elapsed_seconds"] = round(elapsed, 1)
        if len(_completion_times) >= 2:
            first_wu, first_ts = _completion_times[0]
            last_wu, last_ts = _completion_times[-1]
            wu_delta = last_wu - first_wu
            time_delta = last_ts - first_ts
            if wu_delta > 0 and time_delta > 0:
                remaining = total_work - completed_work
                data["eta_seconds"] = round(remaining * time_delta / wu_delta, 1)
            else:
                data["eta_seconds"] = None
        else:
            data["eta_seconds"] = None

        if completed_work - _last_heartbeat >= 2:
            _last_heartbeat = completed_work
            try:
                jm.touch_job(job_id)
            except Exception:
                pass
        if phase in ("hpo", "hpo_trial"):
            evt_name = "hpo_progress"
        elif phase in ("month", "period"):
            evt_name = "month_progress"
        else:
            evt_name = "model_phase"
        _pub(evt_name, job_id, data)
        emit_event(evt_name, job_id=job_id, pct=pct, **data)

        if phase == "simulation_started" and detail:
            _pub("simulation_started", job_id, {
                "model": model,
                "n_periods": detail["n_periods"],
                "bh_curve": detail["bh_curve"],
            })

        elif phase == "hpo_trial" and detail:
            trial_n = detail.get("trial", 0)
            trial_score = detail.get("score")
            trial_state = detail.get("trial_state", "?")
            total_trials = detail.get("total_trials", "?")
            params_keys = list(detail.get("params", {}).keys())
            if trial_score is not None:
                log_print(f"[HPO] Trial {trial_n}/{total_trials}  score={trial_score:.2f}  state={trial_state}  "
                          f"params=[{','.join(params_keys[:6])}...] ({len(params_keys)} total)")
            else:
                log_print(f"[HPO] Trial {trial_n}/{total_trials}  PRUNED  state={trial_state}")
            _pub("hpo_trial_result", job_id, {
                "model": model,
                "trial_number": trial_n,
                "score": trial_score,
                "params": detail.get("params", {}),
                "best_score_so_far": detail.get("best_score_so_far"),
                "trial_state": trial_state,
            })

        elif phase in ("month", "period") and detail:
            _pub("oos_result", job_id, {
                "model": model,
                "period": detail.get("period", 0),
                "total_periods": detail.get("total_periods", 0),
                "equity": detail.get("equity_strategy"),
                "equity_bh": detail.get("equity_bh"),
                "sharpe": detail.get("sharpe"),
                "return_pct": detail.get("return_pct"),
                "trades": detail.get("trades"),
                "drawdown": detail.get("drawdown"),
                "win_rate": detail.get("win_rate"),
                "precision": detail.get("precision_macro"),
                "f1": detail.get("f1_macro"),
                "directional_accuracy": detail.get("directional_accuracy"),
                "active_rate": detail.get("active_rate"),
                "train_sharpe": detail.get("train_sharpe"),
                "sharpe_gap_pct": detail.get("sharpe_gap_pct"),
                "signals_raw": detail.get("signals_raw"),
                "signals_passed_gate": detail.get("signals_passed_gate"),
                "signal_coverage": detail.get("signal_coverage"),
                "profit_per_hit": detail.get("profit_per_hit"),
                "outperformance": detail.get("outperformance"),
            })

    return _progress_cb


celery_app = None
_celery_available = False

# In-memory mapping of job_id → celery_task_id for force-stop revocation
_JOB_TASK_IDS: Dict[str, str] = {}

# Desktop mode: threading.Event per job for thread-safe cancellation
import threading as _threading
_cancellation_events: Dict[str, _threading.Event] = {}


def request_cancellation(job_id: str) -> bool:
    """Signal cancellation for a running job (desktop mode)."""
    evt = _cancellation_events.get(job_id)
    if evt is not None:
        evt.set()
        return True
    return False


def _is_cancelled(job_id: str) -> bool:
    """Check if job has been cancelled via DB flag OR threading event."""
    evt = _cancellation_events.get(job_id)
    if evt is not None and evt.is_set():
        return True
    try:
        store = DataStore(settings.db_full_path)
        from api.services import JobManager as _JM
        jm = _JM(store)
        job = jm.get_job(job_id)
        if job and job.get("status") == "failed" and "stopped" in str(job.get("error", "")).lower():
            return True
    except Exception:
        pass
    return False


def revoke_task(job_id: str) -> bool:
    """Revoke the Celery task / signal cancellation for a job_id (force stop).

    On Windows, terminate=True uses TerminateProcess which is a hard kill
    that skips all cleanup handlers, leaving child processes orphaned.
    Instead we set the threading event so the backtester's _force_stop_checker
    raises KeyboardInterrupt -> cleanup runs gracefully. If the task hasn't
    stopped in 5s, escalate with terminate as last resort.

    Reads Celery task_id from the jobs DB table for cross-worker revocation.
    """
    # Try in-memory dict first, fall back to DB
    task_id = _JOB_TASK_IDS.pop(job_id, None)
    if not task_id:
        try:
            store = DataStore(settings.db_full_path)
            from api.services import JobManager as _JM
            jm = _JM(store)
            task_id = jm.get_task_id(job_id)
        except Exception:
            pass
    revoked = False

    # Phase 1: gentle signal via threading event (works cross-platform)
    evt = _cancellation_events.get(job_id)
    if evt is not None:
        evt.set()
        revoked = True

    # Phase 2: deferred hard kill (only if gentle signal doesn't work)
    if task_id and _celery_available and celery_app is not None:
        try:
            import threading as _thr
            def _escalate():
                import time
                time.sleep(5.0)
                try:
                    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                except Exception:
                    pass
            _thr.Thread(target=_escalate, daemon=True).start()
            revoked = True
        except Exception:
            pass

    return revoked


def _check_force_stopped(job_id: str):
    """Check if the job has been force-stopped. Raises KeyboardInterrupt if stopped."""
    if _is_cancelled(job_id):
        raise KeyboardInterrupt("Force stopped by user")


_wsl_available_cache = None


def _wsl_available() -> bool:
    global _wsl_available_cache
    if _wsl_available_cache is not None:
        return _wsl_available_cache
    try:
        import subprocess as _sp
        result = _sp.run(["wsl", "--status"], capture_output=True, text=True, timeout=10)
        _wsl_available_cache = result.returncode == 0
    except Exception:
        _wsl_available_cache = False
    return _wsl_available_cache


def _has_gpu_models(models: list[str]) -> bool:
    from pipeline.runtime import GPU_RECOMMENDED_MODELS
    return any(m in GPU_RECOMMENDED_MODELS for m in models)


def _run_backtest_via_wsl(job_id: str, config: dict) -> bool:
    """Run backtest in WSL2 for GPU acceleration. Returns True if dispatched, False if fallback needed."""
    models = config.get("models", [])

    if not _has_gpu_models(models):
        return False

    if not _wsl_available():
        _pub("wsl_fallback", job_id, {"reason": "WSL2 not available, using CPU fallback"})
        emit_event("wsl_fallback", job_id=job_id, reason="WSL2 not available")
        return False

    from api.services import JobManager
    from pipeline.data.data_sqlite import DataStore

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)
    jm.update_status(job_id, "running", error=None)
    _pub("job_started", job_id, {"pair": config.get("pair"), "models": models, "runtime": "wsl_gpu"})
    emit_event("job_started", job_id=job_id, runtime="wsl_gpu")

    months_wsl = config.get("months", 3)
    hpo_intensity_wsl = config.get("hpo_intensity", "quick")
    n_trials_override_wsl = config.get("n_trials")
    period_unit_wsl = config.get("period_unit", "months")
    total_work_wsl = _compute_total_work(models, months_wsl, hpo_intensity_wsl, period_unit=period_unit_wsl, n_trials_override=n_trials_override_wsl)
    progress_cb = _make_progress_callback(job_id, total_work_wsl, jm)

    project_root = settings.project_root
    results_dir = os.path.join(project_root, "results")

    import tempfile
    job_file_path = None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"wsl_job_{job_id}_", delete=False) as f:
        wsl_config = dict(config)
        wsl_config["results_dir"] = results_dir
        json.dump(wsl_config, f)
        job_file_path = f.name

    wsl_job_path_unix = job_file_path.replace("\\", "/").replace("C:", "/mnt/c")

    cmd = [
        "wsl", "bash", "-c",
        f"cd /mnt/c/Users/rafa/ML_Trading/thesisproj && "
        f"export PYTHONPATH=/mnt/c/Users/rafa/ML_Trading/thesisproj && "
        f"export MLB_THREADS=1 && "
        f"~/thesisproj-venv/bin/python pipeline/wsl_runner.py "
        f"--job-file={wsl_job_path_unix} --job-id={job_id}"
    ]

    try:
        import subprocess as _sp
        proc = _sp.Popen(
            cmd, stdout=_sp.PIPE, stderr=_sp.PIPE,
            text=True, cwd=project_root,
        )

        _wsl_procs[job_id] = proc

        for line in iter(proc.stdout.readline, ""):
            if _is_cancelled(job_id):
                proc.terminate()
                break
            line = line.strip()
            if line.startswith("[WSL_PROGRESS:"):
                try:
                    rest = line[len("[WSL_PROGRESS:"):]
                    parts = rest.split(":", 2)
                    if len(parts) >= 3:
                        _, evt_name, data_str = parts
                        data_str = data_str.rstrip("]")
                        data = json.loads(data_str)
                        _pub(evt_name, job_id, data)
                        emit_event(evt_name, job_id=job_id, **data)
                except Exception:
                    pass
            elif line.startswith("[WSL_EVENT:"):
                try:
                    rest = line[len("[WSL_EVENT:"):]
                    parts = rest.split(":", 2)
                    if len(parts) >= 3:
                        _, evt_name, msg = parts
                        msg = msg.rstrip("]")
                        _pub(evt_name, job_id, {"message": msg})
                except Exception:
                    pass
            elif line.startswith("[WSL_CB:"):
                try:
                    rest = line[len("[WSL_CB:"):]
                    parts = rest.split(":", 1)
                    if len(parts) >= 2:
                        _, payload_str = parts
                        payload_str = payload_str.rstrip("]")
                        payload = json.loads(payload_str)
                        progress_cb(payload["p"], payload["m"], payload.get("d"))
                except Exception:
                    pass

        proc.wait(timeout=30)
        _wsl_procs.pop(job_id, None)

        rc = proc.returncode
        if rc != 0:
            stderr_output = proc.stderr.read()
            raise RuntimeError(f"WSL process exited with code {rc}: {stderr_output[:500]}")

        result = {
            "pair": config.get("pair"),
            "models": models,
            "config": config,
            "metrics": [],
            "runtime": "wsl_gpu",
        }
        jm.update_status(job_id, "completed", result=result)
        _pub("job_complete", job_id, {"metrics": []})
        emit_event("job_complete", job_id=job_id, n_models=len(models))
        return True

    except Exception as e:
        _wsl_procs.pop(job_id, None)
        tb = traceback.format_exc()
        jm.update_status(job_id, "failed", error=f"WSL error: {e}\n{tb}")
        _pub("job_failed", job_id, {"error": str(e)})
        emit_event("job_failed", job_id=job_id, error=str(e)[:200])
        return True
    finally:
        if job_file_path and os.path.exists(job_file_path):
            try:
                os.unlink(job_file_path)
            except Exception:
                pass


_wsl_procs: dict = {}

if not IS_DESKTOP:
    try:
        from celery import Celery

        celery_app = Celery(
            "fx_pipeline",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend,
        )

        celery_app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            timezone="UTC",
            enable_utc=True,
            task_track_started=True,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            broker_connection_retry_on_startup=True,
            task_time_limit=3600,
            beat_schedule={
                "prefetch-news-hourly": {
                    "task": "prefetch_news",
                    "schedule": 3600.0,
                },
                "prefetch-calendar-daily": {
                    "task": "prefetch_calendar",
                    "schedule": 86400.0,
                },
            },
        )
        _celery_available = True
    except Exception:
        _celery_available = False

    if _celery_available:
        try:
            from celery.signals import worker_shutdown

            @worker_shutdown.connect
            def _on_worker_shutdown(**kwargs):
                print("[Celery] Worker shutting down, cleaning up...")
                try:
                    from api.shutdown import shutdown_cleanup
                    shutdown_cleanup(settings.db_full_path)
                except Exception:
                    pass
        except Exception:
            pass


_redis_client = None
_redis_available: bool | None = None


def _get_redis():
    global _redis_client, _redis_available
    if _redis_available is False:
        raise ConnectionError("Redis unavailable")
    if _redis_client is None:
        import redis as _redis_mod
        _redis_client = _redis_mod.from_url(settings.redis_url, socket_connect_timeout=2)
        try:
            _redis_client.ping()
            _redis_available = True
        except Exception:
            _redis_available = False
            _redis_client = None
            raise
    return _redis_client


_job_events: Dict[str, List[Dict]] = {}
_job_events_lock = threading.Lock()
_JOB_EVENTS_MAX = 5000


def _append_event(job_id: str, event: Dict):
    evt_name = event.get("event", "unknown")
    model = event.get("model", "-")
    import logging as _logging
    _logging.debug(f"[EVENT-WRITE] job={job_id[:8]} event={evt_name} model={model}")
    with _job_events_lock:
        if job_id not in _job_events:
            _job_events[job_id] = []
        lst = _job_events[job_id]
        lst.append(event)
        if len(lst) > _JOB_EVENTS_MAX:
            _job_events[job_id] = lst[-_JOB_EVENTS_MAX:]
    # Persist to SQLite so events are visible cross-process (API server <-> Celery worker)
    try:
        store = DataStore(settings.db_full_path)
        store.append_job_event(job_id, json.dumps(_sanitize_for_json(event)))
        store.trim_job_events(job_id, _JOB_EVENTS_MAX)
    except Exception as _e:
        _logging.warning(f"[event-store] SQLite append failed for {job_id}: {_e}")
    # Also push to Redis list as a fast-path optimization when Redis is up
    try:
        r = _get_redis()
        r.rpush(f"job_events:{job_id}", json.dumps(_sanitize_for_json(event)))
        r.ltrim(f"job_events:{job_id}", -_JOB_EVENTS_MAX, -1)
        r.expire(f"job_events:{job_id}", 86400)
    except Exception:
        global _redis_client, _redis_available
        _redis_client = None
        _redis_available = False


def get_job_events(job_id: str, after: int = 0) -> List[Dict]:
    # SQLite is the authoritative cross-process store
    try:
        store = DataStore(settings.db_full_path)
        events = store.get_job_events(job_id, after=after)
        import logging as _logging
        _logging.debug(f"[EVENT-READ] job={job_id[:8]} after={after} count={len(events)}")
        return events
    except Exception as _e:
        import logging as _logging
        _logging.warning(f"[event-store] SQLite read failed for {job_id} after={after}: {_e}")
    # Fallback to in-memory (desktop mode / SQLite unavailable)
    with _job_events_lock:
        lst = _job_events.get(job_id, [])
        return lst[after:]


def clear_job_events(job_id: str):
    with _job_events_lock:
        _job_events.pop(job_id, None)
    try:
        store = DataStore(settings.db_full_path)
        store.clear_job_events(job_id)
    except Exception:
        pass
    try:
        r = _get_redis()
        r.delete(f"job_events:{job_id}")
    except Exception:
        global _redis_client, _redis_available
        _redis_client = None
        _redis_available = False


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with null for valid JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _pub(event: str, job_id: str, data: Dict[str, Any]):
    """Publish a progress event to Redis pub/sub and in-memory buffer."""
    msg = {"event": event, "job_id": job_id, **data}
    msg = _sanitize_for_json(msg)
    _append_event(job_id, msg)
    try:
        r = _get_redis()
        r.publish(f"job:{job_id}", json.dumps(msg))
    except Exception:
        global _redis_client, _redis_available
        _redis_client = None
        _redis_available = False


def _sanitize_metrics(metrics_list: list) -> list:
    """Ensure all metric values are JSON-safe (no NaN, no Timestamp, no numpy types)."""

    def _coerce(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, dict):
            return {k2: _coerce(v2) for k2, v2 in v.items()}
        if isinstance(v, list):
            return [_coerce(item) for item in v]
        if isinstance(v, (np.datetime64, pd.Timestamp)):
            return str(v)
        return v

    safe = []
    for row in metrics_list:
        out = {}
        for k, v in row.items():
            out[k] = _coerce(v)
        safe.append(out)
    return safe


def _capture_environment() -> str:
    """Capture the current Python environment as a pip freeze string."""
    try:
        import subprocess
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True, stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return ""


def _run_backtest_impl(job_id: str, config: Dict[str, Any]):
    """Core backtest logic -- executed by Celery worker or in-process (desktop mode)."""
    project_root = settings.project_root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)
    _dbg = settings.debug

    from api.services import JobManager
    from pipeline.data.data_sqlite import DataStore

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)

    # Track this task for force-stop revocation
    _current_task_id = getattr(_run_backtest_impl, "request", None)
    if _current_task_id and hasattr(_current_task_id, "id"):
        _JOB_TASK_IDS[job_id] = _current_task_id.id
    # Register cancellation event so gentle force-stop signals work
    _cancellation_events[job_id] = threading.Event()
    # Ensure job row exists (defense against race-condition deletes)
    inserted = jm.ensure_job_exists(job_id, "backtest", config)
    if inserted:
        print(f"[job-recovery] Re-created missing job row for {job_id[:8]}", flush=True)
    jm.update_status(job_id, "running", error=None)

    pair = config.get("pair", "EURUSD")
    models = config.get("models", ["logistic"])
    start = config.get("start_date") or None
    end = config.get("end_date") or None
    months = config.get("months", 3)
    repeats = config.get("repeats", 1)
    seed = config.get("seed", 42)
    hpo_intensity = config.get("hpo_intensity", "quick")
    n_trials_override = config.get("n_trials")
    trading_costs = config.get("trading_costs", True)
    period_unit = config.get("period_unit", "months")
    parent_job_id = config.get("parent_job_id")
    pip_freeze = _capture_environment()

    tb = config.get("thread_budget")
    if tb and isinstance(tb, int) and tb >= 1:
        os.environ["MLB_THREADS"] = str(tb)
        os.environ["BLAS_THREADS_PER_TRIAL"] = str(tb)
        os.environ["XGB_JOBS"] = str(tb)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "SKLEARN_JOBS", "RF_JOBS"):
            os.environ.setdefault(k, str(tb))
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=tb)
        except Exception:
            pass
    import tempfile
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.path.join(tempfile.gettempdir(), f"joblib_{job_id[:12]}"))

    if end is None:
        from datetime import date
        today = date.today()
        first_of_current = today.replace(day=1)
        last_of_prev = first_of_current - pd.Timedelta(days=1)
        end = last_of_prev.strftime("%Y-%m-%d")

    try:
        from pipeline.data.data_sqlite import DataStore as _DS
        _store = _DS(settings.db_full_path)
        _tf_list = _store.list_timeframes(pair)
        if _tf_list:
            _rng = _store.get_date_range(pair, _tf_list[0])
            if _rng:
                _data_end = str(_rng[1])[:10]
                if end > _data_end:
                    end = _data_end
    except Exception:
        pass

    total_work = _compute_total_work(models, months, hpo_intensity, period_unit=period_unit, n_trials_override=n_trials_override)
    jm.update_status(job_id, "running")
    _pub("job_started", job_id, {"pair": pair, "models": models, "total_work": total_work})
    emit_event("job_started", job_id=job_id, pair=pair, models=",".join(models), total_work=total_work)

    _progress_cb = _make_progress_callback(job_id, total_work, jm)

    all_metrics = []

    # Dispatch GPU models to WSL2 if available (returns True if handled)
    if _run_backtest_via_wsl(job_id, config):
        return

    # --- S12.1: Pre-fetch news once per job (deployment only) ---
    _news_agg = None
    _econ_events = None
    _llm_agg = None
    _filtered_articles = []
    config_overrides = config.get("config_overrides", {})
    _use_news = bool(config_overrides.get("use_news", config.get("use_news", True)))
    if _use_news:
        import logging as _logging
        _nl = _logging.getLogger(__name__)

        _timeframe = config.get("timeframe", "H1")
        _freq_map = {"M30": "30min", "H1": "1h", "H4": "4h"}
        _news_freq = _freq_map.get(_timeframe, "1h")

        try:
            from news.scraper import NewsScraper, ECONOMIC_EVENTS
            from news.sentiment import SentimentAnalyzer

            scraper = NewsScraper()
            articles = scraper.fetch_all()

            _start_dt = pd.Timestamp(start) if start else pd.Timestamp(end) - pd.Timedelta(days=365)
            _end_dt = pd.Timestamp(end) + pd.Timedelta(days=1)

            _filtered_articles = [
                a for a in articles
                if _start_dt <= pd.Timestamp(a.timestamp) <= _end_dt
            ]

            if _filtered_articles:
                _news_backend = str(config_overrides.get("news_sentiment_backend", "vader"))
                analyzer = SentimentAnalyzer(backend=_news_backend)
                scored = analyzer.score_articles(_filtered_articles)
                _news_agg = analyzer.aggregate_to_df(scored, freq=_news_freq)

                _years = set()
                for y in range(_start_dt.year, _end_dt.year + 1):
                    _years.add(y)
                _econ_events = []
                for y in sorted(_years):
                    _econ_events.extend(
                        NewsScraper.economic_calendar_events(y, list(ECONOMIC_EVENTS))
                    )
        except Exception as _news_pre_exc:
            _nl.warning("News pre-fetch failed (will skip news features): %s", _news_pre_exc)

        _llm_enabled = bool(config_overrides.get("llm_sentiment_enabled", False))
        if _llm_enabled and _filtered_articles:
            try:
                from pipeline.llm.sentiment import LLMSentimentEngine

                _llm_cfg = {
                    "llm_sentiment_enabled": _llm_enabled,
                    "llm_backend": config_overrides.get("llm_backend", "ollama"),
                    "llm_model": config_overrides.get("llm_model", "llama3"),
                    "llm_api_key": config_overrides.get("llm_api_key", ""),
                    "llm_weight": config_overrides.get("llm_weight", 0.7),
                    "llm_batch_size": config_overrides.get("llm_batch_size", 10),
                    "llm_cache_ttl_hours": config_overrides.get("llm_cache_ttl_hours", 720),
                }
                llm_engine = LLMSentimentEngine(config=_llm_cfg)
                llm_scored = llm_engine.score_articles(_filtered_articles, pair=pair)
                _llm_agg = llm_engine.aggregate_to_df(llm_scored, freq=_news_freq)
            except Exception as _llm_pre_exc:
                _nl.warning("LLM sentiment pre-fetch failed (will skip LLM features): %s", _llm_pre_exc)

    try:
        for cycle_idx, model_type in enumerate(models):
            _check_force_stopped(job_id)

            _pub("cycle_started", job_id, {
                "model": model_type,
                "cycle_number": cycle_idx + 1,
                "total_cycles": len(models) * int(repeats),
            })
            _pub("model_training", job_id, {"model": model_type, "status": "starting"})
            _pub("model_phase", job_id, {"model": model_type, "phase": "hpo", "total_work": total_work})

            from copy import deepcopy
            from pipeline.metrics.metrics_tuples import CLASS_DEFAULTS
            from pipeline.backtester.composed import MLBacktester

            feat_cfg = deepcopy(CLASS_DEFAULTS["features"])
            feat_cfg.update(_convert_model_overrides(config.get("config_overrides", {})))

            # Resolve trial count: override (0–150) or intensity-based lookup
            if n_trials_override is not None and n_trials_override >= 0:
                n_trials_hdr = n_trials_override
                n_startup_hdr = max(n_trials_override // 2, 1) if n_trials_override > 0 else 0
            else:
                tc = _get_trial_counts(hpo_intensity, model_type)
                n_trials_hdr = max(tc.get("random", 3) + tc.get("bayes", 3), 10)
                n_startup_hdr = tc.get("random", 3)

            # Run repeats: each gets a different seed for HPO reproducibility
            rep_metrics = []
            for rep in range(1, int(repeats) + 1):
                if repeats > 1:
                    print(f"[REPEAT] {model_type}: rep {rep}/{repeats}")
                    _progress_cb(f"repeat", model_type, {"rep": rep, "total_repeats": repeats, "model": model_type})

                rep_seed = int(seed) + rep - 1

                bt = MLBacktester(
                    symbol=pair,
                    start=start,
                    end=end,
                    trading_costs=trading_costs,
                    model_type=model_type,
                    features_config=feat_cfg,
                    db_path=settings.db_full_path,
                )
                bt._progress_callback = _progress_cb
                bt._force_stop_checker = lambda: _is_cancelled(job_id)
                bt._job_id = job_id

                # S12.1: Inject pre-fetched news into backtester (deployment only)
                if _news_agg is not None:
                    bt._news_aggregated = _news_agg
                    bt._news_economic_events = _econ_events
                if _llm_agg is not None:
                    bt._llm_aggregated = _llm_agg

                from api.process_cleanup import register_backtester as _reg_bt_task
                _reg_bt_task(job_id, bt)

                base_cfg = deepcopy(CLASS_DEFAULTS["features"])
                base_cfg.update(deepcopy(CLASS_DEFAULTS["cv"]))
                base_cfg["model_type"] = model_type
                base_cfg["rep"] = rep
                base_cfg["trading_costs"] = trading_costs
                base_cfg["n_trials"] = n_trials_hdr
                base_cfg["n_startup_trials"] = n_startup_hdr
                base_cfg["use_cached_global_hpo"] = (n_trials_hdr <= 0)
                base_cfg["seed"] = rep_seed
                base_cfg["period_unit"] = period_unit
                base_cfg.update(_convert_model_overrides(config.get("config_overrides", {})))

                df_sim = bt.real_trading_simulation(
                    base_cfg,
                    models_to_test=[model_type],
                    months=months,
                )

                metrics_row = {"model": model_type, "rep": rep, "seed": rep_seed}
                equity_series = pd.Series(dtype=np.float64)
                buyhold_series = pd.Series(dtype=np.float64)
                bar_concat = getattr(bt, "bar_concat", None)
            if bar_concat is not None and not bar_concat.empty and "cstrategy_cont" in bar_concat.columns:
                equity_series = bar_concat["cstrategy_cont"].astype(np.float64)
                if "creturns_cont" in bar_concat.columns:
                    buyhold_series = bar_concat["creturns_cont"].astype(np.float64)

            trade_log = getattr(bt, "trade_log", None)

            if df_sim is not None and not df_sim.empty:
                metrics_row["total_trades"] = int(df_sim["trades"].sum()) if "trades" in df_sim else 0
                if not equity_series.empty and len(equity_series) > 1:
                    final_eq = float(equity_series.iloc[-1])
                    metrics_row["total_return_pct"] = float(final_eq - 1.0)
                    cum_max = np.maximum.accumulate(equity_series.values)
                    dd_arr = (equity_series.values - cum_max) / np.where(cum_max > 0, cum_max, 1.0)
                    max_dd = float(np.min(dd_arr))
                    metrics_row["max_drawdown"] = max_dd

                    n_bars = len(equity_series)
                    if n_bars > 1 and hasattr(equity_series.index, '__len__'):
                        try:
                            t0 = equity_series.index[0]
                            t1 = equity_series.index[-1]
                            if hasattr(t0, 'timestamp') and hasattr(t1, 'timestamp'):
                                years_span = max((t1.timestamp() - t0.timestamp()) / (365.25 * 86400), 1e-6)
                            else:
                                years_span = max(n_bars / 12096.0, 1e-6)
                        except Exception:
                            years_span = max(n_bars / 12096.0, 1e-6)
                    else:
                        years_span = max(n_bars / 12096.0, 1e-6)

                    cagr = (float(final_eq) ** (1.0 / years_span) - 1.0) if final_eq > 0 else 0.0
                    metrics_row["cagr"] = cagr
                    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
                    metrics_row["calmar_ratio"] = calmar

                    equity_points = []
                    for i, (ts, val) in enumerate(zip(equity_series.index, equity_series.values)):
                        t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else i
                        equity_points.append({"time": t, "value": round(float(val), 6)})
                    metrics_row["equity_curve"] = equity_points

                    if not buyhold_series.empty and len(buyhold_series) == len(equity_series):
                        bh_points = []
                        for i, (ts, val) in enumerate(zip(buyhold_series.index, buyhold_series.values)):
                            t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else i
                            bh_points.append({"time": t, "value": round(float(val), 6)})
                        metrics_row["buy_hold_curve"] = bh_points
                    else:
                        metrics_row["buy_hold_curve"] = []

                    dd_series = pd.Series(dd_arr, index=equity_series.index)
                    dd_points = []
                    for i, (ts, val) in enumerate(zip(dd_series.index, dd_series.values)):
                        t = int(ts.timestamp()) if hasattr(ts, 'timestamp') else i
                        dd_points.append({"time": t, "value": round(float(val), 6)})
                    metrics_row["drawdown_curve"] = dd_points
                else:
                    metrics_row.setdefault("equity_curve", [])
                    metrics_row.setdefault("buy_hold_curve", [])
                    metrics_row.setdefault("drawdown_curve", [])
                    metrics_row.setdefault("total_return_pct", 0.0)
                    metrics_row.setdefault("max_drawdown", 0.0)
                    metrics_row.setdefault("cagr", 0.0)
                    metrics_row.setdefault("calmar_ratio", 0.0)

                if "strategy_return" in df_sim.columns:
                    rets = df_sim["strategy_return"].dropna().values
                    if len(rets) > 2:
                        ann_ret = np.mean(rets) * 12
                        ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
                        metrics_row["sharpe"] = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
                        downside = rets[rets < 0]
                        if len(downside) > 1:
                            downside_vol = np.std(downside, ddof=1) * np.sqrt(12)
                            metrics_row["sortino"] = float(ann_ret / downside_vol) if downside_vol > 0 else 0.0
                        elif len(downside) == 1:
                            metrics_row["sortino"] = float(ann_ret / abs(downside[0])) if abs(downside[0]) > 1e-9 else 0.0
                        else:
                            metrics_row["sortino"] = 0.0
                    else:
                        metrics_row["sharpe"] = 0.0
                        metrics_row["sortino"] = 0.0
                    metrics_row["win_rate"] = float((rets > 0).mean()) if len(rets) > 0 else 0.0
                    gross_wins = float(np.sum(rets[rets > 0])) if np.any(rets > 0) else 0.0
                    gross_losses = abs(float(np.sum(rets[rets < 0]))) if np.any(rets < 0) else 0.0
                    metrics_row["profit_factor"] = gross_wins / gross_losses if gross_losses > 1e-9 else (float('inf') if gross_wins > 0 else 0.0)
                    if metrics_row["profit_factor"] is not None and np.isinf(float(metrics_row["profit_factor"])):
                        metrics_row["profit_factor"] = None
                    total_trades = metrics_row.get("total_trades", 0)
                    if total_trades > 0:
                        metrics_row["avg_trade"] = float(np.sum(rets) / total_trades)
                for col, key in [("active_rate", "active_rate"), ("directional_accuracy", "directional_accuracy"),
                                 ("precision_macro", "precision_macro"), ("f1_macro", "f1_macro")]:
                    if col in df_sim.columns:
                        vals = df_sim[col].dropna()
                        metrics_row[key] = float(vals.mean()) if not vals.empty else 0.0

                months_out = []
                for _, row in df_sim.iterrows():
                    month_label = str(row.get("test_start", ""))[:7] if pd.notna(row.get("test_start")) else ""
                    months_out.append({
                        "month": month_label,
                        "return_pct": float(row.get("strategy_return", 0)) if pd.notna(row.get("strategy_return")) else None,
                        "win_rate": float(row.get("win_rate", 0)) if pd.notna(row.get("win_rate")) else None,
                        "trades": int(row.get("trades", 0)) if pd.notna(row.get("trades")) else 0,
                        "sharpe": float(row.get("sharpe", 0)) if pd.notna(row.get("sharpe")) else None,
                        "max_drawdown": float(row.get("drawdown", 0)) if pd.notna(row.get("drawdown")) else None,
                        "active_rate": float(row.get("active_rate", 0)) if pd.notna(row.get("active_rate")) else None,
                    })
                metrics_row["monthly_results"] = months_out
            else:
                metrics_row["equity_curve"] = []
                metrics_row["buy_hold_curve"] = []
                metrics_row["drawdown_curve"] = []
                metrics_row["monthly_results"] = []
                metrics_row["total_return_pct"] = 0.0
                metrics_row["max_drawdown"] = 0.0
                metrics_row["sharpe"] = 0.0
                metrics_row["sortino"] = 0.0
                metrics_row["profit_factor"] = 0.0
                metrics_row["cagr"] = 0.0
                metrics_row["calmar_ratio"] = 0.0
                metrics_row["total_trades"] = 0
                metrics_row["win_rate"] = 0.0
                metrics_row["active_rate"] = 0.0

            if trade_log is not None and not trade_log.empty:
                safe_trades = trade_log.reset_index(drop=True)
                for col in safe_trades.columns:
                    if pd.api.types.is_datetime64_any_dtype(safe_trades[col]):
                        safe_trades[col] = safe_trades[col].astype(str)
                safe_trades = safe_trades.where(pd.notnull(safe_trades), None)

                safe_trades.rename(columns={
                    "entry_time": "entry_date",
                    "exit_time": "exit_date",
                    "side": "direction",
                    "pnl_pct": "return_pct",
                    "bars_held": "duration_bars",
                }, inplace=True)

                if "direction" in safe_trades.columns:
                    safe_trades["direction"] = safe_trades["direction"].map(
                        lambda x: "BUY" if x == "long" else ("SELL" if x == "short" else str(x))
                    )

                metrics_row["trades"] = json.loads(safe_trades.to_json(orient="records", date_format="iso"))
            else:
                metrics_row["trades"] = []

            metrics_row["hpo_param_importance"] = None
            metrics_row["hpo_trials"] = None
            metrics_row["best_study"] = None
            metrics_row["hpo_study_meta"] = None
            metrics_row["hpo_learning_summary"] = None
            metrics_row["hpo_sensitivity"] = None

            try:
                import glob as _glob
                results_base = os.path.join(project_root, "results")
                pattern = os.path.join(results_base, "**", "param_importances.json")
                found = _glob.glob(pattern, recursive=True)
                if found:
                    latest = max(found, key=os.path.getmtime)
                    with open(latest, "r") as _f:
                        imps = json.load(_f)
                    if isinstance(imps, dict):
                        metrics_row["hpo_param_importance"] = [
                            {"param": k, "importance": v} for k, v in imps.items()
                        ]
            except Exception:
                pass

            try:
                if hasattr(bt, "_optuna_study") and bt._optuna_study is not None:
                    study = bt._optuna_study
                    trials_data = []
                    for t in study.trials:
                        user_attrs = {}
                        try:
                            if hasattr(t, "user_attrs") and t.user_attrs:
                                for ka, va in t.user_attrs.items():
                                    if isinstance(va, (float, int, str, bool, type(None))):
                                        user_attrs[ka] = va
                                    elif isinstance(va, (np.floating,)):
                                        user_attrs[ka] = float(va)
                                    elif isinstance(va, (np.integer,)):
                                        user_attrs[ka] = int(va)
                                    elif isinstance(va, (list,)):
                                        user_attrs[ka] = [float(x) if isinstance(x, (np.floating,)) else
                                                          int(x) if isinstance(x, (np.integer,)) else x
                                                          for x in va[:20]]
                        except Exception:
                            user_attrs = {}
                        trials_data.append({
                            "trial_number": t.number,
                            "value": t.value if t.value is not None else float("nan"),
                            "params": dict(t.params) if t.params else {},
                            "state": str(t.state).split(".")[-1] if hasattr(t, "state") else None,
                            "duration_sec": t.duration.total_seconds() if hasattr(t, "duration") and t.duration else None,
                            "user_attrs": user_attrs,
                        })
                    if trials_data:
                        metrics_row["hpo_trials"] = _sanitize_metrics([trials_data])[0] if trials_data else None
                        metrics_row["hpo_trials"] = trials_data

                    best_trial_num = 0
                    best_value = None
                    best_params = {}
                    try:
                        if study.best_trial is not None:
                            best_trial_num = study.best_trial.number
                            best_value = float(study.best_trial.value) if study.best_trial.value is not None else None
                            best_params = dict(study.best_trial.params) if study.best_trial.params else {}
                    except Exception:
                        pass
                    metrics_row["best_study"] = {
                        "best_trial": best_trial_num,
                        "best_value": best_value,
                        "best_params": best_params,
                    }

                    n_completed = n_pruned = n_failed = 0
                    total_dur = 0.0
                    for t in study.trials:
                        st = str(t.state).split(".")[-1] if hasattr(t, "state") else ""
                        if st == "COMPLETE":
                            n_completed += 1
                        elif st == "PRUNED":
                            n_pruned += 1
                        else:
                            n_failed += 1
                        if hasattr(t, "duration") and t.duration:
                            total_dur += t.duration.total_seconds()
                    sampler_name = None
                    try:
                        sampler_name = study.sampler.__class__.__name__ if hasattr(study, "sampler") and study.sampler else None
                    except Exception:
                        pass
                    direction = None
                    try:
                        direction = str(study.direction).split(".")[-1] if hasattr(study, "direction") else None
                    except Exception:
                        pass
                    metrics_row["hpo_study_meta"] = {
                        "study_name": getattr(study, "study_name", None),
                        "direction": direction,
                        "n_trials": len(study.trials),
                        "n_completed": n_completed,
                        "n_pruned": n_pruned,
                        "n_failed": n_failed,
                        "sampler_type": sampler_name,
                        "total_duration_sec": round(total_dur, 2) if total_dur > 0 else None,
                    }
            except Exception:
                pass

            try:
                import glob as _glob_learn
                results_base = os.path.join(project_root, "results")
                pattern = os.path.join(results_base, "**", "learning_summary.json")
                found = _glob_learn.glob(pattern, recursive=True)
                if found:
                    latest = max(found, key=os.path.getmtime)
                    with open(latest, "r") as _f:
                        ls = json.load(_f)
                    cl_delta = ls.get("cliffs_delta_post_vs_startup", None)
                    if cl_delta is not None:
                        abs_d = abs(float(cl_delta))
                        if abs_d < 0.147:
                            interp = "negligible"
                        elif abs_d < 0.33:
                            interp = "small"
                        elif abs_d < 0.474:
                            interp = "medium"
                        else:
                            interp = "large"
                    else:
                        interp = None
                    startup_best = ls.get("startup_best")
                    post_best = ls.get("post_best")
                    if (startup_best is not None and post_best is not None and
                            startup_best and post_best and startup_best != 0 and np.isfinite(startup_best) and np.isfinite(post_best)):
                        uplift_pct = round(float((post_best - startup_best) / abs(startup_best)) * 100, 2)
                    else:
                        uplift_pct = None
                    metrics_row["hpo_learning_summary"] = {
                        "cliff_delta": cl_delta,
                        "delta_interpretation": interp,
                        "startup_median_score": ls.get("startup_median"),
                        "post_startup_median_score": ls.get("post_median"),
                        "share_beating_startup": ls.get("share_post_above_startup_median"),
                        "best_uplift_pct": uplift_pct,
                        "startup_trials": ls.get("n_startup_complete", 0),
                        "post_startup_trials": ls.get("n_post_complete", 0),
                    }
            except Exception:
                pass

            try:
                if metrics_row["hpo_trials"] and len(metrics_row["hpo_trials"]) >= 3:
                    trials = metrics_row["hpo_trials"]
                    numeric_params = set()
                    param_values = {}
                    for t in trials:
                        for pk, pv in (t.get("params") or {}).items():
                            if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                                numeric_params.add(pk)
                                param_values.setdefault(pk, []).append((t["trial_number"], float(pv), t.get("value")))
                    sensitivity = []
                    for pk in sorted(numeric_params):
                        pts = [(v, val) for _, v, val in param_values[pk] if val is not None and np.isfinite(val)]
                        if len(pts) < 3:
                            continue
                        try:
                            from scipy.stats import spearmanr
                            vals = [v for v, _ in pts]
                            objs = [o for _, o in pts]
                            if len(set(vals)) < 2 or len(set(objs)) < 2:
                                continue
                            rho, _ = spearmanr(vals, objs)
                        except Exception:
                            continue
                        best_v_set = set()
                        for _, v, val in param_values[pk]:
                            if best_params and pk in best_params and abs(v - float(best_params[pk])) < 1e-9:
                                best_v_set.add(val if val is not None and np.isfinite(val) else None)
                        std_best = float(np.std(list(x for x in best_v_set if x is not None))) if best_v_set else None
                        all_vals = sorted(set(vals))
                        range_pct = None
                        if len(all_vals) >= 2 and all_vals[-1] != all_vals[0]:
                            range_pct = float((max(all_vals) - min(all_vals)) / max(abs(all_vals[0]), 1e-9))
                        sensitivity.append({
                            "param": pk,
                            "index": round(float(rho), 4),
                            "std_at_best": round(std_best, 6) if std_best is not None else None,
                            "range_at_best": round(range_pct, 4) if range_pct is not None else None,
                            "perturbation_direction": "increasing" if rho > 0 else "decreasing",
                        })
                    if sensitivity:
                        metrics_row["hpo_sensitivity"] = sensitivity
            except Exception:
                pass

            try:
                from pipeline.metrics.overfitting import compute_overfitting_report, compute_period_breakdown
                wfo_records = getattr(bt, "_wfo_monthly_records", [])
                hpo_best = None
                if hasattr(bt, "_optuna_study") and bt._optuna_study is not None:
                    hpo_best = float(bt._optuna_study.best_value)
                overfit = compute_overfitting_report(wfo_records, model_type, hpo_best_value=hpo_best,
                                                     n_hpo_trials=n_trials_hdr)
                # S16.7: fANOVA interaction effects from Optuna study
                try:
                    if hasattr(bt, "_optuna_study") and bt._optuna_study is not None:
                        from pipeline.metrics.overfitting import compute_fanova_interactions
                        overfit.interaction_effects = compute_fanova_interactions(bt._optuna_study)
                except Exception:
                    pass
                metrics_row["overfitting"] = {
                    "overfit_score": overfit.overfit_score,
                    "risk_level": overfit.risk_level,
                    "risk_color": overfit.risk_color,
                    "train_oos_gap_pct": overfit.train_oos_gap_pct,
                    "temporal_degradation_pct": overfit.temporal_degradation_pct,
                    "sharpe_ci": overfit.sharpe_ci,
                    "return_ci": overfit.return_ci,
                    "maxdd_ci": overfit.maxdd_ci,
                    "cv_sharpe_mean": overfit.cv_sharpe_mean,
                    "cv_sharpe_std": overfit.cv_sharpe_std,
                    "cv_return_mean": overfit.cv_return_mean,
                    "cv_return_std": overfit.cv_return_std,
                    "min_trl_trades": overfit.min_trl_trades,
                    "sufficient_trades": overfit.sufficient_trades,
                    "n_periods": overfit.n_periods,
                    "n_signal_periods": overfit.n_signal_periods,
                    "signal_gap_pct": overfit.signal_gap_pct,
                    "is_mean_sharpe": overfit.is_mean_sharpe,
                    "oos_mean_sharpe": overfit.oos_mean_sharpe,
                    "dsr_min_sharpe": overfit.dsr_min_sharpe,
                    "psr": overfit.psr,
                    "dsr_value": overfit.dsr_value,
                    "interaction_effects": overfit.interaction_effects,
                }
                metrics_row["walkforward_periods"] = compute_period_breakdown(wfo_records)
            except Exception as _overfit_err:
                if settings.debug:
                    print(f"[overfitting] compute failed for {model_type}: {_overfit_err}")
                metrics_row["overfitting"] = None
                metrics_row["walkforward_periods"] = []

            # S16.4: Plain-English backtest summary
            try:
                from pipeline.metrics.summary_generator import generate_summary
                metrics_row["summary_text"] = generate_summary(metrics_row, config)
            except Exception:
                metrics_row["summary_text"] = None

            # S16.3: Training diagnostics (feature importance, confusion matrix, confidence bands)
            try:
                from pipeline.metrics.diagnostics import (
                    compute_feature_importance,
                    compute_prediction_histogram,
                    compute_confidence_bands,
                    aggregate_confusion_matrices,
                    TrainingDiagnosticsData,
                )
                _diag = {}
                # Feature importance from per-period capture
                _fi_tuples = getattr(bt, "_diagnostics_feature_importance", [])
                if _fi_tuples:
                    from pipeline.metrics.diagnostics import FeatureImportanceEntry
                    _diag["feature_importance"] = [
                        {"feature": f, "importance": float(i)} for f, i in _fi_tuples
                    ]
                else:
                    _diag["feature_importance"] = []

                # Importance method + feature family distribution
                try:
                    from pipeline.metrics.diagnostics import get_importance_method, classify_feature_families
                    _diag["importance_method"] = get_importance_method(model_type)
                    _feat_names = getattr(bt, "_diagnostics_feature_names", None)
                    if not _feat_names:
                        _feat_names = [f for f, _ in _fi_tuples] if _fi_tuples else []
                    _diag["feature_families"] = classify_feature_families(_feat_names) if _feat_names else {}
                except Exception:
                    _diag["importance_method"] = "unknown"
                    _diag["feature_families"] = {}

                # S16: VIF collinearity check for logistic models
                _diag["vif_warnings"] = []
                if model_type in ("logistic", "logit") and _fi_tuples:
                    try:
                        from pipeline.metrics.diagnostics import compute_vif
                        _vif_feats = [f for f, _ in _fi_tuples]
                        _res = getattr(bt, "results", None)
                        if _res is not None and hasattr(_res, "columns") and _vif_feats:
                            _cols = [c for c in _vif_feats if c in _res.columns]
                            if _cols:
                                _X_vif = np.asarray(_res[_cols].dropna()[:500], dtype=np.float64)
                                _vif = compute_vif(_X_vif)
                                for i, v in zip(_cols, _vif):
                                    if np.isfinite(v) and v > 10:
                                        _diag["vif_warnings"].append({"feature": i, "vif": round(float(v), 1)})
                    except Exception:
                        pass

                # Confusion matrix from results attrs
                _cm = None
                try:
                    _res = getattr(bt, "results", None)
                    if _res is not None and hasattr(_res, "attrs") and "confusion_matrix" in _res.attrs:
                        _cm_raw = _res.attrs["confusion_matrix"]
                        _cm = aggregate_confusion_matrices([_cm_raw]).matrix if _cm_raw is not None else None
                except Exception:
                    pass
                _diag["confusion_matrix"] = {
                    "matrix": _cm,
                    "labels": ["Short", "Flat", "Long"],
                } if _cm is not None else None

                # Prediction histogram & confidence bands from max_conf
                _conf = getattr(bt, "_last_conf_stats_max_conf", None)
                _res = getattr(bt, "results", None)
                if _conf is not None and len(_conf) > 0 and _res is not None and hasattr(_res, "columns"):
                    _hist = compute_prediction_histogram([_conf])
                    _diag["prediction_histogram"] = [
                        {"bin_start": b.bin_start, "bin_end": b.bin_end, "bin_center": b.bin_center, "count": b.count}
                        for b in _hist
                    ]
                    # Confidence bands: need outcome_arrays and return_arrays
                    _outcome_arr = None
                    _ret_arr = None
                    try:
                        if "pred" in _res.columns and "true_direction" in _res.columns:
                            _pred_dir = _res["pred"].fillna(0).values
                            _true_dir = _res["true_direction"].fillna(0).values
                            _outcome_arr = ((_pred_dir * _true_dir) > 0).astype(float)
                        if "returns" in _res.columns:
                            _ret_arr = _res["returns"].fillna(0).values
                    except Exception:
                        pass
                    if _outcome_arr is not None and _ret_arr is not None:
                        _min_len = min(len(_conf), len(_outcome_arr), len(_ret_arr))
                        _bands = compute_confidence_bands(
                            [_conf[:_min_len]],
                            [_outcome_arr[:_min_len]],
                            [_ret_arr[:_min_len]],
                        )
                        _diag["confidence_bands"] = [
                            {"band_min": b.band_min, "band_max": b.band_max, "count": b.count,
                             "accuracy": b.accuracy, "mean_return": b.mean_return}
                            for b in _bands
                        ]
                    else:
                        _diag["confidence_bands"] = []
                else:
                    _diag["prediction_histogram"] = []
                    _diag["confidence_bands"] = []

                metrics_row["diagnostics"] = _diag
            except Exception as _diag_err:
                if _dbg:
                    print(f"[diagnostics] compute failed for {model_type}: {_diag_err}")
                metrics_row["diagnostics"] = None

            all_metrics.append(metrics_row)

            try:
                from api.shutdown import wal_checkpoint_periodic
                wal_checkpoint_periodic(settings.db_full_path)
            except Exception:
                pass

            _pub("model_training", job_id, {"model": model_type, "status": "complete", "metrics": metrics_row})

            # Persist model snapshot to disk during execution (silent — model
            # object will be freed after the backtest finishes). Only visible
            # when the user explicitly clicks "Save Model" on the Results page.
            _snap_path = None
            try:
                from pipeline.models.model_persistence import save_snapshot
                _model_obj = getattr(bt, "_last_trained_model", None) or getattr(bt, "model", None)
                if _model_obj is not None:
                    _cov_thr = getattr(bt, "_coverage_conf_thr", None) or getattr(bt, "_deep_coverage_thr", None)
                    _feat_names = getattr(bt, "_diagnostics_feature_names", None) or []
                    _fc = getattr(bt, "features_config", None) or {}
                    _cal_method = _fc.get("calibrate_method", "sigmoid") if _fc else "sigmoid"
                    _snap_path = save_snapshot(
                        model=_model_obj,
                        model_type=model_type,
                        best_params=_fc,
                        coverage_conf_thr=float(_cov_thr) if _cov_thr is not None else None,
                        feature_names=list(_feat_names) if _feat_names else None,
                        features_config=_fc,
                        calibrate_method=str(_cal_method),
                        train_start=str(start)[:10] if start else None,
                        train_end=str(end)[:10] if end else None,
                        seed=int(rep_seed) if rep_seed is not None else None,
                        pip_freeze=pip_freeze,
                        parent_job_id=parent_job_id,
                        metrics=metrics_row,
                    )
            except Exception:
                pass

            metrics_row["snapshot_path"] = _snap_path

            bt.free(release_data=True)
            del bt

        result = {
            "pair": pair,
            "models": models,
            "config": config,
            "metrics": _sanitize_metrics(all_metrics),
        }

        jm.update_status(job_id, "completed", result=result)
        _pub("job_complete", job_id, {"metrics": all_metrics})
        emit_event("job_complete", job_id=job_id, n_models=len(all_metrics))
        return result

    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            _JOB_TASK_IDS.pop(job_id, None)
            jm.update_status(job_id, "failed", error="Force stopped by user")
            _pub("job_failed", job_id, {"error": "Stopped by user"})
            emit_event("job_failed", job_id=job_id, error="Stopped by user")
            from api.process_cleanup import cleanup_job as _cleanup_task_job
            _cleanup_task_job(job_id)
            return
        tb = traceback.format_exc()
        jm.update_status(job_id, "failed", error=f"{e}\n{tb}")
        _pub("job_failed", job_id, {"error": str(e)})
        emit_event("job_failed", job_id=job_id, error=str(e)[:200])
        from api.process_cleanup import cleanup_job as _cleanup_task_job
        _cleanup_task_job(job_id)
        raise
    except BaseException:
        _JOB_TASK_IDS.pop(job_id, None)
        jm.update_status(job_id, "failed", error="Task killed or interrupted")
        _pub("job_failed", job_id, {"error": "Task killed or interrupted"})
        emit_event("job_failed", job_id=job_id, error="Task killed or interrupted")
        from api.process_cleanup import cleanup_job as _cleanup_task_job
        _cleanup_task_job(job_id)
        raise
    finally:
        _JOB_TASK_IDS.pop(job_id, None)
        _cancellation_events.pop(job_id, None)
        from api.process_cleanup import cleanup_job as _cleanup_final
        _cleanup_final(job_id)


def _download_data_impl(job_id: str, pair: str, years: int = 10, base_timeframe: str = "M30"):
    """Core download logic -- executed by Celery worker or in-process (desktop mode)."""
    project_root = settings.project_root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)

    from api.services import JobManager
    from pipeline.data.data_sqlite import DataStore
    from pipeline.data.pair_config import get_pair_config, PAIR_REGISTRY

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)

    jm.update_status(job_id, "running")
    _pub("download_started", job_id, {"pair": pair})

    try:
        from pipeline.data.data_downloader import download_pair

        try:
            cfg = get_pair_config(pair)
        except ValueError:
            db_pair = store.get_pair(pair)
            if db_pair is None:
                raise
            class _PairCfg:
                symbol = db_pair["symbol"]
                oanda_name = db_pair["oanda_name"]
                pip_value = db_pair["pip_value"]
                lot_size = db_pair.get("lot_size", 100000.0)
                base_currency = db_pair.get("base_currency", pair[:3])
                quote_currency = db_pair.get("quote_currency", pair[3:])
                typical_spread_bps = db_pair.get("typical_spread_bps", 1.0)
            cfg = _PairCfg()

        store.insert_pairs([{
            "symbol": cfg.symbol,
            "oanda_name": cfg.oanda_name,
            "pip_value": cfg.pip_value,
            "lot_size": cfg.lot_size,
            "base_currency": cfg.base_currency,
            "quote_currency": cfg.quote_currency,
            "typical_spread_bps": cfg.typical_spread_bps,
        }])

        saved = download_pair(
            instrument=cfg.oanda_name,
            store=store,
            years=years,
            pair_symbol=cfg.symbol,
            base_timeframe=base_timeframe,
        )

        jm.update_status(job_id, "completed", result={"pair": pair, "granularities": saved})
        _pub("download_complete", job_id, {"pair": pair})
        return {"pair": pair, "granularities": saved}

    except Exception as e:
        jm.update_status(job_id, "failed", error=str(e))
        _pub("download_failed", job_id, {"error": str(e)})
        raise


def _run_forward_test_impl(job_id: str, config: dict):
    """Execute a forward test: load saved model, run prediction-only on date range."""
    project_root = settings.project_root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    from api.services import JobManager
    from pipeline.data.data_sqlite import DataStore

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)

    jm.update_status(job_id, "running")
    _pub("job_started", job_id, {"type": "forward_test"})

    try:
        from pipeline.forward_test import run_forward_test
        from pipeline.models.model_registry_disk import get_all_deployed

        model_id = config.get("model_id", "")
        snapshot_path = config.get("snapshot_path", "")
        pair = config.get("pair", "EURUSD")
        timeframe = config.get("timeframe", "H1")
        start_date = config.get("start_date", "")
        end_date = config.get("end_date", "")
        position_sizing = config.get("position_sizing", "fixed")
        trading_costs = config.get("trading_costs", True)

        if not snapshot_path or not os.path.isdir(snapshot_path):
            # Try to resolve from deployed models
            rows = get_all_deployed(settings.db_full_path)
            for r in rows:
                if r.get("id") == model_id:
                    snapshot_path = r.get("snapshot_path", "")
                    break

        if not snapshot_path or not os.path.isdir(snapshot_path):
            raise FileNotFoundError(f"Snapshot path not found for model {model_id}")

        result = run_forward_test(
            snapshot_path=snapshot_path,
            pair=pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            position_sizing=position_sizing,
            trading_costs=trading_costs,
        )

        jm.update_status(job_id, "completed", result={
            "pair": pair,
            "models": [result.get("model_type", "unknown")],
            "config": config,
            "metrics": [{
                "model": result.get("model_type", "unknown"),
                "sharpe": result["metrics"]["sharpe"],
                "total_return_pct": result["metrics"]["total_return_pct"],
                "win_rate": result["metrics"]["win_rate"],
                "max_drawdown_pct": result["metrics"]["max_drawdown_pct"],
                "total_trades": result["metrics"]["total_trades"],
                "diagnostics": result.get("diagnostics"),
            }],
            "forward_test": result,
        })
        _pub("job_complete", job_id, {"metrics": result["metrics"]})
        emit_event("job_complete", job_id=job_id, n_models=1)
        return result

    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            jm.update_status(job_id, "failed", error="Force stopped by user")
            _pub("job_failed", job_id, {"error": "Stopped by user"})
            emit_event("job_failed", job_id=job_id, error="Stopped by user")
            return
        tb = traceback.format_exc()
        jm.update_status(job_id, "failed", error=f"{e}\n{tb}")
        _pub("job_failed", job_id, {"error": str(e)})
        emit_event("job_failed", job_id=job_id, error=str(e)[:200])
        raise


# --- Public API: dispatch to Celery or in-process depending on availability ---

def _prefetch_news_impl():
    """Background task: pre-fetch RSS articles to keep cache warm."""
    import logging
    _news_log = logging.getLogger(__name__)
    try:
        from news.scraper import NewsScraper
        scraper = NewsScraper()
        articles = scraper.fetch_all()
        _news_log.info("News prefetch: %d articles cached", len(articles))
        return {"status": "ok", "articles": len(articles)}
    except Exception as exc:
        _news_log.warning("News prefetch failed: %s", exc)
        return {"status": "error", "reason": str(exc)[:200]}


def _prefetch_calendar_impl():
    """Background task: refresh economic calendar cache daily."""
    import logging
    _cal_log = logging.getLogger(__name__)
    try:
        from news.scraper import NewsScraper
        events = NewsScraper.fetch_calendar_live()
        _cal_log.info("Calendar prefetch: %d events cached", len(events))
        return {"status": "ok", "events": len(events)}
    except Exception as exc:
        _cal_log.warning("Calendar prefetch failed: %s", exc)
        return {"status": "error", "reason": str(exc)[:200]}


if _celery_available and celery_app is not None:
    run_backtest_task = celery_app.task(name="run_backtest")(_run_backtest_impl)
    download_data_task = celery_app.task(name="download_data")(_download_data_impl)
    run_forward_test_task = celery_app.task(name="run_forward_test")(_run_forward_test_impl)
    prefetch_news_task = celery_app.task(name="prefetch_news")(_prefetch_news_impl)
    prefetch_calendar_task = celery_app.task(name="prefetch_calendar")(_prefetch_calendar_impl)
else:
    class _SyncTask:
        """Celery-compatible task interface for synchronous (desktop) execution."""

        def delay(self, *args, **kwargs):
            self._func(*args, **kwargs)
            return _SyncResult()

        def apply_async(self, *args, **kwargs):
            self._func(*args[0] if args else [], **kwargs)
            return _SyncResult()

    class _SyncResult:
        """Mimics AsyncResult for synchronous tasks."""
        @property
        def id(self):
            return str(uuid.uuid4())

    class RunBacktestSync(_SyncTask):
        _func = staticmethod(_run_backtest_impl)

    class DownloadDataSync(_SyncTask):
        _func = staticmethod(_download_data_impl)

    class PrefetchNewsSync(_SyncTask):
        _func = staticmethod(_prefetch_news_impl)

    class PrefetchCalendarSync(_SyncTask):
        _func = staticmethod(_prefetch_calendar_impl)

    run_backtest_task = RunBacktestSync()
    download_data_task = DownloadDataSync()
    prefetch_news_task = PrefetchNewsSync()
    prefetch_calendar_task = PrefetchCalendarSync()
