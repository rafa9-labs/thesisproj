"""Celery application and task definitions."""
from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from celery import Celery

from api.config import settings

TRIAL_COUNTS = {
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

CV_BLOCKS_DEFAULT = 5


def _compute_total_work(models: list[str], months: int, cv_blocks: int = CV_BLOCKS_DEFAULT) -> int:
    total = 0
    for m in models:
        tc = TRIAL_COUNTS.get(m, {"random": 3, "bayes": 3})
        n_trials = tc.get("random", 3) + tc.get("bayes", 3)
        n_trials = max(n_trials, 10)
        hpo_work = n_trials * cv_blocks
        sim_work = months
        total += hpo_work + sim_work
    return max(total, 1)

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
)


def _pub(event: str, job_id: str, data: Dict[str, Any]):
    """Publish a progress event to Redis pub/sub."""
    try:
        import redis as _redis

        r = _redis.from_url(settings.redis_url)
        msg = json.dumps({"event": event, "job_id": job_id, **data})
        r.publish(f"job:{job_id}", msg)
    except Exception:
        pass


def _sanitize_metrics(metrics_list: list) -> list:
    """Ensure all metric values are JSON-safe (no NaN, no Timestamp, no numpy types)."""
    safe = []
    for row in metrics_list:
        out = {}
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                out[k] = None
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                out[k] = bool(v)
            else:
                out[k] = v
        safe.append(out)
    return safe


@celery_app.task(bind=True, name="run_backtest")
def run_backtest_task(self, job_id: str, config: Dict[str, Any]):
    """Execute a backtest pipeline run inside a Celery worker."""
    project_root = settings.project_root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)

    from api.services import JobManager
    from pipeline.data_sqlite import DataStore

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)

    pair = config.get("pair", "EURUSD")
    models = config.get("models", ["logistic"])
    start = config.get("start_date") or None
    end = config.get("end_date") or None
    months = config.get("months", 3)
    repeats = config.get("repeats", 1)
    trading_costs = config.get("trading_costs", True)

    if end is None:
        from datetime import date
        today = date.today()
        first_of_current = today.replace(day=1)
        last_of_prev = first_of_current - pd.Timedelta(days=1)
        end = last_of_prev.strftime("%Y-%m-%d")

    try:
        from pipeline.data_sqlite import DataStore as _DS
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

    total_work = _compute_total_work(models, months)
    jm.update_status(job_id, "running")
    _pub("job_started", job_id, {"pair": pair, "models": models, "total_work": total_work})

    completed_work = 0

    def _progress_cb(phase: str, model: str, detail: dict | None = None):
        nonlocal completed_work
        data = {"model": model, "phase": phase}
        if detail:
            data.update(detail)
        if phase == "hpo_trial":
            cv_b = detail.get("cv_blocks", CV_BLOCKS_DEFAULT) if detail else CV_BLOCKS_DEFAULT
            completed_work += cv_b
        elif phase == "month":
            completed_work += 1
        pct = min(round((completed_work / total_work) * 100, 1), 100) if total_work > 0 else 0
        data["completed_work"] = completed_work
        data["total_work"] = total_work
        data["progress_pct"] = pct
        evt_name = f"{phase}_progress" if phase in ("hpo", "month") else "model_phase"
        _pub(evt_name, job_id, data)

    all_metrics = []

    try:
        for model_type in models:
            _pub("model_training", job_id, {"model": model_type, "status": "starting"})
            _pub("model_phase", job_id, {"model": model_type, "phase": "hpo", "total_work": total_work})

            from copy import deepcopy
            from pipeline.metrics_tuples import CLASS_DEFAULTS
            from pipeline.backtester.composed import MLBacktester

            feat_cfg = deepcopy(CLASS_DEFAULTS["features"])
            feat_cfg.update(config.get("config_overrides", {}))

            tc = TRIAL_COUNTS.get(model_type, {"random": 3, "bayes": 3})
            n_trials_hdr = max(tc.get("random", 3) + tc.get("bayes", 3), 10)

            bt = MLBacktester(
                symbol=pair,
                start=start,
                end=end,
                trading_costs=trading_costs,
                model_type=model_type,
                features_config=feat_cfg,
            )
            bt._progress_callback = _progress_cb

            base_cfg = deepcopy(CLASS_DEFAULTS["features"])
            base_cfg.update(deepcopy(CLASS_DEFAULTS["cv"]))
            base_cfg["model_type"] = model_type
            base_cfg["rep"] = 1
            base_cfg["trading_costs"] = trading_costs
            base_cfg["n_trials"] = n_trials_hdr
            base_cfg["n_startup_trials"] = tc.get("random", 3)
            base_cfg.update(config.get("config_overrides", {}))

            df_sim = bt.real_trading_simulation(
                base_cfg,
                models_to_test=[model_type],
                months=months,
            )

            metrics_row = {"model": model_type}
            equity_series = pd.Series(dtype=np.float64)
            bar_concat = getattr(bt, "bar_concat", None)
            if bar_concat is not None and not bar_concat.empty and "cstrategy_cont" in bar_concat.columns:
                equity_series = bar_concat["cstrategy_cont"].astype(np.float64)

            if df_sim is not None and not df_sim.empty:
                metrics_row["total_trades"] = int(df_sim["trades"].sum()) if "trades" in df_sim else 0
                if not equity_series.empty and len(equity_series) > 1:
                    final_eq = float(equity_series.iloc[-1])
                    metrics_row["total_return"] = final_eq
                    metrics_row["total_return_pct"] = (final_eq - 1.0) * 100.0
                    cum_max = np.maximum.accumulate(equity_series.values)
                    dd_arr = (equity_series.values - cum_max) / np.where(cum_max > 0, cum_max, 1.0)
                    metrics_row["max_drawdown"] = float(np.min(dd_arr))
                if "strategy_return" in df_sim.columns:
                    rets = df_sim["strategy_return"].dropna().values
                    if len(rets) > 2:
                        ann_ret = np.mean(rets) * 12
                        ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
                        metrics_row["sharpe"] = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
                    metrics_row["win_rate"] = float((rets > 0).mean()) if len(rets) > 0 else 0.0
                for col, key in [("active_rate", "active_rate"), ("directional_accuracy", "directional_accuracy"),
                                 ("precision_macro", "precision_macro"), ("f1_macro", "f1_macro")]:
                    if col in df_sim.columns:
                        vals = df_sim[col].dropna()
                        metrics_row[key] = float(vals.mean()) if not vals.empty else 0.0

            metrics_row["equity_curve"] = equity_series.tolist() if not equity_series.empty else []

            if df_sim is not None and not df_sim.empty:
                safe_df = df_sim.reset_index()
                for col in safe_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
                        safe_df[col] = safe_df[col].astype(str)
                safe_df = safe_df.where(pd.notnull(safe_df), None)
                metrics_row["monthly_df"] = json.loads(safe_df.to_json(orient="records", date_format="iso"))
            else:
                metrics_row["monthly_df"] = []

            all_metrics.append(metrics_row)
            _pub("model_training", job_id, {"model": model_type, "status": "complete", "metrics": metrics_row})

            bt.free(release_data=True)
            del bt

        result = {
            "pair": pair,
            "models": models,
            "metrics": _sanitize_metrics(all_metrics),
        }

        jm.update_status(job_id, "completed", result=result)
        _pub("job_complete", job_id, {"metrics": all_metrics})
        return result

    except Exception as e:
        tb = traceback.format_exc()
        jm.update_status(job_id, "failed", error=f"{e}\n{tb}")
        _pub("job_failed", job_id, {"error": str(e)})
        raise


@celery_app.task(bind=True, name="download_data")
def download_data_task(self, job_id: str, pair: str, years: int = 10):
    """Download pair data from OANDA and insert into SQLite."""
    project_root = settings.project_root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)

    from api.services import JobManager
    from pipeline.data_sqlite import DataStore

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)

    jm.update_status(job_id, "running")
    _pub("download_started", job_id, {"pair": pair})

    try:
        from pipeline.data_downloader import download_pair
        from pipeline.pair_config import get_pair_config

        cfg = get_pair_config(pair)
        saved = download_pair(
            instrument=cfg.oanda_name,
            granularities=["M30", "H1", "H4"],
            years=years,
            output_dir=settings.csv_data_dir,
        )

        from pipeline.data_migrator import migrate_pair
        for gran in ["M30", "H1", "H4"]:
            csv_key = f"{gran}"
            if csv_key in saved:
                migrate_pair(store, saved[csv_key], pair, gran, force=True)

        jm.update_status(job_id, "completed", result={"pair": pair, "files": list(saved.keys())})
        _pub("download_complete", job_id, {"pair": pair})
        return {"pair": pair, "files": list(saved.keys())}

    except Exception as e:
        jm.update_status(job_id, "failed", error=str(e))
        _pub("download_failed", job_id, {"error": str(e)})
        raise
