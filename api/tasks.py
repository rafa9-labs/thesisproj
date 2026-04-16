"""Celery application and task definitions."""
from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict

from celery import Celery

from api.config import settings

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
    start = config.get("start_date", "2024-01-01")
    end = config.get("end_date", "2024-12-01")
    months = config.get("months", 3)
    repeats = config.get("repeats", 1)
    trading_costs = config.get("trading_costs", True)

    jm.update_status(job_id, "running")
    _pub("job_started", job_id, {"pair": pair, "models": models})

    all_metrics = []

    try:
        for model_type in models:
            _pub("model_training", job_id, {"model": model_type, "status": "starting"})

            from copy import deepcopy
            from pipeline.metrics_tuples import CLASS_DEFAULTS
            from pipeline.backtester.composed import MLBacktester

            feat_cfg = deepcopy(CLASS_DEFAULTS["features"])
            feat_cfg.update(config.get("config_overrides", {}))

            bt = MLBacktester(
                symbol=pair,
                start=start,
                end=end,
                trading_costs=trading_costs,
                model_type=model_type,
                features_config=feat_cfg,
            )

            base_cfg = deepcopy(CLASS_DEFAULTS["backtest"])
            base_cfg.update(config.get("config_overrides", {}))

            df_sim = bt.real_trading_simulation(
                base_cfg,
                models_to_test=[model_type],
                months=months,
            )

            metrics_row = {}
            if df_sim is not None and not df_sim.empty:
                metrics_row = {
                    "model": model_type,
                    "total_return": float(df_sim["cstrategy_cont"].sum()) if "cstrategy_cont" in df_sim else None,
                    "total_trades": int(df_sim["trades"].sum()) if "trades" in df_sim else 0,
                }
                if "cstrategy_cont" in df_sim:
                    rets = df_sim["cstrategy_cont"]
                    if rets.std() > 0:
                        metrics_row["sharpe"] = float(rets.mean() / rets.std() * (252 * 48) ** 0.5)
                    cum = (1 + rets).cumprod()
                    dd = (cum / cum.cummax() - 1).min()
                    metrics_row["max_drawdown"] = float(dd)
                    wins = (rets > 0).sum()
                    total = (rets != 0).sum()
                    metrics_row["win_rate"] = float(wins / total) if total > 0 else 0.0

            all_metrics.append(metrics_row)
            _pub("model_training", job_id, {"model": model_type, "status": "complete", "metrics": metrics_row})

            bt.free(release_data=True)
            del bt

        result = {
            "pair": pair,
            "models": models,
            "metrics": all_metrics,
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
