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
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from api.config import settings

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
from logging_config import emit_event

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


def _compute_total_work(models: list[str], months: int, hpo_intensity: str | None = None, cv_blocks: int = CV_BLOCKS_DEFAULT, period_unit: str = "months") -> int:
    from config import convert_month_count_to_periods
    n_periods = convert_month_count_to_periods(months, period_unit)
    total = 0
    for m in models:
        tc = _get_trial_counts(hpo_intensity, m)
        n_trials = tc.get("random", 3) + tc.get("bayes", 3)
        n_trials = max(n_trials, 10)
        hpo_work = n_trials * cv_blocks
        sim_work = n_periods
        total += hpo_work + sim_work
    return max(total, 1)

celery_app = None
_celery_available = False

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
        )
        _celery_available = True
    except Exception:
        _celery_available = False


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
    with _job_events_lock:
        if job_id not in _job_events:
            _job_events[job_id] = []
        lst = _job_events[job_id]
        lst.append(event)
        if len(lst) > _JOB_EVENTS_MAX:
            _job_events[job_id] = lst[-_JOB_EVENTS_MAX:]
    # Push to Redis list so Celery worker events are visible to the API server
    try:
        r = _get_redis()
        r.rpush(f"job_events:{job_id}", json.dumps(event))
        r.ltrim(f"job_events:{job_id}", -_JOB_EVENTS_MAX, -1)
        r.expire(f"job_events:{job_id}", 86400)
    except Exception:
        global _redis_client, _redis_available
        _redis_client = None
        _redis_available = False


def get_job_events(job_id: str, after: int = 0) -> List[Dict]:
    # Prefer Redis so worker events are visible cross-process
    try:
        r = _get_redis()
        raw = r.lrange(f"job_events:{job_id}", 0, -1)
        if raw:
            events = [json.loads(item) for item in raw]
            return events[after:]
    except Exception:
        global _redis_client, _redis_available
        _redis_client = None
        _redis_available = False
    # Fallback to in-memory (desktop mode / Redis unavailable)
    with _job_events_lock:
        lst = _job_events.get(job_id, [])
        return lst[after:]


def clear_job_events(job_id: str):
    with _job_events_lock:
        _job_events.pop(job_id, None)
    try:
        r = _get_redis()
        r.delete(f"job_events:{job_id}")
    except Exception:
        global _redis_client, _redis_available
        _redis_client = None
        _redis_available = False


def _pub(event: str, job_id: str, data: Dict[str, Any]):
    """Publish a progress event to Redis pub/sub and in-memory buffer."""
    msg = {"event": event, "job_id": job_id, **data}
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


def _run_backtest_impl(job_id: str, config: Dict[str, Any]):
    """Core backtest logic -- executed by Celery worker or in-process (desktop mode)."""
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
    seed = config.get("seed", 42)
    hpo_intensity = config.get("hpo_intensity", "quick")
    trading_costs = config.get("trading_costs", True)
    period_unit = config.get("period_unit", "months")

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

    total_work = _compute_total_work(models, months, hpo_intensity, period_unit=period_unit)
    jm.update_status(job_id, "running")
    _pub("job_started", job_id, {"pair": pair, "models": models, "total_work": total_work})
    emit_event("job_started", job_id=job_id, pair=pair, models=",".join(models), total_work=total_work)

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
        elif phase == "period":
            completed_work += 1
        pct = min(round((completed_work / total_work) * 100, 1), 100) if total_work > 0 else 0
        data["completed_work"] = completed_work
        data["total_work"] = total_work
        data["progress_pct"] = pct
        if phase in ("hpo", "hpo_trial"):
            evt_name = "hpo_progress"
        elif phase in ("month", "period"):
            evt_name = "month_progress"
        else:
            evt_name = "model_phase"
        _pub(evt_name, job_id, data)
        emit_event(evt_name, job_id=job_id, pct=pct, **data)

        if phase == "hpo_trial" and detail:
            _pub("hpo_trial_result", job_id, {
                "model": model,
                "trial_number": detail.get("trial", 0),
                "score": detail.get("score"),
                "params": detail.get("params", {}),
                "best_score_so_far": detail.get("best_score_so_far"),
                "trial_state": detail.get("trial_state", "COMPLETE"),
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
            })

    all_metrics = []

    try:
        for model_type in models:
            _pub("model_training", job_id, {"model": model_type, "status": "starting"})
            _pub("model_phase", job_id, {"model": model_type, "phase": "hpo", "total_work": total_work})

            from copy import deepcopy
            from pipeline.metrics_tuples import CLASS_DEFAULTS
            from pipeline.backtester.composed import MLBacktester

            feat_cfg = deepcopy(CLASS_DEFAULTS["features"])
            feat_cfg.update(_convert_model_overrides(config.get("config_overrides", {})))

            tc = _get_trial_counts(hpo_intensity, model_type)
            n_trials_hdr = max(tc.get("random", 3) + tc.get("bayes", 3), 10)

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

            # --- Inject news & sentiment data if enabled ---
            if feat_cfg.get("use_news", True):
                try:
                    from news.scraper import NewsScraper
                    from news.sentiment import SentimentAnalyzer

                    scraper = NewsScraper()
                    articles = scraper.fetch_all()
                    pair_val = pair or "EURUSD"
                    filtered = NewsScraper.filter_by_pair(articles, pair_val)
                    backend = feat_cfg.get("news_sentiment_backend", "vader")
                    analyzer = SentimentAnalyzer(backend=backend)
                    scored = analyzer.score_articles(filtered)
                    news_aggregated = analyzer.aggregate_to_df(scored, freq="1h")
                    econ_events = scraper.economic_calendar_events()
                    bt._news_aggregated = news_aggregated
                    bt._news_economic_events = econ_events
                    _pub("news_loaded", job_id, {"articles": len(filtered), "backend": backend})
                except Exception as _news_err:
                    _pub("news_skip", job_id, {"reason": str(_news_err)[:200]})

            # --- Inject LLM sentiment data if enabled ---
            if feat_cfg.get("llm_sentiment_enabled", True):
                try:
                    from pipeline.llm.sentiment import LLMSentimentEngine

                    llm_engine = LLMSentimentEngine(config=feat_cfg)
                    llm_articles = getattr(bt, "_news_raw_articles", [])
                    if not llm_articles:
                        try:
                            from news.scraper import NewsScraper
                            scraper = NewsScraper()
                            llm_articles = scraper.fetch_all()
                        except Exception:
                            pass

                    pair_val = pair or "EURUSD"
                    llm_filtered = NewsScraper.filter_by_pair(llm_articles, pair_val)
                    scored_llm = llm_engine.score_articles(llm_filtered, pair=pair_val)
                    llm_aggregated = llm_engine.aggregate_to_df(scored_llm, freq="1h")
                    bt._llm_aggregated = llm_aggregated
                    _pub("llm_loaded", job_id, {
                        "articles": len(llm_filtered),
                        "backend": feat_cfg.get("llm_backend", "ollama"),
                    })
                    llm_engine.close()
                except Exception as _llm_err:
                    _pub("llm_skip", job_id, {"reason": str(_llm_err)[:200]})

            base_cfg = deepcopy(CLASS_DEFAULTS["features"])
            base_cfg.update(deepcopy(CLASS_DEFAULTS["cv"]))
            base_cfg["model_type"] = model_type
            base_cfg["rep"] = 1
            base_cfg["trading_costs"] = trading_costs
            base_cfg["n_trials"] = n_trials_hdr
            base_cfg["n_startup_trials"] = tc.get("random", 3)
            base_cfg["seed"] = seed
            base_cfg["period_unit"] = period_unit
            base_cfg.update(_convert_model_overrides(config.get("config_overrides", {})))

            df_sim = bt.real_trading_simulation(
                base_cfg,
                models_to_test=[model_type],
                months=months,
            )

            metrics_row = {"model": model_type}
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
                    metrics_row["win_rate"] = float((rets > 0).mean()) if len(rets) > 0 else 0.0
                    gross_wins = float(np.sum(rets[rets > 0])) if np.any(rets > 0) else 0.0
                    gross_losses = abs(float(np.sum(rets[rets < 0]))) if np.any(rets < 0) else 0.0
                    metrics_row["profit_factor"] = gross_wins / gross_losses if gross_losses > 1e-9 else (float('inf') if gross_wins > 0 else 0.0)
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

            if trade_log is not None and not trade_log.empty:
                safe_trades = trade_log.reset_index(drop=True)
                for col in safe_trades.columns:
                    if pd.api.types.is_datetime64_any_dtype(safe_trades[col]):
                        safe_trades[col] = safe_trades[col].astype(str)
                safe_trades = safe_trades.where(pd.notnull(safe_trades), None)
                metrics_row["trades"] = json.loads(safe_trades.to_json(orient="records", date_format="iso"))
            else:
                metrics_row["trades"] = []

            metrics_row["hpo_param_importance"] = None
            metrics_row["hpo_trials"] = None

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
                if hasattr(bt, "_optuna_study"):
                    study = bt._optuna_study
                    if study is not None:
                        trials_data = []
                        for t in study.trials:
                            trials_data.append({
                                "trial_number": t.number,
                                "value": t.value if t.value is not None else float("nan"),
                                "params": dict(t.params) if t.params else {},
                            })
                        if trials_data:
                            metrics_row["hpo_trials"] = _sanitize_metrics([trials_data])[0] if trials_data else None
                            metrics_row["hpo_trials"] = trials_data
            except Exception:
                pass

            try:
                from pipeline.overfitting import compute_overfitting_report, compute_period_breakdown
                wfo_records = getattr(bt, "_wfo_monthly_records", [])
                hpo_best = None
                if hasattr(bt, "_optuna_study") and bt._optuna_study is not None:
                    hpo_best = float(bt._optuna_study.best_value)
                overfit = compute_overfitting_report(wfo_records, model_type, hpo_best_value=hpo_best)
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
                }
                metrics_row["walkforward_periods"] = compute_period_breakdown(wfo_records)
            except Exception as _overfit_err:
                if settings.debug:
                    print(f"[overfitting] compute failed for {model_type}: {_overfit_err}")
                metrics_row["overfitting"] = None
                metrics_row["walkforward_periods"] = []

            # S16.3: Training diagnostics (feature importance, confusion matrix, confidence bands)
            try:
                from pipeline.diagnostics import (
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
                    from pipeline.diagnostics import FeatureImportanceEntry
                    _diag["feature_importance"] = [
                        {"feature": f, "importance": float(i)} for f, i in _fi_tuples
                    ]
                else:
                    _diag["feature_importance"] = []

                # Confusion matrix from results attrs
                _cm = None
                try:
                    _res = getattr(bt, "results", None)
                    if _res is not None and hasattr(_res, "attrs") and "confusion_matrix" in _res.attrs:
                        _cm_raw = _res.attrs["confusion_matrix"]
                        _cm = aggregate_confusion_matrices([_cm_raw]).matrix if _cm_raw is not None else None
                except Exception:
                    pass
                _diag["confusion_matrix"] = _cm

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
            _pub("model_training", job_id, {"model": model_type, "status": "complete", "metrics": metrics_row})

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

    except Exception as e:
        tb = traceback.format_exc()
        jm.update_status(job_id, "failed", error=f"{e}\n{tb}")
        _pub("job_failed", job_id, {"error": str(e)})
        emit_event("job_failed", job_id=job_id, error=str(e)[:200])
        raise


def _download_data_impl(job_id: str, pair: str, years: int = 10):
    """Core download logic -- executed by Celery worker or in-process (desktop mode)."""
    project_root = settings.project_root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)

    from api.services import JobManager
    from pipeline.data_sqlite import DataStore
    from pipeline.pair_config import get_pair_config, PAIR_REGISTRY

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)

    jm.update_status(job_id, "running")
    _pub("download_started", job_id, {"pair": pair})

    try:
        from pipeline.data_downloader import download_pair

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
            granularities=["M30", "H1", "H4"],
            years=years,
            pair_symbol=cfg.symbol,
        )

        jm.update_status(job_id, "completed", result={"pair": pair, "granularities": saved})
        _pub("download_complete", job_id, {"pair": pair})
        return {"pair": pair, "granularities": saved}

    except Exception as e:
        jm.update_status(job_id, "failed", error=str(e))
        _pub("download_failed", job_id, {"error": str(e)})
        raise


# --- Public API: dispatch to Celery or in-process depending on availability ---

if _celery_available and celery_app is not None:
    run_backtest_task = celery_app.task(name="run_backtest")(_run_backtest_impl)
    download_data_task = celery_app.task(name="download_data")(_download_data_impl)
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

    run_backtest_task = RunBacktestSync()
    download_data_task = DownloadDataSync()
