"""Backtest runner endpoints."""
import uuid
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.config import settings
from api.dependencies import get_data_store
from api.schemas.backtest import (
    CLASSIC_QUICK_TEST,
    CLASSIC_VALIDATE,
    HPO_TRIAL_MAPS,
    QUICK_TEST_PRESETS,
    BacktestListItem,
    BacktestListResponse,
    BacktestRequest,
    BacktestResultsResponse,
    BacktestResultMetrics,
    BacktestStatusResponse,
    BacktestSubmitResponse,
    BacktestSummaryItem,
    BacktestSummaryResponse,
    CrossPairCurve,
    CrossPairCurvesResponse,
    DateRangePreset,
    DateRangeResponse,
    HeatmapCell,
    HeatmapResponse,
    QuickTestPreset,
    RuntimeEstimateRequest,
    RuntimeEstimateResponse,
)
from api.services import JobManager
from api.tasks import download_data_task, run_backtest_task, IS_DESKTOP
from pipeline.pair_config import VALID_PAIRS

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.get("/presets", response_model=list[QuickTestPreset])
def list_quick_test_presets():
    return QUICK_TEST_PRESETS


@router.get("/date-ranges", response_model=DateRangeResponse)
def get_date_ranges(pair: str = "EURUSD", timeframe: str = "H1"):
    pair = pair.upper()
    if pair not in VALID_PAIRS:
        raise HTTPException(400, f"Unknown pair: {pair}. Available: {VALID_PAIRS}")

    store = get_data_store()
    tfs = store.list_timeframes(pair)
    if timeframe not in tfs:
        tfs = tfs[:1] if tfs else []
        if not tfs:
            raise HTTPException(404, f"No data found for {pair}")

    rng = store.get_date_range(pair, timeframe)
    if not rng:
        raise HTTPException(404, f"No date range for {pair}/{timeframe}")

    data_start, data_end = rng
    data_start_str = str(data_start)[:10]
    data_end_str = str(data_end)[:10]

    today = date.today()
    first_of_current = today.replace(day=1)
    last_of_prev = first_of_current - timedelta(days=1)

    presets = [
        DateRangePreset(
            key="last_1m",
            label="Last 1 month",
            start_date=(last_of_prev - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=last_of_prev.strftime("%Y-%m-%d"),
        ),
        DateRangePreset(
            key="last_3m",
            label="Last 3 months",
            start_date=(last_of_prev - timedelta(days=90)).strftime("%Y-%m-%d"),
            end_date=last_of_prev.strftime("%Y-%m-%d"),
        ),
        DateRangePreset(
            key="last_6m",
            label="Last 6 months",
            start_date=(last_of_prev - timedelta(days=180)).strftime("%Y-%m-%d"),
            end_date=last_of_prev.strftime("%Y-%m-%d"),
        ),
        DateRangePreset(
            key="ytd",
            label="Year to date",
            start_date=today.replace(month=1, day=1).strftime("%Y-%m-%d"),
            end_date=last_of_prev.strftime("%Y-%m-%d"),
        ),
        DateRangePreset(
            key="all",
            label="All available data",
            start_date=data_start_str,
            end_date=data_end_str,
        ),
    ]

    return DateRangeResponse(
        symbol=pair,
        timeframe=timeframe,
        data_start=data_start_str,
        data_end=data_end_str,
        presets=presets,
    )


@router.post("/estimate-runtime", response_model=RuntimeEstimateResponse)
def estimate_runtime(req: RuntimeEstimateRequest):
    trial_map_name = req.hpo_intensity
    per_model_times = {
        "logistic": 0.15,
        "svm": 0.25,
        "decision_tree": 0.1,
        "random_forest": 0.35,
        "xgboost": 0.2,
        "lstm": 1.5,
        "cnn": 1.2,
        "transformer": 2.0,
        "ensemble_adaptive_regime": 3.0,
        "ensemble_cnn_lstm_xgboost": 3.0,
        "dqn": 2.5,
    }

    total_trials = 0
    total_low = 0.0
    total_high = 0.0

    for m in req.models:
        tc = HPO_TRIAL_MAPS.get(trial_map_name, HPO_TRIAL_MAPS["quick"]).get(
            m, {"random": 2, "bayes": 2}
        )
        n_trials = tc["random"] + tc["bayes"]
        total_trials += n_trials

        base_min = per_model_times.get(m, 0.5)
        hpo_time_low = n_trials * base_min * req.months * 0.3
        hpo_time_high = n_trials * base_min * req.months * 0.6
        sim_time_low = base_min * req.months
        sim_time_high = base_min * req.months * 1.5

        total_low += hpo_time_low + sim_time_low
        total_high += hpo_time_high + sim_time_high

    return RuntimeEstimateResponse(
        models=req.models,
        months=req.months,
        hpo_intensity=req.hpo_intensity,
        total_trials=total_trials,
        estimated_minutes_low=round(total_low, 1),
        estimated_minutes_high=round(total_high, 1),
    )


@router.post("", response_model=BacktestSubmitResponse, status_code=202)
def submit_backtest(req: BacktestRequest):
    pair = req.pair.upper()
    if pair not in VALID_PAIRS:
        raise HTTPException(400, f"Unknown pair: {pair}. Available: {VALID_PAIRS}")

    from models.registry import MODEL_REGISTRY
    for m in req.models:
        if m not in MODEL_REGISTRY:
            raise HTTPException(400, f"Unknown model: {m}. Available: {list(MODEL_REGISTRY.keys())}")

    store = get_data_store()
    jm = JobManager(store)

    job_id = str(uuid.uuid4())
    config = {
        "pair": pair,
        "models": req.models,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "months": req.months,
        "repeats": req.repeats,
        "seed": req.seed,
        "hpo_intensity": req.hpo_intensity,
        "n_trials": req.n_trials,
        "parent_job_id": req.parent_job_id,
        "trading_costs": req.trading_costs,
        "config_overrides": req.config_overrides,
    }

    try:
        jm.create_job_atomic(job_id, "backtest", config, max_active=settings.max_concurrent_backtests)
    except RuntimeError:
        raise HTTPException(status_code=409, detail="A backtest is already running. Please wait for completion.")

    if IS_DESKTOP:
        import threading
        from api.tasks import _cancellation_events

        def _run_desktop():
            evt = threading.Event()
            _cancellation_events[job_id] = evt
            try:
                run_backtest_task._func(job_id, config)
            except Exception as exc:
                import traceback
                traceback.print_exc()
            finally:
                _cancellation_events.pop(job_id, None)

        threading.Thread(target=_run_desktop, daemon=True).start()
        return BacktestSubmitResponse(
            job_id=job_id,
            status="pending",
            pair=pair,
            models=req.models,
        )
    else:
        run_backtest_task.delay(job_id, config)

    return BacktestSubmitResponse(
        job_id=job_id,
        status="pending",
        pair=pair,
        models=req.models,
    )


@router.get("", response_model=BacktestListResponse)
def list_backtests(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    store = get_data_store()
    jm = JobManager(store)
    jobs, total = jm.list_jobs_paginated(job_type="backtest", limit=limit, offset=offset)

    items = []
    for j in jobs:
        cfg = j.get("config", {})
        items.append(BacktestListItem(
            job_id=j["id"],
            type=j["type"],
            status=j["status"],
            pair=cfg.get("pair", ""),
            models=cfg.get("models", []),
            created_at=j["created_at"],
        ))

    return BacktestListResponse(jobs=items, total=total, offset=offset, limit=limit)


@router.get("/heatmap", response_model=HeatmapResponse)
def get_heatmap():
    store = get_data_store()
    jm = JobManager(store)
    jobs = jm.list_jobs(job_type="backtest", limit=500)

    models_set: set = set()
    pairs_set: set = set()
    cells: list = []

    for job in jobs:
        if job.get("status") != "completed":
            continue
        result = job.get("result", {})
        cfg = job.get("config", {})
        pair = result.get("pair") or cfg.get("pair", "")
        job_id = job.get("id", "")

        if not pair or not pair:
            continue

        raw_metrics = result.get("metrics", [])
        for m in raw_metrics:
            model_name = m.get("model", "")
            if not model_name:
                continue
            models_set.add(model_name)
            pairs_set.add(pair)
            cells.append(
                HeatmapCell(
                    model=model_name,
                    pair=pair,
                    sharpe=m.get("sharpe"),
                    total_return_pct=m.get("total_return_pct"),
                    win_rate=m.get("win_rate"),
                    max_drawdown=m.get("max_drawdown"),
                    total_trades=m.get("total_trades"),
                    job_id=job_id,
                )
            )

    return HeatmapResponse(
        models=sorted(models_set),
        pairs=sorted(pairs_set),
        cells=cells,
    )


@router.get("/cross-pair-curves", response_model=CrossPairCurvesResponse)
def get_cross_pair_curves(
    model: str = Query(..., description="Model name"),
    pairs: str = Query(..., description="Comma-separated pair list"),
):
    pair_list = [p.strip().upper() for p in pairs.split(",") if p.strip()]

    store = get_data_store()
    jm = JobManager(store)
    jobs = jm.list_jobs(job_type="backtest", limit=500)

    curves: list = []
    used_jobs: set = set()

    for job in jobs:
        if job.get("status") != "completed":
            continue
        result = job.get("result", {})
        cfg = job.get("config", {})
        job_pair = result.get("pair") or cfg.get("pair", "")

        if job_pair not in pair_list:
            continue

        raw_metrics = result.get("metrics", [])
        for m in raw_metrics:
            if m.get("model") != model:
                continue
            curve = m.get("equity_curve")
            if not curve:
                continue
            dedup_key = f"{job_pair}::{job.get('id', '')}"
            if dedup_key in used_jobs:
                continue
            used_jobs.add(dedup_key)
            curves.append(
                CrossPairCurve(
                    model=model,
                    pair=job_pair,
                    equity_curve=curve,
                )
            )

    return CrossPairCurvesResponse(model=model, curves=curves)


@router.get("/results/summary", response_model=BacktestSummaryResponse)
def get_results_summary(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pair: str = Query("", description="Filter by pair"),
    model: str = Query("", description="Filter by model"),
    sort_by: str = Query("created_at", description="Sort column: created_at, sharpe, total_return_pct, win_rate, max_drawdown_pct"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
):
    store = get_data_store()
    jm = JobManager(store)

    all_jobs, _ = jm.list_jobs_paginated(job_type="backtest", limit=2000, offset=0)

    results = []
    for job in all_jobs:
        if job.get("status") != "completed":
            continue
        result = job.get("result", {})
        cfg = job.get("config", {})
        job_pair = result.get("pair") or cfg.get("pair", "")
        job_models = result.get("models") or cfg.get("models", [])

        if pair and job_pair != pair:
            continue
        if model and model not in job_models:
            continue

        raw_metrics = result.get("metrics", [])
        for m in raw_metrics:
            results.append(BacktestSummaryItem(
                job_id=job["id"],
                created_at=job["created_at"],
                pair=job_pair,
                timeframe=cfg.get("timeframe", ""),
                models=job_models,
                sharpe=m.get("sharpe"),
                total_return_pct=m.get("total_return_pct"),
                win_rate=m.get("win_rate"),
                max_drawdown_pct=m.get("max_drawdown"),
                total_trades=m.get("total_trades"),
                status=job["status"],
            ))

    valid_sort_cols = {
        "created_at": "created_at",
        "sharpe": "sharpe",
        "total_return_pct": "total_return_pct",
        "win_rate": "win_rate",
        "max_drawdown_pct": "max_drawdown_pct",
    }
    sort_col = valid_sort_cols.get(sort_by, "created_at")
    reverse = sort_order == "desc"
    results.sort(key=lambda x: getattr(x, sort_col) if getattr(x, sort_col) is not None else 0.0, reverse=reverse)

    total = len(results)
    paged = results[offset : offset + limit]

    return BacktestSummaryResponse(results=paged, total=total, offset=offset, limit=limit)


@router.get("/active", response_model=BacktestListResponse)
def get_active_backtests():
    store = get_data_store()
    jm = JobManager(store)
    jobs = jm.get_active_jobs("backtest")
    items = []
    for j in jobs:
        cfg = j.get("config", {})
        items.append(BacktestListItem(
            job_id=j["id"],
            type=j["type"],
            status=j["status"],
            pair=cfg.get("pair", ""),
            models=cfg.get("models", []),
            created_at=j["created_at"],
        ))
    return BacktestListResponse(jobs=items, total=len(items), offset=0, limit=len(items))


@router.post("/{job_id}/force-stop", response_model=BacktestStatusResponse)
def force_stop_backtest(job_id: str):
    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(400, f"Job status is '{job['status']}', not pending or running")
    jm.force_stop_job(job_id)
    # Revoke the running Celery task
    try:
        from api.tasks import revoke_task
        revoke_task(job_id)
    except Exception:
        pass
    updated = jm.get_job(job_id)
    return BacktestStatusResponse(
        job_id=updated["id"],
        type=updated["type"],
        status=updated["status"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
        error=updated.get("error"),
        progress=updated.get("result"),
    )


@router.get("/{job_id}", response_model=BacktestStatusResponse)
def get_backtest_status(job_id: str):
    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")

    return BacktestStatusResponse(
        job_id=job["id"],
        type=job["type"],
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        error=job.get("error"),
        progress=job.get("result"),
    )


@router.get("/{job_id}/events")
def get_backtest_events(job_id: str, after: int = 0):
    from api.tasks import get_job_events
    import logging as _logging
    events = get_job_events(job_id, after=after)
    _logging.info(f"[EVENTS-API] job={job_id[:8]} after={after} returned={len(events)} total={after + len(events)}")
    return {"events": events, "total": after + len(events)}


def _coerce_curve(raw):
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) == 0:
        return raw if isinstance(raw, list) else None
    first = raw[0]
    if isinstance(first, dict):
        return raw
    if isinstance(first, (int, float)):
        return [{"time": i, "value": float(v)} for i, v in enumerate(raw)]
    return raw


def _parse_diagnostics(raw):
    if raw is None or not isinstance(raw, dict):
        return None
    from api.schemas.backtest import (
        TrainingDiagnostics, FeatureImportanceEntry,
        PredictionHistogramBin, ConfusionMatrixData, ConfidenceBand,
    )
    fi = None
    if "feature_importance" in raw and raw["feature_importance"]:
        try:
            fi = [FeatureImportanceEntry(**e) for e in raw["feature_importance"]]
        except Exception:
            fi = None
    hist = None
    if "prediction_histogram" in raw and raw["prediction_histogram"]:
        try:
            hist = [PredictionHistogramBin(**e) for e in raw["prediction_histogram"]]
        except Exception:
            hist = None
    cm = None
    if "confusion_matrix" in raw and raw["confusion_matrix"]:
        try:
            cm = ConfusionMatrixData(**raw["confusion_matrix"])
        except Exception:
            cm = None
    bands = None
    if "confidence_bands" in raw and raw["confidence_bands"]:
        try:
            bands = [ConfidenceBand(**e) for e in raw["confidence_bands"]]
        except Exception:
            bands = None
    return TrainingDiagnostics(
        feature_importance=fi,
        prediction_histogram=hist,
        confusion_matrix=cm,
        confidence_bands=bands,
        importance_method=raw.get("importance_method"),
        feature_families=raw.get("feature_families"),
        vif_warnings=raw.get("vif_warnings"),
    )


@router.get("/{job_id}/results", response_model=BacktestResultsResponse)
def get_backtest_results(job_id: str):
    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job status is '{job['status']}', not 'completed'")

    result = job.get("result", {})
    cfg = job.get("config", {})
    raw_metrics = result.get("metrics", [])

    metrics = []
    for m in raw_metrics:
        overfit_data = m.get("overfitting")
        overfit = None
        if overfit_data and isinstance(overfit_data, dict):
            from api.schemas.backtest import OverfittingReport as OFReport, OverfittingCI
            sharpe_ci = None
            if overfit_data.get("sharpe_ci"):
                sharpe_ci = OverfittingCI(**overfit_data["sharpe_ci"])
            return_ci = None
            if overfit_data.get("return_ci"):
                return_ci = OverfittingCI(**overfit_data["return_ci"])
            maxdd_ci = None
            if overfit_data.get("maxdd_ci"):
                maxdd_ci = OverfittingCI(**overfit_data["maxdd_ci"])
            overfit = OFReport(
                overfit_score=overfit_data.get("overfit_score", 0.0),
                risk_level=overfit_data.get("risk_level", "low"),
                risk_color=overfit_data.get("risk_color", "green"),
                train_oos_gap_pct=overfit_data.get("train_oos_gap_pct", 0.0),
                temporal_degradation_pct=overfit_data.get("temporal_degradation_pct", 0.0),
                sharpe_ci=sharpe_ci,
                return_ci=return_ci,
                maxdd_ci=maxdd_ci,
                cv_sharpe_mean=overfit_data.get("cv_sharpe_mean"),
                cv_sharpe_std=overfit_data.get("cv_sharpe_std"),
                cv_return_mean=overfit_data.get("cv_return_mean"),
                cv_return_std=overfit_data.get("cv_return_std"),
                min_trl_trades=overfit_data.get("min_trl_trades", 10),
                sufficient_trades=overfit_data.get("sufficient_trades", False),
                n_periods=overfit_data.get("n_periods", 0),
                n_signal_periods=overfit_data.get("n_signal_periods", 0),
                signal_gap_pct=overfit_data.get("signal_gap_pct", 0.0),
                is_mean_sharpe=overfit_data.get("is_mean_sharpe"),
                oos_mean_sharpe=overfit_data.get("oos_mean_sharpe"),
                dsr_min_sharpe=overfit_data.get("dsr_min_sharpe"),
                interaction_effects=overfit_data.get("interaction_effects"),
            )

        wf_periods_raw = m.get("walkforward_periods", [])
        wf_periods = []
        for wp in (wf_periods_raw or []):
            from api.schemas.backtest import WalkForwardPeriod
            wf_periods.append(WalkForwardPeriod(
                period_start=str(wp.get("period_start", "")),
                period_end=str(wp.get("period_end", "")),
                train_start=wp.get("train_start"),
                train_end=wp.get("train_end"),
                test_sharpe=wp.get("test_sharpe"),
                train_sharpe=wp.get("train_sharpe"),
                strategy_return=wp.get("strategy_return"),
                bh_return=wp.get("bh_return"),
                trades=int(wp.get("trades", 0) or 0),
                signals_raw=int(wp.get("signals_raw", 0) or 0),
                signals_passed_gate=int(wp.get("signals_passed_gate", 0) or 0),
                pct_sideways=wp.get("pct_sideways"),
                pct_trend=wp.get("pct_trend"),
                pct_volatile=wp.get("pct_volatile"),
            ))

        metrics.append(BacktestResultMetrics(
            model=m.get("model", ""),
            sharpe=m.get("sharpe"),
            sortino=m.get("sortino"),
            max_drawdown=m.get("max_drawdown"),
            total_return_pct=m.get("total_return_pct"),
            cagr=m.get("cagr"),
            calmar_ratio=m.get("calmar_ratio"),
            win_rate=m.get("win_rate"),
            total_trades=m.get("total_trades"),
            profit_factor=m.get("profit_factor"),
            avg_trade=m.get("avg_trade"),
            active_rate=m.get("active_rate"),
            directional_accuracy=m.get("directional_accuracy"),
            precision_macro=m.get("precision_macro"),
            f1_macro=m.get("f1_macro"),
            equity_curve=_coerce_curve(m.get("equity_curve")),
            buy_hold_curve=_coerce_curve(m.get("buy_hold_curve")),
            drawdown_curve=_coerce_curve(m.get("drawdown_curve")),
            monthly_results=m.get("monthly_results"),
            trades=m.get("trades"),
            hpo_param_importance=m.get("hpo_param_importance"),
            hpo_trials=m.get("hpo_trials"),
            overfitting=overfit,
            walkforward_periods=wf_periods,
            diagnostics=_parse_diagnostics(m.get("diagnostics")),
            summary_text=m.get("summary_text"),
            snapshot_path=m.get("snapshot_path"),
        ))

    return BacktestResultsResponse(
        job_id=job_id,
        pair=result.get("pair", cfg.get("pair", "")),
        models=result.get("models", cfg.get("models", [])),
        config=cfg,
        metrics=metrics,
    )


@router.post("/{job_id}/analyze")
def analyze_backtest_results(job_id: str, model: str = Query("", description="Model to analyze")):
    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job status is '{job['status']}', not 'completed'")

    result = job.get("result", {})
    cfg = job.get("config", {})
    raw_metrics = result.get("metrics", [])

    target = raw_metrics[0] if raw_metrics else {}
    if model:
        for m in raw_metrics:
            if isinstance(m, dict) and m.get("model") == model:
                target = m
                break

    try:
        from pipeline.llm.advisor import analyze_backtest
        analysis = analyze_backtest(target, cfg)
        return {"job_id": job_id, "model": target.get("model", ""), "analysis": analysis}
    except Exception as e:
        return {"job_id": job_id, "model": target.get("model", ""), "analysis": {"error": str(e)}}


# ────────────────────────────────────────────────────────────
#  Experiment diff & tags (Phase B: experiment tracking)
# ────────────────────────────────────────────────────────────
from pydantic import BaseModel


class TagsUpdateRequest(BaseModel):
    action: str
    tag: str


@router.get("/experiments/{job_id}/diff")
def get_experiment_diff(job_id: str, compare: str = Query(..., description="Job ID to compare against")):
    store = get_data_store()
    jm = JobManager(store)
    a = jm.get_job(job_id)
    b = jm.get_job(compare)
    if a is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if b is None:
        raise HTTPException(404, f"Comparison job {compare} not found")

    cfg_a = a.get("config", {}) if isinstance(a.get("config"), dict) else {}
    cfg_b = b.get("config", {}) if isinstance(b.get("config"), dict) else {}
    overrides_a = cfg_a.get("config_overrides", {}) or {}
    overrides_b = cfg_b.get("config_overrides", {}) or {}

    def _flatten(cfg, overrides, prefix=""):
        flat = {}
        for k in ["pair", "models", "months", "repeats", "seed", "hpo_intensity"]:
            if k in cfg:
                flat[prefix + k] = cfg[k]
        for k, v in overrides.items():
            flat[prefix + k] = v
        return flat

    flat_a = _flatten(cfg_a, overrides_a)
    flat_b = _flatten(cfg_b, overrides_b)
    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))

    added_keys = {}
    removed_keys = {}
    changed_values = {}
    unchanged_count = 0

    for k in all_keys:
        va = flat_a.get(k)
        vb = flat_b.get(k)
        if k not in flat_a and k in flat_b:
            added_keys[k] = vb
        elif k in flat_a and k not in flat_b:
            removed_keys[k] = va
        elif va != vb:
            changed_values[k] = {"from": va, "to": vb}
        else:
            unchanged_count += 1

    return {
        "base": {"job_id": job_id, "created_at": a.get("created_at", "")},
        "compare": {"job_id": compare, "created_at": b.get("created_at", "")},
        "added_keys": added_keys,
        "removed_keys": removed_keys,
        "changed_values": changed_values,
    "unchanged_count": unchanged_count,
}


@router.patch("/experiments/{job_id}/tags")
def update_experiment_tags(job_id: str, req: TagsUpdateRequest):
    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")

    import json as _json
    existing = _json.loads(job.get("result", "{}") or "{}")
    tags = list(existing.get("tags", []) if isinstance(existing.get("tags"), list) else [])

    if req.action == "add" and req.tag not in tags:
        tags.append(req.tag)
    elif req.action == "remove":
        tags = [t for t in tags if t != req.tag]

    existing["tags"] = tags
    jm.update_status(job_id, job.get("status", "completed"), result=existing)
    return {"tags": tags}


@router.get("/{job_id}/trades/chart-data")
def get_trade_chart_data(job_id: str, model: str = Query(..., description="Model name")):
    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job status is '{job['status']}', not 'completed'")

    result = job.get("result", {})
    cfg = job.get("config", {})
    pair = result.get("pair") or cfg.get("pair", "")
    timeframe = cfg.get("timeframe", "H1")

    raw_metrics = result.get("metrics", [])
    target_metric = None
    for m in raw_metrics:
        if m.get("model") == model:
            target_metric = m
            break
    if target_metric is None:
        raise HTTPException(404, f"Model '{model}' not found in job results")

    start_date = cfg.get("start_date") or None
    end_date = cfg.get("end_date") or None
    candles_df = store.get_candles(pair, timeframe, start_date, end_date)
    candles = []
    if not candles_df.empty:
        for _, row in candles_df.iterrows():
            t_val = row["time"]
            t_epoch = int(t_val.timestamp()) if hasattr(t_val, "timestamp") else 0
            candles.append({
                "t": t_epoch,
                "o": round(float(row["mid_open"]), 10),
                "h": round(float(row["mid_high"]), 10),
                "l": round(float(row["mid_low"]), 10),
                "c": round(float(row["mid_close"]), 10),
                "volume": int(row.get("volume", 0) or 0),
            })

    candle_lookup = {}
    sorted_epochs = []
    for c in candles:
        candle_lookup[c["t"]] = c
        sorted_epochs.append(c["t"])
    sorted_epochs.sort()

    import bisect

    def find_nearest_candle(epoch: int, tolerance_sec: int = 1800):
        exact = candle_lookup.get(epoch)
        if exact is not None:
            return exact
        idx = bisect.bisect_left(sorted_epochs, epoch)
        best = None
        best_diff = tolerance_sec + 1
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(sorted_epochs):
                cand_epoch = sorted_epochs[candidate_idx]
                diff = abs(cand_epoch - epoch)
                if diff <= tolerance_sec and diff < best_diff:
                    best = candle_lookup[cand_epoch]
                    best_diff = diff
        return best

    import logging
    logger = logging.getLogger(__name__)

    trades = []
    raw_trades = target_metric.get("trades", [])
    if raw_trades:
        for t in raw_trades:
            try:
                entry_raw = t.get("entry_time") or t.get("entry_date") or ""
                exit_raw = t.get("exit_time") or t.get("exit_date") or ""
                if isinstance(entry_raw, str):
                    entry_dt = pd.to_datetime(entry_raw)
                    entry_epoch = int(entry_dt.timestamp())
                else:
                    entry_epoch = int(entry_raw) if entry_raw else 0
                if isinstance(exit_raw, str):
                    exit_dt = pd.to_datetime(exit_raw)
                    exit_epoch = int(exit_dt.timestamp())
                else:
                    exit_epoch = int(exit_raw) if exit_raw else 0

                side = t.get("side", "")
                direction = "BUY" if side in ("long", "buy", "BUY", 1, 1.0) else "SELL"

                entry_c = find_nearest_candle(entry_epoch)
                exit_c = find_nearest_candle(exit_epoch)

                entry_price = round(entry_c["c"], 10) if entry_c else (
                    round(float(t["entry_price"]), 10) if t.get("entry_price") else None
                )
                exit_price = round(exit_c["c"], 10) if exit_c else (
                    round(float(t["exit_price"]), 10) if t.get("exit_price") else None
                )

                trades.append({
                    "trade_id": t.get("trade_id", 0),
                    "entry_time": entry_epoch,
                    "exit_time": exit_epoch,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": t.get("pnl_pct") or t.get("return_pct") or 0,
                })
            except Exception:
                logger.exception("Failed to process trade entry: %s", t)
                continue

    equity_curve = target_metric.get("equity_curve", [])
    if isinstance(equity_curve, list) and len(equity_curve) > 0:
        equity_curve = [
            {"time": e.get("time", 0), "value": e.get("value", 0)}
            for e in equity_curve if isinstance(e, dict)
        ]
    else:
        equity_curve = []

    return {
        "pair": pair,
        "timeframe": timeframe,
        "candles": candles,
        "trades": trades,
        "equity_curve": equity_curve,
    }


@router.get("/{job_id}/debug/events")
def debug_job_events(job_id: str, limit: int = Query(50, ge=1, le=500)):
    """Debug endpoint: return raw events from SQLite for a job, with metadata."""
    from api.dependencies import get_data_store
    from pipeline.data_sqlite import DataStore
    import logging

    store = get_data_store()
    jm = JobManager(store)
    job = jm.get_job(job_id)
    job_exists = job is not None

    sqlite_available = False
    sqlite_count = 0
    sqlite_events = []
    try:
        ds = DataStore(store.db_path)
        sqlite_count = ds.get_job_event_count(job_id)
        raw = ds.get_job_events(job_id, after=0)
        sqlite_events = raw[-limit:]
        sqlite_available = True
    except Exception as e:
        logging.warning(f"[debug] SQLite read failed for {job_id}: {e}")

    in_memory_events = []
    in_memory_count = 0
    try:
        from api.tasks import _job_events
        buf = _job_events.get(job_id, [])
        in_memory_count = len(buf)
        in_memory_events = list(buf)
    except Exception:
        pass

    from api.routers.ws import _get_ws_connections
    ws_connections = _get_ws_connections(job_id)

    return {
        "job_id": job_id,
        "job_exists": job_exists,
        "job_status": job.get("status") if job else None,
        "sqlite": {
            "available": sqlite_available,
            "total_count": sqlite_count,
            "recent_events": sqlite_events,
        },
        "in_memory": {
            "total_count": in_memory_count,
            "recent_events": in_memory_events[-limit:],
        },
        "websocket_connections": ws_connections,
    }


@router.delete("/{job_id}", status_code=204)
def delete_backtest(job_id: str):
    store = get_data_store()
    jm = JobManager(store)
    if not jm.delete_job(job_id):
        raise HTTPException(404, f"Job {job_id} not found")
