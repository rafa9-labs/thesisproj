"""Backtest runner endpoints."""
import uuid
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

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
from api.tasks import download_data_task, run_backtest_task
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
        "trading_costs": req.trading_costs,
        "config_overrides": req.config_overrides,
    }

    jm.create_job(job_id, "backtest", config)
    run_backtest_task.delay(job_id, config)

    return BacktestSubmitResponse(
        job_id=job_id,
        status="pending",
        pair=pair,
        models=req.models,
    )


@router.get("", response_model=BacktestListResponse)
def list_backtests(limit: int = Query(50, ge=1, le=200)):
    store = get_data_store()
    jm = JobManager(store)
    jobs = jm.list_jobs(job_type="backtest", limit=limit)

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

    return BacktestListResponse(jobs=items)


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
            equity_curve=m.get("equity_curve"),
            buy_hold_curve=m.get("buy_hold_curve"),
            drawdown_curve=m.get("drawdown_curve"),
            monthly_results=m.get("monthly_results"),
            trades=m.get("trades"),
            hpo_param_importance=m.get("hpo_param_importance"),
            hpo_trials=m.get("hpo_trials"),
        ))

    return BacktestResultsResponse(
        job_id=job_id,
        pair=result.get("pair", cfg.get("pair", "")),
        models=result.get("models", cfg.get("models", [])),
        config=cfg,
        metrics=metrics,
    )


@router.delete("/{job_id}", status_code=204)
def delete_backtest(job_id: str):
    store = get_data_store()
    jm = JobManager(store)
    if not jm.delete_job(job_id):
        raise HTTPException(404, f"Job {job_id} not found")


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
