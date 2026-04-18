"""Backtest runner endpoints."""
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_data_store
from api.schemas.backtest import (
    BacktestListItem,
    BacktestListResponse,
    BacktestRequest,
    BacktestResultsResponse,
    BacktestResultMetrics,
    BacktestStatusResponse,
    BacktestSubmitResponse,
)
from api.services import JobManager
from api.tasks import download_data_task, run_backtest_task
from pipeline.pair_config import VALID_PAIRS

router = APIRouter(prefix="/backtest", tags=["backtest"])


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
            total_return=m.get("total_return"),
            win_rate=m.get("win_rate"),
            total_trades=m.get("total_trades"),
            profit_factor=m.get("profit_factor"),
            avg_trade=m.get("avg_trade"),
        ))

    return BacktestResultsResponse(
        job_id=job_id,
        pair=result.get("pair", cfg.get("pair", "")),
        models=result.get("models", cfg.get("models", [])),
        metrics=metrics,
    )


@router.delete("/{job_id}", status_code=204)
def delete_backtest(job_id: str):
    store = get_data_store()
    jm = JobManager(store)
    if not jm.delete_job(job_id):
        raise HTTPException(404, f"Job {job_id} not found")
