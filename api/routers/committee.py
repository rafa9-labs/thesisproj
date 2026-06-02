"""Committee backtest & regime analysis API endpoints (Racecar Phases A-E)."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/committee", tags=["committee"])

_COMMITTEE_RESULTS_DIR = Path("results/committee")
_COMMITTEE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pydantic models ─────────────────────────────────────────────────

class RegimeAssignmentSchema(BaseModel):
    models: List[str]
    weights: List[float]


class CommitteeConfigSchema(BaseModel):
    version: int = 1
    regimes: Dict[str, RegimeAssignmentSchema]
    fallback: RegimeAssignmentSchema
    constraints: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegimeMatrixEntry(BaseModel):
    regime: str
    model: str
    sharpe: float
    trades: int
    hit_rate: float


class RegimeMatrixResponse(BaseModel):
    regimes: List[str]
    models: List[str]
    entries: List[RegimeMatrixEntry]
    generated_at: Optional[str] = None


class RegimeLabelPoint(BaseModel):
    timestamp: str
    regime_id: int
    regime_name: str


class RegimeLabelsResponse(BaseModel):
    pair: str
    timeframe: str
    labels: List[RegimeLabelPoint]
    count: int


class CommitteeBacktestRequest(BaseModel):
    config: CommitteeConfigSchema
    pair: str = "EURUSD"
    timeframe: str = "H1"
    train_months: int = 4
    test_months: int = 1
    confidence_threshold: float = 0.5
    seq_len: int = 30


class CommitteeBacktestSubmitResponse(BaseModel):
    job_id: str
    status: str = "submitted"


class CommitteeBacktestResultResponse(BaseModel):
    job_id: str
    status: str
    total_folds: int = 0
    avg_sharpe: float = 0.0
    avg_trades: float = 0.0
    models: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    folds: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_s: float = 0.0


class CommitteeSnapshotInfo(BaseModel):
    version: str
    created_at: str
    models: List[str]


class CommitteeSnapshotListResponse(BaseModel):
    snapshots: List[CommitteeSnapshotInfo]


# ── Config endpoints ────────────────────────────────────────────────

_CONFIG_PATH = Path(os.environ.get("COMMITTEE_CONFIG_PATH", "results/committee/committee_config.json"))


@router.get("/config", response_model=CommitteeConfigSchema)
def get_committee_config():
    """Return the current committee configuration."""
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                return CommitteeConfigSchema(**json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
    default = CommitteeConfig(
        regimes={
            "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
            "trend_down": RegimeAssignment(models=["logistic"], weights=[1.0]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )
    return CommitteeConfigSchema(**default.to_dict())


@router.post("/config")
def save_committee_config(config: CommitteeConfigSchema):
    """Save/update the committee configuration."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)
    return {"status": "ok", "path": str(_CONFIG_PATH)}


# ── Regime matrix endpoint ─────────────────────────────────────────

@router.get("/regime-matrix", response_model=RegimeMatrixResponse)
def get_regime_matrix():
    """Return the regime×model performance matrix from ExpertProfiler output."""
    matrix_path = Path("results/profile/regime_model_matrix.json")
    if matrix_path.exists():
        try:
            with open(matrix_path) as f:
                data = json.load(f)
            entries = _parse_matrix_entries(data)
            return RegimeMatrixResponse(
                regimes=data.get("regimes", []),
                models=data.get("models", []),
                entries=entries,
                generated_at=str(matrix_path.stat().st_mtime),
            )
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    return RegimeMatrixResponse(regimes=[], models=[], entries=[])


def _parse_matrix_entries(data: dict) -> List[RegimeMatrixEntry]:
    entries: List[RegimeMatrixEntry] = []
    regimes = data.get("regimes", [])
    models = data.get("models", [])
    sharpe_matrix = data.get("sharpe", [])
    trade_matrix = data.get("trades", [])
    hitrate_matrix = data.get("hit_rate", [])

    if sharpe_matrix and models and regimes:
        for ri, regime in enumerate(regimes):
            for mi, model in enumerate(models):
                try:
                    entries.append(RegimeMatrixEntry(
                        regime=regime,
                        model=model,
                        sharpe=float(sharpe_matrix[mi][ri]) if ri < len(sharpe_matrix[mi]) else 0.0,
                        trades=int(trade_matrix[mi][ri]) if (
                            trade_matrix and ri < len(trade_matrix[mi])
                        ) else 0,
                        hit_rate=float(hitrate_matrix[mi][ri]) if (
                            hitrate_matrix and ri < len(hitrate_matrix[mi])
                        ) else 0.0,
                    ))
                except (IndexError, ValueError):
                    continue
    return entries


# ── Regime labels endpoint ──────────────────────────────────────────

@router.get("/regime-labels/{pair}/{timeframe}", response_model=RegimeLabelsResponse)
def get_regime_labels(
    pair: str,
    timeframe: str,
    bars: int = Query(500, description="Number of most recent bars to return"),
):
    """Return per-bar regime labels for the most recent N bars of a dataset."""
    from pipeline.regime_utils import detect_regimes, RegimeConfig, _REGIME_NAMES

    timeframe_map = {"H1": "H1", "H4": "H4", "M30": "M30", "M15": "M15"}
    tf = timeframe_map.get(timeframe, "H1")
    csv_path = Path(f"csv_data/{pair}_10_years_{tf}_OANDA.csv")
    if not csv_path.exists():
        csv_path = Path(f"csv_data/fx/{pair}_{tf}.csv")

    if not csv_path.exists():
        return RegimeLabelsResponse(pair=pair, timeframe=timeframe, labels=[], count=0)

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    df = df.tail(bars)
    regime_ids = detect_regimes(df, config=RegimeConfig())

    labels = []
    for idx, rid in enumerate(regime_ids):
        ts = df.index[idx]
        labels.append(RegimeLabelPoint(
            timestamp=str(ts),
            regime_id=int(rid),
            regime_name=_REGIME_NAMES.get(int(rid), "unknown"),
        ))

    return RegimeLabelsResponse(
        pair=pair, timeframe=timeframe, labels=labels, count=len(labels),
    )


# ── Committee backtest endpoints ────────────────────────────────────

@router.post("/backtest", response_model=CommitteeBacktestSubmitResponse)
def submit_committee_backtest(req: CommitteeBacktestRequest):
    """Submit a committee backtest job. Returns immediately with job_id."""
    job_id = f"committee_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    job_dir = _COMMITTEE_RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save config for this job
    with open(job_dir / "config.json", "w") as f:
        json.dump(req.model_dump(), f, indent=2, default=str)

    # Write status as pending
    with open(job_dir / "status.json", "w") as f:
        json.dump({"status": "pending", "created_at": datetime.utcnow().isoformat()}, f)

    # Run synchronously (fast with sklearn models; Celery integration deferred)
    try:
        _run_committee_backtest(job_id, req, job_dir)
    except Exception as e:
        with open(job_dir / "status.json", "w") as f:
            json.dump({"status": "failed", "error": str(e)}, f)
        raise HTTPException(500, f"Committee backtest failed: {e}")

    return CommitteeBacktestSubmitResponse(job_id=job_id)


@router.get("/backtest/{job_id}/results", response_model=CommitteeBacktestResultResponse)
def get_committee_backtest_results(job_id: str):
    """Return results for a completed committee backtest job."""
    job_dir = _COMMITTEE_RESULTS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")

    results_path = job_dir / "results.json"
    if not results_path.exists():
        status_path = job_dir / "status.json"
        if status_path.exists():
            with open(status_path) as f:
                status_data = json.load(f)
            return CommitteeBacktestResultResponse(
                job_id=job_id,
                status=status_data.get("status", "unknown"),
                warnings=[status_data.get("error", "")] if "error" in status_data else [],
            )
        raise HTTPException(404, "Results not yet available")

    with open(results_path) as f:
        data = json.load(f)

    return CommitteeBacktestResultResponse(**data)


def _run_committee_backtest(job_id: str, req: CommitteeBacktestRequest, job_dir: Path):
    """Internal: run the committee backtest synchronously."""
    from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
    from pipeline.committee_backtester import CommitteeBacktester
    from pipeline.regime_utils import RegimeConfig

    # Load data
    timeframe_map = {"H1": "H1", "H4": "H4", "M30": "M30", "M15": "M15"}
    tf = timeframe_map.get(req.timeframe, "H1")
    csv_path = Path(f"csv_data/{req.pair}_10_years_{tf}_OANDA.csv")

    if not csv_path.exists():
        raise HTTPException(400, f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    if "returns" not in df.columns:
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

    # Build config from request
    regimes = {}
    for rname, rdata in req.config.regimes.items():
        regimes[rname] = RegimeAssignment(
            models=rdata.models,
            weights=rdata.weights,
        )
    fallback = RegimeAssignment(
        models=req.config.fallback.models,
        weights=req.config.fallback.weights,
    )
    config = CommitteeConfig(regimes=regimes, fallback=fallback)

    # Run backtest
    bt = CommitteeBacktester(
        config,
        regime_cfg=RegimeConfig(),
        confidence_threshold=req.confidence_threshold,
        seq_len=req.seq_len,
    )
    result = bt.run_wfo(
        df,
        train_months=req.train_months,
        test_months=req.test_months,
        verbose=True,
    )

    # Serialize results
    folds_data = []
    for fold in result.folds:
        folds_data.append({
            "fold_idx": fold.fold_idx,
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "sharpe": fold.sharpe,
            "trades": fold.trades,
            "active_rate": fold.active_rate,
            "win_rate": fold.win_rate,
            "return_val": fold.return_val,
            "drawdown": fold.drawdown,
            "regime_distribution": fold.regime_distribution,
        })

    results_data = {
        "job_id": job_id,
        "status": "completed",
        "total_folds": result.total_folds,
        "avg_sharpe": result.avg_sharpe,
        "avg_trades": result.avg_trades,
        "models": result.models,
        "warnings": result.warnings,
        "folds": folds_data,
        "execution_time_s": result.execution_time_s,
    }

    with open(job_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    with open(job_dir / "status.json", "w") as f:
        json.dump({"status": "completed", "completed_at": datetime.utcnow().isoformat()}, f)


# ── Model store snapshots endpoint ──────────────────────────────────

@router.get("/snapshots", response_model=CommitteeSnapshotListResponse)
def list_committee_snapshots():
    """List available model store snapshots (for deployment inspection)."""
    snapshots_dir = Path("results/snapshots")
    items: List[CommitteeSnapshotInfo] = []

    if snapshots_dir.exists():
        for entry in sorted(snapshots_dir.iterdir(), reverse=True):
            if entry.is_dir():
                manifest = entry / "manifest.json"
                models = []
                if manifest.exists():
                    try:
                        with open(manifest) as f:
                            mdata = json.load(f)
                        models = list(mdata.get("models", {}).keys())
                    except (json.JSONDecodeError, OSError):
                        pass
                items.append(CommitteeSnapshotInfo(
                    version=entry.name,
                    created_at=str(datetime.fromtimestamp(
                        entry.stat().st_mtime).isoformat()),
                    models=models,
                ))
    return CommitteeSnapshotListResponse(snapshots=items[:20])
