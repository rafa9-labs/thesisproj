"""Committee backtest & regime analysis API endpoints (Racecar Phases A-E)."""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ── Racecar Auto-Optimize: full B→C→D pipeline ──────────────────────

class RacecarAutoOptimizeRequest(BaseModel):
    models: List[str] = Field(default_factory=lambda: [
        "logistic", "random_forest", "xgboost",
    ])
    pair: str = "EURUSD"
    timeframe: str = "H1"
    train_months: int = 6
    test_months: int = 1
    profile_trials: int = 5
    committee_top_k: int = 3


class RacecarJobStatus(BaseModel):
    job_id: str
    phase: str  # "profiling", "building", "backtesting", "completed", "failed"
    phase_progress: str = ""  # e.g. "3/5 models"
    started_at: str
    error: str = ""


class RacecarJobResults(BaseModel):
    job_id: str
    status: str
    profile_matrix: Optional[Dict[str, Any]] = None
    committee_config: Optional[Dict[str, Any]] = None
    backtest: Optional[Dict[str, Any]] = None
    total_time_s: float = 0.0


_AUTO_OPTIMIZE_DIR = Path("results/racecar")
_AUTO_OPTIMIZE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/auto-optimize", response_model=RacecarJobStatus)
def start_auto_optimize(req: RacecarAutoOptimizeRequest):
    """Start the full Racecar auto-optimize pipeline (B→C→D).

    Phases:
      B. ExpertProfiler — run all models across WFO folds, build regime×model matrix
      C. CommitteeBuilder — build optimal committee config from matrix
      D. CommitteeBacktester — WFO validation of the committee
    """
    job_id = f"racecar_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    job_dir = _AUTO_OPTIMIZE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    status = RacecarJobStatus(
        job_id=job_id,
        phase="profiling",
        phase_progress="0/{}".format(len(req.models)),
        started_at=datetime.utcnow().isoformat(),
    )
    _write_status(job_dir, status)

    thread = threading.Thread(
        target=_run_auto_optimize,
        args=(job_dir, req, status),
        daemon=True,
    )
    thread.start()
    return status


@router.get("/auto-optimize/{job_id}/status", response_model=RacecarJobStatus)
def get_auto_optimize_status(job_id: str):
    """Poll the current phase and progress of an auto-optimize job."""
    job_dir = _AUTO_OPTIMIZE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    return _read_status(job_dir)


@router.get("/auto-optimize/{job_id}/results", response_model=RacecarJobResults)
def get_auto_optimize_results(job_id: str):
    """Get the final results of a completed auto-optimize job."""
    job_dir = _AUTO_OPTIMIZE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")

    results_path = job_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return RacecarJobResults(**json.load(f))

    status = _read_status(job_dir)
    return RacecarJobResults(
        job_id=job_id,
        status=status.phase,
        error=status.error,
    )


def _write_status(job_dir: Path, status: RacecarJobStatus):
    with open(job_dir / "status.json", "w") as f:
        json.dump(status.model_dump(), f, indent=2, default=str)


def _read_status(job_dir: Path) -> RacecarJobStatus:
    with open(job_dir / "status.json") as f:
        return RacecarJobStatus(**json.load(f))


def _update_phase(job_dir: Path, phase: str, progress: str = ""):
    s = _read_status(job_dir)
    s.phase = phase
    s.phase_progress = progress
    _write_status(job_dir, s)


def _run_auto_optimize(job_dir: Path, req: RacecarAutoOptimizeRequest,
                       status: RacecarJobStatus):
    """Background thread: execute the full Racecar pipeline."""
    t_start = datetime.utcnow()
    try:
        # ── Phase B: Expert Profiler ──
        _update_phase(job_dir, "profiling",
                       f"running {len(req.models)} models")

        from pipeline.expert_profiler import ExpertProfiler, RegimeConfig

        csv_path = Path(f"csv_data/{req.pair}_10_years_{req.timeframe}_OANDA.csv")
        if not csv_path.exists():
            raise HTTPException(400, f"Data not found: {csv_path}")

        profiler = ExpertProfiler(
            data_config={
                "symbol": req.pair,
                "csv_data_path": str(csv_path),
            },
            wfo_config={
                "n_months": req.train_months,
                "n_trials": req.profile_trials,
                "hpo_mode": "static",
                "smoke_test": True,
            },
            regime_cfg=RegimeConfig(),
        )

        profile_result = profiler.profile(
            models=req.models,
            n_months=req.train_months,
            n_trials=req.profile_trials,
            seed=42,
            verbose=False,
        )

        matrix = profile_result.matrix
        if matrix is None or not matrix.models:
            _update_phase(job_dir, "failed", "No models produced valid fold results")
            return

        # Save matrix
        matrix_data = matrix.to_dict()
        with open(job_dir / "regime_matrix.json", "w") as f:
            json.dump(matrix_data, f, indent=2, default=str)

        # ── Phase C: Committee Builder ──
        _update_phase(job_dir, "building", "")

        from pipeline.committee_builder import CommitteeBuilder

        builder = CommitteeBuilder(
            top_k=req.committee_top_k,
            weight_type="sharpe_proportional",
        )
        committee_config = builder.build(
            matrix,
            constraints={
                "max_models_per_regime": req.committee_top_k,
                "min_sharpe": -0.5,
                "min_trades": 3,
            },
        )

        with open(job_dir / "committee_config.json", "w") as f:
            json.dump(committee_config.to_dict(), f, indent=2, default=str)

        # ── Phase D: Committee Backtester ──
        _update_phase(job_dir, "backtesting", "")

        from pipeline.committee_backtester import CommitteeBacktester

        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        if "returns" not in df.columns:
            df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        bt = CommitteeBacktester(
            committee_config,
            regime_cfg=RegimeConfig(),
            confidence_threshold=0.5,
        )
        bt_result = bt.run_wfo(
            df,
            train_months=req.train_months,
            test_months=req.test_months,
            verbose=False,
        )

        # ── Save results ──
        results = {
            "job_id": status.job_id,
            "status": "completed",
            "profile_matrix": matrix_data,
            "committee_config": committee_config.to_dict(),
            "backtest": bt_result.to_summary_dict(),
            "total_time_s": (datetime.utcnow() - t_start).total_seconds(),
        }
        with open(job_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        _update_phase(job_dir, "completed", "")
        print(f"[RACECAR] Auto-optimize {status.job_id} completed in {results['total_time_s']:.1f}s")

    except Exception as e:
        _update_phase(job_dir, "failed", "")
        s = _read_status(job_dir)
        s.error = str(e)
        _write_status(job_dir, s)
        print(f"[RACECAR] Auto-optimize {status.job_id} failed: {e}")


# ── Factory: Iterative Committee Optimizer ──────────────────────────

class FactoryStartRequest(BaseModel):
    models: List[str] = Field(default_factory=lambda: [
        "logistic", "random_forest", "xgboost",
    ])
    proposer: str = "llm"
    llm_backend: str = "deepseek"
    max_iterations: int = 20
    patience: int = 5
    stopping_tolerance: float = 0.02
    regime_sharpe_floor: float = 0.3
    train_months: int = 6


class FactoryIterationRecordOut(BaseModel):
    iteration: int
    action_type: str = ""
    regime: str = ""
    model_add: str = ""
    model_remove: str = ""
    before_sharpe: float = 0.0
    after_sharpe: float = 0.0
    delta_sharpe: float = 0.0
    accepted: bool = False
    rationale: str = ""


class FactoryStatusResponse(BaseModel):
    job_id: str
    phase: str  # "starting", "running", "completed", "failed"
    iteration: int = 0
    total_iterations: int = 0
    current_action: str = ""
    current_regime: str = ""
    before_sharpe: float = 0.0
    after_sharpe: float = 0.0
    delta_sharpe: float = 0.0
    accepted: bool = False
    best_sharpe_so_far: float = 0.0
    stopped: bool = False
    stop_reason: str = ""
    history: List[FactoryIterationRecordOut] = Field(default_factory=list)


class FactoryResultsResponse(BaseModel):
    job_id: str
    status: str
    best_sharpe: float = 0.0
    total_iterations: int = 0
    accepted_count: int = 0
    total_time_s: float = 0.0
    best_config: Optional[Dict[str, Any]] = None
    history: List[FactoryIterationRecordOut] = Field(default_factory=list)
    stop_reason: str = ""


_FACTORY_DIR = Path("results/factory_jobs")
_FACTORY_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/factory/start", response_model=FactoryStatusResponse)
def start_factory_job(req: FactoryStartRequest):
    job_id = f"factory_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    job_dir = _FACTORY_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    status = FactoryStatusResponse(
        job_id=job_id,
        phase="starting",
        total_iterations=req.max_iterations,
    )
    _write_json(job_dir / "status.json", status.model_dump())

    thread = threading.Thread(
        target=_run_factory_job,
        args=(job_dir, job_id, req),
        daemon=True,
    )
    thread.start()
    return status


@router.get("/factory/{job_id}/status", response_model=FactoryStatusResponse)
def get_factory_status(job_id: str):
    job_dir = _FACTORY_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    data = _read_json(job_dir / "status.json")
    return FactoryStatusResponse(**data)


@router.get("/factory/{job_id}/results", response_model=FactoryResultsResponse)
def get_factory_results(job_id: str):
    job_dir = _FACTORY_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    path = job_dir / "results.json"
    if path.exists():
        data = _read_json(path)
        return FactoryResultsResponse(**data)
    status_data = _read_json(job_dir / "status.json")
    return FactoryResultsResponse(
        job_id=job_id,
        status=status_data.get("phase", "unknown"),
        history=status_data.get("history", []),
    )


def _write_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _run_factory_job(job_dir: Path, job_id: str, req: FactoryStartRequest):
    t_start = datetime.utcnow()
    try:
        from pipeline.factory_state import load_state_from_disk
        from pipeline.factory_executor import FactoryExecutor

        # Ensure profiler data exists — run Racecar if needed
        matrix_path = Path("results/racecar/regime_model_matrix.json")
        config_path = Path("results/racecar/committee_config.json")

        if not matrix_path.exists() or not config_path.exists():
            from pipeline.factory_llm import create_llm_proposer
            _update_factory_status(job_dir, "starting", history=[],
                                   current_action="Running Racecar first...")
            _run_racecar_backend(req, matrix_path, config_path)

        state = load_state_from_disk(
            config_path=str(config_path),
            matrix_path=str(matrix_path),
            patience=req.patience,
            tolerance=req.stopping_tolerance,
            floor=req.regime_sharpe_floor,
            max_iter=req.max_iterations,
        )
        if state is None:
            _update_factory_status(job_dir, "failed", stopped=True,
                                   stop_reason="Failed to load state")
            return

        proposer = None
        if req.proposer == "llm":
            from pipeline.factory_llm import create_llm_proposer
            proposer = create_llm_proposer(backend=req.llm_backend)

        executor = FactoryExecutor(
            state=state,
            proposer=proposer,
            train_months=req.train_months,
            test_months=1,
        )

        _update_factory_status(job_dir, "running", current_action="Starting iteration loop")

        # Run loop manually to capture per-iteration status
        executor._load_data()
        _loop_proposer = executor.proposer
        stop_reason = ""
        while True:
            should_stop, reason = state.should_stop()
            if should_stop:
                stop_reason = reason
                _update_factory_status(job_dir, "completed", stopped=True,
                                       stop_reason=reason, history=_state_to_history(state))
                break

            proposal = _loop_proposer.propose(state)
            if proposal.type == "halt":
                stop_reason = "No more untested moves"
                _update_factory_status(job_dir, "completed", stopped=True,
                                       stop_reason=stop_reason,
                                       history=_state_to_history(state))
                break

            current_action = (f"{proposal.type} {proposal.model_add or proposal.model_remove or ''}"
                             .strip())
            _update_factory_status(job_dir, "running",
                                   current_action=current_action,
                                   current_regime=proposal.regime,
                                   iteration=state.iteration + 1)

            record, _ = executor.execute_iteration(proposal)
            if record is None:
                continue

            _update_factory_status(job_dir, "running",
                                   iteration=state.iteration,
                                   current_action=f"{proposal.type} in {proposal.regime}",
                                   before_sharpe=record.before_sharpe,
                                   after_sharpe=record.after_sharpe,
                                   delta_sharpe=record.after_sharpe - record.before_sharpe,
                                   accepted=record.accepted,
                                   best_sharpe_so_far=state.global_best_sharpe,
                                   history=_state_to_history(state))

        # Save final results
        results = {
            "job_id": job_id,
            "status": "completed",
            "best_sharpe": state.global_best_sharpe,
            "total_iterations": state.iteration,
            "accepted_count": sum(1 for r in state.history if r.accepted),
            "total_time_s": (datetime.utcnow() - t_start).total_seconds(),
            "best_config": state.global_best_config,
            "history": [r.to_dict() for r in state.history],
            "stop_reason": reason,
        }
        _write_json(job_dir / "results.json", results)

    except Exception as e:
        _update_factory_status(job_dir, "failed", stopped=True,
                               stop_reason=str(e))
        import traceback
        traceback.print_exc()


def _update_factory_status(job_dir: Path, phase: str, **kwargs):
    data = {}
    status_path = job_dir / "status.json"
    if status_path.exists():
        data = _read_json(status_path)
    data["phase"] = phase
    data.update(kwargs)
    _write_json(status_path, data)


def _state_to_history(state) -> list:
    history = []
    for rec in state.history[-15:]:
        hist = {
            "iteration": rec.iteration,
            "action_type": rec.action.get("type", ""),
            "regime": rec.action.get("regime", ""),
            "model_add": rec.action.get("model_add", ""),
            "model_remove": rec.action.get("model_remove", ""),
            "before_sharpe": rec.before_sharpe,
            "after_sharpe": rec.after_sharpe,
            "delta_sharpe": rec.after_sharpe - rec.before_sharpe,
            "accepted": rec.accepted,
            "rationale": rec.rationale,
        }
        history.append(hist)
    return history


def _run_racecar_backend(req, matrix_path, config_path):
    """Internal: run Racecar pipeline to generate initial profiler data."""
    from pipeline.expert_profiler import ExpertProfiler, RegimeConfig
    from pipeline.committee_builder import CommitteeBuilder

    csv_path = Path(f"csv_data/{req.pair}_10_years_{req.timeframe}_OANDA.csv")
    if not csv_path.exists():
        csv_path = Path("csv_data/EURUSD_10_years_H1_OANDA.csv")

    profiler = ExpertProfiler(
        data_config={"symbol": "EURUSD", "csv_data_path": str(csv_path)},
        wfo_config={"n_months": req.train_months, "n_trials": 3,
                     "hpo_mode": "static", "smoke_test": True},
        regime_cfg=RegimeConfig(),
    )
    result = profiler.profile(models=req.models, n_months=req.train_months,
                               n_trials=3, seed=42, verbose=False)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.save_matrix(result, str(matrix_path))
    builder = CommitteeBuilder(top_k=3, weight_type="sharpe_proportional")
    config = builder.build(result.matrix, constraints={"max_models_per_regime": 3})
    config.to_json(str(config_path))
