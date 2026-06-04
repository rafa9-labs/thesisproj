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
        if "mid_close" in df.columns and "mid_c" not in df.columns:
            df = df.rename(columns={
                "mid_open": "mid_o", "mid_high": "mid_h",
                "mid_low": "mid_l", "mid_close": "mid_c",
            })
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
        "logistic", "svm", "random_forest", "xgboost",
        "lightgbm", "catboost", "lstm", "ensemble_adaptive_regime",
    ])
    pair: str = "EURUSD"
    timeframe: str = "H1"
    train_months: int = 36
    test_months: int = 1
    profile_trials: int = 30
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
                "hpo_sampler": "tpe",
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
            weight_method="sharpe_proportional",
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
        "logistic", "svm", "random_forest", "xgboost",
        "lightgbm", "catboost", "lstm", "ensemble_adaptive_regime",
    ])
    proposer: str = "llm"
    llm_backend: str = "deepseek"
    max_iterations: int = 20
    patience: int = 5
    stopping_tolerance: float = 0.02
    regime_sharpe_floor: float = 0.3
    train_months: int = 36
    pair: str = "EURUSD"
    timeframe: str = "H1"


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

        csv_path = str(Path(f"csv_data/{req.pair}_10_years_{req.timeframe}_OANDA.csv"))
        if not Path(csv_path).exists():
            csv_path = "csv_data/EURUSD_10_years_H1_OANDA.csv"

        executor = FactoryExecutor(
            state=state,
            proposer=proposer,
            data_path=csv_path,
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
        data_config={
            "symbol": req.pair,
            "csv_data_path": str(csv_path),
            "timeframe": req.timeframe,
            "start": None,
            "end": None,
        },
        wfo_config={
            "n_months": req.train_months,
            "n_trials": max(getattr(req, "profile_trials", 30), 30),
            "hpo_mode": "static",
            "hpo_sampler": "tpe",
            "label_threshold": 0.0,
            "confidence_threshold": 0.5,
        },
        regime_cfg=RegimeConfig(),
    )
    result = profiler.profile(models=req.models, n_months=req.train_months,
                                n_trials=max(getattr(req, "profile_trials", 30), 30),
                                seed=42, verbose=False)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.save_matrix(result, str(matrix_path))
    builder = CommitteeBuilder(top_k=3, weight_method="sharpe_proportional")
    config = builder.build(result.matrix, constraints={"max_models_per_regime": 3})
    config.to_json(str(config_path))


# ══════════════════════════════════════════════════════════════════════
# Full Cycle: Racecar (B→C→D) + Factory (optimization) in one shot
# ══════════════════════════════════════════════════════════════════════

class FullCycleRequest(BaseModel):
    models: List[str] = Field(default_factory=lambda: [
        "logistic", "svm", "random_forest", "xgboost",
        "lightgbm", "catboost", "lstm", "ensemble_adaptive_regime",
    ])
    pair: str = "EURUSD"
    timeframe: str = "H1"
    profile_trials_phase0: int = 5
    sweep_n_estimators: int = 100
    sweep_max_depth: int = 5
    committee_top_k: int = 3
    train_months: int = 36
    test_months: int = 1
    hpo_sampler: str = "tpe"
    cv_blocks: int = 3
    cv_val_frac: float = 0.05
    plateau_patience: int = 15
    max_surviving_models: int = 7
    proposer: str = "llm"
    llm_backend: str = "deepseek"
    max_iterations: int = 20
    patience: int = 5
    stopping_tolerance: float = 0.02
    regime_sharpe_floor: float = 0.3
    factory_proxy_months: int = 36
    factory_proxy_folds: int = 3


class FullCycleStatusResponse(BaseModel):
    job_id: str
    phase: str = "starting"
    phase_number: int = 0
    phase_progress: str = ""
    iteration: int = 0
    total_iterations: int = 0
    current_action: str = ""
    best_sharpe_so_far: float = 0.0
    started_at: str = ""
    error: str = ""
    pruned_models: List[str] = Field(default_factory=list)
    surviving_models: List[str] = Field(default_factory=list)
    locked_features_count: int = 0


class FullCycleResultsResponse(BaseModel):
    job_id: str
    status: str = ""
    locked_features_count: int = 0
    pruned_features_count: int = 0
    top_importance_feature: str = ""
    phase0_pruned: List[str] = Field(default_factory=list)
    phase0_survivors: List[str] = Field(default_factory=list)
    racecar_profile_matrix: Optional[Dict[str, Any]] = None
    racecar_committee_config: Optional[Dict[str, Any]] = None
    racecar_backtest: Optional[Dict[str, Any]] = None
    phase3_fold_consistency_cv: float = 0.0
    phase3_fold_consistency_pass: bool = False
    phase3_regime_coverage: Optional[Dict[str, Any]] = None
    phase3_seed_robustness_sharpe: float = 0.0
    phase3_seed_robustness_seeds: int = 3
    phase3_seed_robustness_pass: bool = False
    final_fold_consistency_cv: float = 0.0
    final_fold_consistency_pass: bool = False
    final_regime_coverage: Optional[Dict[str, Any]] = None
    final_seed_robustness_sharpe: float = 0.0
    final_seed_robustness_pass: bool = False
    final_full_wfo: Optional[Dict[str, Any]] = None
    factory_best_sharpe: float = 0.0
    factory_total_iterations: int = 0
    factory_accepted_count: int = 0
    factory_best_config: Optional[Dict[str, Any]] = None
    factory_history: List[Dict[str, Any]] = Field(default_factory=list)
    factory_stop_reason: str = ""
    total_time_s: float = 0.0


class FullCycleHistoryEntry(BaseModel):
    job_id: str
    started_at: str = ""
    status: str = "unknown"
    total_time_s: float = 0.0
    locked_features_count: int = 0
    survivors_count: int = 0
    survivors: List[str] = Field(default_factory=list)
    avg_sharpe: float = 0.0
    phase3_passed: bool = False
    factory_best_sharpe: float = 0.0


class FullCycleHistoryResponse(BaseModel):
    entries: List[FullCycleHistoryEntry] = Field(default_factory=list)
    total_runs: int = 0


_FULL_CYCLE_DIR = Path("results/full_cycle")
_FULL_CYCLE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/full-cycle", response_model=FullCycleStatusResponse)
def start_full_cycle(req: FullCycleRequest):
    """Full cycle: Racecar (B→C→D) → Factory (optimization) → best config saved."""
    job_id = f"fullcycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    job_dir = _FULL_CYCLE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.utcnow().isoformat()
    status = FullCycleStatusResponse(
        job_id=job_id, phase="starting", total_iterations=req.max_iterations,
        started_at=started_at,
    )
    _write_json(job_dir / "status.json", status.model_dump())

    thread = threading.Thread(
        target=_run_full_cycle,
        args=(job_dir, job_id, req, started_at),
        daemon=True,
    )
    thread.start()
    return status


@router.get("/full-cycle/history", response_model=FullCycleHistoryResponse)
def get_full_cycle_history():
    """List all past full cycle runs with key summary metrics."""
    entries: List[FullCycleHistoryEntry] = []
    if not _FULL_CYCLE_DIR.exists():
        return FullCycleHistoryResponse(entries=[], total_runs=0)

    for job_dir in sorted(_FULL_CYCLE_DIR.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        status_path = job_dir / "status.json"
        results_path = job_dir / "results.json"
        if not status_path.exists():
            continue

        try:
            status_data = _read_json(status_path)
            results_data = _read_json(results_path) if results_path.exists() else {}
        except Exception:
            continue

        entries.append(FullCycleHistoryEntry(
            job_id=job_dir.name,
            started_at=status_data.get("started_at", ""),
            status=status_data.get("phase", "unknown"),
            total_time_s=float(results_data.get("total_time_s") or 0.0),
            locked_features_count=int(results_data.get("locked_features_count") or 0),
            survivors_count=len(results_data.get("phase0_survivors") or []),
            survivors=results_data.get("phase0_survivors") or [],
            avg_sharpe=float(results_data.get("phase3_seed_robustness_sharpe") or 0.0),
            phase3_passed=bool(results_data.get("phase3_fold_consistency_pass", False)),
            factory_best_sharpe=float(results_data.get("factory_best_sharpe") or 0.0),
        ))

    return FullCycleHistoryResponse(entries=entries, total_runs=len(entries))


@router.get("/full-cycle/{job_id}/status", response_model=FullCycleStatusResponse)
def get_full_cycle_status(job_id: str):
    job_dir = _FULL_CYCLE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    return FullCycleStatusResponse(**_read_json(job_dir / "status.json"))


@router.get("/full-cycle/{job_id}/results", response_model=FullCycleResultsResponse)
def get_full_cycle_results(job_id: str):
    job_dir = _FULL_CYCLE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    path = job_dir / "results.json"
    if path.exists():
        return FullCycleResultsResponse(**_read_json(path))
    return FullCycleResultsResponse(
        job_id=job_id,
        status=_read_json(job_dir / "status.json").get("phase", "unknown"),
    )


def _update_full_cycle_status(job_dir: Path, phase: str, phase_number: int = 0, **kwargs):
    data = {}
    status_path = job_dir / "status.json"
    if status_path.exists():
        data = _read_json(status_path)
    data["phase"] = phase
    data["phase_number"] = phase_number
    data.update(kwargs)
    _write_json(status_path, data)


def _load_csv_for_committee(pair: str, timeframe: str) -> tuple[Path, pd.DataFrame]:
    """Resolve CSV path and return (path, normalized DataFrame)."""
    csv_path = Path(f"csv_data/{pair}_10_years_{timeframe}_OANDA.csv")
    if not csv_path.exists():
        csv_path = Path("csv_data/EURUSD_10_years_H1_OANDA.csv")
    df = pd.read_csv(csv_path)
    for time_col in ("timestamp", "time"):
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            break
    rename_map = {
        "mid_open": "mid_o", "mid_high": "mid_h",
        "mid_low": "mid_l", "mid_close": "mid_c",
    }
    df = df.rename(columns=rename_map)
    if "returns" not in df.columns:
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)
    return csv_path, df


def _run_full_cycle(job_dir: Path, job_id: str, req: FullCycleRequest, started_at: str):
    t_start = datetime.utcnow()
    try:
        from pipeline.expert_profiler import (
            ExpertProfiler, RegimeConfig, prune_models,
        )
        from pipeline.committee_builder import CommitteeBuilder
        from pipeline.committee_backtester import CommitteeBacktester
        from pipeline.model_families import get_trial_budget
        from pipeline.factory_state import load_state_from_disk
        from pipeline.factory_executor import FactoryExecutor

        csv_path, df = _load_csv_for_committee(req.pair, req.timeframe)

        # ──────────────────────────────────────────────────────────────
        # PHASE -1: FEATURE SWEEP (if locked_features.json doesn't exist)
        # ──────────────────────────────────────────────────────────────
        locked_features_path = Path("results/locked_features.json")
        sweep_report_path = locked_features_path.with_name("locked_features_report.json")
        locked_features = None
        sweep_report_data: Dict[str, Any] = {}

        if locked_features_path.exists():
            try:
                from pipeline.feature_sweep import load_locked_features
                locked_features = load_locked_features(str(locked_features_path))
                if locked_features:
                    print(f"[FULL_CYCLE] Loaded {len(locked_features)} locked features")
                if sweep_report_path.exists():
                    with open(sweep_report_path) as f:
                        sweep_report_data = json.load(f)
            except Exception:
                pass

        if not locked_features:
            from pipeline.feature_sweep import run_phase_minus1
            _update_full_cycle_status(job_dir, "feature_sweep", phase_number=-1,
                                       current_action="Phase -1: expanding features, training shallow RF")
            locked_features, sweep_report_data = run_phase_minus1(
                df, output_path=str(locked_features_path),
                label_threshold=0.0001,
                n_estimators=req.sweep_n_estimators,
                max_depth=req.sweep_max_depth,
                n_folds=3, random_state=42,
            )
            print(f"[FULL_CYCLE] Phase -1 complete: {len(locked_features)} features locked")

        locked_count = len(locked_features) if locked_features else 0
        pruned_count = sweep_report_data.get("pruned_count", 0)
        top_feature = (sweep_report_data.get("locked_features") or [""])[0]

        if locked_features:
            os.environ["MLB_TA_MODE"] = "fixed"
            print("[FULL_CYCLE] TA_MODE locked to 'fixed' (features pre-filtered by Phase -1)")

        # Prepare raw OHLC DataFrame for anchored regime detection (needs 'time' column)
        raw_df_regime = df.reset_index() if df.index.name or "time" not in df.columns else df.copy()
        if "time" not in raw_df_regime.columns:
            raw_df_regime["time"] = raw_df_regime.index

        # ──────────────────────────────────────────────────────────────
        # PHASE 0: PRE-SCREENING (static pass across all candidate models)
        # ──────────────────────────────────────────────────────────────
        def _on_profile_progress(model, idx, total, status, sharpe=None):
            _update_full_cycle_status(
                job_dir, "profiling", phase_number=0,
                current_action=f"Profiling {model} ({idx}/{total})",
                phase_progress=f"{idx}/{total}", iteration=0,
                best_sharpe_so_far=float(sharpe) if sharpe is not None else 0.0,
            )

        _update_full_cycle_status(job_dir, "prescreening", phase_number=0,
                                   phase_progress=f"0/{len(req.models)}",
                                   current_action="Phase 0: screening all models",
                                   locked_features_count=locked_count)
        profiler = ExpertProfiler(
            data_config={"symbol": req.pair, "csv_data_path": str(csv_path)},
            wfo_config={
                "n_months": req.train_months,
                "n_trials": req.profile_trials_phase0,
                "hpo_mode": "static",
                "hpo_sampler": req.hpo_sampler,
                "cv_blocks": req.cv_blocks,
                "cv_val_frac": req.cv_val_frac,
                "plateau_patience": req.plateau_patience,
                "locked_features": locked_features,
            },
            regime_cfg=RegimeConfig(),
        )
        phase0_result = profiler.profile(
            models=req.models, n_months=req.train_months,
            n_trials=req.profile_trials_phase0, seed=42, verbose=False,
            progress_callback=_on_profile_progress,
            raw_df=raw_df_regime,
        )
        matrix = phase0_result.matrix
        if matrix is None or not matrix.models:
            raise RuntimeError("Phase 0: no models produced valid fold results")

        survivors, pruned = prune_models(
            matrix, min_sharpe=0.0, max_models=req.max_surviving_models,
        )
        if not survivors:
            raise RuntimeError(
                f"Phase 0: all {len(req.models)} models failed screening. "
                "Tighten max_surviving_models or add better models."
            )
        _update_full_cycle_status(
            job_dir, "prescreening_complete", phase_number=0,
            current_action=f"{len(survivors)} survivors: {', '.join(survivors)}",
            surviving_models=survivors, pruned_models=pruned,
        )

        # ──────────────────────────────────────────────────────────────
        # PHASE 1: TARGETED HPO (real Optuna on each survivor)
        # ──────────────────────────────────────────────────────────────
        _update_full_cycle_status(job_dir, "tuning", phase_number=1,
                                   current_action="Starting Phase 1 HPO",
                                   phase_progress=f"0/{len(survivors)}")
        # Remove Phase 0 folds for survivors — will be replaced with tuned folds
        matrix.raw_folds = [f for f in matrix.raw_folds if f.model not in survivors]

        hpo_base_config = {
            "symbol": req.pair,
            "csv_data_path": str(csv_path),
            "timeframe": req.timeframe,
            "train_months": req.train_months,
            "test_months": req.test_months,
            "hpo_mode": "static",
            "hpo_sampler": req.hpo_sampler,
            "cv_blocks": req.cv_blocks,
            "cv_val_frac": req.cv_val_frac,
            "plateau_patience": req.plateau_patience,
            "locked_features": locked_features,
        }

        first_df_wfo = None
        hpo_model_params: Dict[str, dict] = {}
        for idx, model_type in enumerate(survivors):
            _update_full_cycle_status(
                job_dir, "tuning", phase_number=1,
                current_action=f"HPO on {model_type} ({idx + 1}/{len(survivors)})",
                phase_progress=f"{idx + 1}/{len(survivors)}",
            )
            n_trials, n_startup = get_trial_budget(model_type)
            model_config = dict(hpo_base_config)
            model_config["model_type"] = model_type
            tuned_folds, df_wfo, best_params = profiler._run_single_model(
                model_type, {**model_config, "n_trials": n_trials,
                             "n_startup_trials": n_startup},
                seed=42, verbose=False,
            )
            if tuned_folds:
                matrix.raw_folds.extend(tuned_folds)
            if best_params:
                hpo_model_params[model_type] = best_params
            if first_df_wfo is None and df_wfo is not None:
                first_df_wfo = df_wfo

        # Rebuild matrix with tuned fold results
        if matrix.raw_folds:
            profiler._attach_regime_distributions(matrix.raw_folds, first_df_wfo)
            matrix = profiler._build_matrix(matrix.raw_folds)

        matrix_data = matrix.to_dict()
        with open(job_dir / "regime_matrix_tuned.json", "w") as f:
            json.dump(matrix_data, f, indent=2, default=str)

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: COMMITTEE ASSEMBLY
        # ──────────────────────────────────────────────────────────────
        _update_full_cycle_status(job_dir, "building", phase_number=2,
                                   current_action="Building committee config")
        builder = CommitteeBuilder(
            top_k=req.committee_top_k, weight_method="sharpe_proportional",
        )
        committee_config = builder.build(
            matrix, constraints={"max_models_per_regime": req.committee_top_k},
        )
        cc_data = committee_config.to_dict()
        if hpo_model_params:
            committee_config.model_params = hpo_model_params
            cc_data["model_params"] = hpo_model_params
        with open(job_dir / "committee_config.json", "w") as f:
            json.dump(cc_data, f, indent=2, default=str)

        # ──────────────────────────────────────────────────────────────
        # PHASE 3: INTERMEDIATE VALIDATION (36-mo WFO + consistency + 3 seeds)
        # ──────────────────────────────────────────────────────────────
        _update_full_cycle_status(job_dir, "validating", phase_number=3,
                                   current_action="Phase 3: 36-month WFO validation")

        bt = CommitteeBacktester(
            committee_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
            model_params=hpo_model_params,
        )
        bt_result = bt.run_wfo(
            df, train_months=req.train_months, test_months=req.test_months,
            verbose=False,
        )

        cv = bt_result.fold_consistency_cv
        cv_pass = bt_result.fold_consistency_pass
        coverage = bt_result.regime_coverage_report(min_trades=30, min_sharpe=0.0)
        all_covered = all(c["covered"] for c in coverage.values())

        # Seed robustness (3 intermediate seeds)
        seed_sharpes = []
        for seed in (42, 101, 202):
            alt_bt = CommitteeBacktester(
                committee_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
                model_params=hpo_model_params,
            )
            alt_r = alt_bt.run_wfo(
                df, train_months=req.train_months, test_months=req.test_months,
                verbose=False,
            )
            seed_sharpes.append(alt_r.avg_sharpe if alt_r else 0.0)
        seed_avg = float(np.mean(seed_sharpes)) if seed_sharpes else 0.0
        seed_pass = all(s > 0.0 for s in seed_sharpes)

        phase3_passed = cv_pass and all_covered and seed_pass
        if not phase3_passed:
            reason_parts = []
            if not cv_pass:
                reason_parts.append(f"fold CV={cv:.3f} >= 1.0")
            if not all_covered:
                uncovered = [r for r, c in coverage.items() if not c["covered"]]
                reason_parts.append(f"regimes not covered: {uncovered}")
            if not seed_pass:
                reason_parts.append(f"seed Sharpes={[f'{s:.3f}' for s in seed_sharpes]}")
            fail_reason = "; ".join(reason_parts)
            _update_full_cycle_status(
                job_dir, "validation_failed", phase_number=3,
                error=f"Phase 3 gate failed: {fail_reason}",
            )
            elapsed = (datetime.utcnow() - t_start).total_seconds()
            results = FullCycleResultsResponse(
                job_id=job_id, status="validation_failed",
                locked_features_count=locked_count,
                pruned_features_count=pruned_count,
                top_importance_feature=top_feature,
                phase0_pruned=pruned, phase0_survivors=survivors,
                phase3_fold_consistency_cv=cv,
                phase3_fold_consistency_pass=cv_pass,
                phase3_regime_coverage=coverage,
                phase3_seed_robustness_sharpe=seed_avg,
                phase3_seed_robustness_seeds=3,
                phase3_seed_robustness_pass=seed_pass,
                total_time_s=elapsed,
            )
            with open(job_dir / "results.json", "w") as f:
                json.dump(results.model_dump(), f, indent=2, default=str)
            return

        # ──────────────────────────────────────────────────────────────
        # PHASE 4: FACTORY OPTIMIZATION (proxy WFO inside loop)
        # ──────────────────────────────────────────────────────────────
        _update_full_cycle_status(job_dir, "optimizing", phase_number=4,
                                   current_action="Starting Factory optimization (proxy WFO)",
                                   phase_progress=f"0/{req.max_iterations}",
                                   iteration=0, best_sharpe_so_far=0.0)

        state = load_state_from_disk(
            config_path=str(job_dir / "committee_config.json"),
            matrix_path=str(job_dir / "regime_matrix_tuned.json"),
            patience=req.patience, tolerance=req.stopping_tolerance,
            floor=req.regime_sharpe_floor, max_iter=req.max_iterations,
        )
        if state is None:
            raise RuntimeError("Failed to load Factory state from committee config")

        proposer = None
        if req.proposer == "llm":
            from pipeline.factory_llm import create_llm_proposer
            proposer = create_llm_proposer(backend=req.llm_backend)

        executor = FactoryExecutor(
            state=state, proposer=proposer, data_path=str(csv_path),
            train_months=req.factory_proxy_months, test_months=req.test_months,
        )

        executor._load_data()
        _loop_proposer = executor.proposer
        reason = ""
        while True:
            should_stop, reason = state.should_stop()
            if should_stop:
                break
            proposal = _loop_proposer.propose(state)
            if proposal.type == "halt":
                reason = "No more untested moves"
                break

            _update_full_cycle_status(
                job_dir, "optimizing", phase_number=4,
                current_action=f"{proposal.type} in {proposal.regime}",
                iteration=state.iteration + 1,
                phase_progress=f"{state.iteration + 1}/{req.max_iterations}",
            )
            record, _ = executor.execute_iteration(proposal)
            if record is None:
                continue
            _update_full_cycle_status(
                job_dir, "optimizing", phase_number=4,
                iteration=state.iteration,
                best_sharpe_so_far=state.global_best_sharpe,
                phase_progress=f"{state.iteration}/{req.max_iterations}",
            )

        # ──────────────────────────────────────────────────────────────
        # FINAL VALIDATION: full 10-year WFO + 5-seed robustness
        # ──────────────────────────────────────────────────────────────
        _update_full_cycle_status(job_dir, "final validation", phase_number=4,
                                   current_action="Final: 10-year WFO + 5-seed robustness")

        final_config = state.config if state.config else committee_config
        final_bt = CommitteeBacktester(
            final_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
            model_params=hpo_model_params,
        )
        final_result = final_bt.run_wfo(
            df, train_months=req.train_months, test_months=req.test_months,
            verbose=False,
        )
        final_summary = final_result.to_summary_dict() if final_result else {}

        # 5-seed robustness on the final config
        final_seed_sharpes = []
        for seed in (42, 101, 202, 789, 999):
            fbt = CommitteeBacktester(
                final_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
                model_params=hpo_model_params,
            )
            fr = fbt.run_wfo(
                df, train_months=req.train_months, test_months=req.test_months,
                verbose=False,
            )
            final_seed_sharpes.append(fr.avg_sharpe if fr else 0.0)
        final_seed_avg = float(np.mean(final_seed_sharpes)) if final_seed_sharpes else 0.0
        final_seed_pass = all(s > 0.0 for s in final_seed_sharpes)

        final_fold_cv = final_result.fold_consistency_cv if final_result else float("inf")
        final_fold_pass = final_result.fold_consistency_pass if final_result else False
        final_coverage = final_result.regime_coverage_report(min_trades=30, min_sharpe=0.0) if final_result else {}

        # Save final committee config to disk (deployment)
        final_config.to_json(str(job_dir / "committee_config_final.json"))

        # ── Save committee snapshot (MLOps reproducibility) ──
        # Train all committee models on full history and save exact weights
        # so deployment loads byte-for-byte identical estimators.
        unique_models = list(set(final_config.all_models()))
        snapshot_dir = job_dir / "committee_snapshot"
        snapshot_saved = False
        try:
            from pipeline.feature_sweep import compute_feature_matrix, FEATURE_NAMES
            from models.registry import build_model
            from pipeline.expert_profiler import _reprefix_params
            import pandas as pd
            import numpy as np

            snapshot_feature_names = locked_features if locked_features else FEATURE_NAMES

            raw_full = pd.read_csv(csv_path)
            for time_col in ("timestamp", "time"):
                if time_col in raw_full.columns:
                    raw_full[time_col] = pd.to_datetime(raw_full[time_col])
                    raw_full = raw_full.set_index(time_col)
                    break
            raw_full = raw_full.rename(columns={
                "mid_open": "mid_o", "mid_high": "mid_h",
                "mid_low": "mid_l", "mid_close": "mid_c",
            })

            feature_matrix = compute_feature_matrix(raw_full, snapshot_feature_names, include_ohlc=False)
            feature_matrix = feature_matrix.dropna()
            X_full = feature_matrix.to_numpy(np.float32)

            fwd = np.log(raw_full.loc[feature_matrix.index, "mid_c"]
                         / raw_full.loc[feature_matrix.index, "mid_c"].shift(1))
            y_full = np.ones(len(X_full), dtype=np.int32)
            threshold = 0.0001
            y_full[np.isfinite(fwd.values) & (fwd.values > threshold)] = 2
            y_full[np.isfinite(fwd.values) & (fwd.values < -threshold)] = 0
            y_full[-1] = 1

            valid = np.isfinite(y_full)
            X_full, y_full = X_full[valid], y_full[valid]

            snapshot_dir.mkdir(parents=True, exist_ok=True)

            for mtype in unique_models:
                try:
                    params = hpo_model_params.get(mtype, {})
                    model = build_model(
                        mtype, use_proba=True, n_features=X_full.shape[1],
                        **_reprefix_params(params, mtype),
                    )
                    model.fit(X_full, y_full)
                    is_tf = hasattr(model, "save") and callable(getattr(model, "save", None))
                    is_tf = is_tf and not hasattr(model, "get_params")
                    if is_tf:
                        tf_path = str(snapshot_dir / f"{mtype}_tf")
                        model.save(tf_path, save_format="tf")
                    else:
                        import joblib
                        joblib.dump(model, str(snapshot_dir / f"{mtype}.joblib"))
                except Exception:
                    import traceback
                    traceback.print_exc()

            # Save metadata
            import json as _json
            meta = {
                "feature_names": list(snapshot_feature_names),
                "models": unique_models,
                "model_params": hpo_model_params,
                "trained_at": datetime.utcnow().isoformat(),
            }
            with open(snapshot_dir / "manifest.json", "w") as f:
                _json.dump(meta, f, indent=2, default=str)
            snapshot_saved = True
        except Exception:
            import traceback
            traceback.print_exc()

        # ── Assemble results ──
        factory_history = [r.to_dict() for r in state.history]
        results = {
            "job_id": job_id,
            "status": "completed",
            "locked_features_count": locked_count,
            "pruned_features_count": pruned_count,
            "top_importance_feature": top_feature,
            "phase0_pruned": pruned,
            "phase0_survivors": survivors,
            "racecar_profile_matrix": matrix_data,
            "racecar_committee_config": cc_data,
            "racecar_backtest": bt_result.to_summary_dict(),
            "phase3_fold_consistency_cv": round(float(cv), 4),
            "phase3_fold_consistency_pass": bool(cv_pass),
            "phase3_regime_coverage": {r: dict(v) for r, v in coverage.items()},
            "phase3_seed_robustness_sharpe": round(float(seed_avg), 4),
            "phase3_seed_robustness_seeds": 3,
            "phase3_seed_robustness_pass": bool(seed_pass),
            "final_full_wfo": final_summary,
            "final_fold_consistency_cv": round(float(final_fold_cv), 4),
            "final_fold_consistency_pass": bool(final_fold_pass),
            "final_regime_coverage": {r: dict(v) for r, v in final_coverage.items()},
            "final_seed_robustness_sharpe": round(float(final_seed_avg), 4),
            "final_seed_robustness_pass": bool(final_seed_pass),
            "factory_best_sharpe": state.global_best_sharpe,
            "factory_total_iterations": state.iteration,
            "factory_accepted_count": sum(1 for r in state.history if r.accepted),
            "factory_best_config": state.global_best_config,
            "factory_history": factory_history,
            "factory_stop_reason": reason,
            "total_time_s": (datetime.utcnow() - t_start).total_seconds(),
            "snapshot_dir": str(snapshot_dir) if snapshot_saved else None,
            "hpo_model_params_count": len(hpo_model_params),
        }
        with open(job_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        _update_full_cycle_status(
            job_dir, "completed", phase_number=4,
            best_sharpe_so_far=state.global_best_sharpe,
        )

    except Exception as e:
        _update_full_cycle_status(job_dir, "failed", phase_number=-1, error=str(e))
        import traceback
        traceback.print_exc()
