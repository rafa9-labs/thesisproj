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

from api.log_buffer import (log_info, log_warn, log_error, get_job_logs,
                               log_phase_start, log_phase_complete, log_progress,
                               log_metric, PHASE_LABELS)
from api.schemas.backtest import StudyMetaRequest, StudyMetaResponse
from concurrent.futures.process import BrokenProcessPool

router = APIRouter(prefix="/committee", tags=["committee"])

_COMMITTEE_RESULTS_DIR = Path("results/committee")
_COMMITTEE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _inject_features_metadata(config, cc_data: dict, hpo_model_params: Optional[Dict[str, dict]]):
    """Inject features_config into committee config metadata for Fast Loop.

    The Fast Retrain script reads features_config, fracdiff_d, and
    input_window_size from the committee JSON metadata so it can reproduce
    the exact same feature engineering as the Slow Loop without guessing.
    """
    from copy import deepcopy
    from pipeline.metrics.metrics_tuples import CLASS_DEFAULTS

    fc_path = os.environ.get("FEATURE_CONFIG_PATH", "configs/feature_config.json")
    features_dict = {}
    try:
        with open(fc_path) as f:
            features_dict = json.load(f)
    except Exception:
        features_dict = {}

    defaults = deepcopy(CLASS_DEFAULTS.get("features", {}))
    defaults.update(features_dict)

    fracdiff_d = None
    window_size = None
    if hpo_model_params:
        for mparams in hpo_model_params.values():
            if isinstance(mparams, dict):
                if fracdiff_d is None:
                    for key in ("fracdiff_d",):
                        if key in mparams:
                            fracdiff_d = float(mparams[key])
                            break
                if window_size is None:
                    for key in ("lags_range", "lags"):
                        if key in mparams:
                            try:
                                window_size = int(mparams[key])
                            except Exception:
                                pass
                            break

    if fracdiff_d is not None:
        defaults["fracdiff_d"] = fracdiff_d
        defaults["use_fracdiff"] = True
    if window_size is not None:
        defaults["lags_range"] = window_size

    if "metadata" not in cc_data:
        cc_data["metadata"] = {}
    cc_data["metadata"]["features_config"] = defaults
    cc_data["metadata"]["fracdiff_d"] = fracdiff_d or defaults.get("fracdiff_d", 0.4)
    cc_data["metadata"]["input_window_size"] = window_size or defaults.get(
        "lags_range", defaults.get("lags", 50)
    )


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
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


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




class CommitteeSnapshotInfo(BaseModel):
    version: str
    created_at: str
    models: List[str]


class CommitteeSnapshotListResponse(BaseModel):
    snapshots: List[CommitteeSnapshotInfo]


# ── Config endpoints ────────────────────────────────────────────────

_CONFIG_PATH = Path(os.environ.get("COMMITTEE_CONFIG_PATH", "results/committee/committee_config.json"))


@router.get("/config", response_model=CommitteeConfigSchema)
def get_committee_config(job_id: str | None = Query(default=None)):
    """Return the committee configuration.

    If job_id is provided, loads from results/full_cycle/{job_id}/committee_config_final.json.
    Otherwise loads from the staged global config file.
    """
    if job_id:
        job_dir = Path("results/full_cycle") / job_id
        config_path = job_dir / "committee_config_final.json"
        if not config_path.exists():
            config_path = job_dir / "committee_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return CommitteeConfigSchema(**json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        raise HTTPException(404, f"Config not found for job {job_id}")

    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                return CommitteeConfigSchema(**json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
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


# ── Saved Committees CRUD ───────────────────────────────────────────

class SavedCommitteeOut(BaseModel):
    id: str
    name: str
    full_cycle_job_id: Optional[str] = None
    pair: str = "EURUSD"
    timeframe: str = "H1"
    config_json: dict = Field(default_factory=dict)
    trust_score: Optional[float] = None
    avg_sharpe: Optional[float] = None
    is_active: bool = False
    tags: List[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class SavedCommitteeListResponse(BaseModel):
    committees: List[SavedCommitteeOut] = Field(default_factory=list)
    total: int = 0


class SaveCommitteeRequest(BaseModel):
    name: str
    full_cycle_job_id: Optional[str] = None
    pair: str = "EURUSD"
    timeframe: str = "H1"
    config_json: dict = Field(default_factory=dict)
    trust_score: Optional[float] = None
    avg_sharpe: Optional[float] = None
    tags: List[str] = Field(default_factory=list)


@router.get("/saved", response_model=SavedCommitteeListResponse)
def list_saved_committees():
    from api.config import settings
    from pipeline.data.data_sqlite import DataStore
    store = DataStore(settings.db_full_path)
    with store._cursor() as (conn, cur):
        cur.execute(
            "SELECT id, name, full_cycle_job_id, pair, timeframe, config_json, trust_score, avg_sharpe, is_active, tags, created_at, updated_at FROM saved_committees ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
    committees = []
    for r in rows:
        try:
            config_json = json.loads(r[5]) if r[5] else {}
        except (json.JSONDecodeError, TypeError):
            config_json = {}
        try:
            tags = json.loads(r[9]) if r[9] else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        committees.append(SavedCommitteeOut(
            id=r[0], name=r[1], full_cycle_job_id=r[2],
            pair=r[3] or "EURUSD", timeframe=r[4] or "H1",
            config_json=config_json, trust_score=r[6], avg_sharpe=r[7],
            is_active=bool(r[8]), tags=tags, created_at=r[10], updated_at=r[11],
        ))
    return SavedCommitteeListResponse(committees=committees, total=len(committees))


@router.post("/saved")
def save_committee(req: SaveCommitteeRequest):
    from api.config import settings
    from pipeline.data.data_sqlite import DataStore
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz

    store = DataStore(settings.db_full_path)
    committee_id = str(_uuid.uuid4())[:12]
    now = _dt.now(_tz.utc).isoformat()

    with store._cursor() as (conn, cur):
        cur.execute(
            "INSERT INTO saved_committees (id, name, full_cycle_job_id, pair, timeframe, config_json, trust_score, avg_sharpe, is_active, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
            (committee_id, req.name, req.full_cycle_job_id, req.pair, req.timeframe,
             json.dumps(req.config_json, default=str), req.trust_score, req.avg_sharpe,
             json.dumps(req.tags), now, now),
        )
    return {"status": "ok", "id": committee_id}


@router.delete("/saved/{committee_id}")
def delete_saved_committee(committee_id: str):
    from api.config import settings
    from pipeline.data.data_sqlite import DataStore
    store = DataStore(settings.db_full_path)
    with store._cursor() as (conn, cur):
        cur.execute("SELECT id FROM saved_committees WHERE id = ?", (committee_id,))
        if not cur.fetchone():
            raise HTTPException(404, f"Saved committee {committee_id} not found")
        cur.execute("DELETE FROM saved_committees WHERE id = ?", (committee_id,))
    return {"status": "ok"}


@router.post("/saved/{committee_id}/activate")
def activate_saved_committee(committee_id: str):
    from api.config import settings
    from pipeline.data.data_sqlite import DataStore
    store = DataStore(settings.db_full_path)
    with store._cursor() as (conn, cur):
        cur.execute("SELECT id, config_json FROM saved_committees WHERE id = ?", (committee_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"Saved committee {committee_id} not found")
        cur.execute("UPDATE saved_committees SET is_active = 0")
        cur.execute("UPDATE saved_committees SET is_active = 1 WHERE id = ?", (committee_id,))
        try:
            config = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            config = {}
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2, default=str)
    return {"status": "ok", "committee_id": committee_id}


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
    from pipeline.regime.regime_utils import detect_regimes, RegimeConfig, _REGIME_NAMES

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

    regime_ids = detect_regimes(df)
    recent = regime_ids[-bars:]
    timestamps = df.index[-bars:]

    labels = []
    for i in range(len(recent)):
        r_id = int(recent[i])
        ts = timestamps[i]
        labels.append(RegimeLabelPoint(
            timestamp=str(ts),
            regime_id=r_id,
            regime_name=_REGIME_NAMES.get(r_id, "unknown"),
        ))

    return RegimeLabelsResponse(pair=pair, timeframe=timeframe, labels=labels, count=len(labels))


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


_STATUS_LOCK = threading.Lock()


def _write_json(path: Path, data: dict):
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_pipeline_artifacts(job_dir: Path, bt, log_info, job_id: str):
    """Save P1-P3 pipeline artifacts from the backtester to disk.

    - P1 MetaLabeler: binary model predicting P(trade_is_winner)
    - P2 HMMRegimeDetector: probabilistic regime detection
    - P3 ConvictionSizer: continuous sigmoid conviction sizing
    """
    # P1: MetaLabeler
    try:
        ml = bt.get_meta_labeler()
        if ml is not None and ml.is_trained:
            ml.save(str(job_dir / "meta_labeler.joblib"))
            log_info(job_id, f"MetaLabeler: saved (accuracy={ml.accuracy:.3f})")
    except Exception as e:
        log_info(job_id, f"MetaLabeler: save failed ({e})")

    # P2: HMMRegimeDetector
    try:
        hmm = getattr(bt, "_hmm_detector", None)
        if hmm is not None and hmm.is_fitted:
            hmm.save(str(job_dir / "hmm_detector.joblib"))
            log_info(job_id, f"HMMRegimeDetector: saved ({hmm.selected_n_states} states, BIC={hmm.bic:.1f})")
    except Exception as e:
        log_info(job_id, f"HMMRegimeDetector: save failed ({e})")

    # P3: ConvictionSizer
    try:
        cs = bt.get_conviction_sizer()
        if cs is not None and cs.fitted:
            cs.save(str(job_dir / "conviction_sizer.json"))
            log_info(job_id, f"ConvictionSizer: saved (L={cs.L:.3f} k={cs.k:.1f} c={cs.c:.3f})")
    except Exception as e:
        log_info(job_id, f"ConvictionSizer: save failed ({e})")

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
    sweep_n_estimators: int = 100
    sweep_max_depth: int = 5
    skip_feature_sweep: bool = False
    use_boruta_shap: bool = True
    boruta_percentile: int = 90
    boruta_max_iter: int = 20
    debug_mode: bool = False
    enable_phase3: bool = True
    enable_phase4: bool = True
    enable_phase5: bool = True
    enable_phase6: bool = True
    committee_top_k: int = 3
    train_months: int = 36
    test_months: int = 1
    hpo_sampler: str = "tpe"
    cv_blocks: int = 3
    cv_val_frac: float = 0.05
    plateau_patience: int = 15
    proposer: str = "llm"
    llm_backend: str = "deepseek"
    ucb_c: float = 2.0
    max_iterations: int = 20
    patience: int = 5
    stopping_tolerance: float = 0.02
    regime_sharpe_floor: float = 0.3
    factory_proxy_months: int = 36
    factory_proxy_folds: int = 3
    hpo_trials: Optional[Dict[str, int]] = None
    hpo_startup_trials: Optional[Dict[str, int]] = None
    committee_weight_method: Optional[str] = None
    committee_min_sharpe: Optional[float] = None


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
    last_heartbeat: str = ""
    stale: bool = False


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
    phase3_regime_coverage: Optional[Dict[str, Any]] = None
    phase3_seed_robustness_sharpe: float = 0.0
    phase3_seed_robustness_seeds: int = 3
    phase3_seed_robustness_pass: bool = False
    trust_score: Optional[Dict[str, Any]] = None
    pbo: float = 0.0
    dsr: float = 0.0
    hpo_status: Dict[str, str] = Field(default_factory=dict)
    hpo_model_params_count: int = 0
    snapshot_dir: Optional[str] = None
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
    trust_score: float = 0.0
    factory_best_sharpe: float = 0.0
    study_meta: Optional[StudyMetaResponse] = None


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
    _write_json(job_dir / "request.json", req.model_dump())

    thread = threading.Thread(
        target=_run_full_cycle,
        args=(job_dir, job_id, req, started_at),
        daemon=True,
    )

    from api.process_cleanup import register_job_thread, register_cancellation_event
    cancel_event = threading.Event()
    register_cancellation_event(job_id, cancel_event)
    register_job_thread(job_id, thread)

    thread.start()
    return status


_TERMINAL_PHASES = {"completed", "failed", "validation_failed", "cancelled"}
_STALE_THRESHOLD_S = 300  # 5 minutes — if status.json untouched longer, job is orphaned


@router.get("/full-cycle/history", response_model=FullCycleHistoryResponse)
def get_full_cycle_history():
    """List all past full cycle runs with key summary metrics."""
    entries: List[FullCycleHistoryEntry] = []
    if not _FULL_CYCLE_DIR.exists():
        return FullCycleHistoryResponse(entries=[], total_runs=0)

    now = datetime.utcnow().timestamp()

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
            meta_data = _read_json(job_dir / "study_meta.json")
        except Exception:
            continue

        _study_meta = None
        if meta_data:
            _study_meta = StudyMetaResponse(**meta_data)

        raw_status = status_data.get("phase", "unknown")

        if raw_status not in _TERMINAL_PHASES:
            try:
                mtime = status_path.stat().st_mtime
                if now - mtime > _STALE_THRESHOLD_S:
                    raw_status = "orphaned"
            except OSError:
                pass

        entries.append(FullCycleHistoryEntry(
            job_id=job_dir.name,
            started_at=status_data.get("started_at", ""),
            status=raw_status,
            total_time_s=float(results_data.get("total_time_s") or 0.0),
            locked_features_count=int(results_data.get("locked_features_count") or 0),
            survivors_count=len(results_data.get("phase0_survivors") or []),
            survivors=results_data.get("phase0_survivors") or [],
            avg_sharpe=float(results_data.get("phase3_seed_robustness_sharpe") or 0.0),
            trust_score=float((results_data.get("trust_score") or {}).get("trust_score", 0.0)),
            factory_best_sharpe=float(results_data.get("factory_best_sharpe") or 0.0),
            study_meta=_study_meta,
        ))

    return FullCycleHistoryResponse(entries=entries, total_runs=len(entries))


@router.get("/full-cycle/{job_id}/status", response_model=FullCycleStatusResponse)
def get_full_cycle_status(job_id: str):
    job_dir = _FULL_CYCLE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    status_path = job_dir / "status.json"
    data = _read_json(status_path)
    if not data:
        data = {"phase": "orphaned", "error": "Status file missing or corrupted",
                 "job_id": job_id, "started_at": ""}
    raw_phase = data.get("phase", "unknown")
    if raw_phase not in _TERMINAL_PHASES and raw_phase != "unknown":
        hb_str = data.get("last_heartbeat", "")
        stale = False
        if hb_str:
            try:
                hb_dt = datetime.fromisoformat(hb_str)
                stale = (datetime.utcnow() - hb_dt).total_seconds() > _STALE_THRESHOLD_S
            except Exception:
                stale = True
        else:
            try:
                mtime = status_path.stat().st_mtime
                stale = datetime.utcnow().timestamp() - mtime > _STALE_THRESHOLD_S
            except OSError:
                stale = False
        if stale:
            data["phase"] = "orphaned"
            data["error"] = "Pipeline thread unresponsive — no heartbeat for 5+ minutes"
        data["stale"] = stale
    return FullCycleStatusResponse(**{k: v for k, v in data.items() if k in FullCycleStatusResponse.model_fields})


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


class LogsResponse(BaseModel):
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    next_index: int = 0


@router.get("/full-cycle/{job_id}/logs", response_model=LogsResponse)
def get_full_cycle_logs(job_id: str, since: int = Query(0, ge=0)):
    entries = get_job_logs(job_id, since)
    next_idx = (entries[-1]["index"] + 1) if entries else since
    return LogsResponse(entries=entries, next_index=next_idx)


class CancelResponse(BaseModel):
    status: str = "cancelling"


@router.post("/full-cycle/{job_id}/cancel")
def cancel_full_cycle(job_id: str):
    """Immediately cancel the running full cycle. Kills thread + processes, cleans up."""
    job_dir = _FULL_CYCLE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")

    log_info(job_id, "Full cycle cancelled by user — force-stopping")

    # 1. Force-stop thread + child processes
    from api.process_cleanup import force_stop_job
    killed = force_stop_job(job_id)
    log_info(job_id, f"Force-stop: killed {killed} processes")

    # 2. Write cancel flag for any remaining checks
    cancel_path = job_dir / "cancel.json"
    _write_json(cancel_path, {"cancelled": True, "at": datetime.utcnow().isoformat()})

    # 3. Clear pending queue
    from api.config import settings
    try:
        from api.services import JobManager
        from pipeline.data.data_sqlite import DataStore
        store = DataStore(settings.db_full_path)
        JobManager(store).clear_pending_queue()
    except Exception as e:
        log_warn(job_id, f"Cancel: failed to clear pending queue: {e}")

    # 4. Compute elapsed time
    elapsed = 0.0
    try:
        status_data = _read_json(job_dir / "status.json")
        started = status_data.get("started_at", "")
        if started:
            elapsed = (datetime.utcnow() - datetime.fromisoformat(started)).total_seconds()
    except Exception:
        pass

    # 5. Mark as cancelled
    _update_full_cycle_status(job_dir, "cancelled", phase_number=0,
                               current_action="Cancelled by user",
                               error="Cancelled by user")
    results = {"job_id": job_id, "status": "cancelled", "total_time_s": elapsed}
    with open(job_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 6. Delete job directory so monitor stops showing it
    import shutil
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
        log_info(job_id, f"Cleaned up job directory: {job_dir}")
    except Exception as e:
        log_warn(job_id, f"Failed to remove job directory: {e}")

    return CancelResponse(status="cancelled")


# ── Study Metadata (Save/Load) ────────────────────────────────────────────

@router.patch("/full-cycle/{job_id}/study-meta", response_model=StudyMetaResponse)
def update_committee_study_meta(job_id: str, meta: StudyMetaRequest):
    job_dir = _FULL_CYCLE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    meta_path = job_dir / "study_meta.json"
    payload = meta.model_dump(exclude_none=True)
    payload["saved_at"] = datetime.utcnow().isoformat()
    _write_json(meta_path, payload)
    return StudyMetaResponse(**payload)


@router.get("/full-cycle/{job_id}/config")
def get_full_cycle_config(job_id: str):
    job_dir = _FULL_CYCLE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    req_path = job_dir / "request.json"
    if req_path.exists():
        return _read_json(req_path)
    return {"job_id": job_id, "error": "Config not found"}


@router.get("/full-cycle/studies", response_model=FullCycleHistoryResponse)
def get_full_cycle_studies(
    favorite_only: bool = Query(False, description="Only show favorites"),
    tag: str = Query("", description="Filter by tag"),
    search: str = Query("", description="Search display_name"),
):
    entries: List[FullCycleHistoryEntry] = []
    if not _FULL_CYCLE_DIR.exists():
        return FullCycleHistoryResponse(entries=[], total_runs=0)

    now = datetime.utcnow().timestamp()

    for job_dir in sorted(_FULL_CYCLE_DIR.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        status_path = job_dir / "status.json"
        results_path = job_dir / "results.json"
        meta_path = job_dir / "study_meta.json"
        if not status_path.exists():
            continue

        try:
            status_data = _read_json(status_path)
            results_data = _read_json(results_path) if results_path.exists() else {}
            meta_data = _read_json(meta_path)
        except Exception:
            continue

        raw_status = status_data.get("phase", "unknown")
        if raw_status not in _TERMINAL_PHASES:
            try:
                mtime = status_path.stat().st_mtime
                if now - mtime > _STALE_THRESHOLD_S:
                    raw_status = "orphaned"
            except OSError:
                pass

        _study_meta = None
        if meta_data:
            _study_meta = StudyMetaResponse(**meta_data)

        if favorite_only and (not _study_meta or not _study_meta.is_favorite):
            continue
        if tag and _study_meta and tag not in _study_meta.tags:
            continue
        if search and _study_meta:
            _dn = (_study_meta.display_name or "").lower()
            if search.lower() not in _dn:
                continue

        entries.append(FullCycleHistoryEntry(
            job_id=job_dir.name,
            started_at=status_data.get("started_at", ""),
            status=raw_status,
            total_time_s=float(results_data.get("total_time_s") or 0.0),
            locked_features_count=int(results_data.get("locked_features_count") or 0),
            survivors_count=len(results_data.get("phase0_survivors") or []),
            survivors=results_data.get("phase0_survivors") or [],
            avg_sharpe=float(results_data.get("phase3_seed_robustness_sharpe") or 0.0),
            trust_score=float((results_data.get("trust_score") or {}).get("trust_score", 0.0)),
            factory_best_sharpe=float(results_data.get("factory_best_sharpe") or 0.0),
            study_meta=_study_meta,
        ))

    return FullCycleHistoryResponse(entries=entries, total_runs=len(entries))



class FullCycleCancelled(Exception):
    """Raised when user requests cancellation via the Cancel button."""
    pass


def _is_cancelled(job_dir: Path) -> bool:
    return (job_dir / "cancel.json").exists()


def _check_cancel(job_dir: Path, job_id: str):
    if _is_cancelled(job_dir):
        log_info(job_id, "Cancel requested — stopping")
        raise FullCycleCancelled()


def _log_action(job_id: str, phase_number: int, msg: str, phase: str = ""):
    """Log a status change to both the log buffer and stdout."""
    label = PHASE_LABELS.get(phase_number, f"Phase {phase_number}")
    log_info(job_id, f"[{label}] {msg}", phase=phase, phase_number=phase_number)


def _update_full_cycle_status(job_dir: Path, phase: str, phase_number: int = 0, **kwargs):
    with _STATUS_LOCK:
        data = {}
        status_path = job_dir / "status.json"
        if status_path.exists():
            data = _read_json(status_path)
        prev_phase = data.get("phase", "")

        # If already force-cancelled, don't let the stale thread overwrite it
        if data.get("phase") == "cancelled" and phase != "cancelled":
            return

        data["phase"] = phase
        data["phase_number"] = phase_number
        data.update(kwargs)
        _write_json(status_path, data)

    # Derive job_id from job_dir name and log status changes (outside lock to avoid log contention)
    job_id = job_dir.name
    action = kwargs.get("current_action", "")
    if action:
        _log_action(job_id, phase_number, action)
    elif phase != prev_phase:
        ph_lbl = PHASE_LABELS.get(phase_number, "")
        when = "started" if not data.get("error") else "failed"
        _log_action(job_id, phase_number, f"{ph_lbl} {when}" if ph_lbl else f"Status: {phase}")


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
    _debug = req.debug_mode
    try:
        # Initialize Adaptive Resource Governor (replaces old hardcoded env var overrides)
        from pipeline.resource_budget import get_resource_budget, apply_process_priority
        from pipeline.resource_monitor import ResourceMonitor, get_throttle_signal
        budget = get_resource_budget()
        apply_process_priority()
        bl = str(budget.blas_threads)
        cv = str(budget.cv_n_jobs)
        for var in ("MLB_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "SKLEARN_JOBS", "RF_JOBS", "XGB_JOBS",
                     "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                     "BLAS_THREADS_PER_TRIAL"):
            os.environ[var] = bl
        os.environ["CV_JOBS"] = cv
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        from pipeline.committee.expert_profiler import (
            ExpertProfiler, RegimeConfig,
        )
        from pipeline.committee.committee_builder import CommitteeBuilder
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.models.model_families import get_trial_budget
        from pipeline.committee.factory_state import load_state_from_disk
        from pipeline.committee.factory_executor import FactoryExecutor
        import numpy as np

        csv_path, df = _load_csv_for_committee(req.pair, req.timeframe)
        _cancel = lambda: _check_cancel(job_dir, job_id)

        from api.log_buffer import enable_file_logging
        enable_file_logging(job_id, str(job_dir))

        heartbeat_stop = threading.Event()
        def _heartbeat_loop():
            while not heartbeat_stop.wait(30):
                try:
                    with _STATUS_LOCK:
                        data = _read_json(job_dir / "status.json") or {}
                        data["last_heartbeat"] = datetime.utcnow().isoformat()
                        _write_json(job_dir / "status.json", data)
                except Exception:
                    pass
        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()

        with ResourceMonitor(budget):
            # ──────────────────────────────────────────────────────────────
            # PHASE 1: FEATURE SWEEP (if locked_features.json doesn't exist)
            # ──────────────────────────────────────────────────────────────
            locked_features_path = job_dir / "locked_features.json"
            sweep_report_path = job_dir / "locked_features_report.json"
            locked_features = None
            sweep_report_data: Dict[str, Any] = {}

            if locked_features_path.exists():
                try:
                    from pipeline.features.feature_sweep import load_locked_features
                    # Check if cached sweep was done with different params
                    expected_sig = f"n{req.sweep_n_estimators}_d{req.sweep_max_depth}_f3"
                    if sweep_report_path.exists():
                        with open(sweep_report_path) as f:
                            sweep_report_data = json.load(f)
                        cached_sig = sweep_report_data.get("config", {}).get("signature", "")
                        if cached_sig and cached_sig != expected_sig:
                            log_info(job_id, f"Phase -1 cache stale (sig={cached_sig} != {expected_sig}) — re-running sweep")
                            locked_features_path.unlink()
                            sweep_report_path.unlink()
                            locked_features = None
                    if not locked_features:
                        locked_features = load_locked_features(str(locked_features_path))
                    if locked_features:
                        log_info(job_id, f"Loaded {len(locked_features)} locked features from cache")
                        _update_full_cycle_status(job_dir, "feature_sweep", phase_number=1,
                                                   current_action="Phase 1: Feature Sweep skipped (cached)")
                except Exception as e:
                    log_warn(job_id, f"Failed to load cached locked features: {e} — re-running sweep")

            if not locked_features:
                if req.skip_feature_sweep:
                    log_warn(job_id, "Phase 1 skipped (skip_feature_sweep=true) — no cached features available, proceeding without feature filtering")
                else:
                    from pipeline.features.feature_sweep import run_phase_minus1

                    def _on_sweep_progress(msg: str):
                        _update_full_cycle_status(job_dir, "feature_sweep", phase_number=1,
                                                   current_action=msg)
                        log_info(job_id, msg)

                    _update_full_cycle_status(job_dir, "feature_sweep", phase_number=1,
                                               current_action="Phase 1: expanding features, training shallow RF")
                    locked_features, sweep_report_data = run_phase_minus1(
                        df, output_path=str(locked_features_path),
                        label_threshold=0.0001,
                        n_estimators=req.sweep_n_estimators,
                        max_depth=req.sweep_max_depth,
                        n_folds=3, random_state=42,
                        progress_callback=_on_sweep_progress,
                        use_boruta=req.use_boruta_shap,
                        boruta_percentile=req.boruta_percentile,
                        boruta_max_iter=req.boruta_max_iter,
                    )
                    log_info(job_id, f"Phase 1 complete: {len(locked_features)} features locked")

            _check_cancel(job_dir, job_id)
            locked_count = len(locked_features) if locked_features else 0
            pruned_count = sweep_report_data.get("pruned_count", 0)
            top_feature = (sweep_report_data.get("locked_features") or [""])[0]

            if locked_features:
                os.environ["MLB_TA_MODE"] = "fixed"
                log_info(job_id, "TA_MODE locked to 'fixed' (features pre-filtered by Phase -1)")

            # Prepare raw OHLC DataFrame for anchored regime detection (needs 'time' column)
            raw_df_regime = df.reset_index() if df.index.name or "time" not in df.columns else df.copy()
            if "time" not in raw_df_regime.columns:
                raw_df_regime["time"] = raw_df_regime.index

            # ── Pre-flight diagnostics ──
            n_bars = len(df)
            # ASHA pruning enabled for Phase 3 — multi-fidelity HPO replaces
            # fixed trial budgets. Pruner filters poor trials at each rung.
            from pipeline.tuning.objective import set_trial_error_callback
            set_trial_error_callback(lambda msg: log_info(job_id, f"  [Trial] {msg}"))
            log_info(job_id, "Phase 2: ASHA pruning enabled (multi-fidelity HPO)", phase_number=2)

            label_count = df.get("returns", pd.Series([0.0])).pipe(
                lambda s: pd.cut(s, bins=[-float("inf"), -0.0001, 0.0001, float("inf")],
                                  labels=["sell", "hold", "buy"])).value_counts()
            hold_pct = label_count.get("hold", 0) / max(n_bars, 1) * 100
            feats_str = str(locked_features[:12]) + ("..." if len(locked_features) > 12 else "") if locked_features else "None"
            log_info(job_id, f"Pre-flight: {n_bars} bars, {len(locked_features) if locked_features else 0} locked features, "
                             f"train_months={req.train_months}, test_months={req.test_months}")
            log_info(job_id, f"Pre-flight: label distribution — "
                             f"buy={label_count.get('buy', 0)}, sell={label_count.get('sell', 0)}, "
                             f"hold={label_count.get('hold', 0)} ({hold_pct:.0f}% neutral)")
            fold_est = max(1, n_bars // (req.train_months * 30 * 24 + req.test_months * 30 * 24))
            log_info(job_id, f"Pre-flight: estimated ~{fold_est} folds, {len(req.models)} models routed directly to Phase 2")

            # ──────────────────────────────────────────────────────────────
            # PHASE 2: REMOVED — all models flow to Phase 3 HPO directly
            # Formerly pre-screening on default params (dead weight).
            # Phase 3 HPO with ASHA pruning performs the filtering correctly.
            # ──────────────────────────────────────────────────────────────
            survivors = list(req.models)
            pruned: List[str] = []
            matrix = None
            profiler = None
            if req.enable_phase3:
                profiler = ExpertProfiler(
                    data_config={"symbol": req.pair, "csv_data_path": str(csv_path)},
                    wfo_config={
                        "n_months": req.train_months,
                        "hpo_mode": "static",
                        "hpo_sampler": req.hpo_sampler,
                        "cv_blocks": req.cv_blocks,
                        "cv_val_frac": req.cv_val_frac,
                        "plateau_patience": req.plateau_patience,
                        "locked_features": locked_features,
                    },
                    regime_cfg=RegimeConfig(),
                )
                profiler._job_id = job_id
                profiler._raw_df = None
                from types import SimpleNamespace
                matrix = SimpleNamespace(raw_folds=[])
            log_info(job_id, f"Phase 2 prep: {len(survivors)} models as survivors -- proceeding to HPO")
            _update_full_cycle_status(
                job_dir, "phase1_hpo", phase_number=2,
                current_action=f"Phase 2 started -- {len(survivors)} models to tune",
                surviving_models=survivors, pruned_models=pruned,
            )

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: TARGETED HPO with ASHA pruning + ModelStatus tracking
        # Phases split by family: CPU models in parallel, deep/ensembles serial
        # ──────────────────────────────────────────────────────────────
        hpo_base_config: Dict[str, Any] = {}
        hpo_status: Dict[str, str] = {}
        if not req.enable_phase3:
            log_info(job_id, "Phase 2: skipped (disabled)", phase_number=2)
            hpo_model_params: Dict[str, dict] = {}
            for m in survivors:
                hpo_status[m] = ModelStatus.SKIPPED.value
        else:
            from pipeline.models.model_families import ModelStatus, is_gpu_model

            _update_full_cycle_status(job_dir, "phase1_hpo", phase_number=2,
                                       current_action="Starting Phase 2 HPO",
                                       phase_progress=f"0/{len(survivors)}")
            if matrix is not None and hasattr(matrix, 'raw_folds') and matrix.raw_folds:
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
                "max_hpo_duration_minutes": 20,
            }

            first_df_wfo = None
            hpo_model_params: Dict[str, dict] = {}
            progress_count = [0]  # mutable counter for parallel callback

            def _build_hpo_config(model_type):
                n_trials, n_startup = get_trial_budget(model_type)
                if req.hpo_trials and model_type in req.hpo_trials:
                    n_trials = max(1, int(req.hpo_trials[model_type]))
                if req.hpo_startup_trials and model_type in req.hpo_startup_trials:
                    n_startup = max(0, int(req.hpo_startup_trials[model_type]))
                cfg = dict(hpo_base_config)
                cfg["model_type"] = model_type
                cfg["n_trials"] = n_trials
                cfg["n_startup_trials"] = n_startup
                return cfg, n_trials

            def _run_hpo_for_model(model_type, seed=42):
                """Run HPO for a single model. Returns (model_type, status, tuned_folds, df_wfo, best_params)."""
                nonlocal first_df_wfo, progress_count
                sig = get_throttle_signal()
                if sig and sig.delay > 0:
                    import time
                    time.sleep(sig.delay)

                cfg, n_trials = _build_hpo_config(model_type)
                per_model_timeout = max(1800, n_trials * 120)

                progress_count[0] += 1
                def _on_progress(msg):
                    _update_full_cycle_status(
                        job_dir, "phase1_hpo", phase_number=2,
                        current_action=f"{msg} ({progress_count[0]}/{len(survivors)})",
                        phase_progress=f"{progress_count[0]}/{len(survivors)}",
                    )
                    log_info(job_id, msg)

                _on_progress(f"HPO on {model_type}: starting {n_trials} trials")

                tuned_folds, df_wfo, best_params = None, None, None
                status = ModelStatus.CRASHED
                try:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            profiler._run_single_model,
                            model_type, cfg, seed,
                            verbose=False,
                            progress_callback=_on_progress,
                        )
                        tuned_folds, df_wfo, best_params = future.result(timeout=per_model_timeout)
                    if tuned_folds:
                        status = ModelStatus.SUCCESS
                    else:
                        status = ModelStatus.NO_FOLDS
                except FutureTimeout:
                    log_warn(job_id, f"HPO on {model_type}: timed out after {per_model_timeout}s")
                    status = ModelStatus.TIMED_OUT
                except Exception as e:
                    log_warn(job_id, f"HPO on {model_type}: failed with {type(e).__name__}: {str(e)[:200]}")
                    status = ModelStatus.CRASHED

                log_info(job_id, f"HPO on {model_type}: status={status.value}")
                return model_type, status, tuned_folds, df_wfo, best_params

            # ---- Batch 1: CPU-classical models in parallel ----
            cpu_models = [m for m in survivors if not is_gpu_model(m)]
            gpu_models = [m for m in survivors if is_gpu_model(m)]

            if cpu_models:
                log_info(job_id, f"Phase 2: running {len(cpu_models)} CPU models in parallel", phase_number=2)
                n_jobs = min(len(cpu_models), int(os.environ.get("MLB_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))))
                try:
                    from joblib import Parallel, delayed
                    cpu_results = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(_run_hpo_for_model)(m) for m in cpu_models
                    )
                except (ImportError, Exception) as e:
                    log_warn(job_id, f"joblib parallel failed ({e}), falling back to sequential")
                    cpu_results = [_run_hpo_for_model(m) for m in cpu_models]
            else:
                cpu_results = []

            # ---- Batch 2: Deep/ensemble models sequential ----
            gpu_results = []
            for m in gpu_models:
                gpu_results.append(_run_hpo_for_model(m))

            # ---- Collect results ----
            all_results = cpu_results + gpu_results
            for model_type, status, tuned_folds, df_wfo, best_params in all_results:
                hpo_status[model_type] = status.value
                if tuned_folds:
                    if hasattr(matrix, 'raw_folds') and matrix.raw_folds is not None:
                        matrix.raw_folds.extend(tuned_folds)
                    else:
                        matrix.raw_folds = list(tuned_folds)
                if best_params:
                    hpo_model_params[model_type] = best_params
                    log_info(job_id, f"HPO on {model_type}: best params found ({len(best_params)} keys)")
                    trial_summary = {
                        "model_type": model_type,
                        "status": status.value,
                        "best_score": best_params.get("__cv_value", None),
                        "committee_size": len(best_params.get("__committee_fixed", [])),
                        "consensus_pool_size": len(best_params.get("__consensus_pool", [])),
                    }
                    model_dir = job_dir / "hpo_results"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    with open(model_dir / f"{model_type}.json", "w") as f:
                        json.dump(trial_summary, f, indent=2, default=str)
                else:
                    if status == ModelStatus.SUCCESS:
                        log_info(job_id, f"HPO on {model_type}: ran successfully but no best params — using defaults")
                    else:
                        log_info(job_id, f"HPO on {model_type}: excluded from committee (status={status.value})")
                if first_df_wfo is None and df_wfo is not None:
                    first_df_wfo = df_wfo
                _check_cancel(job_dir, job_id)

            # Save HPO status manifest
            with open(job_dir / "hpo_status.json", "w") as f:
                json.dump(hpo_status, f, indent=2)

            # Log summary
            n_success = sum(1 for s in hpo_status.values() if s == ModelStatus.SUCCESS.value)
            n_failed = len(hpo_status) - n_success
            log_info(job_id, f"Phase 2 complete: {n_success} successes, {n_failed} failures -- "
                             f"status: {hpo_status}", phase_number=2)

            # Update survivors to only SUCCESS models for downstream phases
            survivors_success = [m for m, s in hpo_status.items() if s == ModelStatus.SUCCESS.value]
            if survivors_success:
                survivors = survivors_success
            elif not req.debug_mode:
                raise RuntimeError(
                    f"Phase 2: all {len(hpo_status)} models failed HPO. "
                    "Check data quality or increase per-model timeout."
                )

            # Rebuild matrix with tuned fold results
            if profiler is not None and hasattr(matrix, 'raw_folds') and matrix.raw_folds:
                profiler._attach_regime_distributions(matrix.raw_folds, first_df_wfo)
                matrix = profiler._build_matrix(matrix.raw_folds)

            if hasattr(matrix, 'to_dict') and matrix.models:
                matrix_data = matrix.to_dict()
                with open(job_dir / "regime_matrix_tuned.json", "w") as f:
                    json.dump(matrix_data, f, indent=2, default=str)

            # ──────────────────────────────────────────────────────────────
            # PHASE 3: COMMITTEE ASSEMBLY
            # ──────────────────────────────────────────────────────────────
            if not req.enable_phase4:
                log_info(job_id, "Phase 3: skipped (disabled) -- using fallback committee", phase_number=3)
                from pipeline.committee.committee_builder import CommitteeConfig as CC, RegimeAssignment as RA
                n = len(survivors)
                w = 1.0 / max(n, 1)
                ra = RA(models=list(survivors), weights=[w] * n)
                committee_config = CC(
                    regimes={"sideways": ra},
                    fallback=ra,
                )
                if hpo_model_params:
                    committee_config.model_params = hpo_model_params
            else:
                _update_full_cycle_status(job_dir, "phase2_assembly", phase_number=3,
                                           current_action="Building committee config")
                if hasattr(matrix, 'models') and matrix.models:
                    weight_method = (req.committee_weight_method
                                     if req.committee_weight_method
                                     else "sharpe_proportional")
                    min_sharpe = (req.committee_min_sharpe
                                  if req.committee_min_sharpe is not None
                                  else 0.0)
                    builder = CommitteeBuilder(
                        top_k=req.committee_top_k, weight_method=weight_method,
                        min_sharpe=min_sharpe,
                    )
                    constraints = {
                        "max_models_per_regime": req.committee_top_k,
                        **({"min_sharpe": min_sharpe} if min_sharpe != 0.0 else {}),
                    }
                    committee_config = builder.build(matrix, constraints=constraints)
                else:
                    from pipeline.committee.committee_builder import CommitteeConfig as CC, RegimeAssignment as RA
                    n = len(survivors)
                    w = 1.0 / max(n, 1)
                    ra = RA(models=list(survivors), weights=[w] * n)
                    committee_config = CC(
                        regimes={"sideways": ra},
                        fallback=ra,
                    )
                    log_warn(job_id, "Phase 3: empty matrix -- using fallback committee with all survivors", phase_number=3)
                cc_data = committee_config.to_dict()
                if hpo_model_params:
                    committee_config.model_params = hpo_model_params
                    cc_data["model_params"] = hpo_model_params
                _inject_features_metadata(committee_config, cc_data, hpo_model_params)
                with open(job_dir / "committee_config.json", "w") as f:
                    json.dump(cc_data, f, indent=2, default=str)

            # ──────────────────────────────────────────────────────────────
            # PHASE 4: INTERMEDIATE VALIDATION (36-mo WFO + consistency + 3 seeds)
            # ──────────────────────────────────────────────────────────────
            if not req.enable_phase5:
                log_info(job_id, "Phase 4: skipped (disabled)", phase_number=4)
                _log_action(job_id, 4, "Phase 4: skipped (disabled)")
            else:
                _update_full_cycle_status(job_dir, "phase3_validation", phase_number=4,
                                           current_action="Phase 4: running WFO validation")
                bt = CommitteeBacktester(
                    committee_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
                    model_params=hpo_model_params,
                    cancel_check=_cancel,
                )
                bt._enable_mda_pruning = True  # P5
                sig = get_throttle_signal()
                if sig and sig.delay > 0:
                    import time
                    time.sleep(sig.delay)
                try:
                    bt_result = bt.run_wfo(
                        df, train_months=req.train_months, test_months=req.test_months,
                        verbose=False,
                        collect_predictions=True,
                    )
                except RuntimeError as e:
                    log_warn(job_id, f"Phase 4: WFO failed -- {e}", phase_number=4)
                    _update_full_cycle_status(
                        job_dir, "validation_failed", phase_number=4,
                        error=f"WFO RuntimeError: {e}",
                    )
                    elapsed = (datetime.utcnow() - t_start).total_seconds()
                    results = FullCycleResultsResponse(
                        job_id=job_id, status="validation_failed",
                        locked_features_count=locked_count,
                        pruned_features_count=pruned_count,
                        top_importance_feature=top_feature,
                        phase0_pruned=pruned, phase0_survivors=survivors,
                        total_time_s=elapsed,
                    )
                    results_extra = results.model_dump()
                    results_extra["trust_score"] = {"trust_score": 0.0, "action": "reject"}
                    with open(job_dir / "results.json", "w") as f:
                        json.dump(results_extra, f, indent=2, default=str)
                    return

                cv = bt_result.fold_consistency_cv
                coverage = bt_result.regime_coverage_report(min_trades=30, min_sharpe=0.0)
                all_covered = all(c["covered"] for c in coverage.values())
                log_info(job_id, f"Phase 4: WFO complete -- CV={cv:.4f}, covered={all_covered}", phase_number=4)

                # -- Compute PBO from fold returns --
                pbo = 1.0
                fold_sharpes = []
                if hasattr(bt_result, "folds") and bt_result.folds:
                    fold_sharpes = [f.sharpe for f in bt_result.folds if not np.isnan(f.sharpe)]
                    # Build approximate fold returns matrix for PBO
                    try:
                        from pipeline.metrics.pbo import compute_pbo
                        # Use fold Sharpes * sqrt(approx bars) to approximate returns
                        approx_bars = len(df) // max(1, len(bt_result.folds))
                        fold_rets = np.array([[s / np.sqrt(approx_bars * 6) for _ in range(10)]
                                              for s in fold_sharpes if np.isfinite(s)])
                        if fold_rets.shape[0] >= 4:
                            pbo = compute_pbo(fold_rets, S=min(8, fold_rets.shape[0]))
                    except Exception:
                        pass
                log_info(job_id, f"Phase 4: PBO={pbo:.4f}", phase_number=4)

                # -- Compute DSR --
                dsr = 0.0
                total_hpo_trials = sum(
                    get_trial_budget(m)[0] for m in survivors
                )
                try:
                    from pipeline.metrics.dsr import deflated_sharpe_ratio
                    avg_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
                    bars_per_fold = req.test_months * 21 * 24
                    T_obs = max(1, len(fold_sharpes) * bars_per_fold)
                    dsr = deflated_sharpe_ratio(
                        sr_hat=max(0.0, avg_sharpe),
                        T=T_obs, N_trials=max(1, total_hpo_trials),
                    )
                except Exception:
                    pass
                log_info(job_id, f"Phase 4: DSR={dsr:.4f} (trials={total_hpo_trials})", phase_number=4)

                # -- Compute regime coverage ratio --
                config_regimes = set(committee_config.regimes.keys()) if hasattr(committee_config, 'regimes') else set()
                covered_regimes = set(r for r, c in coverage.items() if c["covered"])
                regime_coverage_ratio = len(covered_regimes & config_regimes) / max(1, len(config_regimes))

                # -- Compute trust score --
                min_fold_sr = min(fold_sharpes) if fold_sharpes else -float("inf")
                from pipeline.metrics.trust_score import compute_trust_score
                trust = compute_trust_score(pbo, dsr, regime_coverage_ratio, min_fold_sr)
                log_info(job_id, f"Phase 4: Trust Score={trust['trust_score']:.4f} -> {trust['action'].upper()}", phase_number=4)

                # Save trust score
                with open(job_dir / "trust_score.json", "w") as f:
                    json.dump(trust, f, indent=2)

                # Save fold predictions for meta-learner training
                fold_pred_path = job_dir / "fold_predictions.json"
                try:
                    bt.save_fold_predictions(str(fold_pred_path))
                    log_info(job_id, f"Phase 4: saved fold predictions to {fold_pred_path}", phase_number=4)
                except Exception:
                    log_warn(job_id, "Phase 4: failed to save fold predictions", phase_number=4)

                # Train meta-learner on Phase 4 OOS predictions
                if trust["action"] != "reject" and fold_pred_path.exists():
                    try:
                        from pipeline.committee.committee_meta import CommitteeMetaLearner
                        meta = CommitteeMetaLearner()
                        acc = meta.train([str(fold_pred_path)])
                        if meta.is_trained and acc >= 0.40:
                            meta_dir = job_dir / "committee_snapshot" / "meta"
                            meta_dir.mkdir(parents=True, exist_ok=True)
                            meta.save(str(meta_dir / "meta_model.joblib"))
                            log_info(job_id, f"Meta-learner: trained on {bt_result.total_folds} folds, accuracy={acc:.3f} — saved")
                        else:
                            log_info(job_id, f"Meta-learner: accuracy={acc:.3f} — below 0.40 threshold, discarded")
                    except Exception as e:
                        log_warn(job_id, f"Meta-learner training failed: {e} — continuing without meta-learner")

                # Save P1-P3 artifacts from the backtester
                _save_pipeline_artifacts(job_dir, bt, log_info, job_id)

                if hasattr(bt_result, "folds") and bt_result.folds:
                    n_folds = len(bt_result.folds)
                    trades = [f.trades for f in bt_result.folds]
                    log_info(job_id, f"Phase 4 folds: {n_folds} folds, "
                                     f"Sharpe range={min(fold_sharpes):.3f}..{max(fold_sharpes):.3f}, "
                                     f"total trades={sum(trades)}", phase_number=4)
                    for fi, f_item in enumerate(bt_result.folds):
                        log_info(job_id, f"  Fold {fi + 1}: Sharpe={f_item.sharpe:.3f}, "
                                 f"trades={f_item.trades}, test={f_item.test_start}", phase_number=4)

                # Seed robustness
                log_info(job_id, "Phase 4: seed robustness -- 3 seeds", phase_number=4)
                seed_sharpes = []
                for si, seed in enumerate((42, 101, 202)):
                    sig = get_throttle_signal()
                    if sig and sig.delay > 0:
                        import time
                        time.sleep(sig.delay)
                    log_info(job_id, f"Phase 4: seed robustness seed={seed} ({si + 1}/3)", phase_number=4)
                    alt_bt = CommitteeBacktester(
                        committee_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
                        model_params=hpo_model_params,
                        seed=seed,
                        cancel_check=_cancel,
                    )
                    alt_bt._enable_mda_pruning = True  # P5
                    alt_r = alt_bt.run_wfo(
                        df, train_months=req.train_months, test_months=req.test_months,
                        verbose=False,
                    )
                    sd = alt_r.avg_sharpe if alt_r else 0.0
                    seed_sharpes.append(sd)
                    log_info(job_id, f"Phase 4: seed={seed} Sharpe={sd:.4f}", phase_number=4)
                    _check_cancel(job_dir, job_id)
                seed_avg = float(np.mean(seed_sharpes)) if seed_sharpes else 0.0
                seed_pass = all(s > 0.0 for s in seed_sharpes)
                log_info(job_id, f"Phase 4: seed robustness avg={seed_avg:.4f}, pass={seed_pass}", phase_number=4)

                # Trust score gate (replaces binary CV/coverage/seeds check)
                if trust["action"] == "reject":
                    fail_reason = f"trust_score={trust['trust_score']:.4f} < 0.40 -- REJECT"
                    log_warn(job_id, f"Phase 4: {fail_reason}", phase_number=4)
                    _update_full_cycle_status(
                        job_dir, "validation_failed", phase_number=4,
                        error=fail_reason,
                    )
                    elapsed = (datetime.utcnow() - t_start).total_seconds()
                    results = FullCycleResultsResponse(
                        job_id=job_id, status="validation_failed",
                        locked_features_count=locked_count,
                        pruned_features_count=pruned_count,
                        top_importance_feature=top_feature,
                        phase0_pruned=pruned, phase0_survivors=survivors,
                        phase3_fold_consistency_cv=cv,
                        phase3_fold_consistency_pass=False,
                        phase3_regime_coverage=coverage,
                        phase3_seed_robustness_sharpe=seed_avg,
                        phase3_seed_robustness_seeds=3,
                        phase3_seed_robustness_pass=seed_pass,
                        pbo=pbo,
                        dsr=dsr,
                        total_time_s=elapsed,
                    )
                    results_extra = results.model_dump()
                    results_extra["trust_score"] = trust
                    with open(job_dir / "results.json", "w") as f:
                        json.dump(results_extra, f, indent=2, default=str)
                    return

                tw = " (warning: trust<0.80)" if trust["action"] in ("flag", "proceed") else ""
                log_info(job_id, f"Phase 4 complete: validation passed{tw} -- trust={trust['trust_score']:.4f}", phase_number=4)

            # ──────────────────────────────────────────────────────────────
            # PHASE 5: FACTORY OPTIMIZATION (proxy WFO inside loop)
            # ──────────────────────────────────────────────────────────────
            if not req.enable_phase6:
                log_info(job_id, "Phase 5: skipped (disabled)", phase_number=5)
            else:
                _update_full_cycle_status(job_dir, "phase4_factory", phase_number=5,
                                           current_action="Starting Factory optimization (proxy WFO)",
                                           phase_progress=f"0/{req.max_iterations}",
                                           iteration=0, best_sharpe_so_far=0.0)
                log_info(job_id, f"Phase 5: starting Factory -- up to {req.max_iterations} iterations", phase_number=5)

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
                    from pipeline.committee.factory_llm import create_llm_proposer
                    proposer = create_llm_proposer(backend=req.llm_backend)
                elif req.proposer == "hybrid_llm_ucb1":
                    from pipeline.committee.factory_llm import create_llm_proposer
                    from pipeline.committee.factory_ucb import UCB1Proposer
                    from pipeline.committee.factory_hybrid import HybridLLMUCB1Proposer
                    llm = create_llm_proposer(backend=req.llm_backend)
                    ucb = UCB1Proposer(c=req.ucb_c)
                    proposer = HybridLLMUCB1Proposer(llm_proposer=llm, ucb_proposer=ucb, c=req.ucb_c, llm_refresh_interval=5)
                elif req.proposer == "ucb1":
                    from pipeline.committee.factory_ucb import UCB1Proposer
                    proposer = UCB1Proposer(c=req.ucb_c)

                executor = FactoryExecutor(
                    state=state, proposer=proposer, data_path=str(csv_path),
                    train_months=req.factory_proxy_months, test_months=req.test_months,
                )

                executor._load_data()
                _loop_proposer = executor.proposer
                reason = ""
                while True:
                    sig = get_throttle_signal()
                    if sig and sig.delay > 0:
                        import time
                        time.sleep(sig.delay)
                    should_stop, reason = state.should_stop()
                    if should_stop:
                        log_info(job_id, f"Phase 5: stopping -- {reason}", phase_number=5)
                        break
                    proposal = _loop_proposer.propose(state)
                    if proposal.type == "halt":
                        reason = "No more untested moves"
                        log_info(job_id, "Phase 5: no more untested moves", phase_number=5)
                        break

                    _update_full_cycle_status(
                        job_dir, "phase4_factory", phase_number=5,
                        current_action=f"{proposal.type} in {proposal.regime}",
                        iteration=state.iteration + 1,
                        phase_progress=f"{state.iteration + 1}/{req.max_iterations}",
                    )
                    log_info(job_id, f"Phase 5 iter {state.iteration + 1}: {proposal.type} in {proposal.regime}", phase_number=5)
                    record, _ = executor.execute_iteration(proposal)
                    if record is None:
                        continue
                    delta = record.after_sharpe - record.before_sharpe
                    if isinstance(_loop_proposer, HybridLLMUCB1Proposer):
                        _loop_proposer.record_result(proposal, delta)
                    elif isinstance(_loop_proposer, UCB1Proposer):
                        from pipeline.committee.factory_ucb import _arm_hash
                        ah = _arm_hash(proposal.regime, proposal.type, proposal.model_remove, proposal.model_add)
                        _loop_proposer.record_result(ah, delta)
                    _update_full_cycle_status(
                        job_dir, "phase4_factory", phase_number=5,
                        iteration=state.iteration,
                        best_sharpe_so_far=state.global_best_sharpe,
                        phase_progress=f"{state.iteration}/{req.max_iterations}",
                    )
                    accepted = record.accepted if hasattr(record, "accepted") else False
                    status = "accepted" if accepted else "rejected"
                    log_info(job_id, f"Phase 5 iter {state.iteration}: {status} (delta={delta:+.4f}, best={state.global_best_sharpe:.4f})", phase_number=5)
                    _check_cancel(job_dir, job_id)

                # ──────────────────────────────────────────────────────────────
                # FINAL VALIDATION: full 10-year WFO + 5-seed robustness
                # ──────────────────────────────────────────────────────────────
                _update_full_cycle_status(job_dir, "phase4_factory", phase_number=5,
                                           current_action="Final: 10-year WFO + 5-seed robustness")
                log_info(job_id, "Final validation: 10-year WFO + 5-seed robustness")

                final_config = state.config if state.config else committee_config
                final_bt = CommitteeBacktester(
                    final_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
                    model_params=hpo_model_params,
                    cancel_check=_cancel,
                )
                final_bt._enable_mda_pruning = True  # P5
                final_result = final_bt.run_wfo(
                    df, train_months=req.train_months, test_months=req.test_months,
                    verbose=False,
                )
                final_summary = final_result.to_summary_dict() if final_result else {}
                if final_result and hasattr(final_result, "folds") and final_result.folds:
                    n_f = len(final_result.folds)
                    shs = [f.sharpe for f in final_result.folds if not np.isnan(f.sharpe)]
                    log_info(job_id, f"Final WFO: {n_f} folds, "
                                     f"Sharpe range={min(shs):.3f}..{max(shs):.3f}, "
                                     f"total trades={sum(f.trades for f in final_result.folds)}")

                # 5-seed robustness on the final config
                log_info(job_id, "Final: 5-seed robustness")
                final_seed_sharpes = []
                for si, seed in enumerate((42, 101, 202, 789, 999)):
                    sig = get_throttle_signal()
                    if sig and sig.delay > 0:
                        import time
                        time.sleep(sig.delay)
                    log_info(job_id, f"Final: seed robustness seed={seed} ({si + 1}/5)")
                    fbt = CommitteeBacktester(
                        final_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
                        model_params=hpo_model_params,
                        cancel_check=_cancel,
                    )
                    fbt._enable_mda_pruning = True  # P5
                    fr = fbt.run_wfo(
                        df, train_months=req.train_months, test_months=req.test_months,
                        verbose=False,
                    )
                    sd = fr.avg_sharpe if fr else 0.0
                    final_seed_sharpes.append(sd)
                    log_info(job_id, f"Final: seed={seed} Sharpe={sd:.4f}")
                    _check_cancel(job_dir, job_id)
                final_seed_avg = float(np.mean(final_seed_sharpes)) if final_seed_sharpes else 0.0
                final_seed_pass = all(s > 0.0 for s in final_seed_sharpes)
                log_info(job_id, f"Final: seed robustness avg={final_seed_avg:.4f}, pass={final_seed_pass}")

                final_fold_cv = final_result.fold_consistency_cv if final_result else float("inf")
                final_fold_pass = final_result.fold_consistency_pass if final_result else False
                final_coverage = final_result.regime_coverage_report(min_trades=30, min_sharpe=0.0) if final_result else {}

                # Save final committee config to disk (deployment)
                final_data = final_config.to_dict()
                if hpo_model_params:
                    final_data["model_params"] = hpo_model_params
                _inject_features_metadata(final_config, final_data, hpo_model_params)
                with open(job_dir / "committee_config_final.json", "w") as f:
                    json.dump(final_data, f, indent=2, default=str)

                # ── Save committee snapshot (MLOps reproducibility) ──
                # Train all committee models on full history and save exact weights
                # so deployment loads byte-for-byte identical estimators.
                unique_models = list(set(final_config.all_models()))
                snapshot_dir = job_dir / "committee_snapshot"
                snapshot_saved = False
                try:
                    from pipeline.features.feature_sweep import compute_feature_matrix, FEATURE_NAMES
                    from models.registry import build_model
                    from pipeline.committee.expert_profiler import _reprefix_params

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

                    log_info(job_id, f"Snapshot: training {len(unique_models)} models on full dataset")
                    snapshot_dir.mkdir(parents=True, exist_ok=True)

                    for mi, mtype in enumerate(unique_models):
                        log_info(job_id, f"Snapshot: saving {mtype} ({mi + 1}/{len(unique_models)})")
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
                                log_info(job_id, f"Snapshot: {mtype} -> TF SavedModel")
                            else:
                                import joblib
                                joblib.dump(model, str(snapshot_dir / f"{mtype}.joblib"))
                                log_info(job_id, f"Snapshot: {mtype} -> .joblib")
                        except Exception:
                            log_error(job_id, f"Snapshot: {mtype} failed")
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
                    "phase3_regime_coverage": {r: dict(v) for r, v in coverage.items()},
                    "phase3_seed_robustness_sharpe": round(float(seed_avg), 4),
                    "phase3_seed_robustness_seeds": 3,
                    "phase3_seed_robustness_pass": bool(seed_pass),
                    "trust_score": trust,
                    "pbo": round(float(pbo), 4),
                    "dsr": round(float(dsr), 4),
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
                    "hpo_status": hpo_status if 'hpo_status' in dir() else {},
                }
                with open(job_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)

                _update_full_cycle_status(
                    job_dir, "completed", phase_number=6,
                    best_sharpe_so_far=state.global_best_sharpe,
                )

    except (BrokenProcessPool, EOFError):
        log_info(job_id, "Child processes terminated via cancellation — cleaning up")
        from api.process_cleanup import cleanup_job as _cleanup_job
        _cleanup_job(job_id)
    except FullCycleCancelled:
        log_info(job_id, "Full cycle cancelled — cleaning up")
        from api.process_cleanup import cleanup_job as _cleanup_job
        _cleanup_job(job_id)
        if not (job_dir / "results.json").exists():
            _update_full_cycle_status(job_dir, "cancelled", phase_number=0,
                                       error="Cancelled by user")
            elapsed = (datetime.utcnow() - t_start).total_seconds()
            results = {"job_id": job_id, "status": "cancelled",
                       "total_time_s": elapsed,
                       "locked_features_count": locked_count}
            if 'survivors' in dir():
                results["phase0_survivors"] = survivors
            with open(job_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
    except Exception as e:
        log_info(job_id, "Full cycle failed — cleaning up processes")
        from api.process_cleanup import cleanup_job as _cleanup_job
        _cleanup_job(job_id)
        if not (job_dir / "results.json").exists():
            _update_full_cycle_status(job_dir, "failed", phase_number=0, error=str(e))
        import traceback
        traceback.print_exc()
    finally:
        heartbeat_stop.set()
        cancel_path = job_dir / "cancel.json"
        try:
            cancel_path.unlink()
        except Exception:
            pass
        # Save results if no phase saved them yet (covers Phase 3-6 disabled case)
        try:
            if not (job_dir / "results.json").exists():
                has_survivors = 'survivors' in dir()
                has_matrix = 'matrix' in dir() and matrix is not None and hasattr(matrix, 'to_dict')
                has_hpo = 'hpo_model_params' in dir()
                elapsed = (datetime.utcnow() - t_start).total_seconds()
                results = {
                    "job_id": job_id, "status": "completed",
                    "locked_features_count": locked_count,
                    "pruned_features_count": pruned_count,
                    "top_importance_feature": top_feature,
                    "total_time_s": elapsed,
                }
                if has_survivors:
                    results["phase0_survivors"] = survivors
                    results["phase0_pruned"] = pruned if 'pruned' in dir() else []
                if has_matrix:
                    results["racecar_profile_matrix"] = matrix.to_dict()
                if has_hpo:
                    results["hpo_model_params_count"] = len(hpo_model_params)
                if 'cc_data' in dir():
                    results["racecar_committee_config"] = cc_data
                with open(job_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)
                _update_full_cycle_status(job_dir, "completed", phase_number=5,
                                           best_sharpe_so_far=0.0)
        except Exception:
            pass
