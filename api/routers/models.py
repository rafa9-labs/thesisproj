"""Model registry and hyperparameter endpoints."""
import os
from fastapi import APIRouter, HTTPException, Query

from api.schemas.backtest import ModelInfo, ModelListResponse
from api.schemas.hyperparams import (
    HyperparamChoice,
    HyperparamFixed,
    HyperparamRange,
    ModelHyperparams,
    ModelHyperparamsResponse,
)
from config import SEARCH_SPACE

MODEL_DESCRIPTIONS = {
    "logistic": ("Logistic Regression", "classical", "Fast linear classifier with probability calibration"),
    "svm": ("Support Vector Machine", "classical", "Kernel-based classifier with RBF kernel"),
    "random_forest": ("Random Forest", "classical", "Ensemble of decision trees with bagging"),
    "decision_tree": ("Decision Tree", "classical", "Single decision tree classifier"),
    "xgboost": ("XGBoost", "classical", "Gradient-boosted trees with regularisation"),
    "cnn": ("Convolutional Neural Network", "deep", "1D-CNN for pattern recognition on price windows"),
    "lstm": ("LSTM Network", "deep", "Long short-term memory network for sequential data"),
    "transformer": ("Transformer", "deep", "Self-attention architecture for time-series"),
    "dqn": ("Dueling DQN", "rl", "Deep Q-Network reinforcement learning agent"),
    "ensemble_adaptive_regime": ("Adaptive Regime Ensemble", "ensemble", "Regime-aware ensemble combining multiple models"),
    "meta_ensemble": ("Signal Committee", "ensemble", "Multi-model voting committee combining multiple models"),
}


def _parse_param(name: str, spec) -> dict:
    """Convert a SEARCH_SPACE entry into a HyperparamSpec dict."""
    if isinstance(spec, tuple):
        if len(spec) == 3 and spec[2] is True:
            return HyperparamRange(
                type="float_range",
                low=float(spec[0]),
                high=float(spec[1]),
                log_scale=True,
            ).model_dump()
        if len(spec) == 3 and isinstance(spec[2], (int, float)):
            return HyperparamRange(
                type="float_range" if isinstance(spec[0], float) or isinstance(spec[1], float) else "int_range",
                low=float(spec[0]),
                high=float(spec[1]),
                step=float(spec[2]),
            ).model_dump()
        if len(spec) == 2:
            return HyperparamRange(
                type="float_range",
                low=float(spec[0]),
                high=float(spec[1]),
            ).model_dump()
    if isinstance(spec, list):
        return HyperparamChoice(type="choice", values=spec).model_dump()
    return HyperparamFixed(type="fixed", value=spec).model_dump()


router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models():
    models = []
    for name, (display, category, desc) in MODEL_DESCRIPTIONS.items():
        models.append(ModelInfo(
            name=name,
            display_name=display,
            category=category,
            description=desc,
        ))
    return ModelListResponse(models=models)


@router.get("/hyperparams", response_model=ModelHyperparamsResponse)
def get_hyperparams():
    """Return SEARCH_SPACE metadata so the frontend can build per-model hyperparameter UIs."""
    result = []
    for name, (display, category, _) in MODEL_DESCRIPTIONS.items():
        space = SEARCH_SPACE.get(name, {})
        params = {k: _parse_param(k, v) for k, v in space.items()}
        result.append(ModelHyperparams(
            model=name,
            display_name=display,
            category=category,
            tunable=len(params) > 0,
            params={k: v for k, v in params.items()},
        ))
        return ModelHyperparamsResponse(models=result)


# ────────────────────────────────────────────────────────────
#  Deployed Models (Phase B: model registry on disk)
# ────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from typing import List as PyList, Optional
from api.dependencies import get_data_store


class DeployedModelOut(BaseModel):
    id: str
    model_type: str
    snapshot_path: str
    best_sharpe: Optional[float] = None
    best_return: Optional[float] = None
    created_at: str
    status: str
    tags: PyList[str] = Field(default_factory=list)
    parent_job_id: Optional[str] = None
    missing_on_disk: bool = False


class DeployedModelListResponse(BaseModel):
    models: PyList[DeployedModelOut]


class ActivateModelRequest(BaseModel):
    pass  # No body needed


class TagUpdateRequest(BaseModel):
    action: str  # "add" or "remove"
    tag: str


@router.get("/deployed", response_model=DeployedModelListResponse)
def list_deployed_models(
    model_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    from api.config import settings
    from pipeline.model_registry_disk import get_all_deployed

    rows = get_all_deployed(settings.db_full_path)
    models = []
    for r in rows:
        if model_type and r.get("model_type") != model_type:
            continue
        if status and r.get("status") != status:
            continue
        models.append(DeployedModelOut(**r))
    return DeployedModelListResponse(models=models)


@router.post("/deployed/{model_id}/activate")
def activate_deployed_model(model_id: str):
    from api.config import settings
    from pipeline.model_registry_disk import activate_model

    ok = activate_model(model_id, settings.db_full_path)
    if not ok:
        raise HTTPException(404, f"Model {model_id} not found")
    return {"status": "ok"}


@router.delete("/deployed/{model_id}")
def delete_deployed_model(model_id: str):
    from api.config import settings
    from pipeline.model_registry_disk import delete_model

    ok, reason = delete_model(model_id, settings.db_full_path)
    if not ok:
        raise HTTPException(404, reason)
    return {"status": "ok"}


@router.patch("/deployed/{model_id}/tags")
def update_model_tags(model_id: str, req: TagUpdateRequest):
    from api.config import settings
    from pipeline.model_registry_disk import update_tags

    tags = update_tags(model_id, settings.db_full_path, req.action, req.tag)
    if tags is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return {"tags": tags}


# ────────────────────────────────────────────────────────────
#  Live Prediction & Comparison (Phase D: live trading bridge)
# ────────────────────────────────────────────────────────────
from pydantic import BaseModel as PydBaseModel


class PredictRequest(PydBaseModel):
    pair: str = "EURUSD"
    timeframe: str = "H1"


@router.post("/active/predict")
def predict_with_active_model(req: PredictRequest):
    """Load the active model for a pair/timeframe and return a readiness check.

    Returns the active model's type + snapshot ID so the caller knows
    which model would be used. Full feature computation is wired in S21.
    """
    from api.config import settings
    from pipeline.model_persistence import read_metadata, load_model_only, get_active_model_id
    from pipeline.model_registry_disk import get_all_deployed

    rows = get_all_deployed(settings.db_full_path)
    active_by_type: dict[str, dict] = {
        r["model_type"]: r for r in rows if r.get("status") == "active"
    }

    if not active_by_type:
        return {"status": "no_active_model", "message": "No active model deployed"}

    info = active_by_type[list(active_by_type.keys())[0]]
    model_id = info["id"]
    snapshot_path = info.get("snapshot_path", "")
    meta = read_metadata(snapshot_path) if snapshot_path else {}

    return {
        "status": "ready",
        "model_id": model_id,
        "model_type": info.get("model_type"),
        "best_sharpe": info.get("best_sharpe"),
        "snapshot_path": snapshot_path,
        "train_range": f"{meta.get('train_start','?')} → {meta.get('train_end','?')}",
    }


class PredictWithDataRequest(PydBaseModel):
    model_type: str
    features: list[list[float]]  # 2D array: [n_samples, n_features]


@router.post("/active/predict-with-data")
def predict_with_features(req: PredictWithDataRequest):
    """Load active model, predict on provided feature matrix.

    Returns predicted classes + probabilities.
    """
    from api.config import settings
    from pipeline.model_persistence import load_model_only, get_active_model_id
    from pipeline.model_registry_disk import get_all_deployed
    import numpy as np

    model_id = get_active_model_id(req.model_type)
    if not model_id:
        raise HTTPException(404, f"No active model for type {req.model_type}")

    rows = get_all_deployed(settings.db_full_path)
    snapshot_path = None
    for r in rows:
        if r.get("id") == model_id:
            snapshot_path = r.get("snapshot_path")
            break
    if not snapshot_path or not os.path.exists(snapshot_path):
        raise HTTPException(500, "Active model snapshot not found on disk")

    model = load_model_only(snapshot_path)
    X = np.array(req.features, dtype=np.float64)

    try:
        proba = model.predict_proba(X)
    except Exception:
        from datetime import datetime, timezone
        preds = model.predict(X)
        proba = np.zeros((len(preds), 3))
        for i, c in enumerate(preds):
            cls = min(max(int(c) + 1, 0), 2)
            proba[i, cls] = 1.0

    classes = np.argmax(proba, axis=1) - 1
    confidences = np.max(proba, axis=1)

    # Log to live_predictions
    try:
        from pipeline.data_sqlite import DataStore
        from datetime import datetime, timezone
        store = DataStore(settings.db_full_path)
        now = datetime.now(timezone.utc).isoformat()
        with store._cursor() as (conn, cur):
            for i in range(len(classes)):
                cur.execute(
                    "INSERT INTO live_predictions (timestamp, model_id, pair, timeframe, predicted_class, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                    (now, model_id, "EURUSD", "H1", int(classes[i]), float(confidences[i])),
                )
    except Exception:
        pass

    return {
        "model_id": model_id,
        "model_type": req.model_type,
        "predictions": [
            {"class": int(c), "confidence": float(conf)}
            for c, conf in zip(classes, confidences)
        ],
    }


@router.get("/active/compare")
def compare_active_models():
    """Compare live predictions vs saved backtest metrics for active models."""
    from api.config import settings
    from pipeline.data_sqlite import DataStore
    from pipeline.model_registry_disk import get_all_deployed

    active = [r for r in get_all_deployed(settings.db_full_path) if r.get("status") == "active"]
    if not active:
        return {"active_models": [], "message": "No active models deployed"}

    store = DataStore(settings.db_full_path)
    comparisons = []
    for m in active:
        mid = m["id"]
        with store._cursor() as (conn, cur):
            cur.execute(
                "SELECT COUNT(*), AVG(CASE WHEN predicted_class != 0 THEN 1.0 ELSE 0.0 END) FROM live_predictions WHERE model_id = ?",
                (mid,),
            )
            row = cur.fetchone()
        live_count = int(row[0]) if row and row[0] else 0
        live_active_rate = float(row[1]) if row and row[1] and live_count > 0 else 0.0

        comparisons.append({
            "model_id": mid,
            "model_type": m.get("model_type"),
            "backtest_sharpe": m.get("best_sharpe"),
            "backtest_return": m.get("best_return"),
            "live_predictions": live_count,
            "live_active_rate": round(live_active_rate, 4),
        })

    return {"active_models": comparisons}


# ────────────────────────────────────────────────────────────
#  Forward Test — run a saved model on any date range
# ────────────────────────────────────────────────────────────
class ForwardTestRequest(PydBaseModel):
    model_id: str
    pair: str = "EURUSD"
    timeframe: str = "H1"
    start_date: str
    end_date: str
    position_sizing: str = "fixed"
    trading_costs: bool = True


class ForwardTestResponse(PydBaseModel):
    job_id: str
    status: str


@router.post("/{model_id}/forward-test", response_model=ForwardTestResponse, status_code=202)
def forward_test_model(model_id: str, req: ForwardTestRequest):
    from api.config import settings
    from pipeline.model_registry_disk import get_all_deployed
    from pipeline.model_persistence import validate_snapshot
    from api.services import JobManager
    from pipeline.data_sqlite import DataStore
    import uuid

    rows = get_all_deployed(settings.db_full_path)
    model_row = None
    for r in rows:
        if r.get("id") == model_id:
            model_row = r
            break
    if model_row is None:
        raise HTTPException(404, f"Model {model_id} not found in deployed models")

    snapshot_path = model_row.get("snapshot_path", "")
    if not snapshot_path or not os.path.isdir(snapshot_path):
        raise HTTPException(500, "Snapshot directory missing on disk")

    ok, reason = validate_snapshot(snapshot_path)
    if not ok:
        raise HTTPException(400, f"Snapshot validation failed: {reason}")

    store = DataStore(settings.db_full_path)
    jm = JobManager(store)
    job_id = str(uuid.uuid4())
    config = {
        "model_id": model_id,
        "snapshot_path": snapshot_path,
        "pair": req.pair,
        "timeframe": req.timeframe,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "position_sizing": req.position_sizing,
        "trading_costs": req.trading_costs,
    }

    jm.create_job_atomic(job_id, "forward_test", config, max_active=settings.max_concurrent_backtests)

    from api.tasks import IS_DESKTOP
    if IS_DESKTOP:
        import threading
        from api.tasks import _run_forward_test_impl

        def _run_desktop():
            try:
                _run_forward_test_impl(job_id, config)
            except Exception as exc:
                import traceback
                traceback.print_exc()

        threading.Thread(target=_run_desktop, daemon=True).start()
    else:
        from api.tasks import run_forward_test_task
        run_forward_test_task.delay(job_id, config)

    return ForwardTestResponse(job_id=job_id, status="pending")
