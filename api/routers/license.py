"""License API endpoints — activation, deactivation, status, features."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.licensing.manager import get_license_manager, LicenseManager
from api.licensing.gates import (
    FREE_MODELS, PAID_MODELS, ALL_MODELS,
    LOCKED_FEATURES, check_feature, check_model,
    get_available_models, get_locked_models,
)

router = APIRouter(prefix="/license", tags=["license"])


class ActivateRequest(BaseModel):
    license_key: str = Field(..., min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class FeatureCheckRequest(BaseModel):
    feature: str
    model: str | None = None


class DeactivateResponse(BaseModel):
    success: bool
    error: str | None = None


def _mgr() -> LicenseManager:
    return get_license_manager()


@router.get("/status")
def license_status(manager: LicenseManager = Depends(_mgr)):
    status = manager.get_status()
    return status.to_dict()


@router.post("/activate")
async def activate_license(req: ActivateRequest, manager: LicenseManager = Depends(_mgr)):
    result = await manager.activate(req.license_key)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Activation failed"))
    return result


@router.post("/deactivate")
async def deactivate_license(manager: LicenseManager = Depends(_mgr)):
    result = await manager.deactivate()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deactivation failed"))
    return result


@router.post("/verify")
async def verify_license(manager: LicenseManager = Depends(_mgr)):
    result = await manager.verify()
    return result


@router.post("/trial")
def start_trial(manager: LicenseManager = Depends(_mgr)):
    result = manager.start_trial()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Trial already active"))
    return result


@router.post("/check")
def check_access(req: FeatureCheckRequest, manager: LicenseManager = Depends(_mgr)):
    status = manager.get_status()
    feature_ok = check_feature(req.feature, status.plan) if req.feature else True
    model_ok = check_model(req.model, status.plan) if req.model else True
    return {
        "plan": status.plan,
        "feature": req.feature,
        "feature_allowed": feature_ok,
        "model": req.model,
        "model_allowed": model_ok,
    }


@router.get("/features")
def list_features(manager: LicenseManager = Depends(_mgr)):
    status = manager.get_status()
    all_features = {
        "hpo": "Hyperparameter optimization",
        "news_sentiment": "News & sentiment features",
        "advanced_execution": "Advanced execution models (Kelly, volatility sizing)",
        "cross_pair": "Cross-pair comparison",
        "trailing_stop": "Trailing stop orders",
        "breakeven_stop": "Breakeven stop management",
        "partial_close": "Partial close / scale-out",
        "kelly_sizing": "Kelly criterion position sizing",
        "volatility_sizing": "ATR-based volatility position sizing",
        "risk_manager": "Risk management framework",
        "deep_models": "Deep learning models (CNN, LSTM, Transformer)",
        "ensemble_models": "Ensemble models (CNN+LSTM+XGBoost, Adaptive Regime)",
    }
    plan = status.plan
    return {
        "plan": plan,
        "features": {
            name: {"description": desc, "allowed": check_feature(name, plan)}
            for name, desc in all_features.items()
        },
        "models": {
            m: {"allowed": check_model(m, plan)} for m in sorted(ALL_MODELS)
        },
    }