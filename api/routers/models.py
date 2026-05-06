"""Model registry and hyperparameter endpoints."""
from fastapi import APIRouter

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
