"""Model registry endpoint."""
from fastapi import APIRouter

from api.schemas.backtest import ModelInfo, ModelListResponse

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
