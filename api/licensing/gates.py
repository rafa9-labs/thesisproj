"""Feature gates — defines which features are available for each plan.

Free plan:
  - 3 basic models: logistic, xgboost, random_forest
  - Fixed-lot position sizing only
  - No HPO (hyperparameter optimization)
  - No news/sentiment features
  - No cross-pair comparison
  - No advanced execution (trailing stops, breakeven, partial close)

Trial plan (14 days):
  - Full access to all features
  - Expires after 14 days

Pro/Team plan:
  - All features unlocked
"""

from __future__ import annotations

from typing import FrozenSet

FREE_MODELS: FrozenSet[str] = frozenset({"logistic", "xgboost", "random_forest"})
FREE_EXECUTION_TYPES: FrozenSet[str] = frozenset({"fixed_lot"})
LOCKED_FEATURES: FrozenSet[str] = frozenset({
    "hpo",
    "news_sentiment",
    "advanced_execution",
    "cross_pair",
    "trailing_stop",
    "breakeven_stop",
    "partial_close",
    "kelly_sizing",
    "volatility_sizing",
    "risk_manager",
    "deep_models",
    "ensemble_models",
})

PAID_MODELS: FrozenSet[str] = frozenset({
    "svm", "decision_tree", "cnn", "lstm", "transformer",
    "dqn", "gru", "gru_lstm",
    "lightgbm", "catboost",
    "ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime",
    "meta_ensemble", "stacking_ensemble", "regime_classifier",
})

ALL_MODELS: FrozenSet[str] = FREE_MODELS | PAID_MODELS


def check_model(model_name: str, plan: str) -> bool:
    if plan in ("pro", "team", "trial"):
        return True
    return model_name in FREE_MODELS


def check_feature(feature: str, plan: str) -> bool:
    if plan in ("pro", "team", "trial"):
        return True
    return feature not in LOCKED_FEATURES


def get_available_models(plan: str) -> list[str]:
    if plan in ("pro", "team", "trial"):
        return sorted(ALL_MODELS)
    return sorted(FREE_MODELS)


def get_locked_models(plan: str) -> list[str]:
    if plan in ("pro", "team", "trial"):
        return []
    return sorted(PAID_MODELS)