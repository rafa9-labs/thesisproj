"""Model family classification for trial budgets and diversity constraints.

Centralises model-type logic that was previously scattered across
committee_backtester.py, tuning/runner.py, and the committee API.
"""
from __future__ import annotations

from enum import Enum

DEEP_MODELS: frozenset = frozenset({
    "cnn", "lstm", "transformer", "gru", "gru_lstm",
})

CLASSICAL_MODELS: frozenset = frozenset({
    "logistic", "svm", "random_forest", "decision_tree",
    "xgboost", "lightgbm", "catboost",
})

ENSEMBLE_MODELS: frozenset = frozenset({
    "ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost",
    "stacking_ensemble", "meta_ensemble",
})

TREE_MODELS: frozenset = frozenset({
    "xgboost", "lightgbm", "catboost", "random_forest", "decision_tree",
})

LINEAR_MODELS: frozenset = frozenset({
    "logistic", "svm",
})

ALL_MODELS: list[str] = sorted(DEEP_MODELS | CLASSICAL_MODELS | ENSEMBLE_MODELS | {"dqn"})
CORE_MODELS: list[str] = [
    "logistic", "svm", "random_forest", "xgboost",
    "lightgbm", "catboost", "lstm", "ensemble_adaptive_regime",
]


class ModelStatus(str, Enum):
    SUCCESS = "success"
    TIMED_OUT = "timed_out"
    CRASHED = "crashed"
    NO_FOLDS = "no_folds"
    SKIPPED = "skipped"


def get_trial_budget(model_type: str) -> tuple[int, int]:
    """Return (n_trials, n_startup_trials) based on architectural family.

    Deep models require more trials because they have larger search spaces
    and longer per-trial training times.  Ensembles are treated similarly
    because they contain deep components.

    With ASHA pruning enabled, these are upper bounds — the pruner may
    terminate a study earlier if successive trials show no improvement.
    """
    if model_type in DEEP_MODELS:
        return (50, 25)
    if model_type in ENSEMBLE_MODELS:
        return (35, 18)
    return (25, 12)


def get_model_family(model_type: str) -> str:
    """Classify a model into 'deep', 'classical', 'ensemble', or 'unknown'."""
    if model_type in DEEP_MODELS:
        return "deep"
    if model_type in ENSEMBLE_MODELS:
        return "ensemble"
    if model_type in CLASSICAL_MODELS:
        return "classical"
    return "unknown"


def is_gpu_model(model_type: str) -> bool:
    """True if this model type requires GPU-serial execution (TF/PyTorch)."""
    return model_type in DEEP_MODELS or model_type in (
        "ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost",
    )
