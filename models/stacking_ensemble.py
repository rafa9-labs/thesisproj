"""Stacking Ensemble — OOF meta-learner for combining base models.

Unlike MetaEnsemble (which only does hard/soft voting), this implements
proper stacked generalization via sklearn's StackingClassifier, which:
  - Trains each base model on a fold of the training data
  - Generates out-of-fold predictions for every sample
  - Trains a meta-learner (LogisticRegression by default) on OOF predictions

Literature:
  - ResearchGate (2025): "Stacking Ensemble Learning: Combining XGBoost,
    LightGBM, CatBoost, AdaBoost with RF Meta Model — consistently
    outperforms individual models"
  - TheAlphaScientist: exact approach for stock prediction ML
  - Kaggle: stacking is the dominant winning strategy

This implements the BaseModel interface so it plugs into the existing
pipeline with zero changes to backtester/tuning code.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as _np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel


class StackingEnsemble(BaseModel):
    model_type: str = "stacking_ensemble"
    is_deep: bool = False
    supports_proba: bool = True

    def __init__(self, base_models: List[Any],
                 meta_learner: Optional[Any] = None,
                 cv: int = 5,
                 method: str = "auto",
                 **kwargs):
        super().__init__(**kwargs)
        self._base_models = base_models
        self._cv = max(2, int(cv))
        self._method = method.lower()
        self._model_type_names: List[str] = list(kwargs.get("stack_sub_models", []))

        if meta_learner is not None:
            self._meta_learner_builder = meta_learner
        else:
            meta_seed = kwargs.get("seed")
            self._meta_learner_builder = LogisticRegression(
                penalty="l2", solver="lbfgs",
                max_iter=1000, class_weight="balanced",
                random_state=meta_seed if meta_seed else None
            )

        self._stacking_clf: Optional[StackingClassifier] = None
        self._fitted = False

    def fit(self, X, y, **kwargs) -> "StackingEnsemble":
        if len(self._base_models) < 2:
            raise ValueError("StackingEnsemble requires at least 2 base models")

        estimator_tuples = []
        for i, m in enumerate(self._base_models):
            name = self._model_type_names[i] if i < len(self._model_type_names) else f"model_{i}"
            estimator_tuples.append((name, m))

        self._stacking_clf = StackingClassifier(
            estimators=estimator_tuples,
            final_estimator=self._meta_learner_builder,
            cv=self._cv,
            stack_method=self._method,
            n_jobs=1,
            passthrough=True,
        )
        self._stacking_clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            return _np.full(len(X), 0)
        return self._stacking_clf.predict(X)

    def predict_proba(self, X):
        if not self._fitted:
            return _np.zeros((len(X), 3))
        proba = self._stacking_clf.predict_proba(X)
        return _ensure_3class(proba, len(X))

    def get_params(self) -> dict:
        return {
            "cv": self._cv,
            "method": self._method,
            "model_types": self._model_type_names,
            **self._init_kwargs,
        }

    def __repr__(self):
        n = len(self._base_models)
        return f"StackingEnsemble(n_models={n}, cv={self._cv}, types={self._model_type_names})"


def _ensure_3class(proba: _np.ndarray, n_samples: int) -> _np.ndarray:
    nc = proba.shape[1] if proba.ndim == 2 else 1
    if nc == 3:
        return proba
    if nc == 1:
        flat = proba.reshape(-1) if proba.ndim == 2 else proba
        out = _np.zeros((n_samples, 3))
        for i, v in enumerate(flat):
            cls = min(max(int(v) + 1, 0), 2)
            out[i, cls] = 1.0
        return out
    if nc == 2:
        out = _np.zeros((n_samples, 3))
        out[:, 0] = proba[:, 0]
        out[:, 2] = proba[:, 1]
        return out
    return proba[:, :3] if proba.shape[1] >= 3 else proba
