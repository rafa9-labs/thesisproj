"""MetaEnsemble — wraps N model types, trains each, combines predictions.

Combination methods:
  - majority: hard vote on {-1,0,1}, ties → 0
  - soft:     average probabilities
  - weighted: average weighted by per-model Sharpe (from config or equal)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as _np

from models.base_model import BaseModel


class MetaEnsemble(BaseModel):
    model_type: str = "meta_ensemble"
    is_deep: bool = False
    supports_proba: bool = True

    def __init__(self, sub_models: Optional[List[Any]] = None,
                 method: str = "majority",
                 weights: Optional[List[float]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._sub_models: List[Any] = sub_models or []
        self._method = method.lower()
        self._weights = weights
        self._model_types: List[str] = kwargs.get("meta_sub_models", [])

    def fit(self, X, y, **kwargs) -> "MetaEnsemble":
        for m in self._sub_models:
            m.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._sub_models or not self._fitted:
            return _np.zeros((len(X), 3))

        probas = []
        for m in self._sub_models:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
            else:
                preds = m.predict(X)
                p = _np.zeros((len(preds), 3))
                for i, c in enumerate(preds):
                    cls = min(max(int(c) + 1, 0), 2)
                    p[i, cls] = 1.0
            probas.append(_ensure_3class(p, len(X)))

        if self._method == "majority":
            return _majority_vote(probas)
        elif self._method == "weighted" and self._weights:
            w = _np.array(self._weights, dtype=float)
            w = w[:len(probas)] / max(w.sum(), 1e-8)
            return _np.average(probas, axis=0, weights=w)
        else:
            return _np.mean(probas, axis=0)

    def predict(self, X):
        return _np.argmax(self.predict_proba(X), axis=1) - 1

    def get_params(self) -> dict:
        return {
            "method": self._method,
            "model_types": self._model_types,
            "weights": self._weights,
            **self._init_kwargs,
        }

    def __repr__(self):
        n = len(self._sub_models)
        return f"MetaEnsemble(method={self._method}, n_models={n}, types={self._model_types})"


def _majority_vote(probas: List[_np.ndarray]) -> _np.ndarray:
    """Hard majority vote on {-1, 0, 1}, returning (n, 3) probabilities."""
    n = len(probas[0])
    votes = _np.zeros((n, 3), dtype=int)
    for p in probas:
        labels = _np.argmax(p, axis=1)  # 0=buy, 1=flat, 2=sell
        for i, c in enumerate(labels):
            votes[i, c] += 1

    out = _np.zeros((n, 3), dtype=float)
    for i in range(n):
        best = int(_np.argmax(votes[i]))
        tie = _np.sum(votes[i] == votes[i][best]) > 1
        out[i, best] = 0.0 if tie else 1.0
    return out


def _ensure_3class(proba: _np.ndarray, n_samples: int) -> _np.ndarray:
    """Pad probability array to (n_samples, 3)."""
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
