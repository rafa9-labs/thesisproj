"""
RegimeClassifier — Random Forest model for 7-class market regime labeling.

Implements the BaseModel interface so it slots into the pipeline directly.
Used both as a standalone supervised classifier (train on labeled data) and
as a prediction-time router that maps current market state → recommended models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from models.base_model import BaseModel

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from pipeline.regime_utils import (
    _REGIME_NAMES,
    _STABLE_REGIMES_FOR_MODEL,
    RegimeConfig,
)


class RegimeClassifier(BaseModel):
    """Random Forest classifier that labels market bars into 7 regime classes.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default 100).
    max_depth : int or None
        Tree depth limit (default 8, keeps inference fast for live routing).
    min_samples_leaf : int
        Minimum samples per leaf (default 50, prevents overfitting to noise).
    class_weight : str
        Handles class imbalance (default "balanced_subsample").
    random_state : int
        Seed for reproducibility.
    feature_columns : list[str] or None
        Explicit columns to use as features. If None, auto-selects all
        indicator columns (excluding regime/prediction columns).
    regime_cfg : RegimeConfig or None
        Optional threshold config passed through to detect_regimes()
        when using the rule-based fallback.
    """

    model_type: str = "regime_classifier"
    is_deep: bool = False
    supports_proba: bool = True

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
        min_samples_leaf: int = 50,
        class_weight: str = "balanced_subsample",
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
        regime_cfg: Optional[RegimeConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.feature_columns = feature_columns
        self.regime_cfg = regime_cfg or RegimeConfig()

        self._clf: Optional[RandomForestClassifier] = None
        self._model_to_regime: Dict[str, List[int]] = dict(_STABLE_REGIMES_FOR_MODEL)
        self._regime_to_models: Dict[int, List[str]] = self._build_regime_to_models()
        self._feature_names_: List[str] = []
        self._class_names_: List[str] = list(_REGIME_NAMES.values())

    def _build_regime_to_models(self) -> Dict[int, List[str]]:
        """Invert model→regime mapping to regime→models for fast lookup."""
        out: Dict[int, List[str]] = {}
        for model, regimes in self._model_to_regime.items():
            for r in regimes:
                out.setdefault(r, []).append(model)
        return out

    def fit(self, X, y=None, **kwargs) -> "RegimeClassifier":
        """Train the RF on labeled regime data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features. Must be a DataFrame so we can select columns.
            Should contain indicator columns (adx, rsi, bbw, etc.).
        y : array-like, optional
            Target regime labels (0-6). If None, looks for 'regime_7class'
            column in X.
        """
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn is required for RegimeClassifier")

        if isinstance(X, pd.DataFrame):
            if y is None and "regime_7class" in X.columns:
                y = X["regime_7class"].to_numpy(dtype=np.int32)
            # Select feature columns
            if self.feature_columns:
                feat_cols = [c for c in self.feature_columns if c in X.columns]
            else:
                # Auto-select: numeric columns, exclude regime/prediction/metadata
                exclude = {
                    "regime_7class", "regime_name", "regime_id",
                    "regime_trend", "regime_sideways", "regime_volatile",
                    "trend_score", "vol_score",
                    "time", "date", "timestamp", "open", "high", "low", "close",
                    "mid_o", "mid_h", "mid_l", "mid_c",
                    "bid_o", "bid_c", "ask_o", "ask_c",
                    "spread", "volume", "hour", "hour_sin", "hour_cos",
                }
                feat_cols = [c for c in X.columns if c not in exclude and np.issubdtype(X[c].dtype, np.number)]
            self._feature_names_ = feat_cols
            X_feat = X[feat_cols].to_numpy(dtype=np.float32)
        else:
            X_feat = np.asarray(X, dtype=np.float32)
            self._feature_names_ = [f"f{i}" for i in range(X_feat.shape[1])]

        y = np.asarray(y, dtype=np.int32)

        # Remove rows with NaN in features or labels
        valid = np.isfinite(X_feat).all(axis=1) & np.isfinite(y)
        X_clean = X_feat[valid]
        y_clean = y[valid]

        if len(X_clean) == 0:
            raise ValueError("No valid training samples after NaN removal")

        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._clf.fit(X_clean, y_clean)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Return regime class labels (0-6)."""
        self._check_fitted()
        X_feat = self._extract_features(X)
        preds = self._clf.predict(X_feat)
        return preds.astype(np.int8)

    def predict_proba(self, X) -> np.ndarray:
        """Return (n, 7) probability matrix over all regime classes.

        Missing classes (not seen during training) get zero probability.
        """
        self._check_fitted()
        X_feat = self._extract_features(X)
        raw = self._clf.predict_proba(X_feat)
        trained_classes = self._clf.classes_

        proba = np.zeros((raw.shape[0], 7), dtype=np.float32)
        for idx, cls_id in enumerate(trained_classes):
            if 0 <= cls_id < 7:
                proba[:, int(cls_id)] = raw[:, idx]
        return proba

    def recommend_models(
        self,
        X,
        top_k: int = 3,
        min_prob: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """For each bar, recommend top-k model types based on predicted regime.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        top_k : int
            Number of models to recommend per bar.
        min_prob : float
            Minimum regime probability to consider. Regimes below this
            probability are skipped.

        Returns
        -------
        list[dict]
            Each entry has: bar_index, regime_name, regime_prob,
            recommended_models (list of model type strings).
        """
        proba = self.predict_proba(X)
        n = proba.shape[0]

        results: List[Dict[str, Any]] = []
        for i in range(n):
            row_probs = proba[i]
            sorted_idx = np.argsort(-row_probs)

            candidates = []
            for regime_id in sorted_idx:
                if row_probs[regime_id] < min_prob:
                    continue
                if regime_id not in _REGIME_NAMES:
                    continue
                # Find models that support this regime
                regime_models = self._regime_to_models.get(regime_id, [])
                for m in regime_models:
                    if m not in candidates:
                        candidates.append(m)
                if len(candidates) >= top_k:
                    break

            best_regime = int(sorted_idx[0])
            results.append({
                "bar_index": i,
                "regime_id": best_regime,
                "regime_name": _REGIME_NAMES.get(best_regime, "unknown"),
                "regime_prob": float(row_probs[best_regime]),
                "all_probs": {_REGIME_NAMES.get(j, f"c{j}"): float(row_probs[j]) for j in range(7)},
                "recommended_models": candidates[:top_k],
            })
        return results

    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Return feature importance dict after fitting."""
        if self._clf is None or not self._feature_names_:
            return None
        return dict(zip(self._feature_names_, self._clf.feature_importances_.tolist()))

    def update_model_regime_map(self, mapping: Dict[str, List[int]]):
        """Override the static model→regime mapping with empirically learned weights."""
        self._model_to_regime = dict(mapping)
        self._regime_to_models = self._build_regime_to_models()

    def _extract_features(self, X) -> np.ndarray:
        """Convert DataFrame input to feature matrix using stored feature names."""
        if isinstance(X, pd.DataFrame):
            available = [c for c in self._feature_names_ if c in X.columns]
            missing = set(self._feature_names_) - set(available)
            if missing:
                raise ValueError(
                    f"Missing feature columns: {sorted(missing)}. "
                    f"Available: {sorted(available)}"
                )
            return X[available].to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def _check_fitted(self):
        if self._clf is None or not self._fitted:
            raise RuntimeError("RegimeClassifier must be fit() before predict()")

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "feature_columns": self.feature_columns,
        }

    def free(self):
        self._clf = None
        self._fitted = False
