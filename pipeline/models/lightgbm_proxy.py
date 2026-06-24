"""LightGBM proxy feature selector using SHAP importance.

Trains a fast LightGBM model on the training feature set, runs TreeSHAP,
and returns the top-K features by importance. This provides a cheap,
model-agnostic feature pre-filtering step that works for ALL model types
(including deep sequential models — features are flattened for the proxy).

Usage:
    from pipeline.models.lightgbm_proxy import LightGBMProxy
    proxy = LightGBMProxy(top_k=40)
    keep = proxy.select(X_train, y_train)  # returns list of feature names
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

_HAS_LGBM = False
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    lgb = None

_HAS_SHAP = False
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None


class LightGBMProxy:
    """Fast LightGBM + TreeSHAP feature pre-filter."""

    def __init__(
        self,
        top_k: int = 40,
        n_estimators: int = 200,
        random_state: int = 42,
        max_samples: int = 5000,
    ):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self.selected_features_: List[str] = []

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[str]:
        """Select top-K features by SHAP importance via LightGBM proxy.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix (samples x features). For sequential models
            (LSTM/CNN/Transformer), flatten time dimension first.
        y : pd.Series
            Training labels, aligned with X.

        Returns
        -------
        list of str
            Top-K feature names selected by the proxy.
        """
        if not _HAS_LGBM or not _HAS_SHAP:
            return list(X.columns)

        if X.shape[1] <= self.top_k:
            self.selected_features_ = list(X.columns)
            return self.selected_features_

        # Subsample for speed
        if len(X) > self.max_samples:
            idx = np.random.default_rng(self.random_state).choice(
                len(X), size=self.max_samples, replace=False
            )
            X = X.iloc[idx]
            y = y.iloc[idx]

        # Drop constant/null columns (LightGBM handles NaN but not all-constant)
        valid = X.columns[X.nunique() > 1]
        X = X[valid]
        if X.shape[1] <= self.top_k:
            self.selected_features_ = list(X.columns)
            return self.selected_features_

        try:
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            imp = np.abs(shap_vals).mean(axis=0)
            ranked = [X.columns[i] for i in np.argsort(imp)[::-1]]
            self.selected_features_ = ranked[: min(self.top_k, len(ranked))]
        except Exception:
            self.selected_features_ = list(X.columns[: self.top_k])

        return self.selected_features_
