"""
Committee Meta-Learner — learns when to trust the committee's blended signals.

Trains on OOS committee predictions from Phase 5 WFO folds.
A LightGBM classifier learns to predict next-bar direction from:
  - committee signal, confidence, probability distribution
  - market regime, volatility, signal consistency

At inference time, gates the committee: follow, fade, or stay flat.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np


class CommitteeMetaLearner:
    """Meta-learner that gates a regime committee's output.

    Parameters
    ----------
    max_depth : int
        LightGBM max tree depth (default 3, prevents overfitting).
    n_estimators : int
        Number of boosting rounds (default 100).
    override_threshold : float
        Minimum meta-learner confidence to override the committee (default 0.55).
    """

    _FEATURE_NAMES = [
        "committee_signal",
        "committee_confidence",
        "conviction",
        "regime_id",
        "signal_consistency",
        "bar_volatility",
        "bars_since_signal",
    ]

    def __init__(
        self,
        max_depth: int = 3,
        n_estimators: int = 100,
        override_threshold: float = 0.55,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.override_threshold = override_threshold
        self._model = None
        self._trained = False
        self._accuracy: float = 0.0

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def accuracy(self) -> float:
        return self._accuracy

    def build_features(
        self,
        bar: dict,
        prev_signal: int = 0,
        bar_volatility: float = 0.0,
        bars_since_signal: int = 0,
    ) -> List[float]:
        return [
            float(bar.get("committee_signal", 0)),
            float(bar.get("committee_confidence", 0.5)),
            float(bar.get("committee_prob_long", 0.5))
            - float(bar.get("committee_prob_short", 0.5)),
            float(bar.get("regime_id", 6)),
            float(int(bar.get("committee_signal", 0)) == prev_signal)
            if prev_signal is not None
            else 0.0,
            float(bar_volatility),
            float(bars_since_signal),
        ]

    def train(
        self,
        fold_predictions_paths: List[str],
        validation_split: float = 0.20,
    ) -> float:
        """Train on OOS predictions from Phase 5 WFO folds.

        Parameters
        ----------
        fold_predictions_paths : list[str]
            Paths to fold_predictions.json files from CommitteeBacktester.
        validation_split : float
            Fraction of data to hold out for validation (chronological tail).

        Returns
        -------
        float
            Validation accuracy (0–1). F1-weighted for 3-class.
        """
        X_all: List[List[float]] = []
        y_all: List[int] = []

        for path in fold_predictions_paths:
            with open(path) as f:
                folds = json.load(f)
            for fold in folds if isinstance(folds, list) else [folds]:
                bars = fold.get("bars", [])
                if not bars:
                    continue
                prev_signal = 0
                bars_since = 0

                for bar in bars:
                    features = self.build_features(
                        bar,
                        prev_signal=prev_signal,
                        bar_volatility=abs(float(bar.get("next_return", 0.0))),
                        bars_since_signal=bars_since,
                    )
                    X_all.append(features)
                    y_all.append(int(bar.get("next_direction", 1)))

                    sig = int(bar.get("committee_signal", 0))
                    if sig != 0:
                        prev_signal = sig
                        bars_since = 0
                    else:
                        bars_since += 1

        if len(X_all) < 100:
            self._trained = False
            self._accuracy = 0.0
            return 0.0

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int32)

        split_idx = max(100, int(len(X) * (1.0 - validation_split)))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        try:
            from lightgbm import LGBMClassifier

            self._model = LGBMClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                num_class=3,
                random_state=42,
                verbose=-1,
                class_weight="balanced",
            )
            self._model.fit(X_train, y_train)
            self._accuracy = float(self._model.score(X_val, y_val))
            self._trained = True
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier

            self._model = RandomForestClassifier(
                n_estimators=min(50, self.n_estimators),
                max_depth=self.max_depth,
                random_state=42,
                class_weight="balanced",
                n_jobs=1,
            )
            self._model.fit(X_train, y_train)
            self._accuracy = float(self._model.score(X_val, y_val))
            self._trained = True

        return self._accuracy

    def predict(
        self,
        committee_signal: int,
        committee_confidence: float,
        prob_short: float = 0.33,
        prob_flat: float = 0.34,
        prob_long: float = 0.33,
        regime_id: int = 6,
        bar_volatility: float = 0.0,
        bars_since_signal: int = 0,
        prev_signal: int = 0,
    ) -> Tuple[int, float, bool]:
        """Predict trade direction, optionally overriding the committee.

        Parameters
        ----------
        committee_signal : int
            The committee's blended signal (-1 sell, 0 flat, 1 buy).
        committee_confidence : float
            Max probability from the committee (0-1).
        prob_short, prob_flat, prob_long : float
            Committee's 3-class probability distribution.
        regime_id : int
            Current market regime (0-6).
        bar_volatility : float
            Recent bar return magnitude (proxy for vol).
        bars_since_signal : int
            Bars since the last non-zero committee signal.
        prev_signal : int
            The previous bar's committee signal.

        Returns
        -------
        (signal, confidence, overrode) : (int, float, bool)
            Final trade direction, meta-learner confidence, whether
            the meta-learner overrode the committee.
        """
        if not self._trained or self._accuracy < 0.40:
            return committee_signal, committee_confidence, False

        features = np.array([[
            float(committee_signal),
            float(committee_confidence),
            float(prob_long) - float(prob_short),
            float(regime_id),
            float(int(committee_signal == prev_signal)) if prev_signal is not None else 0.0,
            float(bar_volatility),
            float(bars_since_signal),
        ]], dtype=np.float32)

        meta_proba = self._model.predict_proba(features)[0]
        meta_signal = int(np.argmax(meta_proba)) - 1
        meta_conf = float(meta_proba.max())

        if meta_signal != committee_signal and meta_conf >= self.override_threshold:
            return meta_signal, meta_conf, True
        return committee_signal, meta_conf, False

    def save(self, path: str):
        import joblib

        joblib.dump(
            {
                "model": self._model,
                "trained": self._trained,
                "accuracy": self._accuracy,
                "override_threshold": self.override_threshold,
                "max_depth": self.max_depth,
                "n_estimators": self.n_estimators,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CommitteeMetaLearner":
        import joblib

        data = joblib.load(path)
        if isinstance(data, dict):
            meta = cls(
                max_depth=data.get("max_depth", 3),
                n_estimators=data.get("n_estimators", 100),
                override_threshold=data.get("override_threshold", 0.55),
            )
            meta._model = data["model"]
            meta._trained = data["trained"]
            meta._accuracy = data["accuracy"]
            return meta
        meta = cls()
        meta._model = data
        meta._trained = True
        return meta
