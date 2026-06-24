"""
Meta-Labeler — binary secondary model per de Prado AFML Ch. 3.6.

Learns P(trade_is_winner | primary_model_signal, features).  At inference
time gates the primary model: if P(win) < threshold, suppress the trade.

Unlike the CommitteeMetaLearner (which predicts next-bar direction from
committee output), the MetaLabeler predicts *trade profitability*
conditioned on the primary signal being taken.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MetaLabeler:
    """Binary classifier predicting whether a trade signal will be profitable.

    Parameters
    ----------
    override_threshold : float
        Minimum P(win) to take the trade (default 0.50).

    Architecture
    ------------
    LightGBM binary classifier with:
      - max_depth=3  (prevents overfitting on sparse winning trades)
      - class_weight derived from scale_pos_weight (handles imbalance)
      - Features: primary model's 3-class probabilities + confidence
        + conviction spread + directional commitment + regime_id
    """

    _FEATURE_NAMES = [
        "primary_prob_short",
        "primary_prob_flat",
        "primary_prob_long",
        "primary_confidence",
        "conviction_spread",        # P_long - P_short
        "directional_commitment",   # 1 - P_flat
        "regime_id",
    ]

    def __init__(self, override_threshold: float = 0.50):
        self.override_threshold = override_threshold
        self._model: Any = None
        self._trained: bool = False
        self._train_accuracy: float = 0.0

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def accuracy(self) -> float:
        return self._train_accuracy

    # ── Target construction ──────────────────────────────────────────

    @staticmethod
    def build_targets(bar_predictions: List[Dict[str, Any]]) -> np.ndarray:
        """Build binary meta-labels from per-bar committee predictions.

        Meta-label = 1 if the primary model's directional signal matches
        the next-bar return direction (trade would have been profitable).

        Bars where committee_signal == 0 are skipped (no trade → no label).
        """
        y = []
        for bar in bar_predictions:
            signal = int(bar.get("committee_signal", 0))
            if signal == 0:
                continue
            next_return = float(bar.get("next_return", 0.0))
            win = (signal == 1 and next_return > 0) or (signal == -1 and next_return < 0)
            y.append(1 if win else 0)
        return np.array(y, dtype=np.int32)

    # ── Feature construction ─────────────────────────────────────────

    @staticmethod
    def build_features(bar_predictions: List[Dict[str, Any]]) -> np.ndarray:
        """Build feature matrix from per-bar committee predictions.

        Only includes bars with non-zero signals (matching build_targets).
        """
        X = []
        for bar in bar_predictions:
            signal = int(bar.get("committee_signal", 0))
            if signal == 0:
                continue
            p_short = float(bar.get("committee_prob_short", 0.33))
            p_flat = float(bar.get("committee_prob_flat", 0.34))
            p_long = float(bar.get("committee_prob_long", 0.33))
            confidence = float(bar.get("committee_confidence", 0.5))
            regime_id = int(bar.get("regime_id", 6))

            X.append([
                p_short,
                p_flat,
                p_long,
                confidence,
                p_long - p_short,       # conviction_spread
                1.0 - p_flat,            # directional_commitment
                float(regime_id),
            ])
        return np.array(X, dtype=np.float32)

    # ── Training ─────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train on OOS fold predictions.

        Returns in-sample accuracy (for monitoring only — real validation
        happens on subsequent OOS folds via WFO).
        """
        if len(X) < 20:
            self._trained = False
            self._train_accuracy = 0.0
            logger.debug("MetaLabeler: too few samples (%d), skipping", len(X))
            return 0.0

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos < 3 or n_neg < 3:
            self._trained = False
            self._train_accuracy = 0.0
            logger.debug("MetaLabeler: class imbalance too extreme (pos=%d, neg=%d)", n_pos, n_neg)
            return 0.0

        scale_pos_weight = n_neg / max(n_pos, 1.0)

        try:
            from lightgbm import LGBMClassifier

            self._model = LGBMClassifier(
                max_depth=3,
                n_estimators=100,
                scale_pos_weight=min(scale_pos_weight, 10.0),
                random_state=42,
                verbose=-1,
                class_weight="balanced",
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier

            self._model = RandomForestClassifier(
                n_estimators=min(50, len(X)),
                max_depth=3,
                random_state=42,
                class_weight="balanced",
                n_jobs=1,
            )

        self._model.fit(X, y)
        self._trained = True
        self._train_accuracy = float(self._model.score(X, y))
        logger.debug("MetaLabeler: trained on %d samples, accuracy=%.3f, pos=%d neg=%d",
                     len(X), self._train_accuracy, n_pos, n_neg)
        return self._train_accuracy

    # ── Inference ────────────────────────────────────────────────────

    def predict_win_prob(
        self,
        primary_probs: Tuple[float, float, float],
        regime_id: int = 6,
    ) -> float:
        """Predict P(trade_is_winner) given primary model probabilities.

        Parameters
        ----------
        primary_probs : (P_short, P_flat, P_long)
        regime_id : int 0-6

        Returns
        -------
        float in [0, 1]
        """
        if not self._trained or self._model is None:
            return 0.50

        p_short, p_flat, p_long = (float(v) for v in primary_probs)
        confidence = max(p_short, p_flat, p_long)

        features = np.array([[
            p_short,
            p_flat,
            p_long,
            confidence,
            p_long - p_short,
            1.0 - p_flat,
            float(regime_id),
        ]], dtype=np.float32)

        return float(self._model.predict_proba(features)[0, 1])

    def should_trade(
        self,
        primary_signal: int,
        primary_probs: Tuple[float, float, float],
        regime_id: int = 6,
    ) -> Tuple[bool, float]:
        """Decide whether to take the primary model's trade signal.

        Returns (should_trade, P_win).
        """
        if primary_signal == 0:
            return False, 0.0
        if not self._trained:
            return True, 0.50  # pass-through when untrained

        p_win = self.predict_win_prob(primary_probs, regime_id)
        return p_win >= self.override_threshold, p_win

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        import joblib

        joblib.dump(
            {
                "model": self._model,
                "trained": self._trained,
                "accuracy": self._train_accuracy,
                "override_threshold": self.override_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MetaLabeler":
        import joblib

        data = joblib.load(path)
        if isinstance(data, dict):
            meta = cls(
                override_threshold=data.get("override_threshold", 0.50),
            )
            meta._model = data.get("model")
            meta._trained = data.get("trained", False)
            meta._train_accuracy = data.get("accuracy", 0.0)
            return meta
        meta = cls()
        meta._model = data
        meta._trained = True
        return meta
