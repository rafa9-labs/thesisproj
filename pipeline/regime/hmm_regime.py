"""
HMM Regime Detector — probabilistic regime detection via GaussianHMM.

Replaces the rule-based ADX/EMA/ATR classifier in live_committee_runner
and the GMM-based detector in committee_backtester with a proper Hidden
Markov Model that captures state transition dynamics.

Design per de Prado AFML Ch. 4, 15:
  - Unsupervised latent state detection
  - Markov transition matrix models regime persistence
  - BIC selects optimal number of states (3-7)
  - Fixed random seed anchors state semantics across folds
  - Train once on first-fold in-sample data, freeze, reuse for all folds
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Probabilistic regime detection via Hidden Markov Model.

    Parameters
    ----------
    n_states : int or None
        Number of HMM states. If None, auto-selected via BIC scan [3, 7].
    random_state : int
        Fixed seed for reproducible state assignments across folds.
    covariance_type : str
        HMM covariance type (default 'full').
    n_iter : int
        Max EM iterations (default 100).

    Feature set (backward-looking only, no future leakage):
      - 20-bar rolling return mean
      - 20-bar rolling return std (realized vol proxy)
      - Spread / mid ratio
      - 5-bar return autocorrelation
    """

    def __init__(
        self,
        n_states: Optional[int] = None,
        random_state: int = 42,
        covariance_type: str = "full",
        n_iter: int = 100,
    ):
        self.n_states = n_states
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self._hmm = None
        self._fitted = False
        self._state_to_regime: Dict[int, int] = {}
        self._regime_names: List[str] = []
        self._feature_names: List[str] = []
        self._bic: float = float("inf")
        self._selected_n_states: int = 3

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def selected_n_states(self) -> int:
        return self._selected_n_states

    @property
    def bic(self) -> float:
        return self._bic

    # ── Feature computation ───────────────────────────────────────────

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute HMM input features from OHLC DataFrame.

        Expects columns: mid_c (close), mid_o (open), spread (optional).
        All computations are backward-looking (no future data).

        Returns DataFrame with columns:
          ret_mean_20, ret_std_20, ret_ac1_5, spread_ratio
        """
        out = pd.DataFrame(index=df.index)

        # Log returns
        if "mid_c" in df.columns:
            rets = np.log(df["mid_c"] / df["mid_c"].shift(1)).astype(np.float32)
        else:
            rets = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float32)

        # Return mean and std (20-bar rolling)
        out["ret_mean_20"] = rets.rolling(20, min_periods=5).mean().astype(np.float32)
        out["ret_std_20"] = rets.rolling(20, min_periods=5).std().astype(np.float32)

        # Return autocorrelation (5-bar lag)
        out["ret_ac1_5"] = rets.rolling(20, min_periods=5).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 5 else 0.0,
            raw=False,
        ).astype(np.float32)

        # Spread ratio
        if "spread" in df.columns:
            spread = df["spread"].astype(np.float32)
        else:
            spread = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float32)
        mid = df["mid_c"].astype(np.float32) if "mid_c" in df.columns else pd.Series(1.0, index=df.index)
        out["spread_ratio"] = (spread / mid.replace(0, np.nan)).fillna(0.0).astype(np.float32)

        self_feature_names = ["ret_mean_20", "ret_std_20", "ret_ac1_5", "spread_ratio"]

        return out[self_feature_names]

    def _extract_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract clean feature matrix for HMM fitting/prediction."""
        feat_df = self.compute_features(df)
        self._feature_names = list(feat_df.columns)

        X = feat_df.fillna(0.0).to_numpy(np.float32)
        # Drop rows at start where rolling windows haven't filled
        valid = np.isfinite(X).all(axis=1)
        if valid.sum() < 10:
            return X[:0]
        return X[valid]

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, n_states_override: Optional[int] = None) -> "HMMRegimeDetector":
        """Fit HMM on OHLC DataFrame.

        If n_states is None, BIC scans [3, 7] to auto-select.
        After fitting, maps HMM latent states to regime IDs (0-6).
        """
        from hmmlearn.hmm import GaussianHMM

        X = self._extract_feature_matrix(df)
        if len(X) < 20:
            logger.warning("HMMRegimeDetector: too few valid bars (%d), skipping fit", len(X))
            return self

        n_states = n_states_override or self.n_states

        if n_states is None:
            n_states = self._select_n_states_bic(X)
        else:
            n_states = int(n_states)

        self._selected_n_states = n_states
        self._hmm = GaussianHMM(
            n_components=n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",
            verbose=False,
        )
        self._hmm.fit(X)
        self._fitted = True

        # Map HMM states to regime IDs
        self._map_states_to_regimes(df, X)

        logger.info(
            "HMMRegimeDetector: fitted %d states, BIC=%.1f, regime map=%s",
            n_states, self._hmm.bic(X) if hasattr(self._hmm, "bic") else float("inf"),
            self._state_to_regime,
        )
        return self

    def _select_n_states_bic(self, X: np.ndarray) -> int:
        """Select n_states by minimizing BIC over range [3, 7]."""
        from hmmlearn.hmm import GaussianHMM

        best_n = 3
        best_bic = float("inf")

        for n in range(3, 8):
            try:
                hmm = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    init_params="stmc",
                    verbose=False,
                )
                hmm.fit(X)
                bic = hmm.bic(X) if hasattr(hmm, "bic") else float("inf")
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception as e:
                logger.debug("HMM BIC scan: n=%d failed (%s)", n, e)
                continue

        self._bic = best_bic
        logger.info("HMM BIC scan: selected n_states=%d (BIC=%.1f)", best_n, best_bic)
        return best_n

    def _map_states_to_regimes(self, df: pd.DataFrame, X: np.ndarray):
        """Map each HMM latent state to a regime ID (0-6).

        Strategy: predict state for each bar, then compute per-state
        statistics (mean return, return std). Map to regime based on
        these characteristics:

        Regime mapping logic:
          - trend_up (1):    strongest positive mean return
          - trend_down (2):  strongest negative mean return
          - high_volatile (5): highest return std
          - mean_reverting (3): low abs(mean_return), moderate std
          - quiet_squeeze (0): lowest return std
          - breakout (4): high std, moderate mean return
          - sideways (6): fallback
        """
        if not self._fitted or self._hmm is None:
            self._state_to_regime = {i: 6 for i in range(self._selected_n_states)}
            return

        states = self._hmm.predict(X)

        # Compute returns for each bar
        rets = np.log(df["mid_c"].values.astype(np.float64) / np.roll(df["mid_c"].values.astype(np.float64), 1))
        rets = rets[len(df) - len(states):]  # align
        rets = np.nan_to_num(rets, nan=0.0)

        state_stats = {}
        for s in range(self._selected_n_states):
            mask = states == s
            if mask.sum() < 3:
                state_stats[s] = {"mean_ret": 0.0, "std_ret": 0.0, "count": 0}
                continue
            s_rets = rets[mask]
            state_stats[s] = {
                "mean_ret": float(np.mean(s_rets)),
                "std_ret": float(np.std(s_rets)),
                "count": int(mask.sum()),
            }

        # Sort states by characteristics
        means = {s: st["mean_ret"] for s, st in state_stats.items()}
        stds = {s: st["std_ret"] for s, st in state_stats.items()}

        # trend_up: most positive mean return
        trend_up_state = max(means, key=means.get) if means else 0
        # trend_down: most negative mean return
        trend_down_state = min(means, key=means.get) if means else 1
        # high_volatile: highest std
        high_vol_state = max(stds, key=stds.get) if stds else 2

        # Assign remaining states
        remaining = set(range(self._selected_n_states)) - {trend_up_state, trend_down_state, high_vol_state}
        remaining = sorted(remaining)

        # quiet_squeeze: lowest std among remaining
        quiet_state = None
        if remaining:
            quiet_std = min(stds[s] for s in remaining)
            quiet_state = [s for s in remaining if stds[s] == quiet_std][0]
            remaining.remove(quiet_state)

        mapping: Dict[int, int] = {
            trend_up_state: 1,
            trend_down_state: 2,
            high_vol_state: 5,
        }

        if quiet_state is not None:
            mapping[quiet_state] = 0

        # Any remaining states → mean_reverting (3), breakout (4), sideways (6)
        for i, s in enumerate(remaining):
            if i == 0 and len(remaining) == 2:
                mapping[s] = 3  # mean_reverting
            elif i == 1 and len(remaining) == 2:
                mapping[s] = 4  # breakout
            else:
                mapping[s] = 6  # sideways fallback

        self._state_to_regime = mapping
        self._regime_names = [
            "quiet_squeeze", "trend_up", "trend_down", "mean_reverting",
            "breakout", "high_volatile", "sideways",
        ]

        logger.info("HMM state→regime map: %s",
                     {k: self._regime_names[v] for k, v in mapping.items()})

    # ── Prediction ────────────────────────────────────────────────────

    def predict(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict regime ID and soft probabilities for each bar.

        Returns
        -------
        regime_ids : np.ndarray of int8, shape (n_bars,)
            Hard regime classification (0-6).
        probs : np.ndarray of float32, shape (n_bars, n_states)
            Soft state probabilities from HMM. Columns are HMM latent
            states. Need to be remapped to regime space via _state_to_regime.
        """
        if not self._fitted or self._hmm is None:
            n = len(df)
            regime_ids = np.full(n, 6, dtype=np.int8)
            probs = np.zeros((n, 1), dtype=np.float32)
            probs[:, 0] = 1.0
            return regime_ids, probs

        X = self._extract_feature_matrix(df)
        n_total = len(df)
        regime_ids = np.full(n_total, 6, dtype=np.int8)
        probs = np.zeros((n_total, self._selected_n_states), dtype=np.float32)

        if len(X) == 0:
            return regime_ids, probs

        # Get HMM predictions
        hmm_states = self._hmm.predict(X)
        hmm_probs = self._hmm.predict_proba(X)  # (n_valid, n_states)

        # Map HMM states → regime IDs
        valid_len = len(hmm_states)
        offset = n_total - valid_len  # leading NaN bars
        for i in range(valid_len):
            hmm_state = int(hmm_states[i])
            regime_ids[offset + i] = self._state_to_regime.get(hmm_state, 6)
        probs[offset:offset + valid_len] = hmm_probs.astype(np.float32)

        return regime_ids, probs

    def predict_regime_probs(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """Predict regime probability distribution per bar (7 regimes).

        Remaps HMM state probabilities to 7-column regime probability matrix
        via self._state_to_regime mapping.

        Returns
        -------
        np.ndarray of float32, shape (n_bars, 7)
            Columns: [quiet_squeeze, trend_up, trend_down, mean_reverting,
                      breakout, high_volatile, sideways]
        """
        _, hmm_probs = self.predict(df)
        n_bars = hmm_probs.shape[0]

        regime_probs = np.zeros((n_bars, 7), dtype=np.float32)
        for hmm_state, regime_id in self._state_to_regime.items():
            if hmm_state < hmm_probs.shape[1]:
                regime_probs[:, regime_id] += hmm_probs[:, hmm_state]

        # Normalize (handles degraded HMM output)
        row_sums = regime_probs.sum(axis=1, keepdims=True)
        mask = row_sums[:, 0] > 0
        regime_probs[mask] /= row_sums[mask]

        # Bars with no valid HMM output → default to sideways (6)
        no_valid = row_sums[:, 0] == 0
        if no_valid.any():
            regime_probs[no_valid, 6] = 1.0

        return regime_probs

    def predict_hard(self, df: pd.DataFrame) -> np.ndarray:
        """Predict hard regime IDs (0-6) for each bar."""
        regime_ids, _ = self.predict(df)
        return regime_ids

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str):
        import joblib

        joblib.dump(
            {
                "hmm": self._hmm,
                "fitted": self._fitted,
                "state_to_regime": self._state_to_regime,
                "regime_names": self._regime_names,
                "selected_n_states": self._selected_n_states,
                "bic": self._bic,
                "random_state": self.random_state,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "HMMRegimeDetector":
        import joblib

        data = joblib.load(path)
        if isinstance(data, dict):
            det = cls(
                n_states=data.get("selected_n_states", 3),
                random_state=data.get("random_state", 42),
                covariance_type=data.get("covariance_type", "full"),
                n_iter=data.get("n_iter", 100),
            )
            det._hmm = data.get("hmm")
            det._fitted = data.get("fitted", False)
            det._state_to_regime = data.get("state_to_regime", {})
            det._regime_names = data.get("regime_names", [])
            det._selected_n_states = data.get("selected_n_states", 3)
            det._bic = data.get("bic", float("inf"))
            return det
        det = cls()
        det._hmm = data
        det._fitted = True
        return det
