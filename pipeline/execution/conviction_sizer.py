"""
Conviction Sizer — continuous sigmoid position sizing from OOS performance.

Replaces the hardcoded 3-tier step function (0.5/1.0/1.5 at thresholds
0.55/0.65/0.80) with a sigmoid curve fitted from out-of-sample trade PnL
binned by committee confidence.

Design per de Prado AFML Ch. 3, 10:
  - Sizing should be continuous, not discrete — boundary cliffs cause
    unstable position sizes near thresholds.
  - Parameters should be learned from OOS data, not guessed.
  - Uses scipy.optimize.curve_fit to find optimal steepness (k) and
    midpoint (c) for the logistic curve.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def logistic(x: np.ndarray, L: float, k: float, c: float) -> np.ndarray:
    """Logistic (sigmoid) function.

    L / (1 + exp(-k * (x - c)))

    Parameters
    ----------
    x : array-like, probability values [0, 1]
    L : maximum multiplier value
    k : steepness (higher = sharper transition at c)
    c : midpoint (probability where multiplier = L/2)
    """
    return L / (1.0 + np.exp(-k * (np.asarray(x, dtype=np.float64) - c)))


class ConvictionSizer:
    """Continuous conviction-based position sizing via sigmoid curve.

    Parameters
    ----------
    L : float
        Maximum multiplier (default 1.5). Clamped to [0.5, 2.0].
    k : float
        Steepness (default 10.0). Higher = sharper transition.
    c : float
        Midpoint probability where multiplier = L/2 (default 0.65).
    fitted : bool
        Whether parameters were learned from OOS data or are defaults.
    """

    def __init__(self, L: float = 1.5, k: float = 10.0, c: float = 0.65):
        self.L = float(L)
        self.k = float(k)
        self.c = float(c)
        self.fitted: bool = False
        self._r2: float = 0.0
        self._n_trades: int = 0

    @property
    def r2(self) -> float:
        """R-squared of the sigmoid fit (0-1, higher is better)."""
        return self._r2

    @property
    def n_trades(self) -> int:
        """Number of OOS trades used for fitting."""
        return self._n_trades

    # ── Fitting ───────────────────────────────────────────────────────

    def fit(
        self,
        bar_predictions: List[Dict],
        n_bins: int = 10,
    ) -> "ConvictionSizer":
        """Fit sigmoid curve from OOS fold predictions.

        Bins trades by committee confidence, computes avg PnL per bin,
        normalizes PnL to [0.5, 2.0] range, fits logistic curve.

        Parameters
        ----------
        bar_predictions : list[dict]
            Per-bar predictions from committee backtester fold_predictions.
            Each bar must have: committee_signal, committee_confidence, next_return.
        n_bins : int
            Number of confidence bins for fitting (default 10).
        """
        # Extract trades: only bars with non-zero signal
        confidences = []
        pnls = []
        for bar in bar_predictions:
            signal = int(bar.get("committee_signal", 0))
            if signal == 0:
                continue
            conf = float(bar.get("committee_confidence", 0.5))
            ret = float(bar.get("next_return", 0.0))
            pnl = ret * signal  # direction * return = PnL
            confidences.append(conf)
            pnls.append(pnl)

        if len(confidences) < 20:
            logger.debug("ConvictionSizer: too few trades (%d), using defaults", len(confidences))
            return self

        confidences = np.array(confidences, dtype=np.float64)
        pnls = np.array(pnls, dtype=np.float64)

        # Bin by confidence and compute avg PnL
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = []
        bin_avg_pnl = []

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = confidences >= bin_edges[i]
            if mask.sum() >= 3:
                bin_centers.append(float(bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_avg_pnl.append(float(np.mean(pnls[mask])))

        if len(bin_centers) < 3:
            logger.debug("ConvictionSizer: too few bins (%d), using defaults", len(bin_centers))
            return self

        bin_centers = np.array(bin_centers, dtype=np.float64)
        bin_avg_pnl = np.array(bin_avg_pnl, dtype=np.float64)

        # Normalize avg PnL to multiplier range [0.5, 2.0]
        pnl_min = float(np.min(bin_avg_pnl))
        pnl_max = float(np.max(bin_avg_pnl))
        if pnl_max - pnl_min < 1e-8:
            # All bins have same PnL — use flat multiplier
            self.L = 1.0
            self.k = 0.0
            self.c = 0.5
            self.fitted = True
            self._n_trades = len(confidences)
            return self

        y_normalized = 0.5 + 1.5 * (bin_avg_pnl - pnl_min) / (pnl_max - pnl_min)

        # Fit sigmoid via scipy
        try:
            from scipy.optimize import curve_fit

            popt, _ = curve_fit(
                logistic,
                bin_centers,
                y_normalized,
                p0=[self.L, self.k, self.c],
                bounds=([0.5, 1.0, 0.50], [2.0, 30.0, 0.85]),
                maxfev=5000,
            )
            self.L = float(popt[0])
            self.k = float(popt[1])
            self.c = float(popt[2])
            self.fitted = True
            self._n_trades = len(confidences)

            # Compute R²
            y_pred = logistic(bin_centers, self.L, self.k, self.c)
            ss_res = float(np.sum((y_normalized - y_pred) ** 2))
            ss_tot = float(np.sum((y_normalized - np.mean(y_normalized)) ** 2))
            self._r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

            logger.info(
                "ConvictionSizer: fitted L=%.3f k=%.3f c=%.3f R^2=%.3f (%d trades)",
                self.L, self.k, self.c, self._r2, len(confidences),
            )
        except Exception as e:
            logger.warning("ConvictionSizer: curve_fit failed (%s), using defaults", e)

        return self

    # ── Prediction ────────────────────────────────────────────────────

    def get_multiplier(self, confidence: float) -> float:
        """Get position size multiplier for a given committee confidence.

        Parameters
        ----------
        confidence : float
            Committee max probability (0.0 to 1.0).

        Returns
        -------
        float
            Position size multiplier in [0.25, 2.0].
            When fitted: sigmoid(confidence, L, k, c).
            When not fitted: hardcoded fallback (0.5/1.0/1.5 tiers).
        """
        if not self.fitted:
            # Fallback: near-identical to original 3-tier for continuity
            if confidence >= 0.80:
                return 1.5
            elif confidence >= 0.65:
                return 1.0
            elif confidence >= 0.55:
                return 0.5
            return 1.0

        if self.k <= 0:
            return self.L

        mult = logistic(np.array([confidence]), self.L, self.k, self.c)[0]
        return float(np.clip(mult, 0.25, 2.0))

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str):
        import json

        with open(path, "w") as f:
            json.dump(
                {
                    "L": self.L,
                    "k": self.k,
                    "c": self.c,
                    "fitted": self.fitted,
                    "r2": self._r2,
                    "n_trades": self._n_trades,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "ConvictionSizer":
        import json

        with open(path) as f:
            data = json.load(f)
        sizer = cls(
            L=data.get("L", 1.5),
            k=data.get("k", 10.0),
            c=data.get("c", 0.65),
        )
        sizer.fitted = data.get("fitted", False)
        sizer._r2 = data.get("r2", 0.0)
        sizer._n_trades = data.get("n_trades", 0)
        return sizer
