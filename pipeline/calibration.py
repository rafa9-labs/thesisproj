"""Model calibration and uncertainty quantification.

Phase 4.3 -- probability calibration, temperature scaling, conformal prediction.
Backported from utilsNoWFO.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

try:
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    CalibratedClassifierCV = None


class RollingStandardizer:
    """
    Rolling standardizer fit on train (no leakage).
    - fit_transform(X_train): scales train with rolling mean/std (per-column)
    - transform(X_test): scales test using the *last* (mean,std) from train
    """
    window: int = 200
    min_periods: int = 50

    def fit_transform(self, X_train: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        # Single copy to avoid mutating caller's DataFrame; operate in-place on it
        X = X_train.astype(float).copy()

        mu_roll = X.rolling(self.window, min_periods=self.min_periods).mean()
        sd_roll = (
            X.rolling(self.window, min_periods=self.min_periods)
              .std()
              .replace(0, np.nan)
        )

        # Only keep the last rolling stats for test-time scaling
        stats = {"mu_last": mu_roll.iloc[-1], "sd_last": sd_roll.iloc[-1]}

        # In-place standardization on X to avoid allocating X_scaled
        X -= mu_roll
        X /= sd_roll

        return X, stats

    def transform(self, X_test: pd.DataFrame, stats: dict) -> pd.DataFrame:
        mu_last = stats["mu_last"]
        sd_last = stats["sd_last"].replace(0, np.nan)
        return (X_test - mu_last) / sd_last


def calibrate_prefit_and_predict_proba(
    base_estimator, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, method: str = "isotonic"
) -> Tuple[np.ndarray, Optional[object]]:
    """Calibrate probabilities without using the deprecated cv='prefit' path."""
    if CalibratedClassifierCV is None:
        return base_estimator.predict_proba(X_pred), None
    try:
        from sklearn.base import clone
        # Fresh, unfitted copy -- we calibrate with k-fold CV on the train window
        est = clone(base_estimator)
        calibrator = CalibratedClassifierCV(estimator=est, method=method, cv=3)
        calibrator.fit(X_train, y_train)
        proba = calibrator.predict_proba(X_pred)
        return proba, calibrator
    except Exception:
        # Last-resort fallback: use the prefit estimator's own proba
        return base_estimator.predict_proba(X_pred), None


class ConformalClassifier:
    """
    Split-conformal for multiclass:
    - Nonconformity score: 1 - p_true
    - qhat: (ceil((n+1)*(1-alpha))/n)-quantile of calibration scores
    - Predict set S(x) = {k : 1 - p_k <= qhat}
    If |S(x)|==1 we call it 'decisive' and take that class; else abstain.
    """
    alpha: float = 0.1
    qhat_: Optional[float] = None

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        idx = (np.arange(len(y_cal)), y_cal.astype(int))
        nc = 1.0 - proba_cal[idx]
        n = len(nc)
        k = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        k = np.clip(k, 0, n - 1)
        self.qhat_ = float(np.partition(nc, k)[k])
        return self

    def predict_decisions(self, proba_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.qhat_ is not None, "Fit conformal first."
        top_idx = proba_new.argmax(axis=1)
        top_nc = 1.0 - proba_new[np.arange(len(top_idx)), top_idx]
        decisive = top_nc <= self.qhat_
        return decisive, top_idx


def fit_coverage_threshold_on_calibration(proba_cal: np.ndarray,
                                           target_active_rate: float) -> float:
    """
    Given calibrated 3-class probabilities on a calibration slice,
    return the confidence threshold that achieves the requested active rate.
    """
    import numpy as np
    p = np.asarray(proba_cal, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("proba_cal must be (n,3) or (n,2) after sanitize_proba()")
    max_conf = p.max(axis=1)
    
    # Coverage must reflect **trade intent** (short/long strength), not certainty about "flat".
    # Otherwise a dominant flat class makes coverage targeting ineffective.
    if p.shape[1] >= 3:
        max_conf = np.maximum(p[:, 0], p[:, 2])
    else:
        max_conf = p.max(axis=1)
    
    target_active_rate = float(np.clip(target_active_rate, 1e-6, 0.999999))
    # keep top-K% = choose (1 - rate) quantile as threshold
    q = 1.0 - target_active_rate
    thr = float(np.quantile(max_conf, q)) if len(max_conf) else 0.50
    return float(np.clip(thr, 0.0, 1.0))


def apply_temperature_to_proba(proba: np.ndarray, T: float) -> np.ndarray:
    """Softmax temperature scaling in log-prob space (stable & model-agnostic)."""
    T = float(max(1e-3, T))
    logp = np.log(np.clip(proba, 1e-7, 1.0)).astype(np.float64)
    z = logp / T
    z -= z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / np.sum(ez, axis=1, keepdims=True)).astype(np.float32)


def fit_temperature_from_proba(proba: np.ndarray, y_true: np.ndarray) -> float:
    """1-D grid search for T minimizing NLL on a calibration slice."""
    idx = np.arange(len(y_true))
    def nll(p):
        p = np.clip(p[idx, y_true], 1e-7, 1.0)
        return float(-np.mean(np.log(p)))
    Ts = np.concatenate([np.linspace(0.5, 3.0, 26),
                         np.linspace(0.3, 0.5, 5),
                         np.linspace(3.0, 4.0, 5)])
    best_T, best_loss = 1.0, nll(proba)
    for T in Ts:
        L = nll(apply_temperature_to_proba(proba, T))
        if L < best_loss:
            best_T, best_loss = float(T), float(L)
    return float(best_T)


def sanitize_proba(proba):
    """
    Clean and row-normalize predict_proba output:
    - cast to float
    - replace NaN/Inf with 0
    - row-normalize; if a row sums to 0, assign a uniform distribution
    """
    import numpy as np
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        return proba

    # Replace bad values
    bad = ~np.isfinite(proba)
    if bad.any():
        proba[bad] = 0.0

    # Row-normalize
    rowsum = proba.sum(axis=1, keepdims=True)  # shape (n, 1)
    zero_row_idx = np.flatnonzero(rowsum.ravel() == 0.0)  # shape (k,)
    if zero_row_idx.size:
        proba[zero_row_idx, :] = 1.0 / proba.shape[1]
        rowsum = proba.sum(axis=1, keepdims=True)

    proba = proba / rowsum
    # Clamp to avoid exact 0/1 which can blow up log-loss and downstream ratios
    eps = 1e-6
    proba = np.clip(proba, eps, 1.0 - eps)
    # Re-normalize to kill tiny drift from clipping
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba

