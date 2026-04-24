"""
Statistical metrics — Sharpe, PSR, DSR, temperature scaling, CV gates.

Extracted from MLBacktesterNoWFO.py lines 805-919.
"""

import numpy as np

def _apply_temperature_to_proba(proba: np.ndarray, T: float) -> np.ndarray:
    T = float(max(1e-3, T))
    logp = np.log(np.clip(proba, 1e-7, 1.0)).astype(np.float64)
    z = logp / T
    z -= z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / np.sum(ez, axis=1, keepdims=True)).astype(np.float32)

def _fit_temperature_from_proba(proba: np.ndarray, y_true: np.ndarray) -> float:
    idx = (np.arange(len(y_true)), y_true.astype(int))
    def nll(p):
        p = np.clip(p[idx], 1e-7, 1.0)
        return float(-np.mean(np.log(p)))
    Ts = np.concatenate([np.linspace(0.5, 3.0, 26),
                        np.linspace(0.3, 0.5, 5),
                        np.linspace(3.0, 4.0, 5)])
    best_T, best = 1.0, nll(proba)
    for T in Ts:
        L = nll(_apply_temperature_to_proba(proba, T))
        if L < best: best_T, best = float(T), float(L)
    return float(best_T)

from typing import Tuple


from math import sqrt
try:
    from scipy.stats import norm
except Exception:
    norm = None

def deflated_sharpe_ratio(sr: float, n_eff: int, sr_max: float = 0.0,
                          skew: float = 0.0, kurt: float = 3.0,
                          n_trials: int = 1) -> float:
    if n_eff is None or n_eff < 2 or not (sr == sr):
        return -1.0
    return (sr - sr_max) * sqrt(max(n_eff, 1))

    
import re
def _cv_status_is_ok(status: str) -> bool:
    """
    Return True iff the fold should count toward the objective.
    We treat only explicit OK folds as objective-eligible.
    """
    try:
        s = str(status or "").strip()
    except Exception:
        return False
    # Most robust: your table prints "🟢 OK" for passing folds.
    if "🟢" in s and "OK" in s:
        return True
    # Fallback: if someone removed emoji but kept the token.
    if re.search(r"\bOK\b", s):
        return True
    return False



def _psr(sr: float, n_eff: int, sr_bench: float = 0.0, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio: P(SR > sr_bench)."""
    if n_eff is None or n_eff < 2 or not (sr == sr):
        return 0.0
    num = (sr - sr_bench) * sqrt(max(n_eff - 1, 1))
    den = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1.0) * (sr ** 2) / 4.0))
    z = num / den
    if norm is None:
        import math
        return 0.5 * (1.0 + math.erf(z / sqrt(2)))
    return float(norm.cdf(z))

def _dsr_sign(sr: float, n_eff: int, sr_max: float = 0.0) -> float:
    """Very small DSR-style sign proxy (positive => likely above sr_max)."""
    if n_eff is None or n_eff < 2 or not (sr == sr):
        return -1.0
    return (sr - sr_max) * sqrt(max(n_eff, 1))

def _cv_reliability_gate(score: float, trades: int, avg_hold_bars: float, params: dict, cfg: dict) -> tuple[bool, str]:
    """
    Returns (ok, reason). ok=False means 'prune with reason'.
    Pulls defaults from CLASS_DEFAULTS (features) but allows per-trial override via params.
    """
    fcfg = cfg["features"]
    gating_mode      = params.get("gating_mode",        fcfg["gating_mode"])
    min_trades_block = int(params.get("min_trades_per_block",  fcfg["min_trades_per_block"]))
    min_indep_bets   = int(params.get("min_independent_bets",  fcfg["min_independent_bets"]))
    psr_alpha        = float(params.get("psr_alpha",           fcfg["psr_alpha"]))
    dsr_prune        = bool(params.get("dsr_prune",            fcfg["dsr_prune"]))
    floor_cv_final   = float(params.get("floor_cv_final",      fcfg["floor_cv_final"]))

    if trades < min_trades_block:
        return (False, f"trades<{min_trades_block} (got {trades})")

    avg_hold = max(1.0, float(avg_hold_bars))
    # crude but stable effective bets proxy
    n_eff = int(max(min_indep_bets, trades / avg_hold))

    if gating_mode == "bets_psr":
        psr = _psr(score, n_eff, sr_bench=0.0)
        if psr < (1.0 - psr_alpha):   # e.g., <0.95 when alpha=0.05
            return (False, f"PSR<{1-psr_alpha:.2f} (psr={psr:.3f}, n_eff={n_eff})")
        if dsr_prune:
            dsr = _dsr_sign(score, n_eff, sr_max=0.0)
            if dsr <= 0.0:
                return (False, f"DSR<=0 (dsr={dsr:.3f}, n_eff={n_eff})")

    if score <= floor_cv_final:
        return (False, f"score {score:.2f} <= floor {floor_cv_final:.2f}")

    return (True, f"ok (n_eff={n_eff})")

# ============================================================
# Metric helpers: standardize shape & invalid/no-trade metrics
# ============================================================

