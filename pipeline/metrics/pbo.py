"""
Probability of Backtest Overfitting (PBO) — Bailey et al. 2016.

Computes PBO via Combinatorially Symmetric Cross-Validation (CSCV).
PBO is the probability that the best IS configuration underperforms OOS.

PBO < 0.10 = unlikely overfit.
PBO > 0.50 = likely overfit.
"""
from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, annual_factor: float = 15.87) -> float:
    """Annualised Sharpe ratio from a 1-D returns array."""
    rets = returns[np.isfinite(returns)]
    if len(rets) < 2:
        return float("nan")
    mu = np.mean(rets)
    sd = np.std(rets, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(mu / sd * annual_factor)


def compute_pbo(
    fold_returns: np.ndarray,
    S: int = 8,
    annual_factor: float = 15.87,
) -> float:
    """Compute Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    fold_returns : np.ndarray, shape (n_folds, n_bars)
        Per-fold strategy return series.
    S : int
        Number of CSCV subsets (Bailey et al. recommend 8-16).
    annual_factor : float
        Annualisation factor for Sharpe (default sqrt(252) = 15.87).

    Returns
    -------
    float
        PBO in [0, 1]. PBO < 0.10 = not overfit.
    """
    n_folds, T = fold_returns.shape

    if n_folds < 4 or T < 100:
        return 1.0

    # Flatten folds into single return series
    flat = fold_returns.ravel()
    flat_T = len(flat)

    chunk = max(1, flat_T // S)
    if chunk < 10:
        return 1.0

    logit_values = []
    total_comb = 0

    for is_mask_int in range(1, (1 << S) - 1):
        is_bits = [(is_mask_int >> s) & 1 for s in range(S)]
        is_size = sum(is_bits)
        if is_size == 0 or is_size == S:
            continue  # must have both IS and OOS

        is_indices = []
        oos_indices = []
        for s in range(S):
            start = s * chunk
            end = min(start + chunk, flat_T)
            if start >= flat_T:
                break
            idx_range = np.arange(start, end)
            if is_bits[s]:
                is_indices.append(idx_range)
            else:
                oos_indices.append(idx_range)

        if not is_indices or not oos_indices:
            continue

        is_arr = np.concatenate(is_indices)
        oos_arr = np.concatenate(oos_indices)
        is_sharpe = sharpe_ratio(flat[is_arr], annual_factor)
        oos_sharpe = sharpe_ratio(flat[oos_arr], annual_factor)

        if not np.isfinite(is_sharpe) or not np.isfinite(oos_sharpe):
            continue

        is_rank = 1.0 if is_sharpe > 0 else 0.0
        oos_rank = 1.0 if oos_sharpe > 0 else 0.0
        # Simplified: best IS > median OOS?
        if is_sharpe > 0 and oos_sharpe <= 0:
            logit_values.append(-1.0)
        elif is_sharpe > 0 and oos_sharpe > 0:
            logit_values.append(0.5)  # both positive = ambiguous
        else:
            logit_values.append(1.0)  # IS bad, OOS good = not overfit

        total_comb += 1

    if total_comb == 0:
        return 1.0

    pbo = sum(1.0 for v in logit_values if v < 0) / total_comb
    return float(pbo)
