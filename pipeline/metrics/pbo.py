"""
Probability of Backtest Overfitting (PBO) — Bailey et al. 2016.

Computes PBO via Combinatorially Symmetric Cross-Validation (CSCV):
partition the T observations of N configurations into S subsets, form all
C(S, S/2) balanced combinations of subsets (half in-sample, half
out-of-sample), rank every configuration in-sample, and measure where the
best in-sample configuration lands in the out-of-sample distribution.

PBO = P(omega <= 0), where omega = logit of the relative rank of the
best-IS configuration out-of-sample. PBO < 0.10 = unlikely overfit.
PBO > 0.50 = likely overfit.

Reference: D. Bailey, J. Borwein, M. Lopez de Prado, J. Zhu,
"The Probability of Backtest Overfitting", Journal of Computational
Finance, 2016.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def sharpe_ratio(returns: np.ndarray, annual_factor: float = 1.0) -> float:
    """Sharpe ratio of a 1-D returns array (mean/std, no annualisation by default)."""
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return float("nan")
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    if sd < 1e-12:
        return 0.0
    return mu / sd * float(annual_factor)


def _balanced_combinations(S: int, seed: int = 42, max_combos: int = 5000):
    """Yield the balanced subset combinations of S subsets (S/2 in each side).

    For large C(S, S/2), deterministically subsamples ``max_combos`` combos.
    """
    n_is = S // 2
    full = list(itertools.combinations(range(S), n_is))
    total = len(full)
    if total <= max_combos:
        yield from full
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=int(max_combos), replace=False)
    idx = np.sort(idx)
    for i in idx:
        yield full[int(i)]


def compute_pbo(
    performance_matrix: np.ndarray,
    S: int = 16,
    seed: int = 42,
    max_combos: int = 5000,
) -> float:
    """Compute Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    performance_matrix : np.ndarray, shape (N_configs, T)
        Per-period performance (e.g. returns) of each configuration across
        T common periods. Rows = configurations (models/param sets),
        columns = periods. Must be complete (no NaNs).
    S : int
        Number of CSCV subsets (Bailey et al. recommend 8-16, even).
    seed : int
        Seed for deterministic subsampling of combinations when
        C(S, S/2) > max_combos.
    max_combos : int
        Maximum number of balanced combinations to evaluate.

    Returns
    -------
    float
        PBO in [0, 1], or NaN when the input cannot support a valid CSCV
        (fewer than 2 configurations, T < S, or non-finite values).
    """
    mat = np.asarray(performance_matrix, dtype=float)
    if mat.ndim != 2:
        return float("nan")

    N, T = mat.shape
    if N < 2 or T < 4 or not np.isfinite(mat).all():
        return float("nan")

    S = int(S)
    if S < 4 or S > T:
        S = min(T, 16)
    if S < 4:
        return float("nan")
    if S % 2 != 0:
        S -= 1  # must be even for balanced combos
    if S < 4:
        return float("nan")

    # Contiguous partition of the T columns into S subsets.
    chunk = T // S
    bounds = [k * chunk for k in range(S)] + [T]

    omegas: list[float] = []
    for combo in _balanced_combinations(S, seed=seed, max_combos=max_combos):
        is_set = set(combo)
        is_idx, oos_idx = [], []
        for s in range(S):
            lo, hi = bounds[s], bounds[s + 1]
            (is_idx if s in is_set else oos_idx).extend(range(lo, hi))
        if not is_idx or not oos_idx:
            continue

        is_sharpes = np.array([
            sharpe_ratio(mat[c, is_idx]) for c in range(N)
        ], dtype=float)
        oos_sharpes = np.array([
            sharpe_ratio(mat[c, oos_idx]) for c in range(N)
        ], dtype=float)

        finite_is = np.isfinite(is_sharpes)
        if not finite_is.any():
            continue
        best_idx = int(np.nanargmax(np.where(finite_is, is_sharpes, -np.inf)))

        if not np.isfinite(oos_sharpes[best_idx]):
            continue

        # Relative OOS rank of the IS-best config: 1 = worst, N = best.
        oos_rank = int((oos_sharpes < oos_sharpes[best_idx]).sum()) + 1
        u = oos_rank / (N + 1.0)
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        omegas.append(float(math.log(u / (1.0 - u))))

    if not omegas:
        return float("nan")

    omegas = np.asarray(omegas, dtype=float)
    return float(np.mean(omegas <= 0.0))
