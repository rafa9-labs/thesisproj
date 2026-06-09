"""
Deflated Sharpe Ratio — Bailey & Lopez de Prado 2014.

Corrects the Sharpe ratio for: (1) non-normal returns via skewness/kurtosis,
(2) selection bias from running multiple HPO configurations,
(3) sample length limitations.

DSR > 0.95 = Sharpe is genuinely above benchmark (keep).
DSR < 0.90 = flag for review.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def deflated_sharpe_ratio(
    sr_hat: float,
    T: int,
    N_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_star: float = 0.0,
) -> float:
    """Compute Deflated Sharpe Ratio.

    Parameters
    ----------
    sr_hat : float
        Estimated annualised Sharpe ratio.
    T : int
        Number of OOS observations (bars).
    N_trials : int
        Total number of HPO/configurations tried across all models.
    skew : float
        Skewness of the OOS return distribution.
    kurt : float
        Excess kurtosis + 3 of the OOS return distribution.
    sr_star : float
        Benchmark Sharpe ratio (0 = beat cash, >0 for a specific target).

    Returns
    -------
    float
        DSR probability in [0, 1]. Higher = more likely genuinely above benchmark.
    """
    if T < 2:
        return 0.0

    if sr_hat <= sr_star:
        return 0.0

    if N_trials <= 0:
        N_trials = 1

    # Expected maximum SR under N_trials of noise (Bailey & LdP 2014, Eq. 7)
    z_n = norm.ppf(1.0 - 1.0 / N_trials)
    z_ne = norm.ppf(1.0 - 1.0 / (N_trials * np.e))
    E_max_sr = (1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne

    if not np.isfinite(E_max_sr):
        E_max_sr = 2.0  # fallback for single trial

    # Correct SR for non-normality (Bailey & LdP 2014, Eq. 3)
    denom = np.sqrt(max(0.01, 1.0 - skew * sr_hat + (kurt - 3.0) / 4.0 * sr_hat ** 2))
    sr_adj = sr_hat / denom

    # DSR computation (Bailey & LdP 2014, Eq. 2)
    test_stat = (sr_adj - E_max_sr) * np.sqrt(T - 1) / np.sqrt(max(1.0, T - 1))

    if not np.isfinite(test_stat):
        return 0.0

    dsr = float(norm.cdf(test_stat))
    return max(0.0, min(1.0, dsr))
