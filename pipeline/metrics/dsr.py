"""
Deflated Sharpe Ratio — Bailey & Lopez de Prado 2014.

Corrects the Sharpe ratio for:
  (1) non-normal returns via skewness/kurtosis (Eq. 3/6),
  (2) selection bias from running multiple HPO configurations (Eq. 7),
  (3) sample length via the standard error of the SR estimate (Eq. 2/8).

Reference: D. Bailey and M. Lopez de Prado, "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management 40(5), 2014.

Interpretation: DSR is the probability that the estimated Sharpe ratio is
genuinely above the benchmark SR* after correcting for the expected maximum
Sharpe among N_trials of pure noise.

DSR > 0.95 = Sharpe is genuinely above benchmark (keep).
DSR < 0.90 = flag for review.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(N_trials: int) -> float:
    """Expected maximum Sharpe among N_trials noise configurations (Eq. 7).

    Returned in units of the Sharpe ratio at the observation frequency
    (the same frequency as the T observations). Returns 0.0 for
    N_trials <= 1 (no selection bias).
    """
    N_trials = int(N_trials)
    if N_trials <= 1:
        return 0.0
    try:
        z_n = norm.ppf(1.0 - 1.0 / N_trials)
        z_ne = norm.ppf(1.0 - 1.0 / (N_trials * np.e))
    except (ZeroDivisionError, ValueError):
        return 0.0
    e_max = (1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne
    if not np.isfinite(e_max):
        return 0.0
    return float(e_max)


def deflated_sharpe_ratio(
    sr_hat: float,
    T: int,
    N_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_star: float = 0.0,
    periods_per_year: float | None = None,
) -> float:
    """Compute Deflated Sharpe Ratio.

    Parameters
    ----------
    sr_hat : float
        Estimated Sharpe ratio. If ``periods_per_year`` is given, this is the
        ANNUALIZED Sharpe and ``T`` counts observations at that frequency.
        Otherwise ``sr_hat`` must be the Sharpe at the same frequency as ``T``.
    T : int
        Number of OOS observations (bars/periods) at the frequency of ``T``.
    N_trials : int
        Total number of HPO/configurations tried across all models.
    skew : float
        Skewness of the OOS return distribution.
    kurt : float
        RAW kurtosis (excess kurtosis + 3) of the OOS return distribution,
        i.e. scipy.stats.kurtosis(..., fisher=False).
    sr_star : float
        Benchmark Sharpe ratio at the same frequency/scale as ``sr_hat``
        (0 = beat cash, >0 for a specific target).
    periods_per_year : float, optional
        Observations per year of ``T`` (252 daily, 6048 H1, 12096 M30,
        12 monthly). When given, ``sr_hat``/``sr_star`` are treated as
        annualized and converted to observation frequency internally.

    Returns
    -------
    float
        DSR probability in [0, 1]. Higher = more likely genuinely above benchmark.
    """
    try:
        sr_hat = float(sr_hat)
        sr_star = float(sr_star)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(sr_hat) or T < 2:
        return 0.0
    if sr_hat <= sr_star:
        return 0.0

    ppy = float(periods_per_year) if periods_per_year and periods_per_year > 0 else 1.0
    sr_obs = sr_hat / np.sqrt(ppy)
    sr_star_obs = sr_star / np.sqrt(ppy)

    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt) or kurt < 1.0:
        kurt = 3.0

    e_max = expected_max_sharpe(N_trials)

    # Variance of the SR estimate under non-normality (Eq. 3/6),
    # at the observation frequency.
    var_sr = (1.0 - skew * sr_obs + (kurt - 1.0) * sr_obs ** 2 / 4.0) / float(T - 1)
    if var_sr <= 0 or not np.isfinite(var_sr):
        return 0.0
    sigma_sr = float(np.sqrt(var_sr))

    z = (sr_obs - sr_star_obs - e_max * sigma_sr) / sigma_sr
    if not np.isfinite(z):
        return 0.0
    return float(min(1.0, max(0.0, norm.cdf(z))))
