"""Additional metrics backported from utilsNoWFO.py.

Phase 4.4 -- PSR, DSR, Brier/NLL, drawdown curves, rolling stats.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional

try:
    from scipy.stats import norm as _norm
except ImportError:
    _norm = None

from pipeline.metrics.metrics_eval import estimate_frequency_per_year


def _norm_cdf(z: float) -> float:
    if _norm is not None:
        return float(_norm.cdf(z))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _compute_drawdown(equity):
    if equity is None or len(equity) == 0:
        return None
    s = pd.Series(equity) if not isinstance(equity, pd.Series) else equity
    cummax = s.cummax()
    dd = (s - cummax) / cummax.replace(0, np.nan)
    return dd


def compute_brier_and_nll(proba, y_true):
    """
    Multi-class Brier score and negative log-likelihood (NLL).

    Parameters
    ----------
    proba : array-like, shape (n_samples, n_classes)
        Class probabilities per sample.
    y_true : array-like, shape (n_samples,)
        Integer class labels in {0, 1, 2} for 3-class problems.

    Returns
    -------
    brier : float
        Mean multi-class Brier score.
    nll : float
        Negative log-likelihood (cross-entropy).
    """
    import numpy as np
    from sklearn.metrics import log_loss

    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    if proba.ndim != 2 or proba.shape[0] == 0:
        return float("nan"), float("nan")

    # drop rows with non-finite probabilities
    mask = np.isfinite(proba).all(axis=1)
    proba = proba[mask]
    y_true = y_true[mask]

    if proba.shape[0] == 0:
        return float("nan"), float("nan")

    n, k = proba.shape
    # one-hot encode labels
    one_hot = np.eye(k, dtype=float)[y_true]
    brier = float(np.mean((proba - one_hot) ** 2))

    try:
        nll = float(log_loss(y_true, proba, labels=list(range(k))))
    except Exception:
        nll = float("nan")

    return brier, nll


    def _cliffs_delta(a, b):
        if a.size == 0 or b.size == 0:
            return float("nan")
        # rank-biserial: ( #(b>a) - #(a>b) ) / (len(a)*len(b))
        count = 0
        for x in a:
            count += int((b > x).sum()) - int((b < x).sum())
        return float(count) / float(a.size * b.size)


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, periods_per_year=12):
    """
    Bailey & Lopez de Prado (2012) style PSR (simplified, iid assumption).
    Returns probability that SR > sr_benchmark.
    """
    import numpy as np
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 10:
        return float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd == 0:
        return float("nan")
    sr_hat = (mu / sd) * np.sqrt(periods_per_year)
    z = (sr_hat - float(sr_benchmark)) * np.sqrt(n)
    return float(_norm_cdf(z))


def compute_dsr_scores(scores):
    """
    Deflated 'Sharpe' proxy over an array of trial scores.
    Returns a list of DSR-like probabilities (higher is better), one per score.
    Approximates multiple-testing via a Sidak-style family correction.
    """
    x = np.asarray(scores, dtype=float)
    n = int(np.isfinite(x).sum())
    if n <= 1 or np.allclose(np.nanstd(x, ddof=1), 0.0):
        return [0.0 if not np.isfinite(v) else 0.5 for v in x]

    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1))
    out = []
    for v in x:
        if not np.isfinite(v):
            out.append(0.0); continue
        z = (float(v) - mu) / sd
        p_single = 1.0 - _norm_cdf(z)
        p_family = 1.0 - (1.0 - max(1e-12, min(1.0, p_single))) ** n
        dsr = 1.0 - p_family
        out.append(float(dsr))
    return out


def compute_rolling_hit_rate(df, window_bars: int, min_active: int = 1):
    """
    Rolling hit-rate using 1-bar execution delay:
      correct_t = sign(pred_{t-1} * return_t) > 0, ignoring abstentions (pred_{t-1} == 0).
    Returns a Series aligned to df.index with NaN where active<min_active in window.
    Requires df[['pred','returns']].
    """
    import numpy as np, pandas as pd
    if df is None or "pred" not in df or "returns" not in df:
        return pd.Series(index=(df.index if df is not None else None), dtype=float)
    pred_prev = df["pred"].shift(1)
    active = (pred_prev != 0).astype(int)
    correct = ((pred_prev * df["returns"]) > 0).astype(float).where(active == 1)
    act_roll = active.rolling(int(window_bars), min_periods=1).sum()
    good_roll = correct.rolling(int(window_bars), min_periods=1).sum()
    hit = good_roll / act_roll.replace(0, np.nan)
    hit[act_roll < int(min_active)] = np.nan
    return hit


def compute_rolling_sharpe_series(
    equity: "pd.Series",
    window: int,
    frequency_per_year: float | None = None,
) -> "pd.Series":
    """
    Compute an annualised rolling Sharpe ratio from an equity curve (x).

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity (x), strictly positive.
    window : int
        Rolling window length in bars.
    frequency_per_year : float or None
        Bars-per-year for annualisation. If None, inferred from index via
        estimate_frequency_per_year.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio (annualised).
    """
    import numpy as np, pandas as pd

    if equity is None or len(equity) == 0 or window <= 1:
        return pd.Series(dtype=float)

    s = pd.Series(pd.to_numeric(equity, errors="coerce"), index=pd.to_datetime(equity.index))
    s = s.replace([np.inf, -np.inf], np.nan).ffill()

    # Log returns are numerically more stable for cumulative equity
    r = np.log(s).diff()
    r = r.replace([np.inf, -np.inf], np.nan)

    if frequency_per_year is None:
        try:
            frequency_per_year = float(estimate_frequency_per_year(r.index))
        except Exception:
            frequency_per_year = 252.0

    freq = max(1.0, float(frequency_per_year))

    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std(ddof=0)

    # Avoid division by zero
    sigma = sigma.replace(0.0, np.nan)
    sharpe = (mu / sigma) * np.sqrt(freq)

    return sharpe


def compute_drawdown_curve(equity: "pd.Series") -> "pd.Series":
    """
    Public helper: convert an equity curve (x) into a drawdown curve in percent.

    Uses the internal _compute_drawdown (fractional drawdown, negative values)
    and scales by 100.
    """
    import pandas as pd
    dd_frac = _compute_drawdown(equity)
    if dd_frac is None or len(dd_frac) == 0:
        return pd.Series(dtype=float)
    return dd_frac * 100.0

