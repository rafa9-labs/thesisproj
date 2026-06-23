"""
Overfitting detection and reporting engine.

Computes train/OOS divergence, bootstrap confidence intervals,
CV fold stability, and a composite overfitting risk score (0-100)
with green/yellow/red color coding.

Consumed by api.tasks._run_backtest_impl and the diagnostics endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OverfittingReport:
    model: str = ""
    overfit_score: float = 0.0
    risk_level: str = "low"
    risk_color: str = "green"

    train_oos_gap_pct: float = 0.0
    temporal_degradation_pct: float = 0.0

    sharpe_ci: Dict[str, float | None] = field(default_factory=dict)
    return_ci: Dict[str, float | None] = field(default_factory=dict)
    maxdd_ci: Dict[str, float | None] = field(default_factory=dict)

    cv_sharpe_mean: float | None = None
    cv_sharpe_std: float | None = None
    cv_return_mean: float | None = None
    cv_return_std: float | None = None

    min_trl_trades: int = 10
    sufficient_trades: bool = False

    n_periods: int = 0
    n_signal_periods: int = 0

    signal_gap_pct: float = 0.0
    is_mean_sharpe: float | None = None
    oos_mean_sharpe: float | None = None
    dsr_min_sharpe: float | None = None
    psr: float | None = None
    dsr_value: float | None = None
    interaction_effects: List[Dict[str, Any]] | None = None


def _optimal_block_length(values: np.ndarray) -> int:
    """
    Politis & White (2004) automatic block length selection.
    Returns the optimal mean block length for stationary bootstrap.
    Minimum result is 3 for very short series; capped at n//3.
    """
    n = len(values)
    if n < 6:
        return 3
    try:
        acf1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    except (ValueError, IndexError):
        acf1 = 0.0
    if np.isnan(acf1):
        acf1 = 0.0
    acf1 = abs(acf1)
    b_hat = max(3, min(int(n / 3), int(round(n ** (1.0 / 3.0) * (2.0 * acf1 / (1.0 - acf1 ** 2)) ** (2.0 / 3.0)))))
    return max(3, min(b_hat, n // 3))


def _block_bootstrap_ci(
    values: np.ndarray,
    stat_fn=None,
    n_boot: int = 2000,
    alpha: float = 0.10,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Stationary block bootstrap confidence interval for time-series data.
    Preserves within-block serial correlation via randomly-sized contiguous
    blocks drawn from a geometric distribution.

    Parameters
    ----------
    values : np.ndarray
        Time-series values (e.g., monthly Sharpe ratios, returns).
    stat_fn : callable, optional
        Statistic function (default: mean).
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (0.10 = 90% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (low, high, mean) : tuple of float
    """
    if stat_fn is None:
        stat_fn = lambda x: float(np.nanmean(x))

    n = len(values)
    if n < 6:
        # Fall back to i.i.d. for very short series
        return _iid_bootstrap_ci(values, stat_fn, n_boot, alpha, seed)

    mean_block = _optimal_block_length(values)
    rng = np.random.default_rng(seed)

    stats = []
    for _ in range(n_boot):
        sample = _stationary_block_sample(values, n, mean_block, rng)
        stats.append(stat_fn(sample))

    stats = np.asarray(stats, dtype=float)
    stats = stats[np.isfinite(stats)]
    if len(stats) < 10:
        m = stat_fn(values)
        return (m, m, m)

    lo = float(np.percentile(stats, alpha / 2 * 100))
    hi = float(np.percentile(stats, (1 - alpha / 2) * 100))
    mean_val = float(np.mean(stats))
    return (lo, hi, mean_val)


def _stationary_block_sample(values: np.ndarray, n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a stationary bootstrap sample of length n.
    Blocks are drawn from a geometric distribution p = 1/mean_block.
    """
    p = 1.0 / max(mean_block, 1)
    sample = []
    while len(sample) < n:
        start = rng.integers(0, len(values))
        length = max(1, int(rng.geometric(p)))
        for k in range(length):
            if len(sample) >= n:
                break
            idx = (start + k) % len(values)
            sample.append(values[idx])
    return np.array(sample, dtype=float)


def _iid_bootstrap_ci(
    values: np.ndarray,
    stat_fn=None,
    n_boot: int = 2000,
    alpha: float = 0.10,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """I.i.d. bootstrap fallback for very short series (< 6 periods)."""
    if stat_fn is None:
        stat_fn = lambda x: float(np.nanmean(x))
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 3:
        m = stat_fn(values)
        return (m, m, m)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        stats.append(stat_fn(sample))
    stats = np.asarray(stats, dtype=float)
    stats = stats[np.isfinite(stats)]
    if len(stats) < 10:
        m = stat_fn(values)
        return (m, m, m)
    lo = float(np.percentile(stats, alpha / 2 * 100))
    hi = float(np.percentile(stats, (1 - alpha / 2) * 100))
    mean_val = float(np.mean(stats))
    return (lo, hi, mean_val)


def _temporal_degradation(monthly_sharpes: np.ndarray) -> float:
    """
    Compare first half vs second half Sharpe to detect performance decay.
    Returns degradation percentage (positive = degrading).
    """
    n = len(monthly_sharpes)
    if n < 6:
        return 0.0
    mid = n // 2
    first_half = monthly_sharpes[:mid]
    second_half = monthly_sharpes[mid:]
    fh = float(np.nanmean(first_half))
    sh = float(np.nanmean(second_half))
    denom = max(abs(fh), 0.01)
    return (fh - sh) / denom * 100.0


def _overfit_composite_score(
    gap_pct: float,
    degradation_pct: float,
    cv_std: float,
    ci_width_ratio: float,
) -> float:
    score = 0.0
    score += 0.30 * min(max(gap_pct, 0.0) / 60.0, 1.0)
    score += 0.25 * min(max(degradation_pct, 0.0) / 50.0, 1.0)
    score += 0.25 * min(cv_std / 0.40, 1.0)
    score += 0.20 * min(ci_width_ratio / 1.5, 1.0)
    score *= 100.0
    return float(min(score, 100.0))


def _classify_risk(score: float) -> Tuple[str, str]:
    if score <= 25:
        return ("low", "green")
    elif score <= 55:
        return ("medium", "yellow")
    return ("high", "red")


def compute_overfitting_report(
    monthly_records: List[dict],
    model_type: str = "",
    hpo_best_value: Optional[float] = None,
    n_hpo_trials: Optional[int] = None,
) -> OverfittingReport:
    """
    Compute a comprehensive overfitting report from walk-forward monthly records.

    Parameters
    ----------
    monthly_records : list[dict]
        Per-month records with keys: 'sharpe', 'strategy_return', 'trades',
        'test_start', 'test_end'. May also include 'train_sharpe'.
    model_type : str
        Model type identifier.
    hpo_best_value : float, optional
        Best HPO trial value used as in-sample proxy when train_sharpe is
        not available.
    n_hpo_trials : int, optional
        Total number of HPO trials run. Used to compute DSR-adjusted minimum Sharpe
        threshold for statistical significance.

    Returns
    -------
    OverfittingReport
    """
    report = OverfittingReport(model=model_type)

    if not monthly_records:
        return report

    sharpes_arr = np.array([r.get("sharpe", np.nan) for r in monthly_records], dtype=float)
    returns_arr = np.array([r.get("strategy_return", np.nan) for r in monthly_records], dtype=float)
    trades_arr = np.array([r.get("trades", 0) or 0 for r in monthly_records], dtype=int)

    sharpes_finite = sharpes_arr[np.isfinite(sharpes_arr)]
    returns_finite = returns_arr[np.isfinite(returns_arr)]

    report.n_periods = len(monthly_records)
    report.n_signal_periods = int(np.sum(trades_arr > 0))
    report.sufficient_trades = bool(np.sum(trades_arr) >= report.min_trl_trades)

    # --- Block bootstrap CIs (time-series aware) ---
    if len(sharpes_finite) >= 3:
        slo, shi, smean = _block_bootstrap_ci(sharpes_finite)
        report.sharpe_ci = {"low": slo, "high": shi, "mean": smean}
    else:
        report.sharpe_ci = {"low": None, "high": None, "mean": None}

    if len(returns_finite) >= 3:
        total_ret = lambda x: float(np.sum(x))
        rlo, rhi, rmean = _block_bootstrap_ci(returns_finite, stat_fn=total_ret)
        report.return_ci = {"low": rlo, "high": rhi, "mean": rmean}
    else:
        report.return_ci = {"low": None, "high": None, "mean": None}

    if len(sharpes_finite) >= 3:
        def maxdd_stat(x: np.ndarray, n_mc: int = 1000):
            rng = np.random.default_rng(42)
            n = len(x)
            if n < 3:
                return 0.0
            dds = []
            for _ in range(min(n_mc, 500)):
                path = np.cumsum(rng.choice(x, size=n, replace=True))
                if len(path) == 0:
                    continue
                cmax = np.maximum.accumulate(path)
                dd = (path - cmax).min()
                dds.append(float(dd))
            return float(np.mean(dds)) if dds else 0.0

        maxdd = lambda x: maxdd_stat(x)
        dlo, dhi, dmean = _block_bootstrap_ci(returns_finite, stat_fn=maxdd)
        report.maxdd_ci = {"low": dlo, "high": dhi, "mean": dmean}
    else:
        report.maxdd_ci = {"low": None, "high": None, "mean": None}

    # --- Train/OOS gap ---
    train_sharpes = np.array([
        r.get("train_sharpe", np.nan) for r in monthly_records
    ], dtype=float)
    train_sharpes_finite = train_sharpes[np.isfinite(train_sharpes)]

    if len(train_sharpes_finite) >= 3 and len(sharpes_finite) >= 3:
        ts_mean = float(np.nanmean(train_sharpes_finite))
        oos_mean = float(np.nanmean(sharpes_finite))
        gap = (ts_mean - oos_mean) / max(abs(ts_mean), 0.01) * 100.0
        report.train_oos_gap_pct = gap
        report.is_mean_sharpe = ts_mean
        report.oos_mean_sharpe = oos_mean
    elif hpo_best_value is not None and np.isfinite(hpo_best_value) and len(sharpes_finite) >= 3:
        oos_mean = float(np.nanmean(sharpes_finite))
        gap = (hpo_best_value - oos_mean) / max(abs(hpo_best_value), 0.01) * 100.0
        report.train_oos_gap_pct = gap
        report.is_mean_sharpe = hpo_best_value
        report.oos_mean_sharpe = oos_mean
    else:
        report.train_oos_gap_pct = 0.0

    # --- Temporal degradation ---
    report.temporal_degradation_pct = _temporal_degradation(sharpes_arr)

    # --- CV stability (prefer IS fold Sharpes from CV when available) ---
    cv_fold_all = []
    for r in monthly_records:
        cf = r.get("cv_fold_sharpes")
        if isinstance(cf, (list, np.ndarray)) and len(cf) > 0:
            cv_fold_all.extend([float(v) for v in cf if np.isfinite(float(v))])
    cv_fold_arr = np.array(cv_fold_all, dtype=float) if cv_fold_all else np.array([])

    if len(cv_fold_arr) >= 3:
        report.cv_sharpe_mean = float(np.nanmean(cv_fold_arr))
        report.cv_sharpe_std = float(np.nanstd(cv_fold_arr, ddof=1))
    elif len(sharpes_finite) >= 3:
        # Fallback: use OOS monthly Sharpes
        report.cv_sharpe_mean = float(np.nanmean(sharpes_finite))
        report.cv_sharpe_std = float(np.nanstd(sharpes_finite, ddof=1))
    if len(returns_finite) >= 3:
        report.cv_return_mean = float(np.nanmean(returns_finite))
        report.cv_return_std = float(np.nanstd(returns_finite, ddof=1))

    # --- Signal gap ---
    if report.n_periods > 0:
        report.signal_gap_pct = (
            (report.n_periods - report.n_signal_periods) / report.n_periods * 100.0
        )

    # --- Composite score ---
    ci_width = 0.0
    if report.sharpe_ci.get("high") is not None and report.sharpe_ci.get("low") is not None:
        mean_sharpe = report.sharpe_ci.get("mean", 0.01)
        ci_width = (report.sharpe_ci["high"] - report.sharpe_ci["low"]) / max(abs(mean_sharpe or 0.01), 0.01)

    report.overfit_score = _overfit_composite_score(
        gap_pct=report.train_oos_gap_pct,
        degradation_pct=report.temporal_degradation_pct,
        cv_std=report.cv_sharpe_std or 0.0,
        ci_width_ratio=ci_width,
    )
    report.risk_level, report.risk_color = _classify_risk(report.overfit_score)

    # --- DSR minimum Sharpe threshold for statistical significance ---
    if n_hpo_trials is not None and len(sharpes_finite) >= 6:
        report.dsr_min_sharpe = _compute_dsr_min_sharpe(
            n_hpo_trials, len(sharpes_finite), len(trades_arr)
        )

    # --- PSR / DSR (Probabilistic & Deflated Sharpe Ratios) ---
    total_trades = int(np.sum(trades_arr))
    if len(sharpes_finite) >= 3 and len(returns_finite) >= 3:
        oos_sr = report.oos_mean_sharpe if report.oos_mean_sharpe is not None else float(np.nanmean(sharpes_finite))
        try:
            from pipeline.metrics import _psr
            from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurt
            _skew = float(_scipy_skew(returns_finite, bias=False)) if len(returns_finite) >= 4 else 0.0
            _kurt = float(_scipy_kurt(returns_finite, bias=False, fisher=False)) if len(returns_finite) >= 4 else 3.0
            n_eff = max(total_trades, len(returns_finite))
            report.psr = round(float(_psr(oos_sr, n_eff, sr_bench=0.0, skew=_skew, kurt=_kurt)), 4)
        except Exception:
            report.psr = None

        try:
            from pipeline.dsr import deflated_sharpe_ratio
            n_trials_use = n_hpo_trials if n_hpo_trials is not None and n_hpo_trials > 0 else 1
            report.dsr_value = round(float(deflated_sharpe_ratio(
                oos_sr, T=max(total_trades, len(returns_finite)),
                N_trials=n_trials_use, skew=_skew, kurt=_kurt, sr_star=0.0
            )), 4)
        except Exception:
            report.dsr_value = None

    return report


def _compute_dsr_min_sharpe(n_hpo_trials: int, n_oos_periods: int, n_total_trades: int) -> float:
    """Minimum annualized Sharpe for statistical significance after multiple testing.

    Based on the Deflated Sharpe Ratio framework (Lopez de Prado, Bailey et al. 2014).
    Accounts for:
    - Number of HPO trials (multiple testing correction via Sidak)
    - Number of OOS periods (estimation error)
    - Minimum trades (reliability floor)

    Returns a float: the minimum Sharpe ratio a strategy must achieve to be
    considered statistically significant at the 95% confidence level.
    """
    from math import sqrt
    from scipy.stats import norm as _norm

    m = max(n_hpo_trials, 1)
    n = max(n_oos_periods, 6)

    # Sidak correction for multiple testing
    adj_alpha = 1.0 - (1.0 - 0.05) ** (1.0 / m)
    z_adj = _norm.ppf(1.0 - adj_alpha / 2.0)

    # Minimum Sharpe for the given number of OOS observations
    min_sr = z_adj / sqrt(n)

    # Reliability penalty for very few trades
    if n_total_trades < 10:
        min_sr *= 1.5
    elif n_total_trades < 30:
        min_sr *= 1.2

    return round(float(min_sr), 3)


def compute_period_breakdown(monthly_records: List[dict]) -> List[dict]:
    """
    Build per-period transparency records suitable for API return.
    Expands minimal monthly records into the Walk-Forward Panel schema.

    Parameters
    ----------
    monthly_records : list[dict]
        Per-month records from _wfo_monthly_records.

    Returns
    -------
    list[dict]
        List with keys: period_start, period_end, train_start, train_end,
        test_sharpe, train_sharpe, strategy_return, bh_return, trades,
        signals_raw, signals_passed_gate, pct_sideways, pct_trend, pct_volatile.
    """
    out = []
    for r in monthly_records:
        entry = {
            "period_start": str(r.get("test_start", "")),
            "period_end": str(r.get("test_end", "")),
            "train_start": str(r.get("train_start", "")) if r.get("train_start") is not None and str(r.get("train_start")) != "nan" else None,
            "train_end": str(r.get("train_end", "")) if r.get("train_end") is not None and str(r.get("train_end")) != "nan" else None,
            "test_sharpe": float(r.get("sharpe", float("nan"))),
            "train_sharpe": float(r.get("train_sharpe", float("nan"))),
            "strategy_return": float(r.get("strategy_return", float("nan"))),
            "bh_return": float(r.get("bh_return", float("nan"))),
            "trades": int(r.get("trades", 0) or 0),
            "signals_raw": int(r.get("signals_raw", 0) or 0),
            "signals_passed_gate": int(r.get("signals_passed_gate", 0) or 0),
            "pct_sideways": float(r.get("pct_sideways", float("nan"))),
            "pct_trend": float(r.get("pct_trend", float("nan"))),
            "pct_volatile": float(r.get("pct_volatile", float("nan"))),
            "sharpe_gap_pct": float(r.get("sharpe_gap_pct", float("nan"))),
            "return_gap_pct": float(r.get("return_gap_pct", float("nan"))),
            "cv_fold_sharpes": r.get("cv_fold_sharpes", []),
        }
        out.append(entry)
    return out


def compute_fanova_interactions(study) -> List[Dict[str, Any]]:
    """Compute fANOVA interaction effects from an Optuna study.

    Uses Optuna's built-in FanovaImportanceEvaluator to decompose
    objective variance into main effects and pairwise interaction effects.

    Returns a list of interaction entries sorted by importance descending:
        [{param1: "lr", param2: "dropout", interaction_pct: 23.0}, ...]
    """
    result: List[Dict[str, Any]] = []
    try:
        from optuna.importance import FanovaImportanceEvaluator
        evaluator = FanovaImportanceEvaluator()
        importance = evaluator.evaluate(study)

        for param_name, fanova_data in importance.items():
            main_effect = float(getattr(fanova_data, "individual_importance", 0) or 0)
            total_effect = float(getattr(fanova_data, "total_importance", 0) or 0)
            interaction_pct = round((total_effect - main_effect) * 100, 1)

            if interaction_pct > 0.5:
                result.append({
                    "param": param_name,
                    "main_pct": round(main_effect * 100, 1),
                    "interaction_pct": interaction_pct,
                    "total_pct": round(total_effect * 100, 1),
                })

        result.sort(key=lambda x: x["interaction_pct"], reverse=True)
    except Exception:
        pass

    return result[:10]
