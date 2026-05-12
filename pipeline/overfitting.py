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


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn=None,
    n_boot: int = 2000,
    alpha: float = 0.10,
    seed: int = 42,
) -> Tuple[float, float, float]:
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

    # --- Bootstrap CIs ---
    if len(sharpes_finite) >= 3:
        slo, shi, smean = _bootstrap_ci(sharpes_finite)
        report.sharpe_ci = {"low": slo, "high": shi, "mean": smean}
    else:
        report.sharpe_ci = {"low": None, "high": None, "mean": None}

    if len(returns_finite) >= 3:
        total_ret = lambda x: float(np.sum(x))
        rlo, rhi, rmean = _bootstrap_ci(returns_finite, stat_fn=total_ret)
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
        dlo, dhi, dmean = _bootstrap_ci(returns_finite, stat_fn=maxdd)
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

    # --- CV stability ---
    if len(sharpes_finite) >= 3:
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

    return report


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
        }
        out.append(entry)
    return out
