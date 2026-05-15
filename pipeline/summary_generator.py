"""Plain-English backtest summary generator.

Template-based (no LLM dependency). Produces a 3-4 sentence
paragraph describing a completed backtest in language a trader
can understand without reading a statistics manual.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional

FRIENDLY_MODEL_NAMES: Dict[str, str] = {
    "logistic": "Logistic Regression",
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "xgboost": "XGBoost",
    "cnn": "CNN",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "dqn": "Dueling DQN",
    "ensemble_adaptive_regime": "Adaptive Regime",
    "ensemble_cnn_lstm_xgboost": "CNN+LSTM+Boost Ensemble",
}


def generate_summary(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a plain-English summary of a completed backtest.

    Parameters
    ----------
    metrics : dict
        The per-model metrics dict from api/tasks.py (metrics_row).
    config : dict, optional
        The original backtest config (pair, timeframe, dates, months).

    Returns
    -------
    str
        A 3-4 sentence summary paragraph.
    """
    config = config or {}
    model_key = str(metrics.get("model", ""))
    model_name = FRIENDLY_MODEL_NAMES.get(model_key, model_key or "Unknown")
    pair = str(config.get("pair", "EURUSD"))
    timeframe = str(config.get("timeframe", "H1"))
    start = _fmt_date(config.get("start_date", ""))
    end = _fmt_date(config.get("end_date", ""))

    sharpe = _safe_float(metrics.get("sharpe"))
    win_rate = _safe_float(metrics.get("win_rate"))
    max_dd = _safe_float(metrics.get("max_drawdown"))
    total_ret = _safe_float(metrics.get("total_return_pct"))
    trades = _safe_int(metrics.get("total_trades"))

    overfit = metrics.get("overfitting") or {}
    risk_level = str(overfit.get("risk_level", "low"))
    overfit_score = _safe_float(overfit.get("overfit_score"))
    is_sharpe = _safe_float(overfit.get("is_mean_sharpe"))
    oos_sharpe = _safe_float(overfit.get("oos_mean_sharpe"))
    gap_pct = _safe_float(overfit.get("train_oos_gap_pct"))

    periods = metrics.get("walkforward_periods") or []

    sentences = []

    # --- Sentence 1: What was tested and the headline numbers ---
    sentences.append(_sentence_intro(model_name, pair, timeframe, start, end, sharpe, win_rate, max_dd, total_ret, trades))

    # --- Sentence 2: Overfitting assessment ---
    sentences.append(_sentence_overfitting(risk_level, overfit_score, gap_pct, is_sharpe, oos_sharpe, trades))

    # --- Sentence 3: Regime / best period ---
    regime_sentence = _sentence_regimes(periods)
    if regime_sentence:
        sentences.append(regime_sentence)

    return " ".join(sentences).strip()


def _sentence_intro(model_name, pair, tf, start, end, sharpe, win_rate, max_dd, total_ret, trades):
    sharpe_desc = _describe_sharpe(sharpe)
    wr_str = f"with {win_rate:.0%} win rate" if np.isfinite(win_rate) else ""
    dd_str = f"and {max_dd:.1%} max drawdown" if np.isfinite(max_dd) else ""
    ret_str = f"returning {total_ret:+.1%} net" if np.isfinite(total_ret) else ""
    trades_str = f"across {trades} trades" if trades > 0 else "with no trades"

    parts = [
        f"{model_name} on {pair} {tf}",
        f"from {start} to {end}" if start else "",
        f"achieved a {sharpe_desc} of {sharpe:.2f}",
        wr_str,
        dd_str,
        ret_str,
        trades_str + ".",
    ]
    return " ".join(p for p in parts if p)


def _sentence_overfitting(risk_level, score, gap_pct, is_sharpe, oos_sharpe, trades):
    if trades < 10:
        return "Statistical reliability is limited by very few trades — results should be interpreted cautiously."

    if risk_level == "low":
        return (
            f"Overfitting risk is low (score: {score:.0f}, green) — "
            f"in-sample and out-of-sample performance align closely."
        )
    elif risk_level == "medium":
        parts = [f"Overfitting risk is moderate (score: {score:.0f}, yellow)."]
        if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe):
            parts.append(
                f"In-sample Sharpe ({is_sharpe:.2f}) exceeds out-of-sample ({oos_sharpe:.2f}), "
                f"a gap of {gap_pct:.0f}%."
            )
        return " ".join(parts)
    else:
        parts = [
            f"Overfitting risk is high (score: {score:.0f}, red) — "
            f"performance is likely unreliable for live trading."
        ]
        if np.isfinite(gap_pct) and gap_pct > 0:
            parts.append(f"Train/OOS gap is {gap_pct:.0f}%.")
        return " ".join(parts)


def _sentence_regimes(periods: List[Dict]) -> Optional[str]:
    """Build a sentence about which regime performed best, if data available."""
    if not periods:
        return None

    all_regimes: Dict[str, List[float]] = {"sideways": [], "trend": [], "volatile": []}
    regime_sharpes: Dict[str, List[float]] = {"sideways": [], "trend": [], "volatile": []}

    for p in periods:
        test_sharpe = _safe_float(p.get("test_sharpe"))
        if not np.isfinite(test_sharpe):
            continue
        for regime in ("sideways", "trend", "volatile"):
            pct = _safe_float(p.get(f"pct_{regime}"))
            if np.isfinite(pct) and pct >= 0.33:
                regime_sharpes[regime].append(test_sharpe)

    # Find the best and worst regimes
    means = {}
    for regime, sr_list in regime_sharpes.items():
        if sr_list:
            means[regime] = float(np.nanmean(sr_list))
    if len(means) < 2:
        return None

    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    best_sr = means[best]
    worst_sr = means[worst]

    if best_sr <= 0 and worst_sr <= 0:
        return "No regime produced positive risk-adjusted returns."

    return (
        f"It performed best in {best} markets "
        f"(Sharpe {best_sr:.2f})"
        f"{f' and struggled in {worst} conditions (Sharpe {worst_sr:.2f}).' if worst_sr < best_sr else '.'}"
    )


def _describe_sharpe(sharpe: float) -> str:
    if not np.isfinite(sharpe):
        return "Sharpe"
    if sharpe >= 2.0:
        return "exceptional Sharpe"
    elif sharpe >= 1.0:
        return "strong Sharpe"
    elif sharpe >= 0.5:
        return "moderate Sharpe"
    else:
        return "Sharpe"


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _fmt_date(d: Any) -> str:
    if not d:
        return ""
    s = str(d)
    return s[:10] if len(s) >= 10 else s
