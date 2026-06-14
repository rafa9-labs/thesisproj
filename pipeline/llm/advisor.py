"""LLM advisor for backtest result analysis and next-study planning.

Takes a completed backtest result (metrics_row + config) and returns
a structured 3-section diagnostic report:
  1. Multiple Testing Risk (DSR-based)
  2. Friction Attrition (transaction cost impact)
  3. Regime & Feature Decay (robustness across market conditions)

Reuses the LLM backend config from pipeline/llm/sentiment.py when available
to enrich the computed analysis with natural-language insights.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List

import numpy as np


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _i(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _compute_dsr_section(
    metrics: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    overfit = metrics.get("overfitting") or {}
    oos_sharpe = _f(overfit.get("oos_mean_sharpe") or metrics.get("sharpe"))
    dsr_min = _f(overfit.get("dsr_min_sharpe") or 0)
    n_periods = _i(overfit.get("n_periods") or 0)
    risk_level = str(overfit.get("risk_level") or "unknown")
    n_trials = _i(metrics.get("n_trials") or config.get("n_trials") or 0)

    title = ""
    severity = "info"
    detail = ""
    recommendation = ""

    if np.isfinite(dsr_min) and dsr_min > 0:
        if oos_sharpe >= dsr_min:
            title = "DSR passes significance threshold"
            severity = "green"
            detail = (
                f"Deflated Sharpe Ratio adjusted for {n_periods} OOS periods"
                + (f" and {n_trials} HPO trials" if n_trials > 0 else "")
                + f". Reported Sharpe {oos_sharpe:.2f} exceeds the multiple-testing"
                + f" threshold of {dsr_min:.2f} at 95% confidence."
            )
            recommendation = "Result is statistically significant. Proceed to next study or deployment validation."
        else:
            title = "Reported Sharpe likely inflated by selection bias"
            severity = "red"
            gap_pct = abs((oos_sharpe - dsr_min) / max(abs(dsr_min), 0.01)) * 100
            detail = (
                f"Deflated Sharpe Ratio adjusted for {n_periods} OOS periods"
                + (f" and {n_trials} HPO trials" if n_trials > 0 else "")
                + f". Reported Sharpe {oos_sharpe:.2f} falls below the DSR threshold"
                + f" of {dsr_min:.2f}. Estimated over-optimism: {gap_pct:.0f}%."
                + " The result may be a statistical fluke from HPO selection bias."
            )
            recommendation = "Reduce HPO trials, increase OOS periods, or use cross-validation with out-of-sample holdout."
    elif n_trials > 0 or n_periods > 0:
        title = "Multiple testing assessment"
        severity = "amber"
        detail = (
            f"Executed {n_trials} HPO trials over {n_periods} OOS periods."
            + " Without DSR data, assume reported Sharpe may be 30-50% inflated due to selection bias."
        )
        recommendation = "Enable overfitting assessment with DSR computation for accurate significance testing."
    else:
        title = "No HPO or OOS data available"
        severity = "info"
        detail = "Run a backtest with HPO trials > 0 and multiple walk-forward periods for multiple-testing risk assessment."
        recommendation = "Configure HPO trials >= 10 and months >= 6 for baseline DSR computation."

    return {
        "title": title,
        "severity": severity,
        "detail": detail,
        "recommendation": recommendation,
    }


def _compute_friction_section(
    metrics: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    avg_trade = _f(metrics.get("avg_trade") or 0)
    win_rate = _f(metrics.get("win_rate") or 0)
    total_trades = _i(metrics.get("total_trades") or 0)
    sharpe = _f(metrics.get("sharpe") or 0)
    pair = str(config.get("pair") or "EURUSD")

    pip_value = 0.0001
    spread_pips = 1.0
    slippage_pips = 0.5
    total_cost_pips = spread_pips + slippage_pips

    title = ""
    severity = "info"
    detail = ""
    recommendation = ""

    if not np.isfinite(avg_trade) or total_trades == 0:
        title = "Insufficient trade data for friction analysis"
        severity = "info"
        detail = "No trades or avg_trade data available. Friction attrition cannot be computed."
        recommendation = "Run a backtest that produces at least 20 trades for a meaningful friction assessment."
        return {"title": title, "severity": severity, "detail": detail, "recommendation": recommendation}

    avg_win_pips = 0.0
    if np.isfinite(avg_trade) and total_trades > 0 and np.isfinite(win_rate) and win_rate > 0:
        avg_win_pips = abs(avg_trade) * 10000 / max(win_rate, 0.01)
    else:
        avg_win_pips = abs(avg_trade) * 10000

    if avg_win_pips > 0:
        cost_pct = min(100, (total_cost_pips / avg_win_pips) * 100)
    else:
        cost_pct = 100.0

    if cost_pct > 50:
        title = f"Transaction costs consume {cost_pct:.0f}% of gross alpha"
        severity = "red"
        detail = (
            f"Average trade magnitude: {avg_win_pips:.1f} pips."
            + f" At estimated {spread_pips} pip spread + {slippage_pips} pip slippage"
            + f" (total {total_cost_pips} pips), friction consumes {cost_pct:.0f}%"
            + f" of average profitable trade returns across {total_trades} trades."
        )
        recommendation = "This strategy is highly sensitive to execution quality. Consider longer holding periods, wider stops, or models with higher per-trade alpha."
    elif cost_pct > 20:
        title = f"Transaction costs consume {cost_pct:.0f}% of gross alpha"
        severity = "amber"
        detail = (
            f"Average trade magnitude: {avg_win_pips:.1f} pips."
            + f" At {total_cost_pips} pips estimated total cost, friction consumes"
            + f" {cost_pct:.0f}% of gross returns over {total_trades} trades."
        )
        recommendation = "Moderate friction impact. Monitor execution quality in live trading. Consider adding a cost-aware filter to skip low-alpha signals."
    else:
        title = f"Friction attrition is manageable ({cost_pct:.0f}% of returns)"
        severity = "green"
        detail = (
            f"Average trade magnitude: {avg_win_pips:.1f} pips."
            + f" At {total_cost_pips} pips estimated cost, only {cost_pct:.0f}% of"
            + f" gross returns is consumed by spread and slippage over {total_trades} trades."
        )
        recommendation = "Transaction costs are well-covered by per-trade alpha. Strategy is cost-robust for most execution conditions."

    return {
        "title": title,
        "severity": severity,
        "detail": detail,
        "recommendation": recommendation,
    }


def _compute_regime_section(
    metrics: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    wf = metrics.get("walkforward_periods") or []
    fi_list = (metrics.get("diagnostics") or {}).get("feature_importance") or []

    title = ""
    severity = "info"
    detail = ""
    recommendation = ""

    if not wf:
        title = "No walk-forward regime data"
        severity = "info"
        detail = "Run a multi-period backtest to assess regime robustness and feature stability across market conditions."
        recommendation = "Configure months >= 6 for regime coverage analysis."
        return {"title": title, "severity": severity, "detail": detail, "recommendation": recommendation}

    n_periods = len(wf)
    avg_trend = np.mean([_f(p.get("pct_trend")) for p in wf if _f(p.get("pct_trend")) > 0]) if wf else 0
    avg_volatile = np.mean([_f(p.get("pct_volatile")) for p in wf if _f(p.get("pct_volatile")) > 0]) if wf else 0
    avg_sideways = np.mean([_f(p.get("pct_sideways")) for p in wf if _f(p.get("pct_sideways")) > 0]) if wf else 0

    best_r, best_sr = "N/A", float("-inf")
    worst_r, worst_sr = "N/A", float("inf")
    for p in wf:
        sr = _f(p.get("test_sharpe"))
        for r in ("trend", "sideways", "volatile"):
            pct = _f(p.get(f"pct_{r}"))
            if pct >= 0.25 and np.isfinite(sr):
                if sr > best_sr:
                    best_sr, best_r = sr, r
                if sr < worst_sr:
                    worst_sr, worst_r = sr, r

    regime_detail = ""
    if np.isfinite(best_sr) and np.isfinite(worst_sr):
        gap_width = best_sr - worst_sr
        regime_detail = (
            f"Over {n_periods} OOS periods, best performance in {best_r} regimes"
            + f" (avg Sharpe {best_sr:.2f}), worst in {worst_r} regimes"
            + f" (avg Sharpe {worst_sr:.2f}). Performance gap: {gap_width:.2f}."
        )
    else:
        regime_detail = f"Walk-forward across {n_periods} periods. Trend coverage: {avg_trend:.0%}, Sideways: {avg_sideways:.0%}, Volatile: {avg_volatile:.0%}."

    feature_decay_detail = ""
    if isinstance(fi_list, list) and len(fi_list) > 1:
        top1 = fi_list[0]
        top1_name = str(top1.get("feature", "unknown"))
        top1_imp = _f(top1.get("importance"))
        if len(fi_list) > 1:
            second_imp = _f(fi_list[1].get("importance", 0))
            concentration = top1_imp / max(second_imp, 0.001)
            if concentration > 3:
                feature_decay_detail = (
                    f" Model heavily reliant on '{top1_name}' ({top1_imp:.2%} importance,"
                    + f" {concentration:.1f}x the next feature)."
                    + " Single-feature dependency increases risk of regime-shift decay."
                )
            elif concentration > 1.5:
                feature_decay_detail = (
                    f" Top feature '{top1_name}' at {top1_imp:.2%} importance."
                    + " Moderate diversification across features."
                )
            else:
                feature_decay_detail = (
                    f" Top feature '{top1_name}' at {top1_imp:.2%} importance."
                    + " Well-diversified feature attribution."
                )

    if np.isfinite(best_sr) and np.isfinite(worst_sr) and (best_sr - worst_sr) > 1.0:
        title = f"Large regime performance gap ({best_sr - worst_sr:.1f} Sharpe)"
        severity = "red"
        detail = regime_detail + feature_decay_detail
        recommendation = (
            f"Consider regime-specific model specialization for {worst_r} conditions,"
            + " or use an ensemble that blends trend/range strategies."
        )
    elif avg_volatile > 0.3:
        title = f"Good volatile-regime exposure ({avg_volatile:.0%})"
        severity = "green"
        detail = regime_detail + feature_decay_detail
        recommendation = "Strategy has been tested in rough market conditions. Regime robustness is validated."
    elif avg_sideways > 0.3:
        title = f"Significant sideways-market exposure ({avg_sideways:.0%})"
        severity = "amber"
        detail = regime_detail + feature_decay_detail
        recommendation = "Sideways markets are challenging for trend-following models. Verify that performance holds in flat conditions."
    else:
        title = "Regime coverage across market conditions"
        severity = "info"
        detail = regime_detail + feature_decay_detail
        recommendation = "Extend backtest to include diverse market regimes for more robust assessment."

    return {
        "title": title,
        "severity": severity,
        "detail": detail,
        "recommendation": recommendation,
    }


def compute_structured_analysis(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the 3-section structured analysis from existing metrics.

    No LLM required. Works purely from the structured backtest data.
    """
    config = config or {}
    return {
        "dsr_analysis": _compute_dsr_section(metrics, config),
        "friction_analysis": _compute_friction_section(metrics, config),
        "regime_analysis": _compute_regime_section(metrics, config),
    }


def _build_prompt(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Format structured backtest data into an LLM prompt."""
    model = str(metrics.get("model", "unknown"))
    pair = str(config.get("pair", "EURUSD"))
    tf = str(config.get("timeframe", "H1"))
    sd = str(config.get("start_date", "") or "")[:10]
    ed = str(config.get("end_date", "") or "")[:10]
    n_trials = _i(metrics.get("n_trials") or config.get("n_trials") or 10)
    repeats = _i(metrics.get("repeats") or config.get("repeats") or 1)

    sharpe = _f(metrics.get("sharpe"))
    win_rate = _f(metrics.get("win_rate"))
    max_dd = _f(metrics.get("max_drawdown"))
    total_ret = _f(metrics.get("total_return_pct"))
    trades = _i(metrics.get("total_trades"))

    overfit = metrics.get("overfitting") or {}
    risk_level = str(overfit.get("risk_level", "unknown"))
    risk_score = _f(overfit.get("overfit_score"))
    is_sharpe = _f(overfit.get("is_mean_sharpe"))
    oos_sharpe = _f(overfit.get("oos_mean_sharpe"))
    gap_pct = _f(overfit.get("train_oos_gap_pct"))
    dsr_min = _f(overfit.get("dsr_min_sharpe"))

    fi_text = ""
    fi_list = metrics.get("diagnostics", {}).get("feature_importance") or []
    if isinstance(fi_list, list) and fi_list:
        top5 = fi_list[:5]
        fi_lines = [f"{e.get('feature','?')}: {e.get('importance',0):.2%}" for e in top5]
        fi_text = "\n".join(fi_lines)

    wf = metrics.get("walkforward_periods") or []
    n_periods = len(wf)
    n_signal = sum(1 for p in wf if (p.get("trades", 0) or 0) > 0) if wf else 0

    regime_info = ""
    if wf:
        best_r, best_sr = "N/A", float("-inf")
        worst_r, worst_sr = "N/A", float("inf")
        for p in wf:
            sr = _f(p.get("test_sharpe"))
            for r in ("trend", "sideways", "volatile"):
                pct = _f(p.get(f"pct_{r}"))
                if pct >= 0.33 and np.isfinite(sr):
                    if sr > best_sr:
                        best_sr, best_r = sr, r
                    if sr < worst_sr:
                        worst_sr, worst_r = sr, r
        if best_r != "N/A":
            regime_info = f"Best regime: {best_r} (Sharpe {best_sr:.2f}), Worst: {worst_r} (Sharpe {worst_sr:.2f})"

    presets_text = """Available Quick Start presets (format: key=label:models):
- baseline=Baseline Check: logistic
- signal=Quick Signal: xgboost
- classical_min=Classical Minimal: logistic
- classical_std=Classical Standard: logistic,svm,xgboost
- classical_deep=Classical Deep: 5 models
- classical_prod=Classical Production: 5 models, deep HPO
- deep_min=Deep Minimal: lstm
- deep_std=Deep Standard: lstm,cnn
- deep_prod=Deep Production: cnn,lstm,transformer
- ensemble_min=Ensemble Minimal: adaptive_regime
- ensemble_std=Ensemble Standard: 2 ensembles
- ensemble_prod=Ensemble Production: 2 ensembles, deep HPO
- rl_min=RL Minimal: dqn
- rl_prod=RL Production: dqn, deep HPO"""

    return f"""You are a quantitative trading research assistant. Analyze these backtest results and suggest the NEXT study to run. Return ONLY valid JSON, no markdown or commentary.

Backtest Configuration:
Model: {model} | Pair: {pair} {tf} | Period: {sd} to {ed}
HPO: {n_trials} trials x {repeats} runs

Results:
Sharpe: {sharpe:.2f} | Win Rate: {win_rate:.0%} | Max DD: {max_dd:.1%}
Total Return: {total_ret:+.1%} | Trades: {trades}

Overfitting Assessment:
Risk: {risk_level} (score {risk_score:.0f}/100) | Train/OOS gap: {gap_pct:.0f}%
IS Sharpe: {is_sharpe:.2f} | OOS Sharpe: {oos_sharpe:.2f}
Significance threshold: Sharpe {chr(8805)} {dsr_min:.2f}

Top Features:
{fi_text}

Walk-Forward: {n_periods} periods, {n_signal} with trades
{regime_info}

{presets_text}

Return this EXACT JSON structure:
{{"insight": "ONE key finding about what worked or didn't", "recommended_preset": "one of the preset keys above", "reason": "why this preset is recommended", "parameter_changes": ["specific parameter suggestion 1", "suggestion 2"], "predicted_improvement": "rough estimate of what might improve", "warning": "ONE caution, or null if none"}}"""


def analyze_backtest(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze a completed backtest and return a structured 3-section diagnostic report.

    Returns the structured analysis (always available, no LLM required) plus
    optional LLM-enriched insight if an LLM backend is configured.
    """
    config = config or {}
    structured = compute_structured_analysis(metrics, config)

    prompt = _build_prompt(metrics, config)

    feats_config = config.get("features_config", config)
    llm_backend = os.environ.get("LLM_BACKEND", str(feats_config.get("llm_backend", "ollama")))
    llm_model = os.environ.get("LLM_MODEL", str(feats_config.get("llm_model", "llama3")))
    llm_api_key = os.environ.get("OPENAI_API_KEY", str(feats_config.get("llm_api_key", "")))

    response_text = _call_llm(prompt, llm_backend, llm_model, llm_api_key)

    llm_enrichment = None
    if response_text is not None:
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:]) if len(lines) > 1 else text
            if text.endswith("```"):
                text = text[:-3].strip()
            llm_enrichment = json.loads(text)
        except json.JSONDecodeError:
            pass

    result: Dict[str, Any] = {
        "insight": llm_enrichment.get("insight", "") if llm_enrichment else "",
        "recommended_preset": llm_enrichment.get("recommended_preset", "") if llm_enrichment else "",
        "reason": llm_enrichment.get("reason", "") if llm_enrichment else "",
        "parameter_changes": llm_enrichment.get("parameter_changes", []) if llm_enrichment else [],
        "predicted_improvement": llm_enrichment.get("predicted_improvement", "") if llm_enrichment else "",
        "warning": llm_enrichment.get("warning", None) if llm_enrichment else None,
        "dsr_analysis": structured["dsr_analysis"],
        "friction_analysis": structured["friction_analysis"],
        "regime_analysis": structured["regime_analysis"],
    }

    return result


def _call_llm(prompt: str, backend: str, model: str, api_key: str) -> Optional[str]:
    try:
        if backend == "ollama":
            return _call_ollama(prompt, model)
        elif backend == "openai":
            return _call_openai(prompt, model, api_key)
        elif backend == "anthropic":
            return _call_anthropic(prompt, model, api_key)
        else:
            return _call_ollama(prompt, model)
    except Exception:
        return None


def _call_ollama(prompt: str, model: str) -> str:
    import requests
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    import requests
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.3},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    import requests
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": model, "max_tokens": 500, "messages": [{"role": "user", "content": prompt}]},
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]
