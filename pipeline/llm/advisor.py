"""LLM advisor for backtest result analysis and next-study planning.

Takes a completed backtest result (metrics_row + config) and asks
an LLM to analyze it, returning a JSON-formatted suggestion for
the next study to run. Reuses the LLM backend config from
pipeline/llm/sentiment.py (llm_backend, llm_model, llm_api_key).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np


def _build_prompt(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Format structured backtest data into an LLM prompt."""
    model = str(metrics.get("model", "unknown"))
    pair = str(config.get("pair", "EURUSD"))
    tf = str(config.get("timeframe", "H1"))
    sd = str(config.get("start_date", "") or "")[:10]
    ed = str(config.get("end_date", "") or "")[:10]
    n_trials = int(metrics.get("n_trials", config.get("n_trials", 10)))
    repeats = int(metrics.get("repeats", config.get("repeats", 1)))

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

    # Feature importance (top 5)
    fi_text = ""
    fi_list = metrics.get("diagnostics", {}).get("feature_importance") or []
    if isinstance(fi_list, list) and fi_list:
        top5 = fi_list[:5]
        fi_lines = [f"{e.get('feature','?')}: {e.get('importance',0):.2%}" for e in top5]
        fi_text = "\n".join(fi_lines)

    # Walk-forward summary
    wf = metrics.get("walkforward_periods") or []
    n_periods = len(wf)
    n_signal = sum(1 for p in wf if (p.get("trades", 0) or 0) > 0) if wf else 0

    # Regime best/worst
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

    # Available presets for recommendation
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


def analyze_backtest(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze a completed backtest and return a suggested next study.

    Parameters
    ----------
    metrics : dict
        The per-model metrics_row from api/tasks.py
    config : dict, optional
        Original backtest config

    Returns
    -------
    dict with keys: insight, recommended_preset, reason, parameter_changes,
    predicted_improvement, warning, error (if LLM failed)
    """
    config = config or {}
    prompt = _build_prompt(metrics, config)

    # Determine backend config
    feats_config = config.get("features_config", config)
    llm_backend = os.environ.get("LLM_BACKEND", str(feats_config.get("llm_backend", "ollama")))
    llm_model = os.environ.get("LLM_MODEL", str(feats_config.get("llm_model", "llama3")))
    llm_api_key = os.environ.get("OPENAI_API_KEY", str(feats_config.get("llm_api_key", "")))

    response_text = _call_llm(prompt, llm_backend, llm_model, llm_api_key)

    if response_text is None:
        return {"error": "LLM backend unavailable", "insight": _fallback_insight(metrics)}

    try:
        # Strip markdown fences
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "LLM response not valid JSON", "raw_text": response_text[:300]}


def _call_llm(prompt: str, backend: str, model: str, api_key: str) -> Optional[str]:
    """Call the configured LLM backend and return the response text."""
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


def _fallback_insight(metrics: Dict[str, Any]) -> str:
    sharpe = _f(metrics.get("sharpe"))
    overfit = metrics.get("overfitting") or {}
    risk = str(overfit.get("risk_level", "unknown"))
    return (
        f"Your model achieved Sharpe {sharpe:.2f} with {risk} overfitting risk. "
        "Consider running the Classical Standard preset with more indicators "
        "or the Ensemble Minimal preset to compare architectures."
    )
