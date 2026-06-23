from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, List
from math import pi

import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib

# Headless / non-interactive plotting by default
matplotlib.use("Agg")


import os as _os
SKIP_PLOTS = bool(int(_os.environ.get("SKIP_PLOTS", "0")))

# --- METRICS SCHEMA (exactly 16) ---
N_METRICS = 16
METRIC_NAMES = [
    "cstrategy", "outperformance", "creturns", "sharpe", "drawdown", "trades",
    "geo_mean_ann", "directional_accuracy", "precision_macro", "f1_macro",
    "active_rate", "profit_per_hit", "return_per_trade", "win_rate",
    "strategy_volatility", "kurtosis",
]

import numpy as _np

def _coerce_direction_labels(arr, labels=(-1, 0, 1), deadzone: float = 0.5):
    """
    Coerce predictions/targets into discrete {-1,0,+1} labels.
    - If floats: apply deadzone around 0, then sign-binarize.
    - If ints: map any unexpected labels to 0 (neutral).
    """
    a = _np.asarray(arr)
    if a.size == 0:
        return a.astype(int)
    if a.dtype.kind in ("f", "c"):
        a = _np.where(a > deadzone, 1, _np.where(a < -deadzone, -1, 0))
    else:
        try:
            a = a.astype(int, copy=False)
        except Exception:
            a = a.astype(float)
            a = _np.where(a > deadzone, 1, _np.where(a < -deadzone, -1, 0))
    # sanitize unexpected labels
    valid = _np.isin(a, _np.array(labels, dtype=int))
    if not bool(_np.all(valid)):
        a = _np.where(valid, a, 0)
    return a.astype(int, copy=False)

def ensure_metric_tuple(x, fill=np.nan):
    """
    Return a 16-length tuple. If x is scalar, broadcast; if shorter, pad; if longer, trim.
    """
    import numpy as _np
    if x is None:
        return tuple([fill] * N_METRICS)
    if isinstance(x, (int, float)):
        return tuple([float(x)] + [fill] * (N_METRICS - 1))
    try:
        arr = list(x)
    except Exception:
        return tuple([fill] * N_METRICS)
    if len(arr) < N_METRICS:
        arr = arr + [fill] * (N_METRICS - len(arr))
    elif len(arr) > N_METRICS:
        arr = arr[:N_METRICS]
    return tuple(float(v) if v is not None else fill for v in arr)

def validate_metrics_shape(x, context: str | None = None):
    """
    Ensure metric tuple has the expected length.

    Parameters
    ----------
    x : Any
        Metric-like object (tuple/list/np.ndarray).
    context : str, optional
        Optional label for error messages; ignored by callers that
        don't need it.
    """
    ctx = f" in {context}" if context else ""
    if x is None:
        raise ValueError(f"Metrics is None{ctx}")
    try:
        n = len(x)
    except Exception:
        raise ValueError(f"Metrics must be a sequence of length {N_METRICS}{ctx}")
    if n != N_METRICS:
        raise ValueError(f"Metric arity != {N_METRICS} (got {n}){ctx}")
    # Cast to floats but DO NOT pad/trim -- the whole point is to catch drift.
    seq = list(x)
    return tuple(float(v) if v is not None else float("nan") for v in seq)


def combine_block_scores(block_scores, config):
    """
    NaN-robust trimmed mean with coverage gate. Returns float or np.nan.
    Expects block_scores as list[float or nan].
    """
    import numpy as _np
    arr = _np.asarray(block_scores, dtype=float)
    valid_mask = _np.isfinite(arr)
    k = int(valid_mask.sum())
    K = int(len(arr))
    if K == 0:
        return float("nan")

    coverage = float(k / K)
    min_cov = float(config.get("cv_min_coverage", 0.60))
    if k == 0 or coverage < min_cov:
        return float("nan")

    vals = _np.sort(arr[valid_mask])
    trim_frac = float(config.get("cv_trim_frac", 0.10))
    trim_n = int(round(k * trim_frac))
    if trim_n > 0 and 2 * trim_n < k:
        vals = vals[trim_n:-trim_n]
    return float(_np.nanmean(vals)) if vals.size else float("nan")

def is_coverage_intent(features_config) -> bool:
    """
    True if config indicates coverage/active-rate gating should be used.

    Coverage intent is satisfied by ANY of:
    - gating_mode / gate_mode == "coverage"
    - target_active_rate > 0
    - target_coverage > 0

    This is used to keep CV and real-trading logic consistent and avoid
    silent fallbacks to static confidence thresholds.
    """
    fc = (features_config or {})
    mode = str(fc.get("gating_mode", fc.get("gate_mode", "threshold"))).lower()

    def _as_pos_float(x):
        try:
            v = float(x)
            return v if v > 0.0 else 0.0
        except Exception:
            return 0.0

    tar = _as_pos_float(fc.get("target_active_rate", 0.0))
    tcov = _as_pos_float(fc.get("target_coverage", 0.0))
    return (mode == "coverage") or (tar > 0.0) or (tcov > 0.0)


def freeze_confidence_threshold(features_config, default_conf, coverage_conf_thr=None):
    """
    Decide the single confidence threshold to use everywhere.
    If coverage/active-rate gating is enabled AND a coverage-calibrated threshold was computed, use it.

    Key rule:
      - If target_active_rate (or target_coverage) > 0, we treat it as coverage gating,
        even if gating_mode wasn't explicitly set to "coverage".
    """
    import numpy as _np

    fc = (features_config or {})
        
    # Single source of truth for coverage intent
    try:
        _use_cov = bool(is_coverage_intent(fc))
    except Exception:
        _use_cov = False

    # Tripwire: coverage intent but missing calibrated threshold
    if _use_cov and coverage_conf_thr is None:
        # Deterministic, auditable behavior:
        # return NaN so callers can penalize in CV and treat as invalid/diagnostic in real sim.
        return float("nan")
    
    if _use_cov and coverage_conf_thr is not None:
        try:
            thr = float(coverage_conf_thr)
        except Exception:
            thr = _np.nan
        if _np.isfinite(thr):
            # Clamp to configured bounds (but never mask NaN tripwire above)
            try:
                thr_min = float(fc.get("min_conf_thr_cov", fc.get("min_conf_thr", 0.0)))

            except Exception:
                thr_min = 0.0
            try:
                thr_max = float(fc.get("max_conf_thr_cov", fc.get("max_conf_thr", 1.0)))
            except Exception:
                thr_max = 1.0
            if _np.isfinite(thr_max) and _np.isfinite(thr_min) and thr_max >= thr_min:
                thr = float(_np.clip(thr, thr_min, thr_max))
            return float(thr)

    thr = float(fc.get("confidence_threshold", default_conf))
    # Clamp scalar threshold too (keeps behavior consistent with dynamic thr clipping)
    if _np.isfinite(thr):
        try:
            thr_min = float(fc.get("min_conf_thr", 0.0))
        except Exception:
            thr_min = 0.0
        try:
            thr_max = float(fc.get("max_conf_thr", 1.0))
        except Exception:
            thr_max = 1.0
        if _np.isfinite(thr_max) and _np.isfinite(thr_min) and thr_max >= thr_min:
            thr = float(_np.clip(thr, thr_min, thr_max))
    return float(thr)

def compute_metrics(
    returns,
    positions,
    frequency_per_year=None,
    sharpe_cap=None,
    use_hac: bool = True,
    hac_max_lag="auto",
    min_active_obs: int = 25,
    std_floor: float = 1e-8,
):
    """
    Computes Sharpe (annualized), max drawdown, and trade count with robust guards:
    - annualization uses *estimated* bars-per-year from the index unless provided,
    - std can use HAC (Newey-West) to handle autocorrelation,
    - degenerate samples (too few active obs / near-zero vol) -> Sharpe = 0,
    - optional soft cap on Sharpe (env SHARPE_CAP or passed-in).
    """
    import os, numpy as np
    returns = returns.dropna()

    if frequency_per_year is None:
        try:
            frequency_per_year = float(estimate_frequency_per_year(returns.index))
        except Exception:
            frequency_per_year = 252.0  # conservative fallback
    ann_factor = float(np.sqrt(max(1.0, frequency_per_year)))

    # Treat tiny values as inactive to avoid ~0 variance artifacts
    active = returns[np.abs(returns) > 1e-12]
    n_active = int(active.size)

    if n_active < int(min_active_obs):
        sharpe = 0.0
    else:
        if use_hac:
            std = float(hac_std(active, max_lag=hac_max_lag))
        else:
            std = float(active.std(ddof=1))
        mean = float(active.mean())
        sharpe = (mean / std) * ann_factor if (np.isfinite(std) and std >= std_floor) else 0.0

    # Soft cap
    if sharpe_cap is None:
        try:
            cap_env = os.environ.get("SHARPE_CAP")
            if cap_env is not None:
                sharpe_cap = float(cap_env)
        except Exception:
            sharpe_cap = None
    if sharpe_cap is not None and sharpe_cap > 0:
        sharpe = float(np.clip(sharpe, -sharpe_cap, sharpe_cap))

    # Max drawdown
    cum = returns.cumsum().apply(np.exp)
    drawdown = (cum / cum.cummax() - 1).min() if not cum.empty else 0.0

    # Trade count (directional fills; robust to fractional sizing)
    # NOTE: position_exec may be fractional (vol targeting / TWAP ramps). Counting
    # abs(diff) on the raw series will over-count "trades" whenever size is
    # adjusted while direction stays the same. For reporting + reliability gates
    # we count only *directional* fills based on sign(position).
    try:
        p = positions
        if p is None:
            trades = 0
        else:
            if hasattr(p, "fillna"):
                p = p.fillna(0)
            p_arr = np.asarray(p, dtype=float)
            if p_arr.size <= 1:
                trades = 0
            else:
                p_dir = np.sign(p_arr)
                trades = int(np.sum(np.abs(np.diff(p_dir))))
    except Exception:
        trades = 0


    return round(sharpe, 2), round(drawdown, 4), trades

def compute_geometric_mean_annualized(returns):
    """
    returns: Series of per-period log returns
    """
    import numpy as np
    n = len(returns)
    if n == 0:
        return np.nan
    compounded = np.exp(returns.sum())
    try:
        bars_per_year = float(estimate_frequency_per_year(returns.index))
    except Exception:
        bars_per_year = 252.0
    annual_factor = bars_per_year / max(1, n)
    return compounded ** annual_factor - 1


def set_global_determinism(seed=42):
    import os
    import random
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # GPU determinism (optional)
    

def rolling_slope(arr, window):
    """Computes rolling slope via linear regression for each window."""
    idx = np.arange(window)
    def _slope(x):
        if np.any(np.isnan(x)): return np.nan
        return np.polyfit(idx, x, 1)[0]
    return arr.rolling(window).apply(_slope, raw=True)

def ensure_list(val):
    if isinstance(val, list):
        return val
    elif val is None:
        return [{}]
    else:
        return [val]
    
def ensure_dict(obj):
    import json
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}

def print_feature_stats(df, label):
    """
    Debug-only summary stats for features (mean/std/min/max).
    Respects LOG_MODE via log_print(level="DEBUG").
    """
    try:
        stats = df.describe().T[["mean", "std", "min", "max"]]
    except Exception:
        # Fallback: if those columns are missing, just print full describe
        stats = df.describe().T

    log_print(f"\n[DEBUG] {label} feature stats:", level="DEBUG")
    log_print(stats.to_string(), level="DEBUG")
    
def filter_params(params, prefix):
    import numpy as _np
    plen = len(prefix)
    out = {k[plen:]: v for k, v in params.items() if str(k).startswith(prefix)}
    # normalize NaN-like values to None (safe for sklearn switches like class_weight)
    for k, v in list(out.items()):
        if isinstance(v, float) and (_np.isnan(v) or _np.isinf(v)):
            out[k] = None
        elif isinstance(v, str) and v.strip().lower() in ("nan", "none", "null", ""):
            out[k] = None
    return out

def print_conf_stats(conf_array, label: str = "", median_thr: float | None = None, thr=None):
    """
    Debug-only: min/mean/max for a confidence array (+ optional threshold).
    Prints 'n/a' instead of NaN to avoid noisy logs.
    Respects LOG_MODE via log_print(level="DEBUG").
    """
    import numpy as np

    suffix = f" [{label}]" if label else ""
    

    # Optional: report the (median) threshold used for gating
    thr_str = "n/a"
    try:
        if thr is not None:
            _t = float(thr)
            if np.isfinite(_t):
                thr_str = f"{_t:.6f}"
    except Exception:
        thr_str = "n/a"

    if conf_array is None:
        thr_s = "n/a" if median_thr is None or (isinstance(median_thr, float) and np.isnan(median_thr)) else f"{float(median_thr):.6f}"
        log_print(f"GateDiag: thr_med={thr_str} | max_conf_min/mean/max: n/a n/a n/a{suffix}", level="DEBUG")

        return

    arr = np.asarray(conf_array, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        thr_s = "n/a" if median_thr is None or (isinstance(median_thr, float) and np.isnan(median_thr)) else f"{float(median_thr):.6f}"
        log_print(f"Gate stats: thr_med={thr_s} | conf(min/mean/max)=n/a n/a n/a{suffix}", level="DEBUG")
        return

    thr_s = "n/a" if median_thr is None or (isinstance(median_thr, float) and np.isnan(median_thr)) else f"{float(median_thr):.6f}"
    q = np.quantile(arr, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    iqr = float(q[2] - q[0])
    msg = (
        f"Gate stats: thr_med={thr_s} | "
        f"Gate stats: thr_med={thr_s} | std={arr.std():.6f} iqr={iqr:.6f} "
        f"p50={q[1]:.6f} p75={q[2]:.6f} p90={q[3]:.6f} p95={q[4]:.6f} p99={q[5]:.6f} | "
        f"conf(min/mean/max)={arr.min():.6f} {arr.mean():.6f} {arr.max():.6f}{suffix}"
    )
    log_print(msg, level="COMPACT")


# --- compute_full_evaluation_metrics extracted to pipeline/metrics_eval.py ---
from pipeline.metrics_eval import (  # noqa: F401
    compute_full_evaluation_metrics,
    _macro_prec_f1_from_confusion,
)

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


import numpy as np
import pandas as pd
from math import sqrt, e
from scipy.stats import norm

_EPS = 1e-12

def _safe_mean(x): 
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan

def _lisbon_now():
    try:
        from zoneinfo import ZoneInfo  # Py3.9+
        import datetime as _dt
        return _dt.datetime.now(ZoneInfo("Europe/Lisbon"))
    except Exception:
        import datetime as _dt
        return _dt.datetime.now()

def _format_run_stamp(dt=None):
    dt = dt or _lisbon_now()
    # 26_09_25__20_26  (DD_MM_YY__HH_MM)
    return dt.strftime("%d_%m_%y__%H_%M")

def make_results_run_dir(stamp=None, base_dir="results"):
    """
    Create and return <base_dir>/<STAMP> (Lisbon time). If the environment
    variable RESULTS_RUN_DIR is set, reuse it (idempotent) and infer STAMP
    from its basename. Returns (out_dir, stamp).
    """
    import os
    # Reuse a study folder if already provided
    env_run = os.environ.get("RESULTS_RUN_DIR")
    if env_run and (stamp is None):
        os.makedirs(env_run, exist_ok=True)
        return env_run, os.path.basename(env_run.rstrip(os.sep))

    if stamp is None:
        stamp = _format_run_stamp()
    out_dir = os.path.join(base_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, stamp

def init_study_tree(run_dir: str):
    """
    Create the top-level container for a study.

    New layout: we only ensure the run folder exists here. The aggregated
    comparison bucket is created lazily by `comparison_dirs(run_dir)`,
    which writes to `<run>/ALL/...`.
    """
    import os
    os.makedirs(run_dir, exist_ok=True)
    # NOTE: do **not** create `<run>/All` here -- it was an unused legacy
    # folder and only produced an empty, confusing directory.


def model_category(model_type: str) -> str:
    """
    Map model_type -> family bucket with the capitalization you asked for:
      - Classical: logistic, logistic_ovr, svm, decision_tree, random_forest, xgboost, lightgbm, catboost
      - RL:        cnn, lstm, transformer, gru, gru_lstm
      - DQN:       dqn (its own top-level group)
      - Ensembles: any name starting with 'ensemble_' or named stacking_ensemble/meta_ensemble
    Default is Classical.
    """
    m = (model_type or "").lower()
    if m in {"cnn", "lstm", "transformer", "gru", "gru_lstm"}:
        return "RL"
    if m in {"dqn"}:
        return "DQN"
    if m.startswith("ensemble_") or m in {"stacking_ensemble", "meta_ensemble"}:
        return "Ensembles"
    return "Classical"

def target_coverage_policy(model_type: str | None) -> float:
    """Return the **global** policy target signal coverage (active rate).

    Policy (B1): All model families use the *same* target coverage to ensure
    apples-to-apples comparisons (no exposure-budget confounds).

    Global target:
      - target_active_rate / target_coverage: 0.25

    Notes
    -----
    - `model_type` is retained for backwards compatibility, but intentionally ignored.
    """
    _ = model_type  # compatibility: parameter intentionally unused
    return 0.15
 

def enforce_target_coverage_policy(features_config: dict, model_type: str | None = None) -> dict:
    """In-place: set both target_active_rate and target_coverage to policy target."""
    fc = features_config if isinstance(features_config, dict) else {}
    mt = model_type or fc.get("model_type")
    tgt = float(target_coverage_policy(mt))
    fc["target_active_rate"] = tgt
    fc["target_coverage"] = tgt
    return fc


def _infer_family(model_type: str) -> str:
    """
    Backwards-compatible alias used by MLBacktester for mapping a model_type
    into a high-level family name (Classical / RL / DQN / Ensembles).

    This just forwards to model_category() so the logic is centralized here.
    """
    return model_category(model_type)


def friendly_model_name(model_type: str) -> str:
    m = (model_type or "").lower()
    mapping = {
        "logistic": "Logistic",
        "svm": "SVM",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "cnn": "CNN",
        "lstm": "LSTM",
        "gru": "GRU",
        "gru_lstm": "GRU-LSTM",
        "transformer": "Transformer",
        "dqn": "DQN",
        "rl": "RL",
        "ensemble_cnn_lstm_xgboost": "CNN+LSTM+XGB",
        "ensemble_transformer_xgb_dqn": "Transformer+XGB+DQN",
        "ensemble_adaptive_regime": "Adaptive Regime",
        "meta_ensemble": "Signal Committee",
        "stacking_ensemble": "Stacking Ensemble",
    }
    return mapping.get(m, model_type)

def ensure_model_dirs(base_run_dir: str, model_type: str, model_name: str | None = None):
    """
    Create (if needed) and return the model's base directory and a small map
    describing where to put artifacts for that model.

    NEW layout (no family, no per-month folders):
      <run>/<ModelDisplay>/
        |-- csv/
        |-- graphs/
        +-- heatmaps/

    All per-month / per-repetition files go into these subfolders and encode
    the month / rep in the *filename* (e.g. csv_month_1.csv, monthly_equity_1.png).
    """
    import os

    display = friendly_model_name(model_name or model_type)

    # Each model now lives directly under the run directory, no "Classical/RL/..." family.
    model_base = os.path.join(base_run_dir, display)

    csv_dir = os.path.join(model_base, "csv")
    graphs_dir = os.path.join(model_base, "graphs")
    heatmaps_dir = os.path.join(model_base, "heatmaps")

    for p in (model_base, csv_dir, graphs_dir, heatmaps_dir):
        os.makedirs(p, exist_ok=True)

    # 'final' kept for backward-compat: callers expect final["csv"/"graphs"/"heatmaps"]
    return model_base, {
        "base": model_base,
        "final": {"csv": csv_dir, "graphs": graphs_dir, "heatmaps": heatmaps_dir},
        "csv": csv_dir,
        "graphs": graphs_dir,
        "heatmaps": heatmaps_dir,
        # legacy key, but no separate Months/ folder anymore
        "months_root": model_base,
    }


def comparison_dirs(run_dir):
    """
    Build cross-model comparison dirs for a run.

    NEW layout: a single aggregated bucket:

      <run>/ALL/
        |-- csv/
        |-- graphs/
        +-- heatmaps/

    Per-model outputs live in <run>/<ModelDisplay>/... created by ensure_model_dirs().
    """
    import os

    def _mk(p: str) -> str:
        os.makedirs(p, exist_ok=True)
        return p

    def _bucket(root: str) -> dict:
        root = _mk(root)
        return {
            "dir": root,
            "csv": _mk(os.path.join(root, "csv")),
            "graphs": _mk(os.path.join(root, "graphs")),
            "heatmaps": _mk(os.path.join(root, "heatmaps")),
        }

    # Keep dict key "All" for callers, but store on disk in "ALL/"
    return {
        "All": _bucket(os.path.join(run_dir, "ALL")),
    }


# --- utilsNoWFO.py: full cumulative growth plot --------------------------------
def sanitize_proba(proba):
    """
    Clean and row-normalize predict_proba output:
    - cast to float
    - replace NaN/Inf with 0
    - row-normalize; if a row sums to 0, assign a uniform distribution
    """
    import numpy as np
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        return proba

    # Replace bad values
    bad = ~np.isfinite(proba)
    if bad.any():
        proba[bad] = 0.0

    # Row-normalize
    rowsum = proba.sum(axis=1, keepdims=True)  # shape (n, 1)
    zero_row_idx = np.flatnonzero(rowsum.ravel() == 0.0)  # shape (k,)
    if zero_row_idx.size:
        proba[zero_row_idx, :] = 1.0 / proba.shape[1]
        rowsum = proba.sum(axis=1, keepdims=True)

    proba = proba / rowsum
    # Clamp to avoid exact 0/1 which can blow up log-loss and downstream ratios
    eps = 1e-6
    proba = np.clip(proba, eps, 1.0 - eps)
    # Re-normalize to kill tiny drift from clipping
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba

TRAIN_TEST_MONTHS = {
    "logistic":    {"train": (36, 36), "test": (1, 1)},
    "svm":         {"train": (36, 36), "test": (1, 1)},
    "decision_tree":{"train": (36, 36), "test": (1, 1)},
    "random_forest":{"train": (36, 36), "test": (1, 1)},
    "xgboost":     {"train": (36, 36), "test": (1, 1)},
    "lightgbm":    {"train": (36, 36), "test": (1, 1)},
    "catboost":    {"train": (36, 36), "test": (1, 1)},

    # keep as-is if you want strict comparability vs classical
    "lstm":        {"train": (48, 48), "test": (1, 1)},
    "cnn":         {"train": (48, 48), "test": (1, 1)},
    "gru":         {"train": (48, 48), "test": (1, 1)},
    "gru_lstm":    {"train": (48, 48), "test": (1, 1)},

    # data-hungrier / higher-capacity models
    "transformer": {"train": (48, 48), "test": (1, 1)},
    "dqn":         {"train": (60, 60), "test": (1, 1)},
    "rl":          {"train": (60, 60), "test": (1, 1)},

    # ensembles inherit deep-data needs + benefit from more regime variety
    "ensemble_cnn_lstm_xgboost": {"train": (48, 48), "test": (1, 1)},
    "ensemble_adaptive_regime":  {"train": (48, 48), "test": (1, 1)},
    "meta_ensemble":             {"train": (48, 48), "test": (1, 1)},
    "stacking_ensemble":         {"train": (48, 48), "test": (1, 1)},
}

TRAIN_TEST_MONTHS_DEBUG = {
    "logistic":    {"train": (6, 12), "test": (1, 1)},
    "svm":         {"train": (6, 12), "test": (1, 1)},
    "decision_tree":{"train": (6, 12), "test": (1, 1)},
    "random_forest":{"train": (6, 12), "test": (1, 1)},
    "xgboost":     {"train": (6, 12), "test": (1, 1)},
    "lightgbm":    {"train": (6, 12), "test": (1, 1)},
    "catboost":    {"train": (6, 12), "test": (1, 1)},
    "lstm":        {"train": (6, 12), "test": (1, 1)},
    "cnn":         {"train": (6, 12), "test": (1, 1)},
    "gru":         {"train": (6, 12), "test": (1, 1)},
    "gru_lstm":    {"train": (6, 12), "test": (1, 1)},
    "transformer": {"train": (6, 12), "test": (1, 1)},
    "dqn":         {"train": (6, 12), "test": (1, 1)},
    "rl":          {"train": (6, 12), "test": (1, 1)},
    "ensemble_transformer_xgb_dqn": {"train": (6, 12), "test": (1, 1)},
    "ensemble_cnn_lstm_xgboost":   {"train": (6, 12), "test": (1, 1)},
    "ensemble_adaptive_regime":    {"train": (6, 12), "test": (1, 1)},
    "meta_ensemble":               {"train": (6, 12), "test": (1, 1)},
    "stacking_ensemble":           {"train": (6, 12), "test": (1, 1)},
}

def get_train_window_bounds(
    start_date,
    train_periods: int,
    data_start,
    window_type: str = "rolling",
    unit: str = "months",
):
    """Compute (train_start, train_end) respecting the window_type contract.

    rolling:   train_start = start_date  (slides forward with test_start)
    expanding: train_start = data_start  (anchored at data origin, never slides)

    Returns (train_start, train_end)
    """
    from config import period_offset

    if window_type == "expanding":
        train_start = data_start
    else:
        train_start = start_date

    train_end = start_date + period_offset(train_periods, unit=unit)
    return train_start, train_end


def _ensure_dt(s):
    import pandas as pd
    return pd.to_datetime(s, utc=True, errors="coerce")

def build_model_monthly_pivots(combined_df):
    """
    From a multi-model monthly results DataFrame, build:
      - equity_pivot: index=month (DatetimeIndex), cols=model, values=cumulative equity
      - returns_pivot: index=month (DatetimeIndex), cols=model, values=monthly strategy return
      - bh_equity: Series of BH cumulative equity across months (aligned to index)

    Correct aggregation across repeats:
      For each month k and model m, we compute the GEOMETRIC MEAN of the monthly factor
      cstrategy across repeats, then take a CUMPROD over months to form the equity curve.
      BH is computed via the geometric mean of the monthly BH factor (creturns).

    Expected columns in combined_df:
      ['model_type','test_end','cstrategy','creturns','equity_strategy','equity_bh', ...]
    """
    import numpy as np
    import pandas as pd

    if combined_df is None or len(combined_df) == 0:
        return (pd.DataFrame(), pd.DataFrame(), None)

    df = combined_df.copy()
    df["test_end"] = _ensure_dt(df["test_end"])  # existing helper in utils

    # Month key as tz-naive timestamp
    test_end_naive = df["test_end"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["month_dt"] = test_end_naive.dt.to_period("M").dt.to_timestamp()

    # Guard required fields
    if "cstrategy" not in df.columns:
        # if missing, try to reconstruct from equity_strategy via month-on-month ratios (not ideal)
        if "equity_strategy" in df.columns:
            df = df.sort_values(["model_type","test_end"])
            df["cstrategy"] = df.groupby("model_type")["equity_strategy"].pct_change().add(1.0)
        else:
            raise ValueError("combined_df must have 'cstrategy' (monthly factor).")

    # --- Geometric mean of monthly factors across repeats ---
    def _gmean(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce")
        x = x[(x > 0) & np.isfinite(x)]
        if x.empty:
            return np.nan
        return float(np.exp(np.mean(np.log(x))))

    # 1) Strategy: per (month, model) geometric mean of 'cstrategy'
    gm = (df.groupby(["month_dt","model_type"])["cstrategy"]
            .apply(_gmean)
            .rename("gm_factor")
            .reset_index())

    # 2) Pivot geometric means -> monthly factors table
    gm_pivot = (gm.pivot(index="month_dt", columns="model_type", values="gm_factor")
                  .sort_index())

    # 3) Cumulative equity = cumprod of the geometric-mean factors
    equity_pivot = gm_pivot.copy()
    equity_pivot = equity_pivot.ffill()   # in case of sparse months
    equity_pivot = equity_pivot.cumprod()

    # 4) Monthly returns pivot (mean factor - 1)
    returns_pivot = gm_pivot - 1.0

    # 5) Buy&Hold: geometric mean of 'creturns' per month across repeats, then cumprod
    bh_equity = None
    if "creturns" in df.columns:
        bh_gm = (df.groupby("month_dt")["creturns"].apply(_gmean).sort_index())
        bh_equity = bh_gm.cumprod().reindex(equity_pivot.index)

    return equity_pivot, returns_pivot, bh_equity


# --- Ranking helpers (GM equity, HAC-SR, PSR, DSR) ---------------------------
def _max_drawdown_from_equity(eq):
    """
    Max drawdown computed on a cumulative equity series (>=0).
    Returns positive drawdown magnitude, e.g., 0.25 means -25%.
    """
    import numpy as np, pandas as pd
    if eq is None:
        return float("nan")
    s = pd.Series(eq).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    roll_max = s.cummax()
    dd = (s / roll_max) - 1.0
    return float(abs(dd.min()))

def _mode_safe(x):
    import pandas as pd, numpy as np
    s = pd.Series(x)
    s = s.replace({np.nan: None})
    try:
        return s.mode(dropna=True).iloc[0]
    except Exception:
        # fallback: first non-null
        for v in s:
            if v is not None and v == v:
                return v
        return None

def build_model_ranking(combined_df, min_months: int = 1):
    """
    Build a cross-model ranking table from the concatenated monthly results produced
    by `real_trading_simulation` across repeats.

    Uses:
      [0x2022] Final cumulative equity built from the *geometric mean* of monthly factors (per month).
      [0x2022] HAC (Newey-West) Sharpe on monthly *simple* returns (12/yr).
      [0x2022] PSR (probability SR > 0) and a cross-model DSR-like deflation on the SR vector.
      [0x2022] Calmar from annualized return / max drawdown on the GM equity curve.

    Returns a DataFrame with one row per model_type and columns:
      ['rank','model','months','trades','active','SR','PSR','DSR','Calmar',
       'AnnRet','FinalEq','DA','Prec','F1','Profit/Hit','LabelThr','EffConf','lags']
    """
    import numpy as np, pandas as pd

    if combined_df is None or len(combined_df) == 0:
        return pd.DataFrame()

    df = combined_df.copy()

    # Ensure canonical dtypes
    num_cols = ["strategy_return","trades","active_rate","directional_accuracy",
                "precision_macro","f1_macro","profit_per_hit",
                "label_threshold","confidence_threshold","lags",
                "cstrategy","creturns"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pivots built from geometric mean per month across repeats
    try:
        equity_pivot, returns_pivot, bh_equity = build_model_monthly_pivots(df)
    except Exception:
        # Fail-safe: construct simple pivots without GM
        returns_pivot = (df.pivot_table(index="test_end", columns="model_type",
                                        values="strategy_return", aggfunc="mean"))
        equity_pivot = (df.pivot_table(index="test_end", columns="model_type",
                                       values="cstrategy", aggfunc="mean")).cumprod()

    # Collect per-model stats
    rows = []
    models = sorted(df["model_type"].dropna().unique())
    for m in models:
        sub = df[df["model_type"] == m].copy()
        months = int(pd.to_numeric(sub["strategy_return"], errors="coerce").notna().sum())

        if months < int(min_months):
            continue

        # Monthly simple returns vector (across repeats)
        r = pd.to_numeric(sub["strategy_return"], errors="coerce").dropna()
        # HAC Sharpe (Newey-West) -> sr_hat annualized (12/yr)
        if len(r) >= 3:
            mu = float(r.mean())
            sd_hac = float(hac_std(r.values, max_lag="andrews"))
            sr = (mu / sd_hac) * (12.0 ** 0.5) if sd_hac > 0 else float("nan")
            psr = probabilistic_sharpe_ratio(r.values, sr_benchmark=0.0, periods_per_year=12)
        else:
            sr, psr = float("nan"), float("nan")

        # Final equity & annualized return from GM equity curve
        eq = equity_pivot.get(m)
        final_eq = float(eq.iloc[-1]) if isinstance(eq, pd.Series) and len(eq) else float("nan")
        # Annualized from average monthly log-return on GM factors
        try:
            gm = returns_pivot.get(m)  # monthly simple returns from GM factors - 1
            ann_ret = float((1.0 + gm.fillna(0.0).mean())**12 - 1.0) if gm is not None else float("nan")
        except Exception:
            ann_ret = float("nan")

        # Calmar on GM equity curve
        maxdd = _max_drawdown_from_equity(eq) if isinstance(eq, pd.Series) else float("nan")
        calmar = (ann_ret / maxdd) if (isinstance(maxdd, float) and maxdd > 0) else float("nan")

        # Aggregates
        trades = int(pd.to_numeric(sub["trades"], errors="coerce").sum())
        active = float(pd.to_numeric(sub["active_rate"], errors="coerce").mean())
        da     = float(pd.to_numeric(sub["directional_accuracy"], errors="coerce").mean())
        prec   = float(pd.to_numeric(sub["precision_macro"], errors="coerce").mean())
        f1     = float(pd.to_numeric(sub["f1_macro"], errors="coerce").mean())
        pph    = float(pd.to_numeric(sub["profit_per_hit"], errors="coerce").mean())

        # Typical config stats (mode/median)
        label_thr = float(pd.to_numeric(sub["label_threshold"], errors="coerce").median()) if "label_threshold" in sub else np.nan
        eff_conf  = np.nan
        if "confidence_threshold" in sub:
            eff_conf = float(pd.to_numeric(sub["confidence_threshold"], errors="coerce").median())
        # Backward-compat: older monthly CSVs stored the effective threshold under
        # confidence_threshold_used; fall back if canonical is missing/NaN.
        if (not np.isfinite(eff_conf)) and ("confidence_threshold_used" in sub):
            eff_conf = float(pd.to_numeric(sub["confidence_threshold_used"], errors="coerce").median())
        lags_mode = _mode_safe(sub["lags"]) if "lags" in sub else None

        rows.append({
            "model": m, "months": months, "trades": trades, "active": active,
            "SR": sr, "PSR": psr, "Calmar": calmar, "AnnRet": ann_ret, "FinalEq": final_eq,
            "DA": da, "Prec": prec, "F1": f1, "Profit/Hit": pph,
            "LabelThr": label_thr, "EffConf": eff_conf, "lags": lags_mode,
        })

    rank_df = pd.DataFrame(rows)

    if rank_df.empty:
        return rank_df

    # DSR-like deflation across models using the SR vector
    try:
        from utilsNoWFO import compute_dsr_scores as _compute_dsr_scores  # self-import safe
    except Exception:
        _compute_dsr_scores = None
    if _compute_dsr_scores is not None:
        srs = [float(v) if v == v else float("-inf") for v in rank_df["SR"].tolist()]
        dsr = _compute_dsr_scores(srs)
        rank_df["DSR"] = dsr
    else:
        rank_df["DSR"] = float("nan")

    # Sort: DSR desc -> PSR desc -> SR desc -> FinalEq desc -> Calmar desc
    rank_df = rank_df.sort_values(
        by=["DSR","PSR","SR","FinalEq","Calmar"], ascending=[False]*5
    ).reset_index(drop=True)
    rank_df.insert(0, "rank", rank_df.index + 1)

    return rank_df

def save_model_ranking_csv(df_rank, out_dir, filename="model_ranking_final.csv"):
    """Save ranking CSV next to other 'All/csv' artifacts."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df_rank.to_csv(path, index=False)
    return path


def _set_even_time_ticks(ax, idx, n_parts=10, fmt=None, rotation=None):
    """
    Put exactly n_parts+1 evenly spaced ticks from first->last timestamp in idx.

    Parameters
    ----------
    ax : matplotlib axis
    idx : DatetimeIndex / sequence of timestamps
    n_parts : int
        Number of equal *segments* (gives n_parts+1 ticks).
    fmt : str or None
        Optional strftime format for tick labels. If None, choose based on span.
    rotation : float or None
        Optional rotation angle (degrees) for tick labels.
    """
    import pandas as pd
    import matplotlib.dates as mdates
    from matplotlib.ticker import FixedLocator

    if idx is None:
        return
    dt = pd.to_datetime(idx, utc=True, errors="coerce")
    if len(dt) == 0:
        return
    if getattr(dt, "tz", None) is not None:
        dt = dt.tz_convert(None)

    start, end = dt[0], dt[-1]
    if pd.isna(start) or pd.isna(end) or start == end:
        return

    ticks = pd.date_range(start, end, periods=int(n_parts) + 1)

    # Choose default format based on span if no override is given
    if fmt is None:
        span = end - start
        span_days = span.days + span.seconds / 86400.0
        if span_days >= 365:
            fmt = "%Y-%b"
        elif span_days >= 90:
            fmt = "%Y-%m"
        elif span_days >= 14:
            fmt = "%Y-%m-%d"
        elif span_days >= 1:
            fmt = "%m-%d %H:%M"
        else:
            fmt = "%H:%M"

    ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(ticks)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

    # Label rotation / alignment
    rot = 0 if rotation is None else rotation
    ha = "center" if rot == 0 else "right"
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rot)
        lbl.set_ha(ha)
        lbl.set_fontsize(9)


# --- helper: compute drawdown series -----------------------------------------
def _compute_drawdown(equity: "pd.Series") -> "pd.Series":
    """
    Return drawdown series (in fraction, negative values) from an equity curve ([0xd7]).
    """
    import pandas as pd, numpy as np
    if equity is None or len(equity) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(equity, errors="coerce"), index=pd.to_datetime(equity.index))
    s = s.replace([np.inf, -np.inf], np.nan).ffill()
    peak = s.cummax()
    dd = s / peak - 1.0
    return dd


def build_trade_log_from_df(df, bar_minutes=None):
    """
    Build a per-trade log from a per-bar results DataFrame.

    Expects df to contain:
      - index: datetime-like (bar timestamps),
      - 'position': net position (sign gives direction),
      - 'strategy': per-bar log-return (or simple return) of the strategy.

    Returns
    -------
    DataFrame with one row per trade:
      trade_id, entry_time, exit_time, side, side_sign,
      entry_bar, exit_bar, bars_held, holding_minutes,
      gross_log_return, pnl_pct
    """
    import numpy as np
    import pandas as pd

    cols = [
        "trade_id",
        "entry_time",
        "exit_time",
        "side",
        "side_sign",
        "entry_bar",
        "exit_bar",
        "bars_held",
        "holding_minutes",
        "gross_log_return",
        "pnl_pct",
    ]

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)

    if "strategy" not in df.columns:
        return pd.DataFrame(columns=cols)

    # Some evaluation routes store the executed position under 'position_exec'.
    pos_col = None
    for _c in ("position", "position_exec", "pos_exec", "pos"):
        if _c in df.columns:
            pos_col = _c
            break
    if pos_col is None:
        return pd.DataFrame(columns=cols)
 

    # Normalize inputs
    pos = np.sign(pd.to_numeric(df[pos_col]).fillna(0.0).values.astype(float))
    strat = pd.to_numeric(df["strategy"]).fillna(0.0).values.astype(float)
    idx = pd.to_datetime(df.index)

    # Infer bar length in minutes if not supplied
    if bar_minutes is None:
        try:
            if len(idx) >= 2:
                delta_min = (idx[1] - idx[0]).total_seconds() / 60.0
                bar_minutes = max(1, int(round(delta_min)))
            else:
                bar_minutes = 1
        except Exception:
            bar_minutes = 1

    trades = []
    current_side = 0.0
    entry_i = None

    def close_trade(exit_i: int):
        nonlocal current_side, entry_i
        if entry_i is None:
            return
        if exit_i < entry_i:
            exit_i = entry_i

        log_ret = float(np.nansum(strat[entry_i : exit_i + 1]))
        pnl_pct = float(np.exp(log_ret) - 1.0)
        bars_held = int(exit_i - entry_i + 1)

        trades.append(
            {
                "trade_id": len(trades),
                "entry_time": idx[entry_i],
                "exit_time": idx[exit_i],
                "side": "long" if current_side > 0 else "short",
                "side_sign": float(current_side),
                "entry_bar": int(entry_i),
                "exit_bar": int(exit_i),
                "bars_held": bars_held,
                "holding_minutes": int(bars_held * bar_minutes),
                "gross_log_return": log_ret,
                "pnl_pct": pnl_pct,
            }
        )
        current_side = 0.0
        entry_i = None

    # Walk the position series and detect opens / closes / flips
    for i, side in enumerate(pos):
        if current_side == 0.0 and side != 0.0:
            # Opening a new trade
            current_side = side
            entry_i = i
        elif current_side != 0.0:
            if side == 0.0:
                # Closing into flat -> close using this bar
                close_trade(i)
            elif side != current_side:
                # Flip: close old trade at i-1, open new one at i
                close_trade(i - 1)
                current_side = side
                entry_i = i
            else:
                # Same side, keep trade open
                pass

    # If still in a trade at the end, close at the last bar
    if current_side != 0.0 and entry_i is not None:
        close_trade(len(df) - 1)

    if not trades:
        return pd.DataFrame(columns=cols)

    tdf = pd.DataFrame(trades)
    return tdf[cols]


def save_monthly_model_stats(
    df_stats,
    out_dir,
    filename="monthly_model_stats.csv",
    dedup_keys=("model_type", "test_start", "test_end"),
):
    """
    Append/merge a model's monthly stats into a single master CSV:
      <run>/All/csv/monthly_model_stats.csv

    - Unifies columns with any existing file (union of columns).
    - Drops duplicates on dedup_keys (keep latest), if those keys exist.
    - Normalizes test_start/test_end to tz-naive for stable equality checks.
    """
    import os
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    df_new = df_stats.copy()

    # Ensure model_type exists so rows can be grouped later
    if "model_type" not in df_new.columns:
        df_new["model_type"] = "unknown"

    # Normalize dates for stable dedup
    for col in ("test_start", "test_end"):
        if col in df_new.columns:
            df_new[col] = pd.to_datetime(df_new[col], utc=True, errors="coerce").dt.tz_convert(None)

    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
        except Exception:
            df_old = pd.DataFrame()
        # union columns, align, concat
        all_cols = sorted(set(df_old.columns).union(df_new.columns))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # drop duplicates on keys that are present
        keys = [k for k in dedup_keys if k in df_all.columns]
        if keys:
            df_all = df_all.drop_duplicates(subset=keys, keep="last")
        df_all.to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)

    return path

def _coalesce_bh_series(bh_candidates):
    """
    Given a dict {model: Series} for BH curves, return a single BH Series.
    Pick the longest one and combine_first the others.
    """
    import pandas as pd

    if not bh_candidates:
        return None

    # pick the BH series with most non-NaN points as the "spine"
    winner_key = max(bh_candidates, key=lambda k: bh_candidates[k].dropna().shape[0])
    bh = bh_candidates[winner_key].copy()

    for k, s in bh_candidates.items():
        if k == winner_key:
            continue
        bh = bh.combine_first(s)

    return bh.sort_index()


def build_model_bar_compare_df(bt_dict, models=None):
    """
    Build a per-bar comparison DataFrame across models + Buy&Hold from a dict of
    MLBacktester objects.

    Each backtester must have .bar_concat with columns
        ['cstrategy_cont', 'creturns_cont']
    indexed by timestamps.

    Returns columns:
        ['BH'] + [f'{model}_equity' for model in models]
    """
    import pandas as pd

    if models is None:
        models = list(bt_dict.keys())
    models = [m for m in models if m in bt_dict]

    cols = {}
    bh_candidates = {}

    for m in models:
        bt = bt_dict[m]
        df = getattr(bt, "bar_concat", None)
        if df is None or df.empty:
            continue

        # we only need continuous strategy & BH curves
        if not {"cstrategy_cont", "creturns_cont"} <= set(df.columns):
            continue

        dfi = df[["cstrategy_cont", "creturns_cont"]].copy()
        dfi.index = pd.to_datetime(dfi.index, utc=True, errors="coerce")
        dfi = dfi.sort_index()

        cols[f"{m}_equity"] = dfi["cstrategy_cont"]
        bh_candidates[m] = dfi["creturns_cont"]

    if not cols:
        return pd.DataFrame()

    combined = pd.concat(cols, axis=1).sort_index()

    # Insert a single BH baseline (coalesced across models)
    bh = _coalesce_bh_series(bh_candidates)
    if bh is not None:
        combined.insert(0, "BH", bh.reindex(combined.index).combine_first(bh))

    return combined


def save_model_bar_comparison_outputs(
    bt_dict,
    models=None,
    out_prefix="results/model_bar_compare",
    style="nature",
    palette="okabe_ito_no_black",
    bh_color="#666666",
    y_anchor=1.0,
    n_time_parts=10,
    dpi=300,
    line_width=1.0,
    # flexible output routing
    out_dir=None,
    csv_dir=None,
    png_dir=None,
    overlap_mode="intersection",      # 'intersection' (default) or 'union_rebase'
    also_save_intersection=False,     # kept for API backwards-compat; unused here
    annotate_coverage=True,
    save_csv=True,
):
    """
    Per-bar comparison across models + BH.

    overlap_mode:
      - 'union_rebase': rebase to first valid per series, union index, ffill
      - 'intersection': strict overlap only (drop rows where any series is NaN)
      
      rebase_to_first:
      If True (default), each series is normalized to start at 1.0 (divide by its first valid value).
      If False, plots use absolute equity levels as provided (e.g., continuous carry-forward equity)
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

    if not bt_dict:
        print("[WARN][0xfe0f] Empty bt_dict passed to save_model_bar_comparison_outputs.")
        return None

    # ---- build comparison DataFrame -----------------------------------------
    def _build_df_from_bt_dict(d, models_filter=None):
        # If user already passed a dict of Series/DataFrames, respect it
        if all(hasattr(v, "index") and not isinstance(v, (dict, list)) for v in d.values()):
            return pd.DataFrame(d)

        # Otherwise treat it as {model: MLBacktester} and use the helper above
        return build_model_bar_compare_df(d, models=models_filter)

    # ------------------------------------------------------------------ df ---
    df = _build_df_from_bt_dict(bt_dict, models_filter=models)
    if df is None or df.empty:
        print("[WARN][0xfe0f] No data to plot (per-bar comparison).")
        return None

    # Filter to requested models (keep BH if present)
    if models:
        wanted = []
        for m in models:
            col = f"{m}_equity" if m != "BH" and (f"{m}_equity" in df.columns) else m
            if col in df.columns:
                wanted.append(col)
        if wanted:
            keep = (["BH"] if "BH" in df.columns else []) + wanted
            df = df.loc[:, [c for c in keep if c in df.columns]]
        else:
            print(f"[WARN][0xfe0f] None of the requested models found for per-bar plot: {models}")
            return None

    # Ensure datetime index & sorted
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    df = df.sort_index()

    # For plotting only: extend index back to the 1st of the month and
    # draw models as neutral (flat) before the first real bar.
    df = _extend_index_to_calendar_start(df)

    # Rebase ONLY when explicitly requested (matches intent of overlap_mode)
    # - union_rebase: normalize each series to start at 1.0 (divide by first valid)
    # - intersection: keep absolute equity levels as produced by the engine
    if overlap_mode == "union_rebase":
        for c in df.columns:
            s = df[c]
            first = s.dropna().iloc[0] if s.dropna().size else np.nan
            if np.isfinite(first) and first != 0.0:
                df[c] = s / first

    # Make all models neutral before their first trade so curves
    # start on the first day rather than at first trade.
    df = _neutral_fill_before_first_trade(df, skip_cols=None)

    if overlap_mode == "intersection":
        df_plot = df.dropna(how="any")
    else:
        df_plot = df.replace([np.inf, -np.inf], np.nan).ffill()

    if df_plot.empty:
        print("[WARN][0xfe0f] No overlapping data for per-bar plot.")
        return None

    # ---- output paths (respect <run>/<Model>/graphs/) -----------------------
    if (csv_dir is not None) or (png_dir is not None):
        if csv_dir is None:
            csv_dir = png_dir
        if png_dir is None:
            png_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "model_bar_compare.csv")
        png_path = os.path.join(png_dir, "model_bar_compare_bars.png")
    elif out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "model_bar_compare.csv")
        png_path = os.path.join(out_dir, "model_bar_compare_bars.png")
    else:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        csv_path = f"{out_prefix}.csv"
        png_path = f"{out_prefix}_bars.png"

    # CSV of the rebased values
    if save_csv:
        try:
            df_plot.to_csv(csv_path, index=True, float_format="%.10f")
        except Exception as e:
            print(f"[WARN][0xfe0f] Could not save model comparison CSV: {e}")

    # === NEW: (optional future use) derived paths for underwater/rolling ===
    base_png, ext = os.path.splitext(png_path)
    underwater_png_path = f"{base_png}_underwater{ext}"
    rolling_sharpe_png_path = f"{base_png}_rolling_sharpe{ext}"
    _ = (underwater_png_path, rolling_sharpe_png_path)  # silence linters

    # ---- plot ---------------------------------------------------------------
    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, ax = plt.subplots(constrained_layout=True)

        # Coverage labels with *short* model names
        labels = []
        for col in df.columns:
            cov = 100.0 * (df[col].dropna().shape[0] / max(1, df.index.shape[0]))
            base_pretty = _pretty_bar_label_global(col)
            base_short = _short_model_label(base_pretty)

            if col == "BH":
                lbl = base_short  # usually "BH"
            else:
                lbl = base_short
                if annotate_coverage and cov < 100.0:
                    lbl = f"{base_short} ({cov:.0f}% overlap)"

            labels.append((col, lbl))

        # --- Plot BH first if present ----------------------------------------
        if "BH" in df_plot.columns:
            ax.plot(
                df_plot.index,
                df_plot["BH"].astype(float).values,
                linestyle="--",
                linewidth=line_width,
                color=bh_color,
                label="Buy & Hold",
                zorder=2,
            )

        # --- Plot models ------------------------------------------------------
        for col, lbl in labels:
            if col == "BH" or col not in df_plot.columns:
                continue

            color = ax._get_lines.get_next_color()

            ax.plot(
                df_plot.index,
                df_plot[col].astype(float).values,
                linewidth=line_width,
                color=color,
                label=lbl,
                zorder=3,
            )

        ax.set_title("Per-bar Cumulative Equity (Intersection)", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Equity ([0xd7] start capital)")
        ax.grid(True)

        # 10 equal segments, dates like '03/03/25'
        try:
            _set_even_time_ticks(
                ax,
                df_plot.index,
                n_parts=n_time_parts,
                fmt="%d/%m/%y",
                rotation=30,
            )
        except Exception:
            pass

        ax.margins(x=0)

        # Legend on the right, like the thesis example figure.
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            fig.subplots_adjust(right=0.80)
            ax.legend(
                handles,
                labels_,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

        fig.set_size_inches(11.0, 4.8)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return png_path


# ---------------------------------------------------------------------------
# Helper: short, thesis-friendly model labels for bottom bands
# ---------------------------------------------------------------------------
def _short_model_label(name: str) -> str:
    """
    Map long internal model names to short labels for figure band captions.
    This does not affect any logic - only the text shown on the plots.
    """
    if not isinstance(name, str):
        return str(name)

    # Normalise a bit
    key = name.lower().replace(" ", "_")

    mapping = {
        "logistic": "log",
        "logit": "log",
        "logistic_regression": "log",
        "svm": "svm",
        "svc": "svm",
        "random_forest": "rf",
        "rf": "rf",
        "xgboost": "xgb",
        "xgb": "xgb",
        "cnn": "cnn",
        "lstm": "lstm",
        "transformer": "trf",
        "ensemble_cnn_lstm_xgboost": "ens_cnn",
        "ensemble_local_global": "ens_cnn",
        "ensemble_adaptive_regime": "ens_adap",
        "adaptive_regime": "ens_adap",
        "ensemble_transformer_xgb_dqn": "ens_trf",
        "transformer_xgb_dqn": "ens_trf",
    }

    return mapping.get(key, name)


def apply_academic_style(
    theme: str = "nature",
    palette: str = "okabe_ito_no_black",
    bw_line_styles: bool = False,
):
    """
    Global 'academic' style context for all thesis plots.

    - Sets fonts, grid, spines, tick sizes, and figure size.
    - Uses a palette name to select color cycles.
    - Returns a context manager that you use with 'with ...:'.
    """
    import matplotlib as mpl
    from contextlib import contextmanager

    # --- Choose base colors from your existing palette registry -------------
    # If you already have ACADEMIC_THEMES / PALETTES, plug them here.
    # For now we'll use Matplotlib default cycle if we can't resolve it.
    try:
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    except Exception:
        colors = None

    # Base rc settings for "paper" style
    rc = {
        # figure
        "figure.figsize": (11.0, 4.8),
        "figure.dpi": 100,
        "figure.constrained_layout.use": True,

        # fonts
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        # axes & grid
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,

        # spines
        "axes.spines.top": False,
        "axes.spines.right": False,

        # lines
        "lines.linewidth": 1.8,
        "lines.markersize": 4,

        # ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        # legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    }

    # Optional: tweak for B/W printing (more dashes etc.)
    if bw_line_styles:
        rc["axes.prop_cycle"] = mpl.cycler(
            "linestyle",
            ["-", "--", "-.", ":"]
        )

    @contextmanager
    def _style_context():
        with mpl.rc_context(rc):
            if colors is not None and not bw_line_styles:
                # Set color cycle only if we have one and we're not in BW mode
                mpl.rcParams["axes.prop_cycle"] = mpl.cycler("color", colors)
            yield

    return _style_context()


# ===========================
# Optuna progress saver  (PNG only; no CSV; Sharpe sign fixed)
# ===========================
def save_optuna_progress_from_study(
    study, out_prefix, metric_name="Sharpe",
    penalty_value=-9999.0, style="nature", palette="okabe_ito_no_black",
    include_pruned=True, plot_pruned=True
):
    """
    Plot Optuna optimization progress (TRUE Sharpe by trial) to a single PNG.
    - No sign inversion anywhere.
    - Excludes penalty_value points.
    - Dashed line shows running best-so-far (maximize).
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    complete_rows, pruned_rows = [], []

    for t in study.trials:
        state = getattr(t.state, "name", str(t.state))
        if state == "COMPLETE" and t.value is not None and t.value != penalty_value:
            complete_rows.append({"trial_number": t.number, "metric": float(t.value)})
        elif include_pruned and state == "PRUNED":
            ivals = getattr(t, "intermediate_values", {}) or {}
            if ivals:
                steps_vals = sorted(ivals.items())
                vals_clean = [v for _, v in steps_vals if v is not None and v != penalty_value]
                if vals_clean:
                    # TRUE sign (no negation)
                    pruned_rows.append({
                        "trial_number": t.number,
                        "best_intermediate_metric": float(max(vals_clean))
                    })

    if not complete_rows and not pruned_rows:
        print("[WARN][0xfe0f] No eligible trials to plot.")
        return None, None

    df_c = pd.DataFrame(complete_rows)
    if not df_c.empty and "trial_number" in df_c.columns:
        df_c = df_c.sort_values("trial_number")

    df_p = pd.DataFrame(pruned_rows)
    if not df_p.empty and "trial_number" in df_p.columns:
        df_p = df_p.sort_values("trial_number")

    # --- Plot with unified thesis style --------------------------------------
    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, ax = plt.subplots(constrained_layout=True)

        # Completed trials: metric per trial + best-so-far line
        if not df_c.empty:
            df_c["best_so_far"] = df_c["metric"].cummax()
            ax.plot(
                df_c["trial_number"],
                df_c["metric"],
                marker="o",
                linewidth=1.8,
                label=f"{metric_name} per trial",
                zorder=2,
            )
            ax.plot(
                df_c["trial_number"],
                df_c["best_so_far"],
                linestyle="--",
                linewidth=2.0,
                label="Best so far",
                zorder=3,
            )

        # Pruned trials (optional)
        if plot_pruned and not df_p.empty:
            ax.scatter(
                df_p["trial_number"],
                df_p["best_intermediate_metric"],
                marker="x",
                alpha=0.7,
                linewidths=1.2,
                label="Pruned (best interim)",
                zorder=1,
            )

        # Labels & title
        ax.set_xlabel("Trial")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Optuna Optimization Progress -- {metric_name}")
        ax.grid(True)

        # Legend with subtle frame, consistent with other plots
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.9,
        )

        # y-range padding based on whatever we actually plotted
        ys = []
        if not df_c.empty:
            ys += df_c["metric"].tolist()
        if plot_pruned and not df_p.empty:
            ys += df_p["best_intermediate_metric"].tolist()
        if ys:
            ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

        ax.margins(x=0)

        out_png = f"{out_prefix}_progress.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return None, out_png


# --- NEW: small convenience wrapper for all thesis plots --------------------
PAPER_STYLE_DEFAULTS = {
    "style": "nature",              # main theme used in the thesis
    "palette": "okabe_ito_no_black" # color-blind friendly + neutral BH grey
}

PAPER_STYLE_DEFAULTS = {
    "style": "nature",              # your main theme
    "palette": "okabe_ito_no_black" # colorblind-friendly + neutral BH grey
}

def set_paper_style(
    style: str | None = None,
    palette: str | None = None,
    bw_line_styles: bool = False,
):
    """
    Convenience wrapper around apply_academic_style() so all figures share
    the same 'thesis style'.

    - If style/palette are None, use PAPER_STYLE_DEFAULTS.
    - bw_line_styles=True forces different dash patterns for B/W printing.
    """
    style = style or PAPER_STYLE_DEFAULTS["style"]
    palette = palette or PAPER_STYLE_DEFAULTS["palette"]
    return apply_academic_style(
        style,
        palette,
        bw_line_styles=bw_line_styles,
    )


def _features_from_params_names_only(params, base_features):
    """
    Reconstruct feature names from a flat params dict + base_features.
    Keeps indicators off when use_extended_features is False.
    """
    feats = []

    def add(name):
        if name not in feats:
            feats.append(name)

    def _coerce_int_safe(x, default):
        try:
            return int(float(x))
        except Exception:
            return int(default)

    use_ext = bool(params.get("use_extended_features", True))

    # ---- Robustly parse indicator_windows (dict, JSON str, None, NaN, etc.) ----
    ind_src = params.get("indicator_windows", {})
    ind = {}
    if isinstance(ind_src, dict):
        ind = dict(ind_src)
    elif isinstance(ind_src, str) and ind_src.strip():
        try:
            import json as _json
            parsed = _json.loads(ind_src)
            if isinstance(parsed, dict):
                ind = parsed
        except Exception:
            ind = {}
    # else: keep {}

    # Merge any flat window fields into 'ind'
    if "rsi_window" in params:          ind["rsi"]         = _coerce_int_safe(params["rsi_window"], 14)
    if "bb_window"  in params:          ind["bb"]          = _coerce_int_safe(params["bb_window"], 20)
    if "atr_window" in params:          ind["atr"]         = _coerce_int_safe(params["atr_window"], 14)
    if "adx_window" in params:          ind["adx"]         = _coerce_int_safe(params["adx_window"], 14)
    if "stoch_k_window" in params:      ind["stoch"]       = _coerce_int_safe(params["stoch_k_window"], 14)
    if "mtf_ma_fast_window" in params:  ind["mtf_ma_fast"] = _coerce_int_safe(params["mtf_ma_fast_window"], 10)
    if "mtf_ma_slow_window" in params:  ind["mtf_ma_slow"] = _coerce_int_safe(params["mtf_ma_slow_window"], 30)

    # Toggles (only active if extended features are enabled)
    use_sma    = bool(params.get("use_sma", False))   and use_ext
    use_ema    = bool(params.get("use_ema", False))   and use_ext
    use_macd   = bool(params.get("use_macd", False))  and use_ext
    use_rsi    = ("rsi" in ind)                       and use_ext
    use_bbands = ("bb" in ind)                        and use_ext
    use_atr    = ("atr" in ind)                       and use_ext
    use_adx    = ("adx" in ind)                       and use_ext
    use_stoch  = ("stoch" in ind)                     and use_ext
    use_mtf    = (("mtf_ma_fast" in ind) or ("mtf_ma_slow" in ind)) and use_ext

    # Seed indicator feature names
    if use_sma and "sma" in ind: add(f"sma_{int(ind['sma'])}")
    if use_ema and "ema" in ind: add(f"ema_{int(ind['ema'])}")
    if use_macd:
        for c in ["macd_line", "macd_signal", "macd_diff"]:
            add(c)
    if use_rsi:    add(f"rsi_{int(ind['rsi'])}")
    if use_bbands:
        for c in ["bb_upper", "bb_lower", "bb_pct", "bbw"]:
            add(c)
    if use_atr:    add(f"atr_{int(ind['atr'])}")
    if use_adx:    add(f"adx_{int(ind['adx'])}")
    if use_stoch:
        for c in ["stoch_k", "stoch_d"]:
            add(c)
    if use_mtf:
        for c in ["mtf_ma_fast", "mtf_ma_slow"]:
            add(c)

    # Base features always included
    for bf in base_features:
        add(bf)

    # ---- Normalize roll_windows / lag_depth / num_lags BEFORE expansion ----
    rw = params.get("roll_windows", [5])
    if isinstance(rw, (list, tuple)):
        roll_windows = [_coerce_int_safe(w, 5) for w in rw]
    elif rw is None:
        roll_windows = [5]
    else:
        roll_windows = [_coerce_int_safe(rw, 5)]

    try:
        lag_depth = _coerce_int_safe(params.get("lag_depth", 1), 1)
    except Exception:
        lag_depth = 1

    num_lags = params.get("lags", params.get("lags_range", 1))
    num_lags = _coerce_int_safe(num_lags, 1)

    # ---- Expand lags and rolling transforms for all seeds ----
    seeds = list(feats)
    for feat in seeds:
        for k in range(1, lag_depth + 1):
            add(f"{feat}_lag{k}")
        for w in roll_windows:
            for rt in ["rollmean", "rollstd", "rollslope"]:
                add(f"{feat}_{rt}{int(w)}")

    # Raw returns lags (if enabled)
    if params.get("include_raw_lags", True):
        for k in range(1, num_lags + 1):
            add(f"returns_lag{k}")

    return feats

def save_feature_frequency_from_trials(
    study_or_trials,
    base_features,
    out_png="results/feature_frequency_trials.png",
    top_k=30,
    top_percent=0.2,          # keep top 20% trials by score (after sign fix)
    weight_by_score=True,     # weight counts by normalized score
    minimize_objective=True,  # set True if objective is -Sharpe (we need to flip sign)
    style="nature",
    palette="okabe_ito_no_black",
    exclude_prefixes=("returns_lag", "hour"),
    exclude_regex=None,
    collapse_raw_lags=True
):
    """
    Build a feature-frequency heatmap over OPTUNA TRIALS, not monthly winners.
    - Filters to top `top_percent` by score per *entire run* (you can also adapt per-month).
    - Optionally weights feature hits by normalized score.
    - No CSVs written.
    """
    import re, numpy as np, pandas as pd, matplotlib.pyplot as plt

    def _keep_feature(name: str) -> bool:
        if exclude_prefixes and any(name.startswith(p) for p in exclude_prefixes):
            return False
        if exclude_regex and re.search(exclude_regex, name):
            return False
        return True

    def _normalize_name(name: str) -> str:
        if collapse_raw_lags and name.startswith("returns_lag"):
            return "returns_lag*"
        return name

    # Collect trials list
    try:
        import optuna
        if isinstance(study_or_trials, optuna.study.Study):
            trials = study_or_trials.trials
        else:
            trials = list(study_or_trials)
    except Exception:
        trials = list(study_or_trials)

    # Scores (flip sign if needed)
    scores = []
    completed = []
    for t in trials:
        if getattr(t, "state", None).name == "COMPLETE" and t.value is not None:
            s = float(t.value)
            if minimize_objective:
                s = -s
            scores.append(s)
            completed.append(t)

    if not completed:
        print("[WARN][0xfe0f] No COMPLETE trials with scores.")
        return None

    scores = np.array(scores, dtype=float)
    # Keep top X%
    k = max(1, int(np.ceil(len(scores) * float(top_percent))))
    thresh = np.partition(scores, -k)[-k]
    keep_mask = scores >= thresh
    kept = [t for t, m in zip(completed, keep_mask) if m]
    kept_scores = scores[keep_mask]
    # Normalize weights 0..1
    if weight_by_score and kept_scores.max() > kept_scores.min():
        weights = (kept_scores - kept_scores.min()) / (kept_scores.max() - kept_scores.min())
    else:
        weights = np.ones_like(kept_scores)

    rows = []
    for t, w in zip(kept, weights):
        params = dict(getattr(t, "params", {}) or {})
        feats = _features_from_params_names_only(params, list(base_features))
        for f in feats or []:
            if _keep_feature(f):
                rows.append({"feature": _normalize_name(f), "w": float(w)})

    # if not rows:
    #     print("[WARN][0xfe0f] No features found in selected trials.")
    #     return None

    df = pd.DataFrame(rows)
    freq = (df.groupby("feature")["w"].sum()
              .rename("weight").reset_index()
              .sort_values(["weight","feature"], ascending=[False, True]))
    top = freq.head(int(top_k))
    vals = top["weight"].to_numpy().reshape(-1, 1)

    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, ax = plt.subplots()
        im = ax.imshow(vals, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"])
        ax.set_xticks([0]); ax.set_xticklabels(["Weighted freq"])
        ax.set_title(f"Feature Usage Across Trials (Top {int(top_percent*100)}% by score)")
        for j, v in enumerate(vals[:, 0]):
            ax.text(0, j, f"{v:.2f}", ha="center", va="center", fontsize=9)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("Weighted frequency")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

    return out_png


def save_feature_frequency_from_monthly_results(
    df_or_csv,
    base_features,
    out_png="results/feature_frequency_monthly.png",
    top_k=30,
    top_percent=1.0,
    weight_by_score=False,
    minimize_objective=False,
    style="nature",
    palette="okabe_ito_no_black",
    exclude_prefixes=("returns_lag", "hour"),
    exclude_regex=None,
    collapse_raw_lags=True,
    out_csv=None,
    **kwargs
):
    """
    Monthly winners' feature-frequency heatmap/bar with CSV parity.
    Accepts a DataFrame or a CSV path produced by log_simulation_result().
    Saves a PNG and, if out_csv is provided, the matrix used for plotting.

    Notes:
    - Robust for single-month runs (keeps input strictly 2D for imshow).
    - If top_percent < 1.0, it limits the candidate feature pool before top_k.
    - Gracefully skips plotting if no features survive filtering.
    """
    import os, re, math, itertools
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    def _keep_feature(name: str) -> bool:
        if exclude_prefixes and any(name.startswith(p) for p in exclude_prefixes):
            return False
        if exclude_regex and re.search(exclude_regex, name):
            return False
        return True

    def _normalize_name(name: str) -> str:
        if collapse_raw_lags and name.startswith("returns_lag"):
            return "returns_lag*"
        return name

    # Load input
    df = pd.read_csv(df_or_csv) if isinstance(df_or_csv, str) else (df_or_csv.copy() if df_or_csv is not None else None)
    if df is None or getattr(df, "empty", True):
        print("[WARN][0xfe0f] No data for feature-frequency heatmap.")
        return None

    # Collect features per month (expects 'features_used' column from log_simulation_result)
    feats_per_month = []
    for _, row in df.iterrows():
        feats = row.get("features_used", None)

        if isinstance(feats, str):
            try:
                feats = eval(feats)  # produced by our logger; safe here
            except Exception:
                feats = [feats]
        else:
            import numpy as np
            import pandas as pd
            # Treat NaN / None / scalar as "no features"
            if feats is None or (isinstance(feats, float) and np.isnan(feats)):
                feats = []
            elif isinstance(feats, (list, tuple, set, np.ndarray, pd.Series)):
                feats = list(feats)
            else:
                # Any other weird scalar -> ignore
                feats = []

        feats = feats or []
        feats = [_normalize_name(str(f)) for f in feats if _keep_feature(str(f))]
        feats_per_month.append(feats)


    # All unique features that survived filters
    all_feats = sorted(set(itertools.chain.from_iterable(feats_per_month)))
    if not all_feats:
        # Save empty CSV if requested, then exit gracefully
        if out_csv:
            try:
                pd.DataFrame().to_csv(out_csv, index=False)
            except Exception:
                pass
        print("[WARN][0xfe0f] No features survived filtering; nothing to plot.")
        return None

    # Build month x feature count matrix
    counts = [[feats.count(f) for f in all_feats] for feats in feats_per_month]
    mat = pd.DataFrame(counts, columns=all_feats)

    # Persist full matrix (before top_k/top_percent)
    if out_csv:
        try:
            mat.to_csv(out_csv, index=False)
        except Exception as e:
            print(f"[WARN][0xfe0f] Could not write feature-frequency CSV: {e}")

    # Rank features by total usage
    totals = mat.sum(axis=0).sort_values(ascending=False)

    # Apply top_percent first (if < 1.0), then cap by top_k
    if 0 < float(top_percent) < 1.0:
        max_cols = max(1, math.ceil(len(totals) * float(top_percent)))
    else:
        max_cols = len(totals)
    keep_n = min(top_k, max_cols, len(totals))
    keep_cols = list(totals.index[:keep_n])
    mat_plot = mat.loc[:, keep_cols]

    if mat_plot.shape[1] == 0:
        print("[WARN][0xfe0f] No features to plot after top_k/top_percent filtering.")
        return out_png

    # Plot (robust for single row)
    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, ax = plt.subplots(constrained_layout=True)

        arr = mat_plot.values
        # Guarantee a strict 2D array for imshow (M, N). Never (1, 1, N).
        if getattr(arr, "ndim", 2) == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]

        im = ax.imshow(arr, aspect="auto")
        ax.set_yticks(range(arr.shape[0]))
        ax.set_yticklabels([f"month {i+1}" for i in range(arr.shape[0])])
        ax.set_xticks(range(len(keep_cols)))
        ax.set_xticklabels(keep_cols, rotation=60, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Feature usage across months")

        # Adaptive sizing: widen with more features; height scales with months
        fig.set_size_inches(max(8.0, 0.22 * len(keep_cols) + 4.0),
                            max(3.5, 0.6 * arr.shape[0] + 1.0))
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return out_png

from matplotlib import cycler
import matplotlib.pyplot as plt


def save_group_equity_curves(
    equity_pivot: pd.DataFrame,
    bh_equity: Optional[pd.Series],
    out_png: str,
    title: str = "Equity by Model (full duration)",
    include_bh: bool = True,
):
    """
    Plot aggregated equity curves for a set of models (rows = months, cols = models).
    Uses the same academic style as other equity charts and anchors the Y axis
    around 1.0 so the scale is comparable to other strategy-growth plots.
    """
    import matplotlib.pyplot as plt

    if equity_pivot is None or equity_pivot.empty:
        return

    eq = equity_pivot.copy()
    eq = eq.loc[:, ~eq.columns.duplicated()]

    # Align BH equity (if provided) on the same index
    bh_series = None
    if include_bh and bh_equity is not None and not bh_equity.empty:
        bh_series = bh_equity.reindex(eq.index).astype(float)

    with apply_academic_style(theme="nature", palette="tab10"):
        fig, ax = plt.subplots()

        # 1) Optional Buy & Hold
        if include_bh and bh_series is not None:
            ax.plot(
                eq.index, bh_series.values,
                linestyle="--", linewidth=1.5, alpha=0.9,
                label="Buy & Hold",
            )

        # 2) Plot each model
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_idx = 0

        for m in eq.columns:
            vals = eq[m].astype(float).values
            if not np.isfinite(vals).any():
                continue

            c = colors[color_idx % len(colors)]
            color_idx += 1

            ax.plot(
                eq.index,
                vals,
                linewidth=1.8,
                alpha=0.95,
                label=str(m),
                color=c,
            )

        # 3) Y-axis scaling: anchor near 1.0 like other equity charts
        try:
            vals_list = [eq.to_numpy(dtype=float).ravel()]
            if include_bh and bh_series is not None and not bh_series.empty:
                vals_list.append(bh_series.to_numpy(dtype=float).ravel())
            all_vals = np.concatenate(vals_list)
            finite = all_vals[np.isfinite(all_vals)]
            if finite.size:
                y_min = float(np.nanmin(finite))
                y_max = float(np.nanmax(finite))
                bottom = min(1.0, y_min)
                span = (y_max - bottom) if y_max > bottom else max(y_max, 1.0)
                pad = 0.02 * span
                ax.set_ylim(bottom=bottom - pad, top=y_max + pad)
        except Exception as e:
            print(f"[WARN][0xfe0f] Could not set group-equity y-limits: {e}")

        # 4) Labels, grid, ticks
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Equity ([0xd7])")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

        try:
            _set_even_time_ticks(ax, eq.index)
        except Exception:
            pass

        # 5) Legend placement: move outside earlier so it doesn't cover the plot
        labels = [l for l in ax.get_legend_handles_labels()[1] if l]
        n_models = len(labels)

        if n_models > 4:
            # Outside, right-hand side
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles, labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
                title=None,
            )
            fig.subplots_adjust(right=0.78)
        elif n_models > 0:
            ax.legend(
                loc="lower right",
                frameon=False,
                ncol=1,
            )

        # 6) Figure size & save
        fig.set_size_inches(8.0, 4.5)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)


# Optional: only used if you enable calibration / feature selection
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import mutual_info_classif
except Exception:
    CalibratedClassifierCV = None
    mutual_info_classif = None

# ============================
# Feature Engineering Helpers
# ============================

def add_cyclic_hour_features(df: pd.DataFrame, hour_col: str = "hour") -> pd.DataFrame:
    """Add sin/cos encodings for hour-of-day (0..23). Assumes hour_col exists and is integer."""
    if hour_col in df:
        df["hour_sin"] = np.sin(2 * np.pi * df[hour_col] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df[hour_col] / 24.0)
    return df

def build_features_from_params(df, params: dict, base_features):
    """
    Select concrete feature column names from (df, params, base_features).

    Strategy
    --------
    1) Start from any provided `base_features` (if any).
    2) Add indicator columns when toggled in `params` and present in `df`.
    3) Expand each selected indicator with lags and rolling statistics:
       - lags:    _lag{k} for k in [1..lag_depth]
       - rolling: _rollmean{w}, _rollstd{w}, _rollslope{w} for w in `roll_windows`
    4) Optionally include raw returns lags up to `lags` or `lags_range`.

    Notes
    -----
    - This function only *selects* names that already exist in `df`.
      If a requested column doesn't exist, we skip it (with a warning).
    - Keep this list purely declarative; do not mutate `df` here.

    Parameters
    ----------
    df : pandas.DataFrame
        Engineered feature frame (already contains candidate columns).
    params : dict
        Trial/sample parameters (expects keys like 'indicator_windows',
        'use_sma', 'use_ema', 'use_macd', etc., plus 'lag_depth', 'roll_windows',
        'lags' or 'lags_range', and 'include_raw_lags').
    base_features : iterable or None
        Baseline features to always include (if present in `df`).

    Returns
    -------
    list[str]
        Final ordered list of feature column names.
    """
    features = list(base_features or [])
    ind_win = params.get("indicator_windows", {}) or {}

    def add_feat(name: str) -> None:
        """Append `name` if the column exists and isn't already selected."""
        if name in df.columns and name not in features:
            features.append(name)
        # elif name not in df.columns:
        #     print(f"[WARN][0xfe0f] build_features_from_params: '{name}' not in df.columns, skipping.")

    # 1) One-off indicators (controlled by toggles in params)
    if params.get("use_sma", False) and "sma" in ind_win:
        add_feat(f"sma_{ind_win['sma']}")
    if params.get("use_ema", False) and "ema" in ind_win:
        add_feat(f"ema_{ind_win['ema']}")
    if params.get("use_macd", False):
        for macd_col in ("macd_line", "macd_signal", "macd_diff"):
            add_feat(macd_col)
    if params.get("use_rsi", False) and "rsi" in ind_win:
        add_feat(f"rsi_{ind_win['rsi']}")
    if params.get("use_bbands", False):
        for bb_col in ("bb_upper", "bb_lower", "bb_pct", "bbw"):
            add_feat(bb_col)
    if params.get("use_atr", False) and "atr" in ind_win:
        add_feat(f"atr_{ind_win['atr']}")
    if params.get("use_adx", False) and "adx" in ind_win:
        add_feat(f"adx_{ind_win['adx']}")
    if params.get("use_stoch", False):
        for st_col in ("stoch_k", "stoch_d"):
            add_feat(st_col)
    if params.get("use_mtf_ma", False):
        for mtf_col in ("mtf_ma_fast", "mtf_ma_slow"):
            add_feat(mtf_col)

    # Optional SAR/Hour
    if "sar" in df.columns:
        add_feat("sar")
    if params.get("include_hour", False) and "hour" in df.columns:
        add_feat("hour")

    # 2) Window expansions
    lag_depth = int(params.get("lag_depth", 1))

    # accept list OR key strings
    roll_windows = params.get("roll_windows")
    if not roll_windows:
        rk = params.get("roll_windows_key_v2") or params.get("roll_windows_key")
        if rk:
            roll_windows = [int(x) for x in str(rk).split(",") if str(x).strip().isdigit()]
    if not roll_windows:
        roll_windows = [5]
    roll_windows = list(roll_windows)

    num_lags = int(params.get("lags", params.get("lags_range", 1)))

    # --- Feature budget clamp to avoid OOM from expansion ---
    MAX_EXPANDED = int(os.environ.get("MAX_EXPANDED_FEATURES", "5500"))
    import math as _math
    def _estimate_expanded(n_base, lag_depth, roll_windows, num_lags):
        per_feat = 1 + int(lag_depth) + 3 * len(roll_windows)
        return n_base * per_feat + int(num_lags)
    _est = _estimate_expanded(len(features), int(params.get("lag_depth", 1)), list(params.get("roll_windows", [])), int(params.get("lags", params.get("lags_range", 1))))
    _lag_depth = int(params.get("lag_depth", 1))
    _roll_windows = list(params.get("roll_windows", []))
    _num_lags = int(params.get("lags", params.get("lags_range", 1)))
    while _est > MAX_EXPANDED and _lag_depth > 1:
        _lag_depth -= 1
        _est = _estimate_expanded(len(features), _lag_depth, _roll_windows, _num_lags)
    while _est > MAX_EXPANDED and len(_roll_windows) > 1:
        _roll_windows = sorted(_roll_windows)[:max(1, len(_roll_windows)-1)]
        _est = _estimate_expanded(len(features), _lag_depth, _roll_windows, _num_lags)
    while _est > MAX_EXPANDED and _num_lags > 10:
        _num_lags = max(10, int(_math.floor(_num_lags * 0.8)))
        _est = _estimate_expanded(len(features), _lag_depth, _roll_windows, _num_lags)
    # overwrite local params for expansion below
    lag_depth = _lag_depth
    roll_windows = _roll_windows
    num_lags = _num_lags

    # Expand lags and rolling stats for each current feature
    for feat in list(features):
        for k in range(1, lag_depth + 1):
            add_feat(f"{feat}_lag{k}")
        for w in roll_windows:
            add_feat(f"{feat}_rollmean{w}")
            add_feat(f"{feat}_rollstd{w}")
            add_feat(f"{feat}_rollslope{w}")

    # 3) Raw returns lags
    if params.get("include_raw_lags", True):
        for lag in range(1, num_lags + 1):
            add_feat(f"returns_lag{lag}")

    if os.environ.get("LOG_MODE", "COMPACT").upper() == "DEBUG":
        # Only show in LOG_MODE=DEBUG
        log_print(
            "build_features_from_params selected features: " + str(features),
            level="DEBUG",
        )
        
    return features


def bipower_variation(returns: pd.Series, window: int = 30, min_periods: Optional[int] = None) -> pd.Series:
    """Jump-robust bipower variation: (pi/2) * sum |r_t| |r_{t-1}| over window, then sqrt."""
    if min_periods is None:
        min_periods = max(5, window // 3)
    abs_r = returns.abs()
    prod = abs_r * abs_r.shift(1)
    bpv = (pi / 2.0) * prod.rolling(window, min_periods=min_periods).sum()
    return np.sqrt(bpv.clip(lower=0))

# ----------------------------
# Fractional Differentiation
# ----------------------------
def _fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
    """
    Compute fractional differencing weights w_k ~ (-1)^k * comb(d, k).
    Truncate when |w_k| < thresh to keep it fast.
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w, dtype="float64")

def fracdiff(series: pd.Series, d: float = 0.4, max_size: int = 2000, thresh: float = 1e-4) -> pd.Series:
    """
    Fractionally difference a price/return series (no future info).
    Output aligned to original index with NaNs for the warmup.
    """
    series = series.astype("float64")
    w = _fracdiff_weights(d, min(max_size, len(series)), thresh=thresh)
    out = np.full_like(series.values, fill_value=np.nan, dtype="float64")
    k_max = len(w) - 1
    for t in range(k_max, len(series)):
        window = series.values[t - k_max : t + 1]
        out[t] = np.dot(w[::-1], window)
    return pd.Series(out, index=series.index, name=f"fd_{series.name}_d{d:.2f}")


def triple_barrier_labels(
    close: pd.Series,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_holding: int = 48,
    vol: Optional[pd.Series] = None,
    neutral_zone: float = 0.0,
    neutral_zone_is_sigma: bool = False,
) -> pd.Series:
    """
    3-class labels via triple barrier.
    Returns {0: down/short, 2: up/long, 1: neutral/hold} to match the existing 0/1/2 mapping used elsewhere (2=long).
    If neutral_zone > 0, small outcomes inside [-neutral_zone, +neutral_zone] become '2'.

    If neutral_zone_is_sigma is True, `neutral_zone` is interpreted as a
    multiplier on the local volatility series, so the per-bar neutral band is
    neutral_zone * vol[t] (vol must be on the same grid as `close`).
    """
    close = close.astype("float64")
    if vol is None:
        vol = realized_vol(close.pct_change().fillna(0), window=30).bfill().fillna(0)
    pt = pt_mult * vol
    sl = sl_mult * vol
    
    # Optional volatility-scaled neutral band
    if neutral_zone_is_sigma:
        # Per-bar neutral band: k * vol[t]
        nz_series = (neutral_zone * vol).astype("float64")
    else:
        nz_series = None

    n = len(close)
    y = np.full(n, 2, dtype="int8")  # default neutral

    for t0 in range(n):
        c0 = close.iloc[t0]
        ub = c0 * (1.0 + pt.iloc[t0])
        lb = c0 * (1.0 - sl.iloc[t0])
        t1 = min(n - 1, t0 + max_holding)
        path = close.iloc[t0 : t1 + 1]

        hit_up = (path >= ub)
        hit_dn = (path <= lb)
        first_up = path.index[hit_up.argmax()] if hit_up.any() else None
        first_dn = path.index[hit_dn.argmax()] if hit_dn.any() else None

        if first_up is not None and (first_dn is None or first_up <= first_dn):
            y[t0] = 2       # LONG
        elif first_dn is not None and (first_up is None or first_dn < first_up):
            y[t0] = 0       # SHORT
        else:
            ret = (path.iloc[-1] / c0) - 1.0
            # Decide neutral band: either absolute or volatility-scaled
            if neutral_zone_is_sigma and nz_series is not None:
                nz = float(nz_series.iloc[t0])
            else:
                nz = float(neutral_zone)

            if ret > nz:
                y[t0] = 2   # LONG
            elif ret < -nz:
                y[t0] = 0   # SHORT
            else:
                y[t0] = 1   # NEUTRAL
                
    return pd.Series(y, index=close.index, name="label_tb")


def attach_macro_features(
    df_bars: pd.DataFrame,
    macro_specs,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Attach daily or lower-frequency macro features to an intraday bar DataFrame.

    Typical use-case:
    - df_bars: 30-minute EUR/USD bars with a tz-aware DatetimeIndex.
    - macro_specs: either
        * dict {label: csv_path}, or
        * list/tuple of csv_paths (label inferred from filename).
    - Each macro CSV should contain a date/time column (e.g. 'date' or 'time')
      and one or more numeric columns with macro values.

    Behaviour:
    - Parse macro dates, normalise to daily (midnight, no tz).
    - Optionally apply a lag of `lag_days` (shift values) so that bar-day t
      sees macro information from day t - lag_days (to avoid look-ahead).
    - Align to bar dates (index normalised to daily, tz-stripped), then
      forward-fill within and across days.
    - Columns are joined onto df_bars; when multiple sources are provided,
      their numeric columns are concatenated. If a dict is used, labels are
      taken from the dict keys; otherwise, filenames are used as prefixes.
    """
    import os
    import pandas as pd

    if df_bars is None or len(df_bars) == 0:
        return df_bars

    if macro_specs is None or macro_specs == {}:
        return df_bars

    # Normalise macro_specs into an iterable of (label, path)
    items: list[tuple[str, str]] = []
    if isinstance(macro_specs, dict):
        for key, path in macro_specs.items():
            if path is None:
                continue
            items.append((str(key), str(path)))
    else:
        # treat as list/tuple/single path
        if not isinstance(macro_specs, (list, tuple)):
            macro_specs = [macro_specs]
        for path in macro_specs:
            if not path:
                continue
            label = os.path.splitext(os.path.basename(str(path)))[0]
            items.append((label, str(path)))

    if not items:
        return df_bars

    # Prepare bar-level date index (naive daily)
    try:
        bar_dates = pd.to_datetime(df_bars.index).tz_convert(None).normalize()
    except Exception:
        bar_dates = pd.to_datetime(df_bars.index).normalize()

    macro_frames: list[pd.DataFrame] = []

    for label, path in items:
        try:
            mdf = pd.read_csv(path)
        except Exception:
            continue
        if mdf is None or len(mdf) == 0:
            continue

        # Heuristic: find a date-like column
        date_col = None
        for cand in ("date", "Date", "DATE", "time", "Time", "TIME"):
            if cand in mdf.columns:
                date_col = cand
                break
        if date_col is None:
            # fall back to first column
            date_col = mdf.columns[0]

        try:
            mdf[date_col] = pd.to_datetime(mdf[date_col], errors="coerce").dt.normalize()
        except Exception:
            continue
        mdf = mdf.set_index(date_col).sort_index()

        # Keep only numeric macro columns
        mdf = mdf.select_dtypes(include=["number"])
        if mdf.shape[1] == 0:
            continue

        # Apply daily lag by shifting values (not index)
        if lag_days and lag_days > 0:
            mdf = mdf.shift(int(lag_days))

        # Align to bar dates and forward-fill
        mdf_aligned = mdf.reindex(bar_dates).ffill()
        if mdf_aligned is None or len(mdf_aligned) == 0:
            continue

        # Use a prefix when multiple macro sources are present
        if len(items) > 1:
            mdf_aligned = mdf_aligned.add_prefix(f"{label}_")

        # Put back on the original bar index
        mdf_aligned.index = df_bars.index
        macro_frames.append(mdf_aligned)

    if not macro_frames:
        return df_bars

    macro_all = pd.concat(macro_frames, axis=1)
    # Avoid clobbering existing columns: only add new ones
    new_cols = [c for c in macro_all.columns if c not in df_bars.columns]
    if not new_cols:
        return df_bars

    return df_bars.join(macro_all[new_cols], how="left")


# ============================
# Scaling / Calibration / CP
# ============================
@dataclass
class RollingStandardizer:
    """
    Rolling standardizer fit on train (no leakage).
    - fit_transform(X_train): scales train with rolling mean/std (per-column)
    - transform(X_test): scales test using the *last* (mean,std) from train
    """
    window: int = 200
    min_periods: int = 50

    def fit_transform(self, X_train: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        # Single copy to avoid mutating caller's DataFrame; operate in-place on it
        X = X_train.astype(float).copy()

        mu_roll = X.rolling(self.window, min_periods=self.min_periods).mean()
        sd_roll = (
            X.rolling(self.window, min_periods=self.min_periods)
              .std()
              .replace(0, np.nan)
        )

        # Only keep the last rolling stats for test-time scaling
        stats = {"mu_last": mu_roll.iloc[-1], "sd_last": sd_roll.iloc[-1]}

        # In-place standardization on X to avoid allocating X_scaled
        X -= mu_roll
        X /= sd_roll

        return X, stats

    def transform(self, X_test: pd.DataFrame, stats: dict) -> pd.DataFrame:
        mu_last = stats["mu_last"]
        sd_last = stats["sd_last"].replace(0, np.nan)
        return (X_test - mu_last) / sd_last


def calibrate_prefit_and_predict_proba(
    base_estimator, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, method: str = "isotonic"
) -> Tuple[np.ndarray, Optional[object]]:
    """Calibrate probabilities without using the deprecated cv='prefit' path."""
    if CalibratedClassifierCV is None:
        return base_estimator.predict_proba(X_pred), None
    try:
        from sklearn.base import clone
        # Fresh, unfitted copy -- we calibrate with k-fold CV on the train window
        est = clone(base_estimator)
        calibrator = CalibratedClassifierCV(estimator=est, method=method, cv=3)
        calibrator.fit(X_train, y_train)
        proba = calibrator.predict_proba(X_pred)
        return proba, calibrator
    except Exception:
        # Last-resort fallback: use the prefit estimator's own proba
        return base_estimator.predict_proba(X_pred), None

@dataclass
class ConformalClassifier:
    """
    Split-conformal for multiclass:
    - Nonconformity score: 1 - p_true
    - qhat: (ceil((n+1)*(1-alpha))/n)-quantile of calibration scores
    - Predict set S(x) = {k : 1 - p_k <= qhat}
    If |S(x)|==1 we call it 'decisive' and take that class; else abstain.
    """
    alpha: float = 0.1
    qhat_: Optional[float] = None

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        idx = (np.arange(len(y_cal)), y_cal.astype(int))
        nc = 1.0 - proba_cal[idx]
        n = len(nc)
        k = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        k = np.clip(k, 0, n - 1)
        self.qhat_ = float(np.partition(nc, k)[k])
        return self

    def predict_decisions(self, proba_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.qhat_ is not None, "Fit conformal first."
        top_idx = proba_new.argmax(axis=1)
        top_nc = 1.0 - proba_new[np.arange(len(top_idx)), top_idx]
        decisive = top_nc <= self.qhat_
        return decisive, top_idx

# ============================
# Lightweight Feature Selection
# ============================
def select_topk_by_mutual_info(X: pd.DataFrame, y: pd.Series, top_k: int = 64, random_state: int = 42) -> List[int]:
    """
    Returns column indices of top-k features by mutual information on X (2D) vs y.
    Works with DataFrame or ndarray; returns integer indices suitable for np.take.
    """
    if mutual_info_classif is None or top_k >= X.shape[1]:
        return list(range(X.shape[1]))
    Xv = X.values if isinstance(X, pd.DataFrame) else X
    yv = y.values if isinstance(y, pd.Series) else y
    mi = mutual_info_classif(Xv, yv, random_state=random_state, n_jobs=1)
    order = np.argsort(mi)[::-1][:top_k]
    return order.tolist()

def first_tradable_test_bar(index, month_start_ts):
    """
    Return the first timestamp in `index` that is >= month_start_ts.
    If none exists, return None.
    """
    # Be tolerant: callers sometimes pass a DataFrame/Series by mistake.
    if hasattr(index, "index") and not isinstance(index, (pd.Index, pd.DatetimeIndex)):
        index = index.index

    if index is None:
        return None

    idx = pd.DatetimeIndex(index)
    ts = pd.to_datetime(month_start_ts)

    # TZ-safe comparison: align ts to idx timezone if needed
    if getattr(idx, "tz", None) is not None:
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(idx.tz)
        else:
            try:
                ts = ts.tz_convert(idx.tz)
            except Exception:
                # If tz_convert fails for any reason, fall back without conversion
                pass

    pos = idx.searchsorted(ts, side="left")
    if pos < len(idx):
        return idx[pos]

    return None


def compute_required_test_warmup_bars(params: dict) -> int:
    """
    Conservative pre-seed length from your tuning params.
    Considers lags, lag_depth, roll windows, long RV, rolling scaler, fracdiff, and DQN window.
    """
    p = params or {}
    lags = int(p.get("lags", p.get("lags_range", 0)) or 0)
    lag_depth = int(p.get("lag_depth", 1) or 1)
    need_lags = max(lags, lags * lag_depth)

    # roll_windows can be list or "5,10,20"
    rw = []
    if "roll_windows" in p and p.get("roll_windows"):
        if isinstance(p["roll_windows"], str):
            rw = [int(x) for x in p["roll_windows"].split(",") if str(x).strip().isdigit()]
        elif isinstance(p["roll_windows"], (list, tuple)):
            rw = [int(x) for x in p["roll_windows"] if not isinstance(x, (list, tuple))]
    elif "roll_windows_key" in p and p.get("roll_windows_key"):
        rw = [int(x) for x in str(p["roll_windows_key"]).split(",") if str(x).strip().isdigit()]
    need_roll = max(rw) if rw else 0

    rv_long = int(p.get("rv_window_long", 0) or 0)

    scaler_win = int(p.get("scaler_window", 0) or 0) if p.get("use_rolling_scaler") else 0

    fracdiff_warmup = int(p.get("fracdiff_warmup", 0) or 0)
    if not fracdiff_warmup and p.get("use_fracdiff") and float(p.get("fracdiff_d", 0)) > 0:
        fracdiff_warmup = 200  # safe default

    # deep windows / DQN
    mt = str(p.get("model_type", "")).lower()
    need_seq = 0
    if mt in ("cnn", "lstm", "transformer"):
        # For seq models, lags ~ effective window length
        need_seq = max(need_seq, lags)
    if mt == "dqn":
        dqn_cfg = p.get("dqn_config", {}) or {}
        need_seq = max(need_seq, int(dqn_cfg.get("window", lags) or 0))

    # --- NEW: generic window / lookback sweep ---
    generic_win = 0
    for k, v in p.items():
        key = str(k).lower()
        # Do NOT treat training-budget keys as feature warmup requirements
        if "train" in key and "window" in key:
            continue
        if any(tok in key for tok in ("window", "lookback", "period")):
            try:
                val = int(v)
            except Exception:
                continue
            generic_win = max(generic_win, val)

    # Final warm-up requirement = max of all known windows
    warm = max(
        need_lags,
        need_roll,
        rv_long,
        scaler_win,
        fracdiff_warmup,
        need_seq,
        generic_win,
    )

    if warm <= 0:
        return 0

    # Add a safety margin so month-start is definitely fully warmed
    margin = max(10, int(0.10 * warm))
    return int(warm + margin)


# === NEW: frequency & HAC helpers (place after [ANCHOR] imports) ===
def estimate_frequency_per_year(index) -> float:
    """
    Estimate bars-per-year from a DateTimeIndex:
    - compute median bars/day,
    - detect if weekends are mostly present -> use 365, otherwise ~252 trading days.
    """
    import numpy as np
    import pandas as pd

    if not hasattr(index, "tz"):
        try:
            index = pd.to_datetime(index, utc=True, errors="coerce")
        except Exception:
            return 252.0  # safe fallback

    if len(index) < 3:
        return 252.0

    by_day = pd.Series(1.0, index=index).groupby(index.floor("D")).count()
    if by_day.empty:
        return 252.0
    bars_per_day = float(by_day.median())

    # crude weekday/weekend detection
    days = pd.Index(by_day.index)
    weekend_days = int(((days.dayofweek == 5) | (days.dayofweek == 6)).sum())
    frac_weekend = weekend_days / max(1, len(days))
    days_per_year = 365.0 if frac_weekend > 0.10 else 252.0

    return max(1.0, bars_per_day * days_per_year)

def _auto_nw_lag(n: int, mode: str = "sqrt", x=None) -> int:
    """
    mode: "sqrt" (default) or "andrews".
    If "andrews", use a light plug-in via AR(1) rho -> q [0x2248] c * n**(1/5).
    """
    import numpy as np
    n = int(max(1, n))
    m = (mode or "sqrt").lower()
    if m == "andrews":
        # Estimate AR(1) rho from data if provided; else assume modest short memory
        rho = 0.0
        if x is not None:
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if x.size > 3:
                x0, x1 = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
                den = float((x0**2).sum()) or 1.0
                rho = float((x0 * x1).sum() / den)
                rho = float(np.clip(rho, -0.99, 0.99))
        # Bartlett kernel constant (Andrews 1991)
        c = 1.3221  # common plug-in constant for Bartlett
        q = int(max(1, round(c * (n ** 0.2))))
        return q
    # default robust rule-of-thumb
    return int(np.floor(np.sqrt(n)))

def hac_std(x, max_lag="auto") -> float:
    """
    Newey-West (HAC) standard deviation for a 1D array/Series x.
    max_lag: int | "auto" (sqrt(n)) | "andrews" (plug-in).
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return 0.0
    x = x - np.mean(x)
    g0 = np.dot(x, x) / n
    if isinstance(max_lag, str):
        q = _auto_nw_lag(n, mode=max_lag, x=x)
    else:
        q = int(max(0, max_lag))
    if q == 0:
        var = g0
    else:
        var = g0
        for k in range(1, q + 1):
            w = 1.0 - k / (q + 1.0)  # Bartlett
            gamma_k = np.dot(x[:-k], x[k:]) / n
            var += 2.0 * w * gamma_k
    return float(np.sqrt(max(var, 0.0)))


import numpy as _np
from math import sqrt as _sqrt

try:
    from scipy.stats import norm as _norm
except Exception:
    _norm = None

def compute_dsr_scores(scores):
    """
    Deflated 'Sharpe' proxy over an array of trial scores.
    Returns a list of DSR-like probabilities (higher is better), one per score.
    Approximates multiple-testing via a [0x160]id[0xe1]k-style family correction.
    """
    x = _np.asarray(scores, dtype=float)
    n = int(_np.isfinite(x).sum())
    if n <= 1 or _np.allclose(_np.nanstd(x, ddof=1), 0.0):
        # degenerate: everyone ties
        return [0.0 if not _np.isfinite(v) else 0.5 for v in x]

    mu = float(_np.nanmean(x))
    sd = float(_np.nanstd(x, ddof=1))
    out = []
    for v in x:
        if not _np.isfinite(v):
            out.append(0.0); continue
        z = (float(v) - mu) / sd
        # single-test p-value (one-sided)
        if _norm is None:
            import math as _m
            p_single = 0.5 * (1.0 - _m.erf(z / _m.sqrt(2.0)))  # 1 - Phi(z)
        else:
            p_single = 1.0 - float(_norm.cdf(z))
        # family-wise correction across n tests ([0x160]id[0xe1]k)
        p_family = 1.0 - (1.0 - max(1e-12, min(1.0, p_single))) ** n
        dsr = 1.0 - p_family
        out.append(float(dsr))
    return out


def save_optuna_learning_summary(study, out_path, n_startup=10, penalty_value: float | None = None):
    """
    Writes a small JSON with diagnostics that the sampler is learning:
      - Best/median in startup (random) vs post-startup (TPE) trials
      - Cliff's delta effect size (post-startup vs startup)
      - Share of post-startup trials beating startup median
    """
    import json, numpy as np, os

    trials = [t for t in study.trials if (t.value is not None)]
    pen = None if penalty_value is None else float(penalty_value)
    vals = []
    for t in trials:
        if getattr(t, "state", None) and getattr(t.state, "name", "") == "COMPLETE" and (t.value is not None):
            v = float(t.value)
            if pen is None or v != pen:
                vals.append((t.number, v))

    if not vals:
        payload = {"note": "no complete trials with valid scores"}
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f: json.dump(payload, f, indent=2)
        print(f"[WARN][0xfe0f] No eligible trials for learning summary -> {out_path}")
        return out_path

    vals.sort(key=lambda x: x[0])
    startup   = np.array([v for n, v in vals if n < int(n_startup)], dtype=float)
    post      = np.array([v for n, v in vals if n >= int(n_startup)], dtype=float)

    def _cliffs_delta(a, b):
        if a.size == 0 or b.size == 0:
            return float("nan")
        # rank-biserial: ( #(b>a) - #(a>b) ) / (len(a)*len(b))
        count = 0
        for x in a:
            count += int((b > x).sum()) - int((b < x).sum())
        return float(count) / float(a.size * b.size)

    med_start = float(np.median(startup)) if startup.size else float("nan")
    med_post  = float(np.median(post)) if post.size else float("nan")
    best_start = float(np.max(startup)) if startup.size else float("nan")
    best_post  = float(np.max(post)) if post.size else float("nan")
    uplift_best = (best_post - best_start) if (np.isfinite(best_post) and np.isfinite(best_start)) else float("nan")

    share_beating_startup_med = float((post > med_start).mean()) if (post.size and np.isfinite(med_start)) else float("nan")
    delta = _cliffs_delta(startup, post)

    payload = {
        "n_startup": int(n_startup),
        "n_startup_complete": int(startup.size),
        "n_post_complete": int(post.size),
        "startup_median": med_start,
        "post_median": med_post,
        "startup_best": best_start,
        "post_best": best_post,
        "uplift_best": uplift_best,
        "share_post_above_startup_median": round(share_beating_startup_med, 3),
        "cliffs_delta_post_vs_startup": round(delta, 3)
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path

def fit_coverage_threshold_on_calibration(proba_cal: np.ndarray,
                                           target_active_rate: float) -> float:
    """
    Given calibrated 3-class probabilities on a calibration slice,
    return the confidence threshold that achieves the requested active rate.
    """
    import numpy as np
    p = np.asarray(proba_cal, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("proba_cal must be (n,3) or (n,2) after sanitize_proba()")
    max_conf = p.max(axis=1)
    
    # Coverage must reflect **trade intent** (short/long strength), not certainty about "flat".
    # Otherwise a dominant flat class makes coverage targeting ineffective.
    if p.shape[1] >= 3:
        max_conf = np.maximum(p[:, 0], p[:, 2])
    else:
        max_conf = p.max(axis=1)
    
    target_active_rate = float(np.clip(target_active_rate, 1e-6, 0.999999))
    # keep top-K% = choose (1 - rate) quantile as threshold
    q = 1.0 - target_active_rate
    thr = float(np.quantile(max_conf, q)) if len(max_conf) else 0.50
    return float(np.clip(thr, 0.0, 1.0))


def apply_temperature_to_proba(proba: np.ndarray, T: float) -> np.ndarray:
    """Softmax temperature scaling in log-prob space (stable & model-agnostic)."""
    T = float(max(1e-3, T))
    logp = np.log(np.clip(proba, 1e-7, 1.0)).astype(np.float64)
    z = logp / T
    z -= z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / np.sum(ez, axis=1, keepdims=True)).astype(np.float32)


def fit_temperature_from_proba(proba: np.ndarray, y_true: np.ndarray) -> float:
    """1-D grid search for T minimizing NLL on a calibration slice."""
    idx = np.arange(len(y_true))
    def nll(p):
        p = np.clip(p[idx, y_true], 1e-7, 1.0)
        return float(-np.mean(np.log(p)))
    Ts = np.concatenate([np.linspace(0.5, 3.0, 26),
                         np.linspace(0.3, 0.5, 5),
                         np.linspace(3.0, 4.0, 5)])
    best_T, best_loss = 1.0, nll(proba)
    for T in Ts:
        L = nll(apply_temperature_to_proba(proba, T))
        if L < best_loss:
            best_T, best_loss = float(T), float(L)
    return float(best_T)

def _norm_cdf(x):
    # scipy-free standard normal CDF
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, periods_per_year=12):
    """
    Bailey & L[0xf3]pez de Prado (2012) style PSR (simplified, iid assumption).
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

def save_month_equity_graph(
    df,
    out_csv=None,
    out_png="results/monthly_equity.png",
    label_model="Model",
    title=None,
    dpi=300,
    line_width=1.0,
    style="nature",
    palette="okabe_ito_no_black",
):
    """
    Thesis-grade monthly equity:
      [0x2022] Top: strategy (solid) vs BH (dashed)
      [0x2022] Bottom: strategy drawdown (%)
    Also writes a per-bar CSV (suffix `_bars.csv` when given csv_month_k.csv).
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    if df is None or len(df) == 0 or not {"cstrategy_cont","creturns_cont"} <= set(df.columns):
        print("[WARN][0xfe0f] save_month_equity_graph: empty or missing columns -- skipping.")
        return None

    # Optional bars CSV (avoid overwriting the monthly summary CSV)
    if out_csv:
        base = os.path.basename(out_csv)
        if base.startswith("csv_month_") and base.endswith(".csv"):
            root, ext = os.path.splitext(out_csv)
            out_csv = root + "_bars" + ext
        try:
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            df[["cstrategy_cont","creturns_cont"]].to_csv(out_csv, index=True, float_format="%.10f")
        except Exception as e:
            print(f"[WARN][0xfe0f] Could not write monthly bars CSV: {e}")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, (ax, ax_dd) = plt.subplots(2, 1, height_ratios=[3, 1.4], constrained_layout=True, sharex=True)

        eq = pd.to_numeric(df["cstrategy_cont"], errors="coerce")
        bh = pd.to_numeric(df["creturns_cont"], errors="coerce")

        # Top: equity curves
        ax.plot(df.index, eq.values, linewidth=line_width, label=label_model)
        ax.plot(df.index, bh.values, linewidth=1.2, linestyle="--", color="#666666", label="BH")
        try:
            _set_even_time_ticks(ax, df.index, n_parts=8)
        except Exception:
            pass

        y_min = float(np.nanmin([eq.min(), bh.min()]))
        y_top = float(np.nanmax([eq.max(), bh.max()]))
        bottom = min(1.0, y_min)
        pad = 0.02 * (y_top - bottom if y_top > bottom else max(y_top, 1.0))
        ax.set_ylim(bottom=bottom - pad, top=y_top + pad)
        ax.margins(x=0)
        ax.set_title(title or "Monthly Equity", pad=10)
        ax.set_ylabel("Equity ([0xd7])")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", ncol=1, fontsize=9, frameon=False)

        # Bottom: drawdown
        dd = _compute_drawdown(eq)
        ax_dd.plot(dd.index, (dd * 100.0).values, linewidth=1.0)
        ax_dd.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax_dd.set_ylabel("DD")
        ax_dd.grid(True, alpha=0.25)
        try:
            _set_even_time_ticks(ax_dd, df.index, n_parts=8)
        except Exception:
            pass
        try:
            dd_min = float(dd.min())
            ax_dd.set_title(f"Max DD: {dd_min*100.0:.1f}%", fontsize=9, pad=2)
        except Exception:
            pass

        fig.set_size_inches(11.8, 5.0)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return out_png


def save_feature_heatmap_for_single_month(
    df_row,
    out_png="results/feature_heatmap_single.png",
    top_k=30,
    style="nature",
    palette="okabe_ito_no_black",
    exclude_prefixes=("returns_lag", "hour"),
    collapse_raw_lags=True,
):
    """
    df_row: DataFrame with a single row (the month summary), ideally containing 'features_used'.
    Produces a 1[0xd7]K heatmap (counts) and saves to out_png.
    """
    import os, json, ast, numpy as np, pandas as pd, matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    if df_row is None or len(df_row) == 0:
        print("[WARN][0xfe0f] Single-month heatmap: empty df_row -- skipping.")
        return None

    row = df_row.iloc[0]
    feats = None

    # Try 'features_used' first
    if "features_used" in df_row.columns:
        val = row["features_used"]
        try:
            if isinstance(val, str):
                try:
                    feats = json.loads(val)
                except Exception:
                    feats = ast.literal_eval(val)
            elif isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                feats = list(val)
        except Exception:
            feats = None

    # Fallback: mine booleans like use_rsi=True -> 'use_rsi'
    if not feats:
        feats = []
        for c in df_row.columns:
            if str(c).startswith("use_"):
                try:
                    if bool(row[c]):
                        feats.append(str(c))
                except Exception:
                    pass

    # Filter / normalize
    def _keep(name: str) -> bool:
        return not any(name.startswith(p) for p in (exclude_prefixes or []))

    def _norm(name: str) -> str:
        if collapse_raw_lags and name.startswith("returns_lag"):
            return "returns_lag*"
        return name

    feats = [_norm(f) for f in (feats or []) if _keep(str(f))]
    if not feats:
        print("[WARN][0xfe0f] Single-month heatmap: no features found -- skipping.")
        return None

    # Count & take top_k
    s = pd.Series(feats).value_counts().head(int(top_k))
    mat = s.to_numpy().reshape(1, -1)

    plt.figure(figsize=(max(6, len(s) * 0.3), 1.8))
    im = plt.imshow(mat, aspect="auto")
    plt.yticks([0], ["month"])
    plt.xticks(range(len(s.index)), s.index, rotation=60, ha="right")
    for j, v in enumerate(s.values):
        plt.text(j, 0, str(v), va="center", ha="center", fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Feature usage (single month)")
    # Give more room for long x-tick labels
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.32, top=0.9)  # <- added
    try:
        plt.tight_layout()
    except Exception:
        # Some figures with many labels can't satisfy tight_layout; continue anyway
        pass

    plt.savefig(out_png, dpi=140, bbox_inches="tight")  # <- added bbox

    plt.close()
    return out_png

def enforce_day1_eval_anchor(index, month_start_ts):
    """
    Return the first index on the SAME calendar day as month_start_ts, after
    session filtering. If that day has no bars (rare), fall back to the first
    tradable bar >= month_start_ts.
    """
    import pandas as pd
    ts = pd.to_datetime(month_start_ts)
    # First bar on the same day (UTC) at/after ts
    same_day = index[(index >= ts) & (index.normalize() == ts.normalize())]
    if len(same_day) > 0:
        return same_day[0]
    # Fallback: first tradable bar >= ts (e.g., if the day had no bars)
    return first_tradable_test_bar(index, ts)

# --- Online performance monitors for adaptive switching -----------------------

def estimate_bars_per_day(index) -> int:
    """
    Median bars-per-day on the given DatetimeIndex (already session-filtered).
    Falls back to 48 (30min bars * 24h * 0.1 session) if empty.
    """
    import numpy as np, pandas as pd
    if index is None or len(index) == 0:
        return 48
    dt = pd.DatetimeIndex(index)
    by_day = pd.Series(1, index=dt).groupby(dt.normalize()).sum()
    med = float(by_day.median()) if len(by_day) else 48.0
    return int(max(1, round(med)))

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

def find_hit_rate_switch_idx(df, window_bars: int, thr: float = 0.45, start_ts=None):
    """
    Return the FIRST timestamp >= start_ts where rolling hit-rate < thr.
    If none, return None.
    """
    import pandas as pd
    s = df if start_ts is None else df.loc[pd.to_datetime(start_ts):]
    hr = compute_rolling_hit_rate(s, int(window_bars), min_active=1).dropna()
    bad = hr[hr < float(thr)]
    return None if bad.empty else bad.index[0]


def _mk_flat_compat(dirpath: str) -> dict:
    """Make a directory and return a dict mapping all legacy keys to it."""
    import os
    os.makedirs(dirpath, exist_ok=True)
    # expose many aliases so existing callers keep working
    return {
        "base": dirpath,
        "dir": dirpath,
        "path": dirpath,
        "csv": dirpath,
        "graphs": dirpath,
        "heatmaps": dirpath,
    }

def month_dir_path(model_base_dir: str, month_ix: int, *ignored) -> dict:
    """
    Backward-compatible month dir helper.

    Accepts the modern signature (model_base_dir, month_ix).
    Also accepts the legacy 4-arg signature (run_dir, family, month_ix, model_name)
    and internally resolves to the same flat directory.
    """
    import os

    # Legacy shim: month_dir_path(run_dir, family, month_ix, model_name)
    if isinstance(month_ix, (str, int)) and ignored:
        # reinterpret args
        try:
            run_dir, family, m_ix, model_name = model_base_dir, month_ix, ignored[0], ignored[1]
            md = ensure_model_dirs(run_dir, family, model_name)
            month_dir = os.path.join(md["months_root"], str(int(m_ix)))
            return _mk_flat_compat(month_dir)
        except Exception:
            pass  # fall through to modern path

    # Modern: month_dir_path(model_base_dir, month_ix)
    month_dir = os.path.join(model_base_dir, "Months", str(int(month_ix)))
    return _mk_flat_compat(month_dir)


# --- SPA: single-strategy p-value (stationary bootstrap) -----------------------
def _fmt_table_ascii(headers, rows, title=None):
    if not rows:
        return
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len("" if c is None else str(c)))
    def _row(vals):
        return " | ".join(str(v if v is not None else "").ljust(widths[i]) for i, v in enumerate(vals))
    sep = "-+-".join("-" * w for w in widths)
    if title:
        print(f"\n{title}")
    print(_row(headers)); print(sep)
    for r in rows:
        print(_row(r))
    print(sep)

# === Noise-robust feature prefilter utilities =================================
# These helpers remove near-constants, collapse highly correlated columns, and
# (optionally) keep only the top-K features by mutual information with the label.
# Motivation:
#  - Near-constant features add variance without signal [R1].
#  - Collinearity can destabilize model selection & importance [R2].
#  - MI-based filters (e.g., mRMR family) are strong univariate gates before CV [R3].
#
# We keep this as a FILTER (not wrapper) so Optuna still decides the strategy.

from typing import Sequence, Optional, Tuple
import numpy as np
import pandas as pd

def drop_near_constant_features(
    X: pd.DataFrame,
    min_unique_frac: float = 0.005,
    min_std: float = 1e-6,
) -> list[str]:
    """
    Keep columns with enough variation (unique fraction & std).
    - unique fraction: guards discrete/indicator-like features
    - std: guards continuous-scale dead features
    References: simple filter advocated in feature-selection primers [R1].
    """
    n = float(len(X)) if len(X) else 1.0
    uniq_ok = (X.nunique(dropna=True) / max(n, 1.0)) >= float(min_unique_frac)
    std_ok  = X.std(skipna=True) >= float(min_std)
    keep = list(X.columns[uniq_ok & std_ok])
    return keep

def drop_high_corr_features(
    X: pd.DataFrame,
    threshold: float = 0.96,
    prefer_prefixes: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Greedy correlation-threshold pruning: among |rho|>threshold pairs, drop one.
    - We *prefer* keeping columns starting with any of prefer_prefixes when ties occur
      (e.g., realized-volatility or price-normalized spreads you trust more).
    - Uses pairwise-complete Pearson corr (pandas handles NaNs pairwise).
    Reference: correlation filtering to mitigate collinearity effects [R2].
    """
    if X.shape[1] <= 1:
        return list(X.columns)

    # Order columns so preferred prefixes are seen earlier (kept more often).
    cols = list(X.columns)
    if prefer_prefixes:
        def pref_score(c):
            for rank, p in enumerate(prefer_prefixes):
                if c.startswith(str(p)):  # keep earlier
                    return -(len(prefer_prefixes) - rank)
            return 0
        cols = sorted(cols, key=pref_score)

    # Absolute upper-triangular correlation matrix
    corr = X[cols].corr().abs()
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

    to_drop = set()
    for i, ci in enumerate(cols):
        if ci in to_drop:
            continue
        # find highly correlated partners for ci in upper triangle
        high = [cols[j] for j in range(i + 1, len(cols)) if mask_upper[i, j] and corr.iloc[i, j] > threshold]
        for cj in high:
            if cj not in to_drop:
                to_drop.add(cj)

    keep = [c for c in cols if c not in to_drop]
    return keep

def prefilter_features_train(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> list[str]:
    """
    Orchestrates the *pre-CV* filtering stages:
      1) near-constant drop
      2) high-corr pruning
      3) MI top-K gate (optional)

    Notes:
    - y must be aligned with X (same index); we silently align to the intersection.
    - MI step uses your existing select_topk_by_mutual_info() if present.
    """
    try:
        from utilsNoWFO import select_topk_by_mutual_info
    except Exception:
        select_topk_by_mutual_info = None

    # Defaults (safe)
    min_uniq = float(cfg.get("prefilter_min_unique_frac", 0.005))
    min_std  = float(cfg.get("prefilter_min_std", 1e-6))
    rho_thr  = float(cfg.get("prefilter_max_corr", 0.96))
    prefer   = cfg.get("prefilter_prefer_prefixes", ["rv", "ema", "sma", "macd", "adx"])
    mi_topk  = cfg.get("mutual_info_top_k", "sqrt")
    rng      = int(cfg.get("prefilter_random_state", 42))

    # 0) ensure alignment (in case caller forgot)
    if not X.empty and not y.empty:
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

    if X.shape[1] == 0:
        return []

    # 1) drop near-constants (works on original X; no extra copy needed)
    keep = drop_near_constant_features(
        X, min_unique_frac=min_uniq, min_std=min_std
    )
    if not keep:
        return []

    # 2) prune highly correlated on the reduced view only
    X_corr = X[keep]
    keep = drop_high_corr_features(
        X_corr, threshold=rho_thr, prefer_prefixes=prefer
    )
    if not keep:
        return []

    # 3) MI top-K (optional)
    if select_topk_by_mutual_info is None or len(keep) <= 1:
        return list(keep)

    # Minimal fill (means) for MI computation only; does NOT leak into model pipeline
    Xmi = X[keep].fillna(X[keep].mean())
    ymi = y.loc[Xmi.index]

    # derive K
    if isinstance(mi_topk, str) and mi_topk.lower() == "sqrt":
        K = max(1, int(np.sqrt(max(1, Xmi.shape[1]))))
    else:
        K = int(mi_topk)
        K = min(max(1, K), Xmi.shape[1])

    if K >= Xmi.shape[1]:
        return list(Xmi.columns)

    idxs = select_topk_by_mutual_info(
        Xmi, ymi.astype(int), top_k=K, random_state=rng
    )
    keep3 = [Xmi.columns[i] for i in idxs]
    return keep3

def realized_vol(ser, window=96):
    ser = ser.astype(float).fillna(0.0)
    return ser.rolling(int(window), min_periods=max(2, int(window//4))).std(ddof=0)

class CostAwareWrapper:
    """
    Wraps a TradingEnv-like environment so the step reward becomes:

        reward_net = reward_gross
                     - cost_scale * (spread + slippage)
                     - turnover_penalty  (on flips)

    Assumes `action` encodes position state; if action changes vs. previous,
    we charge costs aligned to bar t (arrays are aligned to env steps).

    - `cost_scale` > 1.0 makes transaction costs bite harder (e.g. to
      reflect unmodelled costs or deliberately discourage churn).
    - `turnover_penalty` adds an extra fixed penalty on every flip,
      independent of the spread / slippage arrays.
    """
    def __init__(
        self,
        env,
        *,
        spread=None,
        slippage_bps=None,
        mid_price=None,
        cost_scale: float = 1.0,
        turnover_penalty: float = 0.0,
    ):
        self.env = env
        self.spread = np.asarray(spread, dtype=np.float32) if spread is not None else None
        self.slip   = np.asarray(slippage_bps, dtype=np.float32) if slippage_bps is not None else None
        self.cost_scale = float(cost_scale)
        self.price  = np.asarray(mid_price, dtype=np.float32) if mid_price is not None else None

        self.turnover_penalty = float(turnover_penalty)
        self.t = 0
        self._fallback_t = 0

        # Optional: try to discover a price series so spread (price units) can be converted
        # into fractional return drag consistent with env reward units.
        self._price = None
        try:
            if hasattr(env, "data") and isinstance(env.data, pd.DataFrame):
                if "price" in env.data.columns:
                    self._price = pd.to_numeric(env.data["price"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
                elif "mid_close" in env.data.columns:
                    self._price = pd.to_numeric(env.data["mid_close"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        except Exception:
            self._price = None

    def reset(self, *args, **kwargs):
        self.t = 0
        self.prev_action = 0
        self._fallback_t = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.spread is not None and self.slip is not None:
            try:
                if action != self.prev_action:
                    # IMPORTANT:
                    # TradingEnv computes reward using the *next* bar (idx+1) then increments idx.
                    # After env.step(), env.idx points to the bar that generated this reward.
                    env_idx = getattr(self.env, "idx", None)
                    if env_idx is None:
                        env_idx = self._fallback_t
                    env_idx = int(env_idx)

                    c_sp = float(self.spread[env_idx]) if 0 <= env_idx < int(self.spread.shape[0]) else 0.0
                    c_sl = float(self.slip[env_idx]) * 1e-4 if 0 <= env_idx < int(self.slip.shape[0]) else 0.0

                    # Convert spread to fractional return drag if we have a price series.
                    if self._price is not None and 0 <= env_idx < int(self._price.shape[0]):
                        px = float(self._price[env_idx])
                        if np.isfinite(px) and px > 0:
                            c_sp = c_sp / px

                    total_cost = self.cost_scale * (c_sp + c_sl) + self.turnover_penalty
                    reward = float(reward) - float(total_cost)

                    # Helpful audit hook (ignored by the rest of the pipeline)
                    try:
                        if isinstance(info, dict):
                            info["tx_cost"] = float(total_cost)
                    except Exception:
                        pass
            except Exception:
                pass
        self.prev_action = action
        self._fallback_t += 1
        return state, reward, done, info

    # proxy everything else to the wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)


class RewardProcessWrapper:
    """
    Optional wrapper for the environment's reward to stabilise learning.

    Supports three transforms (can combine):
    - clip_mode = "tanh":  r' = tanh(k * r)
    - clip_mode = "range": r' = clip(r, lo, hi)
    - norm = True: running mean-variance normalization (like PPO/A2C baselines).
    """
    def __init__(
        self,
        env,
        clip_mode=None,
        tanh_k=3.0,
        clip_range=(-1.0, 1.0),
        norm=True,
        norm_beta=0.99,
    ):
        self.env = env
        self.clip_mode = clip_mode
        self.tanh_k = float(tanh_k)
        self.clip_range = tuple(clip_range)
        self.norm = bool(norm)
        self.norm_beta = float(norm_beta)
        self._running_mean = 0.0
        self._running_var = 1.0

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        r = float(reward)

        # Optional clipping
        if self.clip_mode == "tanh":
            import numpy as _np
            r = float(_np.tanh(self.tanh_k * r))
        elif self.clip_mode == "range":
            lo, hi = self.clip_range
            r = float(max(lo, min(hi, r)))

        # Optional running normalization
        if self.norm:
            m = self._running_mean
            v = self._running_var
            beta = self.norm_beta
            m_new = (1 - beta) * r + beta * m
            v_new = (1 - beta) * ((r - m) ** 2) + beta * v
            self._running_mean, self._running_var = m_new, v_new
            if v_new > 0:
                r = (r - m_new) / (v_new ** 0.5)

        return obs, r, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)

import sys        
import os
LOG_MODE = os.getenv("LOG_MODE", "COMPACT").upper()

def log_print(msg: str, level: str = "COMPACT") -> None:
    """
    Centralized logger respecting LOG_MODE.
    SILENT  -> no per-fold prints
    COMPACT -> summary tables only (default)
    DEBUG   -> everything
    """
    if LOG_MODE == "SILENT":
        return
    if LOG_MODE == "COMPACT" and level == "DEBUG":
        return
    print(msg)

# ============================
# Bar-by-bar comparison helpers
# ============================

def _build_bar_compare_dict(all_bt: dict) -> dict:
    """
    Build {name: Series} with 'BH' and each '<model>_equity' from backtester.bar_concat.

    Each backtester should have `bar_concat` with columns:
        - 'cstrategy_cont'  (strategy cumulative equity per bar)
        - 'creturns_cont'   (buy-and-hold cumulative equity per bar)
    """
    series = {}
    for name, bt in (all_bt or {}).items():
        bc = getattr(bt, "bar_concat", None)
        if bc is None or getattr(bc, "empty", True):
            continue

        # Ensure expected column names (fallback if user code set them differently)
        cols = list(bc.columns)
        if "cstrategy_cont" not in cols or "creturns_cont" not in cols:
            # If there are at least two columns, map first->cstrategy, second->bh
            if len(cols) >= 2:
                bc = bc.copy()
                bc.columns = ["cstrategy_cont", "creturns_cont"][:len(cols)]

        # Set BH once (from the first available backtester)
        if "BH" not in series and "creturns_cont" in bc.columns:
            series["BH"] = bc["creturns_cont"]

        # Add this model's equity if present
        if "cstrategy_cont" in bc.columns:
            series[f"{name}_equity"] = bc["cstrategy_cont"]

    return series

def compute_drawdown_curve(equity: "pd.Series") -> "pd.Series":
    """
    Public helper: convert an equity curve ([0xd7]) into a drawdown curve in percent.

    Uses the internal _compute_drawdown (fractional drawdown, negative values)
    and scales by 100.
    """
    import pandas as pd
    dd_frac = _compute_drawdown(equity)
    if dd_frac is None or len(dd_frac) == 0:
        return pd.Series(dtype=float)
    return dd_frac * 100.0


def compute_rolling_sharpe_series(
    equity: "pd.Series",
    window: int,
    frequency_per_year: float | None = None,
) -> "pd.Series":
    """
    Compute an annualised rolling Sharpe ratio from an equity curve ([0xd7]).

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity ([0xd7]), strictly positive.
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


def _pretty_bar_label_global(col: str) -> str:
    """
    Map raw column names like 'cnn_equity' -> 'cnn',
    and ensemble names to short tags, for group plots.
    """
    name = str(col)
    if name == "BH":
        return "Buy & Hold"

    if name.endswith("_equity"):
        name = name[:-7]

    if name == "ensemble_adaptive_regime":
        return "ens_adaptive"
    if name == "ensemble_cnn_lstm_xgboost":
        return "ens_cnn"

    return name


def save_model_underwater_outputs(
    bt_dict,
    models=None,
    out_prefix="results/model_underwater",
    style="nature",
    palette="okabe_ito_no_black",
    bh_color="#666666",
    n_time_parts=10,
    dpi=300,
    line_width=1.0,
    out_dir=None,
    csv_dir=None,
    png_dir=None,
    overlap_mode="intersection",
):
    """
    Multi-model underwater/drawdown plot across models + BH.

    bt_dict can be either:
      - {name: Series} with cumulative equity ([0xd7]) per model and 'BH', or
      - {name: MLBacktester} with .bar_concat containing cstrategy_cont/creturns_cont.
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

    if not bt_dict:
        print("[WARN][0xfe0f] Empty bt_dict passed to save_model_underwater_outputs.")
        return None

    # Same logic as bar comparison for building the DataFrame
    def _build_df_from_bt_dict(d, models_filter=None):
        if all(hasattr(v, "index") and not isinstance(v, (dict, list)) for v in d.values()):
            return pd.DataFrame(d)
        return build_model_bar_compare_df(d, models=models_filter)

    df = _build_df_from_bt_dict(bt_dict, models_filter=models)
    if df is None or df.empty:
        print("[WARN][0xfe0f] No data to plot (underwater).")
        return None

    # Filter to requested models (keep BH if present)
    if models:
        wanted = []
        for m in models:
            col = f"{m}_equity" if m != "BH" and (f"{m}_equity" in df.columns) else m
            if col in df.columns:
                wanted.append(col)
        if wanted:
            keep = (["BH"] if "BH" in df.columns else []) + wanted
            df = df.loc[:, [c for c in keep if c in df.columns]]
        else:
            print(f"[WARN][0xfe0f] None of the requested models found for underwater plot: {models}")
            return None

    # Ensure datetime index & sorted
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    df = df.sort_index()
    
    df = _extend_index_to_calendar_start(df)

    # Rebase ONLY when explicitly requested.
    #
    # IMPORTANT:
    # build_model_bar_compare_df() already pulls continuous equities
    # (e.g., cstrategy_cont / creturns_cont). Rebasing here would
    # incorrectly "reset" curves to 1.0 at each series' first valid bar.
    if overlap_mode == "union_rebase":
        for c in df.columns:
            s = df[c]
            first = s.dropna().iloc[0] if s.dropna().size else np.nan
            if np.isfinite(first) and first != 0.0:
                df[c] = s / first

    # NEW: neutral-fill before first trade so underwater curves exist
    # from day one (drawdown = 0% until the model actually trades).
    df = _neutral_fill_before_first_trade(df, skip_cols=None)

    if overlap_mode == "intersection":
        df_plot = df.dropna(how="any")
    else:
        df_plot = df.replace([np.inf, -np.inf], np.nan).ffill()


    # Output paths (same folder logic as bar comparison)
    if (csv_dir is not None) or (png_dir is not None):
        if csv_dir is None:
            csv_dir = png_dir
        if png_dir is None:
            png_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "model_bar_underwater.csv")
        png_path = os.path.join(png_dir, "model_bar_underwater.png")
    elif out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "model_bar_underwater.csv")
        png_path = os.path.join(out_dir, "model_bar_underwater.png")
    else:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        csv_path = f"{out_prefix}.csv"
        png_path = f"{out_prefix}.png"

    # Build drawdown DataFrame (in %) for CSV
    dd_df = pd.DataFrame(index=df_plot.index)
    for c in df_plot.columns:
        dd_df[_pretty_bar_label_global(c)] = compute_drawdown_curve(df_plot[c])

    try:
        dd_df.to_csv(csv_path, index=True, float_format="%.10f")
    except Exception as e:
        print(f"[WARN][0xfe0f] Could not save underwater CSV: {e}")

    # If you already swapped to set_paper_style in the previous step,
    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, ax = plt.subplots(constrained_layout=True)

        # --- Buy & Hold underwater (dashed, lightly shaded) -----------------
        if "BH" in df_plot.columns:
            dd = compute_drawdown_curve(df_plot["BH"])
            # shaded area under waterline
            ax.fill_between(
                dd.index,
                dd.values,
                0.0,
                color=bh_color,
                alpha=0.10,
                linewidth=0.0,
                zorder=1,
            )
            # dashed line on top
            ax.plot(
                dd.index,
                dd.values,
                linestyle="--",
                linewidth=line_width,
                color=bh_color,
                label="Buy & Hold",
                zorder=2,
            )
                
        for c in df_plot.columns:
            if c == "BH":
                continue

            dd = compute_drawdown_curve(df_plot[c])

            color = ax._get_lines.get_next_color()

            ax.fill_between(
                dd.index,
                dd.values,
                0.0,
                color=color,
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                dd.index,
                dd.values,
                color=color,
                linewidth=line_width,
                label=_pretty_bar_label_global(c),
                zorder=2,
            )


        ax.set_title("Underwater / Drawdown Curve (Intersection)", pad=12)
        ax.set_xlabel("Date")          # was 'Time'
        ax.set_ylabel("Drawdown (%)")
        ax.axhline(0.0, linewidth=1.0, linestyle="--")

        # Light grid is already enabled by the style, but this is harmless
        ax.grid(True)

        try:
            _set_even_time_ticks(
                ax,
                df_plot.index,
                n_parts=n_time_parts,
                fmt="%d/%m/%y",
                rotation=30,   # was 45[0xb0], a bit cleaner tilted less
            )
        except Exception:
            pass

        ax.margins(x=0)

        # Bottom band of model labels (excluding BH)
        model_label_names = [
            _pretty_bar_label_global(c)
            for c in df_plot.columns
            if c != "BH"
        ]
        
        ax.margins(x=0)

        # Legend on the right (no band under the x-axis)
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            fig.subplots_adjust(right=0.80)
            ax.legend(
                handles,
                labels_,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

        plt.close(fig)

    return png_path


def save_model_rolling_performance_outputs(
    bt_dict,
    models=None,
    out_prefix="results/model_rolling_sharpe",
    style="nature",
    palette="okabe_ito_no_black",
    bh_color="#666666",
    n_time_parts=10,
    dpi=300,
    line_width=1.0,   # clearer lines
    out_dir=None,
    csv_dir=None,
    png_dir=None,
    overlap_mode="intersection",
    window_bars=None,
):
    """
    Multi-model rolling Sharpe plot across models + BH.

    bt_dict can be either:
      - {name: Series} with cumulative equity ([0xd7]) per model and 'BH', or
      - {name: MLBacktester} with .bar_concat containing cstrategy_cont/creturns_cont.
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

    if not bt_dict:
        print("[WARN][0xfe0f] Empty bt_dict passed to save_model_rolling_performance_outputs.")
        return None

    def _build_df_from_bt_dict(d, models_filter=None):
        """Build a wide bar-by-bar DataFrame from either:
        (A) a dict of Series/DataFrames keyed by model name, or
        (B) a dict of {model: MLBacktester} objects.
        Robust to duplicate timestamp labels inside any input Series.
        """
        # Case (A): dict of Series/DataFrames (already evaluated outputs)
        # Otherwise treat it as {model: MLBacktester}
        if all(hasattr(v, "index") and not isinstance(v, (dict, list)) for v in d.values()):
            clean = {}
            for k, v in d.items():
                s = v
                # If a DataFrame sneaks in, pick a single column deterministically
                if isinstance(s, pd.DataFrame):
                    if s.shape[1] == 1:
                        s = s.iloc[:, 0]
                    else:
                        # Prefer a column matching the key if it exists, else first column
                        s = s[k] if k in s.columns else s.iloc[:, 0]

                # Coerce to Series (best-effort)
                if not isinstance(s, pd.Series):
                    try:
                        s = pd.Series(s)
                    except Exception:
                        continue

                s = s.copy()

                # Normalize index to timezone-aware timestamps when possible
                try:
                    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
                except Exception:
                    pass

                # Drop NaT and duplicate timestamp labels (keep last)
                if hasattr(s.index, "isna"):
                    s = s.loc[~s.index.isna()]

                if getattr(s.index, "has_duplicates", False):
                    dup_n = int(s.index.duplicated(keep="last").sum())
                    s = s.loc[~s.index.duplicated(keep="last")]
                    # Keep this visible: duplicate bars usually signal an upstream merge/concat issue
                    if dup_n > 0 and str(os.environ.get("LOG_MODE","")).upper() == "DEBUG":
                        print(f"[BarCompare] Dropped {dup_n} duplicate timestamps for '{k}' (kept last).")

                s = s.sort_index()
                clean[k] = s

            return pd.DataFrame(clean)

        # Case (B): dict of {model: MLBacktester} and build from their stored results
        return build_model_bar_compare_df(d, models=models_filter)

    df = _build_df_from_bt_dict(bt_dict, models_filter=models)
    if df is None or df.empty:
        print("[WARN][0xfe0f] No data to plot (rolling performance).")
        return None

    # Filter to requested models (keep BH if present)
    if models:
        wanted = []
        for m in models:
            col = f"{m}_equity" if m != "BH" and (f"{m}_equity" in df.columns) else m
            if col in df.columns:
                wanted.append(col)
        if wanted:
            keep = (["BH"] if "BH" in df.columns else []) + wanted
            df = df.loc[:, [c for c in keep if c in df.columns]]
        else:
            print(f"[WARN][0xfe0f] None of the requested models found for rolling performance plot: {models}")
            return None

    # Ensure datetime index & sorted
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    df = df.sort_index()

    # Rebase to 1.0 at first valid
    for c in df.columns:
        s = df[c]
        first = s.dropna().iloc[0] if s.dropna().size else np.nan
        if np.isfinite(first):
            df[c] = s / first

    # Make all models neutral before first trade
    df = _neutral_fill_before_first_trade(df, skip_cols=None)

    # Overlap handling
    if overlap_mode == "intersection":
        df_plot = df.dropna(how="any")
    else:
        df_plot = df.replace([np.inf, -np.inf], np.nan).ffill()

    if df_plot.empty:
        print("[WARN][0xfe0f] No overlapping data for rolling performance plot.")
        return None

    # Infer frequency & default window if needed
    try:
        freq_year = float(estimate_frequency_per_year(df_plot.index))
    except Exception:
        freq_year = 252.0

    if window_bars is None:
        # ~1-month window in bars, capped for sanity
        window_bars = max(20, min(1000, int(freq_year / 12.0)))

    # Compute rolling Sharpe per series, using short labels
    rs_df = pd.DataFrame(index=df_plot.index)
    for c in df_plot.columns:
        pretty = _pretty_bar_label_global(c)
        short = _short_model_label(pretty)
        rs_df[short] = compute_rolling_sharpe_series(
            df_plot[c],
            window=window_bars,
            frequency_per_year=freq_year,
        )

    # Fill pre-window period so curves visually start on day 1
    rs_df = _neutral_fill_before_first_trade(rs_df, skip_cols=None)

    # Keep curves visually continuous: if volatility is zero in a window the
    # Sharpe is mathematically undefined (NaN). For plotting and CSV we simply
    # forward-fill those NaNs so that the last valid Sharpe estimate is carried
    # through flat, instead of breaking the line.
    rs_df = rs_df.ffill()

    # Output paths
    if (csv_dir is not None) or (png_dir is not None):
        if csv_dir is None:
            csv_dir = png_dir
        if png_dir is None:
            png_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "model_rolling_sharpe.csv")
        png_path = os.path.join(png_dir, "model_rolling_sharpe.png")
    elif out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "model_rolling_sharpe.csv")
        png_path = os.path.join(out_dir, "model_rolling_sharpe.png")
    else:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        csv_path = f"{out_prefix}.csv"
        png_path = f"{out_prefix}.png"

    try:
        rs_df.to_csv(csv_path, index=True, float_format="%.10f")
    except Exception as e:
        print(f"[WARN][0xfe0f] Could not save rolling Sharpe CSV: {e}")

    with set_paper_style(
        style=style,
        palette=palette,
        bw_line_styles=(palette == "print_bw"),
    ):
        fig, ax = plt.subplots(constrained_layout=True)

        # BH column name in rs_df (short label)
        bh_label = _short_model_label(_pretty_bar_label_global("BH"))

        # --- Plot BH rolling Sharpe first, if available ---------------------
        if bh_label in rs_df.columns:
            ax.plot(
                rs_df.index,
                rs_df[bh_label].astype(float).values,
                linestyle="--",
                linewidth=line_width,
                color=bh_color,
                label="Buy & Hold",
                zorder=2,
            )

        # --- Plot model rolling Sharpe curves -------------------------------
        for col in rs_df.columns:
            if col == bh_label:
                continue

            color = ax._get_lines.get_next_color()
            ax.plot(
                rs_df.index,
                rs_df[col].astype(float).values,
                linewidth=line_width,
                label=col,
                color=color,
                zorder=3,
            )

        ax.set_title(f"Rolling Sharpe (~{window_bars} bars window)", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Sharpe")
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.grid(True)

        # 10 equal parts along the x-axis, like the other graphs
        try:
            _set_even_time_ticks(
                ax,
                rs_df.index,
                n_parts=n_time_parts,   # default = 10
                fmt="%d/%m/%y",
                rotation=45,
            )
        except Exception:
            pass

        ax.margins(x=0)

        # Legend on the right, outside the plot area
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            fig.subplots_adjust(right=0.80)
            ax.legend(
                handles,
                labels_,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

        fig.set_size_inches(11.0, 4.8)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return png_path


def _neutral_fill_before_first_trade(df, skip_cols=None):
    """
    For plotting: extend each series' first valid value backwards so that
    all models are defined from the earliest timestamp.

    This avoids dropping early dates in 'intersection' mode when some models
    only start trading later in the month. We keep curves flat (neutral)
    before the first trade.
    """
    import numpy as np

    if df is None or df.empty:
        return df

    if skip_cols is None:
        skip_cols = []

    for c in df.columns:
        if c in skip_cols:
            continue

        s = df[c]
        if not s.notna().any():
            # nothing to do if the column is entirely NaN
            continue

        first_idx = s.first_valid_index()
        if first_idx is None:
            continue

        first_val = s.loc[first_idx]
        if np.isfinite(first_val):
            # Set everything up to and including first_idx to that neutral level
            df.loc[:first_idx, c] = first_val

    return df

def _extend_index_to_calendar_start(df):
    """
    Plotting helper: extend the index backwards to the first calendar day
    of the month of the first timestamp, keeping all new rows NaN.

    Later, _neutral_fill_before_first_trade() will forward-fill these rows
    to the first equity level, so curves look flat (neutral) before the
    first real trading bar.

    Does NOT touch the original backtest objects; only the copy used
    inside save_model_* plotting functions.
    """
    import pandas as pd
    import numpy as np

    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = df.index.sort_values()
    first_ts = idx[0]

    # Anchor: first day of that month (midnight)
    month_start_date = first_ts.normalize().replace(day=1)

    # If we already start on day 1, nothing to do
    if month_start_date == first_ts.normalize():
        return df

    # Daily dates from month start up to the day BEFORE first_ts
    extra_days = pd.date_range(
        start=month_start_date,
        end=first_ts.normalize() - pd.Timedelta(days=1),
        freq="D",
        tz=idx.tz,
    )
    if extra_days.empty:
        return df

    # Keep the same time-of-day as the first real bar
    offset = first_ts - first_ts.normalize()
    extra_index = extra_days + offset

    # Create empty rows (NaN) for all columns on those dates
    extra_df = pd.DataFrame(index=extra_index, columns=df.columns, dtype=float)

    # Concatenate and sort - later we neutral-fill these NaNs
    df_ext = pd.concat([extra_df, df]).sort_index()

    return df_ext

# --- Persistent HPO config storage -------------------------------------------
# Stores best Optuna configs in a global "hpo" folder, sibling to "results".
# You can override the base dir with the MLB_HPO_DIR environment variable.

import os
import json

# Repo root = directory that contains utilsNoWFO.py
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_HPO_BASE_DIR = os.environ.get(
    "MLB_HPO_DIR",
    os.path.join(_REPO_ROOT, "hpo"),
)
os.makedirs(_HPO_BASE_DIR, exist_ok=True)


def get_hpo_config_dir() -> str:
    """Return the directory where persistent HPO configs are stored."""
    return _HPO_BASE_DIR


def _hpo_config_path(model_type: str) -> str:
    safe = str(model_type).replace("/", "_")
    return os.path.join(_HPO_BASE_DIR, f"{safe}_best_config.json")


def _sanitize_for_json(obj):
    """
    Recursively replace NaN / +/-inf with None so that json.dump produces
    valid JSON. Leaves normal numbers/strings/bools untouched.
    """
    import math

    # Dict -> sanitize values
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    # List / tuple -> sanitize each element
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # Plain Python floats/ints
    if isinstance(obj, (float, int)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    # Try to handle numpy scalar types if numpy is installed
    try:
        import numpy as _np  # type: ignore

        if isinstance(obj, (_np.floating, _np.integer)):
            v = float(obj)
            if not math.isfinite(v):
                return None
            return v
    except Exception:
        pass

    # Everything else (str, bool, None, etc.) -> keep as is
    return obj


def save_hpo_config_to_disk(
    model_type: str,
    best_params: dict,
    topN_params: list | None = None,
    study_meta: dict | None = None,
) -> str:
    """
    Persist the best Optuna configuration for a model so that other runs
    (e.g. real_trading_simulation) can reuse it.

    Returns the path of the JSON file written.
    """
    from typing import Any
    payload: dict[str, Any] = {
        "model_type": model_type,
        "best_params": best_params or {},
        "topN_params": topN_params or [],
        "study_meta": study_meta or {},
    }
    path = _hpo_config_path(model_type)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        try:
            # log_print is defined later in this module; it's fine to call here

            log_print(f"[HPO] Saved best config for {model_type} -> {path}", level="COMPACT")
        except Exception:
            print(f"[HPO] Saved best config for {model_type} -> {path}")
    except Exception as e:
        try:
            log_print(f"[HPO] Failed to save best config for {model_type}: {e}", level="COMPACT")
        except Exception:
            print(f"[HPO] Failed to save best config for {model_type}: {e}")
    return path


def load_hpo_config_from_disk(
    model_type: str,
) -> tuple[dict | None, list | None, dict | None]:
    """
    Load the persistent HPO configuration for a model_type, if available.

    Returns (best_params, topN_params, study_meta). Each element may be None.
    """
    path = _hpo_config_path(model_type)
    if not os.path.exists(path):
        return None, None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best_params") or None
        topN = payload.get("topN_params") or None
        meta = payload.get("study_meta") or None
        try:
            log_print(f"[HPO] Loaded saved config for {model_type} from {path}", level="COMPACT")
        except Exception:
            print(f"[HPO] Loaded saved config for {model_type} from {path}")
        return best, topN, meta
    except Exception as e:
        try:
            log_print(f"[HPO] Failed to load config for {model_type} from {path}: {e}", level="COMPACT")
        except Exception:
            print(f"[HPO] Failed to load config for {model_type} from {path}: {e}")
        return None, None, None

import math

def _norm_optuna_direction(direction: str | None) -> str:
    """
    Normalize direction string.
    Returns: "maximize" or "minimize"
    """
    d = str(direction or "maximize").strip().lower()
    return "minimize" if d == "minimize" else "maximize"

def _bad_objective_for_direction(direction: str | None, magnitude: float = 9999.0) -> float:
    d = _norm_optuna_direction(direction)
    m = float(magnitude)
    # maximize => very low is worst; minimize => very high is worst
    return -m if d == "maximize" else +m