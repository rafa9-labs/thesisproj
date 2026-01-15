"""
MLBacktester — Backtesting and real-trading simulation harness.

This module orchestrates model training/selection and real-time style
simulation across classical ML, deep learning, RL (DQN), and ensembles.
"""
# export $(grep -v '^#' .env | xargs)



# ── Standard Library ──────────────────────────────────────────────────────────
import gc as _gc
import glob
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional

# TensorFlow: quiet logs by default and avoid full-GPU preallocation.
# Users can override these via environment variables before launching Python.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # avoid full pre-allocation

# Keep a separate alias for any code that expects `_os`
_os = os

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import optuna
import pandas as pd
import psutil
import tensorflow as tf
import xgboost as xgb
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow import keras
from tqdm import tqdm
from xgboost import XGBClassifier

# scikit-learn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

# Keras callback used in deep-model time caps
Callback = keras.callbacks.Callback
mixed_precision = keras.mixed_precision

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env", override=False)  # read .env without clobbering exported env
except Exception:
    pass

# ── Optional / Soft Dependencies (safe to keep at top) ───────────────────────
try:
    # CSV engine preference used by _load_csv_cached()
    import pyarrow  # noqa: F401
    _CSV_ENGINE = "pyarrow"
except Exception:
    _CSV_ENGINE = "c"

try:
    # Avoid BLAS oversubscription inside blocks
    from threadpoolctl import threadpool_limits as _tp_limits
except Exception:
    _tp_limits = None

try:
    # Used by free() to close figures if any were created
    import matplotlib.pyplot as _plt
except Exception:
    _plt = None

try:
    # Used only for cleanup without touching the main 'tf' alias
    import tensorflow as _tf
except Exception:
    _tf = None

try:
    # Heavy; used in prepare_features for indicators if enabled
    import ta  # noqa: F401
except Exception:
    ta = None

# ── Project-Local (this repo) ────────────────────────────────────────────────
from models.cnn import build_cnn
from models.lstm import build_lstm
from models.transformer import build_transformer

from rl.environment import TradingEnv
from rl.dqn_agent import DQNAgent, filter_dqn_config

from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
from models.ensemble_adaptive_regime import AdaptiveRegimeStrategy

from tuningNoWFO import run_optuna_tuning, final_refit_if_deep, _evaluate_original_no_refit

from utilsNoWFO import (
    # core utils & logging
    set_global_determinism,
    TRAIN_TEST_MONTHS, N_METRICS, METRIC_NAMES,
    ensure_metric_tuple, validate_metrics_shape,

    # I/O / paths
    make_results_run_dir, ensure_model_dirs, comparison_dirs, month_dir_path,
    save_model_bar_comparison_outputs,
    save_model_underwater_outputs,
    save_model_rolling_performance_outputs,
    save_group_equity_curves,
    build_trade_log_from_df,
    save_feature_frequency_from_monthly_results,

    # features & labels
    RollingStandardizer,
    add_cyclic_hour_features,
    build_features_from_params,
    realized_vol, bipower_variation, fracdiff, triple_barrier_labels,
    attach_macro_features,


    # calibration / gating / coverage
    calibrate_prefit_and_predict_proba, ConformalClassifier, sanitize_proba,
    print_feature_stats, print_conf_stats,
    fit_temperature_from_proba, apply_temperature_to_proba,
    fit_coverage_threshold_on_calibration, freeze_confidence_threshold, is_coverage_intent,  enforce_target_coverage_policy, target_coverage_policy,

    # evaluation & metrics
    _build_bar_compare_dict,
    compute_full_evaluation_metrics, combine_block_scores,
    enforce_day1_eval_anchor, first_tradable_test_bar,
    compute_required_test_warmup_bars,

    # run tree helpers
    init_study_tree, model_category, 

    # naming
    friendly_model_name,
    
    # to organize
    save_feature_heatmap_for_single_month,
    save_month_equity_graph,
    save_monthly_model_stats,
    filter_params,
    ensure_dict,
    ensure_list,
    estimate_bars_per_day, 
    compute_rolling_hit_rate,
    find_hit_rate_switch_idx,
    build_model_monthly_pivots,
    build_model_ranking,
    save_model_ranking_csv,
    _fmt_table_ascii,
    
    prefilter_features_train,
    realized_vol,
    
    CostAwareWrapper, RewardProcessWrapper,
    compute_brier_and_nll, log_print,
    
    _ensure_dt,
    SKIP_PLOTS,
)


# --- Quiet TensorFlow logs (no determinism; keep random seeding) ---
import os
# Hide INFO and WARNING logs from TensorFlow; keep ERRORs visible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Tidy shared library path once (helps cut duplicate plugin scans)
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _ld:
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(_ld.split(":")))

# ── Data paths & constants ────────────────────────────────────────────────────
CSV_1H    = "csv_data/EURUSD_10_years_H1_OANDA.csv"
CSV_4H    = "csv_data/EURUSD_10_years_H4_OANDA.csv"
CSV_15MIN = os.environ.get("CSV_15MIN", "csv_data/EURUSD_10_years_M15_OANDA.csv")
CSV_30MIN = os.environ.get("CSV_30MIN", "csv_data/EURUSD_10_years_M30_OANDA.csv")
BASE_CSV  = CSV_30MIN  # switch base timeframe to 15m

# pandas preference used elsewhere in the file
pd.options.mode.copy_on_write = True

# --- Runtime performance knobs (WSL + 3090) -----------------------------------
import os, multiprocessing
try:
    import psutil
except Exception:
    psutil = None

try:
    from threadpoolctl import threadpool_limits as _tp_limits
except Exception:
    _tp_limits = None

# GPU allow-growth (no full preallocation)
try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
        log_print("✅ Enabled dynamic GPU memory allocation.", level="COMPACT")
except Exception:
    pass

# Force-CPU escape hatch (optional)
if os.environ.get("TF_FORCE_CPU", "0") == "1":
    try:
        tf.config.set_visible_devices([], "GPU")
        log_print("⚠️ Forcing CPU: disabled all GPUs because TF_FORCE_CPU=1.", level="COMPACT")
    except Exception:
        pass


CPU_TOTAL  = os.cpu_count() or multiprocessing.cpu_count() or 8
# Prefer an explicit BLAS_THREADS_PER_TRIAL from the environment if set;
# otherwise fall back to MLB_THREADS or (cores - 2).
_blas_env = os.getenv("BLAS_THREADS_PER_TRIAL", "").strip()
if _blas_env:
    SAFE_CORES = max(1, int(_blas_env))
else:
    SAFE_CORES = int(os.getenv("MLB_THREADS", "0")) or max(1, CPU_TOTAL - 2)

for var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
    "SKLEARN_JOBS", "XGB_JOBS", "RF_JOBS", "CV_JOBS", "BLAS_THREADS_PER_TRIAL"
):
    os.environ.setdefault(var, str(SAFE_CORES))

# Ensure BLAS_THREADS_PER_TRIAL is in sync with SAFE_CORES
os.environ["BLAS_THREADS_PER_TRIAL"] = str(SAFE_CORES)

# Apply a live BLAS / OpenMP cap (NumPy, scikit-learn, XGB)
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=SAFE_CORES)
except Exception as e:
    print(f"⚠️ threadpool_limits failed: {e}")


# TensorFlow thread tuning
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(SAFE_CORES)
    tf.config.threading.set_inter_op_parallelism_threads(min(4, SAFE_CORES // 2))
    print(f"🧵 TF threads: intra={SAFE_CORES}, inter={min(4, SAFE_CORES // 2)}")
except Exception as e:
    print(f"⚠️ TF thread setup skipped: {e}")

print(f"🧩 Trial thread budget = {SAFE_CORES} cores active per model fit.")


# Lock TensorFlow to the same thread counts (if available)
try:
    tf.config.threading.set_intra_op_parallelism_threads(SAFE_CORES)
    tf.config.threading.set_inter_op_parallelism_threads(SAFE_CORES)
except Exception:
    pass

# Optional global BLAS cap (keeps numpy/scikit/xgb consistent for this process)
if _tp_limits:
    _tp_limits(limits=SAFE_CORES).__enter__()  # process-lifetime cap

# Simple RAM guard print (unchanged; keep if you like)
if psutil:
    TOTAL_GB = float(psutil.virtual_memory().total) / (1024**3)
    RAM_LIMIT_GB = float(os.environ.get("RAM_LIMIT_GB", str(min(0.85 * TOTAL_GB, TOTAL_GB - 2))))
    print(f"🔧 RAM guard: limit={RAM_LIMIT_GB:.2f} GB (of total {TOTAL_GB:.2f} GB)")

print(f"🧵 Parallelism lock: {SAFE_CORES} threads (logical {CPU_TOTAL}). "
      f"Override any time via MLB_THREADS=<n>.")


# ── Paths & Configs (DQN + Features) ─────────────────────────────────────────
MODEL_DQN_PATH             = "DQNSavedModels/dqn_model.keras"
DQN_GRID_CONFIG_PATH       = "configs/dqn_grid_config.json"
DQN_AGENT_CONFIG_PATH      = "DQNSavedModels/dqn_model_config.json"
FEATURES_PATH              = "configs/feature_config.json"

# Global CSV cache (process-level)
_DATA_CACHE = {}

# --- compact logging defaults (hard-off for debug) ---
LOG_MODE = os.getenv("LOG_MODE", "COMPACT").upper()

def _norm_class_counts(d: object) -> dict:
    """
    Normalize class-count dict keys to plain ints.

    value_counts() often yields keys like np.int64(-1) or -1.0. If logs later do
    raw.get(-1), they miss and show fake zeros.

    Telemetry-only: does not affect trading logic or metrics.
    """
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def print_block_summary(block_id, calib_info, gate_info, reliability,
                        class_dists, block_stats, fold_label: str = "Mini-Block Fold") -> None:
    """
    Per-fold compact summary. All fields are already precomputed by the caller.
    """
    line = "─" * 70
    log_print(f"\n\n\n{line}\n{fold_label} #{block_id}\n{line}", level="COMPACT")

    try:
        _bars_total = int(calib_info.get("bars_total", calib_info.get("bars", 0)) or 0)
    except Exception:
        _bars_total = int(calib_info.get("bars", 0) or 0)
    try:
        _bars_elig = int(calib_info.get("bars_eligible", calib_info.get("bars", _bars_total)) or 0)
    except Exception:
        _bars_elig = int(calib_info.get("bars", _bars_total) or 0)

    log_print(
        f"Coverage target {calib_info['target']:.2f} | "
        f"conf_thr {calib_info['conf_thr']:.3f} | "
        f"bars total {_bars_total} | eligible {_bars_elig}",
        level="COMPACT",
    )
    log_print(
        "Dynamic αβγ → "
        f"base={gate_info['base']:.3f} "
        f"α={gate_info['alpha']:.3f} "
        f"β={gate_info['beta']:.3f} "
        f"γ={gate_info['gamma']:.3f} | "
        f"median_thr={gate_info['median_thr']:.3f}",
        level="COMPACT",
    )
    
    try:
        _rows_total = int(block_stats.get("rows_total", block_stats.get("rows", 0)) or 0)
    except Exception:
        _rows_total = int(block_stats.get("rows", 0) or 0)
    try:
        _rows_elig = int(block_stats.get("rows_eligible", block_stats.get("rows", _rows_total)) or 0)
    except Exception:
        _rows_elig = int(block_stats.get("rows", _rows_total) or 0)

    # ---- Sharpe string computed once (avoid nested f-strings) ----
    _sr_val = block_stats.get("sr", float("nan"))
    try:
        _sr_val = float(_sr_val)
    except Exception:
        _sr_val = float("nan")
    sr_str = "—" if (_sr_val != _sr_val) else f"{_sr_val:+.3f}"

    log_print(
        f"Denoms → "
        f"val_rows={_rows_total} | "
        f"post_feature_bars_total={int(calib_info.get('bars_total', _rows_total) or _rows_total)} | "
        f"eligible={int(calib_info.get('bars_eligible', _rows_elig) or _rows_elig)} | "
        f"eval_bars={int(block_stats.get('rows', _rows_elig) or _rows_elig)}   "
        f"trades={block_stats['trades']}   "
        f"active_rate={block_stats['ar']:.3f}   "
        f"Sharpe={sr_str}",
        level="COMPACT",
    )

    log_print(
        f"Coverage nudge → band ±{gate_info['band']:.2f} "
        f"step {gate_info['step']:.3f}",
        level="COMPACT",
    )
    log_print(
        "Reliability → "
        f"PSRα={reliability['psr_alpha']:.2f} "
        f"cutoff={reliability['cutoff']:.2f} "
        f"min_trades={reliability['min_trades']} "
        f"indep={reliability['min_indep']}",
        level="COMPACT",
    )

    log_print(line, level="COMPACT")
    raw = _norm_class_counts(class_dists.get("raw", {}))
    final = _norm_class_counts(class_dists.get("final", {}))
    
    log_print(
        "Class dist (raw)     "
        f"-1:{raw.get(-1, 0)}   "
        f"0:{raw.get(0, 0)}   "
        f"+1:{raw.get(1, 0)}",
        level="COMPACT",
    )
    log_print(
        "After filter          "
        f"0:{final.get(0, 0)}   "
        f"-1:{final.get(-1, 0)}  "
        f"+1:{final.get(1, 0)}",
        level="COMPACT",
    )

    log_print(line, level="COMPACT")
    sr = block_stats.get("sr", "—")
    if isinstance(sr, (float, int)) and np.isfinite(sr):
        sr_str = f"{float(sr):.3f}"
    else:
        sr_str = "—"

    log_print(
        f"Denoms → "
        f"val_rows={_rows_total} | "
        f"post_feature_bars_total={int(calib_info.get('bars_total', _rows_total) or _rows_total)} | "
        f"eligible={int(calib_info.get('bars_eligible', _rows_elig) or _rows_elig)} | "
        f"eval_bars={int(block_stats.get('rows', _rows_elig) or _rows_elig)}   "
        f"trades={block_stats['trades']}   "
        f"active_rate={block_stats['ar']:.3f}   "
        f"Sharpe={sr_str}",
        level="COMPACT",
    )
    
    # Optional: show trade-intent precision for this fold (post confidence gating)
    try:
        p_int = block_stats.get("precision_intent", None)
        n_int = block_stats.get("intent_bars", None)
        p_int = float(p_int) if p_int is not None else float("nan")
        n_int = int(n_int) if n_int is not None else 0
        if (n_int > 0) and (p_int == p_int):  # not NaN
            log_print(f"Intent precision p={p_int:.3f} (n={n_int})", level="COMPACT")
    except Exception:
        pass
    log_print(line, level="COMPACT")
    
def print_pruned_block_summary(
    block_id: int,
    reason: str,
    rows: int | None = None,
    trades: int | None = None,
    active_rate: float | None = None,
    sharpe: float | None = None,
    fold_label: str = "Mini-Block Fold",
) -> None:
    """
    Compact summary for Mini-Block folds that were pruned or marked invalid.
    Mirrors the style of `print_block_summary` but focuses on the prune/invalid reason.
    """
    line = "─" * 70

    # Fallbacks for stats
    rows_str = "—" if rows is None else str(int(rows))
    trades_str = "—" if trades is None else str(int(trades))

    if isinstance(active_rate, (float, int)) and np.isfinite(active_rate):
        ar_str = f"{float(active_rate):.3f}"
    else:
        ar_str = "—"

    if isinstance(sharpe, (float, int)) and np.isfinite(sharpe):
        sr_str = f"{float(sharpe):.3f}"
    else:
        sr_str = "—"

    # Keep reason on a single, not-too-long line
    reason_str = str(reason).replace("\n", " ").strip()
    if len(reason_str) > 200:
        reason_str = reason_str[:197] + "..."

    log_print(
        f"\n\n\n{line}\n{fold_label} #{block_id} [PRUNED / INVALID]\n{line}",
        level="COMPACT",
    )
    log_print(f"Reason: {reason_str}", level="COMPACT")
    log_print(
        f"rows={rows_str}   trades={trades_str}   "
        f"active_rate={ar_str}   Sharpe={sr_str}",
        level="COMPACT",
    )
    log_print(line, level="COMPACT")



def _load_csv_cached(path, parse_dates=None, index_col=None):
    key = (path, tuple(parse_dates or []), index_col)
    if key not in _DATA_CACHE:
        df = pd.read_csv(path, parse_dates=parse_dates, engine=_CSV_ENGINE)
        if index_col:
            df.set_index(index_col, inplace=True)
        _DATA_CACHE[key] = df
    return _DATA_CACHE[key].copy()

# Silence pandas/ta deprecation noise from PSAR internals in ta.trend
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="ta.trend",
)

def _hard_free():
    try:
        import tensorflow as _tf
        _tf.keras.backend.clear_session()
    except Exception:
        pass
    import gc, time
    gc.collect()
    time.sleep(0.05)
    
def _apply_low_ram_overrides(cfg: dict) -> dict:
    """Shrink memory-heavy knobs when RAM is tight or MLB_LOW_RAM=1."""
    import os, psutil
    cfg = dict(cfg or {})
    avail = psutil.virtual_memory().available / (1024 ** 3)
    trigger = float(os.getenv("LOW_RAM_TRIGGER_GB", "1.25"))
    force = os.getenv("MLB_LOW_RAM", "0") in ("1","true","True")
    if not force and avail >= trigger:
        return cfg
    # caps that materially reduce memory:
    cfg["feature_top_k"] = int(min(cfg.get("feature_top_k", 192), 128))
    cfg["ensemble_deep_max_train_windows"] = int(min(cfg.get("ensemble_deep_max_train_windows", 10000), 10000))
    for k in ("cnn_batch_size","lstm_batch_size","transformer_batch_size"):
        if k in cfg: cfg[k] = int(min(64, int(cfg.get(k, 64))))
    for k in ("cnn_epochs","lstm_epochs","transformer_epochs"):
        if k in cfg: cfg[k] = int(min(15, int(cfg.get(k, 20))))
    # XGBoost memory savers
    cfg["xgb_tree_method"] = "hist"
    cfg["xgb_grow_policy"] = "lossguide"
    cfg["xgb_max_bin"] = 256
    cfg["xgb_n_estimators"] = int(min(int(cfg.get("xgb_n_estimators", 350)), 300))
    # Avoid extra scaling buffers
    cfg["use_rolling_scaler"] = False
    print(f"🧊 LOW-RAM overrides applied (avail≈{avail:.2f}GB).")
    return cfg

def _load_default_dqn_cfg(path: str) -> dict:
    """
    Load a baseline DQN config from JSON (e.g. dqn_grid_config.json).
    If the file is missing or invalid, return an empty dict so that
    _coerce_dqn_cfg can fill in safe defaults.
    """
    cfg = {}
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg = json.load(f) or {}
                if isinstance(cfg, dict):
                    cfg.setdefault("_cfg_source", "grid_defaults")
            if not isinstance(cfg, dict):
                print(f"⚠️ DQN config at {path} is not a dict; ignoring.")
                cfg = {}
        else:
            print(f"⚠️ DQN default config file not found at {path}; using built-in defaults.")
    except Exception as e:
        print(f"⚠️ Failed to load DQN default config from {path}: {e}")
        cfg = {}

    return cfg


def _coerce_dqn_cfg(cfg: dict, *, strict: bool = False) -> dict:
    """
    Normalize and guardrail DQN config so training actually runs.

    NOTE (your requirement):
    - If config is loaded from dqn_grid_config.json, we treat it as source-of-truth
      and DO NOT override user values (no clamping).
    - We still fill defaults for missing keys so the program runs.
    """
    cfg = dict(cfg or {})

    # Respect JSON defaults as source-of-truth when marked.
    strict = bool(strict) or (cfg.get("_cfg_source") == "grid_defaults")

    # ---- Start from candidate values / defaults ----
    # In strict mode: do not inflate/alter values beyond basic validity.
    # In non-strict mode: allow a bit of "guardrailing" for coherence.
    def _as_int(key: str, default: int, minv: int | None = None) -> int:
        v = cfg.get(key, default)
        try:
            v = int(v)
        except Exception:
            v = default
        if minv is not None:
            v = max(minv, v)
        return v

    bs   = _as_int("batch_size", 64, minv=1)
    buf  = _as_int("buffer_size", 50000, minv=1)
    warm = _as_int("warmup_steps", max(5000, 2 * bs), minv=0)

    # Keep warmup < buffer
    if warm >= buf:
        if strict:
            warm = max(0, buf - 1)
        else:
            buf = max(buf, warm + bs)

    # Batch must fit in buffer
    if bs > buf:
        if strict:
            bs = buf
        else:
            bs = max(32, buf // 2)

    cfg["batch_size"]   = bs
    cfg["buffer_size"]  = buf
    cfg["warmup_steps"] = warm

    # ---- Fill defaults only (never override explicitly provided keys) ----
    cfg.setdefault("gamma", 0.99)
    cfg.setdefault("epsilon", 1.0)
    cfg.setdefault("epsilon_min", 0.05)
    cfg.setdefault("epsilon_decay", 0.999)          # legacy fallback
    cfg.setdefault("epsilon_decay_steps", 200000)   # default horizon
    cfg.setdefault("learning_rate", 0.0005)
    cfg.setdefault("replay_freq", 4)
    cfg.setdefault("target_update_freq", 2000)
    cfg.setdefault("episodes", 50)
    cfg.setdefault("action_size", 3)

    # ---- Type coercion + basic validity (no “policy” overrides) ----
    # episodes: respect exactly if provided (no max(30, ...))
    try:
        cfg["episodes"] = max(1, int(cfg.get("episodes", 50)))
    except Exception:
        cfg["episodes"] = 50

    # epsilon_decay_steps: strict => only ensure >=1; non-strict => ensure a sane lower bound
    try:
        if strict and "epsilon_decay_steps" in cfg:
            cfg["epsilon_decay_steps"] = max(1, int(cfg["epsilon_decay_steps"]))
        else:
            cfg["epsilon_decay_steps"] = max(1, int(cfg.get("epsilon_decay_steps", 200000)))
    except Exception:
        cfg["epsilon_decay_steps"] = 200000

    # epsilon_min: strict => accept as-is (just coerce to float); non-strict => clamp to [0.01, 0.2]
    try:
        eps_min = float(cfg.get("epsilon_min", 0.05))
        if strict and "epsilon_min" in cfg:
            cfg["epsilon_min"] = eps_min
        else:
            cfg["epsilon_min"] = min(0.2, max(0.01, eps_min))
    except Exception:
        cfg["epsilon_min"] = 0.05

    return cfg


# ── Global HPO config helpers ────────────────────────────────────────────────
# Use the same default as utilsNoWFO (repo-root /hpo) to avoid CWD-dependent bugs.
HPO_CONFIG_DIR = os.environ.get(
    "MLB_HPO_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpo"),
)

def _ensure_hpo_dir():
    try:
        os.makedirs(HPO_CONFIG_DIR, exist_ok=True)
    except Exception:
        pass

def save_hpo_config_to_disk(model_type: str, best_params: dict, topN_params=None):
    """
    Persist tuned hyperparameters for a given model_type so that they can be
    reused later (e.g. in real_trading_simulation) without re-running Optuna.
    """
    _ensure_hpo_dir()

    safe_best = _sanitize_for_json(best_params or {})
    safe_topN = _sanitize_for_json(topN_params) if topN_params else None

    payload = {
        "model_type": str(model_type),
        "best_params": safe_best,
    }
    if safe_topN:
        payload["topN_params"] = safe_topN

    path = os.path.join(HPO_CONFIG_DIR, f"model_{model_type}_hpo.json")
    try:
        with open(path, "w") as f:
            # default=str just in case something weird slips through (e.g. Timestamps)
            json.dump(payload, f, indent=2, default=str)
        print(f"[HPO] Saved config for {model_type} to {path}")
    except Exception as e:
        print(f"[HPO] Warning: could not save HPO config for {model_type}: {e}")


def _sanitize_for_json(obj):
    """
    Recursively replace NaN / +/-inf with None so that json.dump produces
    valid JSON. Leaves normal numbers/strings/bools untouched.
    """
    import math

    # Dict → sanitize values
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    # List / tuple → sanitize each element
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

    # Everything else (str, bool, None, etc.) → keep as is
    return obj

def load_hpo_config_from_disk(model_type: str):
    """
    Load previously tuned hyperparameters for model_type. Returns (best, topN).
    If nothing is found, returns (None, None).
    Compatibility notes:
        - Supports both file names:
            1) model_<model>_hpo.json   (MLBacktester save/load)
            2) <model>_best_config.json (utilsNoWFO save_hpo_config_to_disk)
        - Supports both schemas:
            { "best_params": {...}, "topN_params": [...] }
            { "best": {...},       "topN": [...] }
    """
    # Candidate paths (first match wins)
    candidates = []

    # Preferred: MLBacktester naming
    candidates.append(os.path.join(HPO_CONFIG_DIR, f"model_{model_type}_hpo.json"))

    # utilsNoWFO naming (safe-escaped)
    safe = str(model_type).replace("/", "_")
    candidates.append(os.path.join(HPO_CONFIG_DIR, f"{safe}_best_config.json"))
    candidates.append(os.path.join(HPO_CONFIG_DIR, f"{model_type}_best_config.json"))

    # If utilsNoWFO uses an absolute base dir and MLB_HPO_DIR isn't set, also check it.
    try:
        from utilsNoWFO import get_hpo_config_dir  # local import
        _base = str(get_hpo_config_dir())
        candidates.append(os.path.join(_base, f"{safe}_best_config.json"))
        candidates.append(os.path.join(_base, f"model_{model_type}_hpo.json"))
    except Exception:
        pass

    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not path:
        return None, None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[HPO] Warning: could not load HPO config for {model_type} from {path}: {e}")
        return None, None
    
    best = data.get("best_params") or data.get("best")
    topN = data.get("topN_params") or data.get("topN") or []

    # Fallback: if schema is flat (params at root), treat the whole dict (minus obvious metadata) as best.
    if not isinstance(best, dict) or not best:
        if isinstance(data, dict):
            drop_keys = {
                "model_type", "direction", "study_name", "schema_version", "generated_at_utc", "source_files",
                "best_params", "best", "topN_params", "topN", "trials",
            }
            best = {k: v for k, v in data.items() if (k not in drop_keys and not str(k).startswith("__"))}
        else:
            best = {}

    # Strip internal metadata keys that can leak into model constructors
    if isinstance(best, dict):
        best = {k: v for k, v in best.items() if not str(k).startswith("__")}

    # Attach a tiny committee pool for runtime consensus (Top-3 if available).
    # __* keys are stripped above, but consensus needs a pool on the evaluated params dict.
    try:
        if isinstance(best, dict) and isinstance(topN, list) and len(topN) >= 2:
            pool = [dict(x) for x in topN[:3] if isinstance(x, dict)]
            if len(pool) >= 2:
                best["__top3_params"] = pool
    except Exception:
        pass

    return best, topN
    
def _apply_temperature_to_proba(proba: np.ndarray, T: float) -> np.ndarray:
    T = float(max(1e-3, T))
    logp = np.log(np.clip(proba, 1e-7, 1.0)).astype(np.float64)
    z = logp / T
    z -= z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / np.sum(ez, axis=1, keepdims=True)).astype(np.float32)

def _fit_temperature_from_proba(proba: np.ndarray, y_true: np.ndarray) -> float:
    idx = (np.arange(len(y_true)), y_true.astype(int))
    def nll(p):
        p = np.clip(p[idx], 1e-7, 1.0)
        return float(-np.mean(np.log(p)))
    Ts = np.concatenate([np.linspace(0.5, 3.0, 26),
                        np.linspace(0.3, 0.5, 5),
                        np.linspace(3.0, 4.0, 5)])
    best_T, best = 1.0, nll(proba)
    for T in Ts:
        L = nll(_apply_temperature_to_proba(proba, T))
        if L < best: best_T, best = float(T), float(L)
    return float(best_T)

from typing import Tuple


from math import sqrt
try:
    from scipy.stats import norm
except Exception:
    norm = None

def deflated_sharpe_ratio(sr: float, n_eff: int, sr_max: float = 0.0,
                          skew: float = 0.0, kurt: float = 3.0,
                          n_trials: int = 1) -> float:
    if n_eff is None or n_eff < 2 or not (sr == sr):
        return -1.0
    return (sr - sr_max) * sqrt(max(n_eff, 1))

    
import re
def _cv_status_is_ok(status: str) -> bool:
    """
    Return True iff the fold should count toward the objective.
    We treat only explicit OK folds as objective-eligible.
    """
    try:
        s = str(status or "").strip()
    except Exception:
        return False
    # Most robust: your table prints "🟢 OK" for passing folds.
    if "🟢" in s and "OK" in s:
        return True
    # Fallback: if someone removed emoji but kept the token.
    if re.search(r"\bOK\b", s):
        return True
    return False



def _psr(sr: float, n_eff: int, sr_bench: float = 0.0, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio: P(SR > sr_bench)."""
    if n_eff is None or n_eff < 2 or not (sr == sr):
        return 0.0
    num = (sr - sr_bench) * sqrt(max(n_eff - 1, 1))
    den = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1.0) * (sr ** 2) / 4.0))
    z = num / den
    if norm is None:
        import math
        return 0.5 * (1.0 + math.erf(z / sqrt(2)))
    return float(norm.cdf(z))

def _dsr_sign(sr: float, n_eff: int, sr_max: float = 0.0) -> float:
    """Very small DSR-style sign proxy (positive => likely above sr_max)."""
    if n_eff is None or n_eff < 2 or not (sr == sr):
        return -1.0
    return (sr - sr_max) * sqrt(max(n_eff, 1))

def _cv_reliability_gate(score: float, trades: int, avg_hold_bars: float, params: dict, cfg: dict) -> tuple[bool, str]:
    """
    Returns (ok, reason). ok=False means 'prune with reason'.
    Pulls defaults from CLASS_DEFAULTS (features) but allows per-trial override via params.
    """
    fcfg = cfg["features"]
    gating_mode      = params.get("gating_mode",        fcfg["gating_mode"])
    min_trades_block = int(params.get("min_trades_per_block",  fcfg["min_trades_per_block"]))
    min_indep_bets   = int(params.get("min_independent_bets",  fcfg["min_independent_bets"]))
    psr_alpha        = float(params.get("psr_alpha",           fcfg["psr_alpha"]))
    dsr_prune        = bool(params.get("dsr_prune",            fcfg["dsr_prune"]))
    floor_cv_final   = float(params.get("floor_cv_final",      fcfg["floor_cv_final"]))

    if trades < min_trades_block:
        return (False, f"trades<{min_trades_block} (got {trades})")

    avg_hold = max(1.0, float(avg_hold_bars))
    # crude but stable effective bets proxy
    n_eff = int(max(min_indep_bets, trades / avg_hold))

    if gating_mode == "bets_psr":
        psr = _psr(score, n_eff, sr_bench=0.0)
        if psr < (1.0 - psr_alpha):   # e.g., <0.95 when alpha=0.05
            return (False, f"PSR<{1-psr_alpha:.2f} (psr={psr:.3f}, n_eff={n_eff})")
        if dsr_prune:
            dsr = _dsr_sign(score, n_eff, sr_max=0.0)
            if dsr <= 0.0:
                return (False, f"DSR<=0 (dsr={dsr:.3f}, n_eff={n_eff})")

    if score <= floor_cv_final:
        return (False, f"score {score:.2f} <= floor {floor_cv_final:.2f}")

    return (True, f"ok (n_eff={n_eff})")

# ============================================================
# Metric helpers: standardize shape & invalid/no-trade metrics
# ============================================================

def _empty_metrics(context: str = "") -> tuple:
    """
    Return a shape-correct metric tuple filled with NaNs.

    Used for:
    - invalid folds,
    - failed evaluations,
    - situations where we want to signal "no usable metrics" but
      keep the schema stable to avoid shape errors downstream.
    """
    raw = [np.nan] * N_METRICS
    metrics = ensure_metric_tuple(raw)
    # context helps debug which path produced empty metrics
    try:
        validate_metrics_shape(metrics, context=context or "empty_metrics")
    except Exception:
        # If even validation fails, fall back to a plain tuple (still correct length)
        metrics = tuple(raw)
    return metrics


def _safe_metrics_return(raw_metrics, context: str = "") -> tuple:
    """
    Strict contract enforcement: fail fast on metric arity drift (prevents silent corruption).
    """
    # Validate WITHOUT coercion (raises on mismatch)
    validate_metrics_shape(raw_metrics, context=context or "evaluation")
    # Safe to cast now (length already proven correct)
    return ensure_metric_tuple(raw_metrics)


# -----------------------------------------------------------------------------
# Output toggles: control which per-model artifacts are written.
# -----------------------------------------------------------------------------
SAVE_TRADES = {
    # Monthly summary of trades per rep
    # (1 row per month, per rep) → <run>/repetition_k/<Model>/csv/monthly_trade_summary_repK.csv
    "monthly_summary_per_rep_csv": True,

    # Per-trade BH vs model comparison (entry/exit)
    # → <run>/repetition_k/<Model>/csv/trade_entry_exit_compare_repK.csv
    "trade_entry_exit_compare_csv": True,

    # Reserved for future use:
    # "per_trade_month_csv": False,
    # "rep_summary_csv": False,
}

SAVE_EQUITY = {
    # Per-month equity PNG for each valid month of a given rep
    # → <Model>/graphs/monthly_equity_k.png
    # Disabled by default; enable explicitly when needed.
    "per_month_equity_png": True,

    # Mean equity over reps (full horizon) per model
    # (run-level mean curves; currently unused)
    "mean_equity_over_reps": True,
}

SAVE_METRICS = {
    # Per-month metrics CSV written during wrap-up
    # → <Model>/Months/k/csv/csv_month_k.csv
    # Disabled by default; enable explicitly when needed.
    "per_month_metrics_csv": True,

    # Aggregated monthly results:
    # → <RUN_DIR>/model_stats/monthly_results_all_<model>.csv
    "monthly_results_all_csv": True,

    # Split by rep:
    # → <RUN_DIR>/repetition_k/<Model>/csv/monthly_results_rep<k>_<model>.csv
    "monthly_results_per_rep_csv": True,
}

SAVE_FEATURES = {
    # Per-month feature heatmap:
    # → <Model>/Months/k/heatmaps/feature_heatmap_k.png
    "monthly_heatmap_png": False,

    # Features/config text dump:
    # → <Model>/Months/k/csv/featuresconfigused_k.txt
    "featuresconfig_txt": False,  # can turn on if you want the heavy dumps
}

# ---------------------------------------------------------------------
# Global defaults (single source of truth) — module-level
# ---------------------------------------------------------------------
CLASS_DEFAULTS = {
    "features": {
        # --- Sessions & leakage control (AFML-consistent) ---
        "session_filter_mode": "both",
        "session_filter_on_train": True,
        "final_embargo_bars": 0,
        "enforce_day1_start": True,

        # --- Feature pipeline / lags ---
        "lag_depth": 1,
        "roll_windows": [5, 10, 30, 60],
        "include_hour": True,
        "include_hour_cyclic": True,
        
        # Table prints
        "eval_print_causality_debug": False,

        # --- Feature slice cache (per-run df_out cache) ---
        "slice_cache_enabled": False,

        # --- Realized-volatility features (short/long windows) ---
        "use_rv_features": True,
        "rv_window_short": 48,
        "rv_window_long": 240,

        # --- Indicator state features (oscillators & volatility regimes) ---
        "use_indicator_states": False,
        "rsi_overbought_level": 70,
        "rsi_oversold_level": 30,
        "stoch_overbought_level": 80,
        "stoch_oversold_level": 20,
        "bbw_compress_threshold": 0.05,
        "bbw_expand_threshold": 0.20,

        # --- Donchian-style price channels / breakouts ---
        "use_donchian": True,
        "donchian_window_short": 20,
        "donchian_window_long": 60,

        # --- Fractional differencing (AFML-style) ---
        "use_fracdiff": False,
        "fracdiff_d": 0.5,

        # --- MTF housekeeping ---
        "mtf_fillna_method": "ffill",

        # --- Canonical indicators (toggled by strategies) ---
        "use_rsi": False,
        "use_macd": False,
        "use_ema": False,
        "use_adx": False,
        "use_bbands": False,
        "use_stoch": False,
        "use_atr": False,
        "use_mtf_ma": False,

        # --- Indicator windows (standard TA defaults) ---
        "indicator_windows": {
            "sma": 20,
            "ema": 20,
            "rsi": 14,
            "atr": 14,
            "adx": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_window": 20,
            "bb_dev": 2.0,
            "stoch_k": 14,
            "stoch_d": 3,
            "mtf_ma_fast_window": 10,
            "mtf_ma_slow_window": 50,
        },

        # --- Macro feature hook (daily / lower-frequency) ---
        "use_macro_features": False,
        "macro_sources": {},
        "macro_lag_days": 1,

        # --- Global HPO policy ---
        "tune_once": True,

        # --- Deep / windowing knobs ---
        "cnn_use_seq_windows": True,
        "lstm_use_seq_windows": True,
        "transformer_train_stride": 1,
        "cnn_train_stride": 1,
        "lstm_train_stride": 1,
        "deep_max_train_windows": 30000,

        # --- Runtime coverage calibration mode ---
        "runtime_coverage_mode": "rolling_quantile",

        # --- Spread/slippage protection baseline ---
        "eval_use_spread_guard": True,
        "eval_spread_cap": 0.00040,
        "slippage_factor": 1.0,
        "eval_impact_eta": 0.0,

        # --- CV-time caps for keras models (early stopping regime) ---
        "deep_cv_max_epochs": 12,
        "deep_cv_batch_size": 256,
        "deep_cv_patience": 6,

        # --- Per-model CV caps (multi-fidelity HPO) ---
        "cnn_cv_max_epochs": 6,
        "cnn_cv_batch_size": 256,
        "cnn_cv_train_stride": 3,
        "cnn_cv_max_train_windows": 4000,

        "lstm_cv_max_epochs": 8,
        "lstm_cv_batch_size": 256,
        "lstm_cv_train_stride": 2,
        "lstm_cv_max_train_windows": 5000,

        "transformer_cv_max_epochs": 6,
        "transformer_cv_batch_size": 256,
        "transformer_cv_train_stride": 2,
        "transformer_cv_max_train_windows": 7000,

        # --- Extra-strict CV caps for ensemble deep heads (CNN/LSTM inside ensemble) ---
        "cnn_ens_cv_max_epochs": 6,
        "lstm_ens_cv_max_epochs": 8,
        "cnn_ens_cv_max_train_windows": 4000,
        "lstm_ens_cv_max_train_windows": 5000,

        # --- Probability calibration (classical + deep) ---
        "calibrate_method": "isotonic",
        "deep_calibrate": True,
        "deep_calibration_method": "temperature",
        "deep_calibration_frac": 0.10,
        "deep_calibration_min_samples": 500,

        # --- Trade gating — PSR/DSR-based reliability filters ---
        "gating_mode": "bets_psr",

        # --- Final experiment coverage policy (fixed; comparable across models) ---
        "target_active_rate": 0.15,
        "runtime_active_band_margin": 0.08,
        "runtime_conf_nudge": 0.005,
        "runtime_coverage_window": 192,


        # Fallback floor when coverage intent is OFF.
        # When coverage intent is ON (target_active_rate/target_coverage > 0),
        # train-anchored coverage calibration overrides this.
        "confidence_threshold": 0.80,

        # Never "force trades" by lowering conf_thr unless explicitly enabled.
        "allow_conf_backoff_cv": False,
        "allow_conf_backoff_eval": False,

        # Real-sim bump applied on top of target_active_rate (opt-in only)
        "real_sim_target_active_mult": 1.00,
        "real_sim_target_active_cap": 0.25,
        "allow_real_sim_target_active_mult": False,

        # --- Reliability / pruning helpers ---
        "min_trades_per_block": 0,
        "min_independent_bets": 20,
        "psr_alpha": 0.24,
        "dsr_prune": True,
        "floor_cv_final": -6.0,

        # --- Per-model confidence tweaks (architectural bias) ---
        "lstm_conf_relax": 0.15,
        "lstm_conf_floor": 0.40,

        # --- Labeling guards & triple-barrier events ---
        "use_triple_barrier": True,
        "tb_pt_mult": 2.0,
        "tb_sl_mult": 2.0,
        "tb_max_holding": 48,
        "tb_neutral_zone": 1.0,
        "tb_neutral_zone_is_sigma": True,
        "print_labeling_debug": False,

        # --- Prefilter / stability selection helpers ---
        "use_prefilter": True,
        "prefilter_min_unique_frac": 0.005,
        "prefilter_min_std": 1e-6,
        "prefilter_max_corr": 0.96,
        "prefilter_prefer_prefixes": ["rv", "ema", "sma", "macd", "adx"],
        "mutual_info_top_k": "sqrt",
        "prefilter_random_state": 42,

        # --- Ensemble throttles & fusion ---
        "ensemble_train_stride": 1,
        "ensemble_deep_max_train_windows": 15000,
        "fusion_alpha": 0.6,

        # --- Regime threshold defaults for AdaptiveRegimeStrategy ---
        "adx_thresh_q": 0.70,

        # --- Reporting / artifact controls ---
        "save_monthly_equity_plots": True,
        "save_monthly_feature_heatmaps": False,

        # --- Dynamic edge-vs-cost gating coefficients ---
        "alpha_vol_z": 0.004,
        "beta_spread_norm": 0.008,
        "gamma_slip_norm": 0.004,
        "slip_norm_bps": 0.25,
        "min_slip_norm_bps": 0.05,
        "vol_z_cap": 6.0,
        "spread_norm_cap": 5.0,
        "slip_ratio_cap": 6.0,
        "min_conf_thr": 0.33,
        "max_conf_thr": 0.90,
        "min_conf_thr_cov": 0.0,
        "max_conf_thr_cov": 0.90,

        # --- Top-N consensus & meta-analysis (runtime) ---
        "deploy_topN_consensus": False,

        # IMPORTANT: turn this OFF for the consensus experiment so results are not “adaptive mode”
        # (adaptive top3 can switch behavior / fall back and muddy your thesis comparison)
        "use_adaptive_top3_for_main_results": False,

        "topN_classical": 3,
        "topN_deep": 3,
        "topN_ensemble": 3,
        "topN_default": 3,

        "consensus_pool_max_trials": 0,
        "topN_style_lock": False,

        # Make it accept your Top-3 even if #2/#3 are worse than #1
        "topN_min_perf_frac": 0.00,

        # Make “geometry similarity” basically never reject members (Top-3 only -> don’t over-filter)
        "topN_geom_radius": 9.0,

        # Keep the tolerances (they won’t matter much once geom_radius is huge, but harmless)
        "topN_lags_tol": 4.0,
        "topN_depth_tol": 1.0,
        "topN_target_tol": 0.05,

        # Make correlation filter basically never drop a member
        "topN_max_corr": 0.9999,

        "print_topN_debug": True,

        "deploy_param_heatmaps": False,
        "topN_for_heatmaps": 5,
        "deploy_feature_freq": False,
        "top_feature_percent": 1.0,

        # --- Evaluation / risk / execution defaults ---
        "eval_use_vol_target": True,
        "eval_vol_target_ann": 0.10,
        "eval_vol_floor": 1e-6,
        "eval_vol_lookback": 96,
        "eval_max_leverage": 1.5,

        "eval_use_scaleout_trail": True,
        "eval_tp1_z": 1.5,
        "eval_trail_k": 3.0,
        "eval_trail_dynamic_vol": True,
        "eval_move_stop_to_be": True,
        "eval_max_holding_bars": 0,

        # Reduce hyper-churn: block flips for the first N bars after entry (exits-to-flat allowed).
        "eval_min_holding_bars": 3,

        "eval_use_twap_execution": True,
        "eval_twap_span_bars": 2,
        "eval_twap_freeze_size_at_entry": True,

        "eval_use_regime_adaptive": True,
        "eval_regime_source": "sigma",
        "eval_regime_q_low": 0.33,
        "eval_regime_q_high": 0.66,
        "eval_tp1_z_calm": 1.2,
        "eval_tp1_z_normal": 1.5,
        "eval_tp1_z_volatile": 1.8,
        "eval_trail_k_calm": 2.5,
        "eval_trail_k_normal": 3.0,
        "eval_trail_k_volatile": 3.5,
        "eval_print_regime_debug": False,
        "eval_print_trail_debug": False,
        "eval_twap_print_debug": False,

        "eval_use_kill_switch": True,
        "eval_kill_mode": "sigma",
        "eval_kill_limit_pct": 0.02,
        "eval_kill_sigma": 3.0,
        "eval_kill_until_session_end": True,
        "eval_cooloff_bars": 30,
        "eval_kill_min_limit_pct": 0.005,
        "eval_kill_min_sigma": 1.0,
        "eval_kill_max_sigma": 6.0,
        "eval_kill_max_cooloff_bars": 480,
        "eval_kill_print_debug": False,

        # --- Output / plotting profile ---
        "output_profile": "thesis",
        "light_output": False,

        "enable_pbo_mcs_analysis": False,

        "allow_param_fallback": False,
        "min_trades_for_wfo": 0,

        # --- Regime features ---
        "use_regime_features": True,
        "regime_num_states": 3,
        "regime_trend_quantile": 0.7,
        "regime_vol_high_quantile": 0.7,
        "regime_vol_low_quantile": 0.4,
        "regime_vol_window": 20,

        # --- DQN reward shaping ---
        "env_cost_scale_dqn": 1.0,
        "env_turnover_penalty_dqn": 0.0002,
    },

    "cv": {
        
        "use_cached_global_hpo": True,
        "n_trials": 0,
        
        # --- CV geometry ---
        "cv_mode": "mini_block",
        "cv_blocks": 5,
        "cv_min_train_frac": 0.75,
        "cv_val_frac": 0.05,
        "cv_embargo_bars": 0,
        "cv_embargo_frac": 0.01,
        "cv_fit_blocks_exact": True,
        "cv_tail_anchor": True,

        # --- Monthly-roll legacy knobs kept for compatibility ---
        "cv_target_folds": 5,
        "cv_val_months": 1.0,
        "cv_train_months": None,
        "bars_per_month_hint": 1000,
        "cv_sliding_stride_frac": None,

        # --- Fold aggregation / robustness ---
        "cv_fold_aggregator": "ivw_sharpe_capped",
        "cv_sr_cap": 4.0,
        "cv_sr_var_floor": 0.75,
        "cv_weight_blend_neff": 0.30,
        "cv_neff_mode": "trades",
        "cv_min_eff_n": 0.0,
        "cv_tail_weight": 1.00,
        "cv_z_weights": "sqrt_n",
        "cv_z_cap": 8.0,
        "cv_sr_ref": 0.0,
        "cv_huber_delta": 1.50,
        "cv_catoni_alpha": 0.50,
        "cv_std_penalty": 0.0,
        "cv_coverage_gamma": 1.50,

        # --- CV validity gates (stop 1-fold gaming) ---
        "cv_min_coverage": 0.80,
        "cv_min_valid_fraction": 0.80,
        "cv_prune_on_low_valid_fraction": True,

        # --- reliability / activity gates ---
        "cv_min_trades_per_block": 30,
        "cv_min_indep_bets_per_block": 12,
        "cv_gate_min_folds": 4,
        "cv_gate_min_active_rate": 0.02,
        "cv_gate_min_sr": 0.00,

        # --- active-rate hygiene ---
        "cv_min_active_rate": 0.005,
        "cv_active_rate_low": 0.00,
        "cv_active_rate_high": 1.00,
        "cv_active_rate_margin": 0.03,
        "cv_low_active_lambda": 0.25,

        # Soft penalties for missing active-rate band in CV
        "cv_soft_active_low_lambda": 1.0,
        "cv_soft_active_high_lambda": 1.0,

        # Soft penalties for turnover outside family band
        "cv_turnover_low": None,
        "cv_turnover_high": None,
        "cv_turnover_low_lambda": 1.0,
        "cv_turnover_high_lambda": 2.0,
        "cv_trade_shortfall_lambda": 0.0,

        # --- numeric stability ---
        "cv_min_volatility": 1e-6,

        "cv_invalid_share_penalty": 5.0,

        # --- Evaluation cost model knobs (CV) ---
        "eval_use_trading_costs": False,
        "eval_spread_pips": 0.8,
        "eval_slip_mode": "tworegime",
        "eval_slip_bps_lo": 0.08,
        "eval_slip_bps_med": 0.16,
        "eval_slip_bps_hi": 0.30,
        "vol_window_bars": 96,
        "high_vol_q": 0.85,
        "high_vol_conf_bump": 0.0,

        "turnover_penalty_lambda": 0.1,

        # --- Debug/log controls ---
        "print_cv_debug": False,
        "print_cv_fold_scores": False,
        "cv_log_precision": 8,
        "cv_use_psr_trim": False,

        # --- Pruning controls ---
        "prune_min_folds": 3,
        "prune_iqr_mult": 1.0125,
        "prune_abs_floor_sr": -8.0,

        # --- Trade cap controls ---
        "cv_dynamic_trades_cap_frac": 0.675,
        "cv_max_trades_per_block": 500,

        # --- Alternative aggregation knobs (kept for compatibility) ---
        "cv_agg_mode": "tanh_mean",
        "cv_tanh_s": 10.0,
        "cv_trim_frac": 0.20,
        "cv_psr_power": 1.0,
        "cv_use_recency_weight": False,
        "cv_recency_power": 1.0,

        # --- CSCV / PBO-related knobs (kept for compatibility) ---
        "cv_cscv_penalty_weight": 0.30,
        "cv_cscv_min_rank_corr": 0.20,
        "cv_cscv_disqualify": False,
        "cv_strict_pruning": False,
        "cv_prune_relax": 0.50,

        "cv_prune_precision_intent": False,

        # --- Optuna plateau stopping ---
        "plateau_min_trials": 20,
        "plateau_patience": 15,
        "plateau_delta": 0.02,

        # --- Disable extra stages: mini-fold → consensus → real trading (only) ---
        "robustness_eval": False,
        "robust_seeds": [1111, 2222, 3333],
        "robust_require_pass": False,
        "verify_topn_monthly_roll": False,
    },
}

# Convenience mirrors to avoid NameError and accidental mutation
DEFAULT_FEATURES = deepcopy(CLASS_DEFAULTS["features"])
DEFAULT_CV       = deepcopy(CLASS_DEFAULTS["cv"])


class MLBacktester:
    def __init__(
        self,
        symbol,
        start,
        end,
        trading_costs: bool = True,
        use_extended_features: bool = True,
        model_type: str = "svm",
        slippage_factor: float = 0.5,
        features_config: dict | None = None,
        use_oof: bool = False,
    ):
        """
        Initialize the backtester for a specific instrument and date range.

        Parameters
        ----------
        symbol : str
            Financial instrument (e.g., 'EURUSD').
        start, end : str | pd.Timestamp
            Backtest window (inclusive).
        trading_costs : bool
            If True, incorporate trading costs in evaluation.
        use_extended_features : bool
            If True, use engineered technical features.
        model_type : str
            Model identifier (e.g., 'svm', 'cnn', 'lstm', 'xgboost', etc.).
        slippage_factor : float
            Slippage coefficient to model execution friction.
        features_config : dict | None
            Configuration for feature generation (indicator windows, toggles, etc.).
        use_oof : bool
            If True, enables Out-of-Fold stacking (for ensemble models).
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        # If trading_costs is explicitly provided at construction, it must not be overwritten
        # by any loaded/merged config later (GlobalHPO reuse, etc.).
        self._trading_costs_locked = (trading_costs is not None)
        self.trading_costs = True if trading_costs is None else bool(trading_costs)

        self.use_extended_features = use_extended_features
        self.model_type = model_type
        self.slippage_factor = float(slippage_factor) if slippage_factor is not None else 1.0
        self.use_oof = use_oof  # control OOF stacking
        self.model = None
        self.results = None
        
        # CV diagnostics: last evaluated fold frame and per-CV-fold frames
        # Used only during Optuna-style CV runs (_in_optuna_cv True).
        self._cv_last_eval_df = None
        self._cv_fold_eval_frames: list = []
        
        # Accumulator for WFO/WFS monthly records (used by PBO/MCS analysis)
        self._wfo_monthly_records: list[dict] = []
        
        # Showing first bars of the trading month
        self._dbg_first_bars = False     # opt-in only
        self._in_cv = False              # set True inside CV wrappers
        self._in_real_sim = False        # set True inside real_trading_sim()


        # ✅ Instance-private copy so in-class mutations never leak outward
        self.features_config = deepcopy(features_config) if features_config else {}
        
        # --- Resolve slippage_factor (explicit config > ctor arg). ---
        # Prevent silent 0.0 when trading_costs are enabled.
        try:
            if isinstance(self.features_config, dict) and ("slippage_factor" in self.features_config):
                self.slippage_factor = float(self.features_config.get("slippage_factor"))
            elif bool(self.trading_costs) and float(getattr(self, "slippage_factor", 0.0) or 0.0) == 0.0:
                # Legacy default was 0.0; treat as 'unset' unless explicitly provided in config.
                self.slippage_factor = 1.0
                if self._is_debug():
                    print("[Costs] slippage_factor missing; defaulting to 1.0 (set features_config['slippage_factor'] to override).")
        except Exception:
            # never fail init due to a bad config knob
            pass
        
        # Feature-slice cache is *off by default*.
        # Rationale: prepare_features() is usually invoked on unique slices
        # (train/test/month/fold), so caching retains large frames with ~0 reuse.
        if isinstance(self.features_config, dict):
            self.features_config.setdefault("feat_cache_enabled", False)
            
            # NOTE: "slice_cache_enabled" is the canonical flag (default OFF).
            # Back-compat: honor older configs that used "feat_cache_enabled".
            if "slice_cache_enabled" not in self.features_config and "feat_cache_enabled" in self.features_config:
                self.features_config["slice_cache_enabled"] = bool(self.features_config.get("feat_cache_enabled", False))
            self.features_config.setdefault("slice_cache_enabled", False)

        # --- Feature cache / FeatureBank (per-run, per-symbol) ---
        self._feat_cache: dict = {}
        self._feat_cache_hits = 0
        self._feat_cache_misses = 0
        self._feat_cache_est_bytes = 0
        self._feat_cache_mode_logged = False  # log cache mode once per run
        
        # Log slice-cache mode explicitly (once per phase) to avoid ambiguity.
        self._feat_cache_logged_cv = False
        self._feat_cache_logged_noncv = False
        
        self._feature_bank_full = None      # type: Optional[pd.DataFrame]
        self._feature_bank_meta = {}        # small dict with base feature names, etc.
        self._feature_bank_key  = None      # signature of data + config used to build the bank
        self._feature_bank_src  = None      # optional stable source df for bank builds (see set_feature_bank_source)


        # Will be populated in get_data()
        self.data = None
        self.df_1h = None
        self.df_4h = None


        # Load all required timeframes & compute base returns
        self.get_data()

    # --- Logging mode helper ---
    def _is_debug(self):
        # Respect module-level LOG_MODE default ("COMPACT") unless explicitly overridden.
        # Also allow an instance-level debug flag.
        try:
            if bool(getattr(self, "debug", False)):
                return True
        except Exception:
            pass
        return os.environ.get("LOG_MODE", LOG_MODE).upper() == "DEBUG"
    
    def _sanitize_runtime_coverage_nudge(self, band, step, *, ctx: str = ""):
        """Clamp and stabilize runtime active-rate 'coverage nudge' params.

        Ensures the nudge step is not larger than half the band (prevents flip-flop),
        applies a small minimum band when enabled, and stores the actually-used values
        for truthful fold/month logging.
        """
        try:
            band_f = float(band)
        except Exception:
            band_f = 0.0
        try:
            step_f = float(step)
        except Exception:
            step_f = 0.0

        if not np.isfinite(band_f):
            band_f = 0.0
        if not np.isfinite(step_f):
            step_f = 0.0

        band_old, step_old = band_f, step_f

        # Clamp to sane ranges (band==0 disables nudge)
        if band_f > 0.0:
            band_f = max(0.01, min(band_f, 0.25))
        else:
            band_f = 0.0

        if step_f > 0.0:
            step_f = min(step_f, 0.25)
        else:
            step_f = 0.0

        adjusted = False

        # Stability: step must not exceed half the band.
        if band_f > 0.0 and step_f > 0.0 and step_f > 0.5 * band_f:
            step_f = 0.5 * band_f
            adjusted = True

        # Persist actually-used values for fold/month summaries.
        try:
            self._last_runtime_active_band_used = float(band_f)
            self._last_runtime_conf_step_used = float(step_f)
        except Exception:
            pass

        if adjusted and self._is_debug():
            tag = f" ({ctx})" if ctx else ""
            print(
                f"[Gate✔] Coverage nudge params adjusted{tag}: "
                f"band {band_old:.4f}→{band_f:.4f}, step {step_old:.4f}→{step_f:.4f}"
            )

        return band_f, step_f

    
    def _safe_float(self, v, fallback_key: str | None = None) -> float:
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            pass
        if fallback_key:
            try:
                _attrs = getattr(getattr(self, "results", None), "attrs", {}) or {}
                x = float(_attrs.get(fallback_key, float("nan")))
                return x
            except Exception:
                return float("nan")
        return float("nan")
    
    def _tf_cleanup(self):
        try:
            import tensorflow as tf
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
        except Exception:
            pass
        try:
            _gc.collect()
        except Exception:
            pass


    def _safe_int(self, v, fallback_key: str | None = None) -> int:
        try:
            x = int(v)
            if x != 0:
                return x
        except Exception:
            pass
        if fallback_key:
            try:
                _attrs = getattr(getattr(self, "results", None), "attrs", {}) or {}
                return int(_attrs.get(fallback_key, 0) or 0)
            except Exception:
                return 0
        return 0

    
    # --- Calibration metrics helper (used by deep models to feed Patch #2 selection penalty) ---
    def _set_last_calib_metrics(self, proba_cal, y_cal, ctx: str = ""):
        """
        Compute and store calibration metrics (Brier, NLL, n) on a calibration slice.
        Exception-safe: if something fails, metrics become NaN and run continues.
        """
        try:
            from utilsNoWFO import compute_brier_and_nll
        except Exception:
            compute_brier_and_nll = None

        try:
            n = int(len(y_cal)) if y_cal is not None else 0
        except Exception:
            n = 0

        brier = float("nan")
        nll = float("nan")
        try:
            if compute_brier_and_nll is not None and n > 0:
                brier, nll = compute_brier_and_nll(proba_cal, y_cal)
        except Exception:
            brier, nll = float("nan"), float("nan")

        # Store for CV aggregation (tuningNoWFO Patch #2)
        try:
            setattr(self, "_last_calib_brier", float(brier))
            setattr(self, "_last_calib_nll", float(nll))
            setattr(self, "_last_calib_n", int(n))
        except Exception:
            pass

        # Optional audit log
        try:
            cfg = getattr(self, "features_config", {}) or {}
            if bool(cfg.get("print_cv_debug", False)) or bool(os.environ.get("HPO_SELECT_DEBUG", "0") == "1"):
                print(f"[Calib][Metrics] brier={float(brier):.6f} nll={float(nll):.6f} n={int(n)} ctx={ctx}")
        except Exception:
            pass
    
    def set_feature_bank_source(self, df):
        """Set a stable source DataFrame for FeatureBank base indicators.

        Why this exists:
        - In real_trading_simulation() the engine repeatedly overwrites `self.data`
        with month-sized slices.
        - Base indicators (ATR/ADX/RSI/etc.) are *causal per timestamp* and can be
        computed once over a larger, stable span and then reindexed to slices.

        This is a performance patch only: it does not change feature definitions.
        """
        self._feature_bank_src = df

        # Force rebuild next time _ensure_feature_bank() runs (span changed).
        self._feature_bank_full = None
        self._feature_bank_meta = {}
        self._feature_bank_key = None
    


    def _guard_label_mix_directional(
        self,
        y_train,
        label_threshold: float,
        context: str = "FOLD",
        min_dir_samples: int = 5,
    ) -> bool:
        """
        Sanity check on 3-class labels with convention:
          0 = SHORT, 1 = NEUTRAL, 2 = LONG.

        We only enforce minimum counts on *directional* classes (0, 2),
        and allow the neutral class to be arbitrarily large or small.

        Returns
        -------
        bool
            True  → label mix is acceptable for training.
            False → fold should be skipped as structurally degenerate.
        """
        # Empty labels → nothing to train on
        if y_train is None or len(y_train) == 0:
            print(f"⚠️ [{context}] Skipping fold: empty label vector.")
            return False
        y_arr = np.asarray(y_train)
        u_tr, c_tr = np.unique(y_arr, return_counts=True)
        label_counts = dict(zip(u_tr, c_tr))

        if self._is_debug():
            print(f"[{context}] Label counts (train): {label_counts} | thr={label_threshold}")

        NEUTRAL_CLASS = 1
        dir_mask = (u_tr != NEUTRAL_CLASS)
        u_dir = u_tr[dir_mask]
        c_dir = c_tr[dir_mask]

        # No directional labels at all → useless for trading
        if len(u_dir) == 0:
            print(f"⚠️ [{context}] Skipping fold: no directional labels in train {label_counts}")
            return False

        # Both SHORT and LONG present → each must have at least min_dir_samples
        if len(u_dir) >= 2 and (c_dir.min() if len(c_dir) else 0) < min_dir_samples:
            print(f"⚠️ [{context}] Skipping fold: poor directional label mix in train {label_counts}")
            return False

        # Only one directional class present (e.g. only LONG) → require enough events
        if len(u_dir) == 1 and c_dir[0] < min_dir_samples:
            print(f"⚠️ [{context}] Skipping fold: too few directional events in train {label_counts}")
            return False

        return True

    def _resolve_conf_thr(self, default_conf: float) -> float:
        """
        Decide the effective confidence threshold for this run.

        This is called for *every* trial, CV fold, WFO month, and
        real-trading simulation. Any tweaks here are baked in from the
        start – no feedback from test results.
        """
        cfg_f = getattr(self, "features_config", {}) or {}

        # 1) Base threshold: coverage-calibrated or manual
        cov_thr = getattr(self, "_coverage_conf_thr", None)
        thr = freeze_confidence_threshold(cfg_f, default_conf, cov_thr)
        
        
        # If coverage intent exists but we couldn't compute a calibrated threshold,
        # treat this as a *rare* fallback outside CV (CV should penalize).
        try:
            import numpy as _np
            _cov_intent = bool(is_coverage_intent(cfg_f))
            _thr_ok = _np.isfinite(float(thr))
        except Exception:
            _cov_intent, _thr_ok = False, True

        # Ensure diagnostics vars always exist (prevents UnboundLocalError in CV tripwire)
        cal_rows = int(getattr(self, "_last_cov_cal_rows", 0) or 0)

        if _cov_intent and (not _thr_ok):
            in_cv = bool(getattr(self, "_in_cv", False) or getattr(self, "_in_optuna_cv", False))
            if in_cv:
                # CV tripwire: never proceed with NaN thresholds (mask becomes a no-op).
                # Use a hard 0-trade gate so the fold is penalized deterministically.
                try:
                    max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))
                except Exception:
                    max_conf_thr = 0.90
                print(
                    f"[Calib][Coverage][TRIPWIRE][CV] conf_thr=nan cal_rows={cal_rows} "
                    f"reason=missing_coverage_thr → forcing_conf_thr={max_conf_thr:.4f}"
                )
                thr = float(max_conf_thr)
            else:

                # deterministic fallback (explicit + auditable)
                fb = float(cfg_f.get("confidence_threshold", default_conf))
                ctx = "eval"
                try:
                    if bool(getattr(self, "_in_real_sim", False)):
                        mx = int(cfg_f.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                        ctx = f"real_m{mx}"
                except Exception:
                    pass
                try:
                    tar = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.0)) or 0.0)
                except Exception:
                    tar = 0.0
                cal_rows = int(getattr(self, "_last_cov_cal_rows", 0) or 0)
                print(
                    f"[Calib][Coverage][FALLBACK] conf_thr={fb:.6f} "
                    f"target_active_rate={tar:.6f} cal_rows={cal_rows} ctx={ctx} "
                    f"reason=missing_coverage_thr"
                )
                thr = fb

        # Book-keeping
        self._last_conf_thr_init = float(
            cfg_f.get("confidence_threshold", default_conf)
        )
        self._last_conf_thr_used = float(thr)

        return float(thr)

    def _emit_conf_gate_snapshot(
        self,
        *,
        model_type: str,
        eval_context: Optional[str],
        conf_requested: float,
        base_thr: float,
        thr_vec: "np.ndarray",
        eval_idx: Optional["np.ndarray"],
        dyn_abg: bool = True,
    ) -> None:
        
        cfg_f = getattr(self, "features_config", {}) or {}
        cov_intent = bool(is_coverage_intent(cfg_f))
        cov_thr = getattr(self, "_coverage_conf_thr", None)

        if cov_intent and cov_thr is not None:
            try:
                cov_thr_f = float(cov_thr)
            except Exception:
                cov_thr_f = float("nan")
        else:
            cov_thr_f = float("nan")

        if cov_intent:
            source = "coverage_calibrated" if np.isfinite(cov_thr_f) else "coverage_intent_missing"
        else:
            source = "static"

        try:
            if eval_idx is not None and hasattr(eval_idx, "size") and eval_idx.size > 0:
                used_m = float(np.nanmedian(thr_vec[eval_idx]))
            else:
                used_m = float(np.nanmedian(thr_vec))
        except Exception:
            used_m = float("nan")

        try:
            used_m = float(getattr(self, "_last_conf_thr_used", used_m))
        except Exception:
            pass

        cov_s = f"{cov_thr_f:.3f}" if np.isfinite(cov_thr_f) else "NA"
        ctx_s = str(eval_context or "")
        dyn_s = "on" if dyn_abg else "off"

        print(
            f"🔒 [ConfGate] model={model_type} ctx={ctx_s} "
            f"conf_requested={float(conf_requested):.3f} conf_base={float(base_thr):.3f} "
            f"conf_used_median={used_m:.3f} source={source} cov_thr={cov_s} dyn_abg={dyn_s}"
        )

    
    # === NEW: expose config merge as instance method (used by tuner & eval) ===
    def _merge_params_into_features_config(self, bp: dict, force_lags: int | None = None) -> dict:
        """
        Merge order:
          existing run config  << nested model sub-configs  << flat tuned keys.
        Then apply defaults ONLY to fill missing keys (defaults must not overwrite tuned keys).
        Returns the merged features_config dict.
        """
        try:
            bp = dict(bp or {})
            
            # --- Materialize derived keys for faithful replay ---------------------------------
            # Optuna trials often store selector keys (e.g., roll_windows_key_v2) and/or
            # per-indicator "*_window_core" primitives. During CV these are materialized
            # into concrete structures (roll_windows list, indicator_windows dict). If the
            # monthly real-trading loop replays a params dict missing these materialized
            # keys, apply_feature_defaults() can silently fall back to JSON defaults (e.g.,
            # roll_windows=[5,10,20]), producing different confidence distributions and
            # no-trade months. This block only fills *missing* derived fields.

            # 0) roll_windows: derive from roll_windows_key_v2/key if needed
            try:
                rk = bp.get("roll_windows_key_v2") or bp.get("roll_windows_key")
                if "roll_windows" not in bp and rk is not None:
                    bp["roll_windows"] = [int(x) for x in str(rk).split(",") if str(x).strip() != ""]
                    print(f"[HPOReplay] materialized roll_windows={bp.get('roll_windows')} from rk={rk}")
                # Drop selector aliases to avoid confusing downstream logs/snapshots
                bp.pop("roll_windows_key_v2", None)
                bp.pop("roll_windows_key", None)
            except Exception:
                pass

            # 1) indicator_windows: build from core window primitives if missing
            try:
                if "indicator_windows" not in bp or not isinstance(bp.get("indicator_windows"), dict):
                    iw = {}
                    if "sma_window_core" in bp: iw["sma"] = int(bp["sma_window_core"])
                    if "ema_window_core" in bp: iw["ema"] = int(bp["ema_window_core"])
                    if "rsi_window_core" in bp: iw["rsi"] = int(bp["rsi_window_core"])
                    if "atr_window_core" in bp: iw["atr"] = int(bp["atr_window_core"])
                    if "adx_window_core" in bp: iw["adx"] = int(bp["adx_window_core"])
                    if "bb_window_core" in bp:  iw["bb_window"] = int(bp["bb_window_core"])
                    if "bb_dev_core" in bp:     iw["bb_dev"] = float(bp["bb_dev_core"])

                    mv = bp.get("macd_core_variant", None)
                    if mv is not None and "macd_fast" not in iw:
                        try:
                            a, b, c = list(mv)
                            iw["macd_fast"] = int(a)
                            iw["macd_slow"] = int(b)
                            iw["macd_signal"] = int(c)
                        except Exception:
                            pass

                    if "mtf_ma_fast_window" in bp: iw["mtf_ma_fast_window"] = int(bp["mtf_ma_fast_window"])
                    if "mtf_ma_slow_window" in bp: iw["mtf_ma_slow_window"] = int(bp["mtf_ma_slow_window"])

                    if iw:
                        bp["indicator_windows"] = iw
            except Exception:
                pass


            # Drop CV-policy keys (never belong to features_config)
            for k in list(bp.keys()):
                if str(k).startswith("cv_"):
                    bp.pop(k, None)

            # Start from current features_config
            base = dict(self.features_config) if isinstance(self.features_config, dict) else {}

            # 1) Pull in namespaced sub-configs (cnn_config, lstm_config, transformer_config, xgb_config, dqn_config, rf_config, logit_config)
            for nested in ("cnn_config", "lstm_config", "transformer_config",
                           "xgb_config", "dqn_config", "rf_config", "logit_config"):
                val = bp.get(nested)
                if isinstance(val, dict):
                    base.update(val)

            # 2) Flat tuned keys win
            base.update(bp)

            # 3) Enforce specific keys if provided
            if force_lags is not None:
                base["lags"] = int(force_lags)
            if "use_fracdiff" in bp:
                base["use_fracdiff"] = bool(bp["use_fracdiff"])
            if "calibrate_method" in bp:
                base["calibrate_method"] = str(bp["calibrate_method"]).lower()

            # Materialize and let defaults fill only missing stuff
            self.features_config = base
            if hasattr(self, "apply_feature_defaults"):
                self.apply_feature_defaults()

            # Return a copy for external consumers (tuner)
            return dict(self.features_config)
        
        except Exception as e:
            print(f"[WARN] Could not merge best_params into features_config: {e}")
            return dict(self.features_config) if isinstance(self.features_config, dict) else {}

    def _short_param_string(self, p: dict) -> str:
        def _getint(k, default=0):
            try: return int(p.get(k, default))
            except: return default

        lags = _getint("lags", _getint("lags_range", 0))
        d    = _getint("lag_depth", 0)

        roll_a = str(p.get("roll_windows_key", "")).strip()
        roll_b = str(p.get("roll_windows_key_v2", "")).strip()
        # render like 5|10,20 if both present, else whichever exists
        roll = (roll_a + ("|" if (roll_a and roll_b) else "") + roll_b) if (roll_a or roll_b) else "-"

        strat = str(p.get("strategy_type", "-"))
        # include only the common, short, human-scan keys
        keys_by_strat = {
            "volatility": ["atr_window"],
            "confirmation": ["adx_window", "mtf_ma_fast_window", "mtf_ma_slow_window"],
            "contrarian": ["rsi_window","bb_window","bb_dev","stoch_k_window","stoch_d_window"],
            "momentum": ["ema_window","rsi_window"],
        }
        want = keys_by_strat.get(strat, [])
        bits = []
        for k in want:
            if k in p:
                label = k.replace("_window","").replace("mtf_ma_","ma_")
                bits.append(f"{label}={p[k]}")
        strat_str = f"{strat}({','.join(bits)})" if bits else strat

        tb_on = bool(p.get("use_triple_barrier", False))
        tb_bits = []
        if "tb_pt_mult" in p: tb_bits.append(f"pt={p['tb_pt_mult']}")
        if "tb_sl_mult" in p: tb_bits.append(f"sl={p['tb_sl_mult']}")
        if "tb_max_holding" in p: tb_bits.append(f"hold={p['tb_max_holding']}")
        tb_str = f"TB={'on' if tb_on else 'off'}" + ((" " + " ".join(tb_bits)) if tb_bits else "")

        return f"lags={lags} d={d} roll={roll}  strat={strat_str} {tb_str}"

    def _should_dump_decisions(self) -> bool:
        return (
            getattr(self, "_dbg_first_bars", False) and
            not getattr(self, "_in_cv", False) and
            getattr(self, "_in_real_sim", False)
        )

    def apply_feature_defaults(self, params: dict | None = None) -> dict:
        """Merge user/trial params over DEFAULT_FEATURES safely."""
        base = deepcopy(DEFAULT_FEATURES)
        if isinstance(getattr(self, "features_config", None), dict):
            base.update(self.features_config)   # class-level or previous
        if isinstance(params, dict):
            base.update(params)                 # trial-level wins
        self.features_config = base
        
        
        # ---------------------------------------------------------------
        # B2: Calibration safety clamps (stability; avoids degenerate cal windows)
        # ---------------------------------------------------------------
        try:
            def _clipf(x, lo, hi, default):
                try:
                    v = float(x)
                except Exception:
                    v = float(default)
                return float(max(lo, min(hi, v)))

            def _clipi(x, lo, hi, default):
                try:
                    v = int(x)
                except Exception:
                    v = int(default)
                return int(max(lo, min(hi, v)))

            # keep calibration fraction in a conservative band
            base["deep_calibration_frac"] = _clipf(base.get("deep_calibration_frac", 0.10), 0.08, 0.20, 0.10)
            base["classical_calibration_frac"] = _clipf(
                base.get("classical_calibration_frac", base.get("deep_calibration_frac", 0.10)), 0.08, 0.20, 0.10
            )
            # keep calibration min samples reasonable
            base["deep_calibration_min_samples"] = _clipi(base.get("deep_calibration_min_samples", 500), 500, 5000, 500)
            base["classical_calibration_min_samples"] = _clipi(
                base.get("classical_calibration_min_samples", base.get("deep_calibration_min_samples", 500)), 500, 5000, 500
            )
        except Exception:
            pass

        
        # ---------------------------------------------------------------
        # HARD DISABLE: CV thin-trades fallback (must never "invent" trades)
        # CV must match real_trading_simulation behavior.
        # ---------------------------------------------------------------
        try:
            in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        except Exception:
            in_cv = False
        if in_cv and bool(base.get("allow_thin_trades_fallback", False)):
            base["allow_thin_trades_fallback"] = False
        # Warn once per instance to avoid spam
            if not bool(getattr(self, "_warned_thin_trades_disabled", False)):
                print("🚫 [CV] allow_thin_trades_fallback is HARD-DISABLED (CV must not invent trades).")
                self._warned_thin_trades_disabled = True
                
        # --- B1 Policy: enforce GLOBAL target coverage (signal intent) for ALL models ---
        try:
            _mt = base.get("model_type", getattr(self, "model_type", None))
            enforce_target_coverage_policy(base, model_type=_mt)
            
            # Hard-assert (non-fatal) + one-time log after ALL merges.
            # This prevents silent drift from later overrides and gives a single
            # authoritative line you can trust in logs.
            try:
                tar = float(base.get("target_active_rate", base.get("target_coverage", 0.0)) or 0.0)
                exp = float(target_coverage_policy(_mt) or 0.0)
                if exp > 0.0 and abs(tar - exp) > 1e-9:
                    # Re-enforce (should be redundant); keep non-fatal to avoid breaking long runs.
                    print(f"⚠️ [CoveragePolicy][ASSERT] target_active_rate drifted to {tar:.6f}; re-enforcing policy={exp:.6f}")
                    enforce_target_coverage_policy(base, model_type=_mt)
                    tar = float(base.get("target_active_rate", base.get("target_coverage", 0.0)) or 0.0)

                if not bool(getattr(self, "_printed_coverage_policy", False)):
                    tc = float(base.get("target_coverage", tar) or tar)
                    gm = base.get("gating_mode", base.get("gate_mode", None))
                    in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                    in_real = bool(getattr(self, "_in_real_sim", False))
                    where = "CV" if in_cv else ("REAL" if in_real else "RUN")
                    print(
                        f"🎯 [CoveragePolicy][{where}] model_type={_mt} gating_mode={gm} "
                        f"target_active_rate={tar:.3f} target_coverage={tc:.3f} (policy-locked)"
                    )
                    self._printed_coverage_policy = True
            except Exception:
                pass

            
            self.features_config = base
        except Exception:
            pass
        return base

    def apply_cv_defaults(self, cv_cfg: dict | None = None) -> dict:
        """Merge user/runner CV cfg over DEFAULT_CV safely."""
        base = deepcopy(DEFAULT_CV)
        if isinstance(cv_cfg, dict):
            base.update(cv_cfg)
        return base


    @classmethod
    def set_global_defaults(cls, section: str, updates: dict):
        """Optional: tweak defaults globally at runtime, e.g., set_global_defaults('cv', {'cv_blocks': 5})."""
        if section in cls.CLASS_DEFAULTS and isinstance(updates, dict):
            cls.CLASS_DEFAULTS[section].update(updates)

        
    def __repr__(self) -> str:
        """Readable summary of key configuration."""
        return (
            f"MLBacktester(symbol={self.symbol}, start={self.start}, end={self.end}, "
            f"trading_costs={self.trading_costs}, use_extended_features={self.use_extended_features}, "
            f"model_type={self.model_type})"
        )


    def get_data(self) -> None:
        """
        Load and preprocess raw market data for the specified window.

        - 30m data → index=time (tz-aware), rename to price/high/low, compute log returns.
        - 1H / 4H data → loaded for multi-timeframe (MTF) features.
        - Precompute/Load 'mtf_ma_fast' (1H fast MA, shifted) and 'mtf_ma_slow' (4H slow MA, shifted).
        Uses tuned windows from features_config['indicator_windows'] and prefers precomputed columns if present.
        """
        # ---- 30m base data ----
        raw = _load_csv_cached(BASE_CSV, parse_dates=["time"], index_col="time")

        # normalize column names expected downstream
        raw.rename(columns={"mid_close": "price", "mid_high": "high", "mid_low": "low"}, inplace=True)
        raw = raw[["price", "high", "low", "spread"]]

        # compute log-returns
        raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))

        # 🔽 Downcast numeric columns to float32 to save RAM
        for col in ("price", "high", "low", "spread", "returns"):
            if col in raw.columns:
                raw[col] = raw[col].astype("float32")

        # ensure tz-aware index before slicing
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
        self.data = raw.loc[self.start:self.end].dropna()

        # Optionally attach macro features (daily / lower-frequency) to bar-level data
        cfg = self.features_config or {}
        if cfg.get("use_macro_features", False):
            macro_specs = cfg.get("macro_sources") or cfg.get("macro_csv_paths") or {}
            if macro_specs:
                try:
                    lag_days = int(cfg.get("macro_lag_days", 1))
                except Exception:
                    lag_days = 1
                try:
                    self.data = attach_macro_features(
                        self.data,
                        macro_specs=macro_specs,
                        lag_days=lag_days,
                    )
                except Exception as _e:
                    if self._is_debug():
                        print(f"⚠️ Failed to attach macro features: {_e}")

        # --- One-time NY session mask (02:00–13:00 NYT) ---
        try:
            _ny_times = self.data.index.tz_convert("America/New_York")
            _ny_active = (_ny_times.hour >= 2) & (_ny_times.hour <= 13)
            self._ny_mask = pd.Series(_ny_active, index=self.data.index)
        except Exception as _e:
            print(f"⚠️ Failed to precompute NY session mask: {_e}")
            self._ny_mask = pd.Series(True, index=self.data.index)  # safe fallback

        # ---- 1H and 4H for MTF features (cached) ----
        self.df_1h = _load_csv_cached(CSV_1H, parse_dates=["time"], index_col="time")
        self.df_4h = _load_csv_cached(CSV_4H, parse_dates=["time"], index_col="time")

        # 🔽 Downcast numeric columns in 1H / 4H to float32 as well
        for _df in (self.df_1h, self.df_4h):
            for _col in _df.columns:
                # only downcast numeric dtypes
                if pd.api.types.is_numeric_dtype(_df[_col].dtype):
                    _df[_col] = _df[_col].astype("float32")

        # ---- Precompute/Load MTF MAs on full history (shift(1) to avoid leakage) ----
        try:
            ind = (self.features_config or {}).get("indicator_windows", {}) or {}
            fast_w = int(ind.get("mtf_ma_fast_window", 10))   # default 10 (1H)
            slow_w = int(ind.get("mtf_ma_slow_window", 50))   # default 50 (4H)

            df1 = self.df_1h.copy()
            df4 = self.df_4h.copy()

            # Prefer precomputed columns if present; else compute from mid_close
            fast_candidates = [
                f"mtf_1h_ma{fast_w}", "mtf_1h_ma_fast", f"ma_1h_{fast_w}", f"ma_fast_{fast_w}"
            ]
            slow_candidates = [
                f"mtf_4h_ma{slow_w}", "mtf_4h_ma_slow", f"ma_4h_{slow_w}", f"ma_slow_{slow_w}"
            ]
            col_fast = next((c for c in fast_candidates if c in df1.columns), None)
            col_slow = next((c for c in slow_candidates if c in df4.columns), None)

            if col_fast is None:
                if "mid_close" not in df1:
                    raise KeyError("1H CSV missing 'mid_close' for MTF compute")
                df1["mtf_1h_ma_fast"] = (
                    df1["mid_close"]
                    .rolling(fast_w, min_periods=fast_w)
                    .mean()
                    .shift(1)
                )
                col_fast = "mtf_1h_ma_fast"

            if col_slow is None:
                if "mid_close" not in df4:
                    raise KeyError("4H CSV missing 'mid_close' for MTF compute")
                df4["mtf_4h_ma_slow"] = (
                    df4["mid_close"]
                    .rolling(slow_w, min_periods=slow_w)
                    .mean()
                    .shift(1)
                )
                col_slow = "mtf_4h_ma_slow"

            # Normalize names for merge
            df1 = df1[[col_fast]].reset_index().rename(columns={col_fast: "mtf_1h_ma_fast"})
            df4 = df4[[col_slow]].reset_index().rename(columns={col_slow: "mtf_4h_ma_slow"})

            # Align timestamps to minute grid so merge_asof matches 30m bars robustly
            df1["time"] = pd.to_datetime(df1["time"], utc=True) + pd.Timedelta(minutes=1)
            df4["time"] = pd.to_datetime(df4["time"], utc=True) + pd.Timedelta(minutes=1)

            # Merge onto current window
            base = self.data.reset_index().rename(columns={"index": "time"})
            base["time"] = pd.to_datetime(base["time"], utc=True)

            mtf_fast = pd.merge_asof(
                base.sort_values("time"), df1.sort_values("time"), on="time", direction="backward"
            ).set_index("time")["mtf_1h_ma_fast"]

            mtf_slow = pd.merge_asof(
                base.sort_values("time"), df4.sort_values("time"), on="time", direction="backward"
            ).set_index("time")["mtf_4h_ma_slow"]

            # assign to self.data aligned to index
            self.data["mtf_ma_fast"] = mtf_fast.reindex(self.data.index).astype("float32")
            self.data["mtf_ma_slow"] = mtf_slow.reindex(self.data.index).astype("float32")

            if self._is_debug():
                print(f"[MTF] fast_w={fast_w}, slow_w={slow_w} (mtf_ma_fast/slow ready)")

        except Exception as _e:
            print(f"⚠️ Precompute/Load MTF features failed: {_e}")

    @staticmethod
    def rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """
        Efficient O(n) rolling slope using cumulative sums.
        Much faster than per-window polyfit.
        """
        x = np.arange(window, dtype=float)
        Sx = x.sum()
        Sxx = (x * x).sum()
        n = window
        den = n * Sxx - Sx * Sx

        y = series.astype(float).to_numpy()
        # handle NaNs safely
        y_filled = np.where(np.isfinite(y), y, 0.0)

        csum_y = np.cumsum(y_filled)
        csum_xy = np.cumsum(y_filled * np.arange(len(y), dtype=float))

        Sy  = csum_y[window-1:] - np.concatenate(([0.0], csum_y[:-window]))
        Sxy = csum_xy[window-1:] - np.concatenate(([0.0], csum_xy[:-window]))

        # numerator for slope
        num = n * (Sxy - np.arange(window-1, len(y)) * Sy) - Sx * Sy
        slope = num / den

        out = np.full_like(y, np.nan, dtype=float)
        out[window-1:] = slope
        return pd.Series(out, index=series.index)
    
    def _ensure_feature_bank(self):
        """
        Build a simple FeatureBank of base indicators over the current `self.data`
        slice if not already present.

        Design:
        - Only base TA indicators + composite features are stored.
        - Lag/rolling expansions and raw returns_lag* are still done per-slice
          inside `prepare_features` to avoid RAM blow-up.
        - The bank is keyed by (data span + toggles + indicator_windows +
          RV/fracdiff settings + price_col). If that signature changes, the bank
          is rebuilt.
        """
        import pandas as pd
        import numpy as np
        import os, psutil

        # Prefer a stable source span if provided; else default to self.data
        src = getattr(self, "_feature_bank_src", None)
        if src is None:
            src = getattr(self, "data", None)
        if src is None or len(src) == 0:
            return

        cfg = self.features_config or {}
        ind_win = (cfg.get("indicator_windows", {}) or {})

        # Compute the desired key cheaply (even under low RAM) so we never reuse a stale bank.
        idx = pd.DatetimeIndex(src.index)
        first_idx = idx[0]
        last_idx  = idx[-1]
        toggles_on = tuple(sorted(k for k, v in cfg.items() if str(k).startswith("use_") and bool(v)))
        key = (
            first_idx,
            last_idx,
            int(len(idx)),
            toggles_on,
            tuple(sorted((k, str(v)) for k, v in ind_win.items())),
            bool(cfg.get("use_rv_features", False)),
            int(cfg.get("rv_window_short", 30)),
            int(cfg.get("rv_window_long", 120)),
            bool(cfg.get("use_fracdiff", False)),
            float(cfg.get("fracdiff_d", 0.4)),
            cfg.get("price_col", "price"),
        )

        # If existing bank matches this signature, keep it
        if getattr(self, "_feature_bank_full", None) is not None and getattr(self, "_feature_bank_key", None) == key:
            return

        # Low-RAM guard: if we skip building, CLEAR any mismatched/stale bank so it cannot be reused.
        avail_gb   = psutil.virtual_memory().available / (1024 ** 3)
        trigger_gb = float(os.getenv("LOW_RAM_TRIGGER_GB", "1.25"))
        force_off  = os.getenv("MLB_FEATUREBANK_OFF", "0") in ("1", "true", "True")
        # Optional: keep FeatureBank off during Optuna CV unless explicitly allowed
        in_cv = bool(getattr(self, "_in_optuna_cv", False))
        if in_cv and not bool(cfg.get("featurebank_in_cv", False)):
            force_off = True

        if force_off or avail_gb < trigger_gb:
            # ensure no stale reuse
            self._feature_bank_full = None
            self._feature_bank_meta = {}
            self._feature_bank_key  = None
            if self._is_debug():
                print(
                    f"[FeatureBank] Disabled/cleared (avail={avail_gb:.2f} GB, trigger={trigger_gb:.2f} GB, "
                    f"force_off={force_off}, in_cv={in_cv})"
                )
            return


        # Otherwise rebuild (clear old one first to free RAM ASAP)
        self._feature_bank_full = None
        self._feature_bank_meta = {}
        self._feature_bank_key  = None

        try:
            # We call prepare_features in "base_only" mode so it computes only
            # base indicators + composites, with NO lag/rolling expansion.
            lags_default = int(cfg.get("lags_range", cfg.get("lags", 10)))
            lag_depth    = int(cfg.get("lag_depth", 1))
            roll_windows = cfg.get("roll_windows", [5])
            if roll_windows is None:
                roll_windows = [5]

            df_feat, base_feats = self.prepare_features(
                src,
                lags=lags_default,
                lag_depth=lag_depth,
                roll_windows=roll_windows,
                base_only=True,        # <-- NEW flag
            )

            # Keep only numeric columns and downcast to float32 to control RAM
            fb = df_feat.select_dtypes(include=["number"]).astype("float32", copy=False)

            self._feature_bank_full = fb
            self._feature_bank_meta = {"base_features": list(base_feats)}
            self._feature_bank_key  = key

            if self._is_debug():
                print(
                    f"[FeatureBank] Built base-indicator bank: "
                    f"shape={fb.shape}, base_features={len(base_feats)}"
                )

        except Exception as e:
            # Fail silently (with debug print) and fall back to per-slice path
            if self._is_debug():
                print(f"⚠️ FeatureBank build failed; falling back to per-slice TA: {e}")
            self._feature_bank_full = None
            self._feature_bank_meta = {}
            self._feature_bank_key  = None

    def _attach_regime_columns(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """
        Add trend_score, vol_score, regime_id, and optional one-hot regime_* columns.
        Uses adx_col/vol_col + adx_thresh/vol_thresh from cfg or sane defaults.


        - 0 = SIDEWAYS
        - 1 = TREND
        - 2 = VOLATILE/CHOPPY
        """
        # Default ADX column used as trend proxy
        adx_col = cfg.get("adx_col") or "adx_14"

        # Default volatility proxy: realized volatility with the "short" window
        vol_col = cfg.get("vol_col")
        if not vol_col:
            rv_short = int(cfg.get("rv_window_short", 48))
            vol_col = f"rv_{rv_short}"

        adx_thr = float(cfg.get("adx_thresh", 20.0))
        # Prefer the train-anchored high-vol threshold already used by the cost model
        # (keeps regime segmentation consistent + avoids scale issues / collapse-to-volatile)
        if cfg.get("high_vol_thr") is not None:
            vol_thr = float(cfg.get("high_vol_thr"))
        else:
            vol_thr = float(cfg.get("vol_thresh", 0.001))  # fallback

        # 1) Guard: if these cols don’t exist, just skip and return df unmodified
        if adx_col not in df.columns or vol_col not in df.columns:
            # no regime annotation possible; keep compatibility
            df["regime_id"] = 1  # or SIDEWAYS default
            return df

        trend_score = df[adx_col].astype("float64")
        vol_score   = df[vol_col].astype("float64")

        # 2) Regime classification
        regime = np.full(len(df), 0, dtype="int8")   # 0 = SIDEWAYS

        trend_mask = trend_score >= adx_thr
        vol_high   = vol_score  >  vol_thr

        regime[trend_mask & ~vol_high] = 1  # TREND
        regime[~trend_mask & vol_high] = 2  # VOLATILE
        regime[trend_mask & vol_high]  = 2  # strong but wild → treat as VOLATILE for now

        df["trend_score"] = trend_score
        df["vol_score"]   = vol_score
        df["regime_id"]   = regime

        # Optional one-hots (helps classical models)
        df["regime_trend"]    = (regime == 1).astype("int8")
        df["regime_sideways"] = (regime == 0).astype("int8")
        df["regime_volatile"] = (regime == 2).astype("int8")
        
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        lags: int,
        lag_depth: int = 1,
        roll_windows: list[int] = [5],
        base_only: bool = False,
    ):

        """
        Create feature matrix using toggles & windows in `self.features_config`.

        Pipeline
        --------
        1) Compute base indicators (as toggled in config).
        2) Add SAR unconditionally.
        3) Include MTF moving averages if present (computed in get_data()).
        4) Momentum extensions: EMA–SMA spread, price–MA z-scores, crossover bins, slope differential.
        5) Composite features: re-entry + momentum, extension/ATR with low ADX, squeeze→expansion, ATR channels,
        trend confirmation, MTF alignment, volatility-managed momentum, MACD/ATR ratio.
        6) Expand with lags/rolling stats; add hour features; drop NAs on active features.

        Notes
        -----
        - Adds columns to a copy of `df`; does not mutate input.
        - Respects indicator_windows and use_* toggles in `self.features_config`.
        - Higher-TF MAs must be precomputed in get_data() (shifted to avoid look-ahead).
        """

        # CLEANUP: cache debug flag once
        debug = bool(getattr(self, "_is_debug", lambda: False)())

        # CLEANUP: tiny local helper to avoid silent exception swallowing
        def _debug_once(tag: str, exc: Exception):
            # DEBUG: print a given exception at most once per function lifetime per instance
            if not debug:
                return
            attr = f"_prepare_features_exc_once__{tag}"
            if getattr(self, attr, False):
                return
            try:
                print(f"⚠️ [prepare_features][{tag}] {type(exc).__name__}: {exc}")
            except Exception:
                # Last-resort: never crash due to logging
                pass
            setattr(self, attr, True)

        # CLEANUP: tiny local helper to reduce repeated H/L/C extraction blocks
        def _get_hlc(_df: pd.DataFrame, _price_col: str):
            hi_ = _df.get("high", _df.get(_price_col))
            lo_ = _df.get("low", _df.get(_price_col))
            cl_ = _df.get("close", _df.get(_price_col))
            return hi_, lo_, cl_

        # Safety belt: normalize index to DatetimeIndex for stable caching / FeatureBank alignment
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
        except Exception as e:
            # CLEANUP/DEBUG: keep non-fatal behavior, but don't swallow silently in debug
            _debug_once("idx_to_datetime", e)

        # ---------- 0) Cache ----------
        if not hasattr(self, "_feat_cache"):
            self._feat_cache = {}

        # Telemetry-only: track approx bytes retained in the cache (truthful "current cache size")
        if not hasattr(self, "_feat_cache_bytes"):
            self._feat_cache_bytes = {}

        cfg = self.features_config or {}
        ind_win = (cfg.get("indicator_windows", {}) or {})

        # Feature-slice caching is now opt-in (default OFF) and always bypassed during Optuna CV.
        # Rationale: cache_key includes slice boundaries → reuse is usually ~0 in walk-forward/monthly runs.
        in_cv = bool(getattr(self, "_in_optuna_cv", False))
        # Canonical flag is "slice_cache_enabled" (default OFF).
        # Back-compat: allow older configs that used "feat_cache_enabled".
        slice_cache_enabled = bool(cfg.get("slice_cache_enabled", cfg.get("feat_cache_enabled", False)))
        cache_enabled = slice_cache_enabled and (not in_cv) and (not base_only)

        # Emit cache mode once per run so it is always obvious whether we are caching or not.
        if (LOG_MODE in {"COMPACT", "DEBUG"}) and (not getattr(self, "_feat_cache_mode_logged", False)):
            msg = "[FEAT_CACHE] enabled (opt-in)" if slice_cache_enabled else "[FEAT_CACHE] disabled (default)"
            if slice_cache_enabled and in_cv:
                msg += " but BYPASSED during Optuna CV"
            print(msg)
            self._feat_cache_mode_logged = True  # CLEANUP: was False → caused repeated prints forever

        # Optional safety net: cap number of cached engineered slices (default 0 = unlimited).
        # This is a *resource* guard only: it must not affect results because the cache is
        # only an optimization (equivalent to recomputing features).
        try:
            feat_cache_max_entries = int(cfg.get("feat_cache_max_entries", 0) or 0)
        except Exception:
            feat_cache_max_entries = 0

        #  --- Normalize roll_windows early (needed for cache_key safety) ---
        rw_cfg = cfg.get("roll_windows", roll_windows if roll_windows is not None else [5])
        if isinstance(rw_cfg, str):
            roll_windows = [int(x.strip()) for x in rw_cfg.split(",") if x.strip()]
        elif isinstance(rw_cfg, (list, tuple)):
            roll_windows = [int(x) for x in rw_cfg]
        else:
            roll_windows = [int(rw_cfg)]

        # Build a cache key that reflects settings affecting columns
        start_idx = df.index[0] if len(df.index) > 0 else None
        end_idx   = df.index[-1] if len(df.index) > 0 else None
        toggles_on = tuple(sorted(k for k, v in cfg.items() if str(k).startswith("use_") and v))
        cache_key = (
            start_idx, end_idx,
            int(cfg.get("lags_range", cfg.get("lags", lags if lags is not None else 10))),
            int(cfg.get("lag_depth", lag_depth if lag_depth is not None else 1)),
            tuple(roll_windows),
            tuple(sorted((k, str(v)) for k, v in ind_win.items())),
            toggles_on,
            bool(cfg.get("include_raw_lags", True)),
            bool(cfg.get("include_hour", True)),
            bool(cfg.get("use_rv_features", False)),
            int(cfg.get("rv_window_short", 30)),
            int(cfg.get("rv_window_long", 120)),
            bool(cfg.get("use_fracdiff", False)),
            float(cfg.get("fracdiff_d", 0.4)),
            bool(cfg.get("include_hour_cyclic", True)),
            cfg.get("price_col", "price"),
        )

        cached = (self._feat_cache.get(cache_key) if cache_enabled else None)
        if cached is not None:
            # Reuse previously engineered features for this exact slice/config combo.
            df_cached, feat_cached = cached

            # Diagnostics: cache hit
            self._feat_cache_hits = int(getattr(self, "_feat_cache_hits", 0)) + 1
            if LOG_MODE in {"COMPACT", "DEBUG"}:
                try:
                    n_entries = len(self._feat_cache) if isinstance(self._feat_cache, dict) else -1
                    n_feats = len(feat_cached)
                    hits = int(getattr(self, "_feat_cache_hits", 0))
                    misses = int(getattr(self, "_feat_cache_misses", 0))
                    denom = hits + misses
                    hit_rate = (hits / denom) if denom > 0 else 0.0
                    cur_bytes = int(getattr(self, "_feat_cache_cur_bytes", 0))
                    do_print = (LOG_MODE == "DEBUG") or (self._feat_cache_hits % 25 == 0)
                    if do_print:
                        print(
                            f"[FEAT_CACHE] HIT  entries={n_entries} "
                            f"hits={hits} misses={misses} hit_rate={hit_rate:.2%} "
                            f"cache_mb={cur_bytes/1024/1024:.1f} feats={n_feats}"
                        )
                except Exception as e:
                    # DEBUG: don't spam, but don't hide forever
                    _debug_once("feat_cache_hit_diag", e)

            # Keep last-used features up to date for downstream logging.
            feat_cached = list(feat_cached)
            self._last_used_features = list(feat_cached)
            return df_cached, feat_cached  # (df_out, features)

        # ---------- 1) Params & toggles ----------
        # Windows (accept *_window aliases)
        window_sma = int(ind_win.get("sma", ind_win.get("sma_window", 20)))
        window_ema = int(ind_win.get("ema", ind_win.get("ema_window", 20)))
        window_rsi = int(ind_win.get("rsi", ind_win.get("rsi_window", 14)))

        macd_fast   = int(ind_win.get("macd_fast", 12))
        macd_slow   = int(ind_win.get("macd_slow", 26))
        macd_signal = int(ind_win.get("macd_signal", 9))

        bb_window = int(ind_win.get("bb_window", 20))
        bb_dev    = float(ind_win.get("bb_dev", 2.0))

        atr_win = int(ind_win.get("atr", ind_win.get("atr_window", 14)))
        adx_win = int(ind_win.get("adx", ind_win.get("adx_window", 14)))

        stoch_k_win = int(ind_win.get("stoch_k", ind_win.get("stoch_k_window", 14)))
        stoch_d_win = int(ind_win.get("stoch_d", ind_win.get("stoch_d_window", 3)))

        # Feature toggles (defaults keep backward-compatibility)
        toggles = dict(
            use_sma   = cfg.get("use_sma", True),
            use_ema   = cfg.get("use_ema", True),
            use_rsi   = cfg.get("use_rsi", True),
            use_macd  = cfg.get("use_macd", True),
            use_bbands= cfg.get("use_bbands", True),
            use_atr   = cfg.get("use_atr", True),
            use_adx   = cfg.get("use_adx", True),
            use_stoch = cfg.get("use_stoch", True),
            use_mtf_ma= cfg.get("use_mtf_ma", True),
        )

        # Indicator state configuration (oscillator & volatility regimes)
        use_indicator_states    = bool(cfg.get("use_indicator_states", False))
        rsi_overbought_level    = float(cfg.get("rsi_overbought_level", 70))
        rsi_oversold_level      = float(cfg.get("rsi_oversold_level", 30))
        stoch_overbought_level  = float(cfg.get("stoch_overbought_level", 80))
        stoch_oversold_level    = float(cfg.get("stoch_oversold_level", 20))
        bbw_compress_threshold  = float(cfg.get("bbw_compress_threshold", 0.05))
        bbw_expand_threshold    = float(cfg.get("bbw_expand_threshold", 0.20))

        # Momentum extensions
        use_ma_spread      = bool(cfg.get("use_ma_spread", False))
        use_price_ma_z     = bool(cfg.get("use_price_ma_z", False))
        use_crossover_bins = bool(cfg.get("use_crossover_bins", False))
        use_slope_diff     = bool(cfg.get("use_slope_diff", False))

        # Composite toggles
        use_reentry_mom        = bool(cfg.get("use_reentry_mom", False))
        use_ext_atr_low_adx    = bool(cfg.get("use_ext_atr_low_adx", False))
        use_squeeze_expansion  = bool(cfg.get("use_squeeze_expansion", False))
        use_atr_channel_break  = bool(cfg.get("use_atr_channel_breakout", False))
        use_trend_confirm      = bool(cfg.get("use_trend_confirm", False))
        use_mtf_alignment      = bool(cfg.get("use_mtf_alignment", False))
        use_vol_managed_mom    = bool(cfg.get("use_vol_managed_mom", False))
        use_macd_atr_ratio     = bool(cfg.get("use_macd_atr_ratio", False))

        # Effective lags / depth / rolling windows
        num_lags = int(lags if lags is not None else 10)
        if "lags" in cfg:       num_lags = int(cfg["lags"])
        if "lags_range" in cfg: num_lags = int(cfg["lags_range"])

        lag_depth = int(cfg.get("lag_depth", lag_depth if lag_depth is not None else 1))

        # ---------- 2) Base indicators ----------
        df = df.copy()
        price_col = cfg.get("price_col", "price")
        if price_col not in df.columns:
            price_col = "close" if "close" in df.columns else price_col

        base_cols: dict[str, pd.Series] = {}
        base_features: list[str] = []

        # CLEANUP: one canonical regime list used in two places later
        regime_cols = [
            "trend_score",
            "vol_score",
            "regime_id",
            "regime_trend",
            "regime_sideways",
            "regime_volatile",
        ]

        # Decide whether to reuse precomputed FeatureBank (only when not building it)
        use_fb = (
            not base_only
            and getattr(self, "_feature_bank_full", None) is not None
            and isinstance(getattr(self, "_feature_bank_meta", None), dict)
            and bool(getattr(self, "_feature_bank_meta", {}).get("base_features"))
        )

        if use_fb:
            try:
                fb = self._feature_bank_full
                meta = self._feature_bank_meta or {}
                base_features = list(meta.get("base_features", []))

                # Align FeatureBank slice to current df index
                fb_slice = fb.reindex(df.index)

                # Attach base/composite features from the bank
                df = pd.concat([df, fb_slice[base_features]], axis=1)
                df = df.loc[:, ~df.columns.duplicated(keep="last")]

            except Exception as e:
                if debug:
                    print(
                        f"⚠️ FeatureBank reuse failed in prepare_features; "
                        f"falling back to per-slice TA: {e}"
                    )
                use_fb = False
                base_features = []
                base_cols = {}

        if not use_fb:
            if "returns" in df:
                base_cols["rolling_std_20"] = df["returns"].rolling(20).std()
                base_features.append("rolling_std_20")

            # SMA / EMA
            if toggles["use_sma"] and price_col in df:
                name = f"sma_{window_sma}"
                base_cols[name] = ta.trend.sma_indicator(df[price_col], window=window_sma)
                base_features.append(name)

            if toggles["use_ema"] and price_col in df:
                name = f"ema_{window_ema}"
                base_cols[name] = ta.trend.ema_indicator(df[price_col], window=window_ema)
                base_features.append(name)

            # MACD (line, signal, diff)
            if toggles["use_macd"] and price_col in df:
                macd_obj = ta.trend.MACD(
                    df[price_col],
                    window_slow=macd_slow,
                    window_fast=macd_fast,
                    window_sign=macd_signal,
                )
                base_cols["macd_line"]   = macd_obj.macd()
                base_cols["macd_signal"] = macd_obj.macd_signal()
                base_cols["macd_diff"]   = macd_obj.macd_diff()
                base_features += ["macd_line", "macd_signal", "macd_diff"]

            # RSI
            if toggles["use_rsi"] and price_col in df:
                name = f"rsi_{window_rsi}"
                base_cols[name] = ta.momentum.RSIIndicator(df[price_col], window=window_rsi).rsi()
                base_features.append(name)

            # Bollinger Bands (+ width, %B)
            if toggles["use_bbands"] and price_col in df:
                bb = ta.volatility.BollingerBands(df[price_col], window=bb_window, window_dev=bb_dev)
                upper, lower = bb.bollinger_hband(), bb.bollinger_lband()
                base_cols["bb_upper"] = upper
                base_cols["bb_lower"] = lower
                base_cols["bb_pct"]   = (df[price_col] - lower) / (upper - lower)
                base_cols["bbw"]      = bb.bollinger_wband()
                base_features += ["bb_upper", "bb_lower", "bb_pct", "bbw"]

            # ATR
            if toggles["use_atr"]:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    name = f"atr_{atr_win}"
                    base_cols[name] = ta.volatility.AverageTrueRange(hi, lo, cl, window=atr_win).average_true_range()
                    base_features.append(name)

                    # ATR-normalized spread (optional toggle)
                    if toggles.get("use_spread_over_atr", False) and ("spread" in df.columns):
                        eps = 1e-8
                        atr_series = base_cols[name].astype(float).replace(0.0, np.nan)
                        spread_series = df["spread"].astype(float)
                        spread_atr = (spread_series / atr_series).replace([np.inf, -np.inf], np.nan)
                        base_cols[f"spread_atr_{atr_win}"] = spread_atr
                        base_features.append(f"spread_atr_{atr_win}")

            # Donchian-style price channels (multi-horizon high/low bands)
            use_donchian = bool(cfg.get("use_donchian", False))
            if use_donchian:
                hi = df.get("high", df.get(price_col))
                lo = df.get("low", df.get(price_col))
                cl = df.get("close", df.get(price_col))
                if (hi is not None) and (lo is not None) and (cl is not None):
                    w_s = int(cfg.get("donchian_window_short", 20))
                    w_l = int(cfg.get("donchian_window_long", 60))
                    for w in sorted({w_s, w_l}):
                        dc_high = hi.rolling(w, min_periods=max(5, w // 3)).max()
                        dc_low  = lo.rolling(w, min_periods=max(5, w // 3)).min()
                        up_col  = f"donchian_up_{w}"
                        dn_col  = f"donchian_dn_{w}"
                        bu_col  = f"donchian_break_up_{w}"
                        bd_col  = f"donchian_break_dn_{w}"
                        base_cols[up_col] = dc_high
                        base_cols[dn_col] = dc_low
                        base_cols[bu_col] = (cl >= dc_high).astype("int8")
                        base_cols[bd_col] = (cl <= dc_low).astype("int8")
                        base_features += [up_col, dn_col, bu_col, bd_col]

            # ADX
            if toggles["use_adx"]:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    name = f"adx_{adx_win}"
                    base_cols[name] = ta.trend.ADXIndicator(hi, lo, cl, window=adx_win).adx()
                    base_features.append(name)

            # Stochastic (K, D)
            if toggles.get("use_stoch", True):
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    k, d = int(stoch_k_win), int(stoch_d_win)
                    Hh = hi.rolling(k, min_periods=k).max()
                    Ll = lo.rolling(k, min_periods=k).min()
                    stoch_k = 100.0 * (cl - Ll) / (Hh - Ll).replace(0.0, np.nan)
                    base_cols["stoch_k"] = stoch_k
                    base_cols["stoch_d"] = stoch_k.rolling(d, min_periods=d).mean()
                    base_features += ["stoch_k", "stoch_d"]

            # Indicator states (oscillators & volatility compression/expansion)
            if use_indicator_states:
                if toggles.get("use_rsi", True):
                    rsi_col = f"rsi_{window_rsi}"
                    if rsi_col in base_cols:
                        rsi_ser = base_cols[rsi_col].astype(float)
                        rsi_state = pd.Series(0, index=rsi_ser.index, dtype="int8")
                        rsi_state[rsi_ser >= rsi_overbought_level] = 1
                        rsi_state[rsi_ser <= rsi_oversold_level] = -1
                        base_cols["rsi_state"] = rsi_state
                        if "rsi_state" not in base_features:
                            base_features.append("rsi_state")

                if toggles.get("use_stoch", True) and "stoch_k" in base_cols:
                    stoch_ser = base_cols["stoch_k"].astype(float)
                    st_state = pd.Series(0, index=stoch_ser.index, dtype="int8")
                    st_state[stoch_ser >= stoch_overbought_level] = 1
                    st_state[stoch_ser <= stoch_oversold_level] = -1
                    base_cols["stoch_state"] = st_state
                    if "stoch_state" not in base_features:
                        base_features.append("stoch_state")

                if toggles.get("use_bbands", True) and "bbw" in base_cols:
                    bbw_ser = base_cols["bbw"].astype(float)
                    vol_state = pd.Series(0, index=bbw_ser.index, dtype="int8")
                    vol_state[bbw_ser <= bbw_compress_threshold] = -1
                    vol_state[bbw_ser >= bbw_expand_threshold]   = 1
                    base_cols["vol_state_bbw"] = vol_state
                    if "vol_state_bbw" not in base_features:
                        base_features.append("vol_state_bbw")

            # SAR always on
            hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
            if (hi is not None) and (lo is not None) and (cl is not None):
                base_cols["sar"] = ta.trend.PSARIndicator(hi, lo, cl).psar()
                base_features.append("sar")

            # MTF MAs (provided by get_data)
            if toggles["use_mtf_ma"]:
                for c in ("mtf_ma_fast", "mtf_ma_slow"):
                    if c in df.columns and c not in base_features:
                        base_features.append(c)

            # Optional realized-volatility & bipower variation — add before expansion so they get lags/rolls
            if cfg.get("use_rv_features", False) and "returns" in df:
                w_s = int(cfg.get("rv_window_short", 30))
                w_l = int(cfg.get("rv_window_long", 120))
                def _rv(s, w):
                    rv2 = s.pow(2).rolling(w, min_periods=max(5, w//3)).sum()
                    return np.sqrt(rv2)
                def _bpv(s, w):
                    abs_r = s.abs(); prod = abs_r * abs_r.shift(1)
                    bpv = (np.pi/2.0) * prod.rolling(w, min_periods=max(5, w//3)).sum()
                    return np.sqrt(bpv.clip(lower=0))
                base_cols[f"rv_{w_s}"]  = _rv(df["returns"], w_s)
                base_cols[f"rv_{w_l}"]  = _rv(df["returns"], w_l)
                base_cols[f"bpv_{w_s}"] = _bpv(df["returns"], w_s)
                base_cols[f"bpv_{w_l}"] = _bpv(df["returns"], w_l)
                base_cols[f"rv_roc_{w_s}"] = base_cols[f"rv_{w_s}"].pct_change()
                base_cols[f"rv_roc_{w_l}"] = base_cols[f"rv_{w_l}"].pct_change()
                base_features += [f"rv_{w_s}", f"rv_{w_l}", f"bpv_{w_s}", f"bpv_{w_l}", f"rv_roc_{w_s}", f"rv_roc_{w_l}"]

            # Optional fractional-diff seed (included in expansion)
            def _fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
                w = [1.0]
                for k in range(1, size):
                    w_k = -w[-1] * (d - (k - 1)) / k
                    if abs(w_k) < thresh:
                        break
                    w.append(w_k)
                return np.array(w, dtype="float64")
            def _fracdiff(series: pd.Series, d: float = 0.4, max_size: int = 2000, thresh: float = 1e-4) -> pd.Series:
                s = series.astype("float64")
                w = _fracdiff_weights(d, min(max_size, len(s)), thresh=thresh)
                out = np.full(len(s), np.nan, dtype="float64")
                kmax = len(w) - 1; vals = s.values
                for t in range(kmax, len(s)):
                    window = vals[t - kmax : t + 1]
                    out[t] = float(np.dot(w[::-1], window))
                return pd.Series(out, index=s.index, name=f"fd_{getattr(series, 'name','x')}_d{d:.2f}")
            if cfg.get("use_fracdiff", False) and price_col in df:
                d = float(cfg.get("fracdiff_d", 0.4))
                fd = _fracdiff(df[price_col], d=d)
                base_cols[fd.name] = fd
                base_features.append(fd.name)

            # ---------- 3) Momentum extensions (AFTER base indicators) ----------
            price_s = df.get(price_col)
            sma_col = f"sma_{window_sma}"
            ema_col = f"ema_{window_ema}"
            _eps = 1e-8

            if use_ma_spread and (ema_col in base_cols) and (sma_col in base_cols):
                base_cols["ema_sma_spread"] = (base_cols[ema_col] - base_cols[sma_col])
                base_features.append("ema_sma_spread")

            if use_price_ma_z and (price_s is not None):
                if sma_col in base_cols:
                    sd_sma = price_s.rolling(window_sma, min_periods=max(5, window_sma//3)).std(ddof=0)
                    base_cols[f"price_sma_z_{window_sma}"] = (price_s - base_cols[sma_col]) / (sd_sma + _eps)
                    base_features.append(f"price_sma_z_{window_sma}")
                if ema_col in base_cols:
                    sd_ema = price_s.rolling(window_ema, min_periods=max(5, window_ema//3)).std(ddof=0)
                    base_cols[f"price_ema_z_{window_ema}"] = (price_s - base_cols[ema_col]) / (sd_ema + _eps)
                    base_features.append(f"price_ema_z_{window_ema}")

            if use_crossover_bins:
                if (sma_col in base_cols) and (price_s is not None):
                    base_cols["price_gt_sma"] = (price_s > base_cols[sma_col]).astype(int)
                    base_features.append("price_gt_sma")
                if (ema_col in base_cols) and (price_s is not None):
                    base_cols["price_gt_ema"] = (price_s > base_cols[ema_col]).astype(int)
                    base_features.append("price_gt_ema")
                trend_proxy = base_cols.get("macd_diff")
                if trend_proxy is None and (ema_col in base_cols) and (sma_col in base_cols):
                    trend_proxy = (base_cols[ema_col] - base_cols[sma_col])
                if trend_proxy is not None:
                    base_cols["ma_cross_up"] = (trend_proxy > 0).astype(int)
                    base_cols["ma_cross_dn"] = (trend_proxy < 0).astype(int)
                    base_features += ["ma_cross_up", "ma_cross_dn"]

            if use_slope_diff:
                w_sd = max(5, min(window_ema, window_sma)//2)
                x = base_cols.get("macd_diff")
                if x is None and (ema_col in base_cols) and (sma_col in base_cols):
                    x = (base_cols[ema_col] - base_cols[sma_col])
                if x is not None:
                    base_cols[f"ma_spread_slope{w_sd}"] = self.rolling_slope(pd.Series(x).ffill(), w_sd)
                    base_features.append(f"ma_spread_slope{w_sd}")

            # ---------- 4) Composite features (built from existing columns) ----------
            ema_s = base_cols.get(ema_col)
            atr_s = base_cols.get(f"atr_{atr_win}") if f"atr_{atr_win}" in base_cols else None
            adx_s = base_cols.get(f"adx_{adx_win}") if f"adx_{adx_win}" in base_cols else None
            bbw_s = base_cols.get("bbw")
            macd_d = base_cols.get("macd_diff")
            rsi_s  = base_cols.get(f"rsi_{window_rsi}") if f"rsi_{window_rsi}" in base_cols else None

            if ema_s is None and (price_s is not None):
                ema_s = ta.trend.ema_indicator(price_s, window=window_ema)
            if atr_s is None:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    atr_s = ta.volatility.AverageTrueRange(hi, lo, cl, window=atr_win).average_true_range()
            if adx_s is None:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    adx_s = ta.trend.ADXIndicator(hi, lo, cl, window=adx_win).adx()
            if bbw_s is None and price_s is not None:
                bb_tmp = ta.volatility.BollingerBands(price_s, window=bb_window, window_dev=bb_dev)
                bbw_s = bb_tmp.bollinger_wband()

            if use_reentry_mom and (price_s is not None) and (rsi_s is not None):
                if "bb_pct" in base_cols:
                    bb_pct = base_cols["bb_pct"]
                else:
                    bb_tmp = ta.volatility.BollingerBands(price_s, window=bb_window, window_dev=bb_dev)
                    upper, lower = bb_tmp.bollinger_hband(), bb_tmp.bollinger_lband()
                    bb_pct = (price_s - lower) / (upper - lower + _eps)
                reenter = ((bb_pct.shift(1) < 0) & (bb_pct >= 0)).astype(float)
                rsi_slope = self.rolling_slope(rsi_s.ffill(), 5)
                base_cols["reentry_mom"] = reenter * rsi_slope.clip(lower=0.0)
                base_features.append("reentry_mom")

            if use_ext_atr_low_adx and (price_s is not None) and (ema_s is not None) and (atr_s is not None) and (adx_s is not None):
                ext_atr = (price_s - ema_s).abs() / (atr_s + _eps)
                adx_norm = (adx_s / 50.0).clip(0.0, 1.0)
                base_cols["ext_atr_low_adx"] = ext_atr * (1.0 - adx_norm)
                base_features.append("ext_atr_low_adx")

            if use_squeeze_expansion and (bbw_s is not None) and (adx_s is not None):
                w_sq = int(cfg.get("squeeze_window", 300))
                q    = float(cfg.get("squeeze_quantile", 0.10))
                def _pct_rank_last(x: np.ndarray) -> float:
                    s = pd.Series(x)
                    return float(s.rank(pct=True).iloc[-1]) if len(s) else np.nan
                bbw_rank = bbw_s.rolling(w_sq, min_periods=max(30, w_sq//5)).apply(_pct_rank_last, raw=True)
                adx_sl = self.rolling_slope(adx_s.ffill(), 5).clip(lower=0.0)
                base_cols["squeeze_expansion"] = ((q - bbw_rank).clip(lower=0.0)) * adx_sl
                base_features.append("squeeze_expansion")

            if use_atr_channel_break and (price_s is not None) and (ema_s is not None) and (atr_s is not None):
                m = float(cfg.get("atr_channel_mult", 1.5))
                base_cols["atr_ch_up"] = ((price_s - ema_s) / (atr_s + _eps)) - m
                base_cols["atr_ch_dn"] = ((ema_s - price_s) / (atr_s + _eps)) - m
                base_features += ["atr_ch_up", "atr_ch_dn"]

            if use_trend_confirm and (price_s is not None) and (ema_s is not None) and (adx_s is not None):
                adx_sl = self.rolling_slope(adx_s.ffill(), 5).clip(lower=0.0)
                macd_ok = (macd_d > 0).astype(float) if macd_d is not None else 1.0
                price_ok = (price_s > ema_s).astype(float)
                base_cols["trend_confirm"] = price_ok * macd_ok * adx_sl
                base_features.append("trend_confirm")

            if use_mtf_alignment and (price_s is not None) and (ema_s is not None) and ("mtf_ma_fast" in df):
                mtf_sl = self.rolling_slope(df["mtf_ma_fast"].ffill(), 5)
                base_cols["mtf_align"] = ((price_s > ema_s).astype(float)) * (mtf_sl > 0).astype(float)
                base_features.append("mtf_align")

            if use_vol_managed_mom and (price_s is not None) and (ema_s is not None) and (atr_s is not None):
                base_cols["mom_vmm"] = (price_s - ema_s) / (atr_s + _eps)
                base_features.append("mom_vmm")

            if use_macd_atr_ratio and (macd_d is not None) and (atr_s is not None):
                base_cols["macd_atr"] = macd_d / (atr_s + _eps)
                base_features.append("macd_atr")

            # ---------- 5) One-shot concat of base columns ----------
            if base_cols:
                df = pd.concat([df, pd.DataFrame(base_cols, index=df.index)], axis=1)
                df = df.loc[:, ~df.columns.duplicated(keep="last")]

            # ---- Regime features (trend_score, vol_score, regime_id/one-hot) ----
            if bool(cfg.get("use_regime_features", True)):
                df = self._attach_regime_columns(df, cfg)
                for c in regime_cols:  # CLEANUP: single source of truth
                    if c in df.columns and c not in base_features:
                        base_features.append(c)

        # --- Base-only mode for FeatureBank build ----------------------------
        if base_only:
            self._last_used_features = list(base_features)
            return df.copy(), list(base_features)

        # ---------- 6) Lags and rolling expansions ----------
        new_cols = {}
        missing_for_expansion = []  # CLEANUP: aggregate missing warnings

        if cfg.get("include_raw_lags", True) and "returns" in df:
            for lag in range(1, num_lags + 1):
                new_cols[f"returns_lag{lag}"] = df["returns"].shift(lag)

        for feat in base_features:
            if feat not in df.columns:
                missing_for_expansion.append(feat)
                continue
            for k in range(1, lag_depth + 1):
                new_cols[f"{feat}_lag{k}"] = df[feat].shift(k)
            for w in roll_windows:
                new_cols[f"{feat}_rollmean{w}"]  = df[feat].rolling(w).mean()
                new_cols[f"{feat}_rollstd{w}"]   = df[feat].rolling(w).std()
                new_cols[f"{feat}_rollslope{w}"] = self.rolling_slope(df[feat], w)

        # DEBUG: print once, not per feature
        if debug and missing_for_expansion:
            print(
                f"⚠️ Skipping lag/rolls for {len(missing_for_expansion)} base_features missing in df "
                f"(showing up to 8): {missing_for_expansion[:8]}"
            )

        # ---------- 7) Hour ----------
        if cfg.get("include_hour", True):
            new_cols["hour"] = df.index.hour

        if cfg.get("include_hour_cyclic", True):
            try:
                hour_vals = df.index.hour.to_numpy(dtype="float32", copy=False)
            except Exception as e:
                hour_vals = None
                _debug_once("hour_to_numpy", e)

            if hour_vals is not None and len(hour_vals) > 0:
                hour_rad = 2.0 * np.pi * hour_vals / 24.0
                new_cols["hour_sin"] = np.sin(hour_rad)
                new_cols["hour_cos"] = np.cos(hour_rad)

        df_out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # ---------- 8) Finalize feature list, fill, dropna ----------
        features: list[str] = [f for f in (list(new_cols.keys()) + base_features) if f in df_out.columns]

        if bool(cfg.get("use_regime_features", True)):
            for c in regime_cols:  # CLEANUP: single list
                if c in df_out.columns and c not in features:
                    features.append(c)

        # Forward-fill MTF if requested
        mtf_fillna = cfg.get("mtf_fillna_method", None)
        if mtf_fillna == "ffill":
            for mtf_col in ("mtf_ma_fast", "mtf_ma_slow"):
                if mtf_col in df_out:
                    df_out[mtf_col] = df_out[mtf_col].ffill()
                    for col in df_out.columns:
                        if col.startswith(mtf_col + "_"):
                            df_out[col] = df_out[col].ffill()

        dropna_subset = [f for f in features if f in df_out.columns]
        if dropna_subset:
            df_out.dropna(subset=dropna_subset, inplace=True)
        else:
            if debug:
                print("⚠️ No valid features for dropna_subset, running full dropna().")
            df_out.dropna(inplace=True)

        # Deduplicate columns if any
        if df_out.columns.duplicated().any():
            dup_cols = df_out.columns[df_out.columns.duplicated()].tolist()
            if debug:
                print(f"⚠️ Duplicate columns detected and removed: {dup_cols}")
            else:
                # CLEANUP: compact one-liner in non-debug
                print(f"⚠️ Duplicate columns detected and removed: n={len(dup_cols)}")
            df_out = df_out.loc[:, ~df_out.columns.duplicated()]

        # Keep last-used feature list for logging/diagnostics
        features = list(features)
        self._last_used_features = list(features)

        # Store engineered slice in per-run cache so later calls can reuse it.
        # We store the features as an immutable tuple to avoid accidental mutation.
        if cache_enabled:
            self._feat_cache[cache_key] = (df_out, tuple(features))

            # ---- Patch: hard cap / eviction (optional) ----
            if feat_cache_max_entries > 0:
                try:
                    evicted = 0
                    while isinstance(self._feat_cache, dict) and len(self._feat_cache) > feat_cache_max_entries:
                        oldest_key = next(iter(self._feat_cache))
                        try:
                            self._feat_cache.pop(oldest_key, None)
                        except Exception:
                            break

                        try:
                            if hasattr(self, "_feat_cache_bytes") and isinstance(self._feat_cache_bytes, dict):
                                b = int(self._feat_cache_bytes.pop(oldest_key, 0) or 0)
                                if hasattr(self, "_feat_cache_cur_bytes"):
                                    self._feat_cache_cur_bytes = max(
                                        0, int(getattr(self, "_feat_cache_cur_bytes", 0)) - b
                                    )
                        except Exception as e:
                            _debug_once("feat_cache_eviction_bytes", e)

                        evicted += 1

                    if evicted:
                        self._feat_cache_evictions = int(getattr(self, "_feat_cache_evictions", 0)) + int(evicted)
                        if LOG_MODE in {"COMPACT", "DEBUG"}:
                            try:
                                cur_bytes = int(getattr(self, "_feat_cache_cur_bytes", 0))
                                print(
                                    f"[FEAT_CACHE] EVICT evicted={evicted} cap={feat_cache_max_entries} "
                                    f"entries_now={len(self._feat_cache)} cache_mb={cur_bytes/1024/1024:.1f}"
                                )
                            except Exception as e:
                                _debug_once("feat_cache_evict_diag", e)
                except Exception as e:
                    _debug_once("feat_cache_eviction_outer", e)

            # Diagnostics: cache miss/store
            self._feat_cache_misses = int(getattr(self, "_feat_cache_misses", 0)) + 1
            if LOG_MODE in {"COMPACT", "DEBUG"}:
                try:
                    n_entries = len(self._feat_cache) if isinstance(self._feat_cache, dict) else -1
                    _deep_mem = bool(LOG_MODE == "DEBUG" or getattr(self, "debug", False))
                    try:
                        est_bytes = int(df_out.memory_usage(deep=_deep_mem).sum())
                    except Exception:
                        est_bytes = int(df_out.memory_usage(deep=False).sum())

                    prev = 0
                    try:
                        prev = int(self._feat_cache_bytes.get(cache_key, 0))
                    except Exception:
                        prev = 0
                    try:
                        self._feat_cache_bytes[cache_key] = est_bytes
                    except Exception as e:
                        _debug_once("feat_cache_bytes_set", e)

                    cur_bytes = int(getattr(self, "_feat_cache_cur_bytes", 0))
                    cur_bytes = max(0, int(cur_bytes) + int(est_bytes) - int(prev))
                    self._feat_cache_cur_bytes = int(cur_bytes)

                    self._feat_cache_est_bytes = int(getattr(self, "_feat_cache_est_bytes", 0)) + est_bytes

                    hits = int(getattr(self, "_feat_cache_hits", 0))
                    misses = int(getattr(self, "_feat_cache_misses", 0))
                    denom = hits + misses
                    hit_rate = (hits / denom) if denom > 0 else 0.0
                    do_print = (
                        (LOG_MODE == "DEBUG")
                        or (self._feat_cache_misses in {1, 2, 5, 10, 20, 50, 100})
                        or (self._feat_cache_misses % 25 == 0)
                    )
                    if do_print:
                        print(
                            f"[FEAT_CACHE] MISS entries={n_entries} "
                            f"hits={hits} misses={misses} hit_rate={hit_rate:.2%} "
                            f"+{est_bytes/1024/1024:.1f}MB cache_mb={cur_bytes/1024/1024:.1f} "
                            f"feats={len(features)}"
                        )
                except Exception as e:
                    _debug_once("feat_cache_miss_diag", e)

        return df_out, features

    
    def scale_features(self, df, features, means=None, stds=None, log_id=None):
        """
        Standardizes feature columns and optionally logs mean/std per fold.

        Parameters:
        - df (pd.DataFrame): Data with features.
        - features (list): Columns to scale.
        - means/stds (pd.Series): Optional for applying saved scaling.
        - log_id (str): Optional string identifier to log per-fold stats.

        Returns:
        - df_scaled, means, stds
        """
        if means is None or stds is None:
            means = df[features].mean()
            stds  = df[features].std()

        # Avoid divide-by-zero
        stds = stds.where(stds != 0, 1e-8)

        df = df.copy()
        df[features] = (df[features] - means) / stds

        # Replace infs that can still appear from pathological inputs
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)

        return df, means, stds

    def label_with_neutral(self, returns, threshold):
        """
        Creates classification labels for ML based on return thresholds.

        Parameters:
        - returns (np.array or pd.Series): Returns series.
        - threshold (float): Absolute return threshold to define class boundaries.

        Returns:
        - np.array: Array of integer labels:
            - 2 (buy/long) if returns > threshold,
            - 0 (sell/short) if returns < -threshold,
            - 1 (neutral/hold) otherwise.
        """

        labels = np.where(returns > threshold, 2, np.where(returns < -threshold, 0, 1))
        # compact stats only
        unique, counts = np.unique(labels, return_counts=True)
        if self._is_debug():
            print("Label counts:", dict(zip(unique, counts)), f"| thr={threshold}")
            
        return labels

    def _fit_keras_with_cv_controls(
        self,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        base_epochs=20,
        base_batch=128,
        verbose=0,
        validation_split_if_needed=0.10,
        extra_callbacks=None,
    ):
        """
        Centralizes keras.fit so deep models run fast when inside Optuna CV.
        - Uses model.early_stop_callback if it exists.
        - If no explicit (X_val, y_val) but EarlyStopping exists, uses a **time-ordered tail holdout**
        (never Keras validation_split with shuffling).
        - During CV (_in_optuna_cv=True), caps epochs/batch from features_config.
        """

        # Defaults
        epochs = int(base_epochs)
        batch  = int(base_batch)
        callbacks = []

        # Respect EarlyStopping created by builders
        early_cb = getattr(model, "early_stop_callback", None)
        if early_cb is not None:
            callbacks.append(early_cb)

        # Any extra callbacks (e.g., time-limit)
        if extra_callbacks:
            callbacks.extend(extra_callbacks if isinstance(extra_callbacks, (list, tuple)) else [extra_callbacks])

        # CV caps (multi-fidelity proxy during Optuna CV)
        if getattr(self, "_in_optuna_cv", False):
            cfg = getattr(self, "features_config", {}) or {}
            # We optionally tag models with a short name ("cnn", "lstm", "transformer")
            # so that different deep families can have different CV budgets.
            #
            # Research basis:
            # - Prechelt (1998, Neural Networks 11(4)) shows that early stopping
            #   + limited epochs can reduce deep training time ~4x with minimal
            #   generalisation loss.
            # - Wu et al. (2020, AISTATS) and Won et al. (2025, ICT Express) argue
            #   for multi-fidelity HPO: use cheaper fidelities (fewer epochs/samples)
            #   during tuning and reserve full budgets for final refit.
            model_tag = getattr(model, "_mlb_model_tag", None)

            # Requested (from model config / Optuna) vs CV caps (compute control)
            req_epochs = int(epochs)
            req_batch  = int(batch)

            # Global CV caps (fallback)
            epochs_cap_default = int(cfg.get("deep_cv_max_epochs", req_epochs))
            batch_cap_default  = int(cfg.get("deep_cv_batch_size", req_batch))

            if model_tag:
                # Optional per-model overrides, e.g. cnn_cv_max_epochs, lstm_cv_max_epochs, ...
                epochs_cap = int(cfg.get(f"{model_tag}_cv_max_epochs", epochs_cap_default))
                batch_cap  = int(cfg.get(f"{model_tag}_cv_batch_size",  batch_cap_default))
            else:
                epochs_cap = epochs_cap_default
                batch_cap  = batch_cap_default

            # Apply caps (so tuning matters up to the cap)
            epochs = min(req_epochs, epochs_cap)
            batch  = min(req_batch,  batch_cap)
            
            print(
                f"[DEEP-CV] model={model_tag or 'generic'} | "
                f"epochs={epochs} (req={req_epochs}, cap={epochs_cap}), "
                f"batch_size={batch} (req={req_batch}, cap={batch_cap}) "
                f"(patience={getattr(early_cb, 'patience', 'NA')})"
            )

        # --- Time-ordered validation (tail split) ---
        use_tail_val = (X_val is None and y_val is None and early_cb is not None
                        and validation_split_if_needed and validation_split_if_needed > 0.0)

        if use_tail_val:
            n = int(getattr(X_train, "shape", [0])[0])
            n_val = max(1, int(round(n * float(validation_split_if_needed))))
            n_val = min(max(1, n_val), n - 1) if n > 1 else 1
            split = n - n_val

            X_tr, y_tr = X_train[:split], y_train[:split]
            X_v,  y_v  = X_train[split:], y_train[split:]
        else:
            X_tr, y_tr = X_train, y_train
            X_v,  y_v  = X_val,   y_val

        # --- Optional class weights (helps LSTM/Transformer with skewed labels) ---
        class_weight = None
        try:
            cfg = getattr(self, "features_config", {}) or {}
            use_cw = bool(cfg.get("deep_use_class_weight", False))
            # You can also enable per-model: lstm_use_class_weight / transformer_use_class_weight
            if not use_cw:
                use_cw = bool(cfg.get("lstm_use_class_weight", False) or cfg.get("transformer_use_class_weight", False))
            if use_cw:
                y_for_cw = np.ravel(y_tr if use_tail_val else y_train)
                classes = np.unique(y_for_cw)
                if classes.size >= 2:
                    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_for_cw)
                    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
                    print(f"[CLASS-WEIGHT] {class_weight}")
        except Exception as _e:
            print(f"⚠️ class_weight computation skipped: {_e}")

        fit_kwargs = dict(
            x=X_tr, y=y_tr,
            epochs=epochs,
            batch_size=batch,
            verbose=verbose,
            shuffle=False,  # <-- time-series safe
        )


        if X_v is not None and y_v is not None:
            fit_kwargs.update({"validation_data": (X_v, y_v)})

        if class_weight is not None:
            fit_kwargs.update({"class_weight": class_weight})

        history = model.fit(callbacks=callbacks, **fit_kwargs)
        return history
    
    def _debug_dump_first_bars(self, index, raw_classes=None, max_conf=None, final_preds=None, n=10, label=""):
        """
        Print a tiny table with the first n window-end decisions aligned to eval index.
        Only runs if self._dbg_first_bars is True (set in real_trading_simulation).
        """
        try:
            if (not getattr(self, "_dbg_first_bars", False)) or getattr(self, "_in_cv", False):
                return
            import numpy as _np
            idx = pd.Index(index)
            if len(idx) == 0:
                print("ℹ️ [debug-dump] empty index.")
                return
            n = int(min(n, len(idx)))

            def _as_arr(a, length, fill=_np.nan):
                if a is None:
                    return _np.full(length, fill, dtype=float)
                arr = _np.asarray(a)
                return arr[:length] if len(arr) >= length else _np.pad(arr, (0, length - len(arr)), constant_values=fill)

            # make equal length slices
            m = n
            rc  = _as_arr(raw_classes, m, _np.nan)
            mc  = _as_arr(max_conf,   m, _np.nan)
            fp  = _as_arr(final_preds, m, _np.nan)

            print(f"\n🔎 First {m} window-end decisions{(' — '+str(label)) if label else ''}")
            print("timestamp                 | raw_cls | max_conf | final_pred")
            for i in range(m):
                ts = str(idx[i])
                rc_i = "NA" if not _np.isfinite(rc[i]) else f"{int(rc[i])}"
                mc_i = "NA" if not _np.isfinite(mc[i]) else f"{float(mc[i]):.3f}"
                fp_i = "NA" if not _np.isfinite(fp[i]) else f"{int(fp[i])}"
                print(f"{ts:>23} | {rc_i:>7} | {mc_i:>8} | {fp_i:>10}")
        except Exception as e:
            print(f"[debug-dump] skipped: {e}")
                               
                               
    def run_pbo_mcs_analysis(self):
        """
        Post-hoc analysis of WFO/WFS results using a CSCV-style PBO estimate
        and a simple Model Confidence Set (MCS) approximation.

        This function is **read-only** with respect to the main pipeline:
        it only consumes self._wfo_monthly_records and does not modify
        models, thresholds, or trading results.

        Returns
        -------
        dict or None
            Dictionary with keys:
            - 'matrix': DataFrame (index=test_end, columns=strategy_id)
            - 'mean_return': per-strategy mean monthly return
            - 'std_return': per-strategy std of monthly return
            - 'sharpe_like': per-strategy mean/std
            - 'pbo': estimated Probability of Backtest Overfitting (0–1 or NaN)
            - 'mcs_strategies': list of strategy_ids in a simple MCS proxy
        """
        import numpy as _np
        import pandas as _pd

        recs = getattr(self, "_wfo_monthly_records", None)
        if not recs:
            log_print("[PBO/MCS] No monthly records available; skipping analysis.", level="COMPACT")
            return None


        df = _pd.DataFrame(recs)
        if df.empty:
            print("[PBO/MCS] Monthly records DataFrame empty; skipping analysis.")
            return None
        
        # Ensure datetime ordering by month end
        df["test_end"] = _pd.to_datetime(df["test_end"])
        df = df.dropna(subset=["test_end", "strategy_id", "strategy_return"])
        df = df.sort_values(["test_end", "strategy_id"])

        if df.empty:
            log_print(
                "[PBO/MCS] No valid (test_end, strategy_id, strategy_return) rows; skipping.",
                level="COMPACT",
            )
            return None

        # Build strategy × month matrix of monthly returns
        mat = df.pivot_table(
            index="test_end",
            columns="strategy_id",
            values="strategy_return",
            aggfunc="first",
        ).sort_index()

        # Drop strategies with too few months
        min_months = 6
        valid_cols = [c for c in mat.columns if mat[c].notna().sum() >= min_months]
        mat = mat[valid_cols]

        if mat.shape[1] < 2:
            log_print(
                "[PBO/MCS] Need ≥2 strategies with sufficient history for PBO/MCS; skipping.",
                level="COMPACT",
            )
            return None


        # Basic per-strategy summary
        mean_ret = mat.mean(axis=0)
        std_ret = mat.std(axis=0, ddof=1)
        sharpe_like = mean_ret / std_ret.replace(0.0, _np.nan)

        # --- CSCV-style PBO estimate (Bailey et al.) ---
        R = mat.to_numpy(dtype=float)
        T, S = R.shape

        n_splits = min(200, max(20, T * 10))  # scale with #months, but cap for runtime
        omegas = []

        rng = _np.random.default_rng(seed=42)  # deterministic for reproducibility
        for _ in range(n_splits):
            # Random train/test split over months (roughly half-half)
            if T < 4:
                break
            train_idx = _np.sort(rng.choice(T, size=max(2, T // 2), replace=False))
            test_mask = _np.ones(T, dtype=bool)
            test_mask[train_idx] = False
            if test_mask.sum() < 2:
                continue

            train_mask = _np.zeros(T, dtype=bool)
            train_mask[train_idx] = True

            R_train = R[train_mask]
            R_test = R[test_mask]

            # In-sample mean per strategy; pick best
            is_mean = _np.nanmean(R_train, axis=0)
            if _np.all(~_np.isfinite(is_mean)):
                continue
            best_idx = int(_np.nanargmax(is_mean))

            # Out-of-sample performance for all strategies
            oos_mean = _np.nanmean(R_test, axis=0)
            if not _np.isfinite(oos_mean[best_idx]):
                continue

            # Empirical OOS quantile of the chosen strategy
            # rank 1=worst, S=best  → quantile in (0,1)
            ranks = _np.argsort(_np.argsort(oos_mean))  # 0-based rank
            u = (ranks[best_idx] + 1) / float(S + 1e-9)
            if u <= 0.0 or u >= 1.0 or not _np.isfinite(u):
                continue

            # Overfitting statistic ω = logit(u)
            omega = _np.log(u / (1.0 - u))
            omegas.append(omega)

        if omegas:
            omegas = _np.asarray(omegas, dtype=float)
            pbo = float(_np.mean(omegas <= 0.0))
        else:
            pbo = float("nan")

        # --- Simple MCS proxy (NOT full Hansen–Lunde–Nason MCS) ---
        # Keep strategies whose mean return is within 1 std-error of the best.
        T_eff = float(mat.shape[0])
        best_mean = float(mean_ret.max())
        best_se = float(std_ret[mean_ret.idxmax()] / _np.sqrt(max(T_eff, 1.0)))
        # Allow a small band around the best
        band = best_mean - best_se
        mcs_strategies = [sid for sid, mu in mean_ret.items() if mu >= band]

        summary = {
            "matrix": mat,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "sharpe_like": sharpe_like,
            "pbo": pbo,
            "mcs_strategies": mcs_strategies,
        }

        try:
            log_print("\n[PBO/MCS] Per-strategy summary (mean, std, sharpe-like):", level="COMPACT")
            log_print(
                _pd.DataFrame({
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "sharpe_like": sharpe_like,
                }).to_string(),
                level="COMPACT",
            )
            log_print(
                f"[PBO/MCS] Estimated PBO = {pbo:.3f} based on {len(omegas) if omegas else 0} splits.",
                level="COMPACT",
            )
            log_print(
                f"[PBO/MCS] MCS proxy strategies: {mcs_strategies}",
                level="COMPACT",
            )
        except Exception:
            pass


        return summary

    def _is_deep_model_type(self, model_type: str) -> bool:
        """Return True if this model family uses TF/Keras in this engine."""
        mt = str(model_type or "").lower().strip()
        if mt in {"cnn", "lstm", "transformer"}:
            return True
        if mt in {"ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime"}:
            return True
        return False

    def _maybe_configure_tf_runtime_once(self, model_type: str) -> None:
        """Configure TF runtime knobs only when a deep model is actually used (once per instance)."""
        if not self._is_deep_model_type(model_type):
            return
        if getattr(self, "_tf_runtime_configured", False):
            return

        try:
            import os
            _threads = int(os.getenv("BLAS_THREADS_PER_TRIAL", os.getenv("MLB_THREADS", "16")))
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_threads))
            os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(max(2, min(4, _threads // 4))))
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(k, str(_threads))
        except Exception:
            pass

        try:
            import tensorflow as tf
            try:
                for _gpu in tf.config.list_physical_devices("GPU"):
                    try:
                        tf.config.experimental.set_memory_growth(_gpu, True)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
            try:
                tf.config.set_soft_device_placement(True)
            except Exception:
                pass
        except Exception:
            pass

        self._tf_runtime_configured = True

    def _get_cost_arrays_aligned(self, cost_df, index):
        """Align cost columns once and return (returns_series, spread_np, slippage_np)."""
        import numpy as _np
        import pandas as _pd

        if cost_df is None or index is None:
            return _pd.Series([], dtype=float), _np.asarray([], dtype=_np.float32), _np.asarray([], dtype=_np.float32)

        aligned = cost_df.reindex(index)

        # returns: keep as Series for realized_vol() (pandas rolling)
        try:
            rets = _pd.to_numeric(aligned.get("returns", 0.0), errors="coerce").astype(float)
        except Exception:
            rets = _pd.Series(0.0, index=index, dtype=float)

        def _col_to_f32(name: str):
            try:
                s = _pd.to_numeric(aligned.get(name, 0.0), errors="coerce")
                return s.to_numpy(dtype=_np.float32, copy=False)
            except Exception:
                return _np.zeros(len(index), dtype=_np.float32)

        sprd = _col_to_f32("spread")
        slip = _col_to_f32("slippage_bps")
        return rets, sprd, slip


    def _ensure_cost_columns(self, df, config):
        """
        Attach real per-bar cost columns (no synthetic means).
        - 'spread' is copied from self.data (if present).
        - 'slippage_bps' is volatility-aware: base→high on high-vol bars.
        """
        import numpy as _np
        import pandas as pd
        from utilsNoWFO import realized_vol

        # 0) Guard
        try:
            use_costs = bool(config.get("eval_use_trading_costs", getattr(self, "trading_costs", True)))
        except Exception:
            use_costs = bool(getattr(self, "trading_costs", True))
        if (not use_costs) or df is None or len(df) == 0:
            return df

        df = df.copy()

        # 1) Spread: copy the real per-bar series from self.data if missing
        if "spread" not in df.columns:
            try:
                if hasattr(self, "data") and isinstance(self.data, pd.DataFrame) and "spread" in self.data.columns:
                    df["spread"] = self.data["spread"].reindex(df.index)
            except Exception:
                pass  # leave missing; metrics fn tolerates it
        else:
            try:
                _s = pd.to_numeric(df["spread"], errors="coerce").astype(float)
                if hasattr(self, "data") and isinstance(self.data, pd.DataFrame) and "spread" in self.data.columns:
                    _s = _s.combine_first(self.data["spread"].reindex(df.index))
                df["spread"] = _s.fillna(0.0)
            except Exception:
                df["spread"] = 0.0
                
                
        # ANCHOR: # 2) Vol-aware slippage_bps per bar (0.20 base → 0.30 on high vol)
        # Price series is needed to normalize spread (price units) into fractional drag.
        # compute_full_evaluation_metrics() uses 'price' or 'mid_close' if available.
        if ("price" not in df.columns) and ("mid_close" not in df.columns):
            try:
                if hasattr(self, "data") and isinstance(self.data, pd.DataFrame):
                    if "price" in self.data.columns:
                        df["price"] = pd.to_numeric(self.data["price"].reindex(df.index), errors="coerce").astype(float)
                    elif "mid_close" in self.data.columns:
                        df["mid_close"] = pd.to_numeric(self.data["mid_close"].reindex(df.index), errors="coerce").astype(float)
            except Exception:
                pass

        # --- Safe config access (supports config=None + df.attrs fallback) ---
        cfg = config if isinstance(config, dict) else {}
        try:
            cfg_from_attrs = dict(df.attrs.get("features_config", {}) or {})
        except Exception:
            cfg_from_attrs = {}
            
            
        # Also allow fallback to self.features_config (train-anchored values persisted there)
        try:
            cfg_from_self = dict(getattr(self, "features_config", {}) or {})
        except Exception:
            cfg_from_self = {}

        def _get_cfg(k, default=None):
            # explicit config wins, then df.attrs, then self.features_config
            return cfg.get(k, cfg_from_attrs.get(k, cfg_from_self.get(k, default)))

        # 2) Vol-aware slippage_bps per bar (base → high on high vol; MED fallback if thr missing)
        if "slippage_bps" not in df.columns:
            base = float(_get_cfg("eval_slip_bps_lo", _get_cfg("cv_slippage_bps_base", 0.20)))
            high = float(_get_cfg("eval_slip_bps_hi", _get_cfg("cv_slippage_bps_high", 0.30)))
            # Optional middle regime (used as safe fallback if high-vol threshold is missing)
            med  = float(_get_cfg("eval_slip_bps_med", _get_cfg("cv_slippage_bps_med", (base + high) / 2.0)))
            vol_w = int(_get_cfg("vol_window_bars", 48))
            qhi   = float(_get_cfg("high_vol_q", 0.80))

            # Optional override: caller may provide a precomputed (train-anchored) threshold.
            # If not provided, DO NOT derive a threshold from the eval df (leakage).
            thr_override = _get_cfg("high_vol_thr", None)
            try:
                thr_override = float(thr_override) if thr_override is not None else None
            except Exception:
                thr_override = None

            # Last-chance fallback: pull a train-anchored threshold cached on the instance.
            # This prevents the LeakageGuard path when callers forget to pass high_vol_thr.
            if thr_override is None:
                try:
                    _thr_last = getattr(self, "_last_high_vol_thr_train", None)
                    _thr_last = float(_thr_last) if _thr_last is not None else None
                    if _thr_last is not None and _np.isfinite(_thr_last):
                        thr_override = _thr_last
                except Exception:
                    pass
            if thr_override is None:
                try:
                    _thr_fc = (getattr(self, "features_config", {}) or {}).get("high_vol_thr", None)
                    _thr_fc = float(_thr_fc) if _thr_fc is not None else None
                    if _thr_fc is not None and _np.isfinite(_thr_fc):
                        thr_override = _thr_fc
                except Exception:
                    pass

            try:
                if "returns" in df.columns:
                    rv = realized_vol(df["returns"].astype(float), window=vol_w)

                    if thr_override is not None and _np.isfinite(thr_override):
                        thr = thr_override
                        mask = (rv >= thr)
                        if getattr(self, "_is_debug", lambda: False)():
                            print(f"[Costs] Using provided high_vol_thr={thr:.6g} (q={qhi:.2f}, vol_w={vol_w})")

                        # Normal regime-aware assignment
                        df["slippage_bps"] = _np.where(mask, high, base).astype(float)

                    else:
                        # Leakage guard: no train-anchored threshold was passed.
                        # IMPORTANT: fallback to MED slippage (do NOT punish all bars with the max regime).
                        print("[Costs][LeakageGuard] high_vol_thr missing; applying MED slippage for all eval bars.")
                        df["slippage_bps"] = float(med)

                else:
                    # No returns column to compute RV: default to MED (safer than assuming HI, less strict than LO)
                    df["slippage_bps"] = float(med)

            except Exception:
                # Hard fallback
                df["slippage_bps"] = float(med)

        return df


    @contextmanager
    def _persist_results_guard(self, persist_results: bool = True):
        # Always run optional cleanup on context exit (even when persisting results).
        # This is used to aggressively release TensorFlow state between Optuna CV folds.
        if bool(persist_results):
            try:
                yield
            finally:
                try:
                    self._maybe_tf_cleanup()
                except Exception:
                    pass
            return
        _snap = {}
        try:
            d = getattr(self, '__dict__', {}) or {}
            for k, v in list(d.items()):
                if (
                    k in ('results', 'results_full', '_cv_last_eval_df', '_cv_fold_eval_frames')
                    or k.startswith('_last_')
                    or k.startswith('_cv_last_')
                ):
                    _snap[k] = v
            yield
        finally:
            try:
                d = getattr(self, '__dict__', {}) or {}
                cur_keys = set(d.keys())
                snap_keys = set(_snap.keys())
                # Remove any newly-created ephemeral keys
                for k in (cur_keys - snap_keys):
                    if (
                        k in ('results', 'results_full', '_cv_last_eval_df', '_cv_fold_eval_frames')
                        or k.startswith('_last_')
                        or k.startswith('_cv_last_')
                    ):
                        try:
                            delattr(self, k)
                        except Exception:
                            pass
                # Restore snapshots
                for k, v in _snap.items():
                    try:
                        setattr(self, k, v)
                    except Exception:
                        pass
            except Exception:
                pass
            # Optional TF cleanup (CV deep models) even when we snapshot/restore results.
            try:
                self._maybe_tf_cleanup()
            except Exception:
                pass

    def _maybe_tf_cleanup(self):
        """Best-effort memory cleanup hook (primarily for Optuna CV)."""
        try:
            do = bool(getattr(self, "_tf_cleanup_do", False))
            if not do:
                return

            # Drop model reference if requested
            if bool(getattr(self, "_tf_cleanup_del_model", False)):
                try:
                    if hasattr(self, "model"):
                        self.model = None
                except Exception:
                    pass

            # Clear TF/Keras graph state (helps prevent accumulation across folds)
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

            # Release Python-side allocations
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass

            # Small yield to let allocators settle (no functional effect)
            try:
                time.sleep(0.05)
            except Exception:
                pass
        finally:
            # Never let flags leak into the next call
            try:
                self._tf_cleanup_do = False
                self._tf_cleanup_del_model = False
            except Exception:
                pass



    def _fit_deep_calibration_and_coverage(
        self,
        *,
        X_cal,
        y_cal,
        pred_fn,
        model_type: str,
        in_cv: bool,
    ) -> None:
        """
        Unified deep calibration (optional temperature) + coverage threshold fit.

        Contract:
        - X_cal: array-like, first dim is sample/window count
        - y_cal: array-like or None, aligned with X_cal (optional)
        - pred_fn: callable(X)->proba (n, C)
        - model_type: 'cnn'/'lstm'/'transformer'
        - in_cv: True when running Optuna CV folds
        """
        try:
            cfg = getattr(self, "features_config", {}) or {}
            
            if not callable(pred_fn):
                return

            frac = float(cfg.get("deep_calibration_frac", 0.10))
            nmin = int(cfg.get("deep_calibration_min_samples", 500))

            nwin = int(getattr(X_cal, "shape", [0])[0]) if X_cal is not None else 0
            if nwin <= 1:
                return

            # robust clamp
            frac = max(0.01, min(frac, 0.99))
            ncal = max(nmin, int(round(nwin * frac)))
            ncal = min(ncal, nwin - 1) if nwin > 1 else 0
            if ncal < 50:
                return

            X_tail = X_cal[-ncal:]
            y_tail = None
            try:
                if y_cal is not None and len(y_cal) >= ncal:
                    y_tail = np.asarray(y_cal[-ncal:], dtype=int)
            except Exception:
                y_tail = None

            # predict proba on tail
            p_tail = sanitize_proba(pred_fn(X_tail))

            # --- optional: Brier/NLL for selection (only if labels exist)
            try:
                if y_tail is not None:
                    brier, nll = compute_brier_and_nll(p_tail, y_tail.astype(int))
                    self._last_calib_brier = float(brier)
                    self._last_calib_nll   = float(nll)
                    self._last_calib_n     = int(len(y_tail))
                    if bool(cfg.get("print_cv_debug", False)):
                        _ctx = "cv" if in_cv else "eval"
                        print(
                            f"[Calib/deep] model={model_type} ctx={_ctx} "
                            f"brier={float(brier):.6f} nll={float(nll):.6f} n={int(len(y_tail))}"
                        )
            except Exception as _e2:
                if bool(cfg.get("print_cv_debug", False)):
                    print(f"⚠️ [Calib/deep] metrics skipped: {_e2}")

            # --- temperature (keep prior behavior: do NOT do this in CV unless you explicitly enable it)
            use_temp = bool(cfg.get("deep_calibrate", False)) and (
                str(cfg.get("deep_calibration_method", "temperature")).lower() == "temperature"
            )
            allow_temp_in_cv = bool(cfg.get("deep_calibrate_in_cv", False))
            if use_temp and (not in_cv or allow_temp_in_cv):
                if y_tail is not None:
                    try:
                        self._deep_temp_T = float(fit_temperature_from_proba(p_tail, y_tail))
                        p_tail = apply_temperature_to_proba(p_tail, float(self._deep_temp_T))
                        if self._is_debug():
                            _ctx = "cv" if in_cv else "eval"
                            print(
                                f"[Calib] model={model_type} ctx={_ctx} Temp T={float(self._deep_temp_T):.3f} "
                                f"on {int(len(y_tail))} cal rows."
                            )
                    except Exception as _e:
                        if self._is_debug():
                            print(f"⚠️ [Calib] temperature fit skipped: {_e}")

            # --- coverage threshold (requested if gating_mode=coverage OR target_active_rate>0)
            _mode = str(cfg.get("gating_mode", cfg.get("gate_mode", "threshold"))).lower()
            _tar = cfg.get("target_active_rate", None)
            try:
                _tar = float(_tar) if _tar is not None else None
            except Exception:
                _tar = None
            _use_cov_fit = (_mode == "coverage") or (_tar is not None and _tar > 0)

            if _use_cov_fit:
                if in_cv and (not bool(cfg.get("coverage_calibrate_in_cv", True))):
                    return
                tgt = float(_tar) if (_tar is not None and _tar > 0) else float(cfg.get("target_coverage", 0.10))
                thr = float(fit_coverage_threshold_on_calibration(p_tail, tgt))

                # store consistently
                self._deep_coverage_thr = float(thr)
                self._coverage_conf_thr = float(thr)
                try:
                    setattr(self, "_last_cov_cal_rows", int(ncal))
                except Exception:
                    pass
                if in_cv:
                    setattr(self, "_cv_cov_thr_last", float(thr))

                # ctx marker contract
                _ctx = "cv" if in_cv else "eval"
                if not in_cv:
                    try:
                        if bool(getattr(self, "_in_real_sim", False)):
                            mx = int(cfg.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                            _ctx = f"real_m{mx}"
                    except Exception:
                        pass

                print(
                    f"[Calib][Coverage] model={model_type} conf_thr={float(thr):.6f} "
                    f"target_active_rate={float(tgt):.6f} cal_rows={int(ncal)} ctx={_ctx}"
                )

        except Exception as _e:
            print(f"⚠️ [Calib/deep] model={model_type} skipped: {_e}")
            
    def _deep_fit_predict_subprocess(
        self,
        *,
        model_type: str,
        mode: str,  # "seq" or "3d"
        X_train_2d: np.ndarray,
        y_train_1d: np.ndarray,
        X_test_2d: np.ndarray,
        win: int,
        train_stride: int,
        max_train_windows: int,
        batch_size: int,
        epochs: int,
        params: dict,
    ):
        """
        Run deep fit+predict in a fresh subprocess to avoid TF/Keras
        memory accumulation in the main process.
        Returns (proba_test: np.ndarray, coverage_thr: float).
        """
        in_cv = bool(getattr(self, "_in_optuna_cv", False))
        cfg = dict(getattr(self, "features_config", {}) or {})
        allow_in_cv = bool(cfg.get("deep_use_subprocess_in_cv", False)) or \
            str(os.getenv("MLB_DEEP_SUBPROCESS_CV", "0")).lower() in ("1", "true", "yes")
        if in_cv and (not allow_in_cv):
            return None, None

        tmpdir = tempfile.mkdtemp(prefix="mlb_deep_subproc_")
        Xtr_p = os.path.join(tmpdir, "X_train.npy")
        ytr_p = os.path.join(tmpdir, "y_train.npy")
        Xte_p = os.path.join(tmpdir, "X_test.npy")
        proba_out = os.path.join(tmpdir, "proba_test.npy")
        job_json = os.path.join(tmpdir, "job.json")
        out_json = os.path.join(tmpdir, "out.json")

        np.save(Xtr_p, np.asarray(X_train_2d, dtype=np.float32))
        np.save(ytr_p, np.asarray(y_train_1d, dtype=np.int32))
        np.save(Xte_p, np.asarray(X_test_2d, dtype=np.float32))
        
        # Faster inference: allow a separate predict batch size (no effect on outputs)
        try:
            _cfg = dict(getattr(self, "features_config", {}) or {})
            _train_bs = int(batch_size or 128)
            pred_bs = int(_cfg.get("deep_pred_batch_size", max(256, _train_bs * 4)))
            pred_bs_cap = int(_cfg.get("deep_pred_batch_size_cap", 2048))
            pred_bs = int(min(max(16, pred_bs), pred_bs_cap))
        except Exception:
            pred_bs = int(batch_size or 128)

        job = {
            "model_type": str(model_type),
            "mode": str(mode),
            "win": int(win or 0),
            "train_stride": int(train_stride or 1),
            "max_train_windows": int(max_train_windows or 10000),
            "batch_size": int(batch_size or 128),
            "epochs": int(epochs or 20),
            "params": dict(params or {}),
            "features_config": dict(getattr(self, "features_config", {}) or {}),
            "X_train_path": Xtr_p,
            "y_train_path": ytr_p,
            "X_test_path": Xte_p,
            "proba_test_out": proba_out,
            "out_json": out_json,
        }
        with open(job_json, "w", encoding="utf-8") as f:
            json.dump(job, f)

        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_subprocess_worker.py")
        cmd = [sys.executable, worker, "--job_json", job_json]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"⚠️ [DEEP_SUBPROC] failed: {e}")
            return None, None

        try:
            with open(out_json, "r", encoding="utf-8") as f:
                out = json.load(f)
            thr = float(out.get("coverage_thr", np.nan))
            proba = np.load(proba_out)
            return proba, thr
        except Exception as e:
            print(f"⚠️ [DEEP_SUBPROC] load outputs failed: {e}")
            return None, None

    def test_strategy(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        lags,
        confidence_threshold: float = 0.6,
        label_threshold: float = 0.0001,
        persist_results: bool = True,
        eval_context: str | None = None,
    ):
        """
        Train on [train_start, train_end], test on [test_start, test_end], and evaluate trading metrics.

        Returns
        -------
        tuple[float, ...]   # 16 metrics in fixed order
        """

        # CLEANUP: centralized debug/log helpers to prevent print storms (no logic changes)
        _DBG = bool(getattr(self, "debug", False)) or bool(getattr(self, "_debug", False))
        try:
            _DBG = bool(_DBG or (hasattr(self, "_is_debug") and self._is_debug()))
        except Exception:
            pass

        def _dprint(_msg: str):
            # DEBUG: only prints when debug is enabled
            if _DBG:
                print(_msg)

        def _print_once(_key: str, _msg: str, *, debug_only: bool = False):
            # DEBUG: print a message at most once per backtest instance
            if debug_only and not _DBG:
                return
            _flag = f"_ts_once_{_key}"
            if not getattr(self, _flag, False):
                print(_msg)
                setattr(self, _flag, True)

        def _dbg_exc(_label: str, _e: Exception):
            # DEBUG: never swallow exceptions silently
            if _DBG:
                print(f"[test_strategy][{_label}] {type(_e).__name__}: {_e}")
        with self._persist_results_guard(persist_results=persist_results):
        
            # Clear sticky feature-slice cache.
            # In practice cache keys are almost always unique (month-by-month + per-config),
            # so keeping these large frames across calls yields ~0 hits and rising RAM.
            self._clear_feature_cache()
         
            # 🛡️ Set TF runtime knobs *before* importing tensorflow
            # Mirror global intra-trial knob
            _threads = int(os.getenv("BLAS_THREADS_PER_TRIAL", os.getenv("MLB_THREADS", "16")))
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_threads))
            os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(max(2, min(4, _threads // 4))))
            # Keep OpenMP stacks aligned (no down-clamp)
            for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS"):
                os.environ.setdefault(k, str(_threads))

            try:
                # Safe memory growth (no pre-grab of full VRAM)
                for _gpu in tf.config.list_physical_devices("GPU"):
                    try:
                        tf.config.experimental.set_memory_growth(_gpu, True)
                    except Exception:
                        pass
                # Keep TF single-threaded under Optuna/Joblib
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.set_soft_device_placement(True)
            except Exception:
                pass


            # --- FIX: define cfg_f up-front so all branches can read it safely ---
            cfg_f = getattr(self, "features_config", {}) or {}
            # Default to coverage-anchored gating unless explicitly overridden
            if "gating_mode" not in cfg_f:
                cfg_f["gating_mode"] = "coverage"
                self.features_config = cfg_f
                
            # Prevent stale coverage thresholds leaking across runs/models.
            # If coverage intent is enabled but calibration fails, freeze_confidence_threshold()
            # should tripwire (NaN) rather than silently reusing an old threshold.
            try:
                if is_coverage_intent(cfg_f):
                    self._coverage_conf_thr = None
                    if hasattr(self, "_deep_coverage_thr"):
                        delattr(self, "_deep_coverage_thr")
            except Exception:
                pass
        
            
            in_cv = bool(getattr(self, "_in_optuna_cv", False))
            
            # CV memory hygiene: default to no TF cleanup unless a deep model is actually used.
            # Flags are consumed by _persist_results_guard() on exit.
            try:
                self._tf_cleanup_do = False
                self._tf_cleanup_del_model = False
            except Exception:
                pass

        

            # --- Costs knobs (respect constructor lock) ---
            try:
                if not getattr(self, "_trading_costs_locked", False):
                    if "eval_use_trading_costs" in cfg_f:
                        self.trading_costs = bool(cfg_f.get("eval_use_trading_costs", self.trading_costs))
                    elif "trading_costs" in cfg_f:
                        self.trading_costs = bool(cfg_f.get("trading_costs", self.trading_costs))
            except Exception:
                pass

            try:
                if "slippage_factor" in cfg_f:
                    self.slippage_factor = float(cfg_f.get("slippage_factor", self.slippage_factor))
            except Exception:
                pass
            if self._is_debug() and bool(getattr(self, "trading_costs", True)):
                try:
                    _sf = float(getattr(self, "slippage_factor", 0.0) or 0.0)
                    if _sf == 0.0:
                        _print_once("costs_slip_disabled", "[Costs][Warn] trading_costs=True but slippage_factor=0.0. Slippage disabled.")  # CLEANUP
                except Exception:
                    pass
        
            # --- Real-trading guard: if a target_active_rate is set, ensure coverage mode
            # so the existing train-anchored coverage threshold fitting can run.
            # This prevents the system from staying stuck at confidence_threshold=0.8
            # and then getting bumped higher by αβγ, which can easily yield 0 trades.
            try:
                in_real = bool(getattr(self, "_in_real_sim", False))
                gmode = str(cfg_f.get("gating_mode", cfg_f.get("gate_mode", "threshold"))).lower()
                tgt = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.0)) or 0.0)
                if in_real and (not in_cv) and gmode in ("threshold", "", "none") and tgt > 0.0:
                    cfg_f["gating_mode"] = "coverage"
                    self.features_config = cfg_f
                    if self._is_debug():
                        _print_once("gate_auto_cov_real", f"[Gate] Auto-enabled gating_mode='coverage' for real-sim (target_active_rate={tgt:.2f}).")  # CLEANUP
            except Exception:
                pass
        
            self.model = None

            # Clear any sticky feature cache from previous evals
            self._clear_feature_cache()

            # Build / refresh FeatureBank for current data + feature config
            try:
                self._ensure_feature_bank()
            except Exception as e:
                if self._is_debug():
                    _dbg_exc("_ensure_feature_bank", e)  # CLEANUP

            # ---- RAM USAGE: Print at the very start ----
            mem_gb_start = psutil.virtual_memory().used / (1024**3)

            # if self._is_debug():
            #     print(f"[RAM] Start of test_strategy: {mem_gb_start:.2f} GB used")
            _dprint(f"[RAM] Start of test_strategy: {mem_gb_start:.2f} GB used")  # CLEANUP
            # ----------------------------
            # 1) Train/Test slicing (+ NY session on test)
            # ----------------------------
            full_data  = self.data
            train_data = full_data.loc[train_start:train_end]


            # --- warm-up aware test selection (pre-roll before test_start) ---
            true_test_start = pd.to_datetime(test_start)
            test_end        = pd.to_datetime(test_end)
            model_label     = str(self.features_config.get("model_type", self.model_type))
            warmup_need     = int(compute_required_test_warmup_bars({**self.features_config, "model_type": model_label}))
        
            # account for final embargo so pre-roll remains outside test month
            embargo_n = int(self.features_config.get("final_embargo_bars", 0) or 0)
            _total_warmup_need = max(0, warmup_need + embargo_n)

            def _slice_with_warmup(n_extra: int):
                if n_extra <= 0:
                    return full_data.loc[true_test_start:test_end]
                idx_before = full_data.index[full_data.index < true_test_start]
                if len(idx_before) == 0:
                    return full_data.loc[true_test_start:test_end]
                start_pos = max(0, len(idx_before) - n_extra)
                warmup_start = idx_before[start_pos]
                return full_data.loc[warmup_start:test_end]


            # initial pre-roll (build test_data before any filtering/embargo)
            test_data = _slice_with_warmup(_total_warmup_need)

            sess_mode = str(self.features_config.get("session_filter_mode", "both")).lower()

            if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                try:
                    full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                    _ny_times = full_idx.tz_convert("America/New_York")
                    self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
                except Exception as _e:
                    _dbg_exc("ny_mask", _e)  # CLEANUP
                    self._ny_mask = pd.Series(True, index=self.data.index)

            # NEW semantics:
            # - "both":        filter train + test
            # - "test_only":   filter test only
            # - "train_only":  filter train only
            if sess_mode in ("test_only", "both"):
                test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
            if sess_mode in ("train_only", "both"):
                train_data = train_data.loc[self._ny_mask.reindex(train_data.index, fill_value=False)]

        
            # ensure we still have enough warm-up after session filter
            if warmup_need > 0 and len(test_data) > 0:
                have = int((test_data.index < true_test_start).sum())
                if have < _total_warmup_need:
                    # fetch more history and reapply the session filter
                    need_more = _total_warmup_need - have
                    test_data = _slice_with_warmup(_total_warmup_need + need_more)
                    if sess_mode in ("test_only", "both"):
                        test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
                    
            # Final embargo to avoid bleed — but NEVER eat CV mini-fold heads
            try:
                embargo_n = int(self.features_config.get("final_embargo_bars", 0))
                if bool(getattr(self, "_in_optuna_cv", False)):
                    embargo_n = 0  # ⬅️ disable head-drop during CV mini-folds
                if embargo_n > 0 and len(test_data) > embargo_n:
                    test_data = test_data.iloc[embargo_n:].copy()
                    if self._is_debug():
                        print(f"[Embargo] Dropped first {embargo_n} bars from TEST (non-CV only).")

            except Exception as e:
                _dbg_exc("final_embargo_bars", e)  # CLEANUP

            # Evaluation anchor:
            # In real trading we start *after* embargo; in CV we start EXACTLY at the fold start.
            use_strict_day1 = bool(self.features_config.get("enforce_day1_start", True))

            if getattr(self, "_in_real_sim", False):
                use_strict_day1 = True

            first_eval_ts = (
                pd.to_datetime(true_test_start)
                if bool(getattr(self, "_in_optuna_cv", False))
                else (
                    enforce_day1_eval_anchor(test_data.index, true_test_start)
                    if use_strict_day1 else
                    first_tradable_test_bar(test_data.index, true_test_start)
                )
            )

        
            if bool(getattr(self, "_in_optuna_cv", False)):
                if self._is_debug():
                    print(f"[CV/CLASSICAL] Eval anchor forced to fold start: {first_eval_ts} | test_len={len(test_data)} | warmup_need={_total_warmup_need}")

            if first_eval_ts is None:
                print("❌ No tradable bar found in test window.")

                 # IMPORTANT: never persist heavy frames during Optuna CV.
                if in_cv:
                    self.results = None
                    self.results_full = None
                    self._cv_last_eval_df = None
                else:
                    self.results = pd.DataFrame()
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_tradable_test_bar")
            self._expected_eval_start = first_eval_ts
        
            # ------------------------------------------------------------------
            # Patch D: Eligibility diagnostics (CV vs real comparability; no logic change)
            # Records: eligible_bars, embargo_dropped, warmup_need, eval_anchor_ts, etc.
            # ------------------------------------------------------------------
            try:
                _in_cv   = bool(getattr(self, "_in_optuna_cv", False))
                _in_real = bool(getattr(self, "_in_real_sim", False))

                # Month-only slice (no warmup) for “how many bars exist this month?”
                _month_raw = full_data.loc[true_test_start:test_end]
                _raw_n = int(len(_month_raw))
                # Apply the same NY session mask to the month slice (test-side semantics)
                _month_sess = _month_raw
                if sess_mode in ("test_only", "both"):
                    _month_sess = _month_raw.loc[
                        self._ny_mask.reindex(_month_raw.index, fill_value=False)
                    ]
                _sess_n = int(len(_month_sess))
                _sess_drop = int(max(0, _raw_n - _sess_n))

                # Embargo is disabled for CV heads (your existing behavior)
                _embargo_used = int(self.features_config.get("final_embargo_bars", 0) or 0)
                if _in_cv:
                    _embargo_used = 0
                _emb_drop = int(min(_embargo_used, _sess_n))

                _after_emb = _month_sess.iloc[_emb_drop:] if _emb_drop > 0 else _month_sess
                _post_emb_n = int(len(_after_emb))

                # Eligibility after the evaluation anchor (day-1 anchor / fold start)
                _anchor_ts = first_eval_ts
                if _anchor_ts is not None and _post_emb_n > 0:
                    _eligible_n = int(len(_after_emb.loc[_anchor_ts:]))
                else:
                    _eligible_n = int(_post_emb_n)
                _anchor_drop = int(max(0, _post_emb_n - _eligible_n))

                self._last_eligibility_diag = {
                    "in_cv": _in_cv,
                    "in_real": _in_real,
                    "sess_mode": sess_mode,
                    "raw_month_bars": _raw_n,
                    "session_month_bars": _sess_n,
                    "session_dropped": _sess_drop,
                    "final_embargo_bars_used": _embargo_used,
                    "embargo_dropped": _emb_drop,
                    "post_embargo_bars": _post_emb_n,
                    # Additive denominator for GateSummary: total bars on eval grid
                    # (after session filter + final embargo, before anchor selection).
                    "bars_total": _post_emb_n,
                    "warmup_need": int(warmup_need),
                    "warmup_plus_embargo_need": int(_total_warmup_need),
                    "eval_anchor_ts": str(_anchor_ts) if _anchor_ts is not None else None,
                    "eligible_bars": _eligible_n,
                    "anchor_dropped": _anchor_drop,
                }
            except Exception:
                self._last_eligibility_diag = {}

            cfg = self.apply_feature_defaults()
            lag_depth    = cfg.get("lag_depth", 1)
            roll_windows = cfg.get("roll_windows", [5])
            lags_eff = int(cfg.get("lags_range", cfg.get("lags", lags)))
            if self._is_debug():
                print(f"[TEST] effective_lags={lags_eff} (cfg-precedence)")

            # === Feature engineering (TRAIN) ===
            train_data, features = self.prepare_features(
                train_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
            )

            # if self._is_debug():
            #     print("Train data length after prepare_features:", len(train_data))
            #     print("Train data columns after prepare_features:", train_data.columns.tolist())

            if len(train_data) < 100:
                raise ValueError("Training data too short.")
            train_data = train_data.loc[:, ~train_data.columns.duplicated()]

            # --- robust feature prefilter on TRAIN only (near-constant → corr → MI) ---
            keep = None  # ensure defined even if an exception occurs
            if bool((self.features_config or {}).get("use_prefilter", True)) and features:
                try:
                    # # Build a provisional label on the training slice (no leakage into TEST)
                    # ret_fwd_pre = train_data["returns"].shift(-1)
                    # thr = float((self.features_config or {}).get("label_threshold", 1e-4))

                    # # label_with_neutral is a class method; align to train_data index
                    # y_pre = pd.Series(
                    #     self.label_with_neutral(ret_fwd_pre, threshold=thr),
                    #     index=train_data.index
                    # ).astype(int)
                
                    # Build a provisional label on the training slice (no leakage into TEST)
                    cfg = getattr(self, "features_config", {}) or {}

                    tb_on = bool(cfg.get("use_triple_barrier", False))
                    if tb_on:
                        y_pre = triple_barrier_labels(
                            close=train_data["price"],
                            pt_mult=float(cfg.get("tb_pt_mult", 1.5)),
                            sl_mult=float(cfg.get("tb_sl_mult", 1.0)),
                            max_holding=int(cfg.get("tb_max_holding", 48)),
                            neutral_zone=float(cfg.get("tb_neutral_zone", 0.0)),
                            neutral_zone_is_sigma=bool(cfg.get("tb_neutral_zone_is_sigma", False)),
                        ).astype(int)
                    else:
                        ret_fwd_pre = train_data["returns"].shift(-1)
                        thr = float(cfg.get("label_threshold", 1e-4))
                        y_pre = pd.Series(self.label_with_neutral(ret_fwd_pre, threshold=thr),
                                        index=train_data.index).astype(int)


                    # Explicit index intersection for robustness
                    common_idx = train_data.index.intersection(y_pre.index)
                    X_pref = train_data.loc[common_idx, features]
                    y_pref = y_pre.loc[common_idx]

                    # 3-stage prefilter (near-constant → high-corr collapse → MI top-K)
                    keep = prefilter_features_train(
                        X=X_pref,
                        y=y_pref,
                        cfg=(self.features_config or {}),
                    )
                except Exception as e:
                    print(f"⚠️ Prefilter skipped (non-fatal): {e}")
            else:
                if self._is_debug():
                    print("[Prefilter] disabled via config or empty feature list.")

            # Apply the reduced feature set only if it truly shrank
            if keep and len(keep) < len(features):
                if self._is_debug():
                    print(f"[Prefilter] Kept {len(keep)}/{len(features)} features.")
                features = [f for f in features if f in set(keep)]
            else:
                if self._is_debug():
                    print("[Prefilter] No change to feature set.")

            # Impute on TRAIN, then apply to both TRAIN and TEST
            imputer = SimpleImputer(strategy="mean")
            train_imputed = pd.DataFrame(
                imputer.fit_transform(train_data[features]),
                index=train_data.index, columns=features
            )

            # (Test imputation happens later once test_data is prepared)
            train_data_scaled, means, stds = self.scale_features(
                pd.concat([train_data.drop(columns=features), train_imputed], axis=1),
                features, log_id=f"test_train_{train_start.date()}_{train_end.date()}"
            )

            # Drop rows with remaining NaNs (train)
            orig_train_len = len(train_data_scaled)
            if train_data_scaled[features].isna().any().any():
                n_dropped = train_data_scaled[features].isna().any(axis=1).sum()
                print(
                    f"⚠️ Dropping {n_dropped} ({n_dropped/orig_train_len:.2%}) train rows with NaN after impute+scale (test_strategy).",
                    train_data_scaled[features].isna().sum()[train_data_scaled[features].isna().sum() > 0],
                )
                train_data_scaled = train_data_scaled[~train_data_scaled[features].isna().any(axis=1)]
                if len(train_data_scaled) == 0:
                    print("⚠️ All train rows dropped after impute+scale. Skipping fold.")
                    return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:train_all_rows_dropped")

                        # -------------------------------------------------------------
            # Labels (train) — unified regime
            # If TripleBarrier is enabled, all supervised families (classical + deep)
            # train on TB event labels. Otherwise fall back to next-bar (T+1) labels.
            # -------------------------------------------------------------
            cfg_lbl = getattr(self, "features_config", {}) or {}
            tb_on_lbl = bool(cfg_lbl.get("use_triple_barrier", False))

            def _resolve_price_col(_df: "pd.DataFrame") -> str | None:
                """Resolve a close-like series for TB labels without assuming column names."""
                try:
                    if _df is None or len(_df) == 0:
                        return None
                    if "price" in _df.columns:
                        return "price"
                    if "mid_close" in _df.columns:
                        return "mid_close"
                    if "close" in _df.columns:
                        return "close"
                    if {"ask_close", "bid_close"}.issubset(_df.columns):
                        _df["__mid_close__"] = (_df["ask_close"] + _df["bid_close"]) / 2.0
                        return "__mid_close__"
                except Exception:
                    return None
                return None

            _pcol_tr = _resolve_price_col(train_data_scaled)
            if tb_on_lbl and _pcol_tr is None:
                if self._is_debug():
                    print("⚠️ TripleBarrier enabled but no price column; falling back to return-based labels (train).")
                tb_on_lbl = False

            if tb_on_lbl:
                y_train = triple_barrier_labels(
                    close=train_data_scaled[_pcol_tr].astype(float),
                    pt_mult=float(cfg_lbl.get("tb_pt_mult", 1.5)),
                    sl_mult=float(cfg_lbl.get("tb_sl_mult", 1.0)),
                    max_holding=int(cfg_lbl.get("tb_max_holding", 48)),
                    neutral_zone=float(cfg_lbl.get("tb_neutral_zone", 0.0)),
                    neutral_zone_is_sigma=bool(cfg_lbl.get("tb_neutral_zone_is_sigma", False)),
                ).astype(int)
            else:
                # Next-bar returns (T+1)
                _returns_fwd = train_data_scaled["returns"].shift(-1)
                # drop last row with NaN forward return to keep X and y aligned
                train_data_scaled = train_data_scaled.loc[_returns_fwd.notna()].copy()
                y_train = self.label_with_neutral(
                    _returns_fwd.loc[train_data_scaled.index],
                    threshold=label_threshold,
                )


            # Features (aligned to y_train)
            X_train = train_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
            if y_train is None or len(y_train) == 0:
                print("⚠️ No labels generated (all NaN or below threshold). Skipping fold.")
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_labels")
            y_train = y_train.astype(int)
            if self._is_debug():
                print("Label distribution in training set:", pd.Series(y_train).value_counts())


            # Make labels available to windowing helper (transformer/LSTM/CNN)
            train_data_scaled["label"] = y_train

            # -----------------------------------------------------------------
            # Train-anchored high-volatility threshold for cost regime switching
            # -----------------------------------------------------------------
            high_vol_thr_train = None
            def _push_thr_to_attrs(_df, thr):
                try:
                    if _df is None or len(_df) == 0 or thr is None:
                        return
                    fc = dict(_df.attrs.get("features_config", {}) or {})
                    fc["high_vol_thr"] = float(thr)
                    _df.attrs["features_config"] = fc
                except Exception:
                    pass
            
            try:
                from utilsNoWFO import realized_vol as _rv_fn
                _vol_w = int(cfg_f.get("vol_window_bars", 48))
                _qhi   = float(cfg_f.get("high_vol_q", 0.80))
                _rv  = _rv_fn(train_data_scaled["returns"].astype(float), window=_vol_w)
                _thr = float(_rv.quantile(_qhi))
                if np.isfinite(_thr):
                    high_vol_thr_train = _thr
                    if self._is_debug():
                        print(f"[Costs] Train-anchored high_vol_thr={high_vol_thr_train:.8f} (q={_qhi:.2f}, vol_w={_vol_w})")

                    # 1) config path (used by _ensure_cost_columns when config is passed)
                    try:
                        if not isinstance(config, dict):
                            config = {}
                    except Exception:
                        config = {}
                        
                    # Cache on instance for downstream consumers (e.g., Top-N consensus)
                    # when config/attrs propagation is temporarily missing.
                    try:
                        self._last_high_vol_thr_train = float(high_vol_thr_train)
                    except Exception:
                        pass

                    # Also mirror into DataFrame attrs (best-effort).
                    try:
                        _push_thr_to_attrs(train_data_scaled, high_vol_thr_train)
                    except Exception:
                        pass

            except Exception as _e:
                if self._is_debug():
                    print(f"[Costs] Failed to compute train-anchored high_vol_thr: {_e}")
                        

            # Guard: minimum per-class and at least 2 classes
            # Require at least 2 classes and a minimum count per class.
            # In Optuna CV, optionally prune immediately when labels collapse.
            _cv_cfg = getattr(self, "_cv_config_current", None) or getattr(self, "cv_config", None) or {}
            try:
                MIN_CLASS_SAMPLES = int(_cv_cfg.get("cv_min_class_samples", 5))
            except Exception:
                MIN_CLASS_SAMPLES = 5
            unique, counts = np.unique(y_train, return_counts=True)
            class_counts = dict(zip(unique, counts))
            too_few = [cls for cls, count in class_counts.items() if count < MIN_CLASS_SAMPLES]
            if len(too_few) > 0 or len(class_counts) < 2:
                msg = (f"⚠️ Skipping fold: Not enough samples for classes {too_few} "
                       f"or only one class present: {class_counts}")
                print(msg)

                # Early prune (CV only): don't waste compute on a broken label regime
                _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                _prune_lbl  = bool(_cv_cfg.get("cv_prune_on_label_collapse", True))
                if _in_cv_mode and _prune_lbl:
                    try:
                        import optuna as _opt
                        raise _opt.TrialPruned(msg)
                    except Exception:
                        # If Optuna isn't available for some reason, fall back to invalid metrics.
                        pass

                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:insufficient_class_support")

            # === Feature engineering (TEST) ===
            test_data, _ = self.prepare_features(
                test_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
            )

            if test_data is None or test_data.empty:
                print(f"[ERROR] test_data is empty after prepare_features for test period {test_start} - {test_end}")
                # IMPORTANT: never persist heavy frames during Optuna CV.
                if in_cv:
                     self.results = None
                     self.results_full = None
                     self._cv_last_eval_df = None
                else:
                     self.results = pd.DataFrame()
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:empty_test_data")

            test_data_raw_with_extras = test_data.copy()
            if self._is_debug():
                print("Test data length after prepare_features:", len(test_data))

            if len(test_data) < 20:
                raise ValueError("Test data too short.")
            test_data = test_data.loc[:, ~test_data.columns.duplicated()]

            # Align + scale test with train stats
            test_imputed = pd.DataFrame(
                imputer.transform(test_data[features]),
                index=test_data.index, columns=features
            )
            test_data_scaled, _, _ = self.scale_features(
                pd.concat([test_data.drop(columns=features), test_imputed], axis=1),
                features, means, stds, log_id=f"test_eval_{test_start.date()}_{test_end.date()}"
            )

            # Force same columns/order as training
            test_data_scaled = test_data_scaled.reindex(columns=features)

            # Drop rows with NaNs (test)
            orig_test_len = len(test_data_scaled)
            if test_data_scaled[features].isna().any().any():
                n_dropped = test_data_scaled[features].isna().any(axis=1).sum()
                print(
                    f"⚠️ Dropping {n_dropped} ({n_dropped/orig_test_len:.2%}) test rows with NaN after impute+scale (test_strategy).",
                    test_data_scaled[features].isna().sum()[test_data_scaled[features].isna().sum() > 0],
                )
                test_data_scaled = test_data_scaled[~test_data_scaled[features].isna().any(axis=1)]

            test_data_scaled = test_data_scaled.copy().reindex(columns=features)
            X_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)

            if len(X_test) == 0:
                print("❌ [ABORT] Empty X_test after scaling/alignment.")
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:empty_X_test")
        
            # evaluation starts at the first tradable bar of the month (after session filter)
            eval_anchor_ts = getattr(self, "_expected_eval_start", pd.to_datetime(test_start))
            eval_mask = test_data_scaled.index >= eval_anchor_ts
        
            # ------------------------------------------------------------------
            # Patch D (final write): eligibility diagnostics on the POST-FEATURE grid
            # Grid must match the exact dataframe used for gating/evaluation.
            # ------------------------------------------------------------------
            try:
                eval_anchor_ts = getattr(self, "_expected_eval_start", pd.to_datetime(test_start))
                gating_df = test_data_scaled  # exact post-feature, post-dropna, post-scale grid
                bars_total = int(len(gating_df))

                # Eligible bars are those in the evaluation window (>= anchor)
                eligible_mask_pf = gating_df.index >= eval_anchor_ts
                eligible_bars_pf = int(np.sum(eligible_mask_pf))

                # Optional split: bars between true month start and anchor (on the same grid)
                try:
                    _tts = pd.to_datetime(true_test_start)
                except Exception:
                    _tts = pd.to_datetime(test_start)
                anchor_dropped_pf = int(np.sum((gating_df.index >= _tts) & (gating_df.index < eval_anchor_ts)))

                # Warmup bars are everything before true month start (on the same grid)
                warmup_dropped_pf = int(bars_total - eligible_bars_pf - anchor_dropped_pf)
                if warmup_dropped_pf < 0:
                        warmup_dropped_pf = 0

                # Update (not replace) so pre-feature month diagnostics remain available
                _diag_pf = dict(getattr(self, "_last_eligibility_diag", {}) or {})
                _diag_pf.update({
                    "post_feature_grid": True,
                    "bars_total": bars_total,
                    "gating_df_len": bars_total,
                    "eligible_bars": eligible_bars_pf,
                    "warmup_dropped": warmup_dropped_pf,
                    "anchor_dropped": anchor_dropped_pf,
                    "eval_anchor_ts": str(eval_anchor_ts) if eval_anchor_ts is not None else None,
                    "true_test_start": str(_tts) if _tts is not None else None,
                })

                # Fail-loud invariants (debug-only): prevent mixed denominators in GateSummary
                if self._is_debug():
                    _lhs = int(_diag_pf.get("warmup_dropped", 0) or 0) + int(_diag_pf.get("anchor_dropped", 0) or 0) + int(_diag_pf.get("eligible_bars", 0) or 0)
                    if bars_total and _lhs and _lhs != bars_total:
                        print(
                            f"⚠️ [EligDiag] Invariant mismatch on post-feature grid: "
                            f"warm({warmup_dropped_pf})+anch({anchor_dropped_pf})+elig({eligible_bars_pf})={_lhs} vs bars_total={bars_total}"
                        )

                self._last_eligibility_diag = _diag_pf
                # ------------------------------------------------------------
                # Explicit eligibility audit log (no behavior change)
                # Confirms: post-feature denominator, warmup dropped, eval anchor.
                # ------------------------------------------------------------
                try:
                    _ctx = "eval"
                    if bool(getattr(self, "_in_optuna_cv", False)):
                        _ctx = "cv"
                    elif bool(getattr(self, "_in_real_sim", False)):
                        _mx = getattr(self, "_rt_month_idx", None)
                        _ctx = f"real_m{int(_mx)}" if _mx is not None else "real"

                    print(
                        f"[Eligibility] post_feature_bars_total={int(bars_total)} "
                        f"eligible={int(eligible_bars_pf)} "
                        f"warmup_dropped={int(warmup_dropped_pf)} "
                        f"anchor={str(eval_anchor_ts)} "
                        f"ctx={_ctx}"
                    )
                except Exception:
                    pass
            except Exception as _e:
                if self._is_debug():
                    print(f"⚠️ [EligDiag] Post-feature eligibility diag failed (non-fatal): {_e}")

            # Patch D (eligibility diagnostics, post-feature):
            # Recompute on the SAME bar grid used for gating/eval (test_data_scaled),
            # so GateSummary doesn't mix pre-feature monthly counts with post-feature counts.
            try:
                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                _idx = test_data_scaled.index
                _month_start_ts = pd.to_datetime(true_test_start)

                _bars_total = int(len(_idx))
                _warm_drop = int(np.sum(_idx < _month_start_ts))
            
                _anchor_ts = pd.to_datetime(eval_anchor_ts) if eval_anchor_ts is not None else None
                _eligible = int(np.sum(_idx >= _anchor_ts)) if _anchor_ts is not None else _bars_total
                _anch_drop = int(np.sum((_idx >= _month_start_ts) & (_idx < _anchor_ts))) if _anchor_ts is not None else 0
            
                _diag.update(
                    {
                        "bars_total": _bars_total,
                        "eligible_bars": _eligible,
                        "warmup_dropped": _warm_drop,
                        "anchor_dropped": _anch_drop,
                        "eval_anchor_ts": str(_anchor_ts) if _anchor_ts is not None else _diag.get("eval_anchor_ts"),
                    }
                )
                # Back-compat: keep this aligned with gating grid so denom matches what we trade/gate on
                _diag["raw_month_bars"] = _bars_total
                self._last_eligibility_diag = _diag
            except Exception:
                pass


            # ----------------------------
            # 3) Build & fit model
            # ----------------------------
            deep_models = ["cnn", "lstm", "transformer"]
            ensemble_models = [
                "ensemble_cnn_lstm_xgboost",
                "ensemble_adaptive_regime",
            ]

            params = self.features_config.copy()
            model_type = params.pop("model_type", self.model_type)
            # Keep internal tag in sync so gating/threshold logic sees the right model
            self.model_type = model_type
            self._maybe_configure_tf_runtime_once(model_type)
            
            # Optuna CV: release TF graph/session + model refs after this fold.
            # (Runs via _persist_results_guard() even on early returns.)
            try:
                if bool(in_cv) and str(model_type) in set(deep_models):
                    self._tf_cleanup_do = True
                    self._tf_cleanup_del_model = True
            except Exception:
                pass
            params.pop("input_shape", None)

            mem_gb_pre_fit = psutil.virtual_memory().used / (1024**3)
            if self._is_debug():
                print(f"[RAM] Before model fit: {mem_gb_pre_fit:.2f} GB used")

            class _TimeLimit(Callback):
                """Hard wall-clock cap for deep trainings."""
                def __init__(self, seconds):
                    super().__init__()
                    self.seconds = float(seconds) if seconds is not None else None
                    self._start = None
                def on_train_begin(self, logs=None):
                    if self.seconds is not None:
                        self._start = time.time()
                def on_batch_end(self, batch, logs=None):
                    if self.seconds is not None and (time.time() - self._start) > self.seconds:
                        self.model.stop_training = True

            def _maybe_mixed_precision(enable: bool, tag: str):
                if not enable:
                    return
                try:
                    mixed_precision.set_global_policy("mixed_float16")
                    print(f"[{tag}] Mixed precision enabled.")
                except Exception:
                    pass

            def _make_windows_fast(X2d: np.ndarray, win: int, stride: int = 1, labels_1d=None):
                """
                Vectorized sliding windows.
                X2d: (n, f) float32  →  (m, win, f), y_seq (m,) if labels_1d provided, idx_end (m,)
                """
                n = X2d.shape[0]
                if n < win:
                    return None, None, None
                Xv = sliding_window_view(X2d, window_shape=win, axis=0)  # (n-win+1, win, f)
                if stride > 1:
                    Xv = Xv[::stride]
                m = Xv.shape[0]
                idx_end = np.arange(win - 1, win - 1 + m * stride, stride, dtype=int)
                yv = labels_1d[idx_end] if labels_1d is not None else None
                return Xv, yv, idx_end
            
            def _start_idx_for_last_strided_windows(n_rows: int, win: int, stride: int, max_windows: int) -> int:
                """
                Compute the *exact* starting row index so that:
                  make_windows(stride) then take [-max_windows:]
                is identical to:
                  slice df[start_idx:] then make_windows(stride) (and optionally still slice)
                This avoids building huge intermediate arrays during CV/HPO.
                """
                try:
                    n_rows = int(n_rows)
                    win = max(1, int(win))
                    stride = max(1, int(stride))
                    max_windows = int(max_windows) if max_windows is not None else 0
                    if max_windows <= 0:
                        return 0
                    total = n_rows - win + 1  # number of raw windows before stride
                    if total <= 0:
                        return 0
                    m = (total + stride - 1) // stride  # number of windows after stride
                    if m <= max_windows:
                        return 0
                    k0 = m - max_windows
                    return int(k0 * stride)
                except Exception:
                    return 0

            def _start_idx_for_last_stride_rows(n_rows: int, stride: int, max_rows: int) -> int:
                """
                For 3D-feed paths:
                  X[::stride] then take [-max_rows:]
                -> compute start row so slicing first preserves exact same sampled rows.
                """
                try:
                    n_rows = int(n_rows)
                    stride = max(1, int(stride))
                    max_rows = int(max_rows) if max_rows is not None else 0
                    if max_rows <= 0:
                        return 0
                    m = (n_rows + stride - 1) // stride
                    if m <= max_rows:
                        return 0
                    k0 = m - max_rows
                    return int(k0 * stride)
                except Exception:
                    return 0

        
            # ---- branch: deep / ensemble / classical ----
            if model_type in deep_models:

                if model_type == "transformer":
                    # knobs
                    train_stride   = int(params.get("transformer_train_stride", 2))
                    batch_size     = int(params.get("transformer_batch_size", 128))
                    epochs         = int(params.get("transformer_epochs", 20))
                    use_mixed_prec = bool(params.get("transformer_mixed_precision", True)) and bool(tf.config.list_physical_devices("GPU"))

                    in_cv = bool(getattr(self, "_in_optuna_cv", False))
                    if in_cv:
                        # Multi-fidelity: coarser stride + fewer windows during CV.
                        # Wu et al. (2020) and Won et al. (2025) use sample/epoch
                        # count as natural fidelities in deep HPO.
                        cfg = getattr(self, "features_config", {}) or {}
                        stride_min = int(cfg.get("transformer_cv_train_stride", train_stride))
                        train_stride = max(train_stride, stride_min)  # higher stride = fewer windows (cheaper)
                        
                    _maybe_mixed_precision(use_mixed_prec, "Transformer")


                    win = max(2, int(lags_eff))
                    
                    # Cap training windows if requested (compute BEFORE materializing arrays)
                    max_train_windows = int(params.get("deep_max_train_windows", 10000))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        max_cap = int(cfg.get("transformer_cv_max_train_windows", max_train_windows))
                        max_train_windows = min(max_train_windows, max_cap)
                        
                    use_subproc = bool((getattr(self, "features_config", {}) or {}).get("deep_use_subprocess", False)) \
                        or str(os.getenv("MLB_DEEP_SUBPROCESS", "0")).lower() in ("1", "true", "yes")
                    if use_subproc and (not in_cv):
                        X2d_train = train_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                        y1d_train = train_data_scaled["label"].to_numpy(dtype=np.int32, copy=False)
                        X2d_test  = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)

                        proba_sub, thr_sub = self._deep_fit_predict_subprocess(
                            model_type="transformer",
                            mode="seq",
                            X_train_2d=X2d_train,
                            y_train_1d=y1d_train,
                            X_test_2d=X2d_test,
                            win=win,
                            train_stride=train_stride,
                            max_train_windows=max_train_windows,
                            batch_size=batch_size,
                            epochs=epochs,
                            params=dict(params),
                        )
                        if proba_sub is not None:
                            self._deep_subproc_proba = proba_sub
                            if thr_sub is not None and np.isfinite(thr_sub):
                                self._coverage_conf_thr = float(thr_sub)
                                self._deep_coverage_thr = float(thr_sub)
                            # Skip in-proc TF build/fit entirely
                            self.model = None
                            # jump to prediction section
                            goto_predict = True
                        else:
                            goto_predict = False
                    else:
                        goto_predict = False

                    if not goto_predict:
                        # Pre-slice DF so that the resulting (strided) windows are identical to the
                        # previous approach (build-all → stride → take last max_train_windows).
                        _si = _start_idx_for_last_strided_windows(
                            len(train_data_scaled), win, train_stride, max_train_windows
                        )
                        _df_tr = train_data_scaled.iloc[_si:] if _si > 0 else train_data_scaled
                        X2d_train = _df_tr[features].to_numpy(dtype=np.float32, copy=False)
                        y1d_train = _df_tr["label"].to_numpy(dtype=np.int32, copy=False)


                        X_seq_train, y_seq_train, _ = _make_windows_fast(
                            X2d_train, win=win, stride=max(1, train_stride), labels_1d=y1d_train
                        )
                        if X_seq_train is None or len(X_seq_train) == 0:
                            print("❌ [ABORT] Empty training sequences for transformer.")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_empty_train_seq")

                        if X_seq_train.shape[0] > max_train_windows:
                            X_seq_train = X_seq_train[-max_train_windows:]
                            y_seq_train = y_seq_train[-max_train_windows:]

                        params["input_shape"] = (X_seq_train.shape[1], X_seq_train.shape[2])
                        self.model = self.get_model(model_type, **params)
                        # Tag for per-model CV caps (used in _fit_keras_with_cv_controls).
                        setattr(self.model, "_mlb_model_tag", "transformer")

                        # callbacks
                        cb = getattr(self.model, "early_stop_callback", None)


                        print(f"[DEEP] model={model_type} | seq_windows={X_seq_train.shape[0]} "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_seq_train, y_seq_train,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )
                    
                        # Unified deep calibration + coverage threshold (works in CV too)
                        X_cal = X_seq_train
                        y_cal = y_seq_train

                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)
                        if X_cal is not None and callable(pred_fn):
                            self._fit_deep_calibration_and_coverage(
                                X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                                model_type=model_type, in_cv=in_cv
                            )
                            
                        # IMPORTANT: X_seq_train is a view over X2d_train (sliding_window_view),
                        # so free the base arrays too once training+calibration is done.
                        try:
                            del X2d_train, y1d_train, _df_tr
                        except Exception:
                            pass

                        del X_seq_train, y_seq_train, X_cal, y_cal, pred_fn
                        _gc.collect()

                elif model_type == "lstm":
                    lstm_use_seq   = bool(params.get("lstm_use_seq_windows", True))
                    train_stride   = int(params.get("lstm_train_stride", 2))
                    batch_size     = int(params.get("lstm_batch_size", 128))
                    epochs         = int(params.get("lstm_epochs", 20))
                    use_mixed_prec = bool(params.get("lstm_mixed_precision", True)) and bool(
                        tf.config.list_physical_devices("GPU")
                    )

                    in_cv = bool(getattr(self, "_in_optuna_cv", False))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        stride_min = int(cfg.get("lstm_cv_train_stride", train_stride))
                        train_stride = max(train_stride, stride_min)


                    _maybe_mixed_precision(use_mixed_prec, "LSTM")

                    # ─────────────────────────────────────────────
                    # A) LSTM with sequence windows (X_seq_train)
                    # ─────────────────────────────────────────────
                    if lstm_use_seq:
                        win = max(2, int(lags_eff))

                        max_train_windows = int(params.get("deep_max_train_windows", 10000))
                        if in_cv:
                            cfg = getattr(self, "features_config", {}) or {}
                            max_cap = int(cfg.get("lstm_cv_max_train_windows", max_train_windows))
                            max_train_windows = min(max_train_windows, max_cap)

                        _si = _start_idx_for_last_strided_windows(len(train_data_scaled), win, train_stride, max_train_windows)
                        _df_tr = train_data_scaled.iloc[_si:] if _si > 0 else train_data_scaled
                        X2d = _df_tr[features].to_numpy(dtype=np.float32, copy=False)
                        y1d = _df_tr["label"].to_numpy(dtype=np.int32, copy=False)

                        X_seq_train, y_seq_train, _ = _make_windows_fast(
                            X2d,
                            win=win,
                            stride=max(1, train_stride),
                            labels_1d=y1d,
                        )
                        if X_seq_train is None or len(X_seq_train) == 0:
                            print("❌ [ABORT] Empty training sequences for LSTM (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_empty_train_seq")
                    

                        if X_seq_train.shape[0] > max_train_windows:
                            X_seq_train = X_seq_train[-max_train_windows:]
                            y_seq_train = y_seq_train[-max_train_windows:]

                        params.setdefault("lstm_use_early_stopping", True)
                        params["input_shape"] = (X_seq_train.shape[1], X_seq_train.shape[2])

                        self.model = self.get_model(model_type, **params)
                        # Tag for per-model CV caps.
                        setattr(self.model, "_mlb_model_tag", "lstm")
                        cb = getattr(self.model, "early_stop_callback", None)


                        print(
                            f"[DEEP] model={model_type} | seq_windows={X_seq_train.shape[0]} "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}"
                        )

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_seq_train,
                            y_seq_train,
                            X_val=None,
                            y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        X_cal = X_seq_train
                        y_cal = y_seq_train

                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)
                        if X_cal is not None and callable(pred_fn):
                            self._fit_deep_calibration_and_coverage(
                                X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                                model_type=model_type, in_cv=in_cv
                            )
                            
                        # cleanup (seq only)
                        # IMPORTANT: X_seq_train is a view over X2d (sliding_window_view).
                        # Free base arrays/DF slice too to reduce RSS growth.
                        try:
                            del X2d, y1d, _df_tr
                        except Exception:
                            pass
                        
                        del X_seq_train, y_seq_train, X_cal, y_cal, pred_fn
                        _gc.collect()

                    # ─────────────────────────────────────────────
                    # B) LSTM with simple 3D feed (no seq windows)
                    # ─────────────────────────────────────────────
                    else:
                        params["input_shape"] = (X_train.shape[1], 1)
                        self.model = self.get_model(model_type, **params)
                        setattr(self.model, "_mlb_model_tag", "lstm")

                        max_train_windows = int(params.get("deep_max_train_windows", 10000))

                        # Preserve exact rows vs old path: (X_train[::stride] then tail-slice)
                        _si = _start_idx_for_last_stride_rows(X_train.shape[0], train_stride, max_train_windows)
                        if _si > 0:
                            X_tr2 = X_train[_si:]
                            y_tr2 = y_train[_si:]
                        else:
                            X_tr2 = X_train
                            y_tr2 = y_train

                        # (N, features, 1)
                        X_train_3d = X_tr2.astype(np.float32).reshape((X_tr2.shape[0], X_tr2.shape[1], 1))

                        # Optional stride-based downsampling
                        if train_stride > 1:
                            X_train_3d = X_train_3d[::train_stride]
                            y_train_eff = y_tr2[::train_stride]
                        else:
                            y_train_eff = y_tr2

                        # Tail cap (apply to both)
                        if X_train_3d.shape[0] > max_train_windows:
                            X_train_3d  = X_train_3d[-max_train_windows:]
                            y_train_eff = y_train_eff[-max_train_windows:]

                        # Hard guard (prevents silent garbage)
                        if int(X_train_3d.shape[0]) != int(len(y_train_eff)):
                            raise ValueError(
                                f"LSTM 3D-feed X/y mismatch: X={X_train_3d.shape[0]} y={len(y_train_eff)} "
                                f"(train_stride={train_stride}, max_train_windows={max_train_windows}, _si={_si})"
                            )

                        cb = getattr(self.model, "early_stop_callback", None)
                        print(f"[DEEP] model={model_type} | seq_windows=NA(3D-feed) "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_train_3d, y_train_eff,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        # Calibration inputs for shared block
                        X_cal  = X_train_3d
                        y_cal  = y_train_eff
                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)

                        self._fit_deep_calibration_and_coverage(
                            X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                            model_type=model_type, in_cv=in_cv
                        )
                        
                        # Free intermediate bases too (can be large)
                        try:
                            del X_tr2, y_tr2
                        except Exception:
                            pass
                        del X_train_3d, y_train_eff, X_cal, y_cal, pred_fn
                        
                        _gc.collect()

                else:  # CNN
                    cnn_use_seq    = bool(params.get("cnn_use_seq_windows", True))
                    train_stride   = max(1, int(params.get("cnn_train_stride", 3)))
                    batch_size     = int(params.get("cnn_batch_size", 128))
                    epochs         = min(int(params.get("cnn_epochs", 20)), 40)
                    use_mixed_prec = bool(params.get("cnn_mixed_precision", True)) and bool(tf.config.list_physical_devices("GPU"))
                    in_cv = bool(getattr(self, "_in_optuna_cv", False))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        stride_min = int(cfg.get("cnn_cv_train_stride", train_stride))
                        train_stride = max(train_stride, stride_min)

                    max_train_windows = int(params.get("deep_max_train_windows", 10000))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        max_cap = int(cfg.get("cnn_cv_max_train_windows", max_train_windows))
                        max_train_windows = min(max_train_windows, max_cap)


                    _maybe_mixed_precision(use_mixed_prec, "CNN")
                    params.setdefault("cnn_use_early_stopping", True)

                    # We'll set these after the fit, so the calibration block is shared
                    X_cal = None
                    y_cal = None
                    pred_fn = None

                    if cnn_use_seq:
                        # ---- Sequence windowing path ----
                        win = max(2, int(lags_eff))
                        
                        # Pre-slice DF so windows match build-all→stride→tail-slice
                        _si = _start_idx_for_last_strided_windows(len(train_data_scaled), win, train_stride, max_train_windows)
                        _df_tr = train_data_scaled.iloc[_si:] if _si > 0 else train_data_scaled
                        X2d = _df_tr[features].to_numpy(dtype=np.float32, copy=False)
                        y1d = _df_tr["label"].to_numpy(dtype=np.int32, copy=False)
 

                        X_seq_train, y_seq_train, _ = _make_windows_fast(
                            X2d, win=win, stride=train_stride, labels_1d=y1d
                        )
                        if X_seq_train is None or len(X_seq_train) == 0:
                            print("❌ [ABORT] Empty training sequences for CNN (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_empty_train_seq")

                        if X_seq_train.shape[0] > max_train_windows:
                            X_seq_train = X_seq_train[-max_train_windows:]
                            y_seq_train = y_seq_train[-max_train_windows:]

                        params["input_shape"] = (X_seq_train.shape[1], X_seq_train.shape[2])
                        self.model = self.get_model(model_type, **params)
                        # Tag for per-model CV caps.
                        setattr(self.model, "_mlb_model_tag", "cnn")

                        cb = getattr(self.model, "early_stop_callback", None)


                        print(f"[DEEP] model={model_type} | seq_windows={X_seq_train.shape[0]} "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_seq_train, y_seq_train,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        # Calibration inputs for shared block
                        X_cal  = X_seq_train
                        y_cal  = y_seq_train
                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)

                    else:
                        # ---- 3D "image-like" feed path ----
                        params["input_shape"] = (X_train.shape[1], 1)
                        self.model = self.get_model(model_type, **params)

                        # Preserve exact rows vs old path: (X_train[::stride] then tail-slice)
                        _si = _start_idx_for_last_stride_rows(X_train.shape[0], train_stride, max_train_windows)
                        if _si > 0:
                            X_tr2 = X_train[_si:]
                            y_tr2 = y_train[_si:]
                        else:
                            X_tr2 = X_train
                            y_tr2 = y_train

                        X_train_3d = X_tr2.astype(np.float32).reshape((X_tr2.shape[0], X_tr2.shape[1], 1))
                        
                        if train_stride > 1:
                            X_train_3d = X_train_3d[::train_stride]
                            y_train_eff = y_tr2[::train_stride]
                        else:
                            y_train_eff = y_tr2
 

                        if X_train_3d.shape[0] > max_train_windows:
                            X_train_3d  = X_train_3d[-max_train_windows:]
                            y_train_eff = y_train_eff[-max_train_windows:]

                        cb = getattr(self.model, "early_stop_callback", None)

                        print(f"[DEEP] model={model_type} | seq_windows=NA(3D-feed) "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_train_3d, y_train_eff,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        # Calibration inputs for shared block
                        X_cal  = X_train_3d
                        y_cal  = y_train_eff
                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)


                    if X_cal is not None and callable(pred_fn):
                        self._fit_deep_calibration_and_coverage(
                            X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                            model_type=model_type, in_cv=in_cv
                        )
                    try:
                        if cnn_use_seq:
                            # IMPORTANT: X_seq_train is a view over X2d (sliding_window_view).
                            # Free base arrays/DF slice too to reduce RSS growth.
                            try:
                                del X2d, y1d, _df_tr
                            except Exception:
                                pass
                            del X_seq_train, y_seq_train
                        else:
                            try:
                                del X_tr2, y_tr2
                            except Exception:
                                pass
                            del X_train_3d, y_train_eff
                    except Exception:
                        pass
                    try:
                        del X_cal, y_cal, pred_fn
                    except Exception:
                        pass
                    _gc.collect()


            elif model_type in ensemble_models:
                raise RuntimeError(
                    f"{model_type} should not be trained via test_strategy(). "
                    "Use the dedicated ensemble handler functions."
                )
            else:
            
                # Classical ML (logistic/logistic_ovr/svm/rf/xgb/…)
                self.model = self.get_model(model_type, **params)
            
                # Fit FIRST (required for sklearn Pipelines / predict_proba).
                self.model.fit(X_train, y_train)

                # ------------------------------------------------------------
                # Classical coverage-threshold (train-anchored, causal)
                # IMPORTANT: must run AFTER fit, otherwise sklearn Pipelines
                # can raise "Pipeline is not fitted yet."
                # ------------------------------------------------------------
                try:
                    cfg = getattr(self, "features_config", {}) or {}
                    tgt = float(cfg.get("target_active_rate", cfg.get("target_coverage", 0.0)) or 0.0)
                    if tgt > 0.0 and hasattr(self.model, "predict_proba"):
                        frac = float(cfg.get("classical_calibration_frac", cfg.get("deep_calibration_frac", 0.15)))
                        nmin = int(cfg.get("classical_calibration_min_samples", cfg.get("deep_calibration_min_samples", 500)))
                        nwin = int(X_train.shape[0]) if hasattr(X_train, "shape") else 0
                        ncal = max(nmin, int(round(nwin * frac))) if nwin > 0 else 0
                        ncal = min(ncal, nwin - 1) if nwin > 1 else 0
                        if ncal >= 50:
                            X_cal = X_train[-ncal:]
                            p_cal = sanitize_proba(self.model.predict_proba(X_cal))
                            self._coverage_conf_thr = float(fit_coverage_threshold_on_calibration(p_cal, tgt))
                            setattr(self, "_cv_cov_thr_last", float(self._coverage_conf_thr))
                            try:
                                setattr(self, "_last_cov_cal_rows", int(ncal))
                            except Exception:
                                pass
                            _in_cv = False
                            try:
                                _in_cv = bool(getattr(self, "_in_cv", False) or getattr(self, "_in_optuna_cv", False))
                            except Exception:
                                _in_cv = False

                            _ctx = "cv" if _in_cv else "eval"

                            # Only label as real_mX when NOT in CV
                            if not _in_cv:
                                try:
                                    if bool(getattr(self, "_in_real_sim", False)):
                                        mx = int(cfg.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                                        _ctx = f"real_m{mx}"
                                except Exception:
                                    pass
                            print(
                                f"[Calib][Coverage] conf_thr={float(self._coverage_conf_thr):.6f} "
                                f"target_active_rate={float(tgt):.6f} cal_rows={int(ncal)} ctx={_ctx}"
                            )
                except Exception as _e:
                    if self._is_debug():
                        print(f"⚠️ [Calib][Classical] Coverage fit skipped: {_e}")


            # ---- RAM (soft guard) BEFORE prediction ----
            try:
                import gc, time
                free_gb = psutil.virtual_memory().available / (1024**3)
                used_gb = psutil.virtual_memory().used / (1024**3)
                if self._is_debug():
                    print(f"[RAM] Before model predict: used={used_gb:.2f} GB | free={free_gb:.2f} GB | floor={float(os.environ.get('MLB_MIN_FREE_GB','2.5')):.2f} GB")

                # Soft guard: try local cleanup if free RAM is below floor; never raise
                if free_gb < float(os.environ.get("MLB_MIN_FREE_GB", "2.5")):
                    _gc.collect(); time.sleep(0.05)
                    free_retry = psutil.virtual_memory().available / (1024**3)
                    if self._is_debug():
                        print(f"[RAM] After cleanup: free={free_retry:.2f} GB")
                    if free_retry < float(os.environ.get("MLB_MIN_FREE_GB", "2.5")):
                        print(f"⚠️ Low free RAM persists ({free_retry:.2f} GB); continuing without raising.")
            except Exception:
                pass

        
            # ----------------------------
            # 4) Predict (branch specific)
            # ----------------------------
            test_data_for_eval = None

            if model_type in deep_models:
                # Build sliding windows over TEST (stride=1) when in seq mode
                if model_type == "transformer":
                    win = max(2, int(lags_eff))

                    X2d_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                    if X2d_test.shape[0] < win:
                        print("❌ [ABORT] Test set shorter than window size for transformer.")
                        return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_test_too_short")

                    n_win = int(X2d_test.shape[0] - win + 1)
                    idx_end = np.arange(win - 1, win - 1 + n_win, 1, dtype=int)

                    proba = getattr(self, "_deep_subproc_proba", None)
                    if proba is not None:
                        # shape guard: if stale/wrong, ignore and fallback
                        try:
                            if int(proba.shape[0]) != int(n_win):
                                proba = None
                        except Exception:
                            proba = None
                        # consume once
                        self._deep_subproc_proba = None

                    if proba is None:
                        bs = int(params.get("transformer_batch_size", 128))
                        free_gb_pred = psutil.virtual_memory().available / (1024**3)
                        floor_gb = float(os.environ.get("MLB_MIN_FREE_GB", "2.5"))
                        force_chunk = bool(int(os.environ.get("MLB_CHUNK_SEQ_PRED", "0")))
                        use_chunk = force_chunk or (free_gb_pred < floor_gb)

                        if use_chunk:
                            chunk_windows = int(os.environ.get("MLB_PRED_CHUNK_WINDOWS", "4096"))
                            print(f"ℹ️ Low-RAM predict: chunking windows (chunk_windows={chunk_windows}).")
                            proba = self._predict_seq_windows_chunked(
                                self.model, X2d_test, win=win, batch_size=bs, chunk_windows=chunk_windows
                            )
                        else:
                            Xv = sliding_window_view(X2d_test, window_shape=win, axis=0)
                            proba = self.model.predict(Xv, verbose=0, batch_size=bs)
                            
                    proba = sanitize_proba(proba)

                    # apply learned temperature if available
                    if hasattr(self, "_deep_temp_T"):
                        try:
                            proba = _apply_temperature_to_proba(proba, float(self._deep_temp_T))
                        except Exception:
                            pass

                    if proba.shape[1] >= 3:
                        p_short = proba[:, 0]
                        p_long  = proba[:, 2]
                        max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                        decoded_raw = np.asarray(
                            np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0)),
                            dtype=np.int8
                        )
                        raw_classes = np.asarray(
                            np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)),
                            dtype=np.int8
                        )
                    else:
                        raw = np.argmax(proba, axis=1)
                        max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                        raw_classes = np.asarray(raw, dtype=np.int8)
                        decoded_raw = np.asarray(np.where(raw_classes == 1, 1, -1), dtype=np.int8)



                    # --- Edge-vs-Cost gating (dynamic; align on window-end idx) ---
                    cfg_f = getattr(self, "features_config", {}) or {}
                    base_thr = float(self._resolve_conf_thr(confidence_threshold))
                    self._last_conf_thr_init = float(cfg_f.get("confidence_threshold", confidence_threshold))

                    _cfg_cost = dict(cfg_f)
                    if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                        _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                    _td_cost = test_data[["returns"]] if (test_data is not None and "returns" in test_data.columns) else test_data.loc[:, []]
                    _cost_src = self._ensure_cost_columns(_td_cost, _cfg_cost)


                    # Build drivers over all test rows, then sample by idx_end
                    _all_idx = test_data_scaled.index
                    rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)

                    # -------------------------------
                    # Causal volatility scaling patch
                    # -------------------------------
                    vol_w = int(cfg_f.get("vol_window_bars", 48))

                    # 1) Compute volatility scale + denom floor from TRAIN (causal)
                    rv_m_tr, rv_s_tr, den_floor_tr = float("nan"), float("nan"), float("nan")
                    try:
                        _tr_cost = train_data[["returns"]] if (train_data is not None and "returns" in train_data.columns) else train_data.loc[:, []]
                        _cost_train = self._ensure_cost_columns(_tr_cost, _cfg_cost)
                        rets_tr = _cost_train["returns"].astype(float)
                        rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                        rv_m_tr, rv_s_tr = float(np.nanmean(rv_tr)), float(np.nanstd(rv_tr))
                        _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                        if _pos.size > 0:
                            den_floor_tr = float(np.nanmedian(_pos))
                    except Exception:
                        pass

                    # 2) Compute realized vol on TEST, but reuse TRAIN stats
                    rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)

                    # Final z-score: bar-by-bar TEST vol vs TRAIN-based scale
                    if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                        vol_z_all = (rv_all - rv_m_tr) / rv_s_tr
                    else:
                        # Degenerate train stats → neutral vol term (no hidden test-fit fallback).
                        vol_z_all = np.zeros_like(rv_all, dtype=np.float32)

                    # Normalised spread vs vol: use TRAIN-derived floor (or constant) — never test-wide median.
                    den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                    den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
                    spread_norm_all = np.divide(
                        sprd_all,
                        den_all,
                        out=np.zeros_like(sprd_all, dtype=np.float32),
                        where=np.isfinite(den_all),
                    )

                    # αβγ coefficients unchanged
                    a = float(cfg_f.get("alpha_vol_z", 0.01))
                    b = float(cfg_f.get("beta_spread_norm", 0.02))
                    g = float(cfg_f.get("gamma_slip_norm", 0.01))
                    slip_norm_bps = float(cfg_f.get("slip_norm_bps", 10.0))
                    max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))

                    thr_full = (
                        base_thr
                        + a * vol_z_all
                        + b * spread_norm_all
                        + g * (slip_all / max(1e-9, slip_norm_bps))
                    )
                    thr_full = np.clip(thr_full, 0.0, max_conf_thr).astype(np.float32)

                    # Align thresholds to window ends / eval bars (unchanged)
                    try:
                        idx_arr = np.asarray(idx_end, dtype=int)
                    except NameError:
                        idx_arr = np.arange(len(test_data_scaled), dtype=int)
                        idx_end = idx_arr.tolist()

                    thr_vec = thr_full[idx_arr]

                    if self._is_debug():
                        print(
                            f"[Gate✔] Dynamic αβγ active | base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                            f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}"
                        )

                    # ------------------------------------------------------------
                    # IMPORTANT: define the TRUE evaluation universe for seq models
                    # (window-ends that land on eligible eval bars AND are >= anchor)
                    # ------------------------------------------------------------
                    idx_end_arr = np.asarray(idx_end, dtype=int)
                    keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                    # eval_mask is bar-level (session/warmup/eligibility); map to window ends
                    try:
                        _em = np.asarray(eval_mask, dtype=bool)
                        if _em.size == len(test_data_scaled):
                            keep_win &= _em[idx_end_arr]
                    except Exception:
                        pass
                    _eval_idx = np.flatnonzero(keep_win)

                    # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                    try:
                        tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                        band  = float(cfg_f.get("runtime_active_band_margin", 0.05))
                        win_k = int(cfg_f.get("runtime_coverage_window", 96))
                        step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                        # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                        # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                        try:
                            band = float(band)
                            step = float(step)
                        except Exception:
                            band, step = 0.0, 0.0
                        band = max(0.0, band)
                        step = abs(step)
                        if band > 0.0 and step > 0.5 * band:
                            _step_old = step
                            step = max(1e-6, 0.5 * band)
                            try:
                                if bool(getattr(self, "debug", False)):
                                    log_print(
                                        f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                        level="COMPACT",
                                    )
                            except Exception:
                                pass

                        n = int(_eval_idx.size)
                        
                        # --- Rolling-quantile cap (prevents "bunched confidence" => near-zero trades) ---
                        # If current activity is below the lower band, cap thresholds DOWN to the rolling
                        # (1 - tgt) quantile of past confidences so "top tgt%" remains achievable.
                        _low = max(0.0, tgt - band)
                        allow_qcap = bool(cfg_f.get("runtime_allow_rolling_qcap", True))
                        if allow_qcap and win_k > 1 and n >= win_k:
                            try:
                                _dr = decoded_raw[_eval_idx]
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                _act0 = ((_dr != 0) & (_mc >= _tv)).astype(np.float32)
                                if float(np.nanmean(_act0)) < _low:
                                    _q = (
                                        pd.Series(_mc)
                                        .rolling(win_k, min_periods=win_k)
                                        .quantile(1.0 - tgt)
                                        .shift(1)  # causal: use past only
                                        .to_numpy(dtype=np.float32)
                                    )
                                    _m = np.isfinite(_q)
                                    if _m.any():
                                        # cap thresholds only on eval windows
                                        _tv[_m] = np.minimum(_tv[_m], _q[_m])
                                        thr_vec[_eval_idx] = _tv
                                        if self._is_debug():
                                            print(
                                                f"[Gate✔] Rolling-quantile cap active | q={1.0 - tgt:.3f} "
                                                f"win={win_k} | thr_med={float(np.nanmedian(_tv)):.3f}"
                                            )
                            except Exception:
                                pass
                        
                        # preliminary decisions with αβγ only (causal)
                        if n > 0:
                            _dr = decoded_raw[_eval_idx].copy()
                            _mc = max_conf[_eval_idx]
                            _tv = thr_vec[_eval_idx]
                            _mask0 = (_mc < _tv)
                            _dr[_mask0] = 0
                            _act = (_dr != 0).astype(np.float32)
                        else:
                            _act = np.asarray([], dtype=np.float32)
                        if win_k > 1 and n > 0:
                            _cs = np.cumsum(np.insert(_act, 0, 0.0))
                            _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                            _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                            _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                            _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                            _drift = np.nan_to_num(_drift, nan=0.0)
                            min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                            max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                            _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                            thr_vec[_eval_idx] = np.clip(
                                thr_vec[_eval_idx] + _drift,
                                min_conf_thr,
                                max_conf_thr
                            ).astype(np.float32)
                            # ----------------------

                    except Exception as _e:
                        print(f"[Gate] Coverage nudge skipped (transformer-seq): {_e}")
                    self._last_conf_thr_used = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))
                
                    # (D1) Snapshot distributions for post-mortem debug (no look-ahead; uses eval arrays only)
                    try:
                        _nq = int(min(len(max_conf), len(thr_vec)))
                        if _eval_idx.size > 0:
                            _mc = max_conf[_eval_idx]
                            _tv = thr_vec[_eval_idx]
                            self._last_lstm_conf_q = tuple(np.nanquantile(_mc, [0.50, 0.75, 0.90]).astype(float).tolist())
                            self._last_lstm_thr_q  = tuple(np.nanquantile(_tv, [0.50, 0.75, 0.90]).astype(float).tolist())
                    except Exception:
                        pass

                    # Apply confidence filter ONLY on eligible eval windows (others forced flat)
                    final_preds = np.zeros_like(decoded_raw, dtype=int)
                    if _eval_idx.size > 0:
                        final_preds[_eval_idx] = decoded_raw[_eval_idx]
                        _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                        final_preds[_eval_idx[_mask]] = 0

                    # No-trade month → invalid fold (let CV aggregator/Optuna handle)
                
                    if self._is_debug():
                        try:
                            _rawc   = pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                            _finalc = pd.Series(final_preds[keep_win]).value_counts().to_dict()
                            print(f"[DeepGate][Dist][transformer-seq] raw={_rawc} | final={_finalc}")
                        except Exception:
                            pass
                    if (final_preds != 0).sum() == 0:
                        return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_seq_no_trades")
                
                    # Build evaluation frame aligned to end-of-window indices
                    idx_end_kept = idx_end_arr[keep_win]
                    keep = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                    idx_end_kept = idx_end_arr[keep]
                    if idx_end_kept.size == 0:
                        return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_seq_no_eval_windows")
                    eval_index = test_data_scaled.index[idx_end_kept]
                    final_preds_kept = np.asarray(final_preds, dtype=int)[keep_win]

                    test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                    test_data_for_eval["pred"] = final_preds_kept  # CLEANUP: keep-index aligned

                    # Stats for CV / summaries (normalize to -1/0/+1 so the table isn't garbage)
                    try:
                        raw_counts = _norm_class_counts(
                            pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                        )
                        final_counts = _norm_class_counts(
                            pd.Series(final_preds[keep_win]).value_counts().to_dict()
                        )
                        self._last_class_dists = {"raw": raw_counts, "final": final_counts}
                        self._last_conf_stats_label = str(model_type)
                        self._last_conf_stats_max_conf = np.asarray(max_conf[keep_win], dtype=np.float32)
                    except Exception:
                        self._last_class_dists = {"raw": {}, "final": {}}
                        self._last_conf_stats_label = str(model_type)
                        self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

                elif model_type == "lstm":
                    lstm_use_seq = bool(params.get("lstm_use_seq_windows", True))


                    # ==================================================================
                    #  LSTM – SEQ MODE (sliding windows + idx_end)
                    # ==================================================================
                    if lstm_use_seq:
                        win = max(2, int(lags_eff))

                        X2d_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                        if X2d_test.shape[0] < win:
                            print("❌ [ABORT] Test set shorter than window size for LSTM (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_seq_test_too_short")
                        n_win = int(X2d_test.shape[0] - win + 1)
                        idx_end = np.arange(win - 1, win - 1 + n_win, 1, dtype=int)

                        bs = int(params.get("lstm_batch_size", 128))
                        free_gb_pred = psutil.virtual_memory().available / (1024**3)
                        floor_gb = float(os.environ.get("MLB_MIN_FREE_GB", "2.5"))
                        force_chunk = bool(int(os.environ.get("MLB_CHUNK_SEQ_PRED", "0")))
                        use_chunk = force_chunk or (free_gb_pred < floor_gb)

                        if use_chunk:
                            chunk_windows = int(os.environ.get("MLB_PRED_CHUNK_WINDOWS", "4096"))
                            print(f"ℹ️ Low-RAM predict: chunking windows (chunk_windows={chunk_windows}).")
                            proba = self._predict_seq_windows_chunked(
                                self.model, X2d_test, win=win, batch_size=bs, chunk_windows=chunk_windows
                            )
                        else:
                            Xv = sliding_window_view(X2d_test, window_shape=win, axis=0)
                            proba = self.model.predict(Xv, verbose=0, batch_size=bs)

                        proba = sanitize_proba(proba)

                        # apply learned temperature if available
                        if hasattr(self, "_deep_temp_T"):
                            try:
                                proba = _apply_temperature_to_proba(proba, float(self._deep_temp_T))
                            except Exception:
                                pass

                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)

                        # --- Edge-vs-Cost gating (dynamic; align on window-end idx) ---
                        _global_f = (CLASS_DEFAULTS.get("features", {}) if "CLASS_DEFAULTS" in globals() else {})
                        _cfg_raw  = (getattr(self, "features_config", {}) or {})
                        cfg_f     = {**_global_f, **_cfg_raw}

                        # 1) Base confidence threshold (from CV / coverage fit / user override)
                        base_thr = float(self._resolve_conf_thr(confidence_threshold))
                        self._last_conf_thr_init = float(
                            cfg_f.get("confidence_threshold", confidence_threshold)
                        )

                        # 2) Build cost & volatility drivers on the full test index
                        _cfg_cost = dict(cfg_f)
                        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                        _td_cost = test_data[["returns"]] if (test_data is not None and "returns" in test_data.columns) else test_data.loc[:, []]
                        _cost_src = self._ensure_cost_columns(_td_cost, _cfg_cost)

                        _all_idx = test_data_scaled.index
                        rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)

                        # Volatility z-score (realized vol over window)
                        vol_w = int(cfg_f.get(
                            "vol_window_bars",
                            _global_f.get("vol_window_bars", 96)
                        ))

                        # --- Train-anchored vol scaling (avoid ex-post test-month stats) ---
                        rv_m_tr, rv_s_tr, den_floor_tr = np.nan, np.nan, np.nan
                        try:
                            rets_tr = train_data_scaled["returns"].astype(float)
                            rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                            rv_m_tr = float(np.nanmean(rv_tr))
                            rv_s_tr = float(np.nanstd(rv_tr))
                            _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                            if _pos.size > 0:
                                den_floor_tr = float(np.nanmedian(_pos))
                        except Exception:
                            pass

                        rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)
                        if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                            vol_z_all = (rv_all - rv_m_tr) / rv_s_tr
                        else:
                            vol_z_all = np.zeros_like(rv_all, dtype=np.float32)

                        den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                        den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
                        spread_norm_all = np.divide(
                            sprd_all,
                            den_all,
                            out=np.zeros_like(sprd_all, dtype=np.float32),
                            where=np.isfinite(den_all)
                        )

                        # 3) αβγ: volatility-, spread-, and slippage-aware threshold bump
                        a = float(cfg_f.get("alpha_vol_z", _global_f.get("alpha_vol_z", 0.004)))
                        b = float(cfg_f.get("beta_spread_norm", _global_f.get("beta_spread_norm", 0.008)))
                        g = float(cfg_f.get("gamma_slip_norm", _global_f.get("gamma_slip_norm", 0.004)))
                        slip_norm_bps = float(cfg_f.get("slip_norm_bps", _global_f.get("slip_norm_bps", 0.25)))
                        min_slip_norm_bps = float(cfg_f.get("min_slip_norm_bps", _global_f.get("min_slip_norm_bps", 0.05)))
                        slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

                        vol_z_cap = float(cfg_f.get("vol_z_cap", _global_f.get("vol_z_cap", 6.0)))
                        spread_norm_cap = float(cfg_f.get("spread_norm_cap", _global_f.get("spread_norm_cap", 5.0)))
                        slip_ratio_cap = float(cfg_f.get("slip_ratio_cap", _global_f.get("slip_ratio_cap", 6.0)))
                        max_conf_thr = float(cfg_f.get("max_conf_thr", _global_f.get("max_conf_thr", 0.90)))
                        
                        try:
                            idx_arr = np.asarray(idx_end, dtype=int)
                        except NameError:
                            idx_arr = np.arange(len(test_data_scaled), dtype=int)
                            idx_end = idx_arr.tolist()

                        tgt_ar = cfg_f.get("target_active_rate", cfg_f.get("target_coverage", None))
                        base_thr_vec = float(base_thr)  # default: scalar
                        if tgt_ar is not None:
                            try:
                                W = int(cfg_f.get(
                                    "coverage_rolling_window",
                                    _global_f.get("coverage_rolling_window", 48)
                                ))
                                _minp = max(10, W // 3)
                                # Build full-length confidence aligned to bar index, then roll causally.
                                conf_full = np.full(len(test_data_scaled), np.nan, dtype=np.float32)
                                _nfill = int(min(len(max_conf), len(idx_arr)))
                                if _nfill > 0:
                                    conf_full[idx_arr[:_nfill]] = np.asarray(max_conf[:_nfill], dtype=np.float32)
                                conf_s = pd.Series(conf_full, index=test_data_scaled.index).astype(float)
                                thr_roll = (
                                    conf_s.rolling(W, min_periods=_minp)
                                          .quantile(1.0 - float(tgt_ar))
                                          .shift(1)
                                )
                                base_thr_vec = thr_roll.fillna(float(base_thr)).to_numpy(dtype=np.float32)
                                if self._is_debug():
                                    print(
                                        f"[Gate✔][RollingQuantile] W={W} target={float(tgt_ar):.3f} "
                                        f"base_med={float(np.nanmedian(base_thr_vec)):.3f}"
                                    )
                            except Exception as _e:
                                base_thr_vec = float(base_thr)


                        slip_ratio = np.clip(slip_all / max(1e-9, slip_norm_bps), 0.0, slip_ratio_cap)
                        vol_z_all = np.clip(vol_z_all, -vol_z_cap, vol_z_cap)
                        spread_norm_all = np.clip(spread_norm_all, 0.0, spread_norm_cap)

                        thr_full = np.clip(
                            base_thr_vec
                            + a * vol_z_all
                            + b * spread_norm_all
                            + g * slip_ratio,
                            0.0, max_conf_thr
                        ).astype(np.float32)

                        thr_vec = thr_full[idx_arr]

                        print(
                            "[Gate✔] Dynamic αβγ active | "
                            f"base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                            f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}"
                        )
                        
                        # ------------------------------------------------------------
                        # IMPORTANT: define eval-universe for seq models (LSTM-seq)
                        # windows whose END index is eligible AND on/after anchor
                        # ------------------------------------------------------------
                        idx_end_arr = np.asarray(idx_end, dtype=int)
                        keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                        try:
                            _em = np.asarray(eval_mask, dtype=bool)
                            if _em.size == len(test_data_scaled):
                                keep_win &= _em[idx_end_arr]
                        except Exception:
                            pass
                        _eval_idx = np.flatnonzero(keep_win)

                        # 4) Soft coverage-drift nudge (regime-aware, non-forcing)
                        try:
                            # Anchor: Optuna-tuned or config-provided target_active_rate / target_coverage
                            tgt = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))

                            band = float(cfg_f.get(
                                "runtime_active_band_margin",
                                _global_f.get("runtime_active_band_margin", 0.05)
                            ))
                            win_k = int(cfg_f.get(
                                "runtime_coverage_window",
                                _global_f.get("runtime_coverage_window", 96)
                            ))
                            step = float(cfg_f.get(
                                "runtime_conf_nudge",
                                _global_f.get("runtime_conf_nudge", 0.01)
                            ))


                            # Stabilize runtime nudge params (avoid flip-flop when step > band/2)
                            band, step = self._sanitize_runtime_coverage_nudge(band, step, ctx="runtime")

                            # ------------------------------------------------------------
                            # IMPORTANT: restrict nudging + filtering to TRUE eval universe
                            # windows whose END index is eligible AND on/after anchor
                            # ------------------------------------------------------------
                            idx_end_arr = np.asarray(idx_end, dtype=int)
                            keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                            try:
                                _em = np.asarray(eval_mask, dtype=bool)
                                if _em.size == len(test_data_scaled):
                                    keep_win &= _em[idx_end_arr]
                            except Exception:
                                pass
                            _eval_idx = np.flatnonzero(keep_win)

                            n = int(_eval_idx.size)
                            if n > 0 and win_k > 1 and step > 0.0:
                                # Decisions using αβγ-threshold only (on eval windows)
                                _dr = decoded_raw[_eval_idx].copy()
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                _dr[_mc < _tv] = 0
                                _act = (_dr != 0).astype(np.float32)

                                # Rolling active rate (simple moving average) on eval windows
                                roll = np.full_like(_act, np.nan, dtype=np.float32)
                                if n >= win_k:
                                    csum = np.cumsum(_act, dtype=float)
                                    roll[win_k - 1:] = (
                                        csum[win_k - 1:] -
                                        np.concatenate(([0.0], csum[:-win_k]))
                                    ) / float(win_k)

                                low, high = tgt - band, tgt + band
                                low = max(0.0, low)
                                high = min(1.0, high)

                                sel = np.isfinite(roll)
                                below = sel & (roll < low)
                                above = sel & (roll > high)

                                drift = np.zeros(n, dtype=np.float32)
                                drift[below] = -step   # too quiet → lower threshold → more trades
                                drift[above] = step    # too active → raise threshold → fewer trades

                                min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                                # Apply drift to thresholds on eval windows (THIS WAS MISSING BEFORE)
                                thr_vec[_eval_idx] = np.clip(
                                    thr_vec[_eval_idx] + drift,
                                    min_conf_thr,
                                    max_conf_thr
                                ).astype(np.float32)

                            self._last_conf_thr_used = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))
                                                    
                            if self._is_debug():
                                print(
                                    "[Gate✔] Coverage nudge active | "
                                    f"target={tgt:.2f} band=±{band:.2f} "
                                    f"step={step:.3f} | median_used={self._last_conf_thr_used:.3f}"
                                )

                        except Exception as _ee:
                            # Fail-safe: keep αβγ-only thr if nudging breaks
                            self._last_conf_thr_used = float(np.nanmedian(thr_vec))
                            print(f"[Gate] Coverage nudge skipped: {type(_ee).__name__}: {_ee}")


                        # (D1) Snapshot distributions for post-mortem debug (no look-ahead; uses eval arrays only)
                        try:
                            if _eval_idx.size > 0:
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                self._last_lstm_conf_q = tuple(np.nanquantile(_mc, [0.50, 0.75, 0.90]).astype(float).tolist())
                                self._last_lstm_thr_q  = tuple(np.nanquantile(_tv, [0.50, 0.75, 0.90]).astype(float).tolist())
                        except Exception:
                            pass

                        # 5) Apply gating to predictions (seq mode) — only on eval windows, force flat elsewhere
                        final_preds = np.zeros_like(decoded_raw, dtype=int)
                        if _eval_idx.size > 0:
                            final_preds[_eval_idx] = decoded_raw[_eval_idx]
                            _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                            final_preds[_eval_idx[_mask]] = 0

                        # No-trade month → invalid fold (let CV aggregator/Optuna handle)
                        if self._is_debug():
                            try:
                                _rawc   = pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                                _finalc = pd.Series(final_preds[keep_win]).value_counts().to_dict()
                                print(f"[DeepGate][Dist][lstm-seq] raw={_rawc} | final={_finalc}")
                            except Exception:
                                pass

                        if (final_preds != 0).sum() == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_seq_no_trades")

                        # keep only windows ending on/after the first eval bar AND eligible by eval_mask
                        idx_end_kept = idx_end_arr[keep_win]
                        if idx_end_kept.size == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_seq_no_eval_windows")

                        eval_index = test_data_scaled.index[idx_end_kept]
                        final_preds_kept = np.asarray(final_preds, dtype=int)[keep_win]

                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(final_preds_kept, index=eval_index).values

                        # Stats for CV / summaries (normalize to -1/0/+1 so fold table prints correctly)
                        try:
                            raw_counts = _norm_class_counts(
                                pd.Series(decoded_raw[keep_win]).value_counts(dropna=False).to_dict()
                            )
                            final_counts = _norm_class_counts(
                                pd.Series(final_preds[keep_win]).value_counts(dropna=False).to_dict()
                            )

                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf[keep_win], dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)


                    # ==================================================================
                    #  LSTM – 3D-FEED MODE (flat windows, no idx_end)
                    # ==================================================================
                    else:
                        X_test_3d = X_test.astype(np.float32).reshape(
                            (X_test.shape[0], X_test.shape[1], 1)
                        )
                        proba = self.model.predict(
                            X_test_3d, verbose=0,
                            batch_size=int(params.get("lstm_batch_size", 128))
                        )
                        proba = sanitize_proba(proba)

                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)

                        # non-seq branch retains backoff/coverage logic unchanged
                        cfg_f  = getattr(self, "features_config", {}) or {}
                        cv_mode = bool(getattr(self, "_in_optuna_cv", False))

                        # Use unified resolver so:
                        #  - coverage-based threshold is respected (if fitted),
                        #  - LSTM gets its per-model relaxation (lstm_conf_relax / floor).
                        base_conf = float(cfg_f.get("confidence_threshold", confidence_threshold))
                        conf0     = float(self._resolve_conf_thr(base_conf))

                        q75, q50, q90 = np.quantile(max_conf, [0.75, 0.50, 0.90])

                        # (D1) Snapshot confidence distribution quantiles (3D has no thr_vec)
                        try:
                            self._last_lstm_conf_q = (float(q50), float(q75), float(q90))
                        except Exception:
                            pass

                        # --- Debug: threshold vs confidence distribution (eval/WFO only) ---
                        if not cv_mode:
                            n_conf0 = int((max_conf >= conf0).sum())
                            _mtag = getattr(self, "model_type", str(model_type))
                            print(
                                f"[Deep3D][GateDebug] model={_mtag} "
                                f"| conf0={conf0:.3f} q75={q75:.3f} q90={q90:.3f} "
                                f"| n_conf0={n_conf0}/{len(max_conf)}"
                            )

                        allow_cv_backoff   = bool(cfg_f.get("allow_conf_backoff_cv", False))
                        allow_eval_backoff = bool(cfg_f.get("allow_conf_backoff_eval", False))
                        floor_cv   = float(cfg_f.get("conf_backoff_floor_cv", 0.33))
                        floor_eval = float(cfg_f.get("conf_backoff_floor_eval", 0.33))

                        candidates = [conf0]
                        if cv_mode:
                            if allow_cv_backoff:
                                candidates = [conf0, min(conf0, q90), q75]
                                candidates = [max(floor_cv, c) for c in candidates]
                        else:
                            if allow_eval_backoff:
                                candidates = [conf0, min(conf0, q90), q75, 0.33, 0.25]
                                candidates = [max(floor_eval, c) for c in candidates]

                        _seen = set()
                        candidates = [
                            x for x in candidates
                            if (round(x, 6) not in _seen and not _seen.add(round(x, 6)))
                        ]

                        final_preds = None
                        self._last_conf_thr_init = float(conf0)
                        self._last_max_conf_q75  = float(q75)
                        self._last_max_conf_q90  = float(q90)
                        self._last_conf_backoff_steps = 0

                        for thr in candidates:
                            preds_try = np.asarray(decoded_raw, dtype=np.int8).copy()

                            preds_try[max_conf < thr] = 0
                            if np.count_nonzero(preds_try) > 0:
                                if abs(thr - conf0) > 1e-9:
                                    print(
                                        f"⚠️ Confidence threshold relaxed {conf0:.3f} → "
                                        f"{thr:.3f} to avoid 0 trades."
                                    )
                                    self._last_conf_backoff_steps = 1
                                final_preds = preds_try
                                self._last_conf_thr_used = float(thr)
                                try:
                                    # 3D uses a scalar threshold; store as a "degenerate" quantile tuple
                                    self._last_lstm_thr_q = (float(thr), float(thr), float(thr))
                                except Exception:
                                    pass
                            
                            
                                break
                        
                        # HARD guard: thin-trades fallback must never be used in CV.
                        if cv_mode and bool(cfg_f.get("allow_thin_trades_fallback", False)):
                            raise RuntimeError(
                                "CV thin-trades fallback is disabled: remove allow_thin_trades_fallback "
                                "(CV must not invent trades that real months will not take)."
                            )


                        if final_preds is None or np.count_nonzero(final_preds) == 0:
                            # Penalize no-trade configs during Optuna/CV (helps search),
                            # but NEVER poison real_trading_simulation with NaNs.
                            in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            in_real = bool(getattr(self, "_in_real_sim", False))
                            if in_cv and not in_real:
                                if self._is_debug():
                                    print("❗ No trades predicted after filtering — penalizing this parameter set.")
                                    try:
                                        _rawc = pd.Series(decoded_raw).value_counts().to_dict()
                                        _finalc = {} if final_preds is None else pd.Series(final_preds).value_counts().to_dict()
                                        print(f"[DeepGate][Dist][deep3d] raw={_rawc} | final={_finalc}")
                                    except Exception:
                                        pass
                                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_trades_cv")
                            if self._is_debug():
                                print("🟨 No trades predicted after filtering — keeping 0-trade evaluation (real-sim / non-CV).")
                                
                        # FAIL-SAFE: if still None here, force HOLD vector so we don't crash on eval_mask slicing.
                        if final_preds is None:
                            final_preds = np.zeros_like(decoded_raw, dtype=int)
                            try:
                                self._last_conf_thr_used = float(conf0)
                            except Exception:
                                pass
                    
                        # --- Debug: final trade count after eval_mask (eval/WFO only) ---
                        if not cv_mode:
                            n_trades = int(np.count_nonzero(final_preds))
                            _mtag = getattr(self, "model_type", str(model_type))
                            print(
                                f"[Deep3D][GateDebug] model={_mtag} "
                                f"| trades_after_mask={n_trades}"
                            )
 

                        # 3D path uses eval_mask (already computed outside this block)
                        eval_index = test_data_scaled.index[eval_mask]
                        final_preds = final_preds[eval_mask]

                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(
                            final_preds, index=eval_index
                        ).values

                        try:
                            raw_counts = pd.Series(raw_classes).value_counts().to_dict()
                            final_counts = pd.Series(final_preds).value_counts().to_dict()

                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

                else:  # CNN
                    cnn_use_seq = bool(params.get("cnn_use_seq_windows", True))

                    if cnn_use_seq:
                        win = max(2, int(lags_eff))

                        X2d_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                        if X2d_test.shape[0] < win:
                            print("❌ [ABORT] Test set shorter than window size for CNN (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_test_too_short")

                        n_win = int(X2d_test.shape[0] - win + 1)
                        idx_end = np.arange(win - 1, win - 1 + n_win, 1, dtype=int)

                        bs = int(params.get("cnn_batch_size", 128))
                        free_gb_pred = psutil.virtual_memory().available / (1024**3)
                        floor_gb = float(os.environ.get("MLB_MIN_FREE_GB", "2.5"))
                        force_chunk = bool(int(os.environ.get("MLB_CHUNK_SEQ_PRED", "0")))
                        use_chunk = force_chunk or (free_gb_pred < floor_gb)

                        if use_chunk:
                            chunk_windows = int(os.environ.get("MLB_PRED_CHUNK_WINDOWS", "4096"))
                            print(f"ℹ️ Low-RAM predict: chunking windows (chunk_windows={chunk_windows}).")
                            proba = self._predict_seq_windows_chunked(
                                self.model, X2d_test, win=win, batch_size=bs, chunk_windows=chunk_windows
                            )
                        else:
                            Xv = sliding_window_view(X2d_test, window_shape=win, axis=0)
                            proba = self.model.predict(Xv, verbose=0, batch_size=bs)

                        proba = sanitize_proba(proba)
                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)
                            
                            
                        # --- Edge-vs-Cost gating (dynamic; align on window-end idx) ---
                        cfg_f = getattr(self, "features_config", {}) or {}
                        base_thr = float(self._resolve_conf_thr(confidence_threshold))
                        self._last_conf_thr_init = float(cfg_f.get("confidence_threshold", confidence_threshold))

                        _cfg_cost = dict(cfg_f)
                        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                        _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)

                        _all_idx = test_data_scaled.index
                        rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)

                        vol_w = int(cfg_f.get("vol_window_bars", 48))
                    
                        rv_m_tr, rv_s_tr, den_floor_tr = np.nan, np.nan, np.nan
                        try:
                            rets_tr = train_data_scaled["returns"].astype(float)
                            rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                            rv_m_tr = float(np.nanmean(rv_tr))
                            rv_s_tr = float(np.nanstd(rv_tr))
                            _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                            if _pos.size > 0:
                                den_floor_tr = float(np.nanmedian(_pos))
                        except Exception:
                            pass

                        rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)
                        if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                            vol_z_all = (rv_all - rv_m_tr) / rv_s_tr
                        else:
                            vol_z_all = np.zeros_like(rv_all, dtype=np.float32)

                        den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                        den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
                        spread_norm_all = np.divide(sprd_all, den_all, out=np.zeros_like(sprd_all, dtype=np.float32), where=np.isfinite(den_all))

                        # Dynamic αβγ coefficients (small by default) and slippage scaling in bps
                        a = float(cfg_f.get("alpha_vol_z", 0.01))
                        b = float(cfg_f.get("beta_spread_norm", 0.02))
                        g = float(cfg_f.get("gamma_slip_norm", 0.01))
                        slip_norm_bps = float(cfg_f.get("slip_norm_bps", 10.0))
                        max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))
                    
                        thr_full = (
                            base_thr
                            + a * vol_z_all
                            + b * spread_norm_all
                            + g * (slip_all / max(1e-9, slip_norm_bps))
                        )
                        thr_full = np.clip(thr_full, 0.0, max_conf_thr).astype(np.float32)

                        # ✅ Align thresholds with window-end indices (seq) or fallback to full rows
                        try:
                            idx_arr = np.asarray(idx_end, dtype=int)
                        except NameError:
                            idx_arr = np.arange(len(test_data_scaled), dtype=int)
                            idx_end = idx_arr.tolist()

                        thr_vec = thr_full[idx_arr]

                        if self._is_debug():
                            print(
                                f"[Gate✔] Dynamic αβγ active | base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                                f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}"
                            )
                            
                        # ------------------------------------------------------------
                        # IMPORTANT: define eval-universe for seq models (CNN-seq)
                        # windows whose END index is eligible AND on/after anchor
                        # ------------------------------------------------------------
                        idx_end_arr = np.asarray(idx_end, dtype=int)
                        keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                        try:
                            _em = np.asarray(eval_mask, dtype=bool)
                            if _em.size == len(test_data_scaled):
                                keep_win &= _em[idx_end_arr]
                        except Exception:
                            pass
                        _eval_idx = np.flatnonzero(keep_win)

                        # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                        try:
                            tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                            band  = float(cfg_f.get("runtime_active_band_margin", 0.05))
                            win_k = int(cfg_f.get("runtime_coverage_window", 96))
                            step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                            # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                            # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                            try:
                                band = float(band)
                                step = float(step)
                            except Exception:
                                band, step = 0.0, 0.0
                            band = max(0.0, band)
                            step = abs(step)
                            if band > 0.0 and step > 0.5 * band:
                                _step_old = step
                                step = max(1e-6, 0.5 * band)
                                try:
                                    if bool(getattr(self, "debug", False)):
                                        log_print(
                                            f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                            level="COMPACT",
                                        )
                                except Exception:
                                    pass

                            n = int(_eval_idx.size)
                            if n > 0:
                                _pre = decoded_raw[_eval_idx].copy()
                                _mask0 = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                                _pre[_mask0] = 0
                                _act = (_pre != 0).astype(np.float32)
                            if win_k > 1 and n > 0:
                                _cs = np.cumsum(np.insert(_act, 0, 0.0))
                                _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                                _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                                _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                                _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                                _drift = np.nan_to_num(_drift, nan=0.0)
                                min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                                max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                                _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                                thr_vec[_eval_idx] = np.clip(
                                    thr_vec[_eval_idx] + _drift,
                                    min_conf_thr,
                                    max_conf_thr
                                ).astype(np.float32)

                                if self._is_debug():
                                    print(f"[Gate✔] Coverage nudge active | target={tgt:.2f} band=±{band:.2f} step={step:.3f}")
                        except Exception as _e:
                            print(f"[Gate] Coverage nudge skipped (cnn-seq): {_e}")

                        self._last_conf_thr_used = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))
                    

                    
                        # (D1) Snapshot distributions for post-mortem debug (no look-ahead; uses eval arrays only)
                        try:
                            if _eval_idx.size > 0:
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                self._last_lstm_conf_q = tuple(np.nanquantile(_mc, [0.50, 0.75, 0.90]).astype(float).tolist())
                                self._last_lstm_thr_q  = tuple(np.nanquantile(_tv, [0.50, 0.75, 0.90]).astype(float).tolist())
                        except Exception:
                            pass
                        
                        # Apply confidence filter ONLY on eligible eval windows (others forced flat)
                        final_preds = np.zeros_like(decoded_raw, dtype=int)
                        if _eval_idx.size > 0:
                            final_preds[_eval_idx] = decoded_raw[_eval_idx]
                            _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                            final_preds[_eval_idx[_mask]] = 0
                    
                        if self._is_debug():
                            try:
                                _rawc   = pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                                _finalc = pd.Series(final_preds[keep_win]).value_counts().to_dict()
                                print(f"[DeepGate][Dist][cnn-seq] raw={_rawc} | final={_finalc}")
                            except Exception:
                                pass

                        if final_preds is None or (final_preds != 0).sum() == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_no_trades")
                    
                        # keep only windows ending on/after the first eval bar
                        idx_end_kept = idx_end_arr[keep_win]
                        if idx_end_kept.size == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_no_eval_windows")
                    
                        eval_index = test_data_scaled.index[idx_end_kept]
                        final_preds_kept = np.asarray(final_preds, dtype=int)[keep_win]

                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(final_preds_kept, index=eval_index).values

                        try:
                            raw_counts = _norm_class_counts(
                                pd.Series(decoded_raw[keep_win]).value_counts(dropna=False).to_dict()
                            )
                            final_counts = _norm_class_counts(
                                pd.Series(final_preds[keep_win]).value_counts(dropna=False).to_dict()
                            )

                            # Store for CV / summary
                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }

                            # Store confidence stats so CV can aggregate
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf[keep_win], dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

                    else:
                        X_test_3d = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
                        proba = self.model.predict(
                            X_test_3d, verbose=0, batch_size=int(params.get("cnn_batch_size", 128))
                        )
                        proba = sanitize_proba(proba)
                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)

                        # --- Edge-vs-Cost gating (flat 3D variant) ---
                        cfg_f = getattr(self, "features_config", {}) or {}
                        base_thr = float(self._resolve_conf_thr(confidence_threshold))
                        self._last_conf_thr_init = float(cfg_f.get("confidence_threshold", confidence_threshold))

                        _cfg_cost = dict(cfg_f)
                        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                        _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)

                        _eval_idx = test_data_scaled.index
                        rets, sprd, slip = self._get_cost_arrays_aligned(_cost_src, _eval_idx)

                        vol_w = int(cfg_f.get("vol_window_bars", 48))
                        rv = realized_vol(rets, window=vol_w).to_numpy(dtype=np.float32)
                    
                        # --- Causal scaling: compute μ/σ (and a safe floor) from TRAIN only ---
                        # Avoid using full-test-month statistics to set live thresholds.
                        try:
                            rets_tr = train_data_scaled["returns"].astype(float)
                            rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                            rv_m_tr = float(np.nanmean(rv_tr))
                            rv_s_tr = float(np.nanstd(rv_tr))
                            rv_floor_tr = float(np.nanmedian(rv_tr[rv_tr > 0])) if np.any(rv_tr > 0) else float("nan")
                        except Exception:
                            rv_m_tr, rv_s_tr, rv_floor_tr = float("nan"), float("nan"), float("nan")
        
                        if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                            vol_z = (rv - rv_m_tr) / rv_s_tr
                        else:
                            # Degenerate train stats → neutral vol term (no hidden test-fit fallback).
                            vol_z = np.zeros_like(rv, dtype=np.float32)
        
                        # Normalized spread vs vol: use TRAIN-derived floor (or constant) — never test-wide median.
                        den_floor = rv_floor_tr if (np.isfinite(rv_floor_tr) and rv_floor_tr > 1e-8) else 1e-6
                        den = np.where(rv > 1e-8, rv, den_floor).astype(np.float32)
                        spread_norm = np.divide(sprd, den, out=np.zeros_like(sprd, dtype=np.float32), where=np.isfinite(den))

                        a = float(cfg_f.get("alpha_vol_z", 0.004))
                        b = float(cfg_f.get("beta_spread_norm", 0.008))
                        g = float(cfg_f.get("gamma_slip_norm", 0.004))
                        slip_norm_bps = float(cfg_f.get("slip_norm_bps", 0.25))
                        min_slip_norm_bps = float(cfg_f.get("min_slip_norm_bps", 0.05))
                        slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

                        vol_z_cap = float(cfg_f.get("vol_z_cap", 6.0))
                        spread_norm_cap = float(cfg_f.get("spread_norm_cap", 5.0))
                        slip_ratio_cap = float(cfg_f.get("slip_ratio_cap", 6.0))
                        max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))

                        vol_z = np.clip(vol_z, -vol_z_cap, vol_z_cap)
                        spread_norm = np.clip(spread_norm, 0.0, spread_norm_cap)
                        slip_norm = np.clip(slip / slip_norm_bps, 0.0, slip_ratio_cap)

                        thr_vec = np.clip(base_thr + a*vol_z + b*spread_norm + g*slip_norm, 0.0, max_conf_thr).astype(np.float32)
                        
                        # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                        try:
                            tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                            band  = float(cfg_f.get("runtime_active_band_margin", 0.05))
                            win_k = int(cfg_f.get("runtime_coverage_window", 96))
                            step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                            # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                            # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                            try:
                                band = float(band)
                                step = float(step)
                            except Exception:
                                band, step = 0.0, 0.0
                            band = max(0.0, band)
                            step = abs(step)
                            if band > 0.0 and step > 0.5 * band:
                                _step_old = step
                                step = max(1e-6, 0.5 * band)
                                try:
                                    if bool(getattr(self, "debug", False)):
                                        log_print(
                                            f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                            level="COMPACT",
                                        )
                                except Exception:
                                    pass
                                                    
                            n = min(len(decoded_raw), len(thr_vec))
                            
                            # preliminary gating with αβγ only
                            _pre = decoded_raw.copy()
                            _mask0 = (max_conf[:n] < thr_vec[:n])
                            np.putmask(_pre[:n], _mask0, 0)
                            # causal rolling active rate on preliminary decisions
                            _act = (_pre[:n] != 0).astype(np.float32)
                            if win_k > 1 and n >= win_k:
                                _cs = np.cumsum(np.insert(_act, 0, 0.0))
                                _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                                # pad to length n
                                _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                                _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                                _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                                _drift = np.nan_to_num(_drift, nan=0.0)
                                min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                                max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                                _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                                thr_vec[:n] = np.clip(
                                    thr_vec[:n] + _drift[:n],
                                    min_conf_thr,
                                    max_conf_thr
                                ).astype(np.float32)

                                if self._is_debug():
                                    print(f"[Gate✔] Coverage nudge active | target={tgt:.2f} band=±{band:.2f} step={step:.3f}")
                        except Exception as _e:
                            print(f"[Gate] Coverage nudge skipped: {_e}")
                        if self._is_debug():
                            print(f"[Gate✔] Dynamic αβγ active | base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                                f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}")

                        self._last_conf_thr_used = float(np.nanmedian(thr_vec))

                        final_preds = decoded_raw.copy()
                        n = min(len(final_preds), len(thr_vec))
                        mask = (max_conf[:n] < thr_vec[:n])
                        np.putmask(final_preds[:n], mask, 0)

                        if final_preds is None or (final_preds != 0).sum() == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_no_trades")

                        eval_index = test_data_scaled.index[eval_mask]
                        final_preds = final_preds[eval_mask]
                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(final_preds, index=eval_index).values

                        try:
                            raw_counts = pd.Series(raw_classes).value_counts().to_dict()
                            final_counts = pd.Series(final_preds).value_counts().to_dict()

                            # Store for CV / summary
                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }

                            # Store confidence stats so CV can aggregate
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

            else:
                # Classical ML prediction path (with CV backoff + thin-trades fallback)
                # 1) Get class probabilities (calibrated if requested)
                try:
                    proba = self.model.predict_proba(X_test)
                    try:
                        cal_method = str((params.get("calibrate_method")
                                        or (self.features_config or {}).get("calibrate_method")
                                        or "")).lower()
                    except Exception:
                        cal_method = ""
                    if cal_method in ("isotonic", "sigmoid") and str(model_type).lower() != "svm":
                        try:
                            n = int(getattr(X_train, "shape", [0])[0])
                            if n > 0:
                                n_cal = max(200, min(n // 10, 2000))
                                cal_X = X_train[-n_cal:] if n_cal < n else X_train
                                cal_y = y_train[-n_cal:] if n_cal < n else y_train
                                proba, _ = calibrate_prefit_and_predict_proba(
                                    self.model, cal_X, cal_y, X_test, method=cal_method
                                )
                                proba = sanitize_proba(proba)
                        except Exception as _e:
                            print(f"⚠️ Calibration failed ({cal_method}): {_e}")
                except Exception:
                    try:
                        scores = self.model.decision_function(X_test)
                    except Exception as e:
                        raise RuntimeError(f"Model does not support predict_proba/decision_function: {e}")
                    scores = np.atleast_2d(scores)
                    if scores.ndim == 2 and scores.shape[1] == 1:
                        scores = np.column_stack([-scores, scores])
                    scores = scores - np.max(scores, axis=1, keepdims=True)
                    exp = np.exp(scores)
                    proba = exp / np.sum(exp, axis=1, keepdims=True)

                proba = sanitize_proba(proba)

                # 2) Map to 3-class format (short/flat/long)
                classes_attr = getattr(self.model, "classes_", None)
                proba = np.asarray(proba, dtype=np.float32)
                n_rows = proba.shape[0]
                proba3 = np.full((n_rows, 3), 1e-12, dtype=np.float32)

                if classes_attr is not None and proba.ndim == 2 and proba.shape[1] == len(classes_attr):
                    for j, cls in enumerate(classes_attr):
                        if cls in (0, 1, 2):
                            proba3[:, int(cls)] = proba[:, j]
                else:
                    if proba.ndim == 1:
                        p_long = np.clip(proba, 0.0, 1.0)
                        proba3[:, 2] = p_long
                        proba3[:, 0] = 1.0 - p_long
                    elif proba.ndim == 2 and proba.shape[1] == 2:
                        proba3[:, 0] = np.clip(proba[:, 0], 0.0, 1.0)
                        proba3[:, 2] = np.clip(proba[:, 1], 0.0, 1.0)
                    elif proba.ndim == 2 and proba.shape[1] >= 3:
                        proba3 = np.clip(proba[:, :3], 1e-12, 1.0).astype(np.float32)

                proba3 = np.nan_to_num(proba3, nan=1e-12, posinf=1.0, neginf=1e-12)
                proba3 /= np.maximum(proba3.sum(axis=1, keepdims=True), 1.0)
            
                # --- CV-only calibration metrics (Brier / NLL on test window) ---
                try:
                    if getattr(self, "_in_optuna_cv", False):
                        cfg_eval = getattr(self, "features_config", {}) or {}

                        # Use the same label logic as plain thresholding: sign of forward returns
                        thr = float(cfg_eval.get("label_threshold", label_threshold))

                        # Forward one-bar return on the TEST index used for X_test
                        rets = self.data["returns"].reindex(test_data_scaled.index).astype(float)
                        ret_fwd = rets.shift(-1)

                        mask_valid = np.isfinite(ret_fwd.to_numpy())
                        if mask_valid.any():
                            # label_with_neutral → {0:short, 1:flat, 2:long}
                            y_cal = self.label_with_neutral(ret_fwd[mask_valid], thr).astype(int)
                            proba_cal = proba[mask_valid]

                            brier, nll = compute_brier_and_nll(proba_cal, y_cal)

                            # Expose to the CV aggregator (_single_study_cv)
                            self._last_calib_brier = float(brier)
                            self._last_calib_nll   = float(nll)
                            self._last_calib_n     = int(len(y_cal))

                            if bool(cfg_eval.get("print_cv_debug", False)):
                                print(
                                    f"[CV-Calib/test_strategy] "
                                    f"brier={brier:.6f} | nll={nll:.6f} | n={len(y_cal)}"
                                )
                except Exception as _e:
                    # Never break the evaluation path because of calibration
                    cfg_eval = getattr(self, "features_config", {}) or {}
                    if bool(cfg_eval.get("print_cv_debug", False)):
                        print(f"[CV-Calib/test_strategy] Calibration metric failed: {_e}")

                proba = proba3

                # 3) Optional: fit coverage threshold
                try:
                    cfg_f = getattr(self, "features_config", {}) or {}
                    if is_coverage_intent(cfg_f):
                        # Learn a per-fold coverage→threshold mapping on the calibration tail of TRAIN (CV-safe)
                        try:
                            # target coverage knob (accept both names)
                            _tgt = float(cfg_f.get("target_active_rate",
                                           cfg_f.get("target_coverage",
                                           params.get("target_coverage", params.get("target_active_rate", 0.10)))))
                            # build a small tail from the training matrix used above
                            n_tr = int(getattr(X_train, "shape", [0])[0])
                            ncal = max(200, min(n_tr // 10, 2000)) if n_tr > 0 else 0
                            cal_X = X_train[-ncal:] if (ncal and ncal < n_tr) else X_train
                            cal_y = y_train[-ncal:] if (ncal and ncal < n_tr) else y_train

                            # compute calibrated probabilities on the calibration tail if a method was chosen
                            try:
                                _cal_m = str((params.get("calibrate_method") or cfg_f.get("calibrate_method") or "")).lower()
                            except Exception:
                                _cal_m = ""
                            # NOTE: Don't "calibrate again" here for SVM / already-calibrated estimators.
                           # Coverage threshold only needs stable probs on TRAIN tail.
                            _already_calibrated = False
                            try:
                                from sklearn.calibration import CalibratedClassifierCV
                                _already_calibrated = isinstance(self.model, CalibratedClassifierCV) or hasattr(self.model, "calibrated_classifiers_")
                            except Exception:
                                _already_calibrated = hasattr(self.model, "calibrated_classifiers_")

                            if (_cal_m in ("isotonic", "sigmoid")) and (str(model_type).lower() != "svm") and (not _already_calibrated):
                                p_cal, _ = calibrate_prefit_and_predict_proba(
                                    self.model, cal_X, cal_y, cal_X, method=_cal_m
                                )
                                p_cal = sanitize_proba(p_cal)
                            else:
                                p_cal = sanitize_proba(self.model.predict_proba(cal_X))


                            # map coverage → threshold on this run and stash for aggregation (CV only)
                            coverage_thr = float(fit_coverage_threshold_on_calibration(p_cal, _tgt))
                            self._coverage_conf_thr = float(coverage_thr)

                            # keep last for CV collector (only when actually in CV)
                            _in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            if _in_cv:
                                setattr(self, "_cv_cov_thr_last", float(coverage_thr))
                            try:
                                setattr(self, "_last_cov_cal_rows", int(getattr(cal_X, "shape", [0])[0]))
                            except Exception:
                                pass
                            
                            # ctx label must reflect the actual run context (cv vs real_mX vs eval)
                            _ctx = "cv" if _in_cv else "eval"
                            if not _in_cv:
                                try:
                                    if bool(getattr(self, "_in_real_sim", False)):
                                        mx = int(cfg_f.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                                        _ctx = f"real_m{mx}"
                                except Exception:
                                    pass
                            print(
                                f"[Calib][Coverage] conf_thr={float(coverage_thr):.6f} "
                                f"target_active_rate={float(_tgt):.6f} "
                                f"cal_rows={int(getattr(cal_X, 'shape', [0])[0])} ctx={_ctx}"
                            )
                        except Exception as _ee:
                            print(f"⚠️ Coverage threshold fit skipped in CV: {_ee}")
                except Exception as _e:
                    print(f"[Calib] Classical coverage threshold skipped: {_e}")

                # Coverage should be based on **trade intent**, not certainty about "flat".
                if proba.shape[1] >= 3:
                    p_short = proba[:, 0]
                    p_long  = proba[:, 2]
                    max_conf = np.maximum(p_short, p_long)
                    decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                    raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                else:
                    max_conf    = proba.max(axis=1)
                    raw_classes = proba.argmax(axis=1)
                    decoded_raw = np.where(raw_classes == 1, 1, -1)  # best-effort for 2-class

                cfg_f = getattr(self, "features_config", {}) or {}
                conf0 = float(cfg_f.get("confidence_threshold", confidence_threshold))
                try:
                    if is_coverage_intent(cfg_f) and hasattr(self, "_coverage_conf_thr"):
                        conf0 = float(getattr(self, "_coverage_conf_thr"))
                except Exception:
                    pass

                # --- Edge-vs-Cost gating (dynamic threshold) ---
                base_thr = float(self._resolve_conf_thr(conf0))
                self._last_conf_thr_init = float(conf0)

                print(f"[DEBUG][Costs] high_vol_thr_train={high_vol_thr_train} | cfg_high_vol_thr={cfg_f.get('high_vol_thr')}")
                
                # Persist for *all* downstream paths this month (TopN, consensus, cont-metrics)
                try:
                    if high_vol_thr_train is not None:
                        if not hasattr(self, "features_config") or not isinstance(self.features_config, dict):
                            self.features_config = {}
                        self.features_config["high_vol_thr"] = float(high_vol_thr_train)
                        self._high_vol_thr_train = float(high_vol_thr_train)
                except Exception:
                    pass
                
                _cfg_cost = dict(cfg_f)
                if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                    _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)

                _eval_idx = test_data_scaled.index
                rets, sprd, slip = self._get_cost_arrays_aligned(_cost_src, _eval_idx)

                vol_w = int(cfg_f.get("vol_window_bars", 48))
                # --- Train-anchored vol scaling (avoid ex-post test-month stats) ---
                rv_m_tr, rv_s_tr, den_floor_tr = np.nan, np.nan, np.nan
                try:
                    rets_tr = train_data_scaled["returns"].astype(float)
                    rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                    rv_m_tr = float(np.nanmean(rv_tr))
                    rv_s_tr = float(np.nanstd(rv_tr))
                    _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                    if _pos.size > 0:
                        den_floor_tr = float(np.nanmedian(_pos))
                except Exception:
                    pass

                rv = realized_vol(rets, window=vol_w).to_numpy(dtype=np.float32)
                if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                    vol_z = (rv - rv_m_tr) / rv_s_tr
                else:
                    vol_z = np.zeros_like(rv, dtype=np.float32)

                den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                den = np.where(rv > 1e-8, rv, den_floor).astype(np.float32)
                spread_norm = np.divide(sprd, den, out=np.zeros_like(sprd, dtype=np.float32), where=np.isfinite(den))

                min_conf_thr = float(cfg_f.get("min_conf_thr", 0.33))
                max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))

                # safer αβγ defaults (small nudges, not giant jumps)
                a = float(cfg_f.get("alpha_vol_z", 0.004))
                b = float(cfg_f.get("beta_spread_norm", 0.008))
                g = float(cfg_f.get("gamma_slip_norm", 0.004))

                # cap the drivers (prevents spikes from blowing up thr)
                vol_z_cap = float(cfg_f.get("vol_z_cap", 6.0))
                spread_norm_cap = float(cfg_f.get("spread_norm_cap", 5.0))
                slip_ratio_cap = float(cfg_f.get("slip_ratio_cap", 6.0))

                vol_z = np.clip(vol_z, -vol_z_cap, vol_z_cap).astype(np.float32)
                spread_norm = np.clip(spread_norm, 0.0, spread_norm_cap).astype(np.float32)

                # slippage normalization (make denominator never tiny, and cap ratio)
                slip_norm_bps = float(cfg_f.get("slip_norm_bps", 0.25))
                min_slip_norm_bps = float(cfg_f.get("min_slip_norm_bps", 0.05))
                slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

                slip_norm = np.clip(slip / slip_norm_bps, 0.0, slip_ratio_cap).astype(np.float32)

                thr_vec = np.clip(
                    base_thr + a*vol_z + b*spread_norm + g*slip_norm,
                    min_conf_thr,
                    max_conf_thr
                ).astype(np.float32)

                if self._is_debug():
                    print(f"[Gate✔] Dynamic αβγ active | base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                        f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}")

            
                # --- IMPORTANT: restrict gating to ELIGIBLE bars only ---
                # Otherwise you "target coverage" on bars you later drop (warmup/session/anchor),
                # which can yield 0 trades in real-sim despite nonzero signals pre-mask.
                try:
                    _eval_mask = np.asarray(eval_mask, dtype=bool)
                    if _eval_mask.size != decoded_raw.size:
                        _eval_mask = np.ones(decoded_raw.size, dtype=bool)
                except Exception:
                    _eval_mask = np.ones(decoded_raw.size, dtype=bool)
                _eval_idx = np.flatnonzero(_eval_mask)
                            
                # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                try:
                    tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                    band  = float(cfg_f.get("runtime_active_band_margin", 0.05))
                    win_k = int(cfg_f.get("runtime_coverage_window", 96))
                    step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                    # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                    # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                    try:
                        band = float(band)
                        step = float(step)
                    except Exception:
                        band, step = 0.0, 0.0
                    band = max(0.0, band)
                    step = abs(step)
                    if band > 0.0 and step > 0.5 * band:
                        _step_old = step
                        step = max(1e-6, 0.5 * band)
                        try:
                            if bool(getattr(self, "debug", False)):
                                log_print(
                                    f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                    level="COMPACT",
                                )
                        except Exception:
                            pass

                    n = int(_eval_idx.size)
                    if win_k > 1 and n > 0:
                        _pre = decoded_raw[_eval_idx].copy()
                        _mask0 = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                        _pre[_mask0] = 0
                        _act = (_pre != 0).astype(np.float32)

                        if n >= win_k:
                            _cs = np.cumsum(np.insert(_act, 0, 0.0))
                            _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                            _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                        else:
                            _roll = np.full(n, np.nan, dtype=np.float32)

                        _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                        _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                        _drift = np.nan_to_num(_drift, nan=0.0)
                        min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                        max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                        _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                        thr_vec[_eval_idx] = np.clip(
                            thr_vec[_eval_idx] + _drift,
                            min_conf_thr,
                            max_conf_thr
                        ).astype(np.float32)
                        if self._is_debug():
                            print(f"[Gate✔] Coverage nudge active | target={tgt:.2f} band=±{band:.2f} step={step:.3f}")
                except Exception as _e:
                    print(f"[Gate] Coverage nudge skipped (deep-3D): {_e}")


                self._last_conf_thr_used = float(np.nanmedian(thr_vec))

                # Apply confidence filter ONLY on eligible bars (others are forced flat)
                final_preds = np.zeros_like(decoded_raw)
                if _eval_idx.size > 0:
                    final_preds[_eval_idx] = decoded_raw[_eval_idx]
                    _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                    final_preds[_eval_idx[_mask]] = 0

                if final_preds is None or (final_preds != 0).sum() == 0:
                    if self._is_debug():
                        print("❗ No trades predicted after filtering — penalizing this parameter set.")
                    if in_cv:
                        return _safe_metrics_return(
                            (np.nan,) * N_METRICS,
                            context="test_ensemble_strategy:no_trades_cv",
                        )
                    final_preds = np.zeros_like(decoded_raw, dtype=int)

                eval_index = test_data_scaled.index[eval_mask]
                final_preds = final_preds[eval_mask]
                test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                test_data_for_eval["pred"] = pd.Series(final_preds, index=eval_index).values

                try:
                    if self._is_debug():
                        print("Raw prediction distribution:", pd.Series(raw_classes).value_counts())
                    decoded = decoded_raw
                    if self._is_debug():
                        print("Decoded preds (before confidence filter):", pd.Series(decoded).value_counts())
                        print("Final preds (after confidence filter):", pd.Series(final_preds).value_counts())

                    # --- Store class distributions for CV / mini-block summaries ---
                    raw_counts = _norm_class_counts(pd.Series(raw_classes).value_counts(dropna=False).to_dict())
                    final_counts = _norm_class_counts(pd.Series(final_preds).value_counts(dropna=False).to_dict())
                    self._last_class_dists = {"raw": raw_counts, "final": final_counts}

                    # --- Store confidence stats for aggregated diagnostics ---
                    self._last_conf_stats_label = str(model_type)
                    self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)

                    # Print concise confidence summary (unchanged behaviour)
                    _in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                    _in_real = bool(getattr(self, "_in_real_sim", False))
                    _rt_mx = int(getattr(self, "_rt_month_ix", 0) or 0)
                    _ctx_label = "cv" if _in_cv else (f"real_m{_rt_mx}" if _in_real else "eval")

                    if self._is_debug():
                        print_conf_stats(
                            max_conf,
                            label=f"{str(model_type)}@{_ctx_label}",
                            median_thr=float(getattr(self, "_last_conf_thr_used", float("nan"))),
                        )
                except Exception:
                    # Defensive defaults so summaries never break
                    self._last_class_dists = {"raw": {}, "final": {}}
                    self._last_conf_stats_label = str(model_type)
                    self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)


                    # --- Ensure 'returns' exists and is aligned for ALL branches ---
            # IMPORTANT: do NOT shift returns here. compute_full_evaluation_metrics()
            # applies the one-bar execution delay by shifting the predictions (pred.shift(1)).
            # Keeping returns un-shifted (original series) avoids double-lag.
            test_data_for_eval["returns"] = (
                self.data["returns"].reindex(test_data_for_eval.index).astype(float)
            )
            if test_data_for_eval["returns"].isna().any():
                # Drop rows where we truly have missing returns (end-of-data)
                test_data_for_eval = test_data_for_eval.dropna(subset=["returns"])

            # --- Optional session_flag for evaluation (1 = inside NY session, 0 = outside) ---
            # Right now, because test_data_for_eval is already session-filtered when
            # session_filter_mode includes "test", this will be a column of ones.
            # It becomes meaningful as soon as you stop filtering test data but still
            # want to block NEW entries outside session in compute_full_evaluation_metrics.
            try:
                if hasattr(self, "_ny_mask") and self._ny_mask is not None:
                    _sess = self._ny_mask.reindex(test_data_for_eval.index, fill_value=False)
                    test_data_for_eval["session_flag"] = _sess.astype(int)
                else:
                    # No precomputed mask: treat all bars as tradable (session_flag=1)
                    test_data_for_eval["session_flag"] = 1
            except Exception:
                # Fail-soft: if anything goes wrong (index mismatch, etc.), we just
                # skip session gating in the evaluator.
                test_data_for_eval["session_flag"] = 1

            # --- Edge-bar guard: require the next in-filter bar to be contiguous (no overnight open) ---
            _idx = test_data_for_eval.index
            if len(_idx) >= 2:
                gaps = pd.Series(_idx[1:] - _idx[:-1], index=_idx[:-1])
                exp  = gaps.median()  # ≈ base bar length (auto-infers 15m)
                is_edge = gaps > (exp * 1.5)

                # Audit (debug-only): check whether the edge-bar guard is killing sparse signals
                if self._is_debug():
                    try:
                        _ctx = "cv" if bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False)) else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "run")
                        _edge_idx = is_edge.index[is_edge]
                        _pred_ser = test_data_for_eval["pred"]
                        _nz_before = int((_pred_ser != 0).sum())
                        _nz_edge = int((_pred_ser.reindex(_edge_idx).fillna(0) != 0).sum()) if len(_edge_idx) else 0
                        _nz_last = int(bool(_pred_ser.iloc[-1] != 0))
                        print(f"[EdgeGuardAudit][{model_type}] ctx={_ctx} exp={exp} edge_bars={int(is_edge.sum())} nz_before={_nz_before} nz_on_edge={_nz_edge} nz_last={_nz_last}")
                    except Exception:
                        pass

                # Zero out initiations on edge bars and on the very last row
                test_data_for_eval.loc[is_edge.index[is_edge], "pred"] = 0
                test_data_for_eval.iloc[-1, test_data_for_eval.columns.get_loc("pred")] = 0

                if self._is_debug():
                    try:
                        _nz_after = int((test_data_for_eval["pred"] != 0).sum())
                        print(f"[EdgeGuardAudit][{model_type}] nz_after={_nz_after}")
                    except Exception:
                        pass


            # ----------------------------
            # 5) Evaluation + storage
            # ----------------------------
            # The new dynamic edge-vs-cost gating already adjusts thresholds per-bar
            # using α·vol_z + β·spread_norm + γ·slip_norm, so no extra quantile bump is needed.

            cfg_adj = dict(getattr(self, "features_config", {}) or {})
        
            # Propagate train-anchored high-vol threshold into the evaluation cost layer
            # (prevents LeakageGuard / ex-post thresholding on the eval window)
            try:
                if high_vol_thr_train is not None and cfg_adj.get("high_vol_thr") is None:
                    cfg_adj["high_vol_thr"] = float(high_vol_thr_train)
            except Exception:
                pass


            # 1) Ensure real per-bar costs are attached (no synthetic means)
            try:
                if bool(getattr(self, "trading_costs", True)):
                    _eval_df = locals().get("test_eval_df", None) or locals().get("test_data_for_eval", None)
                    if _eval_df is not None:
                        _eval_df_refreshed = self._ensure_cost_columns(_eval_df, cfg_adj)
                        if _eval_df_refreshed is not None:
                            if _eval_df is locals().get("test_eval_df", None):
                                test_eval_df = _eval_df_refreshed
                            else:
                                test_data_for_eval = _eval_df_refreshed
            except Exception:
                pass

            # 2) Compute after-cost metrics
            # Robust handles: avoid NameError if a partial refactor/merge left variables uninitialized.
            _eval_df = locals().get("test_eval_df", None)
            if _eval_df is None:
                _eval_df = locals().get("test_data_for_eval", None)

            _full_df = locals().get("test_data", None)

            # If we still don't have an eval frame, bail safely with the fixed 16-metric contract.
            if _eval_df is None:
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_eval_frame")
        
            try:
                if _eval_df is not None:
                    _eval_df.attrs["features_config"] = cfg_adj
                    _eval_df.attrs["debug_costs"] = bool(self._is_debug())
            except Exception:
                pass

            _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
            if _in_cv_mode:
                _eval_ctx = "cv:fold_or_month_eval:test_strategy"
            elif bool(getattr(self, "_in_real_sim", False)):
                _eval_ctx = "real_sim:month_eval:test_strategy"
            else:
                _eval_ctx = "eval:test_strategy"
                
            # Telemetry-only: persist the true evaluated bar-grid length.
            # This should match ExecAudit bars (position_exec) for this evaluation context.
            try:
                self._last_eval_bars = int(len(_eval_df)) if _eval_df is not None else 0
            except Exception:
                pass

            metrics = compute_full_evaluation_metrics(
                df=_eval_df,
                trading_costs=self.trading_costs,
                slippage_factor=self.slippage_factor,
                eval_context=_eval_ctx,
            )
            
            # Capture trade-intent precision from evaluator (cheap scalar; safe in CV).
            try:
                _attrs = getattr(_eval_df, "attrs", {}) or {}
                self._last_precision_trade = float(_attrs.get("precision_trade", float("nan")))
                self._last_n_trade_preds = int(_attrs.get("n_trade_preds", 0) or 0)
            except Exception:
                self._last_precision_trade = float("nan")
                self._last_n_trade_preds = 0


            if not _in_cv_mode:
                # normal run — keep final month-level results
                # Safety: keep canonical executed position in `position` (downstream expects it).
                if _eval_df is not None and "position_exec" in _eval_df.columns:
                    try:
                        pos = _eval_df.get("position", None)
                        posx = _eval_df["position_exec"]
                        if pos is None:
                            _eval_df["position"] = posx
                        else:
                            same = np.allclose(
                                pos.fillna(0).to_numpy(dtype=float),
                                posx.fillna(0).to_numpy(dtype=float),
                                atol=1e-12, rtol=0.0
                            )
                            if not same:
                                _eval_df["position"] = posx
                    except Exception:
                        _eval_df["position"] = _eval_df["position_exec"]
                self.results = _eval_df.copy() if _eval_df is not None else None
                self.results_full = _full_df.copy() if _full_df is not None else None
                # clear any CV scratch storage
                self._cv_last_eval_df = None
            else:
                # CV run — expose a lightweight copy for the tuner/CV aggregator
                # Keep only execution + PnL columns to avoid retaining the full feature matrix in RAM.
                if _eval_df is not None and not _eval_df.empty:
                    _keep = [
                        c for c in (
                            "timestamp", "time", "price", "close",
                            "pred", "position", "position_exec",
                            "returns", "strategy", "strategy_exec",
                            "cstrategy", "creturns", "cstrategy_cont", "creturns_cont",
                            "regime_id", "regime_id_diag",
                        )
                        if c in _eval_df.columns
                    ]
                    # If we didn't match any of the keep-cols, keep the full eval df (otherwise diagnostics become all-NaN).
                    self._cv_last_eval_df = _eval_df[_keep].copy() if _keep else _eval_df.copy()
                else:
                    self._cv_last_eval_df = None
                # accumulate per-fold frames (small list kept on the instance only during this CV run)
                try:
                   if self._cv_last_eval_df is not None and self._is_debug():
                        _cap = int(os.environ.get("CV_MAX_EVAL_FRAMES", "5"))
                        if _cap > 0 and len(self._cv_fold_eval_frames) < _cap:
                            self._cv_fold_eval_frames.append(self._cv_last_eval_df.copy())
                except Exception:
                    # defensive: ensure _cv_fold_eval_frames exists and is list-like
                    try:
                        if self._cv_last_eval_df is not None and self._is_debug():
                            self._cv_fold_eval_frames = [self._cv_last_eval_df.copy()]
                        else:
                            self._cv_fold_eval_frames = []
                    except Exception:
                        self._cv_fold_eval_frames = []
                # do not persist fold outputs into the long-lived results (CV-only)
                self.results = None
                self.results_full = None

            # Aggressively free memory
            try:
                del X_test
            except Exception:
                pass
            
            # Drop deep-model tensors/arrays if they exist (no-op for classical models).
            try:
                del X_seq_train
            except Exception:
                pass
            try:
                del y_seq_train
            except Exception:
                pass
            try:
                del X_seq_test
            except Exception:
                pass
            try:
                del y_seq_test
            except Exception:
                pass

            # Drop model reference before clearing TF session to improve release behavior.
            try:
                if getattr(self, "model", None) is not None:
                    self.model = None
            except Exception:
                pass
        
            # Release large engineered feature frames ASAP (train/test df_out stored in _feat_cache).
            # We already clear at function start, but clearing here avoids holding those frames
            # until the *next* call and helps reduce RAM high-water across long runs.
            try:
                self._clear_feature_cache()
            except Exception:
                pass

            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            _gc.collect()

            # ✅ Return standardized, validated metrics
            metrics = _safe_metrics_return(metrics, context="test_strategy")
            return metrics


    def evaluate_strategy_adaptive_top3(
        self,
        best_params: dict,
        train_start,
        train_end,
        test_start,
        test_end,
        hit_thr: float = 0.45,
        window_days: int = 5,
    ):
        """
        Adaptive run over a single test month using Top-3 parameter sets:
        - Start with Top-1 (best_params).
        - If last 5 days' rolling hit-rate < hit_thr, switch to Top-2 from next bar.
        - After >=5 further days, if it fails again, switch to Top-3.
        Skips for DQN + Transformer-XGB-DQN ensemble (evaluate once).
        Prints/logs switch events if features_config['log_switch'] is True (default).
        """
        import pandas as pd
        from copy import deepcopy

        log_switch = bool(getattr(self, "features_config", {}).get("log_switch", True))

        model_type = (best_params or {}).get("model_type", getattr(self, "model_type", ""))
        if model_type in {"dqn"}:
            # excluded families for now → plain full-month eval
            return self.evaluate_strategy(best_params, train_start, train_end, test_start, test_end)

        # Pull Top-3 alternates (back-compat with old __top5_params files)
        top_alts = list(best_params.get("__top3_params") or best_params.get("__top5_params") or [])
        top_alts = top_alts[:2]  # only need 2 alternates (ranks 2 & 3)

        # Normalize alternates: ensure required keys survive the merge
        REQUIRED = ["model_type", "use_extended_features", "lags", "label_threshold", "confidence_threshold"]
        def _merge_params(base, alt):
            p = dict(base); p.update(deepcopy(alt or {}))
            for k in REQUIRED:
                if k not in p and k in base:
                    p[k] = base[k]
            return p

        # ---------- Leg #1: run Top-1 on full month ----------
        m1 = self.evaluate_strategy(best_params, train_start, train_end, test_start, test_end)
        df1 = getattr(self, "results", pd.DataFrame()).copy()
        if df1 is None or df1.empty:
            return m1

        # Eval anchor for monitoring (no look-ahead)
        first_eval_ts = getattr(self, "_expected_eval_start", None) or df1.index[0]

        # Rolling window in bars = window_days × bars_per_day (estimated on actual index)
        bpd = max(1, estimate_bars_per_day(df1.index))
        nwin = int(max(1, window_days) * bpd)

        # Precompute rolling hit-rate series & window counts (for readable prints)
        # ANCHOR: # Precompute rolling hit-rate series & window counts (for readable prints)
        # df1["pred"] is already the executed (shifted) series after compute_full_evaluation_metrics().
        pred_exec1 = df1["pred"]
        active1 = (pred_exec1 != 0).astype(int)
        correct1 = ((pred_exec1 * df1["returns"]) > 0).astype(int) * active1
        act_roll1 = active1.rolling(nwin, min_periods=1).sum()
        hit_roll1 = correct1.rolling(nwin, min_periods=1).sum()
        hr_series1 = (hit_roll1 / act_roll1.replace(0, pd.NA))

        # Find first switch timestamp t1 (if any), using only info up to each bar
        t1 = find_hit_rate_switch_idx(df1.loc[first_eval_ts:], nwin, thr=float(hit_thr), start_ts=first_eval_ts)

        if t1 is None:
            # No switch → keep Top-1 for whole month
            self._last_switch_log = []
            if log_switch:
                print(f"✅ No switch: Top-1 held entire month "
                    f"(window={window_days}d, bars/day≈{bpd}).")
            return m1

        # Compute resume timestamp (next bar after t1)
        idx = df1.index
        try:
            pos = idx.get_loc(pd.to_datetime(t1))
        except Exception:
            pos = max(0, idx.searchsorted(pd.to_datetime(t1)))
        if pos >= len(idx) - 1:
            # Trigger at final bar → effectively no room to switch
            self._last_switch_log = [{"at": str(pd.to_datetime(t1)), "to_rank": 2, "note": "triggered_at_end"}]
            if log_switch:
                # window stats at t1
                t1_ts = pd.to_datetime(t1)
                trades_win = int(act_roll1.loc[:t1_ts].iloc[-1]) if len(act_roll1.loc[:t1_ts]) else 0
                hits_win = int(hit_roll1.loc[:t1_ts].iloc[-1]) if len(hit_roll1.loc[:t1_ts]) else 0
                hr_val = float(hr_series1.loc[:t1_ts].iloc[-1]) if len(hr_series1.loc[:t1_ts]) else float("nan")
                print(f"⚠️ Switch triggered at end-of-month ({t1_ts}) but no bars remain. "
                    f"Window={window_days}d | hit-rate={hr_val:.2%} on {trades_win} trades | hits={hits_win}.")
            self.results = df1
            return m1

        start2 = idx[pos + 1]
        # Stats at t1 for logging
        t1_ts = pd.to_datetime(t1)
        trades_win = int(act_roll1.loc[:t1_ts].iloc[-1]) if len(act_roll1.loc[:t1_ts]) else 0
        hits_win = int(hit_roll1.loc[:t1_ts].iloc[-1]) if len(hit_roll1.loc[:t1_ts]) else 0
        hr_val = float(hr_series1.loc[:t1_ts].iloc[-1]) if len(hr_series1.loc[:t1_ts]) else float("nan")

        if log_switch:
            print(f"🔁 [Switch #1] {t1_ts} → switching to Top-2 "
                f"(window={window_days}d | hit-rate={hr_val:.2%} on {trades_win} active trades "
                f"< {hit_thr:.0%}); retrain_end={t1_ts}, resume={start2}.")

        # ---------- Leg #2: train Top-2 up to t1; test from start2..end ----------
        p2 = _merge_params(best_params, (top_alts[0] if len(top_alts) >= 1 else {}))
        _ = self.evaluate_strategy(p2, train_start, pd.to_datetime(t1), pd.to_datetime(start2), test_end)
        df2 = getattr(self, "results", pd.DataFrame()).copy()
        self._last_switch_log = [{
            "at": str(t1_ts),
            "to_rank": 2,
            "window_days": int(window_days),
            "hit_rate": float(hr_val) if pd.notna(hr_val) else None,
            "trades_in_window": int(trades_win),
            "hits_in_window": int(hits_win),
            "resume_ts": str(pd.to_datetime(start2)),
        }]

        # Decide on second switch (to Top-3) — only after another full window of Top-2 data
        t2 = None
        if len(top_alts) >= 2 and not df2.empty:
            bpd2 = max(1, estimate_bars_per_day(df2.index))
            nwin2 = int(max(1, window_days) * bpd2)

            # Precompute Top-2 rolling stats for readable prints
            pred_exec2 = df2["pred"]
            active2 = (pred_exec2 != 0).astype(int)
            correct2 = ((pred_exec2 * df2["returns"]) > 0).astype(int) * active2
            act_roll2 = active2.rolling(nwin2, min_periods=1).sum()
            hit_roll2 = correct2.rolling(nwin2, min_periods=1).sum()
            hr_series2 = (hit_roll2 / act_roll2.replace(0, pd.NA))

            # enforce cooldown/evidence: require at least one full window before checking
            df2_chk = df2.iloc[nwin2 - 1 :] if len(df2) >= nwin2 else df2.iloc[0:0]
            t2 = find_hit_rate_switch_idx(
                df2_chk,
                nwin2,
                thr=float(hit_thr),
                start_ts=(df2_chk.index[0] if len(df2_chk) else None),
            )

        # Build combined and hard-reset execution at the switch bar(s) to avoid carry-over via pred.shift(1)
        if t2 is None:
            combined = pd.concat([df1.loc[:t1_ts], df2], axis=0).sort_index()
            # reset pred at t1 so first bar of df2 executes with 0 prev position
            if t1_ts in combined.index:
                if "raw_pred" in combined.columns:
                    combined.loc[t1_ts, "raw_pred"] = 0
                combined.loc[t1_ts, "pred"] = 0
            self.results = combined
            return compute_full_evaluation_metrics(
                df=combined,
                trading_costs=self.trading_costs,
                slippage_factor=self.slippage_factor,
            )

        # ---------- Leg #3: train Top-3 up to t2; test from next bar ----------
        idx2 = df2.index
        try:
            pos2 = idx2.get_loc(pd.to_datetime(t2))
        except Exception:
            pos2 = max(0, idx2.searchsorted(pd.to_datetime(t2)))
        if pos2 >= len(idx2) - 1:
            combined = pd.concat([df1.loc[:t1_ts], df2], axis=0).sort_index()
            if t1_ts in combined.index:
                if "raw_pred" in combined.columns:
                    combined.loc[t1_ts, "raw_pred"] = 0
                combined.loc[t1_ts, "pred"] = 0
            self.results = combined
            self._last_switch_log.append({"at": str(pd.to_datetime(t2)), "to_rank": 3, "note": "triggered_at_end"})
            if log_switch:
                t2_ts = pd.to_datetime(t2)
                # window stats at t2 on Top-2 series if available
                if 'hr_series2' in locals() and t2_ts in hr_series2.index:
                    trades2 = int(act_roll2.loc[:t2_ts].iloc[-1])
                    hits2 = int(hit_roll2.loc[:t2_ts].iloc[-1])
                    hr2 = float(hr_series2.loc[:t2_ts].iloc[-1])
                    print(f"⚠️ Switch #2 triggered at end-of-month ({t2_ts}) but no bars remain. "
                        f"Window={window_days}d | hit-rate={hr2:.2%} on {trades2} trades | hits={hits2}.")
            return compute_full_evaluation_metrics(
                df=combined,
                trading_costs=self.trading_costs,
                slippage_factor=self.slippage_factor,
            )

        start3 = idx2[pos2 + 1]
        t2_ts = pd.to_datetime(t2)

        if log_switch:
            # pretty window stats on Top-2 at t2
            if 'hr_series2' in locals() and t2_ts in hr_series2.index:
                trades2 = int(act_roll2.loc[:t2_ts].iloc[-1])
                hits2 = int(hit_roll2.loc[:t2_ts].iloc[-1])
                hr2 = float(hr_series2.loc[:t2_ts].iloc[-1])
            else:
                trades2, hits2, hr2 = 0, 0, float("nan")
            print(f"🔁 [Switch #2] {t2_ts} → switching to Top-3 "
                f"(window={window_days}d | hit-rate={hr2:.2%} on {trades2} active trades "
                f"< {hit_thr:.0%}); retrain_end={t2_ts}, resume={start3}.")

        p3 = _merge_params(best_params, (top_alts[1] if len(top_alts) >= 2 else {}))
        _ = self.evaluate_strategy(p3, train_start, pd.to_datetime(t2), pd.to_datetime(start3), test_end)
        df3 = getattr(self, "results", pd.DataFrame()).copy()
        self._last_switch_log.append({
            "at": str(t2_ts),
            "to_rank": 3,
            "window_days": int(window_days),
            "hit_rate": float(hr2) if pd.notna(hr2) else None,
            "trades_in_window": int(trades2),
            "hits_in_window": int(hits2),
            "resume_ts": str(pd.to_datetime(start3)),
        })

        combined = pd.concat([df1.loc[:t1_ts], df2.loc[:t2_ts], df3], axis=0).sort_index()
        # reset pred at switch bars to prevent execution carry-over across legs
        if t1_ts in combined.index:
            if "raw_pred" in combined.columns:
                combined.loc[t1_ts, "raw_pred"] = 0
            combined.loc[t1_ts, "pred"] = 0
            
        if t2_ts in combined.index:
            if "raw_pred" in combined.columns:
                combined.loc[t2_ts, "raw_pred"] = 0
            combined.loc[t2_ts, "pred"] = 0

        self.results = combined
        return compute_full_evaluation_metrics(
            df=combined,
            trading_costs=self.trading_costs,
            slippage_factor=self.slippage_factor,
        )
        
    def free(self, release_data: bool = False):
        """Aggressively release memory held by this backtester instance.

        - During Optuna CV / repeated trials, call free(release_data=False) so self.data survives.
        - After finishing a model / repeat (when the instance won't be reused), call free(release_data=True).
        """

        # 0) Drop TF/Keras models first (graphs/buffers)
        for _attr in ("model", "cnn", "lstm", "transformer"):
            try:
                if hasattr(self, _attr):
                    delattr(self, _attr)
            except Exception:
                pass

        # 1) Drop run artifacts (safe in CV)
        for _attr in (
            "results", "results_full", "bar_concat",
            "_cv_last_eval_df", "_cv_fold_eval_frames",
            "_ensemble_win_cache", "_seq_cache",
            "_optuna_best_for_wfo", "_optuna_top5_for_wfo", "_optuna_consensus_pool_for_wfo",

        ):
            try:
                if hasattr(self, _attr):
                    delattr(self, _attr)
            except Exception:
                pass

        # 2) Clear per-run caches
        try:
            if hasattr(self, "_feat_cache") and isinstance(self._feat_cache, dict):
                self._feat_cache.clear()
        except Exception:
            pass

        # FeatureBank: safe to clear in CV (it will rebuild)
        for _attr in ("_feature_bank_full", "_feature_bank_meta", "_feature_bank_key", "_feature_bank_src"):
            try:
                if hasattr(self, _attr):
                    delattr(self, _attr)
            except Exception:
                pass

        # 3) Only release the dataset when explicitly requested
        if release_data:
            for _attr in ("data", "data_raw", "raw_data", "df_1h", "df_4h"):
                try:
                    if hasattr(self, _attr):
                        # Prefer setting None to keep attribute shape predictable if anything touches it later
                        setattr(self, _attr, None)
                except Exception:
                    pass

        # 4) Kill cached joblib/loky workers (optional; can be heavy per-trial)
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass

        # 5) Close matplotlib figures
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass

        # 6) Clear DL backend + GC
        _hard_free()



        
        
    def _clear_feature_cache(self):
        """Clear per-run feature cache; keep FeatureBank (it is self-keyed)."""
        try:
            # Main engineered-slice cache
            if hasattr(self, "_feat_cache") and isinstance(self._feat_cache, dict):
                self._feat_cache.clear()

            # Patch 2: per-entry byte accounting (telemetry only)
            if hasattr(self, "_feat_cache_bytes") and isinstance(getattr(self, "_feat_cache_bytes", None), dict):
                self._feat_cache_bytes.clear()

            # Reset optional stats used by [FEAT_CACHE] logging (if present)
            for _k in (
                "_feat_cache_hits",
                "_feat_cache_misses",
                "_feat_cache_est_bytes",      # legacy cumulative counter (if present)
                "_feat_cache_cur_bytes",      # truthful: current retained bytes
                "_feat_cache_evictions",      # Patch 3: eviction counter (if present)
            ):
                if hasattr(self, _k):
                    setattr(self, _k, 0)

        except Exception:
            pass


    def test_ensemble_strategy(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        lags,
        label_threshold,
        ensemble_config,
        model_type,
    ):
        """
        Wrapper around the ensemble backtest to prevent TensorFlow/Keras
        graph/session accumulation across:
          - Optuna CV folds/trials
          - real_trading_simulation months

        Ensembles train deep heads (CNN/LSTM/Transformer) and must receive the
        same cleanup treatment as standalone deep models.
        """
        try:
            # Always safe: cleanup runs on exit via _persist_results_guard()
            # and does not change outputs, only releases memory.
            self._tf_cleanup_do = True
            self._tf_cleanup_del_model = True
        except Exception:
            pass

        # Ensure cleanup runs even on early returns/exceptions inside the core.
        with self._persist_results_guard(persist_results=True):
            return self._test_ensemble_strategy_core(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                lags=lags,
                label_threshold=label_threshold,
                ensemble_config=ensemble_config,
                model_type=model_type,
            )


    def _test_ensemble_strategy_core(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        lags,
        label_threshold,
        ensemble_config,
        model_type,
    ):
        """
        Universal backtest for ensemble models (CNN+LSTM+XGB, Adaptive Regime).
        Handles: feature prep, scaling, labels, windowing, fitting, prediction, metrics.

        Returns
        -------
        tuple[float, ...]   # 16 metrics in fixed order; NaNs/-9999 on bailouts
        """
        # Clear any sticky feature cache from previous evals (ensemble path)
        self._clear_feature_cache()
        
        # ---- tiny local helper if not imported elsewhere ----
        def filter_params(d: dict, prefix: str) -> dict:
            if not isinstance(d, dict):
                return {}
            L = len(prefix)
            return {k[L:]: v for k, v in d.items() if isinstance(k, str) and k.startswith(prefix)}

        # ----------------------------
        # Data selection & preparation
        # ----------------------------
        full_data  = self.data
        train_data = full_data.loc[train_start:train_end]

        true_test_start = pd.to_datetime(test_start)
        test_end        = pd.to_datetime(test_end)
        model_label     = str(model_type or "ensemble")
        
        self._proba_came_from_dqn_fusion = False

        # merge flags/knobs from features_config and ensemble_config (ensemble overrides features)
        cfg_f  = (getattr(self, "features_config", {}) or {}).copy()
        ens_cf = (ensemble_config or {}).copy()
        merged = {**cfg_f, **ens_cf}
        
        # Always use fused path when DQN is present
        merged.setdefault("use_dqn_fusion", True)
        

        # Trace context early (avoid NameError in downstream guardrails/logs)
        in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        
        # ------------------------------------------------------------
        # Mirror test_strategy() defaults/normalization for fairness:
        # 1) Default gating_mode to coverage unless explicitly set
        # 2) In real-sim, if target_active_rate is set but gating_mode is threshold,
        #    auto-switch to coverage so activity-rate tuning actually applies
        # 3) Mirror eval cost knobs (eval_use_trading_costs / trading_costs, slippage_factor)
        # ------------------------------------------------------------
        if "gating_mode" not in merged:
            merged["gating_mode"] = "coverage"


        # Prevent stale coverage thresholds leaking across models/months.
        # Ensures coverage intent without a fresh calibration trips the NaN tripwire.
        try:
            if is_coverage_intent(merged):
                self._coverage_conf_thr = None
                if hasattr(self, "_deep_coverage_thr"):
                    delattr(self, "_deep_coverage_thr")
        except Exception:
            pass

        try:
            in_real = bool(getattr(self, "_in_real_sim", False))
            gmode = str(merged.get("gating_mode", merged.get("gate_mode", "threshold"))).lower()
            tgt = float(merged.get("target_active_rate", merged.get("target_coverage", 0.0)) or 0.0)
            if in_real and (not in_cv) and gmode in ("threshold", "", "none") and tgt > 0.0:
                merged["gating_mode"] = "coverage"
                if self._is_debug():
                    print(f"[Gate] Auto-enabled gating_mode='coverage' for ensemble real-sim (target_active_rate={tgt:.2f}).")
        except Exception:
            pass

        try:
            if not getattr(self, "_trading_costs_locked", False):
                if "eval_use_trading_costs" in merged:
                    self.trading_costs = bool(merged.get("eval_use_trading_costs", self.trading_costs))
                elif "trading_costs" in merged:
                    self.trading_costs = bool(merged.get("trading_costs", self.trading_costs))
        except Exception:
            pass

        try:
            if "slippage_factor" in merged:
                self.slippage_factor = float(merged.get("slippage_factor", self.slippage_factor))
        except Exception:
            pass
        
        # --- Confidence threshold handling (avoid pre-calibration tripwire in CV) ---
        # In ensemble CV, the coverage threshold is computed AFTER fitting (train-tail calibration).
        # So we *do not* call _resolve_conf_thr() here (it would run before calibration and spam
        # TRIPWIRE logs / potentially force a placeholder cap). Instead, we carry the requested
        # default through the pipeline and let the later gating stage prefer the calibrated
        # self._coverage_conf_thr when gating_mode='coverage'.
        default_conf = float(ens_cf.get("confidence_threshold", cfg_f.get("confidence_threshold", 0.50)))
        try:
            merged["confidence_threshold"] = float(default_conf)
        except Exception:
            pass
        try:
            # keep a trace of the requested init threshold for diagnostics
            self._last_conf_thr_init = float(default_conf)
        except Exception:
            pass
        try:
            merged.setdefault("model_type", model_label)
        except Exception:
            pass

        
        # --- Warm-up bars: use the SAME feature config as other models ---
        # For classical/CNN/LSTM/Transformer we base warm-up on the feature pipeline
        # (lags_range, lag_depth, triple-barrier horizon, etc.). Use that here too,
        # and only tag the model_type so compute_required_test_warmup_bars() knows
        # which branch to use.
        cfg_for_warmup = dict(self.features_config or {})
        cfg_for_warmup["model_type"] = model_label

        warmup_need = int(compute_required_test_warmup_bars(cfg_for_warmup))

        # account for final embargo so pre-roll remains outside test month
        embargo_n = int(cfg_f.get("final_embargo_bars", 0) or 0)
        _total_warmup_need = max(0, warmup_need + embargo_n)

        # account for final embargo so pre-roll remains outside test month
        embargo_n = int(cfg_f.get("final_embargo_bars", 0) or 0)
        _total_warmup_need = max(0, warmup_need + embargo_n)

        def _slice_with_warmup(n_extra: int):
            if n_extra <= 0:
                return full_data.loc[true_test_start:test_end]
            idx_before = full_data.index[full_data.index < true_test_start]
            if len(idx_before) == 0:
                return full_data.loc[true_test_start:test_end]
            start_pos = max(0, len(idx_before) - n_extra)
            warmup_start = idx_before[start_pos]
            return full_data.loc[warmup_start:test_end]


        # initial pre-roll (build test_data before any filtering/embargo)
        test_data = _slice_with_warmup(_total_warmup_need)

        sess_mode = str(cfg_f.get("session_filter_mode", "both")).lower()

        if not hasattr(self, "_ny_mask") or self._ny_mask is None:
            try:
                full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                _ny_times = full_idx.tz_convert("America/New_York")
                self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
            except Exception as _e:
                print(f"⚠️ Lazy NY mask build failed in ensemble path: {_e}")
                self._ny_mask = pd.Series(True, index=self.data.index)

        if sess_mode in ("test_only", "both"):
            test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
        if sess_mode in ("train_only", "both"):
            train_data = train_data.loc[self._ny_mask.reindex(train_data.index, fill_value=False)]

        if warmup_need > 0 and len(test_data) > 0:
            have = int((test_data.index < true_test_start).sum())
            if have < _total_warmup_need:
                need_more = _total_warmup_need - have
                test_data = _slice_with_warmup(_total_warmup_need + need_more)
                if sess_mode in ("test_only", "both"):
                    test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]

        # Optional final embargo — disable for CV mini-folds
        try:
            embargo_n = int(cfg_f.get("final_embargo_bars", 0))
            if in_cv:
                embargo_n = 0
            if embargo_n > 0 and len(test_data) > embargo_n:
                test_data = test_data.iloc[embargo_n:].copy()
                print(f"[Embargo] Dropped first {embargo_n} test bars (ensemble, non-CV).")
        except Exception as e:
            print(f"⚠️ final_embargo_bars handling failed (ensemble): {e}")

        use_strict_day1 = bool(self.features_config.get("enforce_day1_start", True))
        
        if getattr(self, "_in_real_sim", False):
            use_strict_day1 = True

        first_eval_ts = (
            pd.to_datetime(true_test_start)
            if bool(getattr(self, "_in_optuna_cv", False))
            else (
                enforce_day1_eval_anchor(test_data.index, true_test_start)
                if use_strict_day1 else
                first_tradable_test_bar(test_data.index, true_test_start)
            )
        )  
        
        if in_cv:
            print(f"[CV/ENSEMBLE] Eval anchor forced to fold start: {first_eval_ts} | ...")

        if in_cv:
            print(f"[CV/ENSEMBLE] Eval anchor forced to fold start: {first_eval_ts} | test_len={len(test_data)} | warmup_need={_total_warmup_need}")

        if first_eval_ts is None:
            print("❌ No tradable bar found in test window (ensemble).")
            # Bail safely with fixed 16-metric contract.
            # IMPORTANT: never persist heavy frames during Optuna CV.
            if in_cv:
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return(
                (np.nan,) * N_METRICS,
                context="test_ensemble_strategy:no_tradable",
            )
        self._expected_eval_start = first_eval_ts

        # ----------------------------
        # Feature engineering + scaling
        # ----------------------------
        cfg = self.apply_feature_defaults()
        lag_depth    = cfg.get("lag_depth", 1)
        roll_windows = cfg.get("roll_windows", [5])
        lags_eff = int(cfg.get("lags_range", cfg.get("lags", lags)))
        if getattr(self, "_is_debug", lambda: False)():
            print(f"[ENSEMBLE] effective_lags={lags_eff} (cfg-precedence)")

        train_data, features = self.prepare_features(
            train_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
        )
        test_data, _ = self.prepare_features(
            test_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
        )

        # De-dup columns
        train_data = train_data.loc[:, ~train_data.columns.duplicated()]
        test_data  = test_data.loc[:,  ~test_data.columns.duplicated()]

        # Replace infs & drop NaNs in *active* features to keep indices aligned
        for df in (train_data, test_data):
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if features:
                df.dropna(subset=features, inplace=True)

        # Scale (fit on train → apply to test)
        train_data, means, stds = self.scale_features(train_data, features)
        test_data,  _,    _     = self.scale_features(test_data,  features, means, stds)

        # ----------------------------
        # Labels (T+1) and alignment
        # ----------------------------
        # _ret_fwd_tr = train_data["returns"].shift(-1)
        # train_data  = train_data.loc[_ret_fwd_tr.notna()].copy()
        # y_train     = self.label_with_neutral(_ret_fwd_tr.loc[train_data.index], threshold=float(label_threshold))

        # _ret_fwd_te = test_data["returns"].shift(-1)
        # test_data   = test_data.loc[_ret_fwd_te.notna()].copy()
        # y_test      = self.label_with_neutral(_ret_fwd_te.loc[test_data.index],  threshold=float(label_threshold))
        # Labels and alignment
        
        # ----------------------------
        # Labels and alignment
        # ----------------------------
        cfg_lbl = dict(merged)

        # pick a close price column for triple-barrier
        if "price" in train_data.columns:
            _price_col = "price"
        elif "mid_close" in train_data.columns:
            _price_col = "mid_close"
        elif "close" in train_data.columns:
            _price_col = "close"
        elif {"ask_close", "bid_close"}.issubset(train_data.columns):
            # build a mid-close if only bid/ask are present
            train_data = train_data.copy()
            test_data  = test_data.copy()
            train_data["__mid_close__"] = (train_data["ask_close"] + train_data["bid_close"]) / 2.0
            test_data["__mid_close__"]  = (test_data["ask_close"]  + test_data["bid_close"])  / 2.0
            _price_col = "__mid_close__"
        else:
            _price_col = None  # only used if triple-barrier is on

        tb_on_lbl = bool(cfg_lbl.get("use_triple_barrier", False))

        # If triple-barrier is requested but we can't resolve a price series, fall back safely.
        if tb_on_lbl and (_price_col is None or (_price_col not in train_data.columns) or (_price_col not in test_data.columns)):
            print("⚠️ TripleBarrier enabled but no price column found; falling back to return-based labels.")
            tb_on_lbl = False

        if tb_on_lbl:
            y_train = triple_barrier_labels(
                close=train_data[_price_col],
                pt_mult=float(cfg_lbl.get("tb_pt_mult", 1.5)),
                sl_mult=float(cfg_lbl.get("tb_sl_mult", 1.0)),
                max_holding=int(cfg_lbl.get("tb_max_holding", 48)),
                neutral_zone=float(cfg_lbl.get("tb_neutral_zone", 0.0)),
                neutral_zone_is_sigma=bool(cfg_lbl.get("tb_neutral_zone_is_sigma", False)),
            ).astype(int)

            y_test = triple_barrier_labels(
                close=test_data[_price_col],
                pt_mult=float(cfg_lbl.get("tb_pt_mult", 1.5)),
                sl_mult=float(cfg_lbl.get("tb_sl_mult", 1.0)),
                max_holding=int(cfg_lbl.get("tb_max_holding", 48)),
                neutral_zone=float(cfg_lbl.get("tb_neutral_zone", 0.0)),
                neutral_zone_is_sigma=bool(cfg_lbl.get("tb_neutral_zone_is_sigma", False)),
            ).astype(int)

            # --- Debug: confirm triple-barrier config & class balance (safe: no `params`) ---
            if bool(cfg_lbl.get("print_labeling_debug", False)):
                from collections import Counter
                _pt = float(cfg_lbl.get("tb_pt_mult", 1.5))
                _sl = float(cfg_lbl.get("tb_sl_mult", 1.0))
                _mh = int(cfg_lbl.get("tb_max_holding", 48))
                _nz = float(cfg_lbl.get("tb_neutral_zone", 0.0))
                print(f"[Labeling] TripleBarrier ON | pt={_pt}×σ sl={_sl}×σ hold={_mh} bars neutral={_nz}")
                print(f"[Labeling] Train counts={dict(Counter(y_train))} | Test counts={dict(Counter(y_test))}")
        else:
            _ret_fwd_tr = train_data["returns"].shift(-1)
            train_data  = train_data.loc[_ret_fwd_tr.notna()].copy()
            y_train     = self.label_with_neutral(_ret_fwd_tr.loc[train_data.index], threshold=float(cfg_lbl.get("label_threshold", 0.0)))

            _ret_fwd_te = test_data["returns"].shift(-1)
            test_data   = test_data.loc[_ret_fwd_te.notna()].copy()
            y_test      = self.label_with_neutral(_ret_fwd_te.loc[test_data.index],  threshold=float(cfg_lbl.get("label_threshold", 0.0)))

        if y_train is None or y_test is None or len(y_train) == 0 or len(y_test) == 0:
            print("⚠️ Labels empty in ensemble strategy. Skipping fold.")
            # Avoid returning stale frames from previous months/folds.
            if in_cv:
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return(
                (np.nan,) * N_METRICS,
                context="test_ensemble_strategy:empty_labels",
            )

        # Basic label diagnostics
        u_tr, c_tr = np.unique(y_train, return_counts=True)
        u_te, c_te = np.unique(y_test,  return_counts=True)
        if self._is_debug():
            print("Label counts (train):", dict(zip(u_tr, c_tr)), f"| thr={label_threshold}")
            print("Label counts (test): ", dict(zip(u_te, c_te)))

        # Directional-only label mix guard for ensemble folds
        if not self._guard_label_mix_directional(
            y_train,
            label_threshold=label_threshold,
            context="ENSEMBLE_FOLD",
            min_dir_samples=5,
        ):
            # Avoid returning stale frames from previous months/folds.
            if in_cv:
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return(
                (np.nan,) * N_METRICS,
                context="test_ensemble_strategy:label_mix_guard",
            )

        # Defragment before attaching labels to reduce pandas fragmentation warnings
        train_data = train_data.copy()
        test_data = test_data.copy()
        train_data["label"] = y_train.astype(int)
        test_data["label"] = y_test.astype(int)

        input_shape = (int(lags_eff), len(features))
        
        high_vol_thr_train = None
        try:
            if "returns" in train_data.columns:
                _cfg_cost_src = getattr(self, "features_config", {}) or {}
                vol_w = int(_cfg_cost_src.get("vol_window_bars", 48))
                qhi   = float(_cfg_cost_src.get("high_vol_q", 0.75))
                _rv_tr = realized_vol(train_data["returns"].astype(float), window=vol_w)
                _rv_tr = _rv_tr.dropna()
                if len(_rv_tr) > 0:
                    high_vol_thr_train = float(_rv_tr.quantile(qhi))
        except Exception:
            high_vol_thr_train = None



        # -------------------------------------
        # Windowing helpers (vectorized & cache)
        # -------------------------------------
        def _fallback_make_windows(df, feats, window):
            X_seq, _, idx = self._create_sliding_windows(df, feats, window_size=int(window))
            X_flat = df[feats].iloc[idx].values
            y_arr  = df["label"].iloc[idx].values
            return (
                X_seq.astype(np.float32, copy=False),
                y_arr.astype(np.int64,  copy=False),
                idx,
                X_flat.astype(np.float32, copy=False),
            )

        if not hasattr(self, "_ensemble_win_cache"):
            self._ensemble_win_cache = {}

        maker = getattr(self, "_ensemble_make_windows", None)
        make_windows = (lambda df: maker(df, features, int(lags_eff))) if callable(maker) \
                    else (lambda df: _fallback_make_windows(df, features, int(lags_eff)))

        # Deep windowing (train / test)
        #
        # IMPORTANT (OOM fix):
        # Apply stride/cap on the *raw bars* before creating sliding windows.
        # Caching full train windows across months is also disabled here (it explodes RAM).
        max_train_windows = ens_cf.get("ensemble_deep_max_train_windows", 10000)
        train_stride = ens_cf.get("ensemble_deep_train_stride", ens_cf.get("ensemble_train_stride", 3))

        try:
            if max_train_windows is not None:
                max_train_windows = int(max_train_windows)
        except Exception:
            max_train_windows = 10000
            
        try:
            train_stride = int(train_stride) if train_stride is not None else 1
        except Exception:
            train_stride = 1
        if train_stride < 1:
            train_stride = 1

        def _start_idx_for_last_strided_windows(n_rows, win, stride, max_windows):
            if max_windows is None:
                return 0
            try:
                max_windows = int(max_windows)
            except Exception:
                return 0
            if max_windows <= 0:
                return 0
            if n_rows <= win:
                return 0
            total_windows = n_rows - win + 1
            if stride <= 1:
                need = min(total_windows, max_windows)
                start_window = total_windows - need
            else:
                total_strided = (total_windows + stride - 1) // stride
                need = min(total_strided, max_windows)
                start_window = (total_strided - need) * stride
            return max(0, int(start_window))

        _start_idx = _start_idx_for_last_strided_windows(len(train_data), int(lags_eff), int(train_stride), max_train_windows)
        if _start_idx > 0:
            train_data = train_data.iloc[_start_idx:].copy()
 
        X_seq_train, y_train_win, idx_train, X_flat_train = make_windows(train_data)
        X_seq_test,  y_test_win,  idx_test,  X_flat_test  = make_windows(test_data)
 

        if X_seq_train.shape[0] == 0 or X_seq_test.shape[0] == 0:
            print("⚠️ Ensemble produced zero windows. Skipping fold.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:zero_windows")

        if self._is_debug():
            print(f"[ENSEMBLE-CV] X_seq_test={getattr(X_seq_test,'shape',None)} "
                f"X_flat_test={getattr(X_flat_test,'shape',None)} y_test_win={getattr(y_test_win,'shape',None)}")

        # Extra safety: apply stride/cap on the already-windowed arrays (train only)
        if train_stride > 1:
            X_seq_train  = X_seq_train[::train_stride]
            X_flat_train = X_flat_train[::train_stride]
            y_train_win  = y_train_win[::train_stride]
        if X_seq_train.shape[0] > max_train_windows:
            X_seq_train  = X_seq_train[-max_train_windows:]
            X_flat_train = X_flat_train[-max_train_windows:]
            y_train_win  = y_train_win[-max_train_windows:]

        # ==============================
        # ENSEMBLE: CNN + LSTM + XGBoost
        # ==============================
        if model_type == "ensemble_cnn_lstm_xgboost":
            cnn_cfg  = filter_params(ens_cf, "cnn_")
            lstm_cfg = filter_params(ens_cf, "lstm_")
            xgb_cfg  = filter_params(ens_cf, "xgb_")

            # CV-friendly caps (fast mode for heads + sane XGB defaults)
            if in_cv:
                # Global deep caps (fallback)
                cv_epochs_global = int(cfg_f.get("deep_cv_max_epochs", 12))
                cv_bs_global     = int(cfg_f.get("deep_cv_batch_size", 256))
                cv_pat_global    = int(cfg_f.get("deep_cv_patience", 6))

                # Extra-strict caps for ensemble heads
                cv_epochs_cnn  = int(cfg_f.get("cnn_ens_cv_max_epochs",  cv_epochs_global))
                cv_epochs_lstm = int(cfg_f.get("lstm_ens_cv_max_epochs", cv_epochs_global))
                cv_bs_cnn      = cv_bs_global
                cv_bs_lstm     = cv_bs_global

                # Base window cap for ensemble heads
                base_max_win = int(ens_cf.get("ensemble_deep_max_train_windows", 10000))
                max_win_cnn  = int(cfg_f.get("cnn_ens_cv_max_train_windows",  base_max_win))
                max_win_lstm = int(cfg_f.get("lstm_ens_cv_max_train_windows", base_max_win))

                # Coarser stride only for CV
                ens_cf.setdefault("ensemble_train_stride", 3)

                # Heads (enable real ES + time caps in CV)
                cnn_cfg = dict(cnn_cfg)
                cnn_cfg.setdefault("eval_mode", "cv_fast")
                cnn_cfg.setdefault("cnn_use_early_stopping", True)
                # clip trial-level epochs/batch to ensemble caps
                cnn_cfg["cnn_epochs"] = min(int(cnn_cfg.get("cnn_epochs", cv_epochs_cnn)), cv_epochs_cnn)
                cnn_cfg["cnn_batch_size"] = min(int(cnn_cfg.get("cnn_batch_size", cv_bs_cnn)), cv_bs_cnn)
                cnn_cfg.setdefault("cnn_patience", cv_pat_global)
                cnn_cfg["deep_max_train_windows"] = min(
                    int(cnn_cfg.get("deep_max_train_windows", base_max_win)),
                    max_win_cnn,
                )

                lstm_cfg = dict(lstm_cfg)
                lstm_cfg.setdefault("eval_mode", "cv_fast")
                lstm_cfg.setdefault("lstm_use_early_stopping", True)
                lstm_cfg["lstm_epochs"] = min(int(lstm_cfg.get("lstm_epochs", cv_epochs_lstm)), cv_epochs_lstm)
                lstm_cfg["lstm_batch_size"] = min(int(lstm_cfg.get("lstm_batch_size", cv_bs_lstm)), cv_bs_lstm)
                lstm_cfg.setdefault("lstm_patience", cv_pat_global)
                lstm_cfg["deep_max_train_windows"] = min(
                    int(lstm_cfg.get("deep_max_train_windows", base_max_win)),
                    max_win_lstm,
                )

                # XGB (modern device semantics + ES)
                xgb_cfg = dict(xgb_cfg)
                xgb_cfg.setdefault("n_estimators", min(int(xgb_cfg.get("n_estimators", 400)), 400))
                xgb_cfg.setdefault("n_jobs", int(xgb_cfg.get("n_jobs", 3)))
                xgb_cfg.setdefault("xgb_eval_fraction", float(xgb_cfg.get("xgb_eval_fraction", 0.10)))
                xgb_cfg.setdefault("xgb_early_stopping_rounds", int(xgb_cfg.get("xgb_early_stopping_rounds", 50)))
                xgb_cfg.setdefault("use_oof_meta", False)
                xgb_cfg.setdefault("oof_splits", 3)

                # Match global XGB policy: env-gated GPU (XGB_USE_GPU=1) else CPU.
                use_gpu = (os.environ.get("XGB_USE_GPU", "0") == "1")
                xgb_cfg.setdefault("tree_method", "hist")
                xgb_cfg.pop("predictor", None)
                if use_gpu:
                    xgb_cfg["device"] = os.environ.get("XGB_DEVICE", "cuda")
                else:
                    xgb_cfg.pop("device", None)
            else:
                # 🔹 Non-CV path (refit + final evaluation): ALWAYS use OOF stacking
                xgb_cfg = dict(xgb_cfg)
                xgb_cfg.setdefault("n_estimators", min(int(xgb_cfg.get("n_estimators", 400)), 400))
                xgb_cfg.setdefault("n_jobs", int(xgb_cfg.get("n_jobs", 3)))
                xgb_cfg.setdefault("xgb_eval_fraction", float(xgb_cfg.get("xgb_eval_fraction", 0.10)))
                xgb_cfg.setdefault("xgb_early_stopping_rounds", int(xgb_cfg.get("xgb_early_stopping_rounds", 50)))
                
                # 🔹 Force OOF ON here regardless of tuned value
                xgb_cfg.setdefault("use_oof_meta", False)
                xgb_cfg.setdefault("oof_splits", 3)
                use_gpu = (os.environ.get("XGB_USE_GPU", "0") == "1")
                xgb_cfg.setdefault("tree_method", "hist")
                xgb_cfg.pop("predictor", None)
                if use_gpu:
                    xgb_cfg["device"] = os.environ.get("XGB_DEVICE", "cuda")
                else:
                    xgb_cfg.pop("device", None)

            # --- Train throttling (applies in both CV and final): stride + tail cap ---
            try:
                max_win = int(ens_cf.get("ensemble_deep_max_train_windows", 10000))
            except Exception:
                max_win = 10000
            try:
                train_stride = max(1, int(ens_cf.get("ensemble_train_stride", 1)))
            except Exception:
                train_stride = 1

            if train_stride > 1:
                X_seq_train  = X_seq_train[::train_stride]
                y_train_win  = y_train_win[::train_stride]
                if X_flat_train is not None:
                    X_flat_train = X_flat_train[::train_stride]

            if X_seq_train.shape[0] > max_win:
                X_seq_train  = X_seq_train[-max_win:]
                y_train_win  = y_train_win[-max_win:]
                if X_flat_train is not None and len(X_flat_train) >= len(y_train_win):
                    X_flat_train = X_flat_train[-max_win:]

            lags_eff_local = max(int(lags_eff), 3)

            self.model = EnsembleCNNLSTMXGBoost(
                cnn_config=cnn_cfg,
                lstm_config=lstm_cfg,
                xgb_config=xgb_cfg,
                input_shape=(lags_eff_local, len(features)),
            )

            try:
                self.model.fit(X_seq_train, X_flat_train, y_train_win)

                # --- Calibration on train-tail (no leakage), do it ONCE per fold ---
                try:
                    # merge feature + ensemble cfg (ens_cf overrides)
                    cfg = {**(getattr(self, "features_config", {}) or {}), **(ens_cf or {})}
                    use_temp = bool(cfg.get("deep_calibrate", False)) and \
                               str(cfg.get("deep_calibration_method", "")).lower() == "temperature"
                    need_cov = is_coverage_intent(cfg)

                    # IMPORTANT: set CV calibration defaults on *cfg* (ens_cf.setdefault here was too late)
                    if in_cv:
                        cfg.setdefault("deep_calibration_frac", 0.05)
                        cfg.setdefault("deep_calibration_min_samples", 300)

                    # Only compute if missing (reuse across blocks)
                    in_cv_flag = bool(getattr(self, "_in_optuna_cv", False))
                    # Temperature scaling: keep OFF in CV (extra overhead, not needed for activity control).
                    # In eval (real testing), recompute each window after retrain to avoid stale state.
                    must_cal_temp = (not in_cv_flag) and bool(use_temp)
                    # Coverage threshold: MUST be available in CV or Optuna's target_active_rate is ignored.
                    # Also recompute per evaluation window (train window changes month-to-month).
                    must_cal_cov  = bool(need_cov)

                    if must_cal_temp or must_cal_cov:
                        frac = float(cfg.get("deep_calibration_frac", 0.10))
                        nmin = int(cfg.get("deep_calibration_min_samples", 500))
                        nwin = int(X_seq_train.shape[0])
                        ncal = min(
                            max(nmin, int(round(nwin * max(0.01, min(frac, 0.99))))),
                            (nwin - 1)
                        ) if nwin > 1 else 0

                        if ncal >= 50:
                            X_tail_seq  = X_seq_train[-ncal:]
                            X_tail_flat = X_flat_train[-ncal:] if X_flat_train is not None else None
                            y_tail      = y_train_win[-ncal:].astype(int)
                            p_tail = self.model.predict_proba(X_tail_seq, X_tail_flat)
                            p_tail = sanitize_proba(p_tail)

                            if must_cal_temp:
                                self._deep_temp_T = float(fit_temperature_from_proba(p_tail, y_tail))
                                if self._is_debug():
                                    print(f"[Calib] Ensemble (CNN+LSTM+XGB) T={self._deep_temp_T:.3f} on {len(y_tail)} cal windows.")
                                p_tail = apply_temperature_to_proba(p_tail, self._deep_temp_T)
                                
                            if must_cal_cov:
                                _tgt = float(cfg.get("target_active_rate", cfg.get("target_coverage", 0.10)))
                                try:
                                    _mc = np.asarray(p_tail, dtype=float).max(axis=1)
                                    _mc = _mc[np.isfinite(_mc)]
                                    _q = np.quantile(_mc, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
                                    _in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                                    _ctx = "cv" if _in_cv else (
                                        ("real_m" + str(int(getattr(self, "_rt_month_ix", 0) or 0)))
                                        if bool(getattr(self, "_in_real_sim", False))
                                        else "eval"
                                    )
                                    print(
                                        f"[ConfDist][CalTail][{_ctx}][CLX] std={np.std(_mc):.4f} iqr={(_q[2]-_q[0]):.4f} "
                                        f"p50={_q[1]:.3f} p75={_q[2]:.3f} p90={_q[3]:.3f} p95={_q[4]:.3f} p99={_q[5]:.3f}"
                                    )
                                except Exception:
                                    pass

                                self._coverage_conf_thr = float(fit_coverage_threshold_on_calibration(p_tail, _tgt))

                                # Always log (COMPACT-safe), not only debug.
                                print(
                                    f"[Calib][Coverage][CLX] conf_thr={float(self._coverage_conf_thr):.6f} "
                                    f"target_active_rate={float(_tgt):.6f} cal_rows={int(len(p_tail))} ctx={_ctx}"
                                )
                                setattr(self, "_deep_coverage_thr", float(self._coverage_conf_thr))


                except Exception as _e:
                    print(f"[Calib] Ensemble (CNN+LSTM+XGB) calibration skipped: {_e}")

                # Predict on test
                proba = self.model.predict_proba(X_seq_test, X_flat_test)

            finally:
                try:
                    if hasattr(self.model, "free"):
                        self.model.free()
                except Exception:
                    pass
                try:
                    import gc as _gc
                    tf.keras.backend.clear_session()
                    _gc.collect()
                except Exception:
                    pass

        # ===========================
        # ENSEMBLE: ADAPTIVE REGIME
        # ===========================
        elif model_type == "ensemble_adaptive_regime":
            # Extract configs for the adaptive regime ensemble
            lstm_cfg   = filter_params(ens_cf, "lstm_")
            rf_cfg     = filter_params(ens_cf, "rf_")
            logit_cfg  = filter_params(ens_cf, "logit_")

            if in_cv:
                # Global deep CV caps (fallback)
                cv_epochs_global = int(cfg_f.get("deep_cv_max_epochs", 12))
                cv_bs_global     = int(cfg_f.get("deep_cv_batch_size", 256))
                cv_pat_global    = int(cfg_f.get("deep_cv_patience", 6))

                # Extra-strict caps for the LSTM head inside the adaptive ensemble.
                lstm_ens_epochs_cap = int(cfg_f.get("lstm_ens_cv_max_epochs", cv_epochs_global))
                base_max_win        = int(ens_cf.get("ensemble_deep_max_train_windows", 10000))
                lstm_ens_win_cap    = int(cfg_f.get("lstm_ens_cv_max_train_windows", base_max_win))

                # Coarser stride in CV only (final WFO retrain ignores in_cv).
                ens_cf.setdefault("ensemble_train_stride", 3)

                # LSTM head (clip epochs / batch / windows to ensemble-specific caps)
                lstm_cfg = dict(lstm_cfg)
                lstm_cfg.setdefault("lstm_use_early_stopping", True)
                lstm_cfg["lstm_epochs"] = min(
                    int(lstm_cfg.get("lstm_epochs", lstm_ens_epochs_cap)),
                    lstm_ens_epochs_cap,
                )
                lstm_cfg["lstm_batch_size"] = min(
                    int(lstm_cfg.get("lstm_batch_size", cv_bs_global)),
                    cv_bs_global,
                )
                lstm_cfg.setdefault("lstm_patience", cv_pat_global)
                lstm_cfg["deep_max_train_windows"] = min(
                    int(lstm_cfg.get("deep_max_train_windows", base_max_win)),
                    lstm_ens_win_cap,
                )

                # RF and Logistic heads are cheap; keep their existing CV caps/logic.
                rf_cfg = dict(rf_cfg)
                rf_cfg.setdefault("n_estimators", min(int(rf_cfg.get("n_estimators", 400)), 400))
                rf_cfg.setdefault("max_depth", 8)
                rf_cfg.setdefault("min_samples_leaf", 20)
                rf_cfg.setdefault("n_jobs", int(rf_cfg.get("n_jobs", 3)))
                rf_cfg.setdefault("random_state", int(rf_cfg.get("random_state", 42)))

                logit_cfg = dict(logit_cfg)
                logit_cfg.setdefault("penalty", "l2")
                logit_cfg.setdefault("C", 1.0)
                logit_cfg.setdefault("solver", "lbfgs")
                logit_cfg.setdefault("max_iter", 200)
                logit_cfg.setdefault("n_jobs", int(logit_cfg.get("n_jobs", 3)))

            # sanitize class_weight for LogisticRegression
            cw = logit_cfg.get("class_weight", logit_cfg.get("logit_class_weight", None))
            try:
                if isinstance(cw, float) and (np.isnan(cw) or np.isinf(cw)):
                    cw = None
            except Exception:
                pass
            if isinstance(cw, str):
                s = cw.strip().lower()
                cw = None if s in ("", "none", "null", "nan") else ("balanced" if s == "balanced" else None)
            elif (cw not in (None, "balanced")) and (not isinstance(cw, dict)):
                cw = None
            if cw is not None:
                logit_cfg["class_weight"] = cw
            else:
                logit_cfg.pop("class_weight", None)

            if not hasattr(self, "_ensemble_win_cache"):
                self._ensemble_win_cache = {}
            maker = getattr(self, "_ensemble_make_windows", None)
            make_windows = (lambda df: maker(df, features, int(lags_eff))) if callable(maker) \
                        else (lambda df: _fallback_make_windows(df, features, int(lags_eff)))

            # Deep windowing (train / test)
            #
            # IMPORTANT (OOM fix):
            # Apply stride/cap on the *raw bars* before creating sliding windows,
            # and do NOT cache full train windows across folds/months.
            max_train_windows = ens_cf.get("ensemble_deep_max_train_windows", 10000)
            train_stride = ens_cf.get("ensemble_deep_train_stride", ens_cf.get("ensemble_train_stride", 3))

            try:
                if max_train_windows is not None:
                    max_train_windows = int(max_train_windows)
            except Exception:
                max_train_windows = 10000
                
            try:
                train_stride = int(train_stride) if train_stride is not None else 1
            except Exception:
                train_stride = 1
            if train_stride < 1:
                train_stride = 1

            def _start_idx_for_last_strided_windows(n_rows, win, stride, max_windows):
                if max_windows is None:
                    return 0
                try:
                    max_windows = int(max_windows)
                except Exception:
                    return 0
                if max_windows <= 0:
                    return 0
                if n_rows <= win:
                    return 0
                total_windows = n_rows - win + 1
                if stride <= 1:
                    need = min(total_windows, max_windows)
                    start_window = total_windows - need
                else:
                    total_strided = (total_windows + stride - 1) // stride
                    need = min(total_strided, max_windows)
                    start_window = (total_strided - need) * stride
                return max(0, int(start_window))

            _start_idx = _start_idx_for_last_strided_windows(len(train_data), int(lags_eff), int(train_stride), max_train_windows)
            if _start_idx > 0:
                train_data = train_data.iloc[_start_idx:].copy()

            X_seq_train, y_train_win, idx_train, X_flat_train = make_windows(train_data)
            X_seq_test,  y_test_win,  idx_test,  X_flat_test  = make_windows(test_data)

            if X_seq_train.shape[0] == 0 or X_seq_test.shape[0] == 0:
                print("⚠️ Ensemble (adaptive) produced zero windows. Skipping fold.")
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:adaptive_zero_windows")

            # Extra safety: apply stride/cap on the already-windowed arrays (train only)
            idx_train_np = np.asarray(idx_train, dtype=int)

            if train_stride > 1:
                idx_stride = np.arange(0, X_seq_train.shape[0], train_stride, dtype=int)
                y_stride   = y_train_win[idx_stride]
                u_s, c_s   = np.unique(y_stride, return_counts=True)

                try:
                    MIN_CLASS_SAMPLES_ENSEMBLE = int(
                        getattr(self, "ensemble_min_class_samples", 3)
                    )
                except Exception:
                    MIN_CLASS_SAMPLES_ENSEMBLE = 3

                if (len(u_s) < 2) or (c_s.min() < MIN_CLASS_SAMPLES_ENSEMBLE):
                    # Class-aware stride downsampling:
                    # keep all minority-class windows, stride only the majority,
                    # and top-up any class that would otherwise vanish.
                    try:
                        y_full = np.asarray(y_train_win)
                        u_f, c_f = np.unique(y_full, return_counts=True)
                        counts_full = {int(k): int(v) for k, v in zip(u_f, c_f)}

                        # Classes that are "rare" in the full set (we keep all of them)
                        rare_classes = [int(k) for k, v in zip(u_f, c_f) if int(v) < MIN_CLASS_SAMPLES_ENSEMBLE]
                        if rare_classes:
                            idx_keep_rare = np.nonzero(np.isin(y_full, rare_classes))[0].astype(int)
                            idx_sel = np.unique(np.concatenate([idx_stride, idx_keep_rare]).astype(int))
                        else:
                            idx_sel = np.unique(idx_stride.astype(int))
                        idx_sel.sort()

                        # Top-up any class so it doesn't drop below min(desired, available)
                        sel_set = set(idx_sel.tolist())
                        y_sel = y_full[idx_sel] if idx_sel.size else np.asarray([], dtype=y_full.dtype)
                        for cls, full_cnt in zip(u_f, c_f):
                            cls_i = int(cls)
                            full_cnt_i = int(full_cnt)
                            desired = min(int(MIN_CLASS_SAMPLES_ENSEMBLE), full_cnt_i)
                            cur = int(np.sum(y_sel == cls_i)) if y_sel.size else 0
                            if cur < desired:
                                need = desired - cur
                                candidates = np.nonzero(y_full == cls_i)[0].astype(int)
                                added = []
                                for j in candidates:
                                    if int(j) not in sel_set:
                                        sel_set.add(int(j))
                                        added.append(int(j))
                                        if len(added) >= need:
                                            break
                                if added:
                                    idx_sel = np.unique(np.concatenate([idx_sel, np.asarray(added, dtype=int)]))
                                    idx_sel.sort()
                                    y_sel = y_full[idx_sel]

                        # Only accept if we did not lose any class that exists in full y
                        u_post, c_post = np.unique(y_sel, return_counts=True) if y_sel.size else (np.array([]), np.array([]))
                        if (len(u_post) == len(u_f)) and (len(u_post) >= 2):
                            counts_post = {int(k): int(v) for k, v in zip(u_post, c_post)}
                            print(
                                f"[Ensemble-Adapt][Stride] Applied class-aware stride downsampling "
                                f"(stride={train_stride}) kept_rare={rare_classes} "
                                f"full={counts_full} post={counts_post} sel={len(idx_sel)}/{len(y_full)}"
                            )
                            X_seq_train  = X_seq_train[idx_sel]
                            X_flat_train = X_flat_train[idx_sel]
                            y_train_win  = y_sel
                            idx_train_np = idx_train_np[idx_sel]
                        else:
                            counts_s = {int(k): int(v) for k, v in zip(u_s, c_s)}
                            print(
                                f"[Ensemble-Adapt][Stride] Disabled stride downsampling (stride={train_stride}) "
                                f"full={counts_full} post_stride={counts_s}"
                            )
                    except Exception:
                        counts_s = {int(k): int(v) for k, v in zip(u_s, c_s)}
                        print(
                            f"[Ensemble-Adapt][Stride] Disabled stride downsampling (stride={train_stride}) "
                            f"post_stride={counts_s}"
                        )
                else:
                    X_seq_train  = X_seq_train[idx_stride]
                    X_flat_train = X_flat_train[idx_stride]
                    y_train_win  = y_stride
                    idx_train_np = idx_train_np[idx_stride]


            if X_seq_train.shape[0] > max_train_windows:
                X_seq_train  = X_seq_train[-max_train_windows:]
                X_flat_train = X_flat_train[-max_train_windows:]
                y_train_win  = y_train_win[-max_train_windows:]
                idx_train_np = idx_train_np[-max_train_windows:]

            # Regime features
            adx_col_req = str(ens_cf.get("adx_col", "adx_14"))
            vol_col_req = str(ens_cf.get("vol_col", "rolling_std_20"))

            def _resolve_regime_col(req, df_a, df_b, kind):
                # Pick a regime feature column that exists in BOTH train and test frames.
                if (req in df_a.columns) and (req in df_b.columns):
                    return req

                inter = [c for c in df_a.columns if c in df_b.columns]
                low = {str(c).lower(): c for c in inter}
                req_low = str(req).lower()
                if req_low in low:
                    return low[req_low]

                if kind == "adx":
                    pool = [c for c in inter if "adx" in str(c).lower()]
                else:
                    pool = [c for c in inter if "rolling_std" in str(c).lower()]
                    if not pool:
                        pool = [c for c in inter if ("realized" in str(c).lower() and "vol" in str(c).lower())
                                or str(c).lower().startswith("vol") or "_vol" in str(c).lower()]
                    if not pool:
                        pool = [c for c in inter if str(c).lower().startswith("atr") or "_atr" in str(c).lower()]

                if pool:
                    import re as _re
                    def _num(s):
                        m = _re.findall(r"\d+", str(s))
                        return int(m[0]) if m else None
                    tgt = _num(req)
                    if tgt is not None:
                        scored = []
                        for c in pool:
                            v = _num(c)
                            scored.append((abs(v - tgt) if v is not None else 10**9, str(c)))
                        scored.sort(key=lambda x: x[0])
                        best = scored[0][1]
                        for c in pool:
                            if str(c) == best:
                                return c
                    return pool[0]

                return req

            adx_col = _resolve_regime_col(adx_col_req, train_data, test_data, "adx")
            vol_col = _resolve_regime_col(vol_col_req, train_data, test_data, "vol")

            if (adx_col != adx_col_req) or (vol_col != vol_col_req):
                print(
                    f"[Ensemble-Adapt][RegimeCols] adx_col={adx_col} (req={adx_col_req}) "
                    f"vol_col={vol_col} (req={vol_col_req})"
                )

            missing = []
            if (adx_col not in train_data.columns) or (adx_col not in test_data.columns):
                missing.append(adx_col)
            if (vol_col not in train_data.columns) or (vol_col not in test_data.columns):
                missing.append(vol_col)
            if missing:
                print(
                    f"[Ensemble-Adapt][REGIME][WARN] Missing regime cols {missing} in train/test; "
                    "regime switching will default to 'sideways'."
                )
                # Last resort: create constant columns to keep the pipeline running.
                for col in missing:
                    if col not in train_data.columns:
                        train_data[col] = 0.0
                    if col not in test_data.columns:
                        test_data[col] = 0.0


            regime_source_train = train_data[[adx_col, vol_col]].iloc[idx_train_np]
            regime_source_test  = test_data[[adx_col, vol_col]].iloc[idx_test]

            self.model = AdaptiveRegimeStrategy(
                lstm_config=lstm_cfg,
                rf_config=rf_cfg,
                logit_config=logit_cfg,
                input_shape=input_shape,
                adx_col=adx_col,
                vol_col=vol_col,
                adx_thresh=float(ens_cf.get("adx_thresh", 25)),
                vol_thresh=float(ens_cf.get("vol_thresh", 0.002)),
                adx_thresh_q=(
                    float(ens_cf.get("adx_thresh_q", 0.70))
                    if bool(ens_cf.get("train_lstm_on_trend_only", True))
                    else None
                ),
                train_lstm_on_trend_only=bool(ens_cf.get("train_lstm_on_trend_only", True)),
            )
            try:
                idx_end_pos = np.arange(len(X_seq_train), dtype=int)
                self.model.fit(
                    X_seq_train, X_flat_train, y_train_win,
                    X_flat_with_regime=regime_source_train,
                    idx_end=idx_end_pos,
                )
                
                
                # Coverage calibration for AdaptiveRegime ensemble (mirrors test_strategy coverage gating)
                try:
                    cfg_cal = dict(merged)
                    need_cov_cal = is_coverage_intent(cfg_cal)
                    if in_cv:
                        cfg_cal.setdefault("deep_calibration_frac", 0.05)
                        cfg_cal.setdefault("deep_calibration_min_samples", 300)
                    if need_cov_cal:
                        frac = float(cfg_cal.get("deep_calibration_frac", 0.10))
                        nmin = int(cfg_cal.get("deep_calibration_min_samples", 500))
                        nwin = int(X_seq_train.shape[0])
                        ncal = min(
                            max(nmin, int(round(nwin * max(0.01, min(frac, 0.99))))),
                            (nwin - 1)
                        ) if nwin > 1 else 0
                        if ncal >= 50:
                            X_tail_seq  = X_seq_train[-ncal:]
                            X_tail_flat = X_flat_train[-ncal:] if X_flat_train is not None else None
                            y_tail      = y_train_win[-ncal:].astype(int)
                            rs_tail = None
                            try:
                                rs_tail = regime_source_train.iloc[-ncal:]
                            except Exception:
                                rs_tail = None
                            p_tail = self.model.predict_proba(X_tail_seq, X_tail_flat, regime_source=rs_tail)
                            p_tail = sanitize_proba(p_tail)
                            _tgt = float(cfg_cal.get("target_active_rate", cfg_cal.get("target_coverage", 0.10)))
                            self._coverage_conf_thr = float(fit_coverage_threshold_on_calibration(p_tail, _tgt))
                            setattr(self, "_deep_coverage_thr", float(self._coverage_conf_thr))
                            try:
                                setattr(self, "_last_cov_cal_rows", int(len(p_tail)))
                            except Exception:
                                pass
                            _in_cv2 = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            if _in_cv2:
                                _ctx = "cv"
                            elif bool(getattr(self, "_in_real_sim", False)):
                                _ctx = "real_m" + str(int(getattr(self, "_rt_month_ix", 0) or 0))
                            else:
                                _ctx = "eval"
                            print(
                                f"[Calib][Coverage][AR] conf_thr={float(self._coverage_conf_thr):.6f} "
                                f"target_active_rate={float(_tgt):.6f} cal_rows={int(len(p_tail))} ctx={_ctx}"
                            )
                except Exception as _e:
                    print(f"[Calib] Ensemble (adaptive) coverage calibration skipped: {_e}")

                
                proba = self.model.predict_proba(
                    X_seq_test, X_flat_test, regime_source=regime_source_test
                )
            finally:
                try:
                    import gc as _gc
                    tf.keras.backend.clear_session()
                    _gc.collect()
                except Exception:
                    pass

        else:
            raise ValueError(f"Unknown ensemble model_type: {model_type}")

        # ---------------------------------------
        # Generic postprocessing for all ensembles
        # ---------------------------------------
        if proba is None or (hasattr(proba, "__len__") and len(proba) == 0):
            print("❌ Ensemble produced no probabilities.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:no_probabilities")


        proba = np.asarray(proba, dtype=np.float32)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, np.zeros_like(proba), proba], axis=1)
        elif proba.shape[1] == 2:
            neutral = 1.0 - proba.sum(axis=1, keepdims=True)
            proba = np.hstack([proba[:, :1], neutral, proba[:, 1:2]])

        proba = np.nan_to_num(proba, nan=1e-6, posinf=1.0, neginf=0.0)
        row_sums = np.clip(proba.sum(axis=1, keepdims=True), 1e-9, None)
        proba = proba / row_sums
        
        # Apply temperature only if NOT fused with DQN
        if hasattr(self, "_deep_temp_T") and not bool(getattr(self, "_proba_came_from_dqn_fusion", False)):
            try:
                proba = apply_temperature_to_proba(proba, float(getattr(self, "_deep_temp_T")))
            except Exception:
                pass

        # Coverage should be based on **trade intent**, not certainty about "flat".
        if proba.shape[1] >= 3:
            p_short = proba[:, 0]
            p_long  = proba[:, 2]
            max_conf = np.maximum(p_short, p_long)
            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
        else:
            max_conf    = proba.max(axis=1)
            raw_classes = proba.argmax(axis=1)
            decoded_raw = np.where(raw_classes == 1, 1, -1)  # best-effort for 2-class

        # --- Edge-vs-Cost gating (dynamic; align on ensemble window ends = idx_test) ---
        cfg_gate = dict(merged or {})
        # If using coverage gating, start from the calibrated coverage threshold (fold-safe).
        _gmode = str(cfg_gate.get("gating_mode", cfg_gate.get("gate_mode", "threshold"))).lower()
        base_thr = None
        if _gmode == "coverage":
            try:
                _bt = getattr(self, "_coverage_conf_thr", None)
                if _bt is not None and np.isfinite(float(_bt)):
                    base_thr = float(_bt)
            except Exception:
                base_thr = None
        if base_thr is None:
            base_thr = float(self._resolve_conf_thr(
                float(cfg_gate.get("confidence_threshold", 0.0))
            ))
        _cfg_cost = dict(cfg_gate)
        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
        _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)
        _all_idx = test_data.index
        
        rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)
        
        vol_w = int(cfg_gate.get("vol_window_bars", 48))
        rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)

        rv_m, rv_s = np.nan, np.nan
        den_floor = np.nan
        try:
            if "returns" in train_data.columns:
                _rv_tr = realized_vol(train_data["returns"].astype(float), window=vol_w).to_numpy(dtype=np.float32)
                _rv_tr_f = _rv_tr[np.isfinite(_rv_tr)]
                if _rv_tr_f.size >= 50:
                    rv_m = float(np.nanmean(_rv_tr_f))
                    rv_s = float(np.nanstd(_rv_tr_f))
                    _pos = _rv_tr_f[_rv_tr_f > 0]
                    if _pos.size > 0:
                        den_floor = float(np.nanmedian(_pos))
        except Exception:
            pass
        
        # IMPORTANT (causality): do NOT fall back to test-window stats.
        # If TRAIN stats are unusable, neutralize the volatility term and use a constant denom floor.
        if (not np.isfinite(rv_m)) or (not np.isfinite(rv_s)) or (rv_s <= 0):
            vol_z_all = np.zeros_like(rv_all, dtype=np.float32)
        else:
            vol_z_all = ((rv_all - rv_m) / rv_s).astype(np.float32)

        den_floor = den_floor if (np.isfinite(den_floor) and den_floor > 1e-8) else 1e-6
        den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
            
        spread_norm_all = np.divide(sprd_all, den_all, out=np.zeros_like(sprd_all, dtype=np.float32), where=np.isfinite(den_all))
        
        a = float(cfg_gate.get("alpha_vol_z", 0.004))
        b = float(cfg_gate.get("beta_spread_norm", 0.008))
        g = float(cfg_gate.get("gamma_slip_norm", 0.004))
        slip_norm_bps = float(cfg_gate.get("slip_norm_bps", 0.25))
        min_slip_norm_bps = float(cfg_gate.get("min_slip_norm_bps", 0.05))
        slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

        vol_z_cap = float(cfg_gate.get("vol_z_cap", 6.0))
        spread_norm_cap = float(cfg_gate.get("spread_norm_cap", 5.0))
        slip_ratio_cap = float(cfg_gate.get("slip_ratio_cap", 6.0))
        max_conf_thr = float(cfg_gate.get("max_conf_thr", 0.90))

        vol_z_all = np.clip(vol_z_all, -vol_z_cap, vol_z_cap)
        spread_norm_all = np.clip(spread_norm_all, 0.0, spread_norm_cap)
        slip_ratio = np.clip(slip_all / slip_norm_bps, 0.0, slip_ratio_cap)

        thr_full = np.clip(base_thr + a*vol_z_all + b*spread_norm_all + g*slip_ratio, 0.0, max_conf_thr).astype(np.float32)
        idx_test_arr = np.asarray(idx_test, dtype=int)
        thr_vec = thr_full[idx_test_arr]
        if self._is_debug():
            print(f"[Gate✔] Dynamic αβγ active | base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                        f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}")
            
        # ------------------------------------------------------------
        # IMPORTANT: define the TRUE eval-universe for windows
        # (windows whose END time is on/after the eval anchor).
        # The nudge/gating must only use this universe; otherwise it
        # adapts using warmup windows that are later discarded.
        # ------------------------------------------------------------
        try:
            keep_win = (test_data.index[idx_test_arr] >= self._expected_eval_start)
            _eval_idx = np.flatnonzero(keep_win)
        except Exception:
            keep_win = np.zeros(len(idx_test_arr), dtype=bool)
            _eval_idx = np.asarray([], dtype=int)

        if _eval_idx.size == 0:
            print("❌ No tradable test windows in ensemble after start cut.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:no_eval_windows")

        # --- Soft coverage-drift nudge (mirrors test_strategy runtime control) ---
        try:
            tgt   = float(merged.get("target_active_rate", merged.get("target_coverage", 0.10)))
            band  = float(merged.get("runtime_active_band_margin", 0.05))
            win_k = int(merged.get("runtime_coverage_window", 96))
            step  = float(merged.get("runtime_conf_nudge", 0.01))
            n_nudge = int(_eval_idx.size)
            if n_nudge <= 0:
                raise ValueError("no_eval_windows_for_nudge")


            # Rolling-quantile cap (prevents "bunched confidence" → near-zero trades)
            _low = max(0.0, tgt - band)
            allow_qcap = bool(merged.get("runtime_allow_rolling_qcap", True))
            if allow_qcap and win_k > 1 and n_nudge >= win_k:
                try:
                    _dr = np.asarray(decoded_raw, dtype=int)[_eval_idx]
                    _mc = np.asarray(max_conf, dtype=np.float32)[_eval_idx]
                    _tv = np.asarray(thr_vec, dtype=np.float32)[_eval_idx].copy()
                    _act0 = ((_dr != 0) & (_mc >= _tv)).astype(np.float32)
                    if float(np.nanmean(_act0)) < _low:
                        _q = (
                            pd.Series(_mc)
                              .rolling(win_k, min_periods=win_k)
                              .quantile(1.0 - tgt)
                              .shift(1)  # causal: use past only
                              .to_numpy(dtype=np.float32)
                        )
                        _m = np.isfinite(_q)
                        if _m.any():
                            _tv[_m] = np.minimum(_tv[_m], _q[_m])
                            thr_vec[_eval_idx] = _tv
                            if self._is_debug():
                                print(
                                    f"[Gate✔] Rolling-quantile cap active | q={1.0 - tgt:.3f} "
                                    f"win={win_k} | thr_med={float(np.nanmedian(thr_vec[_eval_idx])):.3f}"
                                )
                except Exception:
                    pass

            # Rolling activity drift control: nudge thresholds up/down to stay inside band
            if win_k > 1 and n_nudge >= win_k:
                _dr = np.asarray(decoded_raw, dtype=int)[_eval_idx]
                _mc = np.asarray(max_conf, dtype=np.float32)[_eval_idx]
                _tv = np.asarray(thr_vec, dtype=np.float32)[_eval_idx]
                _act = ((_dr != 0) & (_mc >= _tv)).astype(np.float32)
                _cs = np.cumsum(np.insert(_act, 0, 0.0))
                _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                _low2, _high2 = max(0.0, tgt - band), min(1.0, tgt + band)
                _drift = np.where(_roll < _low2, -step, np.where(_roll > _high2, step, 0.0)).astype(np.float32)
                _drift = np.nan_to_num(_drift, nan=0.0)
                max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))
                min_conf_thr = float(cfg_f.get("min_conf_thr", 0.33))

                thr_vec[:n_nudge] = np.clip(
                    thr_vec[:n_nudge] + _drift[:n_nudge],
                    min_conf_thr,
                    max_conf_thr
                ).astype(np.float32)
        except Exception as _e:
            if self._is_debug():
                print(f"[Gate] Coverage nudge skipped (ensemble): {_e}")

        # Save median threshold for reporting / safety-ladder reference
        conf_thr_final = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))

        self._last_conf_thr_used = conf_thr_final
        # Apply gating ONLY on eval windows; force flat elsewhere
        final_preds = np.zeros_like(decoded_raw, dtype=int)
        if _eval_idx.size > 0:
            final_preds[_eval_idx] = np.asarray(decoded_raw, dtype=int)[_eval_idx]
            _mask = (np.asarray(max_conf, dtype=np.float32)[_eval_idx] < np.asarray(thr_vec, dtype=np.float32)[_eval_idx])
            final_preds[_eval_idx[_mask]] = 0

        # safety: if gating nuked all trades, step down the threshold
        _allow_backoff = bool(
            merged.get(
                "allow_conf_backoff_cv" if in_cv else "allow_conf_backoff_eval",
                False
            )
        )
        if (final_preds != 0).sum() == 0 and _allow_backoff:
            for t in [0.50, 0.33, 0.20, 0.10, 0.00]:
                if t >= self._last_conf_thr_used:
                    continue
                tmp = np.zeros_like(decoded_raw, dtype=int)
                if _eval_idx.size > 0:
                    tmp[_eval_idx] = np.asarray(decoded_raw, dtype=int)[_eval_idx]
                    tmp[_eval_idx[np.asarray(max_conf, dtype=np.float32)[_eval_idx] < t]] = 0
                if (tmp != 0).sum() > 0:
                    print(f"[Backoff] lowered conf_thr → {t:.2f}; active_rate={np.mean(tmp!=0):.3f}")
                    final_preds = tmp
                    self._last_conf_thr_used = float(t)
                    conf_thr_final = float(t)
                    break
        # quick trace
        print(f"[ENSEMBLE-CV] trades={int((final_preds != 0).sum() if final_preds is not None else 0)} "
            f"at thr={float(getattr(self, '_last_conf_thr_used', 0.0)):.4f}")

        if final_preds is None or (final_preds != 0).sum() == 0:
            if in_cv:
                return _safe_metrics_return(
                    (np.nan,) * N_METRICS,
                    context="test_ensemble_strategy:no_trades_cv",
                )
            final_preds = np.zeros_like(decoded_raw, dtype=int)

        # -------- Start-cut before building result_df (FIX: index/pred alignment) --------
        _mask_keep = np.asarray(keep_win, dtype=bool)
        if not _mask_keep.any():
            print("❌ No tradable test windows in ensemble after start cut.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:deep2d_no_trades")

        # apply mask first, then build aligned result df
        idx_test_masked = idx_test_arr[_mask_keep]
        final_preds     = final_preds[_mask_keep]
        try:
            proba = proba[_mask_keep]
        except Exception:
            pass

        test_index = test_data.index[idx_test_masked]
        result_df  = test_data.loc[test_index].copy()
        result_df["pred"] = final_preds

        # ------------------------------------------------------
        # EXPAND RESULT FRAME TO THE FULL EVAL WINDOW (like others)
        # ------------------------------------------------------
        try:
            # Evaluation should start at _expected_eval_start (day-1 anchor) and
            # end at test_end, same as in test_strategy. Build that index from
            # the master data so we don't silently skip early-month bars.
            full_eval_index = (
                self.data.loc[self._expected_eval_start:test_end].index
            )

            # Keep a copy of the narrow frame to preserve attrs, then reindex.
            base_result = result_df.copy()
            result_df = base_result.reindex(full_eval_index)

            # Where the ensemble did not produce a window / prediction,
            # treat it as "no position" (0) rather than dropping the bar.
            result_df["pred"] = result_df.get("pred", 0).fillna(0).astype(int)

            # Preserve any attrs (features_config, etc.) that were on the
            # narrower frame before reindexing.
            for k, v in getattr(base_result, "attrs", {}).items():
                result_df.attrs.setdefault(k, v)
                
            # Attach model-consistent regimes for diagnostics (primarily used
            # by AdaptiveRegimeStrategy). This ensures the per-regime CV table
            # reflects the same regime logic used inside the strategy rather
            # than heuristic `regime_id_diag` reconstruction.
            try:
                _m = getattr(self, "model", None)
                if "regime_id" not in result_df.columns and (_m is not None) and hasattr(_m, "infer_regime_ids"):
                    _cols = []
                    _adx_col = getattr(_m, "adx_col", None)
                    _vol_col = getattr(_m, "vol_col", None)
                    if _adx_col and _adx_col in self.data.columns:
                        _cols.append(_adx_col)
                    if _vol_col and _vol_col in self.data.columns:
                        _cols.append(_vol_col)
                    if _cols:
                        _rs = self.data[_cols].reindex(result_df.index)
                        result_df["regime_id"] = _m.infer_regime_ids(_rs)
                        print(f"[RegimeDiag] Attached regime_id for diagnostics | cols={_cols} | rows={len(result_df)}")
            except Exception:
                pass
        except Exception as _e:
            print(f"⚠️ Ensemble reindex to full eval window failed, using narrow frame: {_e}")
            # fall back to the original result_df


        # Ensure required columns exist/aligned
        if "spread" not in result_df.columns:
            result_df["spread"] = 0.0

        if "returns" not in result_df.columns:
            # try align from full dataset; else compute from price/close
            if hasattr(self, "data") and isinstance(getattr(self, "data"), pd.DataFrame) and "returns" in self.data.columns:
                result_df["returns"] = self.data["returns"].reindex(result_df.index).astype(float)
            else:
                # last-ditch: build simple returns from price/close if available
                px = None
                for cand in ("price", "close", "Price", "Close"):
                    if cand in result_df.columns:
                        px = result_df[cand].astype(float)
                        break
                if px is not None:
                    result_df["returns"] = px.pct_change().fillna(0.0)
                else:
                    # nothing to do; place zeros so metrics are defined
                    result_df["returns"] = 0.0
                    
        # --- Edge-bar guard for ensembles ---
        _idx = result_df.index
        if len(_idx) >= 2:
            gaps = pd.Series(_idx[1:] - _idx[:-1], index=_idx[:-1])
            exp  = gaps.median()
            is_edge = gaps > (exp * 1.5)

            if self._is_debug():
                try:
                    _ctx = "cv" if bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False)) else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "run")
                    _edge_idx = is_edge.index[is_edge]
                    _pred_ser = result_df["pred"]
                    _nz_before = int((_pred_ser != 0).sum())
                    _nz_edge = int((_pred_ser.reindex(_edge_idx).fillna(0) != 0).sum()) if len(_edge_idx) else 0
                    _nz_last = int(bool(_pred_ser.iloc[-1] != 0))
                    print(f"[EdgeGuardAudit][{model_type}] ctx={_ctx} exp={exp} edge_bars={int(is_edge.sum())} nz_before={_nz_before} nz_on_edge={_nz_edge} nz_last={_nz_last}")
                except Exception:
                    pass

            result_df.loc[is_edge.index[is_edge], "pred"] = 0
            result_df.iloc[-1, result_df.columns.get_loc("pred")] = 0

            if self._is_debug():
                try:
                    _nz_after = int((result_df["pred"] != 0).sum())
                    print(f"[EdgeGuardAudit][{model_type}] nz_after={_nz_after}")
                except Exception:
                    pass



        # Drop rows without returns
        result_df = result_df.dropna(subset=["returns"])

        # If no rows or no active signals, return a clean no-trades tuple (non-CV),
        # but keep CV behavior so folds can be pruned upstream.
        def _no_trades_tuple():
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    float(result_df["returns"].std(ddof=0)) if len(result_df) else 0.0,
                    0.0)

        if len(result_df) == 0:
            if getattr(self, "_in_optuna_cv", False):
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:empty_result_df_cv")

            print("ℹ️ Ensemble evaluation window empty after cleaning. Returning no-trades metrics.")
            return _no_trades_tuple()

        # Heuristic for “no activity”: all zeros (or single class that maps to flat)
        # If your pipeline uses {0,1,2}, adjust this if 1/2 map to long/short.
        has_activity = (result_df["pred"] != 0).any()
        if not has_activity:
            if in_cv:
                # Let CV callers penalize this split; they check trades == 0 and score -9999.
                return _safe_metrics_return(
                    (np.nan,) * N_METRICS,
                    context="test_ensemble_strategy:no_activity_cv",
                )

            # Non-CV: keep going so compute_full_evaluation_metrics produces flat curves + attrs.
            if self._is_debug():
                print("ℹ️ [Ensemble] No trades in this window; computing flat metrics.")
            return _no_trades_tuple()

        # ----------------------------
        # Evaluation + return
        # ----------------------------
        
        # Attach per-bar trading-cost columns on ensembles too
        try:
            if bool(getattr(self, "trading_costs", True)):
                _cfg_cost2 = dict(merged)
                if high_vol_thr_train is not None and _cfg_cost2.get("high_vol_thr") is None:
                    _cfg_cost2["high_vol_thr"] = float(high_vol_thr_train)
                result_df = self._ensure_cost_columns(result_df, _cfg_cost2)
        except Exception:
            pass
        
        # Ensure ensemble evaluation can use the same execution overlays (TWAP / kill-switch / etc.)
        # as single-model evaluation (they are driven by df.attrs["features_config"]).
        try:
            
            # Make the stored snapshot truthful: record the final operative threshold actually used.
            try:
                if 'conf_thr_final' in locals() and np.isfinite(float(conf_thr_final)):
                    merged["confidence_threshold"] = float(conf_thr_final)
                    merged["confidence_threshold_used"] = float(conf_thr_final)
            except Exception:
                pass
            result_df.attrs["features_config"] = dict(merged)
            result_df.attrs["debug_costs"] = bool(self._is_debug())
        except Exception:
            pass

        _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        if _in_cv_mode:
            _eval_ctx = "cv:fold_or_month_eval:test_ensemble_strategy"
        elif bool(getattr(self, "_in_real_sim", False)):
            _eval_ctx = "real_sim:month_eval:test_ensemble_strategy"
        else:
            _eval_ctx = "eval:test_ensemble_strategy"

        metrics = compute_full_evaluation_metrics(
            df=result_df,
            trading_costs=self.trading_costs,
            slippage_factor=self.slippage_factor,
            eval_context=_eval_ctx,
        )
        
        # Capture trade-intent precision from evaluator (cheap scalar; safe in CV).
        try:
            _attrs = getattr(result_df, "attrs", {}) or {}
            self._last_precision_trade = float(_attrs.get("precision_trade", float("nan")))
            self._last_n_trade_preds = int(_attrs.get("n_trade_preds", 0) or 0)
        except Exception:
            self._last_precision_trade = float("nan")
            self._last_n_trade_preds = 0


        # Keep canonical executed position in `position` (downstream expects it).
        try:
            if result_df is not None and "position_exec" in result_df.columns:
                result_df["position"] = result_df["position_exec"]
        except Exception:
            pass
        
        # --- Proactive cleanup to avoid cumulative RAM growth across CV folds ---
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc, time
        gc.collect()
        time.sleep(0.05)

        
        if getattr(self, "_in_optuna_cv", False):
            self._cv_last_eval_df = (
                result_df.copy() if result_df is not None and not result_df.empty else None
            )

            # Only accumulate per-fold frames in debug mode.
            # In normal runs we avoid keeping all folds in memory to prevent
            # RAM drift across Optuna trials.
            try:
                if self._cv_last_eval_df is not None and self._is_debug():
                    try:
                        self._cv_fold_eval_frames.append(self._cv_last_eval_df.copy())
                        #   R3: hard-cap stored CV fold frames to prevent RAM drift in debug runs
                        try:
                            _max_keep = int((getattr(self, 'config', {}) or {}).get('cv_max_fold_eval_frames', 3) or 3)
                        except Exception:
                            _max_keep = 3
                        if _max_keep > 0 and len(self._cv_fold_eval_frames) > _max_keep:
                            self._cv_fold_eval_frames = self._cv_fold_eval_frames[-_max_keep:]
                    except AttributeError:
                        # First time in this process: create the list
                        self._cv_fold_eval_frames = [self._cv_last_eval_df.copy()]
            except Exception:
                if self.debug:
                    self._log("⚠️ Failed to append CV fold eval frame", level="warning")
                    
            self.results = None
            self.results_full = None
        else:
            # Non-CV: persist evaluated frame for downstream plotting/exports.
            self.results = result_df.copy() if result_df is not None else None
            self.results_full = self.results
            self._cv_last_eval_df = None


        # Best-effort TF/GC cleanup between runs (non-invasive)
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()

        metrics = _safe_metrics_return(metrics, context="eval_block_1")
        return metrics


    def test_dqn_strategy(self, train_start, train_end, test_start, test_end, lags, dqn_config: dict | None = None):
        """
        Train a DQN agent on the training window and evaluate on the test window.
        Returns a fixed-length metrics tuple from `compute_full_evaluation_metrics`.
        """

        # 1) Lazy-load feature config for DQN path ONLY if no features_config was provided already
        if not hasattr(self, "_dqn_features_config_set"):
            if not isinstance(getattr(self, "features_config", None), dict) or not self.features_config:
                with open(FEATURES_PATH, "r") as f:
                    self.features_config = json.load(f)
            self._dqn_features_config_set = True

        # B1: ensure DEFAULT_FEATURES fill missing execution knobs for DQN path too
        # (compute_full_evaluation_metrics reads from df.attrs["features_config"])
        try:
            self.apply_feature_defaults()
        except Exception:
            pass

        # Clear any sticky feature cache between DQN runs
        self._clear_feature_cache()

        # 2) Prepare data (apply same session filter + embargo used elsewhere)
        full_data  = self.data
        train_data = full_data.loc[train_start:train_end]

        true_test_start = pd.to_datetime(test_start)
        test_end        = pd.to_datetime(test_end)
        warmup_need     = int(compute_required_test_warmup_bars({
            **self.features_config, "model_type": "dqn", "dqn_config": (dqn_config or {})
        }))
        embargo_n = int(self.features_config.get("final_embargo_bars", 0) or 0)
        _total_warmup_need = max(0, warmup_need + embargo_n)

        def _slice_with_warmup(n_extra: int):
            if n_extra <= 0:
                return full_data.loc[true_test_start:test_end]
            idx_before = full_data.index[full_data.index < true_test_start]
            if len(idx_before) == 0:
                return full_data.loc[true_test_start:test_end]
            start_pos = max(0, len(idx_before) - n_extra)
            warmup_start = idx_before[start_pos]
            return full_data.loc[warmup_start:test_end]

        # initial pre-roll (build test_data before any filtering/embargo)
        test_data = _slice_with_warmup(_total_warmup_need)

        sess_mode = str(self.features_config.get("session_filter_mode", "both")).lower()

        if not hasattr(self, "_ny_mask") or self._ny_mask is None:
            try:
                full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                _ny_times = full_idx.tz_convert("America/New_York")
                self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
            except Exception as _e:
                print(f"⚠️ Lazy NY mask build failed: {_e}")
                self._ny_mask = pd.Series(True, index=self.data.index)

        # NEW semantics:
        # - "both":        filter train + test
        # - "test_only":   filter test only
        # - "train_only":  filter train only
        if sess_mode in ("test_only", "both"):
            test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
        if sess_mode in ("train_only", "both"):
            train_data = train_data.loc[self._ny_mask.reindex(train_data.index, fill_value=False)]

        if warmup_need > 0 and len(test_data) > 0:
            have = int((test_data.index < true_test_start).sum())
            if have < _total_warmup_need:
                need_more = _total_warmup_need - have
                test_data = _slice_with_warmup(_total_warmup_need + need_more)
                if sess_mode in ("test_only", "both"):
                    test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]

        try:
            embargo_n = int(self.features_config.get("final_embargo_bars", 0))
            if bool(getattr(self, "_in_optuna_cv", False)):
                embargo_n = 0
            if embargo_n > 0 and len(test_data) > embargo_n:
                test_data = test_data.iloc[embargo_n:].copy()
                print(f"[Embargo] Dropped first {embargo_n} test bars (DQN, non-CV).")
        except Exception as e:
            print(f"⚠️ final_embargo_bars handling failed (DQN): {e}")

        use_strict_day1 = bool(self.features_config.get("enforce_day1_start", True))
        if getattr(self, "_in_real_sim", False):
            use_strict_day1 = True

        first_eval_ts = (
            pd.to_datetime(true_test_start)
            if bool(getattr(self, "_in_optuna_cv", False))
            else (
                enforce_day1_eval_anchor(test_data.index, true_test_start)
                if use_strict_day1 else
                first_tradable_test_bar(test_data.index, true_test_start)
            )
        )
        if bool(getattr(self, "_in_optuna_cv", False)):
            print(f"[CV/DQN] Eval anchor forced to fold start: {first_eval_ts} | test_len={len(test_data)} | warmup_need={_total_warmup_need}")

        if first_eval_ts is None:
            print("❌ No tradable bar found in test window (DQN).")
            if bool(getattr(self, "_in_optuna_cv", False)):
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_dqn_strategy:no_tradable_bar")

        self._expected_eval_start = first_eval_ts

        # Feature knobs
        cfg          = self.features_config
        lag_depth    = cfg.get("lag_depth", 1)
        roll_windows = cfg.get("roll_windows", [5])
        
        # DQN env knobs live in features_config by default.
        # Allow safe overrides from dqn_config for reward processing + DQN env costs.
        cfg_env = dict(cfg or {})
        try:
            _dcfg = dict(dqn_config or {})
            for _k in (
                "env_reward_clip", "env_reward_tanh_k", "env_reward_clip_range",
                "env_reward_norm", "env_reward_norm_beta",
                "env_cost_scale_dqn", "env_turnover_penalty_dqn",
            ):
                if _k in _dcfg:
                    cfg_env[_k] = _dcfg[_k]
        except Exception:
            cfg_env = dict(cfg or {})


        # Build features (train): persist the exact feature list for DQN consistency
        train_data, features = self.prepare_features(train_data, int(lags), lag_depth=lag_depth, roll_windows=roll_windows)
        train_data = train_data.loc[:, ~train_data.columns.duplicated()]
        if features:
            train_data = train_data.dropna(subset=features).copy()

        self.dqn_feature_list = features

        high_vol_thr_train = None
        try:
            if "returns" in train_data.columns:
                _cfg_cost_src = getattr(self, "features_config", {}) or {}
                vol_w = int(_cfg_cost_src.get("vol_window_bars", 48))
                qhi   = float(_cfg_cost_src.get("high_vol_q", 0.75))
                _rv_tr = realized_vol(train_data["returns"].astype(float), window=vol_w)
                _rv_tr = _rv_tr.dropna()
                if len(_rv_tr) > 0:
                    high_vol_thr_train = float(_rv_tr.quantile(qhi))
        except Exception:
            high_vol_thr_train = None

        # Attach cost columns on the TRAIN slice (no leakage; rewards only)
        try:
            if bool(getattr(self, "trading_costs", True)):
                cfg_cost = dict(getattr(self, "features_config", {}) or {})
                if high_vol_thr_train is not None and cfg_cost.get("high_vol_thr") is None:
                    cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                train_data = self._ensure_cost_columns(train_data.copy(), cfg_cost)
        except Exception as _e:
            print(f"[DQN-cost] Skipped adding cost columns on train_data: {_e}")

        # Build features (test)
        test_data_full, _ = self.prepare_features(test_data, int(lags), lag_depth=lag_depth, roll_windows=roll_windows)
        test_data_full = test_data_full.loc[:, ~test_data_full.columns.duplicated()]
        if features:
            test_data_full = test_data_full.dropna(subset=features).copy()

        test_data_features = test_data_full[self.dqn_feature_list].copy()
        
        # Local DQN config (must be applied to env wrappers + training)
        try:
            cfg_local = dict(dqn_config or {})
        except Exception:
            cfg_local = {}


        # 3) Initialize environment (window=lags, state_size=len(features))
        # IMPORTANT (validity): if we attach CostAwareWrapper (spread/slippage arrays),
        # disable the env's internal fixed slippage penalty to avoid double-charging costs.
        spr = None
        slp = None
        _will_use_cost_wrapper = False
        try:
            if bool(getattr(self, "trading_costs", True)):
                spr = train_data.get("spread", None)
                slp = train_data.get("slippage_bps", None)
                _will_use_cost_wrapper = (spr is not None) and (slp is not None)
        except Exception:
            spr = None
            slp = None
            _will_use_cost_wrapper = False

        _env_slippage = 0.0 if _will_use_cost_wrapper else float(getattr(self, "slippage_factor", 0.0))
        env = TradingEnv(train_data, features, slippage=_env_slippage, window=int(lags))

        # Optional: cost-aware reward using spread/slippage arrays from the TRAIN slice
        if _will_use_cost_wrapper:
            try:
                spr_arr = spr.to_numpy(dtype=np.float32, copy=False)
                slp_arr = slp.to_numpy(dtype=np.float32, copy=False)

                cfg_cost_env = getattr(self, "features_config", {}) or {}
                cost_scale = float(cfg_local.get("env_cost_scale_dqn", cfg_cost_env.get("env_cost_scale_dqn", 1.5)))
                turnover_penalty = float(cfg_local.get("env_turnover_penalty_dqn", cfg_cost_env.get("env_turnover_penalty_dqn", 0.0)))

                # Provide mid-price for spread->return conversion if available
                px_arr = None
                try:
                    for _c in ("mid_close", "close", "price"):
                        if _c in train_data.columns:
                            px_arr = train_data[_c].to_numpy(dtype=np.float32, copy=False)
                            break
                except Exception:
                    px_arr = None
                    
                env = CostAwareWrapper(
                    env,
                    spread=spr_arr,
                    slippage_bps=slp_arr,
                    mid_price=px_arr,
                    cost_scale=cost_scale,
                    turnover_penalty=turnover_penalty,
                )
                if self._is_debug():
                    print(
                        f"[DQN-cost] CostAwareWrapper attached (n={spr_arr.shape[0]}), "
                        f"cost_scale={cost_scale}, turnover_penalty={turnover_penalty}."
                    )
            except Exception as _e:
                print(f"[DQN-cost] Skipped CostAwareWrapper: {_e}")

        # Prefer dqn_config, fall back to features_config
        cfg_env = dict(getattr(self, "features_config", {}) or {})
        cfg_env.update(cfg_local or {})

        rw_clip = cfg_env.get("env_reward_clip", None)              # None|"tanh"|"range"
        rw_tk   = float(cfg_env.get("env_reward_tanh_k", 3.0))
        rw_rng  = tuple(cfg_env.get("env_reward_clip_range", (-1.0, 1.0)))
        rw_norm = bool(cfg_env.get("env_reward_norm", True))
        rw_b    = float(cfg_env.get("env_reward_norm_beta", 0.99))

        if self._is_debug():
            print(f"[DQN-reward] clip={rw_clip} tanh_k={rw_tk} range={rw_rng} norm={rw_norm} beta={rw_b}")

        if rw_clip or rw_norm:
            env = RewardProcessWrapper(
                env,
                clip_mode=rw_clip,
                tanh_k=rw_tk,
                clip_range=rw_rng,
                norm=rw_norm,
                norm_beta=rw_b,
            )

        if not hasattr(env, "reset"):
            raise AttributeError("TradingEnv has no reset(). Update rl/environment.py or import the correct class.")

        print(f"[DQN] Env ready | window={env.window} | feature_dim={len(features)}")

        input_shape = (len(features),)
        # Validity: agent window must match env.window (lags). Override any mismatch.
        _w_cfg = cfg_local.get("window", None)
        cfg_local["window"] = int(lags)
        if _w_cfg is not None:
            try:
                if int(_w_cfg) != int(lags) and self._is_debug():
                    print(f"[DQN] Overriding dqn_config.window={_w_cfg} -> {int(lags)} to match lags/env.window.")
            except Exception:
                pass
        cfg_local.setdefault("reward_switch_penalty", 0.007)

        use_pretrained = bool(cfg.get("dqn_use_pretrained", cfg_local.get("use_pretrained", False)))
        in_real_sim = bool(getattr(self, "_in_real_sim", False))
        
        # Run-scoped pretrained paths (avoid accidental cross-run reuse)
        dqn_model_path = MODEL_DQN_PATH
        dqn_cfg_path   = DQN_AGENT_CONFIG_PATH
        try:
            _run_dir = os.environ.get("RESULTS_RUN_DIR", "") or ""
            if _run_dir.strip():
                os.makedirs(_run_dir, exist_ok=True)
                dqn_model_path = os.path.join(_run_dir, os.path.basename(MODEL_DQN_PATH))
                dqn_cfg_path   = os.path.join(_run_dir, os.path.basename(DQN_AGENT_CONFIG_PATH))
        except Exception:
            pass

        loaded_from_disk = False
        if use_pretrained and in_real_sim:
            try:
                if os.path.exists(dqn_model_path) and os.path.exists(dqn_cfg_path):
                    if self._is_debug():
                        print(f"[DQN] Loading pretrained agent from {dqn_model_path}")
                    from rl.dqn_agent import DQNAgent
                    self.model = DQNAgent.load(dqn_model_path, dqn_cfg_path)
                    loaded_from_disk = True
                else:
                    if self._is_debug():
                        print("[DQN] No pretrained DQN files found; will train from scratch.")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained DQNAgent; training from scratch instead: {e}")
                self.model = None
                loaded_from_disk = False

        if not loaded_from_disk:
            self.model = self.get_model("dqn", input_shape=input_shape, dqn_config=cfg_local, lags=int(lags))
            self.model.fit(env)

            if use_pretrained and in_real_sim:
                try:
                    self.model.save(dqn_model_path, dqn_cfg_path)
                    print(f"💾 DQN model trained and saved to {dqn_model_path} / {dqn_cfg_path}.")

                except Exception as e:
                    print(f"⚠️ Could not save DQN model: {e}")

        # 5) Prediction on test (windowed) + supervised-style gating compatibility
        feats = self.dqn_feature_list
        X_test = test_data_features.to_numpy(dtype=np.float32, copy=False)

        if self.model is None:
            raise RuntimeError("DQN model is not initialized (NoneType)!")
        if X_test.shape[1] != self.model.state_size:
            raise ValueError(
                f"DQN feature mismatch: model expects {self.model.state_size}, got {X_test.shape[1]}.\n"
                f"Train features: {self.dqn_feature_list}\n"
                f"Test features:  {list(test_data_features.columns)}"
            )

        # Keep eval window consistent with env.window (lags), even for loaded agents.
        lags_eff = int(lags)
        try:
            if hasattr(self.model, "window") and int(getattr(self.model, "window")) != int(lags_eff):
                if self._is_debug():
                    print(f"[DQN] Forcing loaded agent window={getattr(self.model, 'window')} -> {lags_eff} to match lags.")
                setattr(self.model, "window", lags_eff)
                if hasattr(self.model, "config") and isinstance(getattr(self.model, "config"), dict):
                    self.model.config["window"] = lags_eff
        except Exception:
            pass

        # ---- Build a gating config that behaves like test_strategy (but without redesign) ----
        cfg_gate = dict(getattr(self, "features_config", {}) or {})
        # Allow a small whitelist of overrides from dqn_config (do NOT allow coverage overrides)
        try:
            for _k in (
                "gating_mode", "gate_mode",
                "target_active_rate", "target_coverage",
                "confidence_threshold",
                "deep_calibration_frac", "deep_calibration_min_samples",
            ):
                if _k in (cfg_local or {}):
                    cfg_gate[_k] = cfg_local[_k]
        except Exception:
            pass
        cfg_gate.setdefault("model_type", "dqn")
        
        # --- B1 Policy: GLOBAL target coverage locked for DQN as well (ignore local overrides) ---
        try:
            enforce_target_coverage_policy(cfg_gate, model_type='dqn')
        except Exception:
            pass

        # Coverage intent + calibration on TRAIN-only tail windows (prevents leakage)
        default_conf = float(cfg_gate.get("confidence_threshold", 0.50))
        cov_intent = bool(is_coverage_intent(cfg_gate))
        rate = float(cfg_gate.get("target_active_rate", cfg_gate.get("target_coverage", 0.0)) or 0.0)

        coverage_thr = None
        if cov_intent and rate > 0.0:
            try:
                frac = float(cfg_gate.get("deep_calibration_frac", 0.10))
                nmin = int(cfg_gate.get("deep_calibration_min_samples", 500))

                X_tr = train_data[self.dqn_feature_list].to_numpy(dtype=np.float32, copy=False)
                ntr = int(X_tr.shape[0])
                if ntr >= lags_eff and (ntr - lags_eff + 1) >= 50:
                    nwin_tr = ntr - lags_eff + 1
                    ncal = max(nmin, int(round(nwin_tr * max(0.01, min(frac, 0.99)))))
                    ncal = min(ncal, nwin_tr)
                    if ncal >= 50:
                        start = nwin_tr - ncal
                        states_cal = np.empty((ncal, lags_eff, X_tr.shape[1]), dtype=np.float32)
                        for j in range(ncal):
                            k = start + j
                            states_cal[j] = X_tr[k:k + lags_eff]

                        if not hasattr(self.model, "predict_proba"):
                            raise AttributeError("DQNAgent missing predict_proba(); add softmax(Q) helper or expose it.")

                        p_cal = sanitize_proba(self.model.predict_proba(states_cal))
                        # V3 (DQN): under coverage intent, calibrate threshold on TRADE confidence only (ignore HOLD)
                        p_use = p_cal[:, [0, 2]] if bool(cov_intent) and p_cal.ndim == 2 and p_cal.shape[1] >= 3 else p_cal
                        coverage_thr = float(fit_coverage_threshold_on_calibration(p_use, rate))

                        if self._is_debug():
                            _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            _ctx = "cv" if _in_cv_mode else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "eval")
                            print(f"[Calib][Coverage][DQN] conf_thr={coverage_thr:.6f} target_active_rate={rate:.6f} cal_windows={int(ncal)} ctx={_ctx}")
            except Exception as _e:
                coverage_thr = None
                if self._is_debug():
                    print(f"[Calib][Coverage][DQN] skipped: {_e}")

        conf_thr = float(freeze_confidence_threshold(cfg_gate, default_conf, coverage_conf_thr=coverage_thr))

        # Tripwire consistency: coverage intent but no calibrated threshold => invalid (nan) metrics
        if cov_intent and (not np.isfinite(conf_thr)):
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_dqn_strategy:coverage_intent_missing_thr")

        # ---- Batch predict windows on TEST: proba -> action + confidence ----
        n = int(X_test.shape[0])
        raw_actions = np.ones(n, dtype=int)          # default HOLD (1)
        max_conf    = np.zeros(n, dtype=np.float32)  # default low confidence

        if n >= lags_eff:
            nwin = n - lags_eff + 1
            states = np.empty((nwin, lags_eff, X_test.shape[1]), dtype=np.float32)
            for j in range(nwin):
                states[j] = X_test[j:j + lags_eff]

            if not hasattr(self.model, "predict_proba"):
                raise AttributeError("DQNAgent missing predict_proba(); add softmax(Q) helper or expose it.")

            proba = sanitize_proba(self.model.predict_proba(states))

            # IMPORTANT (coverage intent):
            # DQN has an explicit HOLD action. If we take argmax over {sell,hold,buy},
            # target_active_rate/coverage gating cannot "pull trades up" when HOLD dominates.
            # Under coverage intent, compute confidence over TRADE actions only (sell/buy),
            # then apply conf_thr (and αβγ dynamic thr_vec) to that trade confidence.
            if bool(cov_intent):
                # trade_p: [sell, buy] only
                trade_p = proba[:, [0, 2]]
                trade_conf = trade_p.max(axis=1)
                trade_dir = np.asarray(np.argmax(trade_p, axis=1), dtype=int)  # 0=sell, 1=buy
                # Map back to action space: 0 -> SELL(0), 1 -> BUY(2)
                a = np.where(trade_dir == 0, 0, 2).astype(int)
                c = trade_conf
            else:
                a = np.asarray(np.argmax(proba, axis=1), dtype=int)
                c = proba.max(axis=1)

            # IMPORTANT: write outputs for BOTH paths
            raw_actions[lags_eff - 1:] = a
            max_conf[lags_eff - 1:]    = c

        # Apply the SAME style of confidence gating as supervised models:
        # below threshold => force HOLD/neutral (action=1)
        if np.isfinite(conf_thr):
            raw_actions[max_conf < float(conf_thr)] = 1

        # --- 5b) Trade-frequency control: minimum bars between position switches ---
        min_switch = int(cfg_local.get("dqn_min_switch_interval", 0))
        if min_switch > 1 and len(raw_actions) > 0:
            filtered = []
            current = int(raw_actions[0])
            last_switch_idx = 0

            for i, a in enumerate(raw_actions):
                if i == 0:
                    filtered.append(current)
                    continue
                if a != current and (i - last_switch_idx) < min_switch:
                    filtered.append(current)
                else:
                    filtered.append(a)
                    if a != current:
                        current = a
                        last_switch_idx = i

            raw_actions = np.asarray(filtered, dtype=int)

        # --- 5c) Map discrete actions to trading signals {-1,0,+1} ---
        action_to_signal = np.array([-1.0, 0.0, 1.0], dtype=float)
        preds = action_to_signal[raw_actions]

        # Assemble aligned eval frame (safe)
        result_df = test_data_full.copy()
        result_df["pred"] = preds

        # Expose confidence stats for real-sim GateDiag (debug-only consumer reads these attrs).
        try:
            self._last_conf_thr_used = float(conf_thr) if np.isfinite(conf_thr) else None
            self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)
        except Exception:
            pass


        # Keep confidence around for debug / audits (no harm; evaluator ignores it)
        try:
            result_df["max_conf"] = pd.Series(max_conf, index=test_data_full.index).astype(float).values
        except Exception:
            pass

        if "returns" not in result_df.columns:
            result_df["returns"] = self.data["returns"].reindex(result_df.index).astype(float)

        result_df = result_df.dropna(subset=["returns"])
        if result_df.index.tz is None:
            result_df.index = result_df.index.tz_localize("UTC")

        # Align DQN evaluation window to the expected start (avoid warmup leakage)
        eval_start = getattr(self, "_expected_eval_start", None)
        if eval_start is not None:
            result_df = result_df.loc[result_df.index >= eval_start].copy()

        # --- Edge-bar guard for DQN (avoid big session gaps / last bar) ---
        _idx = result_df.index
        if len(_idx) >= 2:
            gaps = pd.Series(_idx[1:] - _idx[:-1], index=_idx[:-1])
            exp = gaps.median()
            is_edge = gaps > (exp * 1.5)

            if self._is_debug():
                try:
                    _ctx = "cv" if bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False)) else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "run")
                    _edge_idx = is_edge.index[is_edge]
                    _pred_ser = result_df["pred"]
                    _nz_before = int((_pred_ser != 0).sum())
                    _nz_edge = int((_pred_ser.reindex(_edge_idx).fillna(0) != 0).sum()) if len(_edge_idx) else 0
                    _nz_last = int(bool(_pred_ser.iloc[-1] != 0))
                    print(f"[EdgeGuardAudit][dqn] ctx={_ctx} exp={exp} edge_bars={int(is_edge.sum())} nz_before={_nz_before} nz_on_edge={_nz_edge} nz_last={_nz_last}")
                except Exception:
                    pass

            result_df.loc[is_edge.index[is_edge], "pred"] = 0
            result_df.iloc[-1, result_df.columns.get_loc("pred")] = 0

            if self._is_debug():
                try:
                    _nz_after = int((result_df["pred"] != 0).sum())
                    print(f"[EdgeGuardAudit][dqn] nz_after={_nz_after}")
                except Exception:
                    pass

        # First-10 trace for DQN (now has proba-derived confidence)
        if self._should_dump_decisions():
            try:
                _mc = None
                if "max_conf" in result_df.columns:
                    _mc = result_df["max_conf"].to_numpy(dtype=np.float32, copy=False)
                self._debug_dump_first_bars(
                    result_df.index,
                    raw_classes=None,
                    max_conf=_mc,
                    final_preds=result_df["pred"].values,
                    n=10,
                    label="dqn",
                )
            except Exception:
                pass

        if len(result_df) == 0:
            print("❌ No tradable rows left in DQN result after start cut.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_dqn_strategy:empty_result_after_cut")

        if (result_df["pred"] != 0).sum() == 0:
            print("ℹ️ DQN produced no trades in this window.")

        # Keep zero-lag here; compute_full_evaluation_metrics applies the 1-bar execution delay.
        result_df["pred"] = result_df["pred"].fillna(0.0)

        # 6) Evaluation (unified)
        try:
            cfg_adj = dict(cfg_gate)
            # persist the effective threshold so overlays/audits can see it
            cfg_adj["confidence_threshold"] = float(conf_thr)
            if coverage_thr is not None and np.isfinite(float(coverage_thr)):
                cfg_adj["_coverage_conf_thr"] = float(coverage_thr)

            if high_vol_thr_train is not None and cfg_adj.get("high_vol_thr") is None:
                cfg_adj["high_vol_thr"] = float(high_vol_thr_train)

            # Guarantee a valid spread series aligned to evaluated index
            try:
                if "spread" in getattr(self, "data", pd.DataFrame()).columns:
                    result_df["spread"] = self.data["spread"].reindex(result_df.index).astype(float).fillna(0.0)
                else:
                    result_df["spread"] = 0.0
            except Exception:
                result_df["spread"] = 0.0

            # Attach config for execution overlays (TWAP / kill-switch / etc.)
            try:
                result_df.attrs["features_config"] = dict(cfg_adj)
                result_df.attrs["debug_costs"] = bool(self._is_debug())
            except Exception:
                pass

            if bool(getattr(self, "trading_costs", True)):
                result_df = self._ensure_cost_columns(result_df, cfg_adj)
        except Exception:
            pass

        _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        if _in_cv_mode:
            _eval_ctx = "cv:test_dqn_strategy"
        elif bool(getattr(self, "_in_real_sim", False)):
            _eval_ctx = "real_sim:test_dqn_strategy"
        else:
            _eval_ctx = "eval:test_dqn_strategy"

        metrics = compute_full_evaluation_metrics(
            df=result_df,
            trading_costs=self.trading_costs,
            slippage_factor=self.slippage_factor,
            eval_context=_eval_ctx,
        )
        
        # Capture trade-intent precision from evaluator (cheap scalar; safe in CV).
        try:
            _attrs = getattr(result_df, "attrs", {}) or {}
            self._last_precision_trade = float(_attrs.get("precision_trade", float("nan")))
            self._last_n_trade_preds = int(_attrs.get("n_trade_preds", 0) or 0)
        except Exception:
            self._last_precision_trade = float("nan")
            self._last_n_trade_preds = 0


        # Optional: annotate raw test slice (legacy, enabled only in DEBUG mode)
        if self._is_debug():
            metric_names = [
                "cstrategy", "outperformance", "creturns", "sharpe", "drawdown", "trades",
                "geo_mean_ann", "directional_accuracy", "precision_macro", "f1_macro",
                "active_rate", "profit_per_hit", "return_per_trade", "win_rate",
                "strategy_volatility", "kurtosis",
            ]
            for name, value in zip(metric_names, metrics):
                test_data[name] = value  # raw slice (for parity)

        # ----------------------------
        # Results storage (R4 + R3 cap)
        # ----------------------------
        if not getattr(self, "_in_optuna_cv", False):
            self.results = result_df.copy() if result_df is not None else None
            try:
                _es = getattr(self, "_expected_eval_start", None)
                self.results_full = (
                    test_data_full.loc[test_data_full.index >= _es].copy()
                    if (_es is not None and test_data_full is not None)
                    else (test_data_full.copy() if test_data_full is not None else None)
                )
            except Exception:
                self.results_full = test_data_full.copy() if test_data_full is not None else None
            self._cv_last_eval_df = None
        else:
            self._cv_last_eval_df = (result_df.copy() if (result_df is not None and not result_df.empty) else None)

            try:
                if self._cv_last_eval_df is not None and self._is_debug():
                    _cap = int(os.environ.get("CV_MAX_EVAL_FRAMES", "5"))
                    if _cap > 0 and len(self._cv_fold_eval_frames) < _cap:
                        self._cv_fold_eval_frames.append(self._cv_last_eval_df.copy())
                    try:
                        _max_keep = int((getattr(self, "config", {}) or {}).get("cv_max_fold_eval_frames", 3) or 3)
                    except Exception:
                        _max_keep = 3
                if _max_keep > 0 and len(self._cv_fold_eval_frames) > _max_keep:
                    self._cv_fold_eval_frames = self._cv_fold_eval_frames[-_max_keep:]
            except Exception:
                try:
                    if self._cv_last_eval_df is not None and self._is_debug():
                        self._cv_fold_eval_frames = [self._cv_last_eval_df.copy()]
                    else:
                        self._cv_fold_eval_frames = []
                except Exception:
                    self._cv_fold_eval_frames = []

            self.results = None
            self.results_full = None

        metrics = _safe_metrics_return(metrics, context="eval_block_2")
        return metrics

    
    # ---- Debug helper: print effective thread config for this process ----
    def _print_thread_budget(self, tag: str = ""):
        if self._is_debug():
            try:
                from threadpoolctl import threadpool_info
                pools = threadpool_info()
            except Exception:
                pools = []
            envs = {k: os.getenv(k) for k in [
                "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS","SKLEARN_JOBS","RF_JOBS","XGB_JOBS",
                "TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS"
            ]}
            print(f"🧵 Threads[{tag}] env={envs} pools={pools}")


    def get_model(self, model_type, use_proba: bool = True, **params):
        """
        Construct and return an instance of the specified model.
        Trials stay serialized; each fit uses many threads/GPU.
        """
        self._print_thread_budget(tag=model_type)  # show effective threads for this build

        # Shared per-repeat seed
        seed = None
        try:
            seed = int((getattr(self, "features_config", {}) or {}).get("run_seed", 0)) or None
        except Exception:
            seed = None

        # --- local helper: guard invalid sklearn solver/penalty combos ---
        def _sanitize_logit_params(d: dict, *, ovr: bool = False) -> dict:
            p = dict(d or {})
            p.pop("multi_class", None)
            solver  = str(p.get("solver", "saga")).strip().lower()
            penalty = str(p.get("penalty", "l2")).strip().lower()
            allowed_solvers = {"lbfgs", "newton-cg", "liblinear", "sag", "saga"}
            allowed_penalty = {"l2", "l1", "elasticnet", "none"}
            if solver not in allowed_solvers: solver = "saga"
            if penalty not in allowed_penalty: penalty = "l2"

            if penalty == "l1":
                if solver in {"lbfgs","newton-cg","sag"}: solver = "saga"
                if not ovr and solver == "liblinear":    solver = "saga"
            elif penalty == "elasticnet":
                solver = "saga"
                try:
                    p["l1_ratio"] = min(1.0, max(0.0, float(p.get("l1_ratio", 0.5))))
                except Exception:
                    p["l1_ratio"] = 0.5
            elif penalty in {"none", "", None}:
                penalty = "none"
                if solver == "liblinear": solver = "lbfgs"

            if bool(p.get("dual", False)) and not (solver == "liblinear" and penalty == "l2"):
                p["dual"] = False
            try:
                C = float(p.get("C", 1.0));    p["C"]   = C if C > 0 else 1.0
            except Exception:
                p["C"] = 1.0
            try:
                tol = float(p.get("tol", 1e-3)); p["tol"] = tol if tol > 0 else 1e-3
            except Exception:
                p["tol"] = 1e-3
            try:
                mi = int(p.get("max_iter", 2000)); p["max_iter"] = mi if mi > 0 else 2000
            except Exception:
                p["max_iter"] = 2000
            p["solver"]  = solver
            p["penalty"] = penalty
            return p

        # ========== DEEP MODELS (GPU; light CPU threads for input pipeline) ==========
        if model_type == "cnn":
            cfg = filter_params(params, "cnn_")
            if seed is not None: cfg.setdefault("seed", seed)
            input_shape = params["input_shape"]
            # Match TF thread knobs to env (intra/inter)
            try:
                import tensorflow as _tf
                intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (os.cpu_count() or 8) - 2)
                inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass
            model = build_cnn(input_shape=input_shape, config=cfg)

        elif model_type == "lstm":
            cfg = filter_params(params, "lstm_")
            if seed is not None: cfg.setdefault("seed", seed)
            input_shape = params.get("input_shape")
            if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
                raise ValueError(f"Invalid input_shape for LSTM: {input_shape}")
            try:
                import tensorflow as _tf
                intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (os.cpu_count() or 8) - 2)
                inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass
            model = build_lstm(input_shape=input_shape, config=cfg)
            if model is None:
                raise RuntimeError("build_lstm returned None. Check model config or input shape.")

        elif model_type == "transformer":
            cfg = filter_params(params, "transformer_")
            if seed is not None: cfg.setdefault("seed", seed)
            input_shape = params["input_shape"]
            try:
                import tensorflow as _tf
                intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (os.cpu_count() or 8) - 2)
                inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass
            model = build_transformer(input_shape=input_shape, config=cfg)
        elif model_type == "dqn":
            input_shape = params["input_shape"]
            dqn_cfg = params.get("dqn_config", {}) or filter_params(params, "dqn_")
            dqn_cfg = filter_dqn_config(dqn_cfg or {})
            dqn_cfg["state_size"] = int(input_shape[0])
            if "window" not in dqn_cfg and "lags" in params:
                dqn_cfg["window"] = int(params["lags"])
            if seed is not None:
                dqn_cfg.setdefault("seed", seed)

            # NEW: thread tuning, like other TF models
            try:
                import tensorflow as _tf, os as _os
                intra = int(_os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (_os.cpu_count() or 8) - 2)
                inter = int(_os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass

            dqn_cfg = _coerce_dqn_cfg(dqn_cfg)
            model = DQNAgent(**dqn_cfg)

        # ========================== CLASSICAL (CPU) ==========================
        elif model_type == "svm":
            import numpy as _np
            svm_params = filter_params(params, "svm_")
            kernel = str(svm_params.get("kernel", "rbf")).lower()

            cw = svm_params.get("class_weight", None)
            if isinstance(cw, float) and (_np.isnan(cw) or _np.isinf(cw)):
                cw = None
            elif isinstance(cw, str):
                _cws = cw.strip().lower()
                cw = "balanced" if _cws == "balanced" else (None if _cws in ("", "nan", "none", "null") else None)

            gamma = svm_params.get("gamma", "scale")
            if isinstance(gamma, float):
                gamma = "scale" if (_np.isnan(gamma) or _np.isinf(gamma)) else gamma
            elif isinstance(gamma, str):
                g = gamma.strip().lower()
                if g not in ("scale", "auto"):
                    try:
                        gv = float(gamma); gamma = "scale" if (_np.isnan(gv) or _np.isinf(gv)) else gv
                    except Exception:
                        gamma = "scale"
                else:
                    gamma = g

            def _to_float(x, default):
                try: 
                    v = float(x); 
                    return float(default) if (_np.isnan(v) or _np.isinf(v)) else v
                except Exception:
                    return float(default)
            def _to_int(x, default):
                try: return int(x)
                except Exception: return int(default)

            C         = _to_float(svm_params.get("C", 1.0), 1.0)
            degree    = _to_int(svm_params.get("degree", 3), 3) if kernel == "poly" else 3
            max_iter  = _to_int(svm_params.get("max_iter", 200_000), 200_000)
            tol       = _to_float(svm_params.get("tol", 1e-2), 1e-2)
            shrinking = bool(svm_params.get("shrinking", True))
            cache_sz  = _to_float(svm_params.get("cache_size", 2048.0), 2048.0)

            svc = SVC(
                C=C, gamma=gamma, kernel=kernel, degree=degree,
                class_weight=cw,
                probability=False,  # calibrate below
                max_iter=max_iter, tol=tol, shrinking=shrinking, cache_size=cache_sz,
                decision_function_shape="ovr",
                random_state=seed,
            )

            # Features are already standardized by the global scale_features() path.
            # We therefore pass the bare SVC into CalibratedClassifierCV instead of
            # adding another StandardScaler inside a Pipeline.
            calibrate_method = params.get("calibrate_method", None)
            if calibrate_method not in ("sigmoid", "isotonic"):
                calibrate_method = "isotonic"
            cal_jobs = int(params.get("svm_calib_n_jobs", 0)) or int(os.environ.get("SKLEARN_JOBS", -1))
            model = CalibratedClassifierCV(estimator=svc, cv=3, method=calibrate_method, n_jobs=cal_jobs)

        elif model_type == "random_forest":
            rf_params = ensure_dict(filter_params(params, "rf_"))
            rf_params.setdefault("n_estimators", 300)
            
            # Safety clamp: protects against accidentally loaded giant configs.
            # No effect for normal Optuna ranges; does not increase capacity.
            try:
                rf_params["n_estimators"] = int(rf_params.get("n_estimators", 300))
            except Exception:
                rf_params["n_estimators"] = 300
            rf_params["n_estimators"] = max(1, min(rf_params["n_estimators"], 1200))
                
            rf_params.setdefault("max_depth", 18)
            rf_params.setdefault("min_samples_leaf", 10)
            rf_params.setdefault("max_features", "sqrt")
            rf_params.setdefault("class_weight", "balanced_subsample")

            # --- OOB vs bootstrap guard ---
            # In sklearn, oob_score is only valid if bootstrap=True. If bootstrap is
            # tuned to False, we must disable oob_score to avoid errors.
            bootstrap_flag = rf_params.get("bootstrap", True)
            if not bootstrap_flag:
                # force OOB off when no bootstrap sampling is used
                rf_params["oob_score"] = False
            else:
                # leave OOB on by default when using bootstrap
                rf_params.setdefault("oob_score", True)

            # ------------------------------------------------------------------
            # Threading safety: avoid rf n_jobs=-1 by default (can hard-crash some
            # native stacks under repeated CV/Optuna evaluation).
            # - Default RF_JOBS=1
            # - Treat -1/0 as "use safe default"
            # - Clamp to [1, cpu_count]
            # ------------------------------------------------------------------
            try:
                _safe_rf_jobs = int(os.environ.get("RF_JOBS", "1") or 1)
            except Exception:
                _safe_rf_jobs = 1
            if _safe_rf_jobs in (-1, 0):
                _safe_rf_jobs = max(1, (os.cpu_count() or 1) - 1)

            _rf_n_jobs = rf_params.get("n_jobs", _safe_rf_jobs)
            try:
                _rf_n_jobs = int(_rf_n_jobs)
            except Exception:
                _rf_n_jobs = _safe_rf_jobs
            if _rf_n_jobs in (-1, 0):
                _rf_n_jobs = _safe_rf_jobs
            _rf_n_jobs = max(1, min(_rf_n_jobs, (os.cpu_count() or 1)))
            rf_params["n_jobs"] = _rf_n_jobs
            
            if seed is not None:
                rf_params.setdefault("random_state", seed)
            model = RandomForestClassifier(**rf_params)


        elif model_type == "logistic":
            # single-estimator multinomial; rely on OpenMP inside solver; no joblib nesting
            _raw_logit = filter_params(params, "logit_") or {}
            logit_params = ensure_dict(_raw_logit)
            logit_params = _sanitize_logit_params(logit_params, ovr=False)
            logit_params.setdefault("solver", "saga")
            logit_params.setdefault("penalty", "l2")
            logit_params.setdefault("max_iter", 2000)
            logit_params.setdefault("tol", 1e-3)
            logit_params.setdefault("class_weight", "balanced")
            # `n_jobs` is accepted; used by liblinear/ovr; harmless for saga
            logit_params.setdefault("n_jobs", int(os.environ.get("SKLEARN_JOBS", max(1, (os.cpu_count() or 2) - 1))))
            if seed is not None: logit_params.setdefault("random_state", seed)
            model = Pipeline([("std", StandardScaler()), ("logit", LogisticRegression(**logit_params))])
            
        elif model_type == "decision_tree":
            dt_params = ensure_dict(filter_params(params, "dt_"))

            # Moderately regularised defaults; Optuna-tuned runs override via dt_* keys.
            # - max_depth: shallow-to-medium tree to avoid extreme overfitting.
            # - min_samples_split/leaf: ensure each leaf has enough samples to be stable.
            dt_params.setdefault("max_depth", 12)
            dt_params.setdefault("min_samples_split", 2)
            dt_params.setdefault("min_samples_leaf", 10)

            # FX labels are often imbalanced → use balanced class weights by default.
            dt_params.setdefault("class_weight", "balanced")

            # Mild cost-complexity pruning; Optuna can override via dt_ccp_alpha.
            dt_params.setdefault("ccp_alpha", 1e-4)

            if seed is not None:
                dt_params.setdefault("random_state", seed)

            model = DecisionTreeClassifier(**dt_params)

        elif model_type == "xgboost":
            xgb_params = ensure_dict(filter_params(params, "xgb_"))
            # multiclass objective + good defaults
            xgb_params.setdefault("objective", "multi:softprob")
            xgb_params.setdefault("num_class", 3)
            xgb_params.setdefault("eval_metric", "mlogloss")
            xgb_params.setdefault("importance_type", "gain")
            xgb_params.setdefault("subsample", 0.8)
            xgb_params.setdefault("colsample_bytree", 0.8)
            xgb_params.setdefault("max_depth", 6)
            xgb_params.setdefault("n_estimators", 400)
            
            # Safety clamp: protects against accidentally loaded giant configs.
            # No effect for normal Optuna ranges; does not increase capacity.
            try:
                xgb_params["n_estimators"] = int(xgb_params.get("n_estimators", 400))
            except Exception:
                xgb_params["n_estimators"] = 400
            xgb_params["n_estimators"] = max(1, min(xgb_params["n_estimators"], 1500))
            xgb_params.setdefault("min_child_weight", 1.0)

            # ---- REGULARIZATION KEYS (L2) ----
            # Optuna gives us xgb_lambda → "lambda" after filter_params.
            # XGBClassifier expects "reg_lambda" (sklearn-style name).
            if "lambda" in xgb_params and "reg_lambda" not in xgb_params:
                xgb_params["reg_lambda"] = xgb_params.pop("lambda")
            # Safe default if nothing was supplied
            xgb_params.setdefault("reg_lambda", 1.0)
            # Just in case, drop any stray "lambda"
            xgb_params.pop("lambda", None)

            # ---- THREADS ----
            xgb_params.setdefault(
                "n_jobs",
                int(os.environ.get("XGB_JOBS", max(1, (os.cpu_count() or 2) - 1)))
            )

            # ---- GPU CONTROL (XGBoost ≥ 2.0 style) ----
            use_gpu = os.environ.get("XGB_USE_GPU", "0") == "1"

            if use_gpu:
                # New-style: tree_method + device
                # https://xgboost.readthedocs.io/en/stable/gpu/
                xgb_params.setdefault("tree_method", "hist")
                xgb_params["device"] = os.environ.get("XGB_DEVICE", "cuda")
                # Let XGBoost pick the right GPU predictor internally
                xgb_params.pop("predictor", None)
            else:
                # Pure CPU
                xgb_params.setdefault("tree_method", "hist")
                xgb_params.pop("device", None)
                xgb_params.pop("predictor", None)

            if seed is not None:
                xgb_params.setdefault("random_state", seed)

            # Try GPU, fall back to CPU if GPU config explodes
            try:
                model = XGBClassifier(**xgb_params)
            except Exception as e:
                if use_gpu:
                    print(f"[XGBoost] GPU init failed ({e}); falling back to CPU.")
                    # Strip GPU-specific keys and retry on CPU
                    xgb_params.pop("device", None)
                    xgb_params["tree_method"] = "hist"
                    model = XGBClassifier(**xgb_params)
                else:
                    raise



        elif model_type == "ensemble_adaptive_regime":
            lstm_config = filter_params(params, "lstm_")
            rf_config   = filter_params(params, "rf_")
            logit_config= filter_params(params, "logit_")
            if seed is not None:
                lstm_config.setdefault("seed", seed)
                rf_config.setdefault("random_state", seed)
                logit_config.setdefault("random_state", seed)
            input_shape = params["input_shape"]
            model = AdaptiveRegimeStrategy(
                lstm_config=lstm_config,
                rf_config=rf_config,
                logit_config=logit_config,
                input_shape=input_shape,
                adx_col=params.get("adx_col", "adx_14"),
                vol_col=params.get("vol_col", "rolling_std_20"),
                adx_thresh=params.get("adx_thresh", 25),
                vol_thresh=params.get("vol_thresh", 0.002),
                adx_thresh_q=(
                    float(params.get("adx_thresh_q", 0.70))
                    if bool(params.get("train_lstm_on_trend_only", True))
                    else None
                ),
            )

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if model is None:
            raise ValueError(f"Model creation failed for type {model_type}")
        return model



    def _create_sliding_windows(self, df, features, window_size):
        """
        Vectorized fixed-length sliding windows over time.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain `features` columns and optionally a 'label' column.
        features : list[str]
            Feature columns to window.
        window_size : int
            Length of each temporal window.

        Returns
        -------
        Xv : np.ndarray       shape (n_windows, window_size, n_features)   dtype float32
        yv : np.ndarray       shape (n_windows,)                           dtype int32 (zeros if no 'label' col)
        idx : list[int]       end indices (in `df`) corresponding to each window
        """
        from numpy.lib.stride_tricks import sliding_window_view

        X2d = df[features].to_numpy(dtype=np.float32, copy=False)
        n   = X2d.shape[0]
        w   = int(window_size)
        if n < w:
            return np.empty((0, w, X2d.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32), []

        Xv = sliding_window_view(X2d, window_shape=w, axis=0)  # (n-w+1, w, f)
        Xv = Xv.reshape(-1, w, X2d.shape[1])

        idx = list(range(w - 1, n))
        if "label" in df.columns:
            yv = df["label"].to_numpy(dtype=np.int32, copy=False)[idx]
        else:
            yv = np.zeros((len(idx),), dtype=np.int32)

        return Xv, yv, idx
    
    def _predict_seq_windows_chunked(self, model, X2d: np.ndarray, win: int, batch_size: int, chunk_windows: int = 4096):
        """
        Memory-stable prediction for seq models (CNN/LSTM/Transformer).

        Instead of passing a massive sliding_window_view into Keras in one go
        (which can force large contiguous copies), we generate windows in chunks
        and predict chunk-by-chunk.

        Parameters
        ----------
        model : tf.keras.Model
        X2d : np.ndarray
            Shape (n_rows, n_features), float32 preferred.
        win : int
            Window length.
        batch_size : int
            Keras predict batch size.
        chunk_windows : int
            Number of windows per chunk (not rows). Lower = less peak RAM.

        Returns
        -------
        proba : np.ndarray
            Concatenated model outputs for all windows, shape (n_windows, n_classes).
        """
        from numpy.lib.stride_tricks import sliding_window_view

        try:
            n = int(X2d.shape[0])
            win = max(1, int(win))
            m = n - win + 1
            if m <= 0:
                return np.empty((0, 0), dtype=np.float32)

            chunk_windows = int(chunk_windows) if chunk_windows is not None else 0
            if chunk_windows <= 0:
                chunk_windows = 4096

            outs = []
            for s in range(0, m, chunk_windows):
                e = min(m, s + chunk_windows)

                # Need rows [s : e+win-1] to build exactly (e-s) windows
                X_slice = X2d[s : (e + win - 1)]
                Xv = sliding_window_view(X_slice, window_shape=win, axis=0)  # (e-s, win, f)

                p = model.predict(Xv, verbose=0, batch_size=int(batch_size))
                outs.append(p)

            proba = np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
            # Shape sanity: should match m windows
            if int(getattr(proba, "shape", [0])[0]) != int(m):
                raise ValueError(f"chunked predict produced wrong length: got {proba.shape[0]}, expected {m}")
            return proba

        except Exception as _e:
            # Fallback to one-shot predict (may be memory heavy, but keeps semantics)
            try:
                Xv = sliding_window_view(X2d, window_shape=win, axis=0)
                return model.predict(Xv, verbose=0, batch_size=int(batch_size))
            except Exception:
                raise _e

    
    def get_walk_forward_splits(self, walk_data, train_months_list, test_months_list, max_end):
        """
        Generate WFO splits, shrinking train_months if necessary so that at least
        one split is produced (when data is limited).
        """

        def months_between(a, b):
            # calendar months between two timestamps (floor)
            return (b.year - a.year) * 12 + (b.month - a.month)

        tasks = []
        first = walk_data.index[0]
        avail_months = months_between(first, max_end)

        for train_months in train_months_list:
            for test_months in test_months_list:
                req = train_months + test_months
                if req > avail_months:
                    # Try to salvage: shrink train to the largest feasible (>=6 months)
                    best_train = max(6, avail_months - test_months)
                    if best_train < 6:
                        print(f"[WFO] No feasible split: need {req} months, have {avail_months}. Skipping ({train_months}/{test_months}).")
                        continue
                    print(f"[WFO] Shrinking train_months {train_months}→{best_train} due to limited history ({avail_months} months).")
                    train_months_eff = int(best_train)
                else:
                    train_months_eff = int(train_months)

                # Earliest feasible start that still accommodates train+test
                start_date = first
                end_needed = start_date + pd.DateOffset(months=train_months_eff + test_months)
                if end_needed > max_end:
                    # anchor as late as possible
                    # (this guarantees at least one split when feasible)
                    start_date = max_end - pd.DateOffset(months=train_months_eff + test_months)

                # Build the rolling test steps
                while True:
                    if start_date + pd.DateOffset(months=train_months_eff + test_months) > max_end:
                        break
                    tasks.append((start_date, train_months_eff, test_months))
                    start_date += pd.DateOffset(months=test_months)

        # Debug
        try:
            print(f"[WFO] available_months={avail_months} | requested train/test={train_months_list}/{test_months_list} | splits={len(tasks)}")
        except Exception:
            pass
        return tasks

    
        
    def evaluate_strategy(self, best_params, train_start, train_end, test_start, test_end):
        """
        Route a single train/test evaluation to the correct routine based on `model_type`.

        Notes
        -----
        - Avoids pre-building models here; the called test/eval functions build what they need.
        - Accepts legacy grids that used `lags_range` instead of `lags`.

        Returns
        -------
        tuple[float, ...]
            The standard 16-tuple of metrics produced by the test/eval functions.
        """

        # CLEANUP: local debug + print-once helpers (no algorithm change)
        _dbg = bool(getattr(self, "_is_debug", lambda: False)()) or bool(getattr(self, "debug", False))

        def _dprint(msg: str):
            # DEBUG: quiet unless debug
            if _dbg:
                print(msg)

        def _print_once(key: str, msg: str, debug_only: bool = False):
            # CLEANUP: prevent print storms across CV / real-sim loops
            if debug_only and (not _dbg):
                return
            attr = f"_eval_strategy_once__{key}"
            if getattr(self, attr, False):
                return
            print(msg)
            setattr(self, attr, True)

        def _safe_len(x):
            try:
                return len(x)
            except Exception:
                return None

        # ---- Backward-compat for 'lags_range' ----
        if "lags" not in best_params and "lags_range" in best_params:
            # CLEANUP: keep the audit, avoid spam
            _print_once("lags_range_bc", "[WARN] 'lags' not in best_params, using 'lags_range'.")
            best_params["lags"] = int(best_params["lags_range"])

        # In real_trading_simulation, force each evaluation attempt (Top-N, consensus, etc.)
        # to start from the same deterministic month baseline to avoid config drift.
        in_real_sim = bool(getattr(self, "_in_real_sim", False))
        if in_real_sim:
            _base = getattr(self, "_month_base_features_config", None)
            if isinstance(_base, dict) and _base:
                try:
                    self.features_config = deepcopy(_base)
                except Exception as e:
                    # DEBUG: don't swallow silently
                    _dprint(f"⚠️ [evaluate_strategy] deepcopy(_month_base_features_config) failed: {e}")

        # ---- Basic coercions / defaults ----
        model_type = best_params["model_type"]

        # Respect user-provided toggle if present; otherwise keep current instance setting
        self.use_extended_features = best_params.get(
            "use_extended_features",
            getattr(self, "use_extended_features", True)
        )

        # Safe defaults (avoid KeyError)
        label_threshold = float(best_params.get("label_threshold", 1e-4))

        # IMPORTANT: do not hard-default confidence_threshold here.
        # Resolve it AFTER merging params into features_config so CV and real-sim match.
        confidence_threshold = best_params.get("confidence_threshold", None)
        lags = int(best_params.get("lags", 8))

        # CLEANUP: one unified snapshot printer (logging-only)
        def _print_eval_snapshot(_model: str, _cfg: dict, _lags: int, _conf_thr, _calib):
            if _dbg or in_real_sim:
                print(
                    f"[EVAL-SNAPSHOT] model={_model} | "
                    f"lags={_lags} | "
                    f"lag_depth={_cfg.get('lag_depth')} | "
                    f"roll_windows={_cfg.get('roll_windows') or _cfg.get('roll_windows_key') or _cfg.get('roll_windows_key_v2')} | "
                    f"use_fracdiff={_cfg.get('use_fracdiff')} | "
                    f"confidence_threshold={_conf_thr} | "
                    f"calibrate_method={_calib}"
                )

        # ANCHOR: # ---------- Pure Transformer + XGB (explicit, no DQN) ----------
        # Transformer+XGB path has been retired in this project build.
        # Fail fast if the model_type accidentally appears in a grid / Top-N pool.
        if model_type in {"transformer_xgb", "transformer_xgb_only"}:
            raise ValueError(
                "[ERROR] model_type=transformer_xgb(_only) is not supported in this build. "
                "Remove it from configs / Top-N pools."
            )
            
        if model_type == "dqn":
            # ANCHOR: # ---------- DQN only ----------
            # DQN is routed differently than supervised models; keep the dedicated handler.
            # But still merge best_params into features_config so gating knobs (coverage intent etc.)
            # are visible during this evaluation, then restore config to avoid cross-month drift.
            _cfg_snapshot = deepcopy(self.features_config)
            try:
                self._merge_params_into_features_config(best_params, force_lags=lags)
                metrics = self.test_dqn_strategy(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    lags=lags,
                    dqn_config=best_params.get("dqn_config", {}),
                )
            finally:
                self.features_config = _cfg_snapshot

            if not isinstance(metrics, tuple) or len(metrics) != 16:
                raise ValueError("[ERROR] dqn path did not return 16 metrics")
            return metrics
 

        # ---------- Ensembles: CNN+LSTM+XGB or Adaptive Regime ----------
        elif model_type in {"ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime"}:
            _cfg_snapshot = deepcopy(self.features_config)
            try:
                # Ensure tuned feature toggles and per-model knobs are visible to the pipeline
                self._merge_params_into_features_config(best_params, force_lags=lags)

                # Build/normalize ensemble_config
                ens_cfg = dict(best_params.get("ensemble_config", {}))

                # Allow passing sub-configs directly in best_params; copy them into ensemble_config if missing
                for sub in ("cnn_config", "lstm_config", "transformer_config", "xgb_config"):
                    if sub not in ens_cfg and sub in best_params and isinstance(best_params[sub], dict):
                        ens_cfg[sub] = dict(best_params[sub])

                # Do not propagate confidence_threshold here; coverage/backtests will compute it after
                if "confidence_threshold" in ens_cfg:
                    ens_cfg.pop("confidence_threshold", None)
                if "calibrate_method" in best_params and "calibrate_method" not in ens_cfg:
                    ens_cfg["calibrate_method"] = str(best_params["calibrate_method"]).lower()

                cfg = self.features_config or {}

                # CLEANUP: snapshot line was unconditional; now debug/real-sim only.
                # Also show the *source* of confidence_threshold since ens_cfg removes it intentionally.
                _print_eval_snapshot(
                    _model=model_type,
                    _cfg=cfg,
                    _lags=lags,
                    _conf_thr=best_params.get("confidence_threshold", cfg.get("confidence_threshold")),
                    _calib=ens_cfg.get("calibrate_method") or cfg.get("calibrate_method"),
                )

                metrics = self.test_ensemble_strategy(
                    train_start=train_start, train_end=train_end,
                    test_start=test_start,  test_end=test_end,
                    lags=lags,
                    label_threshold=label_threshold,
                    ensemble_config=ens_cfg,
                    model_type=model_type,
                )
            finally:
                # Prevent param bleed to subsequent runs
                self.features_config = _cfg_snapshot

            # Optional: fail-fast if your pipeline reports effective lags different from tuned
            if hasattr(self, "_effective_lags_last"):
                eff = int(getattr(self, "_effective_lags_last"))
                if eff != lags:
                    raise RuntimeError(
                        f"[ABORT] Effective lags={eff} differs from tuned lags={lags}. "
                        f"Refuse to evaluate with a silently-shrunk spec."
                    )

            if (not isinstance(metrics, tuple)) or (len(metrics) != 16):
                n = _safe_len(metrics)
                raise ValueError(f"[ERROR] test_ensemble_strategy() returned {n} values — expected 16")
            return metrics

        # ---------- CNN / LSTM / Transformer / Classical ML ----------
        else:
            # Merge tuned trial params into a TEMP config that is visible to prepare_features(),
            # run the evaluation, then restore whatever the backtester had before.
            _cfg_snapshot = deepcopy(self.features_config)
            try:
                self._merge_params_into_features_config(best_params, force_lags=lags)
                cfg = self.features_config or {}

                # Resolve confidence_threshold consistently with CV:
                # - use tuned param if present
                # - else use merged cfg if present
                # - else fallback: deep models -> 0.0 (no silent no-trade), others -> 0.80
                if confidence_threshold is None:
                    if model_type in {"cnn", "lstm", "transformer"}:
                        confidence_threshold = float(cfg.get("confidence_threshold", 0.0))
                    else:
                        confidence_threshold = float(cfg.get("confidence_threshold", 0.80))
                else:
                    confidence_threshold = float(confidence_threshold)

                # CLEANUP: unify snapshot printing; keep GateInfo line as audit
                _print_eval_snapshot(
                    _model=(cfg.get("model_type") or model_type),
                    _cfg=cfg,
                    _lags=lags,
                    _conf_thr=confidence_threshold,
                    _calib=cfg.get("calibrate_method"),
                )
                if (_dbg or in_real_sim) and (cfg.get("target_active_rate", None) is not None):
                    _print_once(
                        "gateinfo_target_active_rate",
                        "[GateInfo] target_active_rate is set → coverage-calibrated threshold is used; "
                        "fixed confidence_threshold is ignored.",
                    )

                metrics = self.test_strategy(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    lags=lags,
                    confidence_threshold=confidence_threshold,
                    label_threshold=label_threshold,
                )
            finally:
                # Always restore caller state so Top-N tries and later folds don’t leak config
                self.features_config = _cfg_snapshot

            # Optional: fail-fast on silent lags shrink (if your pipeline sets this attribute)
            if hasattr(self, "_effective_lags_last"):
                eff = int(getattr(self, "_effective_lags_last"))
                if eff != lags:
                    raise RuntimeError(
                        f"[ABORT] Effective lags={eff} differs from tuned lags={lags}. "
                        f"Refuse to evaluate with a silently-shrunk spec."
                    )

            # Validate the fixed-length contract
            if (not isinstance(metrics, tuple)) or (len(metrics) != 16):
                n = _safe_len(metrics)
                raise ValueError(f"[ERROR] test_strategy() returned {n} values — expected 16")

            return metrics


    def _record_wfo_monthly_result(self, result: dict) -> None:
        """
        Append a compact monthly record for PBO/MCS analysis.

        Parameters
        ----------
        result : dict
            The per-month result dict built inside real_trading_simulation.
            Expected keys (if present): 'model_type', 'strategy_type',
            'test_start', 'test_end', 'strategy_return', 'cum_return',
            'sharpe', 'trades'.
        """
        try:
            import pandas as _pd
            mt = result.get("model_type", getattr(self, "model_type", ""))
            st = result.get("strategy_type", None)
            sid = f"{mt}:{st}" if st is not None else str(mt)

            rec = {
                "strategy_id": sid,
                "model_type": mt,
                "strategy_type": st,
                "test_start": _pd.to_datetime(result.get("test_start")),
                "test_end": _pd.to_datetime(result.get("test_end")),
                # monthly returns (continuous): strategy vs BH
                "strategy_return": float(result.get("strategy_return", float("nan"))),
                "bh_return": float(result.get("cum_return", float("nan"))),
                # diagnostics
                "sharpe": float(result.get("sharpe", float("nan"))),
                "trades": int(result.get("trades", 0) or 0),
            }
        except Exception as _e:
            # Never let analysis bookkeeping affect the main pipeline
            if self._is_debug():
                print(f"[PBO/MCS] Failed to build monthly record: {_e}")
            return

        self._wfo_monthly_records.append(rec)
        

    def log_simulation_result(
        self,
        i: int,
        test_start,
        test_end,
        perf: float,
        creturns: float,
        sharpe: float,
        trades: int,
        drawdown: float,
        cumsum: float,
        result: dict,
        csv_path: str,
        directional_accuracy: float,
        precision_macro: float,
        f1_macro: float,
        active_rate: float,
        profit_per_hit: float,
        equity_bh: float | None = None,
    ):
        """
        Append a single fold's summary metrics to a CSV and print a concise log line.

        Parameters
        ----------
        i : int
            Fold index (0-based) — will be logged as month = i+1.
        perf, creturns : float
            Monthly equity factors for strategy and buy&hold (e.g., 0.995, 1.012).
        cumsum : float
            Strategy continuous equity for the month (will be reported as equity - 1 in 'cumsum').
        equity_bh : float | None
            Buy&Hold continuous equity for the month (optional).
        """
        # ensure the output directory exists
        out_dir = os.path.dirname(csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        equity_strategy = float(cumsum)
        eq_bh = float(equity_bh) if equity_bh is not None else np.nan
        
        # Secondary activity metric (signal coverage).
        try:
            _signal_coverage = float(result.get("signal_coverage", np.nan))
        except Exception:
            _signal_coverage = np.nan

        result.update({
            "month": i + 1,
            "test_start": test_start,
            "test_end": test_end,

            # Monthly factors
            "cstrategy": perf,
            "creturns": creturns,
            "outperformance": round(perf - creturns, 6),

            "sharpe": sharpe,
            "drawdown": drawdown,
            "trades": trades,

            # Keep legacy 'cumsum' as cumulative strategy return (equity - 1)
            "cumsum": round(equity_strategy - 1.0, 6),

            # Explicit continuous equities
            "equity_strategy": round(equity_strategy, 6),
            "equity_bh": round(eq_bh, 6) if np.isfinite(eq_bh) else np.nan,
            "equity_outperformance": round(
                (equity_strategy - eq_bh) if np.isfinite(eq_bh) else np.nan, 6
            ),

            # Helpful monthly add-ons
            "strategy_return": round(perf - 1.0, 6),
            "bh_return": round(creturns - 1.0, 6),

            "directional_accuracy": directional_accuracy,
            "precision_macro": precision_macro,
            "f1_macro": f1_macro,
            
            # Trade-intent precision (post-gating, causally aligned). We try result first,
            # then fall back to evaluated df attrs (self.results) if available.
            "precision_intent": self._safe_float(
                result.get("precision_intent", float("nan")),
                fallback_key="precision_intent",
            ),
            "intent_bars": self._safe_int(
                result.get("intent_bars", 0),
                fallback_key="intent_bars",
            ),

            
            # Activity (canonical vs secondary)
            "exec_active_rate": active_rate,
            "signal_coverage": _signal_coverage,
            "profit_per_hit": profit_per_hit,
        })
        
        # --- Schema guard: ensure effective confidence threshold is present under
        # the canonical column name expected by downstream ranking.
        try:
            _ct = self._safe_float(result.get("confidence_threshold", np.nan))
            if not np.isfinite(_ct):
                _ctu = self._safe_float(result.get("confidence_threshold_used", np.nan))
                if np.isfinite(_ctu):
                    result["confidence_threshold"] = float(_ctu)
        except Exception:
            pass

        # Drop heavy config blobs from the main CSV; keep them only in sidecar dumps.
        _cfg_keys = {
            "cnn_config", "lstm_config", "transformer_config",
            "xgb_config", "rf_config", "logit_config", "dqn_config"
        }
        _row = {k: v for k, v in result.items() if k not in _cfg_keys}

        pd.DataFrame([_row]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=not os.path.exists(csv_path)
        )

        # Compact monthly line (Europe/Lisbon assumed externally)
        try:
            _ret_m  = float(perf)
            _bh_m   = float(creturns)
            _earned_s = float(equity_strategy) - 1.0
            _earned_b = float(eq_bh) - 1.0 if not np.isnan(eq_bh) else float("nan")
            _start = str(getattr(test_start, "date", lambda: test_start)())
            _end   = str(getattr(test_end, "date", lambda: test_end)())
            print(
                f"📈 M{i+1} {_start}→{_end} | "
                f"month_factor: Strat {_ret_m:.5f} vs BH {_bh_m:.5f} | "
                f"cum_equity: Strat {float(equity_strategy):.5f} vs BH {float(eq_bh):.5f} | "
                f"cum_pnl: Strat {_earned_s:+.2%} vs BH {_earned_b:+.2%} | "
                f"Sharpe {float(sharpe):+.2f} | Trades {int(trades)} | DD {float(drawdown):.2%}"
              )
        except Exception:
            # Fallback to legacy print if anything goes wrong
            print(
                f"\n📈 Month {i + 1} Results: Strat(m): {perf:.5f} | BH(m): {creturns:.5f} | "
                f"EqStrat: {equity_strategy:.5f} | EqBH: {eq_bh:.5f} | "
                f"Sharpe: {sharpe:.2f} | Trades: {trades} | DD: {drawdown:.2%}"
            )

    def run_strategy(self, config, models_to_test=None, n_trials=30, n_startup_trials=10): 
        """
        Run walk-forward optimization (WFO): for each split, tune with Optuna (sliding CV or mini-block CV),
        refit/evaluate on the held-out test window, and aggregate results.

        Parameters
        ----------
        config : dict
            Experiment configuration (model_type, search spaces, months, etc.).
        models_to_test : list[str] | None
            Subset of model types to consider. If None, uses `config['model_type']`.
        n_trials, n_startup_trials : int
            Optuna trial counts (forwarded via `config` to the tuner).

        Returns
        -------
        (pd.DataFrame, dict | None)
            DataFrame of fold results and the best aggregated parameter combo (or None).
        """
        
        # --- Per-run CV geometry cache (safe: only small integers, no DataFrames) ---
        # This is used inside _single_study_cv to avoid recomputing identical
        # Mini-Block geometry (k_blocks, embargo_bars, etc.) for every Optuna trial.
        # It is reset on each run_strategy call, so it cannot accumulate across runs.
        self._cv_geom_cache = {}

        # ---- Config defaults (centralized) ----
        self.apply_cv_defaults(config)
        
        cfg_f = getattr(self, "features_config", {}) or {}

        # Limit the model set (skip DQN here; it has its own path)
        if models_to_test is None:
            models_to_test = [config.get("model_type","xgboost")]
        
        # Only exclude standalone DQN here;
        models_to_test = [m for m in models_to_test if m != "dqn"]

        if not models_to_test:
            # print("[DEBUG] run_strategy called with no models to test (DQN or empty). Skipping Optuna!")
            return None, None

        log_print(f"Models to test in this WFO: {models_to_test}", level="COMPACT")
        log_print("\n🚀 Running unified strategy tuner with walk-forward optimization...", level="COMPACT")

        model_type = config.get("model_type", "svm")
        use_proba = config.get("use_proba", True)  # currently unused here, kept for parity

        full_data = self.data

        # --- Session filter (NY hours) with tz-safety ---
        walk_limit_start = full_data.index[0]
        walk_data = full_data.loc[walk_limit_start:]

        # Optional: only filter for sessions if you explicitly want to plan WFO on a reduced clock.
        if bool(config.get("wfo_session_filter", False)):
            try:
                if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                    full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                    _ny_times = full_idx.tz_convert("America/New_York")
                    self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
                walk_data = walk_data.loc[self._ny_mask.reindex(walk_data.index, fill_value=False)]
            except Exception as e:
                log_print(f"⚠️ WFO session filter (NY) failed: {e} — proceeding without it.", level="COMPACT")
        max_end = walk_data.index[-1]

        if self._is_debug():
            log_print(
                f"Walk data range after filtering: {walk_data.index[0]} to {walk_data.index[-1]}",
                level="DEBUG",
            )


        def ensure_list(x):
            if isinstance(x, (list, tuple)):
                return list(x)
            if x is None:
                return []
            return [x]

        # --- Respect tuned months from Optuna or defaults ---
        train_months_list = ensure_list(
            config.get("train_months", TRAIN_TEST_MONTHS[model_type]["train"][0])
        )
        test_months_list = ensure_list(
            config.get("test_months", TRAIN_TEST_MONTHS[model_type]["test"][0])
        )

        tasks = self.get_walk_forward_splits(
            walk_data, train_months_list, test_months_list, max_end
        )
        if self._is_debug():
            log_print(f"Number of walk-forward splits: {len(tasks)}", level="DEBUG")

        # === ONE-TIME OPTUNA STUDY (before any parallel folds) ======================
        # Use the first fold's train window for tuning; cache Top-5 on self.
        if not tasks:
            log_print("❌ No WFO tasks generated.", level="COMPACT")
            return None, None

        first_start, first_train_months, first_test_months = tasks[0]
        first_train_end = first_start + pd.DateOffset(months=first_train_months)
        
        # IMPORTANT: training must end strictly BEFORE the first test month begins
        # to avoid boundary leakage (pandas .loc is inclusive on endpoints).
        first_test_start = first_train_end
        idx = walk_data.index

        cv_mode_req = str(config.get("cv_mode", "mini_block")).lower()
        monthly_req = cv_mode_req in {"monthly_roll", "monthly", "month", "month_roll", "rolling_month"}

        if monthly_req:
            # Ensure Optuna's tuning sample contains enough calendar history to support:
            # - rolling train_months per fold (same as real trading), plus
            # - K monthly validation blocks (default 5 months), all strictly before first_test_start.
            try:
                k_blocks = int(config.get("cv_blocks", 5))
            except Exception:
                k_blocks = 5
            try:
                val_months_eff = max(1, int(round(float(config.get("cv_val_months", 1.0)))))
            except Exception:
                val_months_eff = 1

            # Match real trading train window length (WFO uses first_train_months here)
            train_months_eff = int(first_train_months)

            need_months = train_months_eff + (k_blocks * val_months_eff)
            optuna_start = first_test_start - pd.DateOffset(months=int(need_months))

            # Clamp to available data start
            if len(idx) > 0:
                optuna_start = max(optuna_start, idx[0])

            first_train_df = walk_data[(idx >= optuna_start) & (idx < first_test_start)]
        else:
            # Keep legacy behaviour for mini_block, but still end-exclusive for leakage safety
            first_train_df = walk_data[(idx >= first_start) & (idx < first_test_start)]

        
        # --- Transparency log: what exactly is the HPO tuning span? ---
        try:
            if bool(config.get("print_cv_debug", False)) or str(config.get("logmode", "")).lower() in {"compact","verbose"}:
                if len(first_train_df) > 0:
                    _hpo_s = first_train_df.index[0]
                    _hpo_e = first_train_df.index[-1]
                    _nbar  = len(first_train_df)
                    _mode  = "monthly_roll" if monthly_req else "mini_block"
                    log_print(
                        f"[HPO] mode={_mode} | tuning_span={_hpo_s} → {_hpo_e} "
                        f"({_nbar} bars) | boundary(first_test_start)={first_test_start} (end-exclusive)"
                    , level="COMPACT")
                else:
                    log_print(f"[HPO] tuning_span is empty | boundary(first_test_start)={first_test_start} (end-exclusive)", level="COMPACT")
        except Exception:
            pass

        if len(first_train_df) < 150:
            print("❌ Not enough data in the first fold to run tuning.")
            return None, None

        # Base features for the tuning fold (exclude leakage/targets)
        base_features_first = [
            c for c in first_train_df.columns
            if c not in ("returns", "price", "spread", "high", "low", "label", "time")
        ]

        # Coarse windows for legacy sliding fallback (mini-block CV sizes itself)
        min_train_window_first = int(len(first_train_df) * 0.75)
        val_window_first       = max(1, int(len(first_train_df) * 0.25))
        if min_train_window_first + val_window_first > len(first_train_df):
            val_window_first = len(first_train_df) - min_train_window_first
        cv_config_first = {"min_train_window": min_train_window_first, "val_window": val_window_first}
        cv_config_first["cv_n_jobs"] = int(os.environ.get("CV_JOBS", os.cpu_count() or 1))
        cv_config_first["score_for_no_trades"] = -1.0

        # 👉 Make CV knobs visible to the nested _single_study_cv via self.config
        #    (that function reads getattr(self, "config", {}) then apply_cv_defaults(...))
        try:
            self.config = {**getattr(self, "config", {}), **dict(cv_config_first)}
        except Exception:
            self.config = dict(cv_config_first)


        def _single_study_cv(train_data, params, min_train_window, val_window, trial=None, cv_config_override=None):
            """
            Mini-block cross-validation driver for a single Optuna trial.
            Computes the objective J over K folds.
            """
    
            import numpy as np
            
            # ============================================================
            # CV fold-row alignment (kills "random" row drift)
            # - Always append exactly ONE row per fold (OK or invalid)
            # - Overview/pruning/coverage read from fold_rows only
            # ============================================================
            fold_rows: list[dict] = []

            def _base_row(fold_id: int, val_start, val_end, train_rows=None, val_rows=None) -> dict:
                return {
                    "fold_id": int(fold_id),
                    "val_start": val_start,
                    "val_end": val_end,
                    "train_rows": int(train_rows) if train_rows is not None else 0,
                    "val_rows": int(val_rows) if val_rows is not None else 0,
                    "trades": 0,
                    "active": 0.0,
                    "sr": None,
                    "psr": None,
                    "status": "⛔ UNSET",
                    "reason": "",
                }

            def _safe_float(x, default=None):
                try:
                    v = float(x)
                    return v if np.isfinite(v) else default
                except Exception:
                    return default

            def _safe_int(x, default=0):
                try:
                    return int(x)
                except Exception:
                    return default

            def _finalize_row(row: dict, *, trades=None, active=None, sr=None, psr=None, status=None, reason=None):
                if trades is not None: row["trades"] = _safe_int(trades, 0)
                if active is not None: row["active"] = _safe_float(active, 0.0) or 0.0
                if sr is not None: row["sr"] = _safe_float(sr, None)
                if psr is not None: row["psr"] = _safe_float(psr, None)
                if status is not None: row["status"] = str(status)
                if reason is not None: row["reason"] = str(reason)
                return row

            
            
            # ── Reset small per-fold CV diagnostics at the start of each trial ──
            # Without this, _cv_fold_eval_frames accumulates copies of every fold
            # DataFrame across all Optuna trials, causing a slow RAM drift.
            try:
                self._cv_fold_eval_frames = []
            except AttributeError:
                # First CV call in this process: create the attribute
                self._cv_fold_eval_frames = []

            # ---- Minimal pretty table helper ----
            def _fmt_table(headers, rows, title=None):
                col_w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)] if rows else [len(str(h)) for h in headers]
                sep = "+".join("-" * (w + 2) for w in col_w)
                def _row(cells):
                    pads = (cells + [""] * (len(col_w) - len(cells))) if len(cells) < len(col_w) else cells
                    return "|".join(" " + str(c).ljust(w) + " " for c, w in zip(pads, col_w))
                lines = []
                if title:
                    lines.append(f"\n{title}")
                lines.append(sep)
                lines.append(_row(headers))
                lines.append(sep)
                for r in rows:
                    lines.append(_row(r))
                lines.append(sep)
                return "\n".join(lines)

            # (Used only if trial is provided; harmless otherwise)
            def _quantifiable_items(d: dict):
                keep_names = {"model_type","strategy_type","feature_selection","calibrate_method",
                            "roll_windows_key","roll_windows_key_v2"}
                out = []
                for k, v in sorted(d.items()):
                    if k.startswith("_"):
                        continue
                    if isinstance(v, (int, float, bool)):
                        out.append((k, v))
                    elif isinstance(v, str) and (k in keep_names or len(v) <= 16):
                        out.append((k, v))
                return out

            def _print_trial_header_table(params, cfg, trial):
                if trial is None:
                    return
                core = dict(params)
                if "lags" not in core and "lags_range" in core:
                    core["lags"] = int(core.get("lags_range", 0))
                core_tbl = _quantifiable_items(core)
                ind = (cfg or {}).get("indicator_windows", {}) or {}
                ta_rows = []
                for name, val in sorted(ind.items()):
                    flag = cfg.get(f"use_{name}", None)
                    ta_rows.append((name, (val if isinstance(val, (int, float)) else str(val)),
                                (flag if isinstance(flag, bool) else "")))
                print(_fmt_table(["Param","Value"], core_tbl, title=f"📋 Trial #{getattr(trial, 'number', '?')} — Hyperparameters"))
                if ta_rows:
                    print(_fmt_table(["Indicator","Window/Value","Enabled"], ta_rows, title="📐 Indicators (trial view)"))
                    
            _prev_cv  = getattr(self, "_in_cv", False)
            _prev_dbg = getattr(self, "_dbg_first_bars", False)
            self._in_cv = True
            self._dbg_first_bars = False
            _old_cv_flag = getattr(self, "_in_optuna_cv", False)
            
            # --- Calibration accumulators across folds ---
            calib_brier_sum = 0.0
            calib_nll_sum   = 0.0
            calib_n_samples = 0
            
            setattr(self, "_in_optuna_cv", True)
            for _k in ("_deep_temp_T", "_coverage_conf_thr"):
                if hasattr(self, _k):
                    try: delattr(self, _k)
                    except Exception: pass
                    
            # CV memory control: feature-slice caching is bypassed in CV mode, but
            # any previously cached engineered frames (from non-CV runs) can still
            # linger on the instance and bloat RAM. Clear them eagerly at CV entry.
            try:
                self._clear_feature_cache()
            except Exception:
                pass

            # Pull config safely and apply CV defaults if available
            config = getattr(self, "config", {}) if hasattr(self, "config") else {}
            try:
                config = self.apply_cv_defaults(dict(config))
            except Exception:
                config = dict(config)
                
            # without relying on evaluate_cv_func being a bound method.
            if cv_config_override:
                try:
                    config.update(dict(cv_config_override))
                except Exception:
                    pass

            # -------------------------------
            # CV config normalization (single source of truth)
            # -------------------------------
            cv_config = dict(config.get("cv_config", {}) or {})
            for _k in (
                "cv_prune_precision_intent",
                "cv_prune_min_precision_intent",
                "cv_prune_min_intent_bars_fold",
                "cv_prune_min_intent_bars",
            ):
                if _k in config:
                    try:
                        cv_config[_k] = config.get(_k)
                    except Exception:
                        pass


            # CV table behavior (single source of truth)
            table_mode          = str(config.get("cv_table_mode", "compact")).lower()   # "compact" | "verbose" | "full" | "off"
            table_verbose       = bool(config.get("cv_table_verbose", False)) or (table_mode in {"verbose","full"})
            table_only_failures = bool(config.get("cv_table_only_failures", False))

                        # Global pruning relaxation knob
            cv_relax = float(config.get("cv_prune_relax", 1.0))
            cv_relax = max(0.0, min(cv_relax, 1.0))

            # Base gates
            _M_gate_base = int(config.get("cv_min_trades_per_block", 5))
            _r_min_base  = float(config.get("cv_gate_min_active_rate", 0.02))
            _L_gate_base = int(config.get("cv_gate_min_folds", 3))

            # Effective gates after relaxation
            if cv_relax <= 0.0:
                # Fully relaxed: do not gate/prune via these thresholds
                _M_gate_eff = 0
                _r_min_eff  = 0.0
                _L_gate_eff = 1
            else:
                # Larger base thresholds = stricter → scale down by cv_relax
                _M_gate_eff = max(1, int(round(_M_gate_base * cv_relax)))
                _r_min_eff  = max(0.0, _r_min_base * cv_relax)
                _L_gate_eff = max(1, int(round(_L_gate_base * cv_relax)))


            # Unified penalty logger
            def _cv_penalty(reason: str, **kv):
                if bool(config.get("print_cv_debug", False)):
                    extras = " | ".join(f"{k}={v}" for k, v in kv.items())
                    log_print(
                        f"[CV-PENALTY] {reason}" + (f" | {extras}" if extras else ""),
                        level="DEBUG",
                    )


            if bool(config.get("print_cv_debug", False)):
                print(f"[CV] mode={str(config.get('cv_mode','mini_block')).lower()} | model={params.get('model_type','?')}")

            setattr(self, "_in_optuna_cv", True)
            try:
                total_len = len(train_data)

                # Sliding stride hints (only for legacy path; we keep mini-block)
                target_folds = int(config.get("cv_target_folds", 5))
                cv_val_months = float(config.get("cv_val_months", 1.0))
                bars_per_month_hint = int(config.get("bars_per_month_hint", 1000))
                if val_window is None or val_window <= 0:
                    val_window = max(1, int(round(cv_val_months * bars_per_month_hint)))

                embargo = int(config.get("cv_embargo_bars", 0))
                avail = max(0, int(min_train_window) - int(val_window) - embargo)
                forced_stride_frac = config.get("cv_sliding_stride_frac", None)
                if forced_stride_frac is not None:
                    step = max(1, int(round(val_window * float(forced_stride_frac))))
                else:
                    denom = max(1, target_folds - 1)
                    step = max(1, int(avail // denom)) if avail > 0 else max(1, int(val_window // 2))
                step = max(1, min(step, max(1, int(val_window))))

                if min_train_window + val_window > total_len:
                    _cv_penalty("Insufficient data for requested CV window",
                                min_train_window=min_train_window, val_window=val_window, total_len=total_len)
                    return float("nan")

                # Merge per-trial feature config
                self.features_config.update(params)
                cfg = self.apply_feature_defaults()
                
                self._optuna_locked_keys = set(params.keys())
                

                self.features_config = cfg
                # Tripwire: warn if CV/time-caps changed Optuna keys
                if getattr(self, "_optuna_locked_keys", None):
                    _cl = {k for k in self._optuna_locked_keys
                        if k in self.features_config and self.features_config[k] != params.get(k)}
                    if _cl:
                        print(f"⚠️ Optuna keys were changed by CV/time-caps: {sorted(_cl)}")

                

                lags     = int(params.get("lags", params.get("lags_range", 5)))
                conf_thr = float(params.get("confidence_threshold", 0.0))
                model_type_local = params["model_type"]
                # cv_mode_local = str(config.get("cv_mode", "mini_block")).lower()
                
                #  # Month-aligned CV request (Patch M1 builds fold boundaries; Patch M2 will wire
                # # them into the training/validation slices). For now, we *prepare* the fold plan
                # # and keep evaluation on mini_block so behaviour stays stable until M2 lands.
                # monthly_roll_requested = cv_mode_local in {
                #     "monthly_roll", "monthly", "month", "month_roll", "rolling_month"
                # }
                # cv_mode_effective = "mini_block" if monthly_roll_requested else cv_mode_local
                
                # # Accept a future month-aligned CV mode name without breaking older runs.
                # # (Patch M1/M2 will implement this fully; for now we fall back to mini_block.)
                # if cv_mode_local in {"monthly_roll", "monthly", "month", "month_roll", "rolling_month"}:
                #     if bool(config.get("print_cv_debug", False)):
                #         print("[CV] monthly_roll requested but not yet implemented; falling back to mini_block.")
                #     cv_mode_local = "mini_block"
                cv_mode_local = str(config.get("cv_mode", "mini_block")).lower()

                # Month-aligned CV request ("monthly_roll" and aliases).
                # Implemented via month-aligned folds inside the CV loop (see _use_monthly below).
                monthly_roll_requested = cv_mode_local in {
                    "monthly_roll", "monthly", "month", "month_roll", "rolling_month"
                }

                # Normalize aliases so logs and downstream checks are unambiguous.
                cv_mode_effective = "monthly_roll" if monthly_roll_requested else cv_mode_local

                is_dqn_like = model_type_local in {"dqn"}

                # Header (if Optuna)
                try:
                    _print_trial_header_table(params, cfg, trial)
                    if monthly_roll_requested:
                        print(
                            f"🔎 CV geometry: mode={cv_mode_effective} "
                            f"| cv_blocks={int(config.get('cv_blocks', 5))} "
                            f"| val_months={float(config.get('cv_val_months', 1.0)):.2f} "
                            f"| cv_train_months={config.get('cv_train_months', None)}"
                    )
                    else:
                        print(
                            f"🔎 CV geometry: mode={cv_mode_effective} | K={int(config.get('cv_blocks', 5))} "
                            f"| embargo={int(config.get('cv_embargo_bars', 0))} "
                            f"| val_frac={float(config.get('cv_val_frac', 0.09)):.3f} "
                            f"| min_train_frac={float(config.get('cv_min_train_frac', 0.80)):.3f}"
                        )
                    print(f"🔒 Confidence threshold (requested): {conf_thr:.3f} "
                        f"| backoff_cv={bool(cfg.get('allow_conf_backoff_cv', False))} "
                        f"| floor_cv={float(cfg.get('conf_backoff_floor_cv', 0.33)):.2f}")
                except Exception:
                    pass
                

                # --- Month-aligned fold plan (prepared only; will be used in Patch M2) ---
                if monthly_roll_requested and not is_dqn_like:
                    def _safe_int_months(v, default=1):
                        try:
                            if v is None:
                                return int(default)
                            if isinstance(v, (list, tuple)):
                                v = v[0]
                            return max(1, int(round(float(v))))
                        except Exception:
                            return int(default)

                    def _build_monthly_roll_folds(df, k_blocks, train_months_eff, val_months_eff, embargo_bars_eff):
                        # "\"\"Return a fold plan as iloc ranges that align to calendar months.\"\"\"
                        if df is None or len(df) < 10:
                            return []
                        idx = df.index
                        if not hasattr(idx, "to_period"):
                            return []
                        months = pd.Index(idx.to_period("M")).unique().sort_values()
                        if len(months) == 0:
                            return []
                        k_use = min(int(k_blocks), len(months))
                        val_months = months[-k_use:]
                        folds = []
                        idx_values = idx.values
                        for j, m in enumerate(val_months, start=1):
                            val_start = m.start_time
                            val_next  = (m + val_months_eff).start_time
                            vs = int(np.searchsorted(idx_values, np.array(val_start, dtype=idx_values.dtype), side="left"))
                            ve = int(np.searchsorted(idx_values, np.array(val_next,  dtype=idx_values.dtype), side="left"))
                            if ve <= vs:
                                continue
                            tr_end = max(0, vs - int(embargo_bars_eff))
                            if tr_end <= 0:
                                continue
                            tr_end_time = idx[tr_end - 1]
                            tr_start_time = tr_end_time - pd.DateOffset(months=int(train_months_eff))
                            ts = int(np.searchsorted(idx_values, np.array(tr_start_time, dtype=idx_values.dtype), side="left"))
                            ts = max(0, min(ts, tr_end))
                            folds.append({
                                "fold": j,
                                "train_iloc": (ts, tr_end),
                                "val_iloc": (vs, ve),
                                "train_start": idx[ts] if ts < len(idx) else None,
                                "train_end": idx[tr_end - 1] if tr_end - 1 < len(idx) else None,
                                "val_start": idx[vs] if vs < len(idx) else None,
                                "val_end": idx[ve - 1] if ve - 1 < len(idx) else None,
                                "val_month": str(m),
                            })
                        return folds

                    k_blocks_cfg = int(config.get("cv_blocks", 5))
                    val_months_eff = _safe_int_months(config.get("cv_val_months", 1.0), default=1)
                    cv_train_months_cfg = config.get("cv_train_months", None)
                    if cv_train_months_cfg is None:
                        cv_train_months_cfg = config.get("train_months", None)
                    if isinstance(cv_train_months_cfg, (list, tuple)):
                        cv_train_months_cfg = cv_train_months_cfg[0]
                    if cv_train_months_cfg is None:
                        try:
                            cv_train_months_cfg = TRAIN_TEST_MONTHS[model_type_local]["train"][0]
                        except Exception:
                            cv_train_months_cfg = 12
                    train_months_eff = _safe_int_months(cv_train_months_cfg, default=12)
                    tb_max_holding_local = int(self.features_config.get("tb_max_holding", int(config.get("tb_max_holding", 0))))
                    embargo_bars_eff = max(int(config.get("cv_embargo_bars", 0)), tb_max_holding_local)
                    monthly_folds = _build_monthly_roll_folds(train_data, k_blocks_cfg, train_months_eff, val_months_eff, embargo_bars_eff)
                    setattr(self, "_cv_monthly_fold_plan", monthly_folds)
                    if bool(config.get("print_cv_debug", False)) and monthly_folds:
                        rows = []
                        for f in monthly_folds:
                            ts, te = f["train_iloc"]; vs, ve = f["val_iloc"]
                            rows.append([f["fold"], f["val_month"], f"{ts}:{te} ({max(0, te-ts)} bars)", f"{vs}:{ve} ({max(0, ve-vs)} bars)"])
                        print(_fmt_table(["Fold","ValMonth","Train(iloc)","Val(iloc)"], rows,
                                         title="🗓️  Monthly-roll CV fold plan"))

                # -------------------------------
                # Mini-block CV (preferred path)
                # -------------------------------
                if (cv_mode_effective in {"mini_block", "mini", "monthly_roll"}) and not is_dqn_like:

                    # ---- Tiny CV geometry cache (per-run, per-geometry) ----
                    # We only cache small integers (no DataFrames) so this cannot
                    # blow up RAM. Keyed by geometry-only knobs: data length,
                    # cv_blocks, val/min_train fractions and embargo.
                    geom_cache = getattr(self, "_cv_geom_cache", None)
                    if geom_cache is None:
                        geom_cache = {}
                        setattr(self, "_cv_geom_cache", geom_cache)

                    total_len = len(train_data)

                    # Base fractional geometry (depends only on config + total_len)
                    val_frac  = float(config.get("cv_val_frac", 0.09))
                    min_frac  = float(config.get("cv_min_train_frac", 0.80))
                    k_blocks_cfg = int(config.get("cv_blocks", 5))

                    # Stable identifiers for this train_data (month)
                    try:
                        idx0 = train_data.index[0]
                        idx1 = train_data.index[-1]
                    except Exception:
                        idx0 = ("len", total_len)
                        idx1 = None

                    cv_key = (
                        "mini_block_geom",
                        k_blocks_cfg,
                        val_frac,
                        min_frac,
                        int(config.get("cv_embargo_bars", 0)),
                        int(getattr(self, "features_config", {}).get(
                            "tb_max_holding",
                            int(config.get("tb_max_holding", 0)),
                        )),
                        int(total_len),
                        str(idx0),
                        str(idx1),
                    )

                    cached = geom_cache.get(cv_key)
                    if cached is not None:
                        # Fast path: reuse integers from previous trial
                        (
                            k_blocks,
                            tb_max_holding_local,
                            embargo_bars,
                            val_window_local,
                            min_train_local,
                            smin,
                            smax,
                        ) = cached
                    else:
                        # Slow path: compute geometry as before (only once per run)
                        k_blocks = k_blocks_cfg
                        tb_max_holding_local = int(
                            self.features_config.get(
                                "tb_max_holding",
                                int(config.get("tb_max_holding", 0)),
                            )
                        )
                        embargo_bars = max(
                            int(config.get("cv_embargo_bars", 0)),
                            tb_max_holding_local,
                        )

                        val_window_local = max(
                            30,
                            int(round(val_frac * total_len)),
                        )
                        min_train_local = int(round(min_frac * total_len))

                        smin = max(0, min_train_local + int(embargo_bars))
                        smax = total_len - val_window_local

                        # Store only small ints; no DataFrames are cached.
                        geom_cache[cv_key] = (
                            int(k_blocks),
                            int(tb_max_holding_local),
                            int(embargo_bars),
                            int(val_window_local),
                            int(min_train_local),
                            int(smin),
                            int(smax),
                        )

                    if smax <= smin:
                        _cv_penalty("MiniBlockCV invalid split geometry", smax=smax, smin=smin)
                        import optuna as _opt
                        raise _opt.TrialPruned("Broken CV geometry: cannot form blocks (no valid bars)")


                    # Exact-fit mode to preserve requested K blocks
                    exact = bool(config.get("cv_fit_blocks_exact", True))
                    if exact:
                        usable = (smax - smin)
                        if usable <= 0:
                            _cv_penalty("MiniBlockCV invalid geometry (usable<=0)", usable=usable, smax=smax, smin=smin)
                            import optuna as _opt
                            raise _opt.TrialPruned("Broken CV geometry: usable<=0")
                        needed = int(max(0, (k_blocks - 1)) * val_window_local)
                        if usable < needed:
                            new_val = max(30, usable // max(1, k_blocks))
                            if new_val < 30:
                                _cv_penalty("MiniBlockCV cannot shrink below 30 bars", new_val=new_val, K=k_blocks, usable=usable)
                                import optuna as _opt
                                raise _opt.TrialPruned("Broken CV geometry: val_window < 30")
                            if bool(config.get("print_cv_debug", False)):
                                print(f"[MiniBlockCV] Shrinking val_window from {val_window_local} → {new_val} to keep K={k_blocks}")
                            val_window_local = new_val
                            smax = total_len - val_window_local

                    # Build split starts (tail-anchored optional)
                    K = int(k_blocks)
                    while K > 1 and (smax - smin) < ((K - 1) * val_window_local):
                        K -= 1
                    if K < 1:
                        for k_try in range(min(k_blocks, 5), 1, -1):
                            K_try = (smax - smin) - (k_try - 1) * val_window_local
                            K_try = K_try // max(30, val_window_local)
                            if K_try >= 1:
                                k_blocks = k_try
                                K = K_try
                                break
                    if K < 1:
                        K = 1
                        k_blocks = 2
                        val_window_local = max(val_window_local, (smax - smin - embargo_bars) // 2)

                    tail_anchor = bool(config.get("cv_tail_anchor", True))
                    if (not tail_anchor) or (K == 0):
                        slack = (smax - smin) - (K * val_window_local)
                        gap   = 0 if K == 1 else max(0, slack // (K - 1))
                        splits = []
                        cursor = smin
                        for _ in range(K):
                            splits.append(cursor)
                            cursor += val_window_local + gap
                    else:
                        if K == 1:
                            splits = [smax]
                        else:
                            early = K - 1
                            avail_early = (smax - smin) - (early * val_window_local)
                            gap_early = 0 if early == 1 else max(0, avail_early // (early - 1))
                            splits = []
                            cursor = smin
                            for _ in range(early):
                                splits.append(cursor)
                                cursor += val_window_local + gap_early
                            splits.append(smax)

                    # Early-stopping knobs for deep models (treat all ensembles as deep)
                    is_ensemble = isinstance(model_type_local, str) and model_type_local.startswith("ensemble_")
                    is_deep = (model_type_local in {"cnn","lstm","transformer"}) or is_ensemble

                    # =========================
                    # CV pre-setup (deep caps)
                    # =========================
                    if is_deep:
                        cfg = self.apply_feature_defaults()
                        # Remember which keys Optuna actually sampled this trial
                        try:
                            self._optuna_locked_keys = set(params.keys())
                        except Exception:
                            self._optuna_locked_keys = set()

                        cfg["deep_eval_mode"] = "cv_fast"

                        # CV-only caps: install as defaults (won't override trial-set keys)
                        cfg.setdefault("deep_cv_max_epochs", 12)
                        cfg.setdefault("deep_cv_batch_size", 256)
                        cfg.setdefault("deep_cv_patience", 6)

                        # Per-model early stopping defaults (don’t override Optuna if set)
                        cfg.setdefault("cnn_use_early_stopping", True)
                        cfg.setdefault("lstm_use_early_stopping", True)
                        cfg.setdefault("transformer_use_early_stopping", True)

                        cv_pat = int(cfg.get("deep_cv_patience", 5))
                        cfg.setdefault("cnn_patience", cv_pat)
                        cfg.setdefault("lstm_patience", cv_pat)
                        cfg.setdefault("transformer_patience", cv_pat)

                        cfg["skip_perm_importance"] = True
                        self.features_config = cfg

                        # Print if anything Optuna chose got clobbered by CV caps
                        if getattr(self, "_optuna_locked_keys", None):
                            clobbered = {
                                k for k in self._optuna_locked_keys
                                if k in self.features_config and self.features_config[k] != params.get(k)
                            }
                            if clobbered:
                                print(f"⚠️ Optuna keys were changed by CV/time-caps: {sorted(clobbered)}")

                    # Timestamps for split starts (debug)
                    val_starts_ts = []
                    try:
                        for i in splits:
                            if 0 <= int(i) < len(train_data):
                                val_starts_ts.append(train_data.index[int(i)])
                    except Exception:
                        val_starts_ts = []

                    if bool(config.get("print_cv_debug", False)):
                        try:
                            _spl = list(map(int, splits))
                        except Exception:
                            _spl = splits
                        print(f"[MiniBlockCV] Using k={K} blocks | val_window={val_window_local} rows | "
                            f"embargo_bars={embargo_bars} | min_train_local={min_train_local} | "
                            f"splits(starts)={_spl} | lags={lags} | deep_fast_cv={is_deep}")
                        try:
                            _ts = [str(t) for t in val_starts_ts]
                            print(f"[MiniBlockCV] val_starts_ts={_ts}")
                        except Exception:
                            pass

                    # ------------------------------------------------------
                    # helpers for status + dict fmt (used by table printing)
                    # ------------------------------------------------------
                    # Read the generic CV gates once for the status helper
                    _M_gate = int(config.get("cv_min_trades_per_block", 5))
                    _r_min  = float(config.get("cv_gate_min_active_rate", 0.02))
                    
                    _relax = float(config.get("cv_prune_relax", 1.0))
                    _relax = max(0.0, min(_relax, 1.0))
                    
                    if _relax <= 0.0:
                        # Disable Thin-gating: we still flag blatant errors (NoTrades, NaN, etc.)
                        _M_gate_eff = 0
                        _r_min_eff = 0.0
                    else:
                        # Larger thresholds = stricter ⇒ scale down to relax
                        _M_gate_eff = max(0, int(round(_M_gate * _relax)))
                        _r_min_eff  = max(0.0, _r_min * _relax)

                    def _status_for_block(trades, sr, active, all_hold=False, pruned=False):
                        if pruned:
                            return "🪓 prune"
                        if (trades is None) or (int(trades) <= 0) or all_hold:
                            return "⛔ NoTrades"
                        if (sr is None) or (not np.isfinite(sr)):
                            return "⛔ SRNaN"
                        if (int(trades) < _M_gate_eff) or (
                            active is not None and np.isfinite(active) and float(active) < _r_min_eff
                        ):
                            return "🟡 Thin"
                        if float(sr) < 0:
                            return "🔴 Bad"
                        return "🟢 OK"

                    def _fmt_dict(d):
                        try:
                            if isinstance(d, dict):
                                return "{" + ", ".join(f"{k}: {d[k]}" for k in sorted(d)) + "}"
                        except Exception:
                            pass
                        return str(d)
                    
                    def _early_structural_prune_if_hopeless():
                        """
                        Early-stop a trial during MiniBlockCV when:
                        (A) The first few folds are all invalid / no-trades (degenerate config).
                        (B) Even in the best case, we cannot hit the required active-fold coverage.

                        (B) is aligned with the final coverage gate:
                        we only trigger when this trial would be hopeless anyway under the
                        cv_min_coverage target. (A) is a pragmatic speed-up for configs that
                        clearly produce 0 trades everywhere.
                        """
                        # Need Optuna + numpy; otherwise do nothing.
                        try:
                            import numpy as _np
                            import optuna as _opt
                        except Exception:
                            return

                        # How many folds are planned
                        try:
                            K_plan = len(splits)
                        except Exception:
                            K_plan = int(config.get("cv_blocks", 5)) or 1
                        if K_plan <= 1:
                            return

                        processed = len(block_scores)
                        if processed <= 0:
                            return

                        arr = _np.asarray(block_scores[:processed], dtype=float)
                        k_valid = int(_np.isfinite(arr).sum())
                        remaining = max(0, K_plan - processed)

                        # ── (A) Degenerate no-trades heuristic ─────────────────────────────
                        # If we've already evaluated N folds and NONE produced a valid score,
                        # this config is effectively dead: thresholds/gating too strict.
                        # Default N=2; can be tuned via cv_early_all_invalid_patience.
                        patience = int(config.get("cv_early_all_invalid_patience", 2))
                        if processed >= patience and k_valid == 0:
                            msg = (f"[MiniBlockCV:EARLY_DEGENERATE] "
                                   f"{processed} folds, all invalid/no-trades → prune trial")
                            if bool(config.get("print_cv_debug", False)):
                                print(msg)
                            raise _opt.TrialPruned(msg)

                        # ── (B) Structural coverage hopelessness ───────────────────────────
                        # Use cv_min_coverage as the design target. We only cut when even if
                        # all remaining folds were perfect, we cannot reach that coverage.
                        min_cov_base = float(config.get("cv_min_coverage", 0.80))
                        min_cov_base = max(0.0, min(1.0, min_cov_base))

                        # For early-hopeless logic we *do not* weaken this with cv_prune_relax.
                        # If you want it tied to relax, replace min_cov_base with
                        # (min_cov_base * cv_relax).
                        max_possible_valid = k_valid + remaining
                        required_valid = min_cov_base * K_plan

                        if max_possible_valid < required_valid:
                            msg = (f"[MiniBlockCV:EARLY_COVERAGE_PRUNE] "
                                   f"k_valid={k_valid}, remaining={remaining}, "
                                   f"K_plan={K_plan}, min_cov={min_cov_base:.2f}")
                            if bool(config.get("print_cv_debug", False)):
                                print(msg)
                            raise _opt.TrialPruned(msg)

                    
                    # --------------------------------------------
                    # Collectors BEFORE the mini-block evaluation
                    # --------------------------------------------
                    block_scores, block_active_rates, block_trades = [], [], []
                    block_eff_conf, block_rows, block_pruned, block_all_hold = [], [], [], []
                    block_train_rows = []      # NEW: train rows per fold (for tables)
                    block_precision_intent = []
                    block_intent_bars = []


                    block_reasons = []          # human-readable gating reason per block
                    pred_cards, val_ends_ts = [], []
                    val_starts_ts_cv = []
                    val_ends_ts_cv = []
                    
                    block_psr, block_neff = [], []
                    block_sharpe = []           # raw after-cost Sharpe per block
                    block_cov_thr = []          # per-block coverage thresholds (base)
                    
                    # Per-regime accumulators (0=SIDEWAYS,1=TREND,2=VOLATILE)
                    regime_stats = {
                        0: {"sum_ret": 0.0, "sum_ret_sq": 0.0, "trades": 0, "bars": 0},
                        1: {"sum_ret": 0.0, "sum_ret_sq": 0.0, "trades": 0, "bars": 0},
                        2: {"sum_ret": 0.0, "sum_ret_sq": 0.0, "trades": 0, "bars": 0},
                    }
                    
                    # NEW: per-fold eval frame collector for CV diagnostics
                    fold_eval_frames: list = []        # each fold's evaluation DataFrame (if available)
                    per_fold_regime_trades = {0: [], 1: [], 2: []}
                    per_fold_regime_active = {0: [], 1: [], 2: []}
                    per_fold_regime_sharpe = {0: [], 1: [], 2: []}

                    # -------------------------
                    # Evaluate each mini-block
                    # -------------------------
                    # If monthly-roll CV was requested and a fold plan exists (prepared in Patch M1),
                    # use it. Otherwise, default to the existing expanding-window mini-block logic.
                    _cv_mode_req = str(config.get("cv_mode", "mini_block")).lower()
                    _monthly_req = _cv_mode_req in {"monthly_roll","monthly","month","month_roll","rolling_month"}
                    monthly_folds = getattr(self, "_cv_monthly_fold_plan", None) if _monthly_req else None
                    _use_monthly = bool(monthly_folds) and isinstance(monthly_folds, list)
                    
                    # --- Guardrails for monthly fold geometry (Patch M3) ---
                    if _use_monthly:
                        bpm = int(config.get("bars_per_month_hint", 1000))
                        val_m = float(config.get("cv_val_months", 1.0))
                        exp_val_bars = max(10, int(round(bpm * val_m)))

                        # Defaults chosen to be conservative and stable:
                        # - train must be at least ~3 months worth of bars
                        # - val must be at least 60% of expected month bars (handles missing days / session filters)
                        min_train_months = float(config.get("cv_min_train_months_monthly", 3.0))
                        min_val_frac     = float(config.get("cv_min_val_frac_monthly", 0.60))
                        min_valid_folds  = int(config.get("cv_min_valid_folds_monthly", 3))

                        min_train_bars = max(50, int(round(bpm * min_train_months)))
                        min_val_bars   = max(10, int(round(exp_val_bars * min_val_frac)))

                        filtered = []
                        dropped_rows = []
                        for f in monthly_folds:
                            try:
                                ts, te = f.get("train_iloc", (0, 0))
                                vs, ve = f.get("val_iloc", (0, 0))
                                ts, te, vs, ve = int(ts), int(te), int(vs), int(ve)
                            except Exception:
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), "bad iloc parse"])
                                continue

                            if not (0 <= ts < te <= len(train_data) and 0 <= vs < ve <= len(train_data)):
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), "iloc out of range"])
                                continue

                            train_n = te - ts
                            val_n   = ve - vs

                            if train_n < min_train_bars:
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), f"train too short ({train_n} < {min_train_bars})"])
                                continue
                            if val_n < min_val_bars:
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), f"val too short ({val_n} < {min_val_bars})"])
                                continue

                            filtered.append(f)

                        if bool(config.get("print_cv_debug", False)) and dropped_rows:
                            print(_fmt_table(["Fold","ValMonth","DroppedReason"], dropped_rows,
                                             title="⚠️ Monthly-roll CV: dropped folds (guardrails)"))

                        if len(filtered) >= min_valid_folds:
                            monthly_folds = filtered
                            _use_monthly = True
                        else:
                            if bool(config.get("print_cv_debug", False)):
                                print(f"[CV] Monthly-roll folds valid={len(filtered)} < {min_valid_folds}; falling back to mini_block for this trial.")
                            _use_monthly = False


                    if _use_monthly and bool(config.get("print_cv_debug", False)):
                        print(f"[CV] Using monthly-roll fold slicing for this trial (K={len(monthly_folds)})")
                        
                   # Remember what we actually used this trial so the summary table title matches reality
                    try:
                        setattr(self, "_cv_used_monthly_last", bool(_use_monthly))
                    except Exception:
                        pass

                    fold_iter = monthly_folds if _use_monthly else splits
                    fold_label = "CV Fold" if _use_monthly else "Mini-Block Fold"

                    # (B1) Stable per-fold record slots (prevents index drift when folds are pruned/skipped).
                    # NOTE: We keep all existing parallel arrays for compatibility in this patch.
                    fold_records = [None] * int(len(fold_iter))
                    if bool(config.get("print_cv_debug", False)):
                        print(f"[CV][FoldRecords] init slots={len(fold_records)}")
                        
                        
                    # ------------------------------------------------------------------
                    # Single source of truth: normalize cv_config ONCE, before fold loop.
                    # Baseline: nested config["cv_config"]; allow a few flat keys in
                    # `config` to override for backward compatibility.
                    # ------------------------------------------------------------------
                    cv_config = dict(config.get("cv_config", {}) or {})
                    for _k in (
                        "cv_prune_precision_intent",
                        "cv_prune_min_precision_intent",
                        "cv_prune_min_intent_bars_fold",
                        "cv_prune_min_intent_bars",
                    ):
                        if _k in config:
                            cv_config[_k] = config.get(_k)
                            
                    # ----------------------------------------------------------------
                    #   Snapshot/restore all _last_* and _cv_last_* attrs + results fields
                    #   around any diagnostic pass, and scream if the fold guard changes.
                    # ------------------------------------------------------------------
                    def _cv__clone_state(v):
                        try:
                            import pandas as _pd
                            import numpy as _np
                        except Exception:
                            _pd = None
                            _np = None
                        try:
                            from copy import deepcopy as _deepcopy
                        except Exception:
                            _deepcopy = None
                        try:
                            if _pd is not None and isinstance(v, _pd.DataFrame):
                                return v.copy(deep=True)
                            if _pd is not None and isinstance(v, _pd.Series):
                                return v.copy(deep=True)
                            if _np is not None and isinstance(v, _np.ndarray):
                                return v.copy()
                            if isinstance(v, (dict, list, tuple)) and _deepcopy is not None:
                                return _deepcopy(v)
                        except Exception:
                            pass
                        return v

                    def _cv__snapshot_state_for_diagnostics():
                        keys = {"results", "results_full", "_cv_last_eval_df", "_last_eligibility_diag"}
                        try:
                            for k in list(getattr(self, "__dict__", {}).keys()):
                                if k.startswith("_last_") or k.startswith("_cv_last_"):
                                    keys.add(k)
                        except Exception:
                            pass
                        snap, present = {}, {}
                        try:
                            d = getattr(self, "__dict__", {})
                            for k in keys:
                                present[k] = (k in d)
                                if present[k]:
                                    snap[k] = _cv__clone_state(d.get(k))
                        except Exception:
                            pass
                        return snap, present

                    def _cv__restore_state_after_diagnostics(snap, present):
                        for k, was_present in (present or {}).items():
                            if was_present:
                                try:
                                    setattr(self, k, snap.get(k))
                                except Exception:
                                    pass
                            else:
                                if hasattr(self, k):
                                    try:
                                        delattr(self, k)
                                    except Exception:
                                        pass
                        try:
                            d_now = getattr(self, "__dict__", {})
                            for k in list(d_now.keys()):
                                if (k.startswith("_last_") or k.startswith("_cv_last_")) and (k not in (present or {})):
                                    try:
                                        delattr(self, k)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    @contextmanager
                    def _cv_diagnostic_guard(ctx="cv:diagnostic"):
                        snap, present = _cv__snapshot_state_for_diagnostics()
                        g_uuid   = getattr(self, "_cv_fold_guard_uuid", None)
                        g_trades = getattr(self, "_cv_fold_guard_trades_main", None)
                        try:
                            yield
                        finally:
                            try:
                                a_uuid   = getattr(self, "_cv_fold_guard_uuid", None)
                                a_trades = getattr(self, "_cv_fold_guard_trades_main", None)
                                if (a_uuid != g_uuid) or (a_trades != g_trades):
                                    print(
                                        f"🚨 [CV][Patch7] Diagnostic mutated fold guard | ctx={ctx} | "
                                        f"uuid {g_uuid}→{a_uuid} | trades {g_trades}→{a_trades}"
                                    )
                            except Exception:
                                pass
                            _cv__restore_state_after_diagnostics(snap, present)
                            try:
                                self._cv_fold_guard_uuid = g_uuid
                                self._cv_fold_guard_trades_main = g_trades
                            except Exception:
                                pass


                    for j, fold in enumerate(fold_iter, start=1):
                        if _use_monthly:
                            # fold is a dict like: {"train_iloc": (ts, te), "val_iloc": (vs, ve), ...}
                            try:
                                ts, te = fold.get("train_iloc", (0, 0))
                                vs, ve = fold.get("val_iloc", (0, 0))
                            except Exception:
                                ts = te = vs = ve = 0
                            tr  = train_data.iloc[int(ts):int(te)]
                            val = train_data.iloc[int(vs):int(ve)]
                            split = int(vs)  # for logging/penalty context below
                        else:
                            split = fold
                            tr_end_idx = max(0, split - embargo_bars)
                            tr         = train_data.iloc[:tr_end_idx]
                            val        = train_data.iloc[split : split + val_window_local]

                        # record rows + val_end ts
                        # (NEW) also record train rows + true val_start ts for accurate tables
                        try:
                            block_train_rows.append(int(len(tr)))
                        except Exception:
                            block_train_rows.append(0)

                        rows_i = len(val)
                        block_rows.append(rows_i)
                        try:
                            vstart_ts = val.index[0] if rows_i > 0 else None
                        except Exception:
                            vstart_ts = None
                        val_starts_ts_cv.append(vstart_ts)
                        try:
                            vend_ts = val.index[-1] if rows_i > 0 else None
                        except Exception:
                            vend_ts = None

                        val_ends_ts.append(vend_ts)
                        val_ends_ts_cv.append(vend_ts)

                        # Size sanity
                        if len(tr) < max(100, lags + 5) or len(val) < max(20, lags + 1):
                            _cv_penalty(
                                "MiniBlockCV reject split (insufficient rows)",
                                split_start=int(split),
                                len_tr=len(tr),
                                min_len_tr=max(100, lags + 5),
                                len_val=len(val),
                                min_len_val=max(20, lags + 1),
                            )

                            block_scores.append(float("nan"))
                            block_active_rates.append(0.0)
                            block_trades.append(0)
                            block_eff_conf.append(float("nan"))
                            block_sharpe.append(float("nan"))
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))
                            block_pruned.append(False)
                            block_all_hold.append(True)
                            block_reasons.append("TooShort")

                            # (B1/B2-minimal) Commit a fold record into the fixed slot BEFORE continue.
                            # This is the key to preventing "parallel array drift" for this early-exit branch.
                            try:
                                fold_records[int(j) - 1] = {
                                    "fold_idx": int(j),
                                    "train_rows": int(len(tr)),
                                    "val_rows": int(rows_i),
                                    "vstart": vstart_ts,     # prefer these keys everywhere
                                    "vend": vend_ts,

                                    # denoms (safe defaults — eligibility not computed in this early exit)
                                    "post_feature_bars_total": int(rows_i),
                                    "post_feature_eligible": int(rows_i),
                                    "eval_bars": int(rows_i),

                                    "score": float("nan"),
                                    "sharpe": float("nan"),
                                    "psr": float("nan"),
                                    "trades": 0,
                                    "active_rate": 0.0,
                                    "precision_trade": float("nan"),
                                    "n_trade_preds": 0,
                                    "eff_conf": float("nan"),
                                    "reason": "TooShort",
                                    "pruned": False,
                                    "all_hold": True,
                                    "status": "skipped",
                                }
                            except Exception:
                                pass

                            # Pretty debug card for this invalid fold
                            print_pruned_block_summary(
                                block_id=j,
                                reason="MiniBlockCV reject split (insufficient rows)",
                                rows=rows_i,
                                trades=0,
                                active_rate=0.0,
                                sharpe=float("nan"),
                                fold_label=fold_label,
                            )

                            _early_structural_prune_if_hopeless()
                            continue

                        # =========================
                        # Evaluate this mini-block
                        # =========================
                        try:
                            # Reset per-block 'used' nudge params (avoids log bleed from prior blocks)
                            try:
                                self._last_runtime_active_band_used = None
                                self._last_runtime_conf_step_used = None
                            except Exception:
                                pass
                            
                            # Evaluate block (generic: any "ensemble_*" → test_ensemble_strategy)
                            if (
                                isinstance(model_type_local, str)
                                and model_type_local.startswith("ensemble_")
                            ):
                                metrics = self.test_ensemble_strategy(
                                    train_start=tr.index[0],
                                    train_end=tr.index[-1],
                                    test_start=val.index[0],
                                    test_end=val.index[-1],
                                    lags=lags,
                                    label_threshold=params.get(
                                        "label_threshold", 0.0
                                    ),
                                    ensemble_config=params,
                                    model_type=model_type_local,
                                )
                            else:
                                metrics = self.test_strategy(
                                    train_start=tr.index[0],
                                    train_end=tr.index[-1],
                                    test_start=val.index[0],
                                    test_end=val.index[-1],
                                    lags=lags,
                                    confidence_threshold=conf_thr,
                                    label_threshold=params.get(
                                        "label_threshold", 0.0
                                    )
                                )

                            # Require 16-tuple
                            if (not isinstance(metrics, tuple)) or len(metrics) != 16:
                                _cv_penalty(
                                    "MiniBlockCV metrics malformed",
                                    mtype=type(metrics).__name__,
                                    mlen=(
                                        len(metrics)
                                        if hasattr(metrics, "__len__")
                                        else "NA"
                                    ),
                                )

                                block_scores.append(float("nan"))
                                block_active_rates.append(0.0)
                                block_trades.append(0)
                                block_eff_conf.append(float("nan"))
                                block_sharpe.append(float("nan"))
                                block_psr.append(float("nan"))
                                block_neff.append(float("nan"))
                                block_pruned.append(False)
                                block_all_hold.append(True)
                                block_reasons.append("BadMetrics")

                                print_pruned_block_summary(
                                    block_id=j,
                                    reason="MiniBlockCV metrics malformed",
                                    rows=rows_i,
                                    fold_label=fold_label,
                                )
                                
                                try:
                                    _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                    _pf_total = int(_diag.get("bars_total", rows_i) or rows_i)
                                    _pf_elig  = int(_diag.get("eligible_bars", _pf_total) or _pf_total)
                                    _eval_bars = int(getattr(self, "_last_eval_bars", _pf_elig) or _pf_elig)
                                    fold_records[int(j) - 1] = {
                                        "vstart": vstart_ts,
                                        "vend": vend_ts,
                                        "train_rows": int(len(tr)),
                                        "val_rows": int(rows_i),
                                        "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                        "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                        "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                        "psr": float("nan"),
                                        "trades": 0,
                                        "active_rate": 0.0,
                                        "precision_trade": float("nan"),
                                        "n_trade_preds": 0,
                                        "sharpe": float("nan"),
                                        "reason": "BadMetrics",
                                        "pruned": False,
                                        "status": "invalid",
                                    }
                                except Exception:
                                    pass

                                
                                _early_structural_prune_if_hopeless()
                                continue


                            # Unpack metrics
                            (
                                perf,
                                outperf,
                                creturns,
                                sharpe,
                                drawdown,
                                trades,
                                _,
                                _,
                                _,
                                _,
                                active_rate,
                                *_rest,
                            ) = metrics
                            
                            # ============================================================
                            # - Prevents stale reuse of _cv_last_eval_df from prior fold
                            # - Ensures per-regime outputs for invalid folds are NaN/empty
                            # ============================================================
                            _diag_valid = True
                            try:
                                _diag_valid = (int(trades) > 0) and np.isfinite(float(sharpe))
                            except Exception:
                                _diag_valid = False

                            if not _diag_valid:
                                # ------------------------------------------------------------
                                # (e.g., 0 trades after filtering). Without this, the monthly
                                # overview later shows "NO DATA / PRUNED" despite the fold
                                # having run, causing the historic index-drift bug.
                                # ------------------------------------------------------------
                                try:
                                    trades_int = int(trades) if trades is not None else 0
                                except Exception:
                                    trades_int = 0

                                try:
                                    ar_pr = float(active_rate) if (active_rate is not None and np.isfinite(float(active_rate))) else 0.0
                                except Exception:
                                    ar_pr = 0.0

                                reason_tag = "NoTrades" if trades_int <= 0 else "InvalidSR"
                                
                                try:
                                    print_pruned_block_summary(
                                        block_id=j,
                                        reason=reason_tag,
                                        rows=rows_i,
                                        trades=trades_int,
                                        active_rate=float(ar_pr),
                                        sharpe=float(sharpe) if (sharpe is not None and np.isfinite(float(sharpe))) else float("nan"),
                                        fold_label=fold_label,
                                    )
                                except Exception:
                                    pass

                                # Append fold-aligned placeholders (so len(block_*) increments)
                                block_scores.append(float("nan"))
                                block_active_rates.append(float(ar_pr))
                                block_trades.append(int(trades_int))
                                block_eff_conf.append(float("nan"))
                                block_sharpe.append(float("nan"))
                                block_psr.append(float("nan"))
                                block_neff.append(float("nan"))
                                block_pruned.append(False)
                                block_all_hold.append(True)
                                pred_cards.append({})
                                block_reasons.append(reason_tag)

                                # Commit fold-local record into the fixed slot
                                try:
                                    fold_records[int(j) - 1] = {
                                        "vstart": vstart_ts,
                                        "vend": vend_ts,
                                        "train_rows": int(len(tr)),
                                        "val_rows": int(rows_i),
                                        "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                        "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                        "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                        "psr": float("nan"),
                                        "trades": int(trades_int),
                                        "active_rate": float(ar_pr),
                                        "precision_trade": float("nan"),
                                        "n_trade_preds": 0,
                                        "sharpe": float("nan"),
                                        "reason": reason_tag,
                                        "pruned": False,
                                        "status": "invalid",
                                    }
                                except Exception:
                                    pass

                                # Hard-stop any possibility of stale fold reuse.
                                try:
                                    self._cv_last_eval_df = None
                                except Exception:
                                    pass

                                # Keep fold alignment: add an empty placeholder frame
                                # so later per-regime table prints NaNs for this fold.
                                try:
                                    _empty = pd.DataFrame(
                                        {
                                            "regime_id": pd.Series(dtype=int),
                                            "pred": pd.Series(dtype=float),
                                        }
                                    )
                                    fold_eval_frames.append(_empty)
                                except Exception:
                                    pass

                                # Also append fold-aligned per-regime placeholders.
                                try:
                                    for rid in (0, 1, 2):
                                        per_fold_regime_trades[rid].append(0)
                                        per_fold_regime_active[rid].append(float("nan"))
                                        per_fold_regime_sharpe[rid].append(float("nan"))
                                except Exception:
                                    pass

                                _early_structural_prune_if_hopeless()

                                # Skip diagnostic recomputation entirely for this fold.
                                continue

                            
                            # store per-fold evaluation frame for downstream per-regime CV diagnostics
                            # (fold-local copy: diagnostics must never mutate the fold's official result)
                            try:
                                _df0 = getattr(self, "_cv_last_eval_df", None)
                                df_fold = _df0.copy(deep=True) if isinstance(_df0, pd.DataFrame) else None
                                if df_fold is not None and (not df_fold.empty):
                                    # already copied above
                                    fold_eval_frames.append(df_fold)

                                    # best-effort per-regime simple stats for this fold (non-mutating)
                                    df_r = df_fold
                                    if "regime_id" not in df_r.columns:
                                        df_r = df_r.copy()
                                        df_r["regime_id"] = 1
                                        
                                    # ------------------------------------------------------------
                                    # DIAGNOSTIC-ONLY regime id (does NOT affect training/features)
                                    # Use cfg_high_vol_thr when available to avoid "all volatile"
                                    # collapse in regime logs/tables.
                                    # ------------------------------------------------------------
                                    try:
                                        if "regime_id_diag" not in df_r.columns:
                                            _cfgd = {}
                                            try:
                                                _cfgd = dict(getattr(df_r, "attrs", {}).get("features_config", {}) or {})
                                            except Exception:
                                                _cfgd = dict(getattr(self, "features_config", {}) or {})

                                            # choose columns
                                            adx_col = "adx"
                                            if adx_col not in df_r.columns:
                                                adx_w = int(_cfgd.get("adx_window_core", 14))
                                                if f"adx_{adx_w}" in df_r.columns:
                                                    adx_col = f"adx_{adx_w}"
                                            vol_col = None
                                            rv_w = int(_cfgd.get("rv_window_short", 48))
                                            if f"rv_{rv_w}" in df_r.columns:
                                                vol_col = f"rv_{rv_w}"
                                            elif "rv" in df_r.columns:
                                                vol_col = "rv"

                                            adx_thr = float(_cfgd.get("adx_thresh", 20.0))
                                            # prefer train-anchored threshold printed in logs
                                            vol_thr = _cfgd.get("high_vol_thr", None)
                                            if vol_thr is None:
                                                vol_thr = float(_cfgd.get("vol_thresh", 0.001))
                                            else:
                                                vol_thr = float(vol_thr)

                                            if vol_col is not None and adx_col in df_r.columns:
                                                _adx = df_r[adx_col].astype(float).fillna(0.0)
                                                _vol = df_r[vol_col].astype(float).fillna(0.0)
                                                vol_high = (_vol > vol_thr)
                                                trend = (_adx > adx_thr)
                                                # 2=volatile, 1=trend, 0=sideways
                                                df_r = df_r.copy()
                                                df_r["regime_id_diag"] = np.where(vol_high, 2, np.where(trend, 1, 0)).astype(int)
                                    except Exception:
                                        pass

                                    _rid_col = "regime_id_diag" if "regime_id_diag" in df_r.columns else "regime_id"


                                    for rid in (0, 1, 2):
                                        sub = df_r[df_r[_rid_col] == rid]
                                        if sub is None or len(sub) == 0:
                                            continue

                                        # Use whichever prediction column exists (prefer executed "pred")
                                        pred_col = None
                                        for _c in ("pred", "pred_exec", "final_pred", "prediction"):
                                            if _c in sub.columns:
                                                pred_col = _c
                                                break
                                        if pred_col is None:
                                            continue

                                        s = sub[pred_col].fillna(0)
                                        trades_i = int((s != 0).sum())
                                        active_i = float((s != 0).mean())

                                        sharpe_i = float("nan")
                                        try:
                                            _sub_eval = sub.copy()
                                            try:
                                                if bool(getattr(self, "trading_costs", False)):
                                                    _cfg_cost = {}
                                                    try:
                                                        _cfg_cost = dict(getattr(df_r, "attrs", {}).get("features_config", {}) or {})
                                                    except Exception:
                                                        _cfg_cost = dict(getattr(self, "features_config", {}) or {})
                                                    try:
                                                        _sub_eval.attrs["features_config"] = dict(_cfg_cost)
                                                    except Exception:
                                                        pass
                                                    _sub_eval = self._ensure_cost_columns(_sub_eval, _cfg_cost)
                                            except Exception:
                                                pass

                                            _m = compute_full_evaluation_metrics(
                                                df=_sub_eval,
                                                trading_costs=self.trading_costs,
                                                slippage_factor=self.slippage_factor,
                                                eval_context=f"cv:diagnostic:per_regime_metrics:rid={rid}",
                                            )
                                            sharpe_i = float(_m[3]) if _m is not None else float("nan")
                                        except Exception:
                                            pass

                                        per_fold_regime_trades[rid].append(trades_i)
                                        per_fold_regime_active[rid].append(active_i)
                                        per_fold_regime_sharpe[rid].append(sharpe_i)
                            except Exception:
                                pass

                        except Exception as e:
                            # If an inner component requested pruning (Optuna), propagate it.
                            try:
                                import optuna as _opt
                                if isinstance(e, _opt.TrialPruned):
                                    raise
                            except Exception:
                                pass
                            _cv_penalty(
                                "MiniBlockCV exception during block evaluation",
                                split_start=int(split),
                                error=str(e),
                            )
                            if bool(config.get("print_cv_debug", False)):
                                traceback.print_exc()

                            # Detect Optuna structural prune vs generic exception
                            _is_pruned = False
                            try:
                                import optuna as _opt
                                _is_pruned = isinstance(e, _opt.TrialPruned)
                            except Exception:
                                pass

                            reason = f"Pruned: {e}" if _is_pruned else f"Exception: {e}"

                            block_scores.append(float("nan"))
                            block_sharpe.append(float("nan"))
                            block_active_rates.append(0.0)
                            block_trades.append(0)
                            block_eff_conf.append(float("nan"))
                            block_pruned.append(_is_pruned)
                            block_all_hold.append(False)
                            pred_cards.append({})
                            block_reasons.append(reason)
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))

                            # Pretty card for this invalid/pruned fold
                            print_pruned_block_summary(
                                block_id=j,
                                reason=reason,
                                rows=rows_i,
                                fold_label=fold_label,
                            )

                            continue

                        try:
                            # These will be set inside test_strategy/test_ensemble_strategy
                            brier_f = float(getattr(self, "_last_calib_brier", float("nan")))
                            nll_f   = float(getattr(self, "_last_calib_nll",   float("nan")))
                            n_f     = int(getattr(self, "_last_calib_n",       0))
                            if (
                                n_f > 0
                                and np.isfinite(brier_f)
                                and np.isfinite(nll_f)
                            ):
                                calib_brier_sum += brier_f * n_f
                                calib_nll_sum   += nll_f   * n_f
                                calib_n_samples += n_f
                                if bool(config.get("print_cv_debug", False)):
                                    print(
                                        f"[CV-Calib-Fold] brier={brier_f:.6f} | "
                                        f"nll={nll_f:.6f} | n={n_f}"
                                    )
                        except Exception as _e:
                            if bool(config.get("print_cv_debug", False)):
                                print(f"[CV-Calib-Fold] Failed to accumulate calibration: {_e}")
                        
                        # Basic validity after successful metrics retrieval
                        _invalid_sharpe = (
                            sharpe is None
                            or (not np.isfinite(sharpe))
                            or (float(sharpe) <= -9998.0)
                        )

                        _no_trades = (
                            trades is None
                            or (not np.isfinite(trades))
                            or (int(trades) <= 0)
                        )
                        if _invalid_sharpe or _no_trades:
                            # mark invalid block uniformly (no-trade or broken)
                            # IMPORTANT: append to each block_* list EXACTLY ONCE
                            try:
                                trades_int = int(trades) if (trades is not None and np.isfinite(trades)) else 0
                            except Exception:
                                trades_int = 0
                                
                            try:
                                ar_pr = float(active_rate) if (active_rate is not None and np.isfinite(active_rate)) else 0.0
                            except Exception:
                                ar_pr = 0.0

                            block_scores.append(float("nan"))
                            block_active_rates.append(float(ar_pr))
                            block_trades.append(int(trades_int))
                            block_eff_conf.append(float("nan"))
                            block_pruned.append(False)

                            _final_after = getattr(
                                self, "_last_final_preds_dist", None
                            )
                            _all_hold = False
                            try:
                                if isinstance(_final_after, dict):
                                    non_hold = sum(
                                        v
                                        for k, v in _final_after.items()
                                        if str(k) != "0"
                                    )
                                    _all_hold = (int(non_hold) == 0)
                            except Exception:
                                pass

                            block_all_hold.append(bool(_all_hold))
                            pred_cards.append(
                                {
                                    "label_counts": getattr(
                                        self, "_last_label_counts", None
                                    ),
                                    "thr": getattr(
                                        self,
                                        "_last_label_threshold",
                                        params.get("label_threshold", None),
                                    ),
                                    "test_len": getattr(
                                        self, "_last_test_len", rows_i
                                    ),
                                    "raw_preds": getattr(
                                        self,
                                        "_last_raw_pred_dist",
                                        None,
                                    ),
                                    "decoded_before": getattr(
                                        self,
                                        "_last_decoded_preds_before",
                                        None,
                                    ),
                                    "final_after_thr": getattr(
                                        self,
                                        "_last_final_preds_dist",
                                        None,
                                    ),
                                }
                            )
                            
                            reason_tag = "NoTrades" if _no_trades else "InvalidSR"
                            block_reasons.append(reason_tag)
                            

                            print_pruned_block_summary(
                                block_id=j,
                                reason=reason_tag,
                                rows=rows_i,
                                trades=trades_int,
                                active_rate=float(ar_pr),
                                fold_label=fold_label,
                            )

                            try:
                                fold_records[int(j) - 1] = {
                                    "vstart": vstart_ts,
                                    "vend": vend_ts,
                                    "train_rows": int(len(tr)),
                                    "val_rows": int(rows_i),
                                    "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                    "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                    "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                    "psr": float("nan"),
                                    "trades": int(trades_int),
                                    "active_rate": float(ar_pr),
                                    "precision_trade": float(getattr(self, "_last_precision_trade", float("nan"))),
                                    "n_trade_preds": int(getattr(self, "_last_n_trade_preds", 0) or 0),
                                    "sharpe": float("nan"),
                                    "reason": reason_tag,
                                    "pruned": False,
                                    "status": "invalid",
                                }
                            except Exception:
                                pass


                            block_sharpe.append(float("nan"))
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))
                            
                            _early_structural_prune_if_hopeless()
                            continue



                        # --- Research-aligned gates & penalties ---
                        trades_i = int(trades)
                        ar = (
                            float(active_rate)
                            if (active_rate is not None and np.isfinite(active_rate))
                            else 0.0
                        )

                        # Estimate average holding in bars and #independent bets (Lopez de Prado style)
                        # active_rate ≈ (trades * avg_hold) / rows  ⇒ avg_hold ≈ (ar * rows) / trades
                        avg_hold_bars = (
                            float("inf")
                            if trades_i <= 0
                            else (ar * float(rows_i)) / float(trades_i)
                        )
                        indep_bets_est = (
                            0.0
                            if (not np.isfinite(avg_hold_bars) or avg_hold_bars <= 0)
                            else (float(rows_i) / (2.0 * avg_hold_bars))
                        )
                        indep_bets_est = (
                            float(min(float(trades_i), indep_bets_est))
                            if np.isfinite(indep_bets_est)
                            else 0.0
                        )

                        # Activity band around per-trial target
                        # Center the CV active-rate band on the *per-trial* (per-family) target.
                        # NOTE: `config` here is CV config (self.config). The target is a feature-level policy.
                        _f_cfg = getattr(self, "features_config", {}) or {}
                        ar_target = float(_f_cfg.get("target_active_rate", config.get("target_active_rate", 0.10)))
                        ar_margin = float(
                            config.get("cv_active_rate_margin", 0.12)
                        )  # ± absolute
                        ar_low = float(
                            config.get(
                                "cv_active_rate_low",
                                max(0.01, ar_target - ar_margin),
                            )
                        )
                        ar_high = float(
                            config.get(
                                "cv_active_rate_high",
                                min(0.95, ar_target + ar_margin),
                            )
                        )

                        # Dynamic "too-many-trades" cap
                        T_cap_hard = int(
                            config.get(
                                "cv_max_trades_per_block",
                                DEFAULT_CV["cv_max_trades_per_block"],
                            )
                        )
                        cap_frac = float(
                            config.get(
                                "cv_dynamic_trades_cap_frac",
                                DEFAULT_CV["cv_dynamic_trades_cap_frac"],
                            )
                        )
                        T_cap_dyn = int(max(50, cap_frac * float(rows_i)))
                        T_cap = int(min(T_cap_hard, T_cap_dyn))

                        # Reliability gates defaults
                        from math import sqrt
                        try:
                            from scipy.stats import norm
                        except Exception:
                            norm = None

                        def _psr(sr, n_eff, sr_bench=0.0, skew=0.0, kurt=3.0):
                            """Probabilistic Sharpe Ratio: P(SR > sr_bench)."""
                            if n_eff is None or n_eff < 2 or not (sr == sr):
                                return 0.0
                            num = (sr - sr_bench) * sqrt(max(n_eff - 1, 1))
                            den = sqrt(
                                max(
                                    1e-12,
                                    1
                                    - skew * sr
                                    + (kurt - 1.0) * (sr ** 2) / 4.0,
                                )
                            )
                            z = num / den
                            if norm is None:
                                import math

                                return 0.5 * (1.0 + math.erf(z / sqrt(2)))
                            return float(norm.cdf(z))

                        def _dsr_sign(sr, n_eff, sr_max=0.0):
                            """Simple DSR sign proxy."""
                            if n_eff is None or n_eff < 2 or not (sr == sr):
                                return -1.0
                            return (sr - sr_max) * sqrt(max(n_eff, 1))

                        if bool(config.get("print_cv_debug", False)):
                            print(
                                f"[Block {j}] rows={rows_i} trades={trades_i} ar={ar:.3f}"
                            )

                        defaults_features = deepcopy(DEFAULT_FEATURES)
                        defaults_cv = deepcopy(DEFAULT_CV)

                        gating_mode = config.get(
                            "gating_mode",
                            defaults_features.get("gating_mode", "bets_psr"),
                        )
                        min_trades_block = int(
                            config.get(
                                "cv_min_trades_per_block",
                                defaults_cv.get(
                                    "cv_min_trades_per_block",
                                    defaults_features.get(
                                        "min_trades_per_block", 20
                                    ),
                                ),
                            )
                        )
                        min_indep_bets = int(
                            config.get(
                                "cv_min_indep_bets_per_block",
                                defaults_features.get(
                                    "min_independent_bets", 10
                                ),
                            )
                        )
                        psr_alpha = float(
                            config.get(
                                "psr_alpha",
                                defaults_features.get("psr_alpha", 0.10),
                            )
                        )  # PSR cutoff = 1 - psr_alpha
                        dsr_prune = bool(
                            config.get(
                                "dsr_prune",
                                defaults_features.get("dsr_prune", False),
                            )
                        )
                        floor_cv_final = float(
                            config.get(
                                "floor_cv_final",
                                defaults_features.get("floor_cv_final", -4.0),
                            )
                        )

                        if config.get("print_cv_debug", False):
                            print(
                                "[Debug] Reliability gates → "
                                f"psr_alpha={psr_alpha:.2f} (cutoff={1.0 - psr_alpha:.2f}) | "
                                f"min_trades={min_trades_block} | "
                                f"min_indep={min_indep_bets} | "
                                f"dsr_prune={dsr_prune}"
                            )

                        # --- Effective independent bets & PSR for this block (if possible) ---
                        indep_bets = float("nan")
                        psr_block = float("nan")
                        try:
                            if trades_i > 0 and np.isfinite(sharpe):
                                avg_hold_safe = (
                                    float(avg_hold_bars)
                                    if np.isfinite(avg_hold_bars)
                                    and avg_hold_bars > 0
                                    else 1.0
                                )
                                n_eff = trades_i / avg_hold_safe
                                n_eff = max(min_indep_bets, n_eff)
                                indep_bets = float(n_eff)
                                psr_block = float(
                                    _psr(
                                        float(sharpe),
                                        int(round(n_eff)),
                                        sr_bench=0.0,
                                    )
                                )
                        except Exception:
                            indep_bets = float("nan")
                            psr_block = float("nan")

                        # ── Reliability decision: HARD vs SOFT vs OK ──
                        reason = None
                        hard_reject = False

                        # 1) Truly broken → HARD
                        if np.isfinite(T_cap) and np.isfinite(trades_i) and float(trades_i) > float(T_cap):
                            reason = f"OvertradeCap(trades={int(trades_i)} > cap={int(T_cap)})"
                            hard_reject = True
                        elif trades_i <= 0 or not np.isfinite(sharpe):
                            reason = (
                                f"NoTradesOrNaN(trades={trades_i}, sr={sharpe})"
                            )
                            hard_reject = True

                        # 2) Too few trades → SOFT
                        elif trades_i < min_trades_block:
                            reason = (
                                f"TooFewTrades({trades_i}<{min_trades_block})"
                            )

                        # 3) PSR / DSR / Sharpe checks → SOFT (informative)
                        else:
                            try:
                                n_eff_int = int(
                                    max(
                                        min_indep_bets,
                                        trades_i
                                        / max(1.0, float(avg_hold_bars)),
                                    )
                                )
                            except Exception:
                                n_eff_int = int(
                                    max(min_indep_bets, trades_i)
                                )

                            indep_bets = float(n_eff_int)
                            psr = _psr(float(sharpe), n_eff_int)
                            dsr = _dsr_sign(float(sharpe), n_eff_int)
                            psr_block = float(psr)

                            if psr < (1.0 - psr_alpha):
                                reason = (
                                    f"PSR<{1.0 - psr_alpha:.2f} ({psr:.3f})"
                                )
                            elif dsr_prune and dsr <= 0.0:
                                reason = f"DSR≤0 ({dsr:.3f})"
                            elif float(sharpe) <= float(floor_cv_final):
                                reason = (
                                    f"Sharpe≤floor "
                                    f"({float(sharpe):.2f} ≤ {float(floor_cv_final):.2f})"
                                )

                        if reason is not None:
                            block_reasons.append(reason)

                            if hard_reject:
                                # Hard fail → no score; mark NaNs and continue
                                eff_conf_local = float(
                                    getattr(
                                        self,
                                        "_last_conf_thr_used",
                                        conf_thr,
                                    )
                                )
                                block_scores.append(float("nan"))
                                block_active_rates.append(float(ar))
                                block_trades.append(int(trades_i))
                                block_eff_conf.append(eff_conf_local)
                                block_sharpe.append(float("nan"))
                                block_psr.append(float("nan"))
                                block_neff.append(float("nan"))
                                block_pruned.append(True)
                                block_all_hold.append(False)
                                pred_cards.append(
                                    {
                                        "label_counts": getattr(
                                            self,
                                            "_last_label_counts",
                                            None,
                                        ),
                                        "thr": getattr(
                                            self,
                                            "_last_label_threshold",
                                            params.get(
                                                "label_threshold", None
                                            ),
                                        ),
                                        "test_len": rows_i,
                                        "raw_preds": getattr(
                                            self,
                                            "_last_raw_pred_dist",
                                            None,
                                        ),
                                        "decoded_before": getattr(
                                            self,
                                            "_last_decoded_preds_before",
                                            None,
                                        ),
                                        "final_after_thr": getattr(
                                            self,
                                            "_last_final_preds_dist",
                                            None,
                                        ),
                                        "eff_conf": eff_conf_local,
                                        "avg_hold_bars": (
                                            float(avg_hold_bars)
                                            if np.isfinite(
                                                avg_hold_bars
                                            )
                                            else "—"
                                        ),
                                        "indep_bets": "—",
                                        "psr": "—",
                                    }
                                )

                                # Pretty card explaining *why* this fold was hard-pruned
                                print_pruned_block_summary(
                                    block_id=j,
                                    reason=reason,
                                    rows=rows_i,
                                    trades=int(trades_i),
                                    active_rate=float(ar),
                                    sharpe=float(sharpe) if np.isfinite(sharpe) else float("nan"),
                                    fold_label=fold_label,
                                )

                                _early_structural_prune_if_hopeless()
                                continue  # skip scoring for this block

                            # Soft fail → informative only; still score block
                            block_pruned.append(False)
                        else:
                            block_reasons.append("")
                            block_pruned.append(False)


                        # --- Soft activity regularization around [ar_low, ar_high] ---

                        _cd = (
                            CLASS_DEFAULTS
                            if "CLASS_DEFAULTS" in globals()
                            else {}
                        )
                        _cd_cv = _cd.get("cv", {})

                        lam_turn = float(
                            config.get(
                                "turnover_penalty_lambda",
                                _cd_cv.get(
                                    "turnover_penalty_lambda", 0.0
                                ),
                            )
                        )
                        lam_low = float(
                            config.get(
                                "cv_soft_active_low_lambda",
                                _cd_cv.get(
                                    "cv_soft_active_low_lambda", 0.0
                                ),
                            )
                        )
                        lam_high = float(
                            config.get(
                                "cv_soft_active_high_lambda",
                                _cd_cv.get(
                                    "cv_soft_active_high_lambda", 0.0
                                ),
                            )
                        )

                        pen_low = max(0.0, float(ar_low) - float(ar))
                        pen_high = max(0.0, float(ar) - float(ar_high))
                        turnover = float(trades_i) / float(max(1, rows_i))
                        
                                                # --- Soft turnover band penalties (model-family aware) ---
                        model_type_local = str(
                            params.get(
                                "model_type",
                                _f_cfg.get("model_type", getattr(self, "model_type", "")),
                            )
                        )
                        _turn_bands = {
                            # Classical ML
                            "logistic": (0.03, 0.18),
                            "svm": (0.03, 0.18),
                            "decision_tree": (0.03, 0.18),
                            "random_forest": (0.03, 0.18),
                            "xgboost": (0.03, 0.18),
                            # Deep supervised
                            "cnn": (0.02, 0.15),
                            "lstm": (0.02, 0.15),
                            "transformer": (0.02, 0.15),
                            # Ensembles
                            "ensemble_cnn_lstm_xgboost": (0.02, 0.14),
                            "ensemble_adaptive_regime": (0.01, 0.12),
                            # RL
                            "dqn": (0.05, 0.25),
                        }

                        # Allow explicit override from CV config; otherwise use family band.
                        _tlow_cfg = config.get("cv_turnover_low", _cd_cv.get("cv_turnover_low", None))
                        _thigh_cfg = config.get("cv_turnover_high", _cd_cv.get("cv_turnover_high", None))
                        try:
                            turn_low = float(_tlow_cfg) if _tlow_cfg is not None else float(_turn_bands.get(model_type_local, (0.02, 0.18))[0])
                        except Exception:
                            turn_low = float(_turn_bands.get(model_type_local, (0.02, 0.18))[0])
                        try:
                            turn_high = float(_thigh_cfg) if _thigh_cfg is not None else float(_turn_bands.get(model_type_local, (0.02, 0.18))[1])
                        except Exception:
                            turn_high = float(_turn_bands.get(model_type_local, (0.02, 0.18))[1])

                        lam_tlow = float(
                            config.get(
                                "cv_turnover_low_lambda",
                                _cd_cv.get("cv_turnover_low_lambda", 0.0),
                            )
                        )
                        lam_thigh = float(
                            config.get(
                                "cv_turnover_high_lambda",
                                _cd_cv.get(
                                    "cv_turnover_high_lambda",
                                    0.0,
                                ),
                            )
                        )

                        # Normalize penalties so units are comparable across bands.
                        _tl_den = max(1e-12, abs(float(turn_low)))
                        _th_den = max(1e-12, abs(float(turn_high)))
                        pen_turn_low = max(0.0, float(turn_low) - float(turnover)) / _tl_den
                        pen_turn_high = max(0.0, float(turnover) - float(turn_high)) / _th_den


                        # Penalized CV score
                        score_penalized = (
                            float(sharpe)
                            - lam_turn * turnover
                            - (lam_low * pen_low + lam_high * pen_high)
                            - (lam_tlow * pen_turn_low + lam_thigh * pen_turn_high)
                        )
                        
                        if config.get("print_cv_debug", False) and (
                            (pen_low > 0.0) or (pen_high > 0.0) or (pen_turn_low > 0.0) or (pen_turn_high > 0.0)
                        ):
                            print(
                                f"[CV][Penalty] model={model_type_local} sr={float(sharpe):.3f} score={float(score_penalized):.3f} "
                                f"ar={ar:.3f} band=[{ar_low:.3f},{ar_high:.3f}] "
                                f"turn={turnover:.4f} band=[{turn_low:.4f},{turn_high:.4f}] "
                                f"pen_ar=({pen_low:.3f},{pen_high:.3f}) pen_turn=({pen_turn_low:.3f},{pen_turn_high:.3f})"
                            )

                        # --- Record metrics for this block ---

                        # -------------------------------
                        # Fold-level intent-precision gate
                        # -------------------------------
                        _pgate_on = bool(cv_config.get("cv_prune_precision_intent", False))
                        _p_thr = float(cv_config.get("cv_prune_min_precision_intent", 0.38))
                        _p_nmin_fold = int(cv_config.get("cv_prune_min_intent_bars_fold", 30))

                        # Pull fold-local intent precision (post-confidence gating).
                        # Prefer the fold eval df attrs; fall back to self._last_* mirrors.
                        _p_int = float("nan")
                        _n_int = 0
                        try:
                            _eval_df = getattr(self, "_cv_last_eval_df", None)
                            _attrs = getattr(_eval_df, "attrs", {}) or {}
                            _p_int = float(_attrs.get("precision_intent", float("nan")))
                            _n_int = int(_attrs.get("intent_bars", 0) or 0)
                        except Exception:
                            pass
                        try:
                            if not np.isfinite(_p_int):
                                _p_int = float(getattr(self, "_last_precision_intent", float("nan")))
                        except Exception:
                            pass
                        try:
                            if int(_n_int) <= 0:
                                _n_int = int(getattr(self, "_last_intent_bars", 0) or 0)
                        except Exception:
                            pass
                        
                        _reason = ""  # default: no fold invalidation reason
                        # Debug visibility (only when print_cv_debug=True)
                        # Print right before the fold gate can actually trigger (on + eligible).
                        if bool(config.get("print_cv_debug", False)) and _pgate_on and (_n_int >= int(_p_nmin_fold)) and np.isfinite(_p_int):
                            print(f"[CV][IntentGate] on={_pgate_on} thr={_p_thr} nmin_fold={_p_nmin_fold} p={_p_int} n={_n_int}")

                        # If enabled: discard fold like other invalid folds (score -> NaN, reason tagged)
                        if _pgate_on and (_n_int >= int(_p_nmin_fold)) and np.isfinite(_p_int) and (float(_p_int) < float(_p_thr)):

                            # Keep ALL per-fold lists aligned with the normal (non-invalid) append path
                            block_scores.append(float("nan"))
                            block_active_rates.append(float(ar))
                            block_trades.append(int(trades_i))

                            # These exist in your normal path; add placeholders here too
                            try:
                                eff_conf_local = float(getattr(self, "_last_conf_thr_used", conf_thr))
                            except Exception:
                                eff_conf_local = float("nan")
                            try:
                                block_eff_conf.append(float(eff_conf_local))
                            except Exception:
                                pass
                            try:
                                block_sharpe.append(float("nan"))
                            except Exception:
                                pass
                            try:
                                block_psr.append(float("nan"))
                            except Exception:
                                pass
                            try:
                                block_neff.append(float("nan"))
                            except Exception:
                                pass
                            try:
                                block_all_hold.append(False)
                            except Exception:
                                pass
                            try:
                                pred_cards.append(None)
                            except Exception:
                                pass
                            # Optional: if you maintain cov-threshold per fold, keep it aligned too
                            try:
                                block_cov_thr.append(float(getattr(self, "_coverage_conf_thr", float("nan"))))
                            except Exception:
                                pass

                            # Keep intent arrays aligned
                            block_precision_intent.append(float(_p_int))
                            block_intent_bars.append(int(_n_int))

                            try:
                                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                _pf_total = int(_diag.get("bars_total", rows_i) or rows_i)
                                _pf_elig  = int(_diag.get("eligible_bars", _pf_total) or _pf_total)
                                _eval_bars = int(getattr(self, "_last_eval_bars", _pf_elig) or _pf_elig)
                                fold_records[int(j) - 1] = {
                                    "vstart": vstart_ts,
                                    "vend": vend_ts,
                                    "train_rows": int(len(tr)),
                                    "val_rows": int(rows_i),
                
                                    "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                    "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                    "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                    "psr": float(psr_block) if np.isfinite(psr_block) else float("nan"),
                                    "trades": int(trades_i),
                                    "active_rate": float(ar),
                                    "precision_trade": float(getattr(self, "_last_precision_trade", float("nan"))),
                                    "n_trade_preds": int(getattr(self, "_last_n_trade_preds", 0) or 0),
                                    "precision_intent": float(_p_int),
                                    "intent_bars": int(_n_int),
                                    "sharpe": float("nan"),
                                    "reason": _reason,
                                    "pruned": False,
                                    "status": "invalid",
                                }
                            except Exception:
                                pass

                            continue

                        eff_conf_local = float(getattr(self, "_last_conf_thr_used", conf_thr))

                        try:
                            _thr_base = float(getattr(self, "_coverage_conf_thr"))
                            block_cov_thr.append(_thr_base)
                        except Exception:
                            pass

                        _final_after = getattr(self, "_last_final_preds_dist", None)
                        _all_hold = False
                        try:
                            if isinstance(_final_after, dict):
                                non_hold = sum(v for k, v in _final_after.items() if str(k) != "0")
                                _all_hold = (int(non_hold) == 0)
                        except Exception:
                            _all_hold = False

                        block_scores.append(float(score_penalized))
                        block_active_rates.append(float(ar))
                        block_trades.append(int(trades_i))
                        
                        block_eff_conf.append(float("nan"))
                        block_sharpe.append(float("nan"))
                        block_psr.append(float("nan"))
                        block_neff.append(float("nan"))
                        block_pruned.append(False)
                        block_all_hold.append(True)
                        block_reasons.append(str(_reason))
                        pred_cards.append({})


                        # Keep intent arrays aligned
                        block_precision_intent.append(float(_p_int) if np.isfinite(_p_int) else float("nan"))
                        block_intent_bars.append(int(_n_int))

                        pred_cards.append({
                            "label_counts": getattr(self, "_last_label_counts", None),
                            "thr": getattr(self, "_last_label_threshold", params.get("label_threshold", None)),
                            "test_len": getattr(self, "_last_test_len", rows_i),
                            "final_after_thr": _final_after,
                            "eff_conf": eff_conf_local,
                            "avg_hold_bars": (float(avg_hold_bars) if np.isfinite(avg_hold_bars) else "—"),
                            "indep_bets": (float(indep_bets) if np.isfinite(indep_bets) else "—"),
                            "psr": (float(psr_block) if np.isfinite(psr_block) else "—"),
                            "turnover": float(turnover),
                        })

                        # fold-local record (canonical source for overview table)
                        try:
                            fold_records[int(j) - 1] = {
                                "vstart": vstart_ts,
                                "vend": vend_ts,
                                "train_rows": int(len(tr)),
                                "val_rows": int(rows_i),
                                "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(getattr(self, "_last_eval_bars", 0) or 0),
                                "psr": float(psr_block) if np.isfinite(psr_block) else float("nan"),
                                "trades": int(trades_i),
                                "active_rate": float(ar),
                                "precision_trade": float(getattr(self, "_last_precision_trade", float("nan"))),
                                "n_trade_preds": int(getattr(self, "_last_n_trade_preds", 0) or 0),
                                "precision_intent": float(_p_int) if np.isfinite(_p_int) else float("nan"),
                                "intent_bars": int(_n_int),
                                "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
                                "reason": "",
                                "pruned": False,
                                "status": "ok",
                            }
                        except Exception:
                            pass


                        # --- Compact per-fold summary (Mini-Block Fold #j) ---
                        try:
                            cfg_f = getattr(self, "features_config", {}) or {}
                            _cd = CLASS_DEFAULTS.get("features", {}) if "CLASS_DEFAULTS" in globals() else {}

                            # Coverage / calibration
                            target_cov = float(
                                cfg_f.get(
                                    "target_active_rate",
                                    cfg_f.get("target_coverage", 0.10),
                                )
                            )
                            try:
                                base_conf = float(getattr(self, "_coverage_conf_thr"))
                            except Exception:
                                base_conf = float(
                                    cfg_f.get(
                                        "confidence_threshold",
                                        _cd.get("confidence_threshold", 0.0),
                                    )
                                )

                            calib_info = {
                                "target": target_cov,
                                "conf_thr": base_conf,
                                "bars": int(rows_i),
                            }
                            
                            # Reporting denominators (telemetry only):
                            # - bars_total: post-feature eval grid (Eligibility bars_total)
                            # - bars_eligible: actual evaluated bars (matches ExecAudit bars)
                            try:
                                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                _bars_total = int(_diag.get("bars_total", rows_i) or rows_i)
                            except Exception:
                                _bars_total = int(rows_i)
                            try:
                                _bars_eval = int(getattr(self, "_last_eval_bars", 0) or 0)
                                if _bars_eval <= 0:
                                    _bars_eval = int(_diag.get("eligible_bars", _bars_total) or _bars_total)
                            except Exception:
                                _bars_eval = int(_bars_total)

                            calib_info["bars_total"] = int(_bars_total)
                            calib_info["bars_eligible"] = int(_bars_eval)
                            
                            # --- Patch 2: fold reporting denominators (telemetry only) ---
                            # Use the same bar-grid universe as gating/execution (post-feature),
                            # and the same evaluated bars as Gate✔/ExecAudit.
                            try:
                                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                _bars_postfeat_total = int(_diag.get("bars_total", rows_i) or rows_i)
                                _bars_postfeat_elig  = int(_diag.get("eligible_bars", _bars_postfeat_total) or _bars_postfeat_total)
                            except Exception:
                                _bars_postfeat_total = int(rows_i)
                                _bars_postfeat_elig  = int(rows_i)
                            try:
                                _bars_eval = int(getattr(self, "_last_eval_bars", _bars_postfeat_elig) or _bars_postfeat_elig)
                            except Exception:
                                _bars_eval = int(_bars_postfeat_elig)

                            calib_info["bars_total"] = int(_bars_postfeat_total)
                            calib_info["bars_eligible"] = int(_bars_eval)

                            # Dynamic αβγ and coverage nudge
                            alpha = float(cfg_f.get("alpha_vol_z", _cd.get("alpha_vol_z", 0.0)))
                            beta  = float(cfg_f.get("beta_spread_norm", _cd.get("beta_spread_norm", 0.0)))
                            gamma = float(cfg_f.get("gamma_slip_norm", _cd.get("gamma_slip_norm", 0.0)))
                            band  = float(cfg_f.get("runtime_active_band_margin", _cd.get("runtime_active_band_margin", 0.05)))
                            step  = float(cfg_f.get("runtime_conf_nudge", _cd.get("runtime_conf_nudge", 0.01)))
                            
                            # Prefer actually-used (sanitized) nudge params from this block, when available.
                            try:
                                _band_used = getattr(self, "_last_runtime_active_band_used", None)
                                _step_used = getattr(self, "_last_runtime_conf_step_used", None)
                                if _band_used is not None and np.isfinite(_band_used):
                                    band = float(_band_used)
                                if _step_used is not None and np.isfinite(_step_used):
                                    step = float(_step_used)
                            except Exception:
                                pass

                            gate_info = {
                                "base": base_conf,
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "median_thr": float(
                                    getattr(self, "_last_conf_thr_used", base_conf)
                                ),
                                "band": band,
                                "step": step,
                            }

                            # Reliability gate parameters
                            # IMPORTANT: the summary must reflect the SAME knobs used by the actual gate.
                            # Prefer the canonical keys used elsewhere in test_strategy:
                            #   psr_alpha, floor_cv_final, cv_min_indep_bets_per_block
                            psr_alpha = float(
                                config.get(
                                    "psr_alpha",
                                    config.get(
                                        "cv_psr_alpha",
                                        cfg_f.get("psr_alpha", _cd.get("psr_alpha", 0.10)),
                                    ),
                                )
                            )

                            # Clamp to sane range so cutoff computation can't go weird.
                            if (not np.isfinite(psr_alpha)) or (psr_alpha <= 0.0) or (psr_alpha >= 1.0):
                                if config.get("print_cv_debug", False):
                                    print(f"⚠️ [CV][Reliability] psr_alpha out of range ({psr_alpha}); reset → 0.10")
                                psr_alpha = 0.10

                            cutoff = float(
                                config.get(
                                    "floor_cv_final",
                                    config.get(
                                        "cv_sharpe_floor_final",
                                        cfg_f.get("floor_cv_final", _cd.get("floor_cv_final", -4.0)),
                                    ),
                                )
                            )

                            min_trades_block = int(config.get("cv_min_trades_per_block", 5))
                            min_indep_bets = int(
                                config.get(
                                    "cv_min_indep_bets_per_block",
                                    config.get("cv_min_indep_bets", 12),
                                )
                            )

                            reliability = {
                                "psr_alpha": psr_alpha,
                                "cutoff": cutoff,
                                "min_trades": min_trades_block,
                                "min_indep": min_indep_bets,
                            }


                            # Class distributions captured earlier in test_strategy
                            class_dists = getattr(
                                self, "_last_class_dists", {"raw": {}, "final": {}}
                            )

                            block_stats = {
                                # Back-compat key
                                "rows": int(_bars_eval),
                                # New: show raw val slice vs evaluated bar-grid rows
                                "rows_total": int(rows_i),
                                "rows_eligible": int(_bars_postfeat_elig),
                                "trades": int(trades),
                                "ar": float(active_rate),
                                "sr": float(sharpe),
                                "precision_intent": float(getattr(self, "_last_precision_intent", float("nan"))),
                                "intent_bars": int(getattr(self, "_last_intent_bars", 0) or 0),
                            }
 

                            if LOG_MODE in {"COMPACT", "DEBUG"}:
                                print_block_summary(
                                    block_id=j,
                                    calib_info=calib_info,
                                    gate_info=gate_info,
                                    reliability=reliability,
                                    class_dists=class_dists,
                                    block_stats=block_stats,
                                )
                        except Exception:
                            # Summary printing should never break CV
                            pass


                        # --- Interim Optuna reporting & pruning ---
                        try:
                            valid_now = [
                                s for s in block_scores if np.isfinite(s)
                            ]
                            if trial is not None and valid_now:
                                arr_t = np.asarray(valid_now, dtype=float)

                                # Recency-tilted interim mean
                                tail_t = float(
                                    config.get(
                                        "cv_tail_weight", 1.35
                                    )
                                )
                                if arr_t.size > 1:
                                    w_rec = np.array(
                                        [
                                            1.0
                                            + i
                                            * (
                                                (tail_t - 1.0)
                                                / max(
                                                    1,
                                                    arr_t.size - 1,
                                                )
                                            )
                                            for i in range(
                                                arr_t.size
                                            )
                                        ],
                                        dtype=float,
                                    )
                                else:
                                    w_rec = np.ones_like(arr_t)

                                interim = float(
                                    np.average(
                                        arr_t, weights=w_rec
                                    )
                                )

                                step_idx = len(block_scores)
                                trial.report(interim, step=step_idx)

                                if (
                                    os.getenv(
                                        "MLB_DISABLE_OPTUNA_PRUNING",
                                        "0",
                                    )
                                    != "1"
                                ):
                                    relax = float(
                                        config.get(
                                            "cv_prune_relax", 1.0
                                        )
                                    )
                                    relax = max(
                                        0.0, min(relax, 1.0)
                                    )

                                    if relax > 0.0:
                                        base_min_k = float(
                                            config.get(
                                                "prune_min_folds",
                                                2,
                                            )
                                        )
                                        base_abs_fl = float(
                                            config.get(
                                                "prune_abs_floor_sr",
                                                -8.0,
                                            )
                                        )
                                        base_iqr_m = float(
                                            config.get(
                                                "prune_iqr_mult",
                                                0.75,
                                            )
                                        )

                                        k_done = int(
                                            arr_t.size
                                        )
                                        min_k = max(
                                            1,
                                            int(
                                                round(
                                                    base_min_k
                                                    * relax
                                                )
                                            ),
                                        )

                                        if base_abs_fl < 0:
                                            abs_fl = (
                                                base_abs_fl
                                                / max(
                                                    relax,
                                                    1e-6,
                                                )
                                            )
                                        else:
                                            abs_fl = (
                                                base_abs_fl
                                                * relax
                                            )

                                        iqr_m = (
                                            base_iqr_m
                                            * relax
                                        )

                                        if k_done >= 3:
                                            q1, q3 = (
                                                np.percentile(
                                                    arr_t,
                                                    [25, 75],
                                                )
                                            )
                                        else:
                                            q1 = float(
                                                np.min(
                                                    arr_t
                                                )
                                            )
                                            q3 = float(
                                                np.max(
                                                    arr_t
                                                )
                                            )

                                        iqr = max(
                                            1e-12,
                                            (float(q3) - float(q1)),
                                        )
                                        rel_fl = float(
                                            np.median(
                                                arr_t
                                            )
                                        ) - iqr_m * iqr
                                        gate = max(
                                            abs_fl, rel_fl
                                        )

                                        if (
                                            k_done
                                            >= min_k
                                        ) and (
                                            interim
                                            < gate
                                        ):
                                            import optuna as _opt
                                            if bool(config.get("cv_strict_pruning", False)):
                                                # Keep legacy behavior (abort whole trial)
                                                raise _opt.TrialPruned(
                                                    "Pruned early: "
                                                    f"interim={interim:.4f} "
                                                    f"< gate={gate:.4f} "
                                                    f"at step={step_idx}"
                                                )
                                            else:
                                                # Downgrade to fold-level invalidation; the outer
                                                # exception handler will convert this to a NaN score
                                                # and keep evaluating remaining blocks.
                                                raise RuntimeError(
                                                    "FoldPrunedByGate: "
                                                    f"interim={interim:.4f} "
                                                    f"< gate={gate:.4f} "
                                                    f"at step={step_idx}"
                                                )

                                # Also honor Optuna's own pruner
                                import optuna as _opt

                                if trial.should_prune():
                                    raise _opt.TrialPruned(
                                        "Pruned by scheduler "
                                        f"at step={step_idx} "
                                        f"with interim={interim:.4f}"
                                    )

                        except Exception as e:
                            # Propagate real TrialPruned
                            try:
                                import optuna as _opt
                                if isinstance(e, _opt.TrialPruned):
                                    raise
                            except Exception:
                                pass

                            # Normal block error → mark invalid and continue
                            _cv_penalty(
                                "MiniBlockCV exception during block evaluation",
                                split_start=int(split),
                                error=str(e),
                            )
                            block_scores.append(float("nan"))
                            block_active_rates.append(0.0)
                            block_trades.append(0)
                            block_eff_conf.append(float("nan"))
                            block_sharpe.append(float("nan"))
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))
                            _is_pruned = ("TrialPruned" in type(e).__name__) or ("Pruned" in str(e))
                            block_pruned.append(_is_pruned)
                            block_all_hold.append(False)
                            pred_cards.append({})
                            block_reasons.append("Pruned" if _is_pruned else "Exception")
                            _early_structural_prune_if_hopeless()
                            continue
                        
                        
                    # -------------------------------
                    # ALIGNMENT TRIPWIRE (debug only)
                    # -------------------------------
                    if bool(config.get("print_cv_debug", False)):
                        L = len(block_scores)
                        for name, lst in [
                            ("block_active_rates", block_active_rates),
                            ("block_trades", block_trades),
                            ("block_rows", block_rows),
                            ("block_sharpe", block_sharpe),
                            ("block_psr", block_psr),
                            ("block_neff", block_neff),
                            ("block_pruned", block_pruned),
                            ("block_all_hold", block_all_hold),
                            ("block_reasons", block_reasons),
                            ("pred_cards", pred_cards),
                        ]:
                            if len(lst) != L:
                                raise RuntimeError(
                                    f"[CV][ALIGNMENT_BUG] {name} len={len(lst)} != block_scores len={L}"
                                )


                    # --------------------------------------------
                    # Active folds & coverage (pre-aggregator)
                    # --------------------------------------------
                    M_gate = _M_gate_eff
                    L_gate = _L_gate_eff
                    r_min  = _r_min_eff

                    K_plan = int(config.get("cv_blocks", len(block_scores) or 5))
                    N_use  = min(len(block_scores), K_plan)

                    active_mask = [
                        (i < len(block_trades))
                        and (int(block_trades[i]) >= M_gate)
                        and (float(block_active_rates[i]) >= r_min)
                        for i in range(N_use)
                    ]
                    active_folds = int(sum(1 for a in active_mask if a))

                    # -------------------------------
                    # Single place to print mini-table
                    # -------------------------------
                    def _print_mini_tables():
                        """
                        Compact MiniBlockCV overview:
                        - Always 1 row per planned fold.
                        - Uses tracked raw Sharpe (block_sharpe) and PSR (block_psr).
                        - Penalized scores (block_scores) are internal to Optuna.
                        """
                        if table_mode == "off":
                            return

                        # How many folds we intended to have
                        planned_k = int(config.get("cv_blocks", 0)) or len(val_ends_ts) or len(block_scores) or 5
                        N_tbl = planned_k

                        rows_over = []

                        for i in range(N_tbl):

                            # Prefer fold_records (slot-stable) over parallel arrays.
                            fr = None
                            try:
                                fr = fold_records[i] if (i < len(fold_records)) else None
                            except Exception:
                                fr = None

                            # IMPORTANT: "has_data" must reflect whether this fold actually ran,
                            # not whether a parallel array happened to have an entry.
                            if isinstance(fr, dict):
                                rows_i_probe = fr.get("val_rows", 0) or 0
                                tr_rows_probe = fr.get("train_rows", 0) or 0
                                pruned_probe = bool(fr.get("pruned", False))
                                # Treat as "has data" if it produced rows or was explicitly pruned with a reason.
                                has_data = (int(rows_i_probe) > 0) or (int(tr_rows_probe) > 0) or pruned_probe or bool(fr.get("reason", ""))
                            else:
                                has_data = (i < len(block_scores))

                            if isinstance(fr, dict):
                                # IMPORTANT: when fold_records exists, the overview must be driven ONLY by it.
                                # Never index parallel arrays here (that’s the historic drift bug).
                                sc = fr.get("score", float("nan"))
                                tr = fr.get("trades", float("nan"))
                                ar = fr.get("active_rate", float("nan"))
                                sh = fr.get("sharpe", float("nan"))
                                ps = fr.get("psr", float("nan"))
                                rows_i = fr.get("val_rows", 0) or 0
                                tr_rows = fr.get("train_rows", 0) or 0

                                # NEW: denominators (must exist in both branches)
                                pf_total  = int(fr.get("post_feature_bars_total", rows_i) or rows_i)
                                pf_elig   = int(fr.get("post_feature_eligible", pf_total) or pf_total)
                                eval_bars = int(fr.get("eval_bars", pf_elig) or pf_elig)

                                # NEW: intent precision + intent bars
                                pint = fr.get("precision_intent", float("nan"))
                                nint = fr.get("intent_bars", 0) or 0

                                # tolerate older/newer key variants
                                vstart = fr.get("vstart", fr.get("val_start", None))
                                vend   = fr.get("vend",   fr.get("val_end", None))

                                reason = fr.get("reason", "") or ""
                                pruned = bool(fr.get("pruned", False))

                            else:
                                sc = block_scores[i]        if has_data and i < len(block_scores)       else float("nan")
                                tr = block_trades[i]        if has_data and i < len(block_trades)       else float("nan")
                                ar = block_active_rates[i]  if has_data and i < len(block_active_rates) else float("nan")
                                sh = block_sharpe[i]        if has_data and i < len(block_sharpe)       else float("nan")
                                ps = block_psr[i]           if has_data and i < len(block_psr)          else float("nan")
                                rows_i = block_rows[i]      if has_data and i < len(block_rows)         else 0
                                tr_rows = block_train_rows[i] if has_data and i < len(block_train_rows) else 0

                                # Defaults (fallback when fold_records[i] is missing / non-dict)
                                pf_total  = int(rows_i) if rows_i else 0
                                pf_elig   = int(pf_total)
                                eval_bars = int(pf_elig)

                                # NEW: intent precision fallback arrays
                                pint = block_precision_intent[i] if has_data and i < len(block_precision_intent) else float("nan")
                                nint = block_intent_bars[i] if has_data and i < len(block_intent_bars) else 0

                                # val_start / val_end: use captured fold timestamps for correct alignment
                                vstart = val_starts_ts_cv[i] if i < len(val_starts_ts_cv) else None
                                vend   = val_ends_ts_cv[i]   if i < len(val_ends_ts_cv)   else (val_ends_ts[i] if i < len(val_ends_ts) else None)

                                # Human-readable status
                                reason = block_reasons[i] if i < len(block_reasons) else ""
                                pruned = block_pruned[i] if i < len(block_pruned) else False

                            if not has_data:
                                st = "⛔ NO DATA / PRUNED"
                            elif pruned:
                                st = f"⛔ {reason or 'Pruned'}"
                            elif reason:
                                # soft issues / diagnostics
                                if "Bad" in reason or "SRNaN" in reason:
                                    st = f"🔴 {reason}"
                                else:
                                    st = f"⛔ {reason}"
                            else:
                                st = (f"🟡 {reason}" if reason else "🟢 OK")

                            # PSR column: use stored block_psr (based on raw Sharpe & n_eff)
                            if has_data and ps is not None and np.isfinite(ps):
                                psr_str = f"{float(ps):.3f}"
                            else:
                                psr_str = "—"

                            rows_over.append([
                                i + 1,
                                (str(vstart) if vstart is not None else "—"),
                                (str(vend)   if vend   is not None else "—"),
                                int(tr_rows) if tr_rows else 0,
                                int(rows_i) if rows_i else 0,
                                int(pf_total) if pf_total else 0,
                                int(pf_elig) if pf_elig else 0,
                                int(eval_bars) if eval_bars else 0,
                                int(tr) if tr is not None and np.isfinite(tr) else 0,
                                (f"{float(ar):.3f}" if ar is not None and np.isfinite(ar) else "—"),
                                (f"{float(pint):.3f}" if pint is not None and np.isfinite(pint) else "—"),
                                (int(nint) if nint else 0),
                                (f"{float(sh):.3f}" if sh is not None and np.isfinite(sh) else "—"),  # SR = raw Sharpe
                                psr_str,                                                                # PSR(raw SR)
                                st,
                            ])

                        if rows_over:
                            _title = "🗓️  Monthly-roll CV overview" if bool(getattr(self, "_cv_used_monthly_last", False)) else "🧪 Mini-block overview"
                            log_print(
                                _fmt_table(
                                    ["#", "val_start", "val_end", "train_rows", "val_rows", "pf_total", "pf_elig", "eval_bars", "trades", "active", "PrecInt", "nInt", "SR", "PSR", "status"],
                                    rows_over,
                                    title=_title
                                ),
                                level="COMPACT",
                            )

                        if table_verbose:
                            for i in range(N_tbl):
                                # IMPORTANT: if fold_records exists for this slot, drive the verbose
                                # card from it (never from parallel arrays).
                                fr = None
                                try:
                                    fr = fold_records[i] if (i < len(fold_records)) else None
                                except Exception:
                                    fr = None

                                if isinstance(fr, dict):
                                    sc = fr.get("score", float("nan"))
                                    tr = fr.get("trades", float("nan"))
                                    ar = fr.get("active_rate", float("nan"))
                                    all_hold_i = bool(fr.get("all_hold", False))
                                    pruned_i = bool(fr.get("pruned", False))
                                    reason_i = str(fr.get("reason", "") or "")

                                    # tolerate older/newer key variants
                                    vstart = fr.get("vstart", fr.get("val_start", None))
                                    vend   = fr.get("vend",   fr.get("val_end", None))
                                else:
                                    sc = block_scores[i] if i < len(block_scores) else float("nan")
                                    tr = block_trades[i] if i < len(block_trades) else 0
                                    ar = block_active_rates[i] if i < len(block_active_rates) else float("nan")
                                    all_hold_i = (block_all_hold[i] if i < len(block_all_hold) else False)
                                    pruned_i = (bool(block_pruned[i]) if i < len(block_pruned) else False)
                                    reason_i = (block_reasons[i] if i < len(block_reasons) else "")

                                    try:
                                        vstart = (val_starts_ts_cv[i] if i < len(val_starts_ts_cv) else val_starts_ts[i])
                                    except Exception:
                                        vstart = None
                                    try:
                                        vend = (val_ends_ts_cv[i] if i < len(val_ends_ts_cv) else val_ends_ts[i])
                                    except Exception:
                                        vend = None

                                st = _status_for_block(
                                    tr,
                                    sc,
                                    ar,
                                    all_hold=all_hold_i,
                                    pruned=pruned_i,
                                )
                                try:
                                    if reason_i:
                                        st = f"⛔ {reason_i}"
                                except Exception:
                                    pass

                                if table_only_failures and st == "🟢 OK":
                                    continue

                                card = pred_cards[i] if i < len(pred_cards) else {}

                                try:
                                    vstart = str(vstart).split("+")[0].replace("T", " ") if vstart is not None else ""
                                except Exception:
                                    vstart = ""
                                try:
                                    vend = str(vend).split("+")[0].replace("T", " ") if vend is not None else ""
                                except Exception:
                                    vend = ""

                                # PSR for verbose card: same logic as above
                                psr_card = card.get("psr", None)
                                psr_str = "—"
                                try:
                                    if psr_card is not None and psr_card != "—" and np.isfinite(float(psr_card)):
                                        psr_str = f"{float(psr_card):.3f}"
                                    else:
                                        indep = card.get("indep_bets", None)
                                        if (
                                            sc is not None
                                            and np.isfinite(sc)
                                            and indep not in (None, "—")
                                        ):
                                            n_eff = int(max(2, round(float(indep))))
                                            psr_val = _psr(float(sc), n_eff, sr_bench=0.0)
                                            if np.isfinite(psr_val):
                                                psr_str = f"{psr_val:.3f}"
                                except Exception:
                                    psr_str = "—"

                                rows_card = [
                                    ["val_start",        vstart],
                                    ["val_end",          vend],
                                    ["label_counts",     _fmt_dict(card.get("label_counts"))],
                                    ["thr",              card.get("thr", params.get("label_threshold", None))],
                                    ["test_len",         card.get("test_len", (fr.get("val_rows", "") if isinstance(fr, dict) else (block_rows[i] if i < len(block_rows) else "")))],
                                    ["raw_preds",        _fmt_dict(card.get("raw_preds"))],
                                    ["decoded_before",   _fmt_dict(card.get("decoded_before"))],
                                    ["final_after_thr",  _fmt_dict(card.get("final_after_thr"))],
                                    ["eff_conf",         (f"{float(block_eff_conf[i]):.3f}" if i < len(block_eff_conf) and np.isfinite(block_eff_conf[i]) else "—")],
                                    ["indep_bets",       card.get("indep_bets", "—")],
                                    ["psr",              psr_str],
                                    ["status",           st],
                                ]

                                print(
                                    _fmt_table(
                                        ["field", "value"],
                                        rows_card,
                                        title=f"🎯 Predictions — Block {i+1:02d}",
                                    )
                                )

                    # Print once before exit
                    _print_mini_tables()

                    # Gate on active folds:
                    # - If literally ZERO valid folds → structural failure → prune.
                    # - Otherwise: KEEP the trial; few-fold coverage is treated as "very noisy/bad" but informative.
                    if cv_relax > 0.0 and active_folds == 0:
                        msg = (f"[MiniBlockCV:GATE_FAIL] active_folds=0/{K_plan} | "
                            f"M={M_gate}, r_min={r_min:.3f}")
                        if bool(config.get("print_cv_debug", False)):
                            print(msg + " → TrialPruned (no usable folds)")
                        import optuna as _opt
                        raise _opt.TrialPruned(msg)


                    # Soft gate: if too few active folds, just warn; Optuna will downweight this trial.
                    if active_folds < L_gate:
                        if bool(config.get("print_cv_debug", False)):
                            print(f"[MiniBlockCV:GATE_SOFT] active_folds={active_folds}/{K_plan} "
                                  f"<{L_gate} → keeping trial with weak evidence.")


                    # --- Aggregate fold scores (configurable) ---
                    # -------------------------------
                    # CV-level intent precision aggregation (for Optuna pruning / logs)
                    # -------------------------------
                    try:
                        _p_arr = np.asarray(block_precision_intent[:K_plan], dtype=float)
                        _n_arr = np.asarray(block_intent_bars[:K_plan], dtype=float)
                        _m = np.isfinite(_p_arr) & (_n_arr > 0)
                        _n_sum = float(np.sum(_n_arr[_m])) if _m.size else 0.0
                        if _n_sum > 0:
                            precision_intent_cv = float(np.sum(_p_arr[_m] * _n_arr[_m]) / _n_sum)
                            intent_bars_cv = int(np.sum(_n_arr[_m]))
                        else:
                            precision_intent_cv = float("nan")
                            intent_bars_cv = 0
                        setattr(self, "_last_precision_intent_cv", precision_intent_cv)
                        setattr(self, "_last_intent_bars_cv", intent_bars_cv)
                        if trial is not None:
                            try:
                                trial.set_user_attr("precision_intent_cv", precision_intent_cv)
                                trial.set_user_attr("intent_bars_cv", intent_bars_cv)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    
                    arr_all    = np.asarray(block_scores[:K_plan], dtype=float)
                    valid_mask = np.isfinite(arr_all)
                    K_all      = int(arr_all.size)
                    k_valid    = int(valid_mask.sum())
                    coverage   = float(k_valid / max(1, K_all))
                    
                    # --- Hard requirement: minimum valid folds (prevents fold-gaming) ---
                    # Prefer an integer gate over fraction for small K (e.g., K=4).
                    # If set, this is an unrelaxed structural requirement.
                    try:
                        _used_monthly = bool(getattr(self, "_cv_used_monthly_last", False))
                    except Exception:
                        _used_monthly = False

                    min_valid_folds = config.get(
                        "cv_min_valid_folds_monthly" if _used_monthly else "cv_min_valid_folds",
                        None
                    )
                    prune_on_min_valid_folds = bool(config.get("cv_prune_on_min_valid_folds", True))
                    if min_valid_folds is not None:
                        try:
                            min_valid_folds = int(min_valid_folds)
                        except Exception:
                            min_valid_folds = None

                    if min_valid_folds is not None and min_valid_folds > 0:
                        if k_valid < min_valid_folds:
                            msg = (f"[MiniBlockCV:GATE_MIN_VALID_FOLDS] valid={k_valid}/{K_all} "
                                   f"< min_folds={min_valid_folds} → PRUNE")
                            if bool(config.get("print_cv_debug", False)):
                                print(msg)
                            if prune_on_min_valid_folds:
                                import optuna as _opt
                                raise _opt.TrialPruned(msg)
                            else:
                                return float("nan")
                    
                    
                    # --- Hard requirement: minimum valid-fraction (unrelaxed) ---
                    min_valid_frac = float(config.get("cv_min_valid_fraction", 0.80))
                    prune_low_valid = bool(config.get("cv_prune_on_low_valid_fraction", True))
                    if np.isfinite(min_valid_frac) and min_valid_frac > 0.0:
                        if coverage < min_valid_frac:
                            if prune_low_valid:
                                msg = (f"[MiniBlockCV:GATE_MIN_VALID] valid={k_valid}/{K_all} "
                                       f"(cov={coverage:.2f}) < min={min_valid_frac:.2f} → PRUNE")
                                if bool(config.get("print_cv_debug", False)):
                                    print(msg)
                                try:
                                    import optuna as _opt
                                    raise _opt.TrialPruned(msg)
                                except Exception:
                                    return float("nan")
                            else:
                                # We'll keep the trial but apply a penalty later (after aggregation).
                                if bool(config.get("print_cv_debug", False)):
                                    print(f"[MiniBlockCV:GATE_MIN_VALID] cov={coverage:.2f} < {min_valid_frac:.2f} → "
                                          f"KEEP & PENALIZE (cv_invalid_share_penalty × invalid_folds)")

                    min_cov_base = float(config.get("cv_min_coverage", 0.80))

                    if cv_relax <= 0.0:
                        # Fully relaxed: do NOT hard-prune on coverage.
                        # We let NaNs / low coverage propagate as "bad scores" instead.
                        min_cov = 0.0
                    else:
                        # Larger base threshold = stricter; scale down by relaxation.
                        min_cov = max(0.0, min_cov_base * cv_relax)

                    if cv_relax > 0.0:
                        # Normal behavior: if nothing usable OR too little coverage → prune the trial.
                        if k_valid == 0 or coverage < min_cov:
                            msg = f"[MiniBlockCV] coverage={coverage:.2f} < min_cov={min_cov:.2f}"
                            if bool(config.get("print_cv_debug", False)):
                                print(msg + " → TrialPruned")
                            if optuna is not None:
                                raise optuna.TrialPruned(msg)
                            # If Optuna is not available, fall back to "hopeless" score.
                            return float("nan")
                    else:
                        # cv_prune_relax == 0.0 → no coverage-based pruning.
                        # If k_valid == 0, we just log and continue; downstream agg will yield NaN.
                        if k_valid == 0 and bool(config.get("print_cv_debug", False)):
                            print("[MiniBlockCV] cv_prune_relax=0.0 & k_valid=0 → no hard prune (returning NaN later).")

                    vals = np.sort(arr_all[valid_mask])
                    trim_frac = float(config.get("cv_trim_frac", 0.0))
                    trim_n = int(round(k_valid * trim_frac))
                    if trim_n > 0 and (2 * trim_n) < k_valid:
                        vals = vals[trim_n:-trim_n]


                    agg_mode = str(config.get("cv_agg_mode", "tanh_mean")).lower()
                    if agg_mode == "tanh_mean":
                        s0 = float(config.get("cv_tanh_s", 10.0))
                        if (not np.isfinite(s0)) or (s0 <= 0):
                            s0 = 0.0
                        elif s0 < 1.0:
                            print(f"[CVCombine] cv_tanh_s too small ({s0}); clamping to 1.0")
                            s0 = 1.0                    
                        if s0 > 0:
                            vals = s0 * np.tanh(vals / s0)
                        final_score = float(np.nanmean(vals)) if vals.size else float("nan")
                    elif agg_mode == "mean":
                        final_score = float(np.nanmean(vals)) if vals.size else float("nan")
                    elif agg_mode == "median":
                        final_score = float(np.nanmedian(vals)) if vals.size else float("nan")
                    elif agg_mode == "psr_weighted_tanh_mean":
                        # Values (already tanh-capped below if s0>0 like in tanh_mean)
                        s0 = float(config.get("cv_tanh_s", 10.0))
                        if (not np.isfinite(s0)) or (s0 <= 0):
                            s0 = 0.0
                        elif s0 < 1.0:
                            print(f"[CVCombine] cv_tanh_s too small ({s0}); clamping to 1.0")
                            s0 = 1.0
                        if (not np.isfinite(s0)) or (s0 <= 0):
                            s0 = 0.0
                        elif s0 < 1.0:
                            print(f"[CVCombine] cv_tanh_s too small ({s0}); clamping to 1.0")
                            s0 = 1.0
                        vals_full = arr_all.copy()
                        if s0 > 0:
                            vals_full = s0 * np.tanh(vals_full / s0)
                        vals = vals_full[valid_mask]

                        # Build weights = PSR^power  * optional recency tilt
                        psr_power = float(config.get("cv_psr_power", 1.0))
                        # Align PSR weights with the ACTUAL number of scored folds (K_all),
                        # not the planned K_plan. This avoids shape mismatches when some
                        # folds failed before producing PSR.
                        psr_arr_all = np.asarray(
                            (block_psr[:K_all] + [np.nan] * max(0, K_all - len(block_psr))),
                            dtype=float,
                        )

                        w_psr = np.power(np.clip(psr_arr_all, 0.0, 1.0), psr_power)

                        # Defensive alignment: if something is off, truncate to the common length.
                        if w_psr.shape[0] != valid_mask.shape[0]:
                            common = min(w_psr.shape[0], valid_mask.shape[0])
                            w_psr = w_psr[:common]
                            valid_mask = valid_mask[:common]
                            vals = vals_full[:common][valid_mask]

                        w_psr = w_psr[valid_mask]

                        # Optional recency tilt (monotone increasing weights)
                        use_rec = bool(config.get("cv_use_recency_weight", True))
                        if use_rec and vals.size > 1:
                            rec_pow = float(config.get("cv_recency_power", 1.40))
                            w_rec = np.array(
                                [1.0 + i * ((rec_pow - 1.0) / (vals.size - 1)) for i in range(vals.size)],
                                dtype=float
                            )
                        else:
                            w_rec = np.ones_like(vals)

                        # Combine & guard
                        w = w_psr * w_rec
                        if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0:
                            final_score = float(np.nanmean(vals)) if vals.size else float("nan")
                        else:
                            final_score = float(np.average(vals, weights=w)) if vals.size else float("nan")
                    else:
                        final_score = float(np.nanmean(vals)) if vals.size else float("nan")

                    # --- Optional invalid-share penalty (only if we didn't prune earlier) ---
                    try:
                        min_valid_frac = float(config.get("cv_min_valid_fraction", 0.80))
                        prune_low_valid = bool(config.get("cv_prune_on_low_valid_fraction", True))
                        if not prune_low_valid and np.isfinite(final_score):
                            invalid_folds = int(max(0, K_all - k_valid))
                            if k_valid < int(math.ceil(min_valid_frac * max(1, K_all))):
                                per_fold_pen = float(config.get("cv_invalid_share_penalty", 0.5))
                                penalty = per_fold_pen * float(invalid_folds)
                                final_score = float(final_score - penalty)
                                if bool(config.get("print_cv_debug", False)):
                                    print(f"[MiniBlockCV:INVALID_PEN] invalid={invalid_folds} × {per_fold_pen:.2f} "
                                        f"→ −{penalty:.2f} → final={final_score:.4f}")
                    except Exception:
                        pass

                    # --- CSCV / PBO temporal stability penalty ---
                    pbo_weight = float(config.get("cv_cscv_penalty_weight", 0.2))
                    if pbo_weight > 0.0:
                        v = np.asarray(arr_all[valid_mask], dtype=float)
                        if v.size >= 3 and np.all(np.isfinite(v)):
                            # rank correlation between fold performance and time (proxy for CSCV/PBO stability)
                            ranks = np.argsort(np.argsort(v))
                            t_idx = np.arange(v.size)
                            corr = np.corrcoef(ranks, t_idx)[0, 1]

                            # Optional disqualification (very unstable through time)
                            min_corr = float(config.get("cv_cscv_min_rank_corr", np.nan))
                            if np.isfinite(min_corr) and (float(corr) < min_corr) and bool(config.get("cv_strict_pruning", False)):
                                if bool(config.get("print_cv_debug", False)):
                                    print(f"[CSCV-PBO] corr={corr:.3f} < min={min_corr:.2f} → DISQUALIFY (strict)")
                                import optuna as _opt
                                raise _opt.TrialPruned("CSCV/PBO: rank-corr below minimum (strict)")

                            else:

                                # Direction-safe penalty:
                                # - maximize: always DECREASE score by subtracting abs(score)*penalty
                                # - minimize: always INCREASE score by adding abs(score)*penalty (optional; or disable)
                                direction = str(config.get("optuna_direction", "maximize")).lower().strip()
                                is_max = (direction != "minimize")

                                try:
                                    base_before_pbo = float(final_score)
                                except Exception:
                                    base_before_pbo = None

                                # Robust corr handling
                                try:
                                    corr_f = float(corr)
                                except Exception:
                                    corr_f = float("nan")
                                if not np.isfinite(corr_f):
                                    corr_f = 1.0  # corr unavailable => skip CSCV/PBO penalty (pbo_proxy=0)

                                pbo_proxy = max(0.0, 1.0 - corr_f)
                                pen_frac = float(pbo_weight) * float(pbo_proxy)

                                # clip to [0, 1] to avoid pathological amplification
                                if pen_frac < 0.0:
                                    pen_frac = 0.0
                                elif pen_frac > 1.0:
                                    pen_frac = 1.0

                                pen_amt = 0.0
                                if (base_before_pbo is not None) and np.isfinite(base_before_pbo) and (pen_frac > 0.0):
                                    pen_amt = abs(base_before_pbo) * pen_frac
                                    if is_max:
                                        final_score = float(base_before_pbo - pen_amt)
                                    else:
                                        final_score = float(base_before_pbo + pen_amt)  # or: final_score = base_before_pbo
                                # else: leave final_score unchanged

                                # Persist audit attrs (best-effort; never crash objective)
                                try:
                                    if trial is not None:
                                        trial.set_user_attr("cv_cscv_rank_corr", float(corr_f))
                                        trial.set_user_attr("cv_cscv_pbo_proxy", float(pbo_proxy))
                                        trial.set_user_attr("cv_cscv_pen_frac", float(pen_frac))
                                        trial.set_user_attr("cv_cscv_pen_amount", float(pen_amt))
                                        trial.set_user_attr("cv_cscv_pen_direction", direction)
                                except Exception:
                                    pass

                                if bool(config.get("print_cv_debug", False)):
                                    try:
                                        b = float(base_before_pbo)
                                    except Exception:
                                        b = float("nan")
                                    print(
                                        f"[CSCV-PBO] corr={corr_f:.3f} proxy={pbo_proxy:.3f} "
                                        f"pen_frac={pen_frac:.3f} pen_amt={pen_amt:.4f} "
                                        f"dir={direction} base={b:.4f} → final={float(final_score):.4f}"
                                    )

                    if bool(config.get("print_cv_fold_scores", False)) or bool(config.get("print_cv_debug", False)):
                        _prec = int(config.get("cv_log_precision", 8))
                        _raw  = np.round(arr_all[valid_mask], 4).tolist()
                        _fin  = f"{final_score:.{_prec}f}"
                        prefix = "CVCombine" if bool(getattr(self, "_cv_used_monthly_last", False)) else "MiniBlockCV"
                        kept_ids = [i + 1 for i, ok in enumerate(list(valid_mask)) if bool(ok)]
                        print(
                            f"[{prefix}:{agg_mode}] kept_folds={kept_ids}/{K_all} folds={_raw} "
                            f"| k={k_valid}/{K_all} (cov={coverage:.2f}) "
                            f"| trim_frac={trim_frac:.2f} → final={_fin}"
                        )

                # --- Store diagnostics (robust to partial folds) ---
                try:
                    if trial is not None:
                        # Build per-regime table: per-fold rows + median aggregate
                        try:
                            import numpy as _np
                            names = {0: "sideways", 1: "trend", 2: "volatile"}
                            per_fold_rows = []
                            for fidx, df_f in enumerate(fold_eval_frames, start=1):
                                row = {"FOLD": fidx}
                                _cols = getattr(df_f, "columns", [])
                                _rid_col = "regime_id" if ("regime_id" in _cols) else ("regime_id_diag" if ("regime_id_diag" in _cols) else None)
                                if (_rid_col is None) and fidx == 1:
                                    try:
                                        log_print("[MiniBlock][Diag] 'regime_id' not found in fold eval frames; per-regime stats assume all TREND (rid=1).", level="COMPACT")
                                    except Exception:
                                        pass

                                for rid, rname in names.items():
                                    if _rid_col is not None:
                                        sub = df_f[df_f[_rid_col] == rid]
                                    else:
                                        sub = df_f if rid == 1 else df_f.iloc[0:0]

                                    if len(sub) == 0:
                                        row[rname] = {"cstrategy": float("nan"), "sharpe": float("nan"), "trades": 0, "active_rate": float("nan")}
                                    else:
                                        try:
                                            _sub_eval = sub.copy()
                                            try:
                                                if bool(getattr(self, "trading_costs", False)):
                                                    try:
                                                        _sub_eval.attrs["features_config"] = dict(getattr(df_f, "attrs", {}).get("features_config", {}) or {})
                                                    except Exception:
                                                        _sub_eval.attrs["features_config"] = dict(getattr(self, "features_config", {}) or {})
                                                    _sub_eval = self._ensure_cost_columns(_sub_eval, _sub_eval.attrs.get("features_config", {}))
                                            except Exception:
                                                pass

                                            m = compute_full_evaluation_metrics(
                                                df=_sub_eval,
                                                trading_costs=self.trading_costs,
                                                slippage_factor=self.slippage_factor,
                                                eval_context=f"cv:diagnostic:per_regime_metrics:rid={rid}",
                                            )
                                            cstr = float(m[0]) if m is not None else float("nan")
                                            sr = float(m[3]) if m is not None else float("nan")

                                            if (m is not None) and (len(m) > 5) and (m[5] is not None):
                                                tr = int(m[5])
                                            elif "pred" in sub.columns:
                                                tr = int((sub["pred"].fillna(0) != 0).sum())
                                            elif "position_exec" in sub.columns:
                                                tr = int((sub["position_exec"].fillna(0) != 0).sum())
                                            else:
                                                tr = 0

                                            if "pred" in sub.columns:
                                                ar = float((sub["pred"].fillna(0) != 0).mean())
                                            elif "position_exec" in sub.columns:
                                                ar = float((sub["position_exec"].fillna(0) != 0).mean())
                                            else:
                                                ar = float("nan")

                                            row[rname] = {"cstrategy": cstr, "sharpe": sr, "trades": tr, "active_rate": ar}
                                        except Exception:
                                            if "pred" in sub.columns:
                                                tr = int((sub["pred"].fillna(0) != 0).sum())
                                            elif "position_exec" in sub.columns:
                                                tr = int((sub["position_exec"].fillna(0) != 0).sum())
                                            else:
                                                tr = 0
                                            row[rname] = {"cstrategy": float("nan"), "sharpe": float("nan"), "trades": tr, "active_rate": float("nan")}

                                per_fold_rows.append(row)

                            # median-aggregate across folds
                            agg = {}
                            for rid, rname in names.items():
                                vals = {"cstrategy": [], "sharpe": [], "trades": [], "active_rate": []}
                                for r in per_fold_rows:
                                    d = r[rname]
                                    vals["cstrategy"].append(d["cstrategy"])
                                    vals["sharpe"].append(d["sharpe"])
                                    vals["trades"].append(d["trades"])
                                    vals["active_rate"].append(d["active_rate"])
                                def _safe_med(lst):
                                    a = _np.asarray(lst, dtype=float)
                                    a = a[_np.isfinite(a)]
                                    return float(_np.nanmedian(a)) if a.size else float("nan")
                                agg[rname] = {
                                    "cstrategy": _safe_med(vals["cstrategy"]),
                                    "sharpe": _safe_med(vals["sharpe"]),
                                    "trades": int(_np.nanmedian(_np.asarray(vals["trades"], dtype=float))) if vals["trades"] else 0,
                                    "active_rate": _safe_med(vals["active_rate"]),
                                }

                            trial.set_user_attr("per_regime_cv_per_fold", per_fold_rows)
                            trial.set_user_attr("per_regime_cv_median", agg)

                            # Print table like final metric table (compact)
                            try:
                                log_print("\nPer-regime CV table (per-fold rows + median):", level="COMPACT")
                                header = f"{'FOLD':>4} {'REGIME':<10} {'CSTRAT':>10} {'SHARPE':>8} {'TRADES':>8} {'ACTIVE%':>8}"
                                log_print(header)
                                for r in per_fold_rows:
                                    fidx = r["fold"]
                                    for rn in ("sideways", "trend", "volatile"):
                                        v = r[rn]
                                        ar_pct = (v["active_rate"] * 100) if (v["active_rate"] == v["active_rate"]) else float("nan")
                                        log_print(f"{fidx:4d} {rn:<10} {v['cstrategy']:10.4f} {v['sharpe']:8.3f} {int(v['trades']):8d} {ar_pct:8.2f}")
                                log_print("-" * len(header))
                                for rn in ("sideways", "trend", "volatile"):
                                    a = agg[rn]
                                    ar_pct = (a["active_rate"] * 100) if (a["active_rate"] == a["active_rate"]) else float("nan")
                                    log_print(f"{'MED':>4} {rn:<10} {a['cstrategy']:10.4f} {a['sharpe']:8.3f} {int(a['trades']):8d} {ar_pct:8.2f}")
                                log_print("")
                            except Exception:
                                pass
                        except Exception as e:
                            try:
                                if LOG_MODE in {"COMPACT", "DEBUG"} or getattr(self, "debug", False):
                                    print(f"[MiniBlockCV] per-regime CV table failed: {type(e).__name__}: {e}")
                            except Exception:
                                pass


                        arr_tr = np.asarray(block_trades[:K_plan], dtype=float) if 'block_trades' in locals() else np.array([])
                        arr_ar = np.asarray(block_active_rates[:K_plan], dtype=float) if 'block_active_rates' in locals() else np.array([])

                        # Align masks defensively
                        if arr_tr.size and valid_mask.size:
                            m = min(arr_tr.size, valid_mask.size)
                            mask_tr = valid_mask[:m] & np.isfinite(arr_tr[:m])
                            med_tr = float(np.nanmedian(arr_tr[:m][mask_tr])) if mask_tr.any() else float('nan')
                        else:
                            med_tr = float('nan')

                        if arr_ar.size and valid_mask.size:
                            m = min(arr_ar.size, valid_mask.size)
                            mask_ar = valid_mask[:m] & np.isfinite(arr_ar[:m])
                            med_ar = float(np.nanmedian(arr_ar[:m][mask_ar])) if mask_ar.any() else float('nan')
                        else:
                            med_ar = float('nan')

                        trial.set_user_attr("trades_cv", med_tr)
                        trial.set_user_attr("active_rate_cv", med_ar)
                        
                        # Trade-intent precision (median across valid folds).
                        # Defined as P(correct direction | model chose to trade) on each fold.
                        try:
                            _prec_vals = []
                            for _fr in (fold_records or []):
                                if isinstance(_fr, dict):
                                    _prec_vals.append(float(_fr.get("precision_trade", float("nan"))))
                                else:
                                    _prec_vals.append(float("nan"))
                            _arr_p = np.asarray(_prec_vals, dtype=float)
                            if _arr_p.size and valid_mask.size:
                                _m = min(_arr_p.size, valid_mask.size)
                                _mask_p = valid_mask[:_m] & np.isfinite(_arr_p[:_m])
                                med_p = float(np.nanmedian(_arr_p[:_m][_mask_p])) if _mask_p.any() else float("nan")
                            else:
                                med_p = float("nan")
                        except Exception:
                            med_p = float("nan")
                        trial.set_user_attr("precision_trade_cv", med_p)

                        trial.set_user_attr("cv_k_valid", int(k_valid))
                        
                        # Attach intent-precision aggregates for Optuna pruning
                        try:
                            _pi = np.asarray(block_precision_intent, dtype=float)
                            _ni = np.asarray(block_intent_bars, dtype=float)
                            if _pi.size and valid_mask.size:
                                m = min(_pi.size, valid_mask.size)
                                _vm = valid_mask[:m]
                                mask_pi = _vm & np.isfinite(_pi[:m]) & np.isfinite(_ni[:m]) & (_ni[:m] > 0)
                                pi_cv = float(np.nanmedian(_pi[:m][mask_pi])) if mask_pi.any() else float("nan")
                                mask_ni = _vm & np.isfinite(_ni[:m]) & (_ni[:m] > 0)
                                ni_cv = int(np.nansum(_ni[:m][mask_ni])) if mask_ni.any() else 0
                            else:
                                pi_cv = float("nan")
                                ni_cv = 0
                            trial.set_user_attr("precision_intent_cv", float(pi_cv))
                            trial.set_user_attr("intent_bars_cv", int(ni_cv))
                        except Exception:
                            pass
                        # Diagnostics-only: aggregate PSR across CV blocks for Top-N payload
                        try:
                            _psr_vals = np.asarray(block_psr, dtype=float)
                            _psr_vals = _psr_vals[np.isfinite(_psr_vals)]
                            trial.set_user_attr("psr", float(np.nanmedian(_psr_vals)) if _psr_vals.size else float("nan"))
                        except Exception:
                            pass
                except Exception as ex:
                    try:
                        import optuna as _opt
                        if isinstance(ex, _opt.TrialPruned):
                            raise
                    except Exception:
                        pass

                    return final_score
                
                # --- Aggregate CV-derived coverage thresholds across blocks (median) ---
                try:
                    if block_cov_thr:
                        _agg_thr = float(np.nanmedian(np.asarray(block_cov_thr, dtype=float)))
                        setattr(self, "_cv_coverage_thr_agg", _agg_thr)
                        print(
                            f"[CV] Aggregated coverage conf_thr (median of blocks) = {_agg_thr:.3f} | "
                            f"_cv_coverage_thr_agg={_agg_thr:.6f}"
                        )

                except Exception as _ee:
                    print(f"[CV] Coverage threshold aggregation skipped: {_ee}")

                # --- Attach calibration metrics to Optuna trial 
                if trial is not None and calib_n_samples > 0:
                    try:
                        brier_avg = calib_brier_sum / calib_n_samples
                        nll_avg   = calib_nll_sum   / calib_n_samples
                        trial.set_user_attr("brier", float(brier_avg))
                        trial.set_user_attr("nll",   float(nll_avg))
                        if bool(config.get("print_cv_debug", False)):
                            print(
                                f"[CV-Calib] Trial {trial.number}: "
                                f"brier={brier_avg:.6f} | nll={nll_avg:.6f} "
                                f"| n={calib_n_samples}"
                            )
                    except Exception as _e:
                        print(f"[CV-Calib] Failed to attach calibration metrics to trial: {_e}")

                # Successful mini-block CV: return the aggregated score
                return float(final_score)

            finally:
                try:
                    self.free(release_data=False)
                except Exception:
                    pass
                setattr(self, "_in_optuna_cv", _old_cv_flag)
                # Restore CV/debug flags to prevent CV-mode leakage into real_trading_simulation
                try:
                    self._in_cv = _prev_cv
                    self._dbg_first_bars = _prev_dbg
                except Exception:
                    pass



        # Run a single study now and cache Top-5 on self
        month_ix_local = int(config.get("month_ix", 1))
        month_graphs_dir_local = config.get("month_graphs_dir", None) or None

        # Call tuner. If the tuner hasn't been patched to accept month_out_dir/month_ix yet,
        # this will gracefully fall back to the old signature.
        try:
            from inspect import signature as _sig
            _sig_params = set(_sig(run_optuna_tuning).parameters.keys())
        except Exception:
            _sig_params = set()

        # --- Determine Top-N size by model family (Classical/Deep/Ensemble/DQN) ---
        mt_local = str(config.get("model_type", getattr(self, "model_type", ""))).lower()
        cfg_local = getattr(self, "features_config", {}) or {}

        classical = {"logistic", "svm", "decision_tree", "random_forest", "xgboost"}
        rl        = {"cnn", "lstm", "transformer"}
        dqn       = {"dqn"}
        ensembles = {"ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost"}

        if mt_local in classical:
            rt_n = int(cfg_local.get("topN_classical", 4))
        elif mt_local in rl:
            rt_n = int(cfg_local.get("topN_deep", 3))
        elif mt_local in ensembles:
            rt_n = int(cfg_local.get("topN_ensemble", 2))
        elif mt_local in dqn:
            rt_n = int(cfg_local.get("topN_dqn", 2))
        else:
            rt_n = int(cfg_local.get("topN_default", 3))
        rt_n = max(1, rt_n)

        _common_kwargs = dict(
            train_data=first_train_df,
            base_features=base_features_first,
            evaluate_cv_func=_single_study_cv,
            cv_config=cv_config_first,
            models_to_test=models_to_test,
            n_trials=int(config.get("n_trials", 1)),
            return_top_n=rt_n,
            study=None,  # one study here
            sampler_seed=int(self.features_config.get("run_seed", 0)) or None,
        )

        if {"month_out_dir", "month_ix"} <= _sig_params:
            best_params_once, best_score_once, top5_once, study_obj, consensus_pool_once = run_optuna_tuning(
                **_common_kwargs,
                month_out_dir=month_graphs_dir_local,
                month_ix=month_ix_local,
            )
        else: 
            # Legacy tuner signature (no per-month plot routing)
            best_params_once, best_score_once, top5_once, study_obj, consensus_pool_once = run_optuna_tuning(
                **_common_kwargs
            )

        log_print(
            f"🏁 Optuna best trial: #{study_obj.best_trial.number} value={study_obj.best_value:.6f}",
            level="COMPACT",
        )

        
        # Attach Top-5 and consensus pool metadata for downstream use
        if top5_once:
            best_params_once["__top5_params"] = top5_once

        # Normalise consensus pool to a list (may be empty)
        consensus_pool_once = consensus_pool_once or []
        best_params_once["__consensus_pool"] = consensus_pool_once

        # --------------------------------------------------------------
        # Freeze a Top-N committee once after global HPO.
        # This prevents re-selecting the committee each month and avoids
        # any pool/trials_info alignment quirks later.
        # --------------------------------------------------------------
        try:
            from utilsNoWFO import _infer_family  # local import to avoid any cyclical surprises
            fam = _infer_family(str(best_params_once.get("model_type", "")).lower()).strip()
        except Exception:
            fam = "Unknown"

        try:
            cfg_local = getattr(self, "features_config", {}) or {}
            if fam == "Classical":
                N_target = int(cfg_local.get("topN_classical", 3))
            elif fam == "Ensembles":
                N_target = int(cfg_local.get("topN_ensemble", 2))
            elif fam == "RL":
                N_target = int(cfg_local.get("topN_deep", 2))
            else:
                N_target = int(cfg_local.get("topN_default", 2))
            N_target = max(2, int(N_target))
        except Exception:
            N_target = 2

        def _pool_val(d):
            try:
                v = d.get("__cv_value", d.get("cv_value", d.get("value", None)))
                return float(v) if v is not None else float("-inf")
            except Exception:
                return float("-inf")

        try:
            _pool = [dict(x) for x in (consensus_pool_once or []) if isinstance(x, dict)]
            _committee = sorted(_pool, key=_pool_val, reverse=True)[: max(1, min(N_target, len(_pool)))]
            best_params_once["__committee_fixed"] = _committee
            best_params_once["__committee_fixed_n"] = int(len(_committee))
        except Exception:
            # Don't break training if anything unexpected happens here.
            pass


        # Cache on the backtester for WFO / real_trading_simulation helpers
        self._optuna_best_for_wfo = best_params_once
        self._optuna_top5_for_wfo = top5_once or []
        self._optuna_consensus_pool_for_wfo = consensus_pool_once
        
        log_print(
            f"📊 Stored Top-{len(self._optuna_top5_for_wfo or ['best'])} params "
            f"and {len(self._optuna_consensus_pool_for_wfo)} consensus candidates for fallback/consensus use." 
        , level="COMPACT")
        # ============================================================================
        
                # --- HPO-only mode: persist tuned hyperparameters and skip WFO evaluation ---
        if bool(config.get("hpo_only", False)):
            try:
                if bool(config.get("hpo_save_to_disk", False)):
                    # mt_local was defined above when we chose Top-N defaults
                    save_hpo_config_to_disk(mt_local, best_params_once, top5_once)
            except Exception as e:
                print(f"[HPO] Warning: failed to save HPO config for {mt_local}: {e}")
            # In HPO-only mode we return a (None, best_params) tuple so callers
            # can grab the tuned config and run their own evaluation logic.
            return None, best_params_once


        def evaluate_fold(start_date, train_months, test_months):

            train_end = start_date + pd.DateOffset(months=train_months)
            test_end  = train_end + pd.DateOffset(months=test_months)
            if test_end > max_end:
                return None

            # IMPORTANT: end-exclusive slicing to avoid boundary leakage.
            # pandas .loc is inclusive; using < train_end keeps train strictly before test.
            idx_w = walk_data.index
            train_data = walk_data[(idx_w >= start_date) & (idx_w < train_end)]
            test_data  = walk_data[(idx_w >= train_end) & (idx_w < test_end)]
            if len(train_data) < 150 or len(test_data) < 30:
                log_print(
                    f"⚠️ Skipping fold: train={len(train_data)}, test={len(test_data)} too small",
                    level="COMPACT",
                )
                return None

            # Base features (exclude leakage/targets)
            base_features = [
                c for c in train_data.columns
                if c not in ("returns", "price", "spread", "high", "low", "label", "time")
            ]

            # Coarse windows for legacy sliding fallback (mini-block CV has its own sizing)
            min_train_window = int(len(train_data) * 0.75)
            val_window       = int(len(train_data) * 0.25)
            if min_train_window + val_window > len(train_data):
                val_window = len(train_data) - min_train_window
            cv_config = {"min_train_window": min_train_window, "val_window": val_window}
            cv_config["score_for_no_trades"] = -1.0

            # Per-trial CV objective (kept here so second-study fallback can reuse it)
            def evaluate_cv_func(train_data, params, min_train_window, val_window, trial=None):
                # (identical body as above; omitted here for brevity in this snippet)
                return _single_study_cv(train_data, params, min_train_window, val_window, trial=trial)

            # --- NO per-fold Optuna here. Reuse the single study's Top-5 sequentially ---
            best_params = getattr(self, "_optuna_best_for_wfo", None) or {}

            # Seed runtime base threshold from CV aggregation, if available
            try:
                _cv_thr = getattr(self, "_cv_coverage_thr_agg", None)
                if _cv_thr is not None and float(_cv_thr) == float(_cv_thr):  # isfinite without np
                    self._coverage_conf_thr = float(_cv_thr)
                    if self._is_debug():
                        log_print(
                            f"[CV→Runtime] Using aggregated coverage conf_thr={self._coverage_conf_thr:.3f} as base.",
                            level="DEBUG",
                        )
            except Exception as _e:
                if self._is_debug():
                    log_print(f"[CV→Runtime] Coverage conf_thr aggregation not available: {_e}", level="DEBUG")

            top5_params_list = getattr(self, "_optuna_top5_for_wfo", None) or []
            if top5_params_list and "__top5_params" not in best_params:
                best_params["__top5_params"] = top5_params_list

            # --- Prepare Top-5 candidate evaluation (no extra deep refit here) ---
            perf_tuple  = None
            valid_found = False
            selected_params = None  # track chosen candidate for final deep refit


            self.features_config.update(best_params)
            self._optuna_locked_keys = set(best_params.keys())
            base     = dict(best_params)
            raw_topk = best_params.get("__top5_params") or []
            candidates = [base] + [{**base, **deepcopy(alt)} for alt in raw_topk]

            # --- NEW: realism mode uses only the pre-committed CV winner for the candidate list ---
            if not bool(self.features_config.get("allow_param_fallback", False)):
                widx = int(best_params.get("__winner_index", 0))
                if widx <= 0:
                    candidates = [base]
                else:
                    try:
                        chosen = {**base, **deepcopy(raw_topk[widx])}
                        candidates = [chosen]
                    except Exception:
                        candidates = [base]

            REQUIRED_KEYS = ["model_type", "use_extended_features", "lags", "label_threshold"]
            for c in candidates:
                for k in REQUIRED_KEYS:
                    if k not in c and k in base:
                        c[k] = base[k]

            # === Try Top-K candidates sequentially until one meets the WFO trade rule ===
            # NOTE: This now runs even when allow_param_fallback=False, but in that
            #       case `candidates` contains only the chosen CV winner.
            if not valid_found:
                # Configurable minimum trades for *runtime* WFO.
                # Default: 0 → allows "flat but valid" months.
                cfg_f = getattr(self, "features_config", {}) or {}
                min_trades_wfo = int(cfg_f.get("min_trades_for_wfo", 0))

                for idx, params_try in enumerate(candidates):
                    try:
                        self.features_config.update(params_try)
                        perf_tuple = self.evaluate_strategy(
                            params_try,
                            train_start=start_date,
                            train_end=train_end,
                            test_start=train_end,
                            test_end=test_end,
                        )
                        (
                            perf, outperf, creturns,
                            sharpe, drawdown, trades,
                            geo_mean_ann, directional_accuracy, precision_macro,
                            f1_macro, active_rate, profit_per_hit,
                            return_per_trade, win_rate,
                            strategy_volatility, kurtosis
                        ) = perf_tuple

                        try:
                            trades_int = int(trades) if (trades is not None and trades == trades) else 0
                        except Exception:
                            trades_int = 0

                        if trades_int >= min_trades_wfo:
                            print(f"✅ Using Top-{idx+1} candidate (trades={trades_int})")
                            selected_params = dict(params_try)
                            valid_found = True
                            break
                        else:
                            print(
                                f"⚠️ Top-{idx+1} candidate produced {trades_int} trades; "
                                "trying next."
                            )
                    except Exception as e:
                        print(f"❌ Error with Top-{idx+1} candidate: {e}")
                        continue


            # --- Optional SECOND Optuna study (only when allow_param_fallback=True) ---
            if (not valid_found or perf_tuple is None) and bool(self.features_config.get("allow_param_fallback", False)):
                print("⚠️ Top-5 produced no valid result → starting a SECOND Optuna study now (sequential).")

                try:
                    best2, score2, top5_2, _ = run_optuna_tuning(
                        train_data=train_data,
                        base_features=base_features,
                        evaluate_cv_func=evaluate_cv_func,
                        cv_config=cv_config,
                        models_to_test=models_to_test,
                        n_trials=int(config.get("retry_extra_trials", 20)),
                        return_top_n=rt_n,
                        study=None,   # NEW study, but only here, after Top-5 fully failed
                        sampler_seed=int(self.features_config.get("run_seed", 0)) or None,
                    )
                    if top5_2:
                        best2["__top5_params"] = top5_2

                    base2 = dict(best2)
                    raw2 = best2.get("__top5_params") or []
                    candidates2 = [base2] + [{**base2, **deepcopy(alt)} for alt in raw2]

                    for c in candidates2:
                        for k in REQUIRED_KEYS:
                            if k not in c and k in base2:
                                c[k] = base2[k]

                    for idx, params_try in enumerate(candidates2):
                        try:
                            self.features_config.update(params_try)
                            perf_tuple = self.evaluate_strategy(
                                params_try,
                                train_start=start_date, train_end=train_end,
                                test_start=train_end,  test_end=test_end,
                            )
                            (
                                perf, outperf, creturns, sharpe, drawdown, trades,
                                geo_mean_ann, directional_accuracy, precision_macro,
                                f1_macro, active_rate, profit_per_hit,
                                return_per_trade, win_rate,
                                strategy_volatility, kurtosis
                            ) = perf_tuple

                            cfg_f = getattr(self, "features_config", {}) or {}
                            min_trades_wfo = int(cfg_f.get("min_trades_for_wfo", 0))

                            try:
                                trades_int = int(trades) if (trades is not None and trades == trades) else 0
                            except Exception:
                                trades_int = 0

                            if trades_int >= min_trades_wfo:
                                print(
                                    f"✅ Using SECOND-study Top-{idx+1} candidate "
                                    f"(trades={trades_int})"
                                )
                                selected_params = dict(params_try)
                                valid_found = True
                                break
                            else:
                                print(
                                    f"⚠️ SECOND-study Top-{idx+1} candidate produced {trades_int} "
                                    "trades; trying next."
                                )
                        except Exception as e:
                            print(f"❌ Error with SECOND-study Top-{idx+1} candidate: {e}")
                            continue


                except Exception as e:
                    print(f"❌ SECOND Optuna study failed: {e}")

                if not valid_found or perf_tuple is None:
                    print("❗ No valid configuration was found in either study.")
                    return None

            # --- FINAL GUARD: no usable metrics → skip this fold (WFO will log flat month upstream) ---
            if (not valid_found) or (perf_tuple is None):
                print("❗ evaluate_fold: no valid metrics produced for this fold; skipping.")
                return None

            # ------------------------------------------------------------
            # Optuna uses capped, compute-saving CV. After a candidate is
            # selected for this fold/month, run a *single* uncapped refit
            # (stride=1, deep_max_train_windows≈∞, etc.) and report those
            # metrics. This preserves compute during search while ensuring
            # final reported results match the deployment training regime.
            # ------------------------------------------------------------
            try:
                _cfg_f = getattr(self, "features_config", {}) or {}
                _do_refit = bool(_cfg_f.get("final_refit_enabled", True))
                _mt = str((selected_params or {}).get("model_type", "")).strip().lower()
                _is_deep_like = _mt in {
                    "cnn", "lstm", "transformer",
                    "ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime",
                }
                if _do_refit and _is_deep_like and (selected_params is not None):
                    perf_tuple = final_refit_if_deep(
                        backtester=self,
                        best_params=selected_params,
                        train_start=start_date, train_end=train_end,
                        test_start=train_end,  test_end=test_end,
                        overrides={},
                    )
            except Exception as _e:
                try:
                    print(f"⚠️ Final refit skipped/failed; using original metrics. err={_e}")
                except Exception:
                    pass

            # Ensure metrics have the correct arity / structure
            perf_tuple = _safe_metrics_return(perf_tuple, context="wfo_fold")

            # === Save and return results for this fold ===
            (perf, outperf, creturns, sharpe, drawdown, trades,
             geo_mean_ann, directional_accuracy, precision_macro, f1_macro,
             active_rate, profit_per_hit, return_per_trade, win_rate,
             strategy_volatility, kurtosis) = perf_tuple

            return {
                "type": "walk_forward",
                "train_start": start_date,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "performance": perf,
                "outperformance": outperf,
                "return": creturns,
                "sharpe": sharpe,
                "drawdown": drawdown,
                "trades": trades,
                "geo_mean_ann": geo_mean_ann,
                "directional_accuracy": directional_accuracy,
                "precision_macro": precision_macro,
                "f1_macro": f1_macro,
                "active_rate": active_rate,
                "profit_per_hit": profit_per_hit,
                "return_per_trade": return_per_trade,
                "win_rate": win_rate,
                "strategy_volatility": strategy_volatility,
                "kurtosis": kurtosis,
                **best_params,
            }


        start = time.time()
        GPU_MODELS = {
            "cnn", "lstm", "transformer",
            "ensemble_cnn_lstm_xgboost",
            "ensemble_adaptive_regime",
            "xgboost",  # XGBoost with device="cuda" also shares GPU; avoid loky
        }
        is_gpu_model = model_type in GPU_MODELS
        backend = "threading" if is_gpu_model else "loky"

        if model_type in GPU_MODELS:
            print("⚠️ Serializing TF-based trials to avoid GPU contention.")
            n_jobs_actual = 1
        else:
            # Use our unified CPU-centric thread knob
            n_jobs_actual = int(
                os.getenv("MLB_THREADS", os.getenv("BLAS_THREADS_PER_TRIAL", "8"))
            )


        all_results = Parallel(n_jobs=n_jobs_actual, backend=backend)(
            delayed(evaluate_fold)(s, trn, tst) for s, trn, tst in tqdm(tasks, desc="Walk-forward splits")
        )


        print(f"✅ Parallel walk-forward completed in {time.time() - start:.2f} seconds.")

        all_results = [r for r in all_results if r is not None]
        if not all_results:
            print("❌ WFO failed completely.")
            return None, None

        # Serialize config dicts for grouping
        for r in all_results:
            for key in [
                "cnn_config", "lstm_config", "transformer_config", "xgb_config",
                "dqn_config", "rf_config", "logit_config", "indicator_windows",
            ]:
                if key in r and isinstance(r[key], dict):
                    r[key] = json.dumps(r[key], sort_keys=True)

        df_wfo = pd.DataFrame(all_results)

        # Normalize potential leftover dicts to JSON strings
        for col in [
            "cnn_config", "lstm_config", "transformer_config", "xgb_config",
            "dqn_config", "rf_config", "logit_config",
        ]:
            if col in df_wfo.columns:
                df_wfo[col] = df_wfo[col].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else x)

        metric_cols = {
            "type", "train_start", "train_end", "test_start", "test_end",
            "performance", "return", "sharpe", "drawdown", "trades",
            "geo_mean_ann", "directional_accuracy", "precision_macro",
            "f1_macro", "active_rate", "profit_per_hit",
            "return_per_trade", "win_rate", "strategy_volatility", "kurtosis",
        }
        candidates = [c for c in df_wfo.columns if c not in metric_cols and not str(c).startswith("__top5")]

        def _is_scalar_series(s):
            return s.map(lambda x: isinstance(x, (type(None), bool, int, float, str))).all()

        param_cols = [c for c in candidates if _is_scalar_series(df_wfo[c])]
        if not param_cols:
            print("⚠️ No hashable parameter columns to group by; cannot select best combo.")
            return df_wfo, None

        grouped = (
            df_wfo.groupby(param_cols, dropna=False)["performance"]
            .mean().sort_values(ascending=False)
        )
        if grouped.empty:
            print("⚠️ WFO produced no valid rows to rank; cannot select best combo.")
            return df_wfo, None

        topk_df = grouped.reset_index()
        best_combo = topk_df.iloc[0].to_dict()

        # Carry __top5_* helpers through (don’t group on them)
        for helper in ["__top5_info", "__top5_params", "__top5_path"]:
            if helper in df_wfo.columns:
                first_nonnull = df_wfo[helper].dropna()
                if not first_nonnull.empty:
                    best_combo[helper] = first_nonnull.iloc[0]

        # --- Fallback: if for some reason __top5_params did not survive
        # into df_wfo, pull directly from the backtester attribute set
        # by run_optuna_tuning() earlier.
        try:
            topN_fallback = getattr(self, "_optuna_top5_for_wfo", None) or []
        except Exception:
            topN_fallback = []

        if topN_fallback and not best_combo.get("__top5_params"):
            best_combo["__top5_params"] = topN_fallback
            print(f"[TopN] Attached {len(topN_fallback)} tuned configs to best_combo for real-trading consensus.")

        # Deserialize JSON config columns back to dicts, if any
        for key in ["cnn_config", "lstm_config", "transformer_config", "xgb_config", "dqn_config", "rf_config", "logit_config"]:
            if key in best_combo and isinstance(best_combo[key], str):
                try:
                    best_combo[key] = json.loads(best_combo[key])
                except Exception:
                    pass

        return df_wfo, best_combo

    def real_trading_simulation(self, config, models_to_test=None, months=1):
        """
        Simulate sequential (month-by-month) live trading:
        - per month: tune (or short-circuit for DQN), evaluate, log metrics,
        and carry continuous equity to the next month.

        Returns
        -------
        pd.DataFrame
            One row per successfully evaluated month.
        """      
        _prev_real = getattr(self, "_in_real_sim", False)
        _prev_dbg  = getattr(self, "_dbg_first_bars", False)
        # Real-trading sim must not inherit Optuna CV mode.
        _prev_optuna_cv = getattr(self, "_in_optuna_cv", False)

        self.bar_concat = pd.DataFrame()
        self.eq_concat  = pd.DataFrame()
        self._in_real_sim = True
        self._in_optuna_cv = False
        
        # ------------------------------------------------------------
        # Freeze the baseline feature config at entry to real-trading sim
        # so monthly evaluation cannot drift due to prior trial/Month state.
        # ------------------------------------------------------------
        try:
            self._rt_sim_base_features_config = deepcopy(getattr(self, 'features_config', {}) or {})
        except Exception:
            self._rt_sim_base_features_config = {}
            
        
        # repetition index (for file naming); defaults to 1 if not provided
        rep_idx = int(config.get("rep", 1))

        log_print(
            "IN REAL_TRADING_SIMULATION, model_type is: " + str(config.get("model_type")),
            level="DEBUG",
        )


        # keep evaluation costs consistent across all paths
        # BUT: do not override constructor choice if it was explicitly set.
        if not getattr(self, "_trading_costs_locked", False):
            if "eval_use_trading_costs" in config:
                self.trading_costs = bool(config.get("eval_use_trading_costs"))
            elif "trading_costs" in config:
                self.trading_costs = bool(config.get("trading_costs"))
        else:
            # Optional: leave a breadcrumb in debug logs if you want
            # if debug(): print(f"[Costs] Constructor lock active → ignoring config trading_costs override.")
            pass

            
        self.slippage_factor = float(config.get("slippage_factor", self.slippage_factor))

        def _log_flat_month_fallback(
            month_idx,
            train_start,
            train_end,
            test_start,
            test_end,
            model_type,
            full_data,
            prev_position,
            prev_eq_strategy,
            prev_eq_bh,
        ):
            """
            Log a flat no-trades month when we have no usable WFO combo
            or no valid metrics. Returns updated (prev_eq_strategy, prev_eq_bh).
            """
            cfg_f       = getattr(self, "features_config", {}) or {}
            sess_mode   = str(cfg_f.get("session_filter_mode", "both")).lower()
            use_strict  = bool(cfg_f.get("enforce_day1_start", True))

            # In real_trading_simulation always use strict day-1 anchor
            if getattr(self, "_in_real_sim", False):
                use_strict = True

            # Start from raw month slice
            test_bars = full_data.loc[test_start:test_end].copy()

            # Apply NY session filter if used during testing
            if sess_mode in ("test_only", "both"):
                if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                    try:
                        full_idx  = pd.to_datetime(full_data.index, utc=True, errors="coerce")
                        _ny_times = full_idx.tz_convert("America/New_York")
                        # 02:00–13:00 NY
                        self._ny_mask = pd.Series(
                            (_ny_times.hour >= 2) & (_ny_times.hour <= 13),
                            index=full_idx,
                        )
                    except Exception as _e:
                        print(f"⚠️ Lazy NY mask build failed in flat-month fallback: {_e}")
                        self._ny_mask = pd.Series(True, index=full_data.index)
                test_bars = test_bars.loc[self._ny_mask.reindex(test_bars.index, fill_value=False)]
                
            # Tag session_flag for this month slice as well (mainly for future use
            # if we ever feed these bars into compute_full_evaluation_metrics).
            try:
                if hasattr(self, "_ny_mask") and self._ny_mask is not None:
                    _sess_month = self._ny_mask.reindex(test_bars.index, fill_value=False)
                    test_bars["session_flag"] = _sess_month.astype(int)
                else:
                    test_bars["session_flag"] = 1
            except Exception:
                test_bars["session_flag"] = 1


            # Enforce day-1 anchor + warm-up, consistent with evaluation
            month_start_dt = _ensure_dt(test_start)
            if use_strict and not test_bars.empty:
                try:
                    first_tradable = first_tradable_test_bar(test_bars.index, month_start_dt)
                except Exception as _e:
                    print(f"⚠️ first_tradable_test_bar failed in flat-month fallback: {_e}")
                    first_tradable = None

                try:
                    needed = compute_required_test_warmup_bars(
                        {**cfg_f, "model_type": model_type, "lags": int(cfg_f.get("lags", 8))}
                    )
                except Exception:
                    needed = 0

                warmups    = max(int(needed), 0)
                min_anchor = month_start_dt
                if first_tradable is not None and first_tradable > min_anchor:
                    min_anchor = first_tradable

                try:
                    anchor_idx = test_bars.index.get_loc(min_anchor, method="bfill")
                except Exception:
                    anchor_idx = 0

                if warmups > 0:
                    anchor_idx = min(anchor_idx + warmups, max(len(test_bars) - 1, 0))
                test_bars = test_bars.iloc[anchor_idx:]

            # If there are literally no bars after filters, keep both equities flat
            if test_bars.empty:
                df_flat = test_bars.copy()
                try:
                    df_flat.attrs["signal_coverage"] = 0.0
                    df_flat.attrs["end_eq_strategy"] = float(prev_eq_strategy)
                    df_flat.attrs["end_eq_bh"] = float(prev_eq_bh)
                    df_flat.attrs["last_position"] = float(prev_position)
                except Exception:
                    pass
                prev_position_out = float(prev_position)
                monthly_bh_factor      = 1.0
                equity_strategy        = float(prev_eq_strategy)
                equity_bh              = float(prev_eq_bh)
                perf                   = 1.0
                creturns               = monthly_bh_factor
                outperf                = perf - creturns
                sharpe                 = 0.0
                drawdown               = 0.0
                trades                 = 0
                geo_mean_ann           = 0.0
                directional_accuracy   = 0.0
                precision_macro        = 0.0
                f1_macro               = 0.0
                active_rate            = 0.0
                profit_per_hit         = 0.0
                return_per_trade       = 0.0
                win_rate               = 0.0
                strategy_volatility    = 0.0
                kurtosis               = 0.0
            else:
                # Build a minimal flat-strategy df and run it through the SAME engine
                df_flat = test_bars.copy()

                # Shared baseline returns: always from self.data
                df_flat["returns"] = (
                    self.data["returns"].reindex(df_flat.index).astype(float)
                )

                # Strategy is flat all month
                # Real-sim continuity: if we come in holding a position, keep it unless a model says otherwise.
                # This prevents “teleporting to flat” just because WFO returned nothing.
                df_flat["pred"] = float(prev_position)

                # Minimal spread; full cost model will read more columns if present
                if "spread" not in df_flat.columns:
                    df_flat["spread"] = 0.0

                # ------------------------------------------------------------
                # Propagate (or compute) train-anchored high_vol_thr for costs
                # so fallback never triggers LeakageGuard.
                # ------------------------------------------------------------
                cfg_cost = dict(getattr(self, "features_config", {}) or {})

                # 1) If global config (closure) already has a threshold, reuse it.
                try:
                    _thr_cfg = cfg_cost.get("high_vol_thr", None)
                    if _thr_cfg is None and isinstance(config, dict):
                        _thr_cfg = config.get("high_vol_thr", None)
                    _thr_cfg = float(_thr_cfg) if _thr_cfg is not None else None
                    if _thr_cfg is not None and np.isfinite(_thr_cfg):
                        cfg_cost["high_vol_thr"] = float(_thr_cfg)
                except Exception:
                    pass

                # 2) If still missing, compute from TRAIN window only (no leakage).
                try:
                    if cfg_cost.get("high_vol_thr", None) is None:
                        from utilsNoWFO import realized_vol as _rv_fn
                        _vol_w = int(cfg_cost.get("vol_window_bars", 48))
                        _qhi   = float(cfg_cost.get("high_vol_q", 0.80))

                        train_bars = full_data.loc[train_start:train_end].copy()

                        # Apply NY session filter to TRAIN if your pipeline uses it there.
                        if sess_mode in ("train_only", "both"):
                            if hasattr(self, "_ny_mask") and self._ny_mask is not None:
                                train_bars = train_bars.loc[
                                    self._ny_mask.reindex(train_bars.index, fill_value=False)
                                ]

                        if "returns" in train_bars.columns and len(train_bars) > 0:
                            _rv_tr = _rv_fn(train_bars["returns"].astype(float), window=_vol_w)
                            _thr_tr = float(_rv_tr.quantile(_qhi))
                            if np.isfinite(_thr_tr):
                                cfg_cost["high_vol_thr"] = float(_thr_tr)
                except Exception:
                    pass

                # Ensure the evaluator/cost layer can see the config via attrs as well.
                try:
                    df_flat.attrs["features_config"] = cfg_cost
                    df_flat.attrs["debug_costs"] = bool(self._is_debug())
                except Exception:
                    pass

                # Align cost columns if trading_costs are enabled
                if bool(getattr(self, "trading_costs", True)):
                    # Ensure we pass a TRAIN-anchored high_vol_thr into the cost layer,
                    # even in flat-month fallback, to avoid LeakageGuard forcing HIGH slippage.
                    cfg_cost = dict(getattr(self, "features_config", {}) or {})
                    try:
                        _thr = cfg_cost.get("high_vol_thr", None)
                        if _thr is None:
                            from utilsNoWFO import realized_vol as _rv_fn
                            vol_w = int(cfg_cost.get("vol_window_bars", 96))
                            qhi   = float(cfg_cost.get("high_vol_q", 0.85))
                            _train = self.data.loc[train_start:train_end]
                            if (
                                _train is not None
                                and hasattr(_train, "columns")
                                and "returns" in _train.columns
                                and len(_train) > max(vol_w, 5)
                            ):
                                _rv = _rv_fn(_train["returns"].astype(float), window=vol_w)
                                _thr = float(_rv.quantile(qhi))
                                if _thr is not None and np.isfinite(_thr):
                                    cfg_cost["high_vol_thr"] = float(_thr)
                    except Exception:
                        pass
                    try:
                        df_flat = self._ensure_cost_columns(
                            df_flat, cfg_cost
                        )
                    except Exception as _e:
                        print(f"⚠️ _ensure_cost_columns failed in flat-month fallback: {_e}")

                cont_metrics = compute_full_evaluation_metrics(
                    df_flat,
                    trading_costs=self.trading_costs,
                    slippage_factor=self.slippage_factor,
                    prev_position=float(prev_position),
                    prev_eq_strategy=prev_eq_strategy,
                    prev_eq_bh=prev_eq_bh,
                    eval_context="real_sim:flat_month_fallback",
                )
                
                from utilsNoWFO import validate_metrics_shape
                validate_metrics_shape(cont_metrics, context="real_sim:cont_metrics")

                (
                    perf,
                    outperf,
                    creturns,
                    sharpe,
                    drawdown,
                    trades,
                    geo_mean_ann,
                    directional_accuracy,
                    precision_macro,
                    f1_macro,
                    active_rate,
                    profit_per_hit,
                    return_per_trade,
                    win_rate,
                    strategy_volatility,
                    kurtosis,
                ) = cont_metrics

                # Pull continuous end-of-month equities from engine attrs
                equity_strategy    = float(df_flat.attrs.get("end_eq_strategy", prev_eq_strategy))
                equity_bh          = float(df_flat.attrs.get("end_eq_bh", prev_eq_bh))
                prev_position_out = float(df_flat.attrs.get("last_position", prev_position))
                monthly_bh_factor  = float(creturns)

            result = {
                "month": month_idx,
                "model": model_type,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "cstrategy": perf,
                "creturns": creturns,
                "outperformance": outperf,
                "sharpe": sharpe,
                "drawdown": drawdown,
                "trades": trades,
                "geo_mean_ann": geo_mean_ann,
                "directional_accuracy": directional_accuracy,
                "precision_macro": precision_macro,
                "f1_macro": f1_macro,
                "active_rate": active_rate,
                "signal_coverage": float(df_flat.attrs.get("signal_coverage", float("nan"))),
                "profit_per_hit": profit_per_hit,
                "return_per_trade": return_per_trade,
                "win_rate": win_rate,
                "strategy_volatility": strategy_volatility,
                "kurtosis": kurtosis,
            }

            model_name = friendly_model_name(model_type)
            model_base_dir = os.path.join(out_dir, model_name)
            month_dirs = month_dir_path(model_base_dir, month_idx)
            csv_path   = os.path.join(month_dirs["csv"], f"csv_month_{month_idx}.csv")

            self.log_simulation_result(
                i=month_idx - 1,
                test_start=test_start,
                test_end=test_end,
                perf=float(result["cstrategy"]),         # 1.0 for a flat month
                creturns=float(result["creturns"]),      # BH factor
                sharpe=float(result["sharpe"]),          # 0.0
                trades=int(result["trades"]),            # 0
                drawdown=float(result["drawdown"]),      # 0.0
                cumsum=float(equity_strategy),           # carried strategy equity
                result=result,
                csv_path=csv_path,
                directional_accuracy=float(result["directional_accuracy"]),
                precision_macro=float(result["precision_macro"]),
                f1_macro=float(result["f1_macro"]),
                active_rate=float(result["active_rate"]),
                profit_per_hit=float(result["profit_per_hit"]),
                equity_bh=float(equity_bh),
            )

            results.append(result)
            # Keep the cross-month curve continuous even in a flat-month fallback.
            # NOTE: test_bars is raw market data; we want the evaluated *_cont curves.
            try:
                all_dfs.append(df_flat[["cstrategy_cont", "creturns_cont"]].copy())
            except Exception:
                try:
                    import pandas as _pd
                    all_dfs.append(_pd.DataFrame(columns=["cstrategy_cont", "creturns_cont"]))
                except Exception:
                    pass

            try:
                trade_dfs.append(build_trade_log_from_df(df_flat))
            except Exception:
                trade_dfs.append(pd.DataFrame())

            print(
                f"🟨 Month {month_idx}: logged FLAT month "
                f"(no usable config / no valid metrics)."
            )

            prev_eq_strategy = equity_strategy
            prev_eq_bh       = equity_bh
            
            return prev_eq_strategy, prev_eq_bh, prev_position_out



        def get_first(val, default):
            
            if isinstance(val, (list, tuple)):
                return val[0]
            return val if val is not None else default

        def _is_valid_metrics_tuple(t):
            """Expect a 16-tuple. Reject NaN/Inf/sentinel and no-trade configs."""
            if t is None or not isinstance(t, (list, tuple)) or len(t) < 16:
                return False
            perf, outperf, creturns, sharpe, drawdown, trades, geo_mean_ann, \
            directional_accuracy, precision_macro, f1_macro, active_rate, profit_per_hit, \
            return_per_trade, win_rate, volatility, excess_kurtosis = t

            # In small-sample months (or very flat paths), Sharpe/outperf may be NaN.
            # That should NOT zero out the entire month or prune an Optuna trial.
            # We require the equity results themselves to be finite, and treat
            # non-finite diagnostics as 0.0 for validity purposes.
            try:
                sharpe_v = 0.0 if sharpe is None else float(sharpe)
            except Exception:
                sharpe_v = 0.0
            if not np.isfinite(sharpe_v):
                sharpe_v = 0.0

            try:
                outperf_v = float(outperf)
            except Exception:
                outperf_v = 0.0
            if not np.isfinite(outperf_v):
                outperf_v = 0.0

            arr = np.array([perf, creturns])
            if np.any(~np.isfinite(arr)):
                return False

            if perf <= -9999 or creturns <= -9999:
                return False
            try:
                active_v = float(active_rate)
            except Exception:
                active_v = -1.0
            if not np.isfinite(active_v) or active_v <= 0:
                return False

            try:
                trades_v = float(trades)
            except Exception:
                trades_v = -1.0
            if not np.isfinite(trades_v) or trades_v < 0:
                return False

            return True

        model_type = config.get("model_type", "svm")
        log_print(f"\n📣 Starting Real Trading Simulation for {months} month(s)", level="COMPACT")
        log_print(
            f"🧠 Strategy: {model_type.upper()} | Logging results per month...",
            level="COMPACT",
        )

        out_dir, _stamp = make_results_run_dir() 
        full_data = self.data.copy()
        
        # FeatureBank: keep a stable source across month slices so base indicators
        # are computed once and then reindexed to each month.
        try:
            self.set_feature_bank_source(full_data)
        except Exception:
            # Hard-fail would be silly; the system can always fall back to per-slice TA.
            pass


        # --- Global HPO: tune once per run, reuse hyperparameters every month (non-DQN only) ---
        global_hpo_best = None
        global_hpo_topN = None

        if model_type != "dqn":
            # Decide whether to reuse a cached config or force a fresh study.
            # Default behaviour (use_cached_global_hpo=False) is to always run a
            # new global HPO for this run and overwrite any stale cache.
            use_cached_global_hpo = bool(
                config.get(
                    "use_cached_global_hpo",
                    DEFAULT_CV.get("use_cached_global_hpo", False),
                )
            )

            # If the user requested n_trials <= 0, we must load a cached HPO config
            # and skip Optuna entirely (otherwise the tuner will have zero trials and crash).
            _req_trials = int(config.get("n_trials", 0) or 0)
            _force_cached_hpo = (_req_trials <= 0)
            if _force_cached_hpo:
                use_cached_global_hpo = True

            # 1) Optionally try to load from disk (if reuse is allowed)
            if use_cached_global_hpo:
                try:
                    global_hpo_best, global_hpo_topN = load_hpo_config_from_disk(model_type)
                except Exception:
                    global_hpo_best, global_hpo_topN = None, None
                    
            # If we explicitly forced cached HPO (n_trials <= 0), missing cache is a hard error.
            if _force_cached_hpo:
                if (not isinstance(global_hpo_best, dict)) or (not global_hpo_best):
                    raise RuntimeError(
                        f"[HPO] n_trials=0 but no cached HPO config found for {model_type}. "
                        f"Expected under: {HPO_CONFIG_DIR} (set MLB_HPO_DIR to override)."
                    )
                log_print(
                    f"[HPO] Using cached global HPO for {model_type} (n_trials=0).",
                    level="COMPACT",
                )

            # 2) If cache is disabled or missing/invalid, run a single Optuna study now
            if (not _force_cached_hpo) and ((not use_cached_global_hpo) or (not isinstance(global_hpo_best, dict) or not global_hpo_best)):
                log_print(
                    f"[HPO] Running ONE global Optuna study for {model_type} "
                    f"(use_cached_global_hpo={use_cached_global_hpo})...",
                    level="COMPACT",
                )
                hpo_cfg = dict(config)  # shallow copy is enough

                # Ensure HPO-only flags
                hpo_cfg["hpo_only"] = True
                hpo_cfg.setdefault("hpo_save_to_disk", True)

                # Use TRIAL_COUNTS to configure n_trials / n_startup_trials if not already set
                tc = TRIAL_COUNTS.get(model_type, {})
                default_random = int(tc.get("random", hpo_cfg.get("n_startup_trials", 10)))
                default_bayes = int(tc.get("bayes", max(hpo_cfg.get("n_trials", 30) - default_random, 0)))

                if "n_startup_trials" not in hpo_cfg:
                    hpo_cfg["n_startup_trials"] = default_random
                if "n_trials" not in hpo_cfg:
                    hpo_cfg["n_trials"] = default_random + default_bayes

                # IMPORTANT: for HPO we want the full dataset available, not the per-month slice
                self.data = full_data.copy()

                # We only need params for this model_type
                res_hpo = self.run_strategy(
                    hpo_cfg,
                    models_to_test=[model_type],
                    n_trials=hpo_cfg["n_trials"],
                    n_startup_trials=hpo_cfg["n_startup_trials"],
                )
                # In hpo_only mode we return (None, best_params)
                if isinstance(res_hpo, tuple) and len(res_hpo) >= 2 and isinstance(res_hpo[1], dict):
                    global_hpo_best = res_hpo[1]
                else:
                    global_hpo_best = getattr(self, "_optuna_best_for_wfo", None)

                # If the study persisted a Top-N pool, load it back from disk; otherwise
                # fall back to any in-memory Top-5 captured during the study.
                try:
                    _best_tmp, global_hpo_topN = load_hpo_config_from_disk(model_type)
                except Exception:
                    try:
                        global_hpo_topN = getattr(self, "_optuna_top5_for_wfo", None)
                    except Exception:
                        global_hpo_topN = None

            # Cache on the instance for the monthly loop
            self._global_hpo_best = global_hpo_best
            self._global_hpo_topN = global_hpo_topN or []
        else:
            # DQN: no global HPO, keep behaviour unchanged there
            self._global_hpo_best = None
            self._global_hpo_topN = []



        results = []
        all_dfs = []
        trade_dfs = []   # per-month trade DataFrames (aligned with all_dfs / results)

        # Reset per-run PBO/MCS accumulator
        self._wfo_monthly_records = []

        train_months = get_first(
            config.get("train_months"),
            TRAIN_TEST_MONTHS[model_type]["train"][0]
        )
        test_months = get_first(
            config.get("test_months"),
            TRAIN_TEST_MONTHS[model_type]["test"][0]
        )

        # ---- carry continuous state across months ----
        prev_eq_strategy = 1.0
        prev_eq_bh = 1.0
        prev_position = 0.0
        
        # --- Build model output tree (Months/Final) — do NOT rely on a global RUN_DIR ---
        disp_name = friendly_model_name(model_type)
        
        # --- helper: map model_type → family folder name
        def _infer_family(m: str) -> str:
            m = (m or "").lower()
            classical = {"logistic", "svm", "decision_tree", "random_forest", "xgboost"}
            rl        = {"cnn", "lstm", "transformer"}      # deep models only
            dqn       = {"dqn"}
            ensembles = {"ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost"}
            if m in classical:  return "Classical"
            if m in rl:         return "RL"
            if m in dqn:        return "DQN"
            if m in ensembles:  return "Ensembles"
            return "Classical"

        # --- Determine Top-N size by model family for second-study fallback ---
        mt_local = str(config.get("model_type", model_type)).lower()
        fam = _infer_family(mt_local)
        cfg_local = getattr(self, "features_config", {}) or {}
        if fam == "Classical":
            rt_n = int(cfg_local.get("topN_classical", 4))
        elif fam == "RL":
            rt_n = int(cfg_local.get("topN_deep", 3))
        elif fam == "Ensembles":
            rt_n = int(cfg_local.get("topN_ensemble", 2))
        elif fam == "DQN":
            rt_n = int(cfg_local.get("topN_dqn", 2))
        else:
            rt_n = int(cfg_local.get("topN_default", 3))
        rt_n = max(1, rt_n)

        # Prefer a run dir injected by main(), else the env var, else fall back to this method's out_dir
        RUN_DIR_LOCAL = config.get("_run_dir") or os.environ.get("RESULTS_RUN_DIR") or out_dir
        os.makedirs(RUN_DIR_LOCAL, exist_ok=True)
        
        buckets = comparison_dirs(RUN_DIR_LOCAL) 

        # Compat: allow ensure_model_dirs to return (base, dict) OR a dict
        ret = ensure_model_dirs(RUN_DIR_LOCAL, model_type, disp_name)        
        
        
        if isinstance(ret, tuple):
            model_base_dir, model_dirs_meta = ret
        elif isinstance(ret, dict):
            model_dirs_meta = ret
            # try common keys for the base path
            model_base_dir = (model_dirs_meta.get("base")
                            or model_dirs_meta.get("root")
                            or model_dirs_meta.get("dir")
                            or model_dirs_meta.get("path"))
            if not model_base_dir:
                raise RuntimeError("ensure_model_dirs returned dict without a base path key")
        else:
            raise TypeError(f"Unexpected return from ensure_model_dirs: {type(ret)}")

        final_dirs = model_dirs_meta["final"]
        
        os.makedirs(final_dirs["csv"], exist_ok=True)
        csv_path = os.path.join(final_dirs["csv"], f"real_trading_simulation_{model_type}.csv")

        # --- DQN periodic retraining settings (ignored for other models) ---
        if model_type == "dqn":
            # Retrain once every N test months (default: 12).
            dqn_retrain_period = int(config.get("dqn_retrain_period_months", 12))
            dqn_month_counter = 0
        else:
            dqn_retrain_period = None
            dqn_month_counter = 0
            
        # --- helper: deterministic month config fingerprint (auditable) ---
        def _month_cfg_fingerprint(_cfg: dict):
            """Return (short_sha1, compact_dict) for a stable month-config snapshot."""
            try:
                _keys = [
                    "model_type",
                    "gating_mode",
                    "confidence_threshold",
                    "target_active_rate",
                    "calibrate_method",
                    "label_threshold",
                    "lags",
                    "lags_range",
                    "lag_depth",
                    "roll_windows_key",
                    "roll_windows_key_v2",
                    "runtime_active_band_margin",
                    "runtime_conf_nudge",
                    "runtime_coverage_window",
                    "alpha_vol_z",
                    "beta_spread_norm",
                    "gamma_slip_norm",
                    "vol_window_bars",
                    "high_vol_q",
                    "slip_norm_bps",
                    "eval_use_trading_costs",
                ]
                _snap = {k: _cfg.get(k, None) for k in _keys if k in _cfg}
                _ser = json.dumps(_snap, sort_keys=True, default=str)
                _fp = hashlib.sha1(_ser.encode("utf-8")).hexdigest()[:10]
                return _fp, _snap
            except Exception:
                return "na", {}

        for i in range(months):
            # month_idx must exist even if we crash early in the month loop
            month_idx = i + 1
            
            # V2 export safety: always define, then overwrite if evaluator provides it
            _signal_coverage_month = float("nan")
            
            # ------------------------------------------------------------
            # Allows ctx=real_mX tagging in test_strategy() eligibility prints.
            # ------------------------------------------------------------
            try:
                # NOTE: keep both spellings; downstream log/ctx code reads _rt_month_ix.
                setattr(self, "_rt_month_idx", int(month_idx))
                setattr(self, "_rt_month_ix", int(month_idx))

            except Exception:
                pass
            try:
                test_start_naive  = pd.to_datetime(self.start) + pd.DateOffset(months=37 + i)
                test_end_naive    = test_start_naive + pd.DateOffset(months=test_months)
                train_start_naive = test_start_naive - pd.DateOffset(months=train_months)
                train_end_naive   = test_start_naive - pd.Timedelta(minutes=30)

                # make the slices tz-aware (UTC)
                test_start  = test_start_naive.tz_localize('UTC')
                test_end    = test_end_naive.tz_localize('UTC')
                train_start = train_start_naive.tz_localize('UTC')
                train_end   = train_end_naive.tz_localize('UTC')

                if train_end >= test_start:
                    print(f"❗ Sanity check failed: train_end ({train_end}) is not before test_start ({test_start})")
                    continue

                train_start_nominal = test_start - pd.DateOffset(months=train_months)
                train_start = max(full_data.index[0], train_start_nominal)
                if train_start > train_start_nominal:
                    log_print(
                        f"ℹ️ Train window truncated to data start: "
                        f"{train_start_nominal.date()} → {train_start.date()}",
                        level="COMPACT",
                    )
                log_print(
                    f"📆 Month {i+1}/{months}: "
                    f"Tuning on {train_start.date()} → {train_end.date()} | "
                    f"Testing on {test_start.date()} → {test_end.date()}",
                    level="COMPACT",
                )


                data_slice = full_data.loc[train_start - pd.DateOffset(months=1):test_end].copy()
                log_print(
                    f"📊 Data shape for training/testing window: {data_slice.shape}",
                    level="DEBUG",
                )
                
                # ------------------------------------------------------------
                # Compute ONCE per month from the TRAIN window and cache it so
                # every evaluation path (single model, Top-N, consensus) uses
                # the same volatility regime split without leakage.
                # ------------------------------------------------------------
                try:
                    from utilsNoWFO import realized_vol as _realized_vol
                    _thr_month = None

                    _train_for_thr = full_data.loc[train_start:train_end]

                    # Optional: match train-side session filtering if enabled
                    try:
                        _fc = getattr(self, "features_config", {}) or {}
                        _sf_mode = str(_fc.get("session_filter_mode", "none")).lower()
                        _sf_on_train = bool(_fc.get("session_filter_on_train", False))
                        if _sf_on_train and _sf_mode in ("train", "both") and hasattr(self, "_ny_mask"):
                            _m = getattr(self, "_ny_mask")
                            _train_for_thr = _train_for_thr.loc[_m.reindex(_train_for_thr.index).fillna(False).values]
                    except Exception:
                        pass

                    if isinstance(_train_for_thr, pd.DataFrame) and "returns" in _train_for_thr.columns and len(_train_for_thr) > 10:
                        _vol_w = int((config or {}).get("vol_window_bars", (getattr(self, "features_config", {}) or {}).get("vol_window_bars", 96)))
                        _qhi   = float((config or {}).get("high_vol_q", (getattr(self, "features_config", {}) or {}).get("high_vol_q", 0.85)))
                        _rv = _realized_vol(pd.to_numeric(_train_for_thr["returns"], errors="coerce").astype(float), window=_vol_w)
                        _rv = _rv.dropna()
                        if len(_rv) > 0:
                            _thr_month = float(_rv.quantile(_qhi))

                    setattr(self, "_last_high_vol_thr_train", _thr_month)

                    # Mirror into configs used downstream (safe; only affects cost regime split)
                    try:
                        if isinstance(getattr(self, "features_config", None), dict):
                            if _thr_month is not None:
                                self.features_config["high_vol_thr"] = float(_thr_month)
                            else:
                                self.features_config.pop("high_vol_thr", None)
                    except Exception:
                        pass
                    try:
                        if isinstance(config, dict):
                            if _thr_month is not None:
                                config["high_vol_thr"] = float(_thr_month)
                            else:
                                config.pop("high_vol_thr", None)
                    except Exception:
                        pass

                    if bool(getattr(self, "debug", False)):
                        log_print(f"[DEBUG][Costs] high_vol_thr_train={_thr_month} | ctx=real_sim:m{i+1}", level="DEBUG")
                except Exception:
                    # If anything goes wrong, leave the cache unset; downstream will use BASE slippage.
                    try:
                        setattr(self, "_last_high_vol_thr_train", None)
                    except Exception:
                        pass

                    
                # ── DQN short-circuit: evaluate directly (no Optuna) ──
                if model_type == "dqn":
                    self.data = data_slice

                    lags_val = int(
                        config.get(
                            "lags",
                            getattr(self, "features_config", {}).get(
                                "lags",
                                getattr(self, "features_config", {}).get(
                                    "lags_range", 10
                                ),
                            ),
                        )
                    )

                    # ---- DQN periodic retrain logic ----
                    # We always run with use_pretrained=True so that:
                    # - If no model files exist for this month → train + save new policy.
                    # - If files exist → load and reuse the saved policy (no retrain).
                    if dqn_retrain_period:
                        # At the START of each retrain block (every N months),
                        # delete only the *model* so the next evaluation will train a new one.
                        # KEEP the JSON config so your manual settings persist.
                        if (dqn_month_counter % dqn_retrain_period) == 0:
                            try:
                                if os.path.exists(MODEL_DQN_PATH):
                                    os.remove(MODEL_DQN_PATH)
                                    
                                # IMPORTANT: do NOT remove DQN_AGENT_CONFIG_PATH here (keeps pretrained config alongside weights).
                                log_print(
                                    f"[DQN] Starting new retrain block at month {i + 1}; "
                                    f"removed old pretrained DQN model (kept config).",
                                    level="COMPACT",
                                )
                            except Exception as e:
                                log_print(
                                    f"⚠️ Could not remove old DQN model file: {e}",
                                    level="COMPACT",
                                )

                    raw_dqn_cfg = config.get("dqn_config", None)
                    if isinstance(raw_dqn_cfg, dict) and len(raw_dqn_cfg) > 0:
                         dqn_cfg = dict(raw_dqn_cfg)
                         cfg_source = "inline"
                    else:
                         dqn_cfg = _load_default_dqn_cfg(DQN_GRID_CONFIG_PATH)
                         cfg_source = "grid"
                    print(f"[DQN][CFG] source={cfg_source} episodes={dqn_cfg.get('episodes')} path={DQN_GRID_CONFIG_PATH}")

                    # Ensure DQN path uses the pretrained behaviour in test_dqn_strategy
                    dqn_cfg.setdefault("use_pretrained", True)

                    best_combo = {
                        "model_type": "dqn",
                        "lags": lags_val,
                        "use_extended_features": config.get(
                            "use_extended_features", True
                        ),
                        "dqn_config": dqn_cfg,
                    }

                    try:
                        metrics = self.evaluate_strategy(
                            best_combo, train_start, train_end, test_start, test_end
                        )
                        # Increment DQN month counter only on successful evaluation
                        if dqn_retrain_period:
                            dqn_month_counter += 1
                    except Exception as e:
                        print(f"❌ DQN evaluation failed for month {i + 1}: {e}")
                        metrics = None

                else:
                    # ----------- Non-DQN monthly evaluation -----------
                    # If global_hpo_once is enabled and we have a cached global
                    # config, reuse it here and *do not* run Optuna/WFO again.
                    use_global_hpo = bool(config.get("global_hpo_once", True))

                    df_wfo = None
                    best_combo = None
                    wfo_ok = False

                    if use_global_hpo:
                        # Try to reuse the globally tuned parameters from the header
                        best_combo = getattr(self, "_global_hpo_best", None)
                        topN_from_header = getattr(self, "_global_hpo_topN", None) or []

                        if isinstance(best_combo, dict) and best_combo:
                            # If Top-N list exists but is not attached to the config dict,
                            # attach it so consensus logic works.
                            if topN_from_header and "__top5_params" not in best_combo:
                                best_combo["__top5_params"] = topN_from_header

                            wfo_ok = True
                            print(
                                f"[GlobalHPO] Month {i+1}: reusing globally tuned params for "
                                f"{model_type} (no Optuna/WFO in monthly loop)."
                            )
                        else:
                            print(
                                f"[GlobalHPO] Month {i+1}: no cached global HPO config found "
                                f"for {model_type}; falling back to per-month WFO tuning."
                            )

                    if not wfo_ok:
                        # ----------- Legacy Optuna WFO fallback -----------
                        try:
                            self.data = data_slice
                            config["use_proba"] = config.get("use_proba", True)

                            month_dirs = month_dir_path(model_base_dir, i + 1)

                            config["month_ix"] = int(i + 1)
                            config["month_graphs_dir"] = month_dirs["graphs"]

                            # ⚠️ run_strategy may return None (e.g. ensembles with no valid trials)
                            res = self.run_strategy(
                                config,
                                models_to_test=models_to_test,
                                n_trials=config.get("n_trials", 30),
                                n_startup_trials=config.get("n_startup_trials", 10),
                            )
                            if isinstance(res, tuple) and len(res) >= 2:
                                df_wfo, best_combo = res[0], res[1]
                            else:
                                df_wfo, best_combo = None, None

                            # --- Inject Top-5 params into best_combo for Top-N consensus ---
                            try:
                                top5_params_rt = getattr(self, "_optuna_top5_for_wfo", None) or []
                            except Exception:
                                top5_params_rt = []

                            if (
                                isinstance(best_combo, dict)
                                and top5_params_rt
                                and "__top5_params" not in best_combo
                            ):
                                best_combo["__top5_params"] = top5_params_rt
                                print(
                                    f"[TopN] Attached {len(top5_params_rt)} tuned configs "
                                    "to best_combo for real-trading consensus."
                                )

                        except Exception as e:
                            # Hard fail: do NOT try to restart Optuna here.
                            # We want to see the real error and stop the study.
                            print(f"❌ run_strategy failed in primary WFO: {e}")
                            raise

                        # --- WFO result can be empty if every fold is 0-trade or pruned ---
                        wfo_ok = isinstance(df_wfo, pd.DataFrame) and isinstance(best_combo, dict)
                        if not wfo_ok:
                            # Do NOT crash: this just means no usable config was found.
                            # We let the downstream flat-month fallback handle it.
                            print(
                                f"⚠️ WFO returned no usable result for month {i + 1} "
                                f"(df_wfo={type(df_wfo)}, best_combo={type(best_combo)}). "
                                "Skipping model evaluation and logging a flat no-trades month."
                            )

                    # Restore full_data slice before either evaluation or flat-month fallback
                    self.data = full_data.loc[train_start - pd.DateOffset(months=1):test_end].copy()

                    # Metrics placeholder for this month/model (filled by consensus / Top-3 / single-best)
                    metrics = None


                    # Only build / run Top-N + adaptive Top-3 if WFO actually produced a config
                    if wfo_ok:
                             
                        def _evaluate_with_topn_consensus(base_params):
                            """
                            Build a small Top-N ensemble over the tuned parameter set.

                            High-level intent:
                            - Start from a pool of tuned candidates (Optuna Top-N trials per model type).
                            - Filter for (optionally) style coherence + local similarity (geometry ball) + perf floor.
                            - Form a small committee: base + (N_target-1) best neighbours (by CV objective value).
                            - Evaluate each candidate on the SAME month, align on common index, majority-vote preds ∈ {-1,0,+1}.
                            - Compute metrics for the consensus pred using compute_full_evaluation_metrics.

                            Return:
                            metrics tuple (same format as your evaluation path) or None (caller falls back to single-model logic).
                            """
                            import numpy as _np
                            import pandas as _pd
                            import json as _json
                            import math as _math
                            
                            # --- Robust unwrap: accept either {"best_params": {...}} or the inner dict ---
                            # If caller accidentally passes the outer JSON (model_type/best_params/topN_params),
                            # consensus keys live under best_params, so unwrap here.
                            if (
                                isinstance(base_params, dict)
                                and isinstance(base_params.get("best_params"), dict)
                            ):
                                base_params = base_params["best_params"]


                            cfg_local = getattr(self, "features_config", {}) or {}
                            if not bool(cfg_local.get("deploy_topN_consensus", True)):
                                return None

                            mt_local = str(base_params.get("model_type", getattr(self, "model_type", ""))).lower()
                            # keep behaviour unchanged for RL/DQN (and anything you explicitly want excluded)
                            if mt_local in {"dqn"}:
                                return None

                            # ---------- committee size ----------
                            try:
                                from utilsNoWFO import _infer_family
                                family = _infer_family(mt_local)
                            except Exception:
                                family = "Unknown"

                            if family == "Classical":
                                N_target = int(cfg_local.get("topN_classical", 3))
                            elif family in {"RL"}:  # your code uses "RL" bucket for deep supervised
                                N_target = int(cfg_local.get("topN_deep", 2))
                            elif family == "Ensembles":
                                N_target = int(cfg_local.get("topN_ensemble", 2))
                            elif family == "DQN":
                                N_target = int(cfg_local.get("topN_dqn", 2))
                            else:
                                N_target = int(cfg_local.get("topN_default", 2))

                            N_target = max(2, int(N_target))

                            debug_topn    = bool(cfg_local.get("print_topN_debug", True))
                            style_lock    = bool(cfg_local.get("topN_style_lock", True))
                            geom_radius   = float(cfg_local.get("topN_geom_radius", 0.30))
                            min_perf_frac = float(cfg_local.get("topN_min_perf_frac", 0.60))
                            max_corr      = float(cfg_local.get("topN_max_corr", 0.95))

                            # ---------- 1) pool ----------
                            raw_pool = (
                                base_params.get("__committee_fixed")
                                or base_params.get("__consensus_pool")
                                or base_params.get("__top5_params")
                                or base_params.get("__top3_params")
                                or []
                            )

                            pool_src = (
                                "committee_fixed"
                                if isinstance(base_params.get("__committee_fixed"), list) and len(base_params.get("__committee_fixed")) > 0
                                else ("consensus_pool" if "__consensus_pool" in base_params
                                    else ("top5_params" if "__top5_params" in base_params else "top3_params"))
                            )
                            if not raw_pool:
                                return None

                            raw_topk = []
                            for x in (raw_pool or []):
                                try:
                                    raw_topk.append(dict(x or {}))
                                except Exception:
                                    continue
                            if not raw_topk:
                                return None

                            # ---------- 2) CV meta (direction + values) ----------
                            top_info        = base_params.get("__top5_info") or {}
                            trials_info     = list((top_info.get("trials") or [])) if isinstance(top_info, dict) else []
                            dir_str         = str(top_info.get("direction", "maximize")).lower() if isinstance(top_info, dict) else "maximize"
                            is_minimize     = dir_str.startswith("min")

                            # keep pool + trials_info aligned ONLY for Top-K pools that are index-aligned to trials_info.
                            # For consensus/frozen committees, ordering can differ and per-item metadata should be trusted.
                            pool_src = (
                                "committee_fixed" if isinstance(base_params.get("__committee_fixed"), list) and len(base_params.get("__committee_fixed")) > 0
                                else ("consensus_pool" if isinstance(base_params.get("__consensus_pool"), list) and len(base_params.get("__consensus_pool")) > 0
                                      else "top5_params")
                            )

                            if pool_src != "committee_fixed":
                                # trials_info alignment is only valid for top5_params (same ordering).
                                if pool_src == "top5_params":
                                    if trials_info and len(raw_topk) > len(trials_info):
                                        raw_topk = raw_topk[:len(trials_info)]
                            def _meta_for_pool(idx, alt_dict):
                                """Extract trial_number + objective value for this pool row (robust across formats)."""
                                meta = {}
                                try:
                                    # 1) Prefer metadata stored on the pool dict itself (authoritative)
                                    if isinstance(alt_dict, dict):
                                        tn = alt_dict.get("__trial_number", alt_dict.get("trial_number", None))
                                        vv = alt_dict.get("__cv_value", alt_dict.get("cv_value", alt_dict.get("value", None)))
                                        if tn is not None:
                                            meta["trial_number"] = int(tn)
                                        if vv is not None:
                                            meta["value"] = float(vv)

                                    if meta:
                                        return meta

                                    # 2) Fallback: trials_info aligned by index (less reliable)
                                    if trials_info and idx < len(trials_info):
                                        row = trials_info[idx] or {}
                                        tn = row.get("number", row.get("trial_number", None))
                                        vv = row.get("value", row.get("cv_value", row.get("cv", None)))
                                        if tn is not None:
                                            meta["trial_number"] = int(tn)
                                        if vv is not None:
                                            meta["value"] = float(vv)
                                except Exception:
                                    meta = {}
                                return meta


                            def _meta_value(meta):
                                try:
                                    v = float(meta.get("value")) if isinstance(meta, dict) and meta.get("value") is not None else None
                                    return v if (v is not None and _np.isfinite(v)) else None
                                except Exception:
                                    return None

                            def _meta_trial(meta):
                                try:
                                    t = meta.get("trial_number") if isinstance(meta, dict) else None
                                    return int(t) if t is not None else None
                                except Exception:
                                    return None

                            # compute best_val for perf floor
                            best_val = None
                            vals = []

                            if trials_info:
                                for row in trials_info:
                                    try:
                                        v = float((row or {}).get("value"))
                                        if _np.isfinite(v):
                                            vals.append(v)
                                    except Exception:
                                        pass
                            if not vals:
                                for j, alt in enumerate(raw_topk):
                                    mv = _meta_value(_meta_for_pool(j, alt))
                                    if mv is not None:
                                        vals.append(mv)
                            if vals:
                                best_val = float(_np.min(vals) if is_minimize else _np.max(vals))

                            def _passes_perf_floor(v):
                                # Keep only candidates reasonably close to the best objective value.
                                # IMPORTANT: objectives can be negative in trading; ratio comparisons break on negatives.
                                try:
                                    if best_val is None or v is None:
                                        return True
                                    if not (_np.isfinite(float(best_val)) and _np.isfinite(float(v))):
                                        return False
                                    bv = float(best_val)
                                    vv = float(v)
                                    frac = float(min_perf_frac)
                                    if frac <= 0.0:
                                        return True
                                    if frac >= 1.0:
                                        return (vv <= bv) if is_minimize else (vv >= bv)
                                    tol = abs(bv) * (1.0 - frac)
                                    # minimize: allow up to +tol above best; maximize: allow down to -tol below best
                                    return (vv <= bv + tol) if is_minimize else (vv >= bv - tol)
                                except Exception:
                                    return True

                            # ---------- 3) geometry function (DEFINED BEFORE USE) ----------
                            # Ranges aligned with your tuning ranges (roughly); used only for normalization.
                            HP_RANGES = {
                                "lags_range":         (8.0, 40.0),
                                "lag_depth":          (1.0, 4.0),
                                "target_active_rate": (0.15, 0.35),
                                "label_threshold":    (5e-5, 5e-3),
                                "alpha_vol_z":        (0.0, 0.03),
                                "beta_spread_norm":   (0.0, 0.08),
                                "gamma_slip_norm":    (0.0, 0.08),
                            }

                            def _norm_dist(a, b, key):
                                lo, hi = HP_RANGES.get(key, (None, None))
                                if lo is None or hi is None or hi <= lo:
                                    return _math.inf
                                if a is None or b is None:
                                    return _math.inf
                                try:
                                    return abs(float(a) - float(b)) / (hi - lo)
                                except Exception:
                                    return _math.inf

                            def _within_geom_ball(base_cfg, alt_cfg, radius):
                                """Normalized max-distance ball in key structural/cost-sensitive knobs."""
                                try:
                                    r = float(radius or 0.0)
                                    if r <= 0:
                                        return True

                                    base_vals = {
                                        "lags_range":         base_cfg.get("lags_range", base_cfg.get("lags", None)),
                                        "lag_depth":          base_cfg.get("lag_depth", None),
                                        "target_active_rate": base_cfg.get("target_active_rate", base_cfg.get("target_coverage", None)),
                                        "label_threshold":    base_cfg.get("label_threshold", None),
                                        "alpha_vol_z":        base_cfg.get("alpha_vol_z", None),
                                        "beta_spread_norm":   base_cfg.get("beta_spread_norm", None),
                                        "gamma_slip_norm":    base_cfg.get("gamma_slip_norm", None),
                                    }
                                    alt_vals = {
                                        "lags_range":         alt_cfg.get("lags_range", alt_cfg.get("lags", base_vals["lags_range"])),
                                        "lag_depth":          alt_cfg.get("lag_depth", base_vals["lag_depth"]),
                                        "target_active_rate": alt_cfg.get("target_active_rate", alt_cfg.get("target_coverage", base_vals["target_active_rate"])),
                                        "label_threshold":    alt_cfg.get("label_threshold", None),
                                        "alpha_vol_z":        alt_cfg.get("alpha_vol_z", None),
                                        "beta_spread_norm":   alt_cfg.get("beta_spread_norm", None),
                                        "gamma_slip_norm":    alt_cfg.get("gamma_slip_norm", None),
                                    }

                                    max_d = 0.0
                                    any_finite = False
                                    for k in HP_RANGES.keys():
                                        d = _norm_dist(base_vals.get(k), alt_vals.get(k), k)
                                        if not _math.isfinite(d):
                                            continue
                                        any_finite = True
                                        if d > max_d:
                                            max_d = d

                                    # If no comparable dims, don’t block by geometry.
                                    return True if (not any_finite) else (max_d <= r)
                                except Exception:
                                    return True

                            # ---------- 4) filter pool → eligible neighbours ----------
                            base_style = base_params.get("strategy_type", None)

                            eligible_alts = []
                            eligible_meta = []
                            eligible_audit = []
                            rejected_audit = []

                            for idx, alt_dict in enumerate(raw_topk):
                                meta = _meta_for_pool(idx, alt_dict)
                                reasons = []

                                if style_lock:
                                    alt_style = alt_dict.get("strategy_type", None)
                                    if (base_style is not None) and (alt_style is not None) and (str(base_style) != str(alt_style)):
                                        reasons.append("STYLE_MISMATCH")

                                if not _within_geom_ball(base_params, alt_dict, geom_radius):
                                    reasons.append("OUTSIDE_GEOM_RADIUS")

                                mv = _meta_value(meta)
                                if not _passes_perf_floor(mv):
                                    reasons.append("BELOW_PERF_FLOOR")

                                if reasons:
                                    if debug_topn:
                                        rejected_audit.append({
                                            "idx": idx,
                                            "trial": meta.get("trial_number"),
                                            "value": meta.get("value"),
                                            "reasons": reasons,
                                        })
                                    continue

                                eligible_alts.append(alt_dict)
                                eligible_meta.append(meta)
                                if debug_topn:
                                    eligible_audit.append({
                                        "idx": idx,
                                        "trial": meta.get("trial_number"),
                                        "value": meta.get("value"),
                                    })

                            # ---------- 5) build candidates list (BASE FIRST, ALWAYS) ----------
                            base_core = dict(base_params)
                            # strip helper keys so dict equality/dedup is meaningful
                            for k_rm in ("__top5_params", "__top5_info", "__top5_path", "__consensus_pool"):
                                base_core.pop(k_rm, None)

                            # base meta: try winner row if possible, else fallback to best_val
                            base_meta = {"trial_number": None, "value": best_val}
                            try:
                                winner_idx = int(base_params.get("__winner_index", 0))
                            except Exception:
                                winner_idx = 0
                            try:
                                if trials_info and 0 <= winner_idx < len(trials_info):
                                    row = trials_info[winner_idx] or {}
                                    tn = row.get("number", row.get("trial_number", None))
                                    vv = row.get("value", row.get("cv_value", row.get("cv", None)))
                                    if tn is not None:
                                        base_meta["trial_number"] = int(tn)
                                    if vv is not None:
                                        base_meta["value"] = float(vv)
                            except Exception:
                                pass

                            candidates = [base_core]
                            candidate_meta = [base_meta]

                            # merge each eligible alt onto base_core (non-None overrides)
                            for alt_dict, meta in zip(eligible_alts, eligible_meta):
                                merged = dict(base_core)
                                try:
                                    merged.update({k: v for k, v in (alt_dict or {}).items() if v is not None})
                                except Exception:
                                    pass
                                candidates.append(merged)
                                candidate_meta.append(meta or {})

                            # if we somehow ended up with only base, no consensus possible
                            if len(candidates) < 2:
                                return None

                            # ---------- 6) committee size trim: base + best neighbours ----------
                            selected_trials_pre_dedup = []

                            if len(candidates) > N_target:
                                base_cand = candidates[0]
                                base_m    = candidate_meta[0] if candidate_meta else {}

                                alt_pairs = list(zip(candidates[1:], candidate_meta[1:]))

                                # rank neighbours by objective value (finite only); if missing values, they go last
                                finite = [(c, m, _meta_value(m)) for (c, m) in alt_pairs]
                                finite_sorted = [x for x in finite if x[2] is not None]
                                none_sorted   = [x for x in finite if x[2] is None]

                                finite_sorted.sort(key=lambda x: x[2], reverse=(not is_minimize))

                                k_keep = max(0, int(N_target) - 1)
                                picked = finite_sorted[:k_keep]

                                # if not enough finite-valued neighbours, pad with unknown-valued ones (stable order)
                                if len(picked) < k_keep:
                                    picked += none_sorted[:(k_keep - len(picked))]

                                candidates     = [base_cand] + [c for (c, _, _) in picked]
                                candidate_meta = [base_m]    + [m for (_, m, _) in picked]

                            # snapshot pre-dedup trial ids
                            try:
                                for m in (candidate_meta or []):
                                    t = _meta_trial(m)
                                    if t is not None:
                                        selected_trials_pre_dedup.append(t)
                            except Exception:
                                selected_trials_pre_dedup = []

                            # ---------- 7) de-dup candidates (keep meta aligned) ----------
                            seen = set()
                            uniq_cands = []
                            uniq_meta  = []
                            for cand, meta in zip(candidates, candidate_meta):
                                key = _json.dumps({k: v for k, v in (cand or {}).items() if not str(k).startswith("__")},
                                                sort_keys=True, default=str)
                                if key in seen:
                                    continue
                                seen.add(key)
                                uniq_cands.append(cand)
                                uniq_meta.append(meta)

                            candidates = uniq_cands
                            candidate_meta = uniq_meta

                            # collapse pseudo-committee if all from same trial_number (and not missing)
                            try:
                                tids = []
                                for m in (candidate_meta or []):
                                    t = _meta_trial(m)
                                    tids.append(t if t is not None else "__missing__")
                                uniq = set(tids)
                                if len(candidates) > 1 and len(uniq) == 1 and list(uniq)[0] not in (None, "__missing__"):
                                    if debug_topn:
                                        print(f"[TopN] All committee members share trial_number={list(uniq)[0]}; collapsing to Top-1.")
                                    candidates = candidates[:1]
                                    candidate_meta = candidate_meta[:1]
                            except Exception:
                                pass

                            if len(candidates) < 2:
                                if debug_topn:
                                    print("[TopN] <=1 distinct config after size/dedup; skipping consensus.")
                                return None
                            
                            # Cache the final committee (post filter/trim/dedup) so it stays fixed across months
                            try:
                                base_params["__consensus_committee_cache"] = deepcopy(candidates)
                                base_params["__consensus_committee_meta"]  = deepcopy(candidate_meta)
                                if debug_topn:
                                    print(f"[TopN] Cached committee for reuse across months (k={len(candidates)}) src={pool_src}.")
                            except Exception:
                                pass

                            # ---------- 8) debug: committee table (ONE place, no duplicates) ----------
                            if debug_topn:
                                try:
                                    base = candidates[0]
                                    base_style_dbg  = base.get("strategy_type")
                                    base_lags_dbg   = base.get("lags_range", base.get("lags"))
                                    base_depth_dbg  = base.get("lag_depth")
                                    base_target_dbg = base.get("target_active_rate", base.get("target_coverage"))

                                    print(
                                        f"[TopN] Committee ({len(candidates)} configs) | model={mt_local} "
                                        f"| strategy_type={base_style_dbg} | lags={base_lags_dbg} | depth={base_depth_dbg} | target_active={base_target_dbg}"
                                    )

                                    def _fmt(v, nd=4):
                                        try:
                                            if v is None:
                                                return "—"
                                            if isinstance(v, bool):
                                                return str(v)
                                            if isinstance(v, int):
                                                return str(v)
                                            fv = float(v)
                                            if not _np.isfinite(fv):
                                                return "—"
                                            return f"{fv:.{nd}g}"
                                        except Exception:
                                            return str(v)

                                    # model-family specific extras (keep short)
                                    extra_keys = []
                                    if mt_local == "logistic":
                                        extra_keys = ["logit_C", "logit_penalty", "logit_class_weight"]
                                    elif mt_local == "svm":
                                        extra_keys = ["svm_c", "svm_gamma", "svm_kernel", "svm_degree"]
                                    elif mt_local == "xgboost":
                                        extra_keys = ["xgb_eta", "xgb_max_depth", "xgb_subsample", "xgb_colsample_bytree", "xgb_min_child_weight"]
                                    elif mt_local in {"random_forest", "decision_tree"}:
                                        extra_keys = ["max_depth", "min_samples_leaf", "min_samples_split"]
                                    elif mt_local == "cnn":
                                        extra_keys = ["cnn_num_filters", "cnn_num_layers", "cnn_kernel_size", "cnn_dropout_rate"]
                                    elif mt_local == "lstm":
                                        extra_keys = ["lstm_units", "lstm_num_layers", "lstm_dropout_rate"]
                                    elif mt_local == "transformer":
                                        extra_keys = ["transformer_d_model", "transformer_n_heads", "transformer_num_layers", "transformer_dropout_rate"]
                                    elif mt_local.startswith("ensemble"):
                                        extra_keys = ["ensemble_weight_cnn", "ensemble_weight_lstm", "ensemble_weight_xgb", "ensemble_weight_meta"]

                                    headers = ["id", "trial", "value", "lags", "depth", "target", "label_thr", "conf_thr", "alpha", "beta", "gamma", "extra"]
                                    rows = []
                                    for i_c, (cand, meta) in enumerate(zip(candidates, candidate_meta), start=1):
                                        member_id = "base" if i_c == 1 else str(i_c)
                                        extra = []
                                        for k in extra_keys:
                                            if k in cand:
                                                extra.append(f"{k}={_fmt(cand.get(k), nd=4)}")
                                        rows.append({
                                            "id":        member_id,
                                            "trial":     _fmt(_meta_trial(meta), nd=0),
                                            "value":     _fmt(_meta_value(meta), nd=4),
                                            "lags":      _fmt(cand.get("lags_range", cand.get("lags")), nd=0),
                                            "depth":     _fmt(cand.get("lag_depth"), nd=0),
                                            "target":    _fmt(cand.get("target_active_rate", cand.get("target_coverage")), nd=4),
                                            "label_thr": _fmt(cand.get("label_threshold"), nd=4),
                                            "conf_thr":  _fmt(cand.get("confidence_threshold"), nd=4),
                                            "alpha":     _fmt(cand.get("alpha_vol_z"), nd=3),
                                            "beta":      _fmt(cand.get("beta_spread_norm"), nd=3),
                                            "gamma":     _fmt(cand.get("gamma_slip_norm"), nd=3),
                                            "extra":     ", ".join(extra) if extra else "—",
                                        })

                                    colw = {h: len(h) for h in headers}
                                    for h in headers:
                                        for r in rows:
                                            colw[h] = max(colw[h], len(str(r.get(h, ""))))

                                    def _line(d):
                                        return " | ".join(str(d.get(h, "")).ljust(colw[h]) for h in headers)

                                    print("      " + _line({h: h for h in headers}))
                                    print("      " + " | ".join("-" * colw[h] for h in headers))
                                    for r in rows:
                                        print("      " + _line(r))

                                    print(
                                        f"[TopN][Audit] pool_raw={len(raw_topk)} eligible={len(eligible_audit)} rejected={len(rejected_audit)} "
                                        f"selected_pre_dedup_trials={selected_trials_pre_dedup}"
                                    )
                                    if rejected_audit:
                                        rej = [(x.get("trial"), x.get("value"), ",".join(x.get("reasons") or [])) for x in rejected_audit[:40]]
                                        print(f"[TopN][Audit] rejected(trial,value,why)={rej}")

                                except Exception as _e:
                                    print(f"[TopN] (debug: failed committee print → {_e})")
                            else:
                                # optional single-line info (non-debug)
                                if bool(cfg_local.get("topN_deploy", False)):
                                    try:
                                        print(f"[TopN] committee_size={len(candidates)}/{int(N_target)} geom={geom_radius:.3g} floor={min_perf_frac:.3g}")
                                    except Exception:
                                        pass

                            # ---------- 9) evaluate each candidate & collect per-bar pred/returns ----------
                            bar_dfs = []
                            for idx_c, cand in enumerate(candidates, start=1):
                                try:
                                    # evaluate_strategy is assumed to set self.results to a DF for this candidate
                                    _ = self.evaluate_strategy(cand, train_start, train_end, test_start, test_end)
                                    df_c = getattr(self, "results", None)

                                    if df_c is None or getattr(df_c, "empty", True):
                                        continue
                                    if ("pred" not in df_c.columns) or ("returns" not in df_c.columns):
                                        continue

                                    # ANCHOR: df_loc = df_c[["pred", "returns"]].copy()
                                    # IMPORTANT: df_c["pred"] is executed-time (already shifted by compute_full_evaluation_metrics).
                                    # For committee voting we need decision-time signals. Prefer raw_pred if available.
                                    if "raw_pred" in df_c.columns:
                                        _sig = df_c["raw_pred"]
                                    else:
                                        # Best-effort reconstruction: executed pred -> decision-time (undo 1-bar delay)
                                        _sig = df_c["pred"].shift(-1)

                                    df_loc = _pd.DataFrame(
                                        {
                                            "raw_pred": _pd.to_numeric(_sig, errors="coerce").fillna(0.0).astype(float),
                                            "returns":  _pd.to_numeric(df_c["returns"], errors="coerce").fillna(0.0).astype(float),
                                        },
                                        index=df_c.index,
                                    )
                                    df_loc.index = _pd.to_datetime(df_loc.index, utc=True, errors="coerce")
                                    df_loc = df_loc[~df_loc.index.isna()]
                                    if df_loc.empty:
                                        continue
                                    bar_dfs.append(df_loc)

                                except Exception as _e:
                                    if debug_topn:
                                        print(f"[TopN] Candidate {idx_c} failed during consensus eval: {_e}")

                            if len(bar_dfs) < 2:
                                if debug_topn:
                                    print("[TopN] Need ≥2 valid candidates with pred+returns; skipping.")
                                return None

                            # ---------- 10) align on common index ----------
                            common_idx = bar_dfs[0].index
                            for df_c in bar_dfs[1:]:
                                common_idx = common_idx.intersection(df_c.index)
                            common_idx = common_idx.sort_values()

                            if len(common_idx) == 0:
                                if debug_topn:
                                    print("[TopN] No overlapping bars across candidates; skipping.")
                                return None

                            aligned = [df_c.reindex(common_idx) for df_c in bar_dfs]

                            # drop bars where returns missing in any member
                            mask = _np.ones(len(common_idx), dtype=bool)
                            for df_c in aligned:
                                mask &= df_c["returns"].notna().to_numpy()
                            if not mask.any():
                                if debug_topn:
                                    print("[TopN] All overlapping bars invalid after NA filter; skipping.")
                                return None

                            common_idx = common_idx[mask]
                            aligned = [df_c.loc[common_idx] for df_c in aligned]

                            # ---------- 11) optional correlation diversity filter (on RETURNS series) ----------
                            if len(aligned) > 2 and (max_corr < 1.0):
                                try:
                                    base_sig = _pd.to_numeric(aligned[0]["raw_pred"], errors="coerce").fillna(0.0).astype(float)
                                    keep = [True]
                                    for df_c in aligned[1:]:
                                        sig_c = _pd.to_numeric(df_c["raw_pred"], errors="coerce").fillna(0.0).astype(float)
                                        corr = base_sig.corr(sig_c)
                                        # If corr is NaN/None (e.g., constant signals), don't block diversity.
                                        if corr is None or (not _np.isfinite(float(corr))) or abs(float(corr)) <= max_corr:
                                            keep.append(True)
                                        else:
                                            keep.append(False)

                                    if sum(keep) < 2:
                                        if debug_topn:
                                            print("[TopN] Corr-diversity filter left <2 members; skipping.")
                                        return None

                                    aligned = [df for df, k in zip(aligned, keep) if k]
                                except Exception:
                                    pass

                            # ---------- 12) majority vote + metric eval ----------
                            base_df = aligned[0].copy()
                            preds = _np.stack([df_c["raw_pred"].astype(float).to_numpy() for df_c in aligned], axis=0)

                            # Majority vote on {-1,0,+1} (ties go to 0 because sign(0)=0)
                            consensus_raw = _np.sign(preds.sum(axis=0))
                            # Feed decision-time signals; evaluator applies the 1-bar delay exactly once.
                            base_df["raw_pred"] = consensus_raw
                            # Keep pred numeric for preconditions; evaluator overwrites pred from raw_pred anyway.
                            base_df["pred"] = 0.0
                            
                            # Ensure the evaluator sees the same config-driven execution overlays
                            # (TWAP / kill-switch / gating diagnostics) as the single-model path.
                            # Preserve train-anchored high_vol_thr (prevents leakage-guard HIGH slippage)
                            _fc = dict(cfg_local)
                            try:
                                if _fc.get("high_vol_thr") is None:
                                    _thr_prev = (base_df.attrs.get("features_config", {}) or {}).get("high_vol_thr", None)
                                    if _thr_prev is None:
                                        _thr_prev = getattr(self, "_last_high_vol_thr_train", None)
                                    if _thr_prev is not None:
                                        _fc["high_vol_thr"] = float(_thr_prev)
                            except Exception:
                                pass
                            try:
                                base_df.attrs["features_config"] = dict(_fc)
                                base_df.attrs["debug_costs"] = bool(self._is_debug())
                                base_df.attrs["eval_context"] = "real_sim:topN_consensus"
                            except Exception:
                                pass

                            # Ensure cost columns consistent with your single-model evaluation path (best-effort)
                            try:
                                if bool(getattr(self, "trading_costs", False)):
                                    base_df = self._ensure_cost_columns(base_df, _fc)
                            except Exception:
                                pass

                            # Carry state is only meaningful in real_sim; guard in case this helper is ever re-used elsewhere.
                            try:
                                _pp   = prev_position
                                _peqs = prev_eq_strategy
                                _peqb = prev_eq_bh
                            except Exception:
                                _pp = _peqs = _peqb = None

                            metrics_cons = compute_full_evaluation_metrics(
                                df=base_df,
                                trading_costs=self.trading_costs,
                                slippage_factor=self.slippage_factor,
                                prev_position=_pp,
                                prev_eq_strategy=_peqs,
                                prev_eq_bh=_peqb,
                                eval_context="real_sim:topN_consensus",
                            )

                            # expose for downstream logging/plots
                            self.results = base_df

                            try:
                                metrics_cons = _safe_metrics_return(metrics_cons, context="topN_consensus")
                            except Exception:
                                pass

                            return metrics_cons


                    # ------------------------------------------------------------
                    # Month CONFIG (single source of truth)
                    # ------------------------------------------------------------
                    try:
                        import hashlib as _hashlib, json as _json
                        from copy import deepcopy

                        # 1) deterministic month baseline
                        _month_base = deepcopy(DEFAULT_FEATURES)

                        # restore frozen base (if you set it once at sim start)
                        _month_base.update(deepcopy(getattr(self, "_rt_sim_base_features_config", {}) or {}))

                        # IMPORTANT: allow run-level keys that affect gating/calibration/execution
                        _month_allow = set(DEFAULT_FEATURES.keys()) | {
                            "model_type",
                            "gating_mode",
                            "confidence_threshold",
                            "target_active_rate",
                            "calibrate_method",
                            "label_threshold",
                            "lags", "lags_range", "lag_depth",
                            "roll_windows_key", "roll_windows_key_v2",
                            "runtime_active_band_margin", "runtime_conf_nudge", "runtime_coverage_window",
                            "alpha_vol_z", "beta_spread_norm", "gamma_slip_norm",  "real_sim_target_active_mult",
                            "allow_real_sim_target_active_mult",
                            # cost regime (train-anchored)
                            "vol_window_bars", "high_vol_q", "high_vol_thr",
                            "eval_slip_bps_lo", "eval_slip_bps_hi",
                        }
                        
                        if isinstance(config, dict):
                            _month_base.update({k: deepcopy(v) for k, v in config.items() if k in _month_allow})
                            
                        # Ensure train-anchored high-vol threshold is present for the cost model
                        try:
                            _thr_m = getattr(self, "_last_high_vol_thr_train", None)
                            if _thr_m is not None:
                                _month_base["high_vol_thr"] = float(_thr_m)
                            else:
                                _month_base.pop("high_vol_thr", None)
                        except Exception:
                            pass

                        # ensure model_type exists in baseline (do NOT rely on DEFAULT_FEATURES)
                        _month_base["model_type"] = str(_month_base.get("model_type") or model_type)

                        # Store baseline for auditing (do NOT overwrite self.features_config here)
                        setattr(self, "_month_base_features_config", deepcopy(_month_base))

                        # 2) effective month config = base + tuned params (best_combo)
                        _params = best_combo if isinstance(best_combo, dict) else {}
                        # Preserve internal helper keys (e.g., __top5_params / __consensus_pool) for
                        # downstream Top-N consensus logic, while keeping the *effective* month config
                        # free of internal metadata.
                        _params_internal = {k: deepcopy(v) for k, v in _params.items() if str(k).startswith("__")}
                        _params_clean = {k: v for k, v in _params.items() if not str(k).startswith("__")}

                        _effective = deepcopy(_month_base)
                        _effective.update(deepcopy(_params_clean))
                        
                        # Keep the calibration mapping consistent with the CV-selected params.
                        # (Confidence thresholds are only comparable under the same calibration method.)
                        if _params_clean.get("calibrate_method") is not None:
                            _effective["calibrate_method"] = str(_params_clean.get("calibrate_method")).lower()
                            
                        # --- Real-sim only: bump target_active_rate to offset downstream gates (does NOT affect CV) ---
                        try:
                            _mult_raw = _effective.get('real_sim_target_active_mult', None)
                            _allow_mult = bool(_effective.get('allow_real_sim_target_active_mult', False))
                            if _allow_mult and _mult_raw is not None and _effective.get('target_active_rate') is not None:
                                _mult = float(_mult_raw)
                                if _mult != 1.0:
                                    _tar0 = float(_effective.get('target_active_rate'))

                                    # 1) Clamp multiplier (prevents typos like 10.0)
                                    _mult = float(max(0.80, min(1.30, _mult)))

                                    # 2) Cap effective TAR (prevents "trade-all-bars" behavior)
                                    _tar_cap = float(_effective.get('real_sim_target_active_cap', 0.25))

                                    _effective['target_active_rate'] = float(
                                        max(0.0, min(_tar_cap, _tar0 * _mult))
)
                                    log_print(f"[RealSim][Coverage] m{month_idx} target_active_rate base={_tar0:.3f} mult={_mult:.3f} effective={_effective['target_active_rate']:.3f}", level="COMPACT")
                        except Exception:
                            pass

                        if "lags" not in _effective and "lags_range" in _effective:
                            _effective["lags"] = _effective.get("lags_range")
                        _fp_view = {
                            "model_type": str(_effective.get("model_type", "")),
                            "lags": _effective.get("lags"),
                            "lag_depth": _effective.get("lag_depth"),

                            # NEW: what the engine actually uses
                            "roll_windows": (
                                _effective.get("roll_windows")                  # e.g. [10, 30, 60]
                                or _effective.get("roll_windows_key_v2")         # e.g. "10,30,60"
                                or _effective.get("roll_windows_key")            # e.g. "20,60"
                            ),

                            # keep the raw keys too (for traceability)
                            "roll_windows_key": _effective.get("roll_windows_key"),
                            "roll_windows_key_v2": _effective.get("roll_windows_key_v2"),

                            "target_active_rate": _effective.get("target_active_rate"),
                            "confidence_threshold": _effective.get("confidence_threshold"),
                            "calibrate_method": _effective.get("calibrate_method"),
                            "runtime_active_band_margin": _effective.get("runtime_active_band_margin"),
                            "runtime_coverage_window": _effective.get("runtime_coverage_window"),
                            "runtime_conf_nudge": _effective.get("runtime_conf_nudge"),
                            "alpha_vol_z": _effective.get("alpha_vol_z"),
                            "beta_spread_norm": _effective.get("beta_spread_norm"),
                            "gamma_slip_norm": _effective.get("gamma_slip_norm"),
                        }


                        _fp = _hashlib.sha1(_json.dumps(_fp_view, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
                        log_print(
                            f"[CONFIG-FINGERPRINT] m{month_idx} sha1={_fp} | "
                            f"tar={_fp_view['target_active_rate']} conf={_fp_view['confidence_threshold']} cal={_fp_view['calibrate_method']} | "
                            f"lags={_fp_view['lags']} ld={_fp_view['lag_depth']} | "
                            f"rwk={_fp_view['roll_windows_key']} rwk2={_fp_view['roll_windows_key_v2']} | "
                            f"roll_windows={_fp_view['roll_windows']} | "
                            f"rwin={_fp_view['runtime_coverage_window']} nudge={_fp_view['runtime_conf_nudge']} band={_fp_view['runtime_active_band_margin']}",
                            level="COMPACT",
                        )

                        # Guardrail: drift detection for target_active_rate
                        if "target_active_rate" in _params_clean:
                            _tuned = float(_params_clean.get("target_active_rate"))
                            _eff   = float(_effective.get("target_active_rate"))
                            if abs(_tuned - _eff) > 1e-9:
                                _base = float(_month_base.get("target_active_rate", _tuned))
                                if bool(getattr(self, "_in_real_sim", False)):
                                    log_print(
                                        f"[CONFIG] m{month_idx} target_active_rate base={_base} tuned={_tuned} effective={_eff} (real-sim override)",
                                        level="COMPACT",
                                    )
                                else:
                                    log_print(
                                        f"⚠️ [CONFIG-DRIFT] m{month_idx} target_active_rate tuned={_tuned} effective={_eff}",
                                        level="COMPACT",
                                    )

                        # OPTIONAL BUT STRONGLY RECOMMENDED:
                        # make downstream use the *effective* dict, not the pre-cleaned best_combo
                        best_combo = deepcopy(_effective)
                        # Re-attach internal metadata (Top-N pools, audit helpers, etc.)
                        if isinstance(_params_internal, dict) and _params_internal:
                            try:
                                best_combo.update(deepcopy(_params_internal))
                            except Exception:
                                best_combo.update(_params_internal)

                    except Exception as _e_cfg:
                        log_print(f"⚠️ [CONFIG-FINGERPRINT] m{month_idx} failed: {type(_e_cfg).__name__}: {_e_cfg}", level="COMPACT")
                        
                    # ------------------------------------------------------------
                    # Patch 1: Re-fingerprint immediately before each evaluation call
                    # and after a Top-N candidate is accepted (audit determinism).
                    # This rebuilds an effective config from _month_base_features_config
                    # so the fingerprint matches what evaluate_strategy() will see.
                    # ------------------------------------------------------------
                    def _rt_print_config_fingerprint(_params_in, _tag="pre-eval"):
                        try:
                            import hashlib as __hashlib, json as __json
                            from copy import deepcopy as __deepcopy

                            _base = __deepcopy(getattr(self, "_month_base_features_config", {}) or {})
                            _p = _params_in if isinstance(_params_in, dict) else {}
                            _p_clean = {k: v for k, v in _p.items() if not str(k).startswith("__")}

                            _eff = __deepcopy(_base)
                            _eff.update(__deepcopy(_p_clean))
                            if "lags" not in _eff and "lags_range" in _eff:
                                _eff["lags"] = _eff.get("lags_range")

                            _roll = _eff.get("roll_windows") or _eff.get("roll_windows_key") or _eff.get("roll_windows_key_v2")
                            _conf = _eff.get("confidence_threshold")
                            if _conf is None:
                                _conf = 0.0
                                
                            _view = {
                                "model_type": str(_eff.get("model_type", "")),
                                "lags": _eff.get("lags"),
                                "lag_depth": _eff.get("lag_depth"),
                                "roll_windows": _roll,
                                "target_active_rate": _eff.get("target_active_rate"),
                                "confidence_threshold": _conf,
                                "calibrate_method": _eff.get("calibrate_method"),
                                "runtime_active_band_margin": _eff.get("runtime_active_band_margin"),
                                "runtime_coverage_window": _eff.get("runtime_coverage_window"),                                
                                "runtime_conf_nudge": _eff.get("runtime_conf_nudge"),
                                "alpha_vol_z": _eff.get("alpha_vol_z"),
                                "beta_spread_norm": _eff.get("beta_spread_norm"),
                                "gamma_slip_norm": _eff.get("gamma_slip_norm"),
                            }

                            _sha = __hashlib.sha1(
                                __json.dumps(_view, sort_keys=True, default=str).encode("utf-8")
                            ).hexdigest()[:10]

                            log_print(
                                f"[CONFIG-FINGERPRINT] m{month_idx} {_tag} sha1={_sha} | "
                                f"tar={_view['target_active_rate']} conf={_view['confidence_threshold']} cal={_view['calibrate_method']} | "
                                f"lags={_view['lags']} ld={_view['lag_depth']} | "
                                f"| roll_windows={_view['roll_windows']} | "
                                f"rwin={_view['runtime_coverage_window']} nudge={_view['runtime_conf_nudge']} band={_view['runtime_active_band_margin']}",
                                level="COMPACT",
                            )
                            return _eff
                        except Exception as __e:
                            log_print(
                                f"⚠️ [CONFIG-FINGERPRINT] m{month_idx} {_tag} failed: {type(__e).__name__}: {__e}",
                               level="COMPACT",
                            )
                            return _params_in
                    
                    # ---------- Primary evaluation step (consensus → adaptive Top-3 → single best) ----------
                    # For all non-DQN / non-TF-XGB-DQN models, force the Top-N consensus path
                    # to run, even if a previous refit already produced metrics.

                    params_safe = best_combo if isinstance(best_combo, dict) else {}
                    
                    # model type for gating / debug (must exist even if signal_coverage attrs read fails)
                    mt_eval = str(params_safe.get("model_type", getattr(self, "model_type", ""))).lower()

                    # Secondary activity metric from evaluator (non-neutral label coverage)
                    try:
                        _signal_coverage_month = float(eval_df_cont.attrs.get("signal_coverage", float("nan")))
                    except Exception:
                        _signal_coverage_month = float("nan")


                    # Debug-only: explain whether Top-N consensus will run (logging only; no behaviour change)
                    try:
                        _cfg = getattr(self, "features_config", {}) or {}
                        if bool(_cfg.get("print_topN_debug", False)):
                            _has_pool = bool(
                                params_safe.get("__top5_params") or
                                params_safe.get("__consensus_pool") or
                                params_safe.get("__top3_params")
                            )                            
                            print(f"[TopN][Precheck] metrics_is_none={metrics is None} has_pool={_has_pool} deploy={bool(_cfg.get('deploy_topN_consensus', True))} model={mt_eval}")
                    except Exception:
                        pass

                    # Run Top-N consensus only when explicitly enabled.
                    _deploy_topn = True
                    try:
                        _cfg = getattr(self, "features_config", {}) or {}
                        _deploy_topn = bool(_cfg.get("deploy_topN_consensus", True))
                    except Exception:
                        _deploy_topn = True

                    if _deploy_topn and mt_eval not in {"dqn"} and params_safe:
                        _m_cons = _evaluate_with_topn_consensus(params_safe)
                        if _m_cons is not None:
                            metrics = _m_cons
                    # 2) If consensus is disabled or failed, fall back to previous behavior
                    if metrics is None:
                        try:
                            # ADAPTIVE Top-3 (skip DQN and Transformer-XGB-DQN ensemble)
                            has_top3 = bool(
                                best_combo.get("__top3_params") or best_combo.get("__top5_params")
                            )
                            mt = best_combo.get("model_type", getattr(self, "model_type", ""))

                            # New: flag from CLASS_DEFAULTS["features"]
                            use_adaptive_top3 = bool(
                                (self.features_config or {}).get(
                                    "use_adaptive_top3_for_main_results", False
                                )
                            )

                            if has_top3 and mt not in {"dqn"} and use_adaptive_top3:
                                thr = float(self.features_config.get("switch_hit_rate_thr", 0.45))
                                wnd = int(self.features_config.get("switch_window_days", 5))
                                metrics = self.evaluate_strategy_adaptive_top3(
                                    best_combo,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end,
                                    hit_thr=thr,
                                    window_days=wnd,
                                )
                            else:
                                metrics = self.evaluate_strategy(
                                    best_combo,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end,
                                )
                        except Exception as e:
                            print(f"⚠️ evaluate_strategy failed (primary): {e}")
                            metrics = None

                    # ---------- Top-N fallbacks if needed ----------
                    def _safe_build_topn_candidates(base_params):
                        
                        if isinstance(base_params, dict) and isinstance(base_params.get("best_params"), dict):
                            base_params = base_params["best_params"]
                        
                        base = dict(base_params)
                        raw_topk = base_params.get("__top5_params") or []
                        cands = [base] + [{**base, **deepcopy(alt)} for alt in raw_topk]

                        REQUIRED = [
                            "model_type",
                            "use_extended_features",
                            "lags",
                            "label_threshold",
                            "confidence_threshold",
                        ]
                        for c in cands:
                            for k in REQUIRED:
                                if k not in c and k in base:
                                    c[k] = base[k]

                        return cands


                    # If Top-N fallbacks are disabled, keep the primary result even if invalid.
                    if (not _is_valid_metrics_tuple(metrics)) and (
                        not bool(self.features_config.get("allow_param_fallback", False))
                    ):
                        print("🔒 Realism ON: skipping Top-N fallbacks; keeping primary result (may be NaN/0-trade).")
                    elif not _is_valid_metrics_tuple(metrics) and isinstance(best_combo, dict):
                        print("⚠️ Best combo invalid — trying Top-N fallbacks...")

                        for idx, params_try in enumerate(_safe_build_topn_candidates(best_combo), start=1):
                            try:
                                params_eval = _rt_print_config_fingerprint(params_try, _tag=f"top{idx}-pre")
                                alt_metrics = self.evaluate_strategy(
                                    params_eval, train_start, train_end, test_start, test_end
                                )
                                if _is_valid_metrics_tuple(alt_metrics):
                                    _internal = {k: v for k, v in (params_try or {}).items() if str(k).startswith("__")}
                                    best_combo = dict(params_eval) if isinstance(params_eval, dict) else dict(params_try)
                                    best_combo.update(_internal)
                                    _rt_print_config_fingerprint(best_combo, _tag=f"top{idx}-ACCEPT")
                                    best_combo = params_try
                                    metrics = alt_metrics
                                    print(f"    ✅ Using Top-{idx} candidate (non-degenerate result).")
                                    break
                                else:
                                    print(f"    ⚠️ Top-{idx} candidate degenerate (e.g., 0 trades). Trying next...")
                            except Exception as e:
                                print(f"    ✖️ Top-{idx} candidate crashed: {e}")
                                continue

                    # If Top-N fallbacks are disabled, keep the primary result even if invalid.
                    if (not _is_valid_metrics_tuple(metrics)) and (
                        not bool(self.features_config.get("allow_param_fallback", False))
                    ):
                        print(
                            "🔒 Realism ON: skipping Top-N fallbacks; keeping primary result "
                            "(may be NaN/0-trade)."
                        )

                    elif not _is_valid_metrics_tuple(metrics):
                        print("⚠️ Best combo invalid — trying Top-N fallbacks...")

                        for idx, params_try in enumerate(_safe_build_topn_candidates(best_combo), start=1):
                            try:
                                params_eval = _rt_print_config_fingerprint(params_try, _tag=f"top{idx}-pre")
                                alt_metrics = self.evaluate_strategy(
                                    params_eval, train_start, train_end, test_start, test_end
                                )
                                if _is_valid_metrics_tuple(alt_metrics):
                                    _internal = {k: v for k, v in (params_try or {}).items() if str(k).startswith("__")}
                                    best_combo = dict(params_eval) if isinstance(params_eval, dict) else dict(params_try)
                                    best_combo.update(_internal)
                                    _rt_print_config_fingerprint(best_combo, _tag=f"top{idx}-ACCEPT")
                                    print(f"    ✅ Using Top-{idx} candidate (non-degenerate result).")
                                    break
                                else:
                                    print(
                                        f"    ⚠️ Top-{idx} candidate degenerate (e.g., 0 trades). "
                                        "Trying next..."
                                    )
                            except Exception as e:
                                print(f"    ✖️ Top-{idx} candidate crashed: {e}")
                                continue

                        # If still invalid after trying Top-N, just fall through.
                        if not _is_valid_metrics_tuple(metrics):
                            print(
                                "⚠️ Top-N fallbacks exhausted; keeping primary/degenerate "
                                "result for this month."
                            )
                            
                # ------------------------------------------------------------
                # Gate diagnostic (debug-only):
                # Print median threshold actually used vs max_conf distribution
                # so CV vs real-sim mismatches cannot hide silently.
                # ------------------------------------------------------------
                try:
                    from utilsNoWFO import print_conf_stats
                    _thr_med = getattr(self, "_last_conf_thr_used", None)
                    _conf    = getattr(self, "_last_conf_stats_max_conf", None)
                    if self._is_debug():
                        print_conf_stats(_conf, label=f"real_m{month_idx}", thr=_thr_med)
                except Exception:
                    pass
                
                
                # --- Patch B: real-sim metric sanitization (avoid false "no valid trades") ---
                # Some months can produce a few trades but a secondary metric (e.g., Sharpe/PSR)
                # becomes non-finite due to tiny-sample variance. In real-sim we still want to
                # log the month (equity curve/trade stats) rather than force a flat month.
                try:
                    _in_real = bool(getattr(self, "_in_real_sim", False))
                    _in_cv   = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                    if _in_real and (not _in_cv) and isinstance(metrics, tuple) and len(metrics) >= 16:
                        _mm = list(metrics)
                        _repl = []
                        for _i in (4, 12, 13, 14):  # sharpe, psr, dsr, calmar
                            try:
                                if not np.isfinite(float(_mm[_i])):
                                    _mm[_i] = 0.0
                                    _repl.append(_i)
                            except Exception:
                                _mm[_i] = 0.0
                                _repl.append(_i)
                        if _repl:
                            print(f"[RealSim][MetricsSanitize] m{month_idx} replaced non-finite metric(s) at idx={_repl}")
                            metrics = tuple(_mm)
                except Exception:
                    pass

                # Skip if still invalid
                # If still invalid (e.g., no trades), log a flat month instead of skipping
                if not _is_valid_metrics_tuple(metrics):
                    prev_eq_strategy, prev_eq_bh, prev_position = _log_flat_month_fallback(
                        month_idx=month_idx,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        model_type=model_type,
                        full_data=full_data,
                        prev_position=prev_position,
                        prev_eq_strategy=prev_eq_strategy,
                        prev_eq_bh=prev_eq_bh,
                    )
                    continue



                # -------- Build continuous-month DF + carry state --------
                (perf, outperf, ret, sharpe, drawdown, trades,
                geo_mean_ann, directional_accuracy, precision_macro, f1_macro,
                active_rate, profit_per_hit, return_per_trade, win_rate,
                strategy_volatility, kurtosis_val) = metrics

                if hasattr(self, "results") and isinstance(self.results, pd.DataFrame):
                    # Base df with whatever the model produced
                    test_df = self.results.loc[
                        (self.results.index >= test_start) & (self.results.index <= test_end)
                    ].copy()

                    # --- Build a canonical evaluation index for this month ---
                    # 1) Raw month slice from full_data
                    cfg_f = getattr(self, "features_config", {}) or {}
                    sess_mode = str(cfg_f.get("session_filter_mode", "both")).lower()
                    use_strict = bool(cfg_f.get("enforce_day1_start", True))
                    if getattr(self, "_in_real_sim", False):
                        use_strict = True

                    test_bars = full_data.loc[test_start:test_end].copy()

                    # 2) Apply the same NY session filter used during testing
                    if sess_mode in ("test_only", "both"):
                        if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                            try:
                                full_idx = pd.to_datetime(full_data.index, utc=True, errors="coerce")
                                _ny_times = full_idx.tz_convert("America/New_York")
                                # 02:00–13:00 NY
                                self._ny_mask = pd.Series(
                                    (_ny_times.hour >= 2) & (_ny_times.hour <= 13),
                                    index=full_idx,
                                )
                            except Exception as _e:
                                print(f"⚠️ Lazy NY mask build failed in real-trading eval: {_e}")
                                self._ny_mask = pd.Series(True, index=full_data.index)
                        test_bars = test_bars.loc[
                            self._ny_mask.reindex(test_bars.index, fill_value=False)
                        ]

                    # 3) Enforce day-1 calendar anchor (same rule for ALL models)
                    eval_index = test_bars.index
                    if use_strict and not test_bars.empty:
                        month_start_dt = _ensure_dt(test_start)
                        try:
                            first_eval_ts = enforce_day1_eval_anchor(test_bars.index, month_start_dt)
                            eval_index = test_bars.loc[first_eval_ts:].index
                        except Exception as _e:
                            print(f"⚠️ enforce_day1_eval_anchor failed in real-trading eval: {_e}")
                            # fallback: keep full test_bars index

                    # 4) Reindex model outputs onto canonical index
                    if not test_df.empty and len(eval_index) > 0:
                        # Align to the canonical monthly timeline
                        test_df = test_df.reindex(eval_index)

                        # Shared buy-and-hold baseline: always from the same returns stream
                        test_df["returns"] = (
                            self.data["returns"].reindex(test_df.index).astype(float)
                        )

                        # If the model never produced preds for some bars, treat them as flat
                        if "pred" not in test_df.columns:
                            test_df["pred"] = 0.0
                        test_df["pred"] = test_df["pred"].fillna(0.0)
                        
                        # If raw_pred exists (decision-time preds), reindexing can introduce NaNs.
                        # Treat those as flat so causality/shift logic remains consistent.
                        if "raw_pred" in test_df.columns:
                            test_df["raw_pred"] = pd.to_numeric(test_df["raw_pred"], errors="coerce").fillna(0.0)

                        # Ensure evaluator has consistent cost columns (spread, price/mid_close, slippage_bps)
                        # so cont_metrics uses the same cost model as the single-model evaluation path.
                        try:
                            cfg_cost = {}
                            try:
                                cfg_cost = dict((test_df.attrs.get("features_config", {}) or {}))
                            except Exception:
                                cfg_cost = {}

                            # cfg_local (this model/month config) takes precedence when available
                            try:
                                if isinstance(locals().get("cfg_local", None), dict):
                                    cfg_cost = dict(cfg_cost)
                                    cfg_cost.update(dict(locals().get("cfg_local") or {}))
                            except Exception:
                                pass

                            try:
                                test_df.attrs["features_config"] = dict(cfg_cost)
                                test_df.attrs["debug_costs"] = bool(self._is_debug())
                                test_df.attrs["eval_context"] = "real_sim:month_eval:cont_metrics:prep"
                            except Exception:
                                pass

                            if bool(getattr(self, "trading_costs", False)):
                                # Avoid copying the full feature frame when attaching cost columns.
                                _td_cost = test_df[["returns"]] if ("returns" in test_df.columns) else test_df.loc[:, []]
                                _cost_df = self._ensure_cost_columns(_td_cost, cfg_cost)
                                for _c in ("spread", "slippage_bps"):
                                    if _c in _cost_df.columns:
                                        test_df[_c] = _cost_df[_c].reindex(test_df.index)
                                if self._is_debug():
                                    if ("spread" not in test_df.columns) or ("slippage_bps" not in test_df.columns):
                                        print("[Costs][Warn] cont_metrics frame missing spread/slippage_bps after _ensure_cost_columns.")
                        except Exception as _e:
                            if self._is_debug():
                                print(f"⚠️ Cost-column prep failed in real_sim cont_metrics: {_e}")




                        # test_df coming from test_strategy has already been evaluated once
                        # (it already contains continuous curves), meaning its 'pred' is in executed-time.
                        # Reconstruct decision-time 'pred' so cont_metrics applies exactly ONE shift.
                        df_for_cont = test_df
                        if ("cstrategy_cont" in df_for_cont.columns) and ("pred" in df_for_cont.columns):
                            df_for_cont = df_for_cont.copy()
                            df_for_cont["pred"] = df_for_cont["pred"].shift(-1).fillna(0.0)

                        cont_metrics = compute_full_evaluation_metrics(
                            df_for_cont,
                            trading_costs=self.trading_costs,
                            slippage_factor=self.slippage_factor,
                            prev_position=prev_position,
                            prev_eq_strategy=prev_eq_strategy,
                            prev_eq_bh=prev_eq_bh,
                            eval_context="real_sim:month_eval:cont_metrics",
                        )
                        
                        from utilsNoWFO import validate_metrics_shape
                        validate_metrics_shape(cont_metrics, context="real_sim:cont_metrics")

                        (perf, outperf, ret, sharpe, drawdown, trades,
                        geo_mean_ann, directional_accuracy, precision_macro, f1_macro,
                        active_rate, profit_per_hit, return_per_trade, win_rate,
                        strategy_volatility, kurtosis_val) = cont_metrics

                        # IMPORTANT: use the *post-cont_metrics* frame for carry + plots.
                        # test_df came from test_strategy (monthly-rebased). df_for_cont has
                        # been re-evaluated with prev_eq_* so its *_cont curves are truly continuous.
                        eval_df_cont = df_for_cont
                        

                        # V2: secondary activity metric (signal coverage) from evaluator attrs
                        try:
                            _signal_coverage_month = float(eval_df_cont.attrs.get("signal_coverage", float("nan")))
                        except Exception:
                            _signal_coverage_month = float("nan")

                        # carry-out for next month (continuous equities)
                        prev_position    = float(eval_df_cont.attrs.get("last_position", prev_position))
                        prev_eq_strategy = float(eval_df_cont.attrs.get("end_eq_strategy", prev_eq_strategy))
                        prev_eq_bh       = float(eval_df_cont.attrs.get("end_eq_bh", prev_eq_bh))

                        # per-trade log for this month (built from the *continuous* evaluated df)
                        try:
                            trade_df_month = build_trade_log_from_df(eval_df_cont)
                        except Exception as _e:
                            print(f"⚠️ Could not build trade log for month {i + 1}: {_e}")
                            trade_df_month = None

                        # save continuous curves for the cross-month plot
                        all_dfs.append(eval_df_cont[["cstrategy_cont", "creturns_cont"]].copy())
                        trade_dfs.append(trade_df_month)
                        
                    else:
                        print("⚠️ results DataFrame missing required columns — skipping bar concat.")
                else:
                    print("⚠️ No self.results to build bar DF from.")


                # Carry-over equities are already updated just above (prev_eq_strategy / prev_eq_bh)
                monthly_bh_factor = float(ret)                  # BH factor this month (continuous)
                equity_strategy   = float(prev_eq_strategy)     # carried strategy equity
                equity_bh         = float(prev_eq_bh)           # carried BH equity

                # Safely capture optional fields for downstream plotting/reconstruction
                features_used = list(getattr(self, "_last_used_features", []))

                _ct_init = best_combo.get("confidence_threshold_init")
                if _ct_init is None:
                    _ct_init = getattr(self, "_last_conf_thr_init", float("nan"))

                _ct_used = getattr(self, "_last_conf_thr_used", None)
                if _ct_used is None:
                    _ct_used = best_combo.get("confidence_threshold")
                if _ct_used is None:
                    _ct_used = float("nan")

                _backoff = getattr(self, "_last_conf_backoff_steps", 0) or 0
                _max_q75 = getattr(self, "_last_max_conf_q75", float("nan"))
                _max_q90 = getattr(self, "_last_max_conf_q90", float("nan"))
                
                # ------------------------------------------------------------------
                # Patch C: Compact per-month gating summary (real-sim; no behavior change)
                # Prints: active_rate, trades, conf_init/used, eligible bars, anchor,
                # and top 3 “filter” contributors if available.
                # ------------------------------------------------------------------
                try:
                    _diag = getattr(self, "_last_eligibility_diag", {}) or {}

                    # Pull components FIRST (avoid UnboundLocalError / silent swallow)
                    _elig_n = int(_diag.get("eligible_bars", 0) or 0)
                    _sess_d = int(_diag.get("session_dropped", 0) or 0)
                    _emb_d  = int(_diag.get("embargo_dropped", 0) or 0)
                    _warm_d = int(_diag.get("warmup_dropped", 0) or 0)
                    _anch_d = int(_diag.get("anchor_dropped", 0) or 0)
                    _anchor = _diag.get("eval_anchor_ts", None)

                    _sum_parts = _elig_n + _sess_d + _emb_d + _warm_d + _anch_d

                    # bars_total must be additive on the eval grid; warn if inconsistent.
                    _bars_total = int(_diag.get("bars_total", 0) or 0)
                    _post_emb_n = int(_diag.get("post_embargo_bars", 0) or 0)

                    # Ensure denominator is ALWAYS defined for summary formatting.
                    # Preference order:
                    #  1) explicit bars_total (additive eval grid),
                    #  2) post_embargo_bars (same meaning in older diags),
                    #  3) eligible + anchor_dropped (also additive within post-embargo bars),
                    #  4) last-resort fallback to any positive count.
                    if _bars_total > 0:
                        _total_n = int(_bars_total)
                    elif _post_emb_n > 0:
                        _total_n = int(_post_emb_n)
                    else:
                        _ea = int(_elig_n + _anch_d)
                        _total_n = int(max(_ea, _elig_n, _sum_parts, 0))


                    # NOTE: these diagnostic counts are NOT mutually exclusive.
                    # Example: `eligible` is (by construction) a subset of `session`,
                    # so summing will usually exceed `bars_total`. Only warn on impossible
                    # accounting (a component exceeds total bars).
                    _parts = {
                        "eligible": int(_elig_n),
                        "session": int(_sess_d),
                        "embargo": int(_emb_d),
                        "warmup": int(_warm_d),
                        "anchor": int(_anch_d),
                    }
                    _max_part = max(_parts.values()) if _parts else 0
                    if _bars_total > 0 and _max_part > _bars_total:
                        print(f"⚠️ [GateSummary][WARN] impossible eligibility counts: bars_total={_bars_total} max_part={_max_part} parts={_parts}")
                    elif bool(getattr(self, "debug", False)) and _bars_total > 0 and _sum_parts > 0 and _bars_total != _sum_parts:
                        # Overlap is expected; keep this at debug-level only.
                        try:
                            log_print(
                                f"[GateSummary][DEBUG] overlapping eligibility counts (expected): "
                                f"bars_total={_bars_total} sum={_sum_parts} parts={_parts}",
                                level="DEBUG",
                            )
                        except Exception:
                            pass

                    # Conf gate filtered count (uses last max_conf snapshot if present)
                    _conf = getattr(self, "_last_conf_stats_max_conf", None)
                    _conf_filt = None
                    try:
                        # _ct_used is expected to exist in this scope (computed upstream)
                        if _conf is not None and np.size(_conf) > 0 and (_ct_used == _ct_used):
                            _conf_arr = np.asarray(_conf, dtype=float)
                            _conf_filt = int(np.sum(_conf_arr < float(_ct_used)))
                    except Exception:
                        _conf_filt = None

                    _reasons = []
                    if _conf_filt is not None:
                        _reasons.append(("conf", _conf_filt))
                    if _sess_d:
                        _reasons.append(("session", _sess_d))
                    if _emb_d:
                        _reasons.append(("embargo", _emb_d))
                    if _anch_d:
                        _reasons.append(("anchor", _anch_d))

                    _reasons = sorted(_reasons, key=lambda kv: kv[1], reverse=True)[:3]
                    _top = ", ".join([f"{k}:{int(v)}" for k, v in _reasons]) if _reasons else "n/a"

                    # Robust formatting (avoid exceptions if upstream values are nan/missing)
                    _ar = float(active_rate) if active_rate == active_rate else float("nan")
                    _tr = int(trades) if trades == trades else 0
                    _ci = float(_ct_init) if _ct_init == _ct_init else float("nan")
                    _cu = float(_ct_used) if _ct_used == _ct_used else float("nan")

                    log_print(
                        f"[GateSummary][M{i+1}] "
                        f"ar={_ar:.3f} trades={_tr} "
                        f"conf_init={_ci:.3f} conf_used={_cu:.3f} "
                        f"eligible={_elig_n}/{_total_n} anchor={_anchor} "
                        f"drops(sess={_sess_d}, emb={_emb_d}, warm={_warm_d}, anch={_anch_d}) "
                        f"top=[{_top}]",
                        level="COMPACT",
                    )
                except Exception as _e:
                    # No silent pass: if summary fails, it MUST be visible during audits.
                    log_print(f"⚠️ [GateSummary][WARN] failed to print gating summary: {_e}", level="COMPACT")



                # Safely read from best_combo; if it's not a dict, fall back to {}
                params_safe = best_combo if isinstance(best_combo, dict) else {}
                
                 
                # Effective confidence threshold for this month (USED by runtime gating/backoff).
                # Schema rule: monthly CSV column `confidence_threshold` must reflect the effective used value.
                _ct_param = params_safe.get("confidence_threshold", None)
                _ct_eff = self._safe_float(_ct_used)
                if not np.isfinite(_ct_eff):
                    _ct_eff = self._safe_float(_ct_param)


                result = {
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_months": train_months,
                    "test_months": test_months,
                    
                    # Patch D (persist eligibility deltas for auditability)
                    "elig_raw_month_bars": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("raw_month_bars", 0) or 0),
                    "elig_session_dropped": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("session_dropped", 0) or 0),
                    "elig_embargo_dropped": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("embargo_dropped", 0) or 0),
                    "elig_anchor_dropped": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("anchor_dropped", 0) or 0),
                    "elig_warmup_need": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("warmup_need", 0) or 0),
                    "elig_eval_anchor_ts": (getattr(self, "_last_eligibility_diag", {}) or {}).get("eval_anchor_ts", None),
                    "elig_eligible_bars": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("eligible_bars", 0) or 0),

                    # factors/returns (continuous month)
                    "cum_return":          round(monthly_bh_factor - 1.0, 6),
                    "strategy_return":     round(perf - 1.0, 6),

                    # core knobs used by reconstructor
                    "lags":                 params_safe.get("lags"),
                    "lags_range":           params_safe.get("lags_range"),
                    "lag_depth":            params_safe.get("lag_depth"),
                    "roll_windows":         params_safe.get("roll_windows"),
                    "include_raw_lags":     params_safe.get("include_raw_lags"),

                    # thresholds / switches
                    "label_threshold":      params_safe.get("label_threshold"),
                    "confidence_threshold": _ct_eff,
                    "confidence_threshold_param": _ct_param,
                    "confidence_threshold_init": float(_ct_init),
                    "confidence_threshold_used": float(_ct_used),
                    "conf_backoff_steps":        int(_backoff),
                    "max_conf_q75":              float(_max_q75),
                    "max_conf_q90":              float(_max_q90),

                    "use_extended_features": params_safe.get("use_extended_features"),
                    "use_proba":             params_safe.get("use_proba"),
                    "strategy_type":         params_safe.get("strategy_type"),

                    # per-indicator toggles
                    "use_sma":    params_safe.get("use_sma"),
                    "use_ema":    params_safe.get("use_ema"),
                    "use_macd":   params_safe.get("use_macd"),
                    "use_rsi":    params_safe.get("use_rsi"),
                    "use_bbands": params_safe.get("use_bbands"),
                    "use_atr":    params_safe.get("use_atr"),
                    "use_stoch":  params_safe.get("use_stoch"),
                    "use_adx":    params_safe.get("use_adx"),
                    "use_mtf_ma": params_safe.get("use_mtf_ma"),

                    "indicator_windows": params_safe.get("indicator_windows"),

                    # model metadata
                    "model_type": model_type,

                    # performance snapshot (continuous month + carried equities)
                    "cstrategy":           round(float(perf), 6),
                    "creturns":            round(float(monthly_bh_factor), 6),
                    "outperformance":      round(float(perf) - float(monthly_bh_factor), 6),
                    "equity_strategy":     round(equity_strategy, 6),
                    "equity_bh":           round(equity_bh, 6),
                    "equity_outperformance": round(equity_strategy - equity_bh, 6),

                    # detailed metrics
                    "sharpe":              float(sharpe),
                    "drawdown":            float(drawdown),
                    "trades":              int(trades) if trades == trades else 0,
                    "directional_accuracy":float(directional_accuracy),
                    "precision_macro":     float(precision_macro),
                    "f1_macro":            float(f1_macro),
                    "active_rate":         float(active_rate),
                    "signal_coverage":     float(_signal_coverage_month),
                    "profit_per_hit":      float(profit_per_hit),
                    "return_per_trade":    float(return_per_trade),
                    "win_rate":            float(win_rate),
                    "strategy_volatility": float(strategy_volatility),
                    "kurtosis":            float(kurtosis_val),

                    # trace features actually used this month
                    "features_used": features_used,
                }

                # Only try to serialize sub-configs if best_combo is a dict
                if isinstance(best_combo, dict):
                    for key in ["cnn_config", "lstm_config", "transf_config", "dqn_config",
                                "xgb_config", "rf_config", "logit_config"]:
                        if key in best_combo:
                            result[key] = json.dumps(best_combo[key], sort_keys=True)

                # Call stays with your current signature
                self.log_simulation_result(
                    i=i,
                    test_start=test_start, test_end=test_end,
                    perf=float(perf),
                    creturns=float(monthly_bh_factor),
                    sharpe=float(sharpe), trades=int(trades) if trades == trades else 0,
                    drawdown=float(drawdown), cumsum=float(equity_strategy),  # pass strategy equity here
                    result=result, csv_path=csv_path,
                    directional_accuracy=float(directional_accuracy),
                    precision_macro=float(precision_macro),
                    f1_macro=float(f1_macro), active_rate=float(active_rate),
                    profit_per_hit=float(profit_per_hit),
                    equity_bh=float(equity_bh)
                )

                results.append(result)
                
                # PBO/MCS monthly bookkeeping (does not affect trading logic)
                try:
                    self._record_wfo_monthly_result(result)
                except Exception as _e:
                    if self._is_debug():
                        print(f"[PBO/MCS] Failed to record monthly result: {_e}")
                        
                time.sleep(1)
            finally:
                _hard_free()
                import gc as _gc
                _gc.collect()

                # Also clear feature cache after each month to avoid accumulation
                self._clear_feature_cache()
        
        # ---------------------------------------------------------------------
        # Wrap-up: aggregate results, save artifacts into this run's out_dir
        # ---------------------------------------------------------------------
        df_months = pd.DataFrame(results)
        if not df_months.empty:
            print(f"\n✅ Real Trading Simulation Complete ({months} Months)")

            # df_months contains only the valid months; all_dfs has one per-bar DF per valid month in order.
            df_months_reset = df_months.reset_index(drop=True)
            df_rows = df_months_reset.to_dict(orient="records")

            # Month-level artifacts (csv_month_k, featuresconfigused_k.txt,
            # monthly_equity_k.png, feature_heatmap_k.png) are now handled
            # by the bar_concat-based block further below, gated by SAVE_* flags.


            # -------------------------------------------------------------
            # Monthly trade summary for this model & repetition
            # -------------------------------------------------------------
            if SAVE_TRADES.get("monthly_summary_per_rep_csv", True):
                try:
                    import pandas as _pd
                    import os as _os

                    monthly_trade_summaries = []

                    # df_months_reset and trade_dfs are in the same order of valid months
                    for idx, row in df_months_reset.iterrows():
                        if idx >= len(trade_dfs):
                            continue
                        tdf = trade_dfs[idx]
                        if tdf is None or tdf.empty:
                            continue

                        month_idx = int(row.get("month_idx", idx + 1))
                        n_trades = int(len(tdf))

                        if n_trades > 0:
                            wins = tdf["pnl_pct"] > 0
                            win_rate = float(wins.mean())

                            avg_pnl = float(tdf["pnl_pct"].mean())
                            med_pnl = float(tdf["pnl_pct"].median())
                            std_pnl = float(tdf["pnl_pct"].std(ddof=0))

                            avg_hold = float(tdf["holding_minutes"].mean())
                            med_hold = float(tdf["holding_minutes"].median())
                        else:
                            win_rate = avg_pnl = med_pnl = std_pnl = 0.0
                            avg_hold = med_hold = 0.0

                        monthly_trade_summaries.append(
                            {
                                "run_id": _os.path.basename(RUN_DIR_LOCAL),
                                "model_type": model_type,
                                "repetition": int(rep_idx),
                                "month_idx": month_idx,
                                "test_start": row.get("test_start"),
                                "test_end": row.get("test_end"),
                                "n_trades": n_trades,
                                "win_rate": win_rate,
                                "avg_pnl_pct": avg_pnl,
                                "median_pnl_pct": med_pnl,
                                "std_pnl_pct": std_pnl,
                                "avg_holding_minutes": avg_hold,
                                "median_holding_minutes": med_hold,
                            }
                        )

                    if monthly_trade_summaries:
                        monthly_df = _pd.DataFrame(monthly_trade_summaries)
                        monthly_path = _os.path.join(
                            final_dirs["csv"],
                            f"monthly_trade_summary_rep{rep_idx}.csv",
                        )
                        monthly_df.to_csv(
                            monthly_path,
                            index=False,
                            float_format="%.10f",
                        )
                        print(f"✅ Saved monthly trade summary for rep {rep_idx} → {monthly_path}")
                    else:
                        print("ℹ️ No trades recorded; skipping monthly trade summary.")

                except Exception as _e:
                    print(f"⚠️ Could not build monthly trade summary: {_e}")
            else:
                if self._is_debug():
                    print("ℹ️ Monthly trade summary disabled via SAVE_TRADES['monthly_summary_per_rep_csv'].")

            # -------------------------------------------------------------
            # Per-trade BH vs model comparison at entry/exit
            # -------------------------------------------------------------
            if SAVE_TRADES.get("trade_entry_exit_compare_csv", True):
                try:
                    import pandas as _pd
                    import os as _os
                    import math as _math

                    trade_compare_rows = []

                    # df_months_reset, all_dfs and trade_dfs share the same valid-month ordering
                    for idx, row in df_months_reset.iterrows():
                        if idx >= len(trade_dfs) or idx >= len(all_dfs):
                            continue

                        tdf = trade_dfs[idx]
                        mdf = all_dfs[idx]

                        if tdf is None or tdf.empty or mdf is None or mdf.empty:
                            continue
                        if not {"cstrategy_cont", "creturns_cont"} <= set(mdf.columns):
                            continue

                        eq_index = mdf.index

                        for _, tr in tdf.iterrows():
                            try:
                                entry_i = int(tr.get("entry_bar"))
                                exit_i = int(tr.get("exit_bar"))
                            except Exception:
                                continue

                            if entry_i < 0 or exit_i < 0:
                                continue
                            if entry_i >= len(mdf) or exit_i >= len(mdf):
                                continue

                            # ensure order
                            if exit_i < entry_i:
                                entry_i, exit_i = exit_i, entry_i

                            entry_time = eq_index[entry_i]
                            exit_time = eq_index[exit_i]

                            strat_start = float(mdf["cstrategy_cont"].iloc[entry_i])
                            strat_end   = float(mdf["cstrategy_cont"].iloc[exit_i])
                            bh_start    = float(mdf["creturns_cont"].iloc[entry_i])
                            bh_end      = float(mdf["creturns_cont"].iloc[exit_i])

                            def _rel_ret(end, start):
                                try:
                                    if start == 0.0 or not _math.isfinite(start) or not _math.isfinite(end):
                                        return float("nan")
                                except Exception:
                                    return float("nan")
                                return float(end / start - 1.0)

                            bh_ret = _rel_ret(bh_end, bh_start)
                            model_curve_ret = _rel_ret(strat_end, strat_start)

                            pnl_pct = float(tr.get("pnl_pct", float("nan")))
                            edge_vs_bh = float("nan")
                            if _math.isfinite(bh_ret) and _math.isfinite(pnl_pct):
                                edge_vs_bh = float(pnl_pct - bh_ret)

                            trade_compare_rows.append(
                                {
                                    "run_id": _os.path.basename(RUN_DIR_LOCAL),
                                    "model_type": model_type,
                                    "repetition": int(rep_idx),
                                    "month_idx": int(row.get("month_idx", idx + 1)),
                                    "trade_id": tr.get("trade_id"),
                                    "side": tr.get("side"),
                                    "side_sign": tr.get("side_sign"),
                                    "entry_bar": entry_i,
                                    "exit_bar": exit_i,
                                    "entry_time": entry_time,
                                    "exit_time": exit_time,
                                    "bars_held": tr.get("bars_held"),
                                    "holding_minutes": tr.get("holding_minutes"),
                                    "pnl_pct": pnl_pct,
                                    "model_curve_return_pct": model_curve_ret,
                                    "bh_return_pct": bh_ret,
                                    "edge_vs_bh_pct": edge_vs_bh,
                                }
                            )

                    if trade_compare_rows:
                        compare_df = _pd.DataFrame(trade_compare_rows)
                        compare_path = _os.path.join(
                            final_dirs["csv"],
                            f"trade_entry_exit_compare_rep{rep_idx}.csv",
                        )
                        compare_df.to_csv(
                            compare_path,
                            index=False,
                            float_format="%.10f",
                        )
                        print(
                            f"✅ Saved trade entry/exit BH comparison for rep {rep_idx} → "
                            f"{compare_path}"
                        )
                    else:
                        if self._is_debug():
                            print(
                                "ℹ️ No trades with usable equity curves for trade_entry_exit_compare; skipping CSV."
                            )

                except Exception as _e:
                    print(f"⚠️ Could not build trade entry/exit comparison CSV: {_e}")
            else:
                if self._is_debug():
                    print(
                        "ℹ️ Trade entry/exit comparison disabled via "
                        "SAVE_TRADES['trade_entry_exit_compare_csv']."
                    )

            

        # 2) Final (model-level) artifacts
        #    - feature_heatmap_final.png over all months of _this_ model
        #      HEAVY → gated by config + SKIP_PLOTS
        try:
            cfg_local = getattr(self, "features_config", {}) or {}
            do_feat_freq = bool(cfg_local.get("deploy_feature_freq", True))
            if do_feat_freq and not SKIP_PLOTS:
                save_feature_frequency_from_monthly_results(
                    df_months,
                    base_features=[],
                    out_png=os.path.join(final_dirs["graphs"], "feature_heatmap_final.png"),
                    top_k=30,
                    style="nature",
                    palette="okabe_ito_no_black",
                    exclude_prefixes=("returns_lag", "hour"),
                    collapse_raw_lags=True,
                    out_csv=os.path.join(final_dirs["csv"], "feature_frequency_monthly.csv"),
                )
        except Exception as _e:
            if self._is_debug():
                print(f"⚠️ Feature-frequency (model-level) heatmap skipped: {_e}")



        #    - csv over all months of this model
        _csv_exclude = [
            "features_used",
            "dqn_config", "cnn_config", "lstm_config", "transformer_config", "xgb_config", "rf_config", "logit_config",
        ]
        df_months.drop(columns=[c for c in _csv_exclude if c in df_months.columns], errors="ignore") \
            .to_csv(os.path.join(final_dirs["csv"], f"real_trading_simulation_{model_type}.csv"),
                    index=False, float_format="%.10f")

                # (optional) one consolidated TXT with per-month feature/config refs
        try:
            agg_path = os.path.join(final_dirs["csv"], "featuresconfigused_all.txt")
            with open(agg_path, "w", encoding="utf-8") as f:
                for idx, row in df_months.reset_index(drop=True).iterrows():
                    k = idx + 1
                    f.write(f"=== Month {k} ===\n")
                    feats = row.get("features_used", [])
                    f.write("features_used:\n")
                    if isinstance(feats, str):
                        f.write(feats + "\n")
                    else:
                        # Normalize feats: non-iterables (NaN, scalars) → empty list
                        try:
                            from collections.abc import Iterable
                            if not isinstance(feats, Iterable):
                                feats = []
                        except Exception:
                            feats = []
                        for ft in (feats or []):
                            f.write(str(ft) + "\n")
                    for cfg_key in (
                        "cnn_config", "lstm_config", "transformer_config",
                        "xgb_config", "rf_config", "logit_config", "dqn_config"
                    ):
                        if cfg_key in row:
                            f.write(f"\n{cfg_key}:\n{row[cfg_key]}\n")
                    f.write("\n")
        except Exception as _e:
            print(f"⚠️ Could not write aggregated features/config dump: {_e}")

        # Do NOT create a new timestamped folder here; reuse the one from earlier.
        # Just make sure it exists.
        os.makedirs(out_dir, exist_ok=True)

        # One-shot monthly feature-frequency heatmap (across all months in this repeat) -> All/
        # derive the repeat id directly from the data (written in main(): df_sim["rep"] = rep)
        
        rep = int(df_months['rep'].dropna().iloc[0]) if ('rep' in df_months.columns and df_months['rep'].notna().any()) else 1

        save_feature_frequency_from_monthly_results(
            df_months,
            base_features=[],
            out_png=os.path.join(buckets["All"]["heatmaps"], f"feature_frequency_monthly_rep{rep}.png"),
            top_k=30,
            style="nature",
            palette="okabe_ito_no_black",
            exclude_prefixes=("returns_lag", "hour"),
            collapse_raw_lags=True,
            out_csv=os.path.join(buckets["All"]["csv"], f"feature_frequency_monthly_rep{rep}.csv"),
        )
    
        # Per-bar comparison CSV/PNG — run ONCE (single model vs BH)
        try:
            if all_dfs:
                bar_concat = pd.concat(all_dfs).sort_index()
                bar_concat.columns = ["cstrategy_cont", "creturns_cont"]
                self.bar_concat = bar_concat

                bt_dict = {
                    "BH": bar_concat["creturns_cont"],
                    f"{model_type}_equity": bar_concat["cstrategy_cont"],
                }
                cfg_local = getattr(self, "features_config", {}) or {}
                light_output = bool(cfg_local.get("light_output", False))
                if not (SKIP_PLOTS or light_output):
                    save_model_bar_comparison_outputs(
                        bt_dict,
                        csv_dir=final_dirs["csv"],
                        png_dir=final_dirs["graphs"],
                        style="nature",
                        palette="okabe_ito_no_black",
                        bh_color="#666666",
                        n_time_parts=10,
                        dpi=300,
                        line_width=1.2,
                        annotate_coverage=False,
                    )

        except Exception as e:
            print(f"⚠️ Per-bar comparison (single model) failed: {e}")

        # --- Now that bar_concat exists, write month PNGs by slicing it ---
        try:
            cfg_local = getattr(self, "features_config", {}) or {}

            # Combine config flags with SAVE_* toggles
            do_csv = bool(SAVE_METRICS.get("per_month_metrics_csv", False))
            do_feat_txt = bool(SAVE_FEATURES.get("featuresconfig_txt", False))
            do_equity = (
                bool(SAVE_EQUITY.get("per_month_equity_png", False))
                and bool(cfg_local.get("save_monthly_equity_plots", False))
            )
            do_heatmap = (
                bool(SAVE_FEATURES.get("monthly_heatmap_png", False))
                and bool(cfg_local.get("save_monthly_feature_heatmaps", False))
            )

            # If nothing is enabled, skip the whole loop (but keep monthly stats / PBO below)
            if not (do_csv or do_feat_txt or do_equity or do_heatmap):
                if self._is_debug():
                    print(
                        "ℹ️ All SAVE_* per-month artifacts disabled; "
                        "skipping month-level file writes."
                    )
            else:
                for idx, row in df_months.iterrows():
                    month_ix = int(row.get("month_idx", idx + 1))
                    mdirs = month_dir_path(model_base_dir, month_ix)

                    # (a) CSV with only that month row (clean)
                    if do_csv:
                        _csv_exclude = {
                            "features_used",
                            "dqn_config", "cnn_config", "lstm_config", "transformer_config",
                            "xgb_config", "rf_config", "logit_config",
                        }
                        row_csv = {kk: vv for kk, vv in row.items() if kk not in _csv_exclude}

                        pd.DataFrame([row_csv]).to_csv(
                            os.path.join(mdirs["csv"], f"csv_month_{month_ix}.csv"),
                            index=False,
                            float_format="%.10f",
                        )

                    # (a2) Dump features/configs to a TXT file (only if enabled)
                    if do_feat_txt:
                        try:
                            dump_path = os.path.join(
                                mdirs["csv"],
                                f"featuresconfigused_{month_ix}.txt",
                            )
                            with open(dump_path, "w", encoding="utf-8") as f:
                                f.write(f"Month: {month_ix}\n")
                                f.write("\nfeatures_used:\n")
                                feats = row.get("features_used", [])
                                if isinstance(feats, str):
                                    f.write(feats + "\n")
                                else:
                                    # Normalize feats: non-iterables (NaN, scalars) → empty list
                                    try:
                                        from collections.abc import Iterable
                                        if not isinstance(feats, Iterable):
                                            feats = []
                                    except Exception:
                                        feats = []
                                    for ft in (feats or []):
                                        f.write(str(ft) + "\n")
                                for cfg_key in (
                                    "cnn_config", "lstm_config", "transformer_config",
                                    "xgb_config", "rf_config", "logit_config", "dqn_config",
                                ):
                                    if cfg_key in row:
                                        f.write(f"\n{cfg_key}:\n{row[cfg_key]}\n")
                        except Exception as _e:
                            print(
                                f"⚠️ Could not write features/config dump for month {month_ix}: {_e}"
                            )

                    # (b) slice per-bar equity for that month and plot (HEAVY → gated)
                    if do_equity:
                        ts, te = row.get("test_start"), row.get("test_end")
                        if pd.notna(ts) and pd.notna(te):
                            mdf = self.bar_concat.loc[ts:te].copy()
                        else:
                            mdf = self.bar_concat.copy()

                        if (
                            mdf is not None
                            and not mdf.empty
                            and all(
                                c in mdf.columns
                                for c in ("cstrategy_cont", "creturns_cont")
                            )
                        ):
                            save_month_equity_graph(
                                mdf,
                                out_csv=None,
                                out_png=os.path.join(
                                    mdirs["graphs"],
                                    f"monthly_equity_{month_ix}.png",
                                ),
                                label_model=disp_name,
                                title=f"{disp_name} — Month {month_ix}",
                                dpi=300,
                            )

                    # (c) Month-only feature heatmap (HEAVY → gated)
                    if do_heatmap:
                        save_feature_heatmap_for_single_month(
                            pd.DataFrame([row]),
                            out_png=os.path.join(
                                mdirs["heatmaps"],
                                f"feature_heatmap_{month_ix}.png",
                            ),
                        )

        except Exception as e:
            print(f"⚠️ Month artifact export failed: {e}")

        # Persist the monthly stats table for the whole simulation period
        try:
            if not df_months.empty:
                # Use the shared run directory so all models land in the same master CSV
                buckets = comparison_dirs(RUN_DIR_LOCAL)
                save_monthly_model_stats(df_months, buckets["All"]["csv"], filename="monthly_model_stats.csv")

        except Exception as e:
            print(f"⚠️ Saving monthly model stats CSV failed: {e}")
            
        
        # Optional post-hoc PBO/MCS analysis (read-only)
        try:
            cfg = getattr(self, "features_config", {}) or {}
            if bool(cfg.get("enable_pbo_mcs_analysis", False)):
                pbo_mcs_result = self.run_pbo_mcs_analysis()
                # keep for external inspection
                self._pbo_mcs_result = pbo_mcs_result
        except Exception as e:
            print(f"[PBO/MCS] Analysis failed: {e}")


        # Restore caller flags to prevent mode leakage across runs (CV vs real-sim)
        try:
            self._in_real_sim = _prev_real
            self._dbg_first_bars = _prev_dbg
        except Exception:
            pass

        # Restore mode flags (keeps CV-only behavior from leaking into subsequent runs).
        try:
            self._in_optuna_cv = bool(_prev_optuna_cv)
        except Exception:
            pass
        try:
            self._in_real_sim = bool(_prev_real)
        except Exception:
            pass
        try:
            self._dbg_first_bars = bool(_prev_dbg)
        except Exception:
            pass

        return df_months
    
    
    
# TRIAL_COUNTS_FULL = {
#     "logistic":       {"random": 15, "bayes": 45},
#     "svm":            {"random": 15, "bayes": 45},
#     "decision_tree":  {"random": 15, "bayes": 45},
#     "random_forest":  {"random": 15, "bayes": 45},
#     "xgboost":        {"random": 15, "bayes": 45},
#     "lstm":           {"random": 20, "bayes": 50},
#     "cnn":            {"random": 0,  "bayes": 0},
#     "transformer":    {"random": 20, "bayes": 50},
#     "ensemble_adaptive_regime":     {"random": 20, "bayes": 50},
#     "ensemble_cnn_lstm_xgboost":    {"random": 20, "bayes": 50},
#     "dqn":            {"random": 0,  "bayes": 0},
# }


# Used for thesis
# TRIAL_COUNTS = {
#     "logistic":       {"random": 20, "bayes": 40},
#     "svm":            {"random": 20, "bayes": 40},
#     "decision_tree":  {"random": 20, "bayes": 40},
#     "random_forest":  {"random": 20, "bayes": 40},
#     "xgboost":        {"random": 20, "bayes": 40},
#     "lstm":           {"random": 20, "bayes": 40},
#     "cnn":            {"random": 20, "bayes": 40},
#     "transformer":    {"random": 20, "bayes": 40},
#     "ensemble_adaptive_regime":     {"random": 20, "bayes": 40},
#     "ensemble_cnn_lstm_xgboost":    {"random": 20, "bayes": 40},
#     "dqn":            {"random": 0,  "bayes": 0},
# }

# For quick system check.
TRIAL_COUNTS = {
    "logistic":       {"random": 3, "bayes": 3},
    "svm":            {"random": 3, "bayes": 3},
    "decision_tree":  {"random": 3, "bayes": 3},
    "random_forest":  {"random": 3, "bayes": 3},
    "xgboost":        {"random": 3, "bayes": 3},
    "lstm":           {"random": 3, "bayes": 3},
    "cnn":            {"random": 3, "bayes": 3},
    "transformer":    {"random": 3, "bayes": 3},
    "ensemble_adaptive_regime":     {"random": 3, "bayes": 3},
    "ensemble_cnn_lstm_xgboost":    {"random": 3, "bayes": 3},
    "dqn":            {"random": 3,  "bayes": 3},
}



def main() ->  None:
    """
    Run a 36-month real-trading simulation across selected models,
    repeat it N times with different seeds, and rank models across repeats.
    """
    
    # --- local memory cleanup helper (no external deps) ---
    import time, gc
    def _hard_free_local():
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.05)

    # ------------------------------------------------------------
    # FINAL EXPERIMENT: fixed per-run seeds (one full pipeline per seed)
    # This replaces the old "fresh os.urandom seed per repeat" behavior.
    # ------------------------------------------------------------

    
    # SEEDS = [11111, 22222, 33333]
    # SEEDS = [22222, 33333]
    
    SEEDS = [33333]
    REPEATS = 1

    N_REAL_MONTHS = 3 # 36
    END_DATE = "2025-12-01 00:00:00"   # end-of-Aug 2025, inclusive-ish for bar data

    # 1) Load feature configuration
    with open("configs/feature_config.json", "r") as f:
        features_config = json.load(f)
        
    # 🔒 Research-grade defaults for fairness:
    # - All models share the same strict day-1 calendar anchor.
    # - All models share the same NY session filtering rule.
    #   (Change "both" to "test_only" here if you want, but KEEP IT GLOBAL.)
    features_config.setdefault("enforce_day1_start", True)
    if features_config.get("session_filter_mode") is None:
        features_config["session_filter_mode"] = "both"

    # 1.1) EXPERIMENT LOCK: enforce CLASS_DEFAULTS over JSON for reproducibility.
    # JSON can still add extra keys, but it cannot override the experiment defaults.
    for _k, _v in CLASS_DEFAULTS["features"].items():
        features_config[_k] = deepcopy(_v) if isinstance(_v, (dict, list)) else _v

    # 1.5) Create one study run folder and make it global for this process
    RUN_DIR, _ = make_results_run_dir()
    # Softer Optuna RAM defaults: keep a small absolute floor, no percent-of-total gate.
    # The RAM guard is now *soft* (warns + GC) instead of pruning trials.
    need_default = float(os.environ.get("OPTUNA_MIN_FREE_GB", "0.35"))
    # Clamp to a maximum default; user can override via env var if they want stricter limits.
    if need_default > 0.35:
        os.environ["OPTUNA_MIN_FREE_GB"] = "0.35"

    # IMPORTANT: disable the percent-of-total rule by default.
    # This used to force need_gb ~= 3% of total RAM (≈0.76GB on your machine),
    # which caused "low RAM prune" even when ~0.75GB was free.
    os.environ.setdefault("OPTUNA_MIN_FREE_GB_PERCENT", "0.0")

    # Keep a mild relax/floor in case you later want stricter behaviour.
    os.environ.setdefault("OPTUNA_MIN_FREE_GB_RELAX", "0.6")
    os.environ.setdefault("OPTUNA_MIN_FREE_GB_FLOOR", "0.20")


    # If a CUDA GPU is present, also demand some VRAM headroom so we prune instead of OOMing
    try:
        import tensorflow as _tf
        if _tf.config.list_physical_devices("GPU"):
            # ask for at least ~1.0 GB free VRAM before starting a trial
            os.environ.setdefault("OPTUNA_MIN_FREE_VRAM_GB", "1.0")
    except Exception:
        pass

    # Keep BLAS inside each trial reasonable; if user already set
    # BLAS_THREADS_PER_TRIAL (e.g. via .env), respect it. Otherwise
    # fall back to (cores - 2) as a sensible default.
    _safe = max(1, (os.cpu_count() or 8) - 2)
    os.environ.setdefault("BLAS_THREADS_PER_TRIAL", str(_safe))

    # Keep BLAS/OpenMP stacks consistent with the chosen BLAS_THREADS_PER_TRIAL
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, os.environ["BLAS_THREADS_PER_TRIAL"])

    init_study_tree(RUN_DIR)

    os.environ["RESULTS_RUN_DIR"] = RUN_DIR  # ensures internal calls reuse the same folder
    
    # Persistent joblib memmap root for this run (prevents racey temp deletion)
    JOBLIB_ROOT = os.path.abspath("./joblib_tmp")
    os.makedirs(JOBLIB_ROOT, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = JOBLIB_ROOT
    
    log_print(f"🗂️ JOBLIB_TEMP_FOLDER={JOBLIB_ROOT}", level="COMPACT")
    log_print(f"\n📁 Study folder: {RUN_DIR}", level="COMPACT")


    # 2) Choose models (edit this list as you like)
    # Warning free, Hardware Performant and Optimized. 
    MODEL_LIST = [
        # Linear / margin baselines (shallow, low-capacity)
        "logistic", 
        # "svm", 

        # # Tree-based classical ML (nonlinear tabular learners)
        # "random_forest", 
        # "decision_tree",  
        "xgboost",  
        
        # # Deep supervised sequence models (learn temporal structure directly)
        # "lstm", 
        # "cnn", 
        # "transformer", 
        
        # Reinforcement learning (policy/Q-learning)
        # "dqn",  
        
        # Hybrid ensembles (explicit fusion / regime routing)
        # "ensemble_cnn_lstm_xgboost",  
        # "ensemble_adaptive_regime", 
    ]
    
    print(f"\n🧪 Models for real trading simulation: {MODEL_LIST}")

    all_reps = []  # collect combined monthly results across all repeats
    eq_by_model: dict[str, list[pd.DataFrame]] = {}  # collect per-rep equity paths

    for rep in range(1, REPEATS + 1):

        # ------------------------------------------------------------------
        # Per-repetition run directory: <RUN_DIR>/repetition_1, repetition_2, ...
        # Everything for this repetition (per-model, per-month) is routed here.
        # ------------------------------------------------------------------
        rep_run_dir = os.path.join(RUN_DIR, f"repetition_{rep}")
        os.makedirs(rep_run_dir, exist_ok=True)

        # Let downstream helpers know where to write results for this repeat
        os.environ["RESULTS_RUN_DIR"] = rep_run_dir

        # ---- Fixed seed per repeat (research reproducibility) ----
        run_seed = int(SEEDS[rep - 1])
        set_global_determinism(seed=run_seed)
        
        # ---- Seed sanity trace (should match across same-seed reruns) ----
        try:
            import random as _py_random
            log_print(
                f"[SEED-SANITY] seed={run_seed} py={_py_random.random():.12f} np={np.random.randint(0, 2**31-1)}",
                level="COMPACT",
            )
        except Exception:
            pass

        # ✅ Fresh, isolated config for THIS repeat (no cross-repeat bleed)
        features_config_rep = deepcopy(features_config)
        features_config_rep["run_seed"] = int(run_seed)
        # Drop any sticky derived fields that prior runs may have injected
        features_config_rep.pop("eval_seed_sets", None)
        features_config_rep.pop("test_warmup_bars", None)

        log_print(
            f"\n========== 🔁 REPEAT {rep}/{REPEATS} — seed={run_seed} =========="
            f"\n📁 repetition_run_dir = {rep_run_dir}",
            level="COMPACT",
        )

        rep_results: dict[str, pd.DataFrame] = {}
        bt_by_model = {}

        # 3) Simulate N walk-forward months per model
        for model_type in MODEL_LIST:
            log_print(
                f"\n🚦 Running real trading simulation for model: {model_type}",
                level="COMPACT",
            )
            
            # --- CPU/GPU perf profile per category ---
            try:
                cat = model_category(model_type)
            except Exception:
                cat = "unknown"

            # One Optuna trial at a time for all families (n_jobs=1), but let that trial use many cores
            # Respect user-provided MLB_THREADS; otherwise fall back conservatively
            cpu_total  = max(1, (os.cpu_count() or 8))
            safe_cores = int(os.environ.get("MLB_THREADS", str(max(1, cpu_total - 2))))

            os.environ.setdefault("CV_JOBS", str(safe_cores))
            os.environ.setdefault("OPTUNA_N_JOBS", "1")  # keep trials sequential
            os.environ.setdefault("BLAS_THREADS_PER_TRIAL", str(safe_cores))

            for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(k, str(safe_cores))

            # keep these aligned if not set
            os.environ.setdefault("SKLEARN_JOBS", str(safe_cores))
            os.environ.setdefault("XGB_JOBS", str(safe_cores))
            os.environ.setdefault("RF_JOBS", str(safe_cores))



            # Also align common BLAS envs so numpy/scipy/OpenMP agree
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[k] = os.environ["BLAS_THREADS_PER_TRIAL"]

            print(f"⚙️ Perf [{cat}] → BLAS_THREADS_PER_TRIAL={os.environ['BLAS_THREADS_PER_TRIAL']} "
                f"| OPTUNA_N_JOBS={os.environ['OPTUNA_N_JOBS']}")

            
            try:
                trial_cfg = TRIAL_COUNTS.get(model_type, {"random": 5, "bayes": 5})
                base_config = {
                    "model_type": model_type,
                    "rep": rep,  # repetition index used for trade-log file names
                    "n_trials": trial_cfg["random"] + trial_cfg["bayes"],
                    "n_startup_trials": trial_cfg["random"],
                    # Route all artifacts for this repetition into its own subfolder:
                    # e.g. <DATE>/rep_1/logistic/..., <DATE>/rep_1/xgboost/...
                    "_run_dir": rep_run_dir,
                }

                # Apply low-RAM overrides on a *copy* so the original config remains intact
                model_features_cfg = deepcopy(features_config_rep)
                if os.environ.get("MLB_DISABLE_LOW_RAM_OVERRIDES", "0") != "1":
                    model_features_cfg = _apply_low_ram_overrides(model_features_cfg)

                # Instantiate a fresh backtester for this model
                bt = MLBacktester(
                    symbol="EURUSD",
                    start="2019-10-01 00:00:00",
                    end=END_DATE,
                    trading_costs=False,
                    features_config=model_features_cfg,
                    use_oof=("ensemble" in model_type),
                )

                try:
                    df_sim = bt.real_trading_simulation(
                        deepcopy(base_config),
                        models_to_test=[model_type],
                        months=N_REAL_MONTHS,
                    )
                    if df_sim is not None and not df_sim.empty:
                        df_sim = df_sim.copy()
                        df_sim["model_type"] = model_type
                        df_sim["rep"] = rep
                        df_sim["run_seed"] = run_seed
                        rep_results[model_type] = df_sim
                        # Keep only the per-bar equity curves for cross-model plots.
                        # Storing the full backtester keeps large frames/caches alive.
                        from types import SimpleNamespace
                        bc = getattr(bt, "bar_concat", None)
                        if bc is not None and not getattr(bc, "empty", True):
                            _cols = [c for c in ("cstrategy_cont", "creturns_cont") if c in bc.columns]
                            bc_small = bc[_cols].copy() if _cols else bc.iloc[:, :0].copy()
                        else:
                            bc_small = pd.DataFrame()
                        bt_by_model[model_type] = SimpleNamespace(bar_concat=bc_small)


                        # Collect full-horizon equity path for mean-over-reps (this model, this rep)
                        try:
                            bc = getattr(bt, "bar_concat", None)
                            if bc is not None and not getattr(bc, "empty", True):
                                eq_df = bc.copy()

                                # Ensure expected columns
                                cols = list(eq_df.columns)
                                if "cstrategy_cont" not in cols or "creturns_cont" not in cols:
                                    if len(cols) >= 2:
                                        eq_df = eq_df.copy()
                                        eq_df.columns = ["cstrategy_cont", "creturns_cont"][:len(cols)]

                                if "cstrategy_cont" in eq_df.columns:
                                    tmp = eq_df[["cstrategy_cont"]].copy()
                                    tmp["rep"] = int(rep)
                                    tmp["ts"] = tmp.index
                                    # Append to accumulator
                                    eq_by_model.setdefault(model_type, []).append(tmp)
                        except Exception as _e:
                            print(f"⚠️ Could not collect equity path for model {model_type}, rep {rep}: {_e}")
                    else:
                        print(f"⚠️ No rows returned for model {model_type}.")

                except Exception as e:
                    print(f"❌ Simulation failed for {model_type}: {e}")
                    traceback.print_exc()
                finally:
                    # Release model resources ASAP
                    try:
                        if hasattr(bt, "free") and callable(getattr(bt, "free")):
                            bt.free(release_data=True)

                    except Exception:
                        pass
            finally:
                # Hard cleanup between models
                try:
                    _hard_free_local()
                except Exception:
                    pass
                try:
                    import tensorflow as _tf
                    _tf.keras.backend.clear_session()
                except Exception:
                    pass
                import gc, time
                gc.collect()
                time.sleep(0.05)

        # 4) Per-repeat cross-model outputs + save combined monthly
        if rep_results:
            dfs = [df for df in rep_results.values() if not df.empty]
            combined_rep = pd.concat(dfs, ignore_index=True)
            # print("\n📊 Combined monthly results (this repeat):")
            # print(combined_rep.to_string(index=False))

            # === Write this-repeat combined monthly table into the repetition-local 'ALL/csv' ===
            rep_buckets = comparison_dirs(rep_run_dir)
            os.makedirs(rep_buckets["All"]["csv"], exist_ok=True)
            combined_rep_path = os.path.join(
                rep_buckets["All"]["csv"],
                f"combined_monthly_rep{rep}.csv",
            )
            combined_rep.to_csv(combined_rep_path, index=False)
            print(f"✅ Saved per-repeat monthly CSV → {combined_rep_path}")

            # --- Per-repeat ranking across models (this repeat only) ---
            try:
                rep_rank_df = build_model_ranking(combined_rep, min_months=1)
                if rep_rank_df is not None and not rep_rank_df.empty:
                    rep_rank_path = save_model_ranking_csv(
                        rep_rank_df,
                        rep_buckets["All"]["csv"],
                        filename=f"csv_ranking_rep{rep}.csv",
                    )
                    print(f"✅ Saved per-repeat ranking → {rep_rank_path}")
            except Exception as _e:
                print(f"[tables] ranking for repeat {rep} skipped: {_e}")

            # Generate cross-model artifacts only if 2+ models produced per-bar curve
            if len(bt_by_model) > 1:
                try:
                    bt_dict_for_compare = _build_bar_compare_dict(bt_by_model)
                except Exception as e:
                    log_print(
                        f"⚠️ Failed to build bar comparison dict: {e}",
                        level="DEBUG",
                    )
                    bt_dict_for_compare = {}

                if bt_dict_for_compare:
                    # Which models actually produced results this repeat?
                    avail_all = sorted([
                        k.replace("_equity", "")
                        for k in (bt_dict_for_compare or {}).keys()
                        if isinstance(k, str) and k != "BH" and k.endswith("_equity")
                    ])

                    if avail_all:
                        # Create bar-comparison + risk outputs in this repetition's ALL/ bucket.
                        if not SKIP_PLOTS:
                            # Per-bar cumulative equity (existing)
                            save_model_bar_comparison_outputs(
                                bt_dict_for_compare,
                                models=avail_all,
                                csv_dir=rep_buckets["All"]["csv"],   # CSV suppressed inside helper
                                png_dir=rep_buckets["All"]["graphs"],
                                style="nature",
                                palette="okabe_ito_no_black",
                                bh_color="#666666",
                                n_time_parts=10,
                                dpi=300,
                                line_width=1.2,
                                annotate_coverage=False,
                                save_csv=False,
                            )

                            # NEW: multi-model underwater / drawdown curve
                            save_model_underwater_outputs(
                                bt_dict_for_compare,
                                models=avail_all,
                                csv_dir=rep_buckets["All"]["csv"],
                                png_dir=rep_buckets["All"]["graphs"],
                                style="nature",
                                palette="okabe_ito_no_black",
                                bh_color="#666666",
                                n_time_parts=10,
                                dpi=300,
                                line_width=1.2,
                            )

                            # NEW: multi-model rolling Sharpe curve
                            save_model_rolling_performance_outputs(
                                bt_dict_for_compare,
                                models=avail_all,
                                csv_dir=rep_buckets["All"]["csv"],
                                png_dir=rep_buckets["All"]["graphs"],
                                style="nature",
                                palette="okabe_ito_no_black",
                                bh_color="#666666",
                                n_time_parts=10,
                                dpi=300,
                                line_width=1.2,
                                window_bars=None,  # auto ~1-month from frequency
                            )

                        # Rename outputs so each repetition has its own files
                        try:
                            # Bar compare
                            default_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                "model_bar_compare.csv",
                            )
                            default_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                "model_bar_compare_bars.png",
                            )
                            rep_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                f"bar_compare_models_rep{rep}.csv",
                            )
                            rep_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                f"bar_compare_models_rep{rep}.png",
                            )

                            # Underwater
                            under_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                "model_bar_underwater.csv",
                            )
                            under_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                "model_bar_underwater.png",
                            )
                            rep_under_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                f"underwater_models_rep{rep}.csv",
                            )
                            rep_under_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                f"underwater_models_rep{rep}.png",
                            )

                            # Rolling Sharpe
                            roll_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                "model_rolling_sharpe.csv",
                            )
                            roll_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                "model_rolling_sharpe.png",
                            )
                            rep_roll_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                f"rolling_sharpe_models_rep{rep}.csv",
                            )
                            rep_roll_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                f"rolling_sharpe_models_rep{rep}.png",
                            )

                            # Rename if the defaults exist
                            for src, dst in [
                                (default_csv, rep_csv),
                                (default_png, rep_png),
                                (under_csv, rep_under_csv),
                                (under_png, rep_under_png),
                                (roll_csv, rep_roll_csv),
                                (roll_png, rep_roll_png),
                            ]:
                                if os.path.exists(src):
                                    os.replace(src, dst)

                            print(
                                f"✅ Saved bar & risk comparison for rep {rep} → "
                                f"{rep_csv}, {rep_png}"
                            )
                        except Exception as _e:
                            print(
                                f"⚠️ Could not rename comparison outputs for rep {rep}: {_e}"
                            )
                    else:
                        print("⚠️ No models available for 'All' comparison.")
                else:
                    print("⚠️ No bt_dict_for_compare to plot.")
            else:
                print("ℹ️ Single-model (or no per-bar) in this repeat: skipping cross-model plots.")

            all_reps.append(combined_rep)
        else:
            print("\n❌ No valid simulation results produced in this repeat.")

        # --- end-of-repeat cleanup (keep RSS stable across repetitions) ---
        try:
            rep_results.clear()
        except Exception:
            pass
        try:
            bt_by_model.clear()
        except Exception:
            pass
        try:
            _hard_free_local()
        except Exception:
            pass


    if not all_reps:
        print("\n❌ No results in any repeat — nothing to rank.")
        return

    # -------------------------------------------------------------
    # 5) Final: aggregate across repeats (all_reps) and route tables
    # -------------------------------------------------------------
    combined_all = pd.concat(all_reps, ignore_index=True)

    # Global combined monthly table (all repeats, all models)
    #   <RUN_DIR>/combined_monthly_all.csv
    try:
        combined_all_path = os.path.join(RUN_DIR, "combined_monthly_all.csv")
        combined_all.to_csv(
            combined_all_path,
            index=False,
            float_format="%.10f",
        )
        print(f"✅ Saved combined monthly (all repeats) → {combined_all_path}")
    except Exception as e:
        print(f"[tables] combined_monthly_all.csv skipped: {e}")
        
    # Global equity curves by model (all months × repeats), consistent
    # with the ranking table. For each (month, model) we aggregate the
    # monthly 'cstrategy' factors across repeats via a geometric mean,
    # then compound those factors in chronological order starting from 1.0.
    try:
        from utilsNoWFO import (
            build_model_monthly_pivots,
            save_group_equity_curves,
        )  # local import is cheap and robust; comparison_dirs is global

        equity_pivot, returns_pivot, bh_equity = build_model_monthly_pivots(combined_all)
        if equity_pivot is not None and not equity_pivot.empty:
            # use the globally imported comparison_dirs
            global_buckets = comparison_dirs(RUN_DIR)
            out_png = os.path.join(
                global_buckets["All"]["graphs"],
                "model_equity_all_months.png",
            )

            save_group_equity_curves(
                equity_pivot,
                bh_equity,
                out_png=out_png,
                title="Equity by Model (all months × repeats)",
                include_bh=True,
            )
            print(f"✅ Saved global equity curves (all months × repeats) → {out_png}")
        else:
            print("⚠️ Global equity curves skipped: empty equity_pivot.")
    except Exception as e:
        print(f"⚠️ Global equity curve plot skipped: {e}")

    # -------------------------------------------------------------
    # Per-model monthly results
    #   • monthly_results_all_<model>.csv      → <RUN_DIR>/model_stats/
    #   • monthly_results_rep<k>_<model>.csv  → <RUN_DIR>/repetition_k/<Model>/csv/
    # -------------------------------------------------------------
    if not (
        SAVE_METRICS.get("monthly_results_all_csv", True)
        or SAVE_METRICS.get("monthly_results_per_rep_csv", True)
    ):
        print("[tables] Per-model monthly results saving is disabled via SAVE_METRICS.")
    else:
        try:
            import os as _os
            import pandas as _pd
            from utilsNoWFO import friendly_model_name

            # Root-level folder for "all repeats" per-model tables
            model_stats_dir = _os.path.join(RUN_DIR, "model_stats")
            _os.makedirs(model_stats_dir, exist_ok=True)

            model_col = combined_all.get("model_type")
            if model_col is None:
                model_col = combined_all.get("model")

            if model_col is None:
                print("[tables] No 'model_type'/'model' column in combined_all; skipping per-model monthly tables.")
            else:
                models_present = [
                    m for m in model_col.dropna().unique().tolist()
                    if isinstance(m, str) and m.strip()
                ]
                models_present = sorted(models_present)

                for m in models_present:
                    model_df = combined_all[model_col == m].copy()
                    if model_df.empty:
                        continue

                    # (a) All repeats, this model → <RUN_DIR>/model_stats/
                    if SAVE_METRICS.get("monthly_results_all_csv", True):
                        all_path = _os.path.join(
                            model_stats_dir,
                            f"monthly_results_all_{m}.csv",
                        )
                        model_df.to_csv(
                            all_path,
                            index=False,
                            float_format="%.10f",
                        )
                        if SAVE_METRICS.get("verbose", False):
                            print(f"    ↳ Saved all-reps monthly results for {m} → {all_path}")

                    # (b) Per repetition, this model → <RUN_DIR>/repetition_k/<Model>/csv/
                    if SAVE_METRICS.get("monthly_results_per_rep_csv", True) and "rep" in model_df.columns:
                        reps_present = (
                            model_df["rep"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        for rep_val in sorted(reps_present):
                            try:
                                rep_int = int(rep_val)
                            except Exception:
                                continue

                            rep_df = model_df[model_df["rep"] == rep_val].copy()
                            if rep_df.empty:
                                continue

                            rep_dir = _os.path.join(RUN_DIR, f"repetition_{rep_int}")
                            model_folder = friendly_model_name(m)
                            model_base_dir = _os.path.join(rep_dir, model_folder)
                            csv_dir = _os.path.join(model_base_dir, "csv")
                            _os.makedirs(csv_dir, exist_ok=True)

                            rep_path = _os.path.join(
                                csv_dir,
                                f"monthly_results_rep{rep_int}_{m}.csv",
                            )
                            rep_df.to_csv(
                                rep_path,
                                index=False,
                                float_format="%.10f",
                            )
                            if SAVE_METRICS.get("verbose", False):
                                print(f"    ↳ Saved rep-{rep_int} monthly results for {m} → {rep_path}")

                print("✅ Saved per-model monthly_results_all_* and monthly_results_rep*_*.csv in new layout.")
        except Exception as e:
            print(f"[tables] Per-model monthly results skipped due to error: {e}")

    # -------------------------------------------------------------
    # NOTE: We intentionally do NOT save mean equity curves over
    #       repetitions anymore (full_equity_mean_over_reps_*).
    #       Only per-rep equity plots, monthly tables, and global
    #       feature heatmaps are produced at run level.
    # -------------------------------------------------------------

    # 6) Global feature heatmaps across ALL repetitions
    #    • one "all models" heatmap
    #    • one per-model heatmap
    #    Location: <RUN_DIR>/heatmaps/
    if not SKIP_PLOTS:
        try:
            global_heat_dir = os.path.join(RUN_DIR, "heatmaps")
            os.makedirs(global_heat_dir, exist_ok=True)

            # (a) All models together
            try:
                save_feature_frequency_from_monthly_results(
                    combined_all,
                    base_features=[],
                    out_png=os.path.join(
                        global_heat_dir,
                        "feature_heatmap_all_models_all_reps.png",
                    ),
                    top_k=30,
                    top_percent=1.0,
                    weight_by_score=False,
                    minimize_objective=False,
                    style="nature",
                    palette="okabe_ito_no_black",
                    exclude_prefixes=("returns_lag", "hour"),
                    collapse_raw_lags=True,
                    out_csv=os.path.join(
                        global_heat_dir,
                        "feature_heatmap_all_models_all_reps.csv",
                    ),
                )
                print(
                    f"✅ Saved global feature heatmap (all models, all reps) → {global_heat_dir}"
                )
            except Exception as _e:
                print(f"⚠️ Global all-model heatmap skipped: {_e}")

            # (b) Per-model heatmaps
            model_col = combined_all.get("model_type")
            if model_col is None:
                model_col = combined_all.get("model")

            if model_col is not None:
                models_present = sorted(
                    [
                        m for m in model_col.dropna().unique().tolist()
                        if isinstance(m, str) and m.strip()
                    ]
                )
                for m in models_present:
                    df_m = combined_all[model_col == m]
                    if df_m.empty:
                        continue
                    try:
                        save_feature_frequency_from_monthly_results(
                            df_m,
                            base_features=[],
                            out_png=os.path.join(
                                global_heat_dir,
                                f"feature_heatmap_{m}_all_reps.png",
                            ),
                            top_k=30,
                            top_percent=1.0,
                            weight_by_score=False,
                            minimize_objective=False,
                            style="nature",
                            palette="okabe_ito_no_black",
                            exclude_prefixes=("returns_lag", "hour"),
                            collapse_raw_lags=True,
                            out_csv=os.path.join(
                                global_heat_dir,
                                f"feature_heatmap_{m}_all_reps.csv",
                            ),
                        )
                        print(
                            f"✅ Saved global feature heatmap for model={m} → {global_heat_dir}"
                        )
                    except Exception as _e:
                        print(
                            f"⚠️ Global per-model heatmap for {m} skipped: {_e}"
                        )
            else:
                print(
                    "ℹ️ No model_type/model column; skipping global per-model heatmaps."
                )

        except Exception as e:
            print(f"⚠️ Global heatmap generation skipped: {e}")


    # Build & save definitive model ranking across X months (and repeats)
    try:
        rank_df = build_model_ranking(combined_all, min_months=1)

        # Save CSV at run root (next to repetition_1/, repetition_2/, model_stats/, etc.)
        try:
            global_rank_path = os.path.join(RUN_DIR, "csv_ranking_FINAL.csv")
            rank_df.to_csv(
                global_rank_path,
                index=False,
                float_format="%.10f",
            )
            print(f"✅ Saved global ranking across repeats → {global_rank_path}")
        except Exception as _e:
            print(f"[tables] ranking CSV skipped: {_e}")

        # Pretty ASCII table, consistent with other outputs
        cols = [
            "rank","model","months","trades","active","SR","PSR","DSR","Calmar",
            "AnnRet","FinalEq","DA","Prec","F1","Profit/Hit","LabelThr","EffConf","lags"
        ]
        rows = []
        if rank_df is not None and not rank_df.empty:
            for _, r in rank_df.iterrows():
                rows.append([
                    int(r.get("rank", 0)),
                    str(r.get("model", "")),
                    int(r.get("months", 0)) if pd.notna(r.get("months", None)) else "—",
                    int(r.get("trades", 0)) if pd.notna(r.get("trades", None)) else "—",
                    (f"{float(r.get('active', float('nan'))):.5f}"     if pd.notna(r.get("active", None)) else "—"),
                    (f"{float(r.get('SR', float('nan'))):.3f}"         if pd.notna(r.get("SR", None)) else "—"),
                    (f"{float(r.get('PSR', float('nan'))):.3f}"        if pd.notna(r.get("PSR", None)) else "—"),
                    (f"{float(r.get('DSR', float('nan'))):.3f}"        if pd.notna(r.get("DSR", None)) else "—"),
                    (f"{float(r.get('Calmar', float('nan'))):.3f}"     if pd.notna(r.get("Calmar", None)) else "—"),
                    (f"{float(r.get('AnnRet', float('nan'))):.5f}"     if pd.notna(r.get("AnnRet", None)) else "—"),
                    (f"{float(r.get('FinalEq', float('nan'))):.5f}"    if pd.notna(r.get("FinalEq", None)) else "—"),
                    (f"{float(r.get('DA', float('nan'))):.5f}"         if pd.notna(r.get("DA", None)) else "—"),
                    (f"{float(r.get('Prec', float('nan'))):.5f}"       if pd.notna(r.get("Prec", None)) else "—"),
                    (f"{float(r.get('F1', float('nan'))):.5f}"         if pd.notna(r.get("F1", None)) else "—"),
                    (f"{float(r.get('Profit/Hit', float('nan'))):.6f}" if pd.notna(r.get("Profit/Hit", None)) else "—"),
                    (f"{float(r.get('LabelThr', float('nan'))):.6f}"   if pd.notna(r.get("LabelThr", None)) else "—"),
                    (f"{float(r.get('EffConf', float('nan'))):.3f}"    if pd.notna(r.get("EffConf", None)) else "—"),
                    (int(r.get("lags", 0)) if pd.notna(r.get("lags", None)) else "—"),
                ])
        _fmt_table_ascii(
            cols,
            rows,
            title="🏁 Model Ranking (all months × repeats; equity compounded monthly)",
        )

    except Exception as e:
        print(f"[tables] ranking build skipped: {e}")


    # ---------------- Local toggles for this section (no global `config` required) ----------
    PRINT_EVAL_TABLES      = True    # master switch: prints ONLY the complex per-run table
    EVAL_TABLE_MAX_ROWS    = 40      # max rows for the per-run table


    def _safe_first(group, cols, default=None):
        for c in cols:
            if c in group and not group[c].dropna().empty:
                return group[c].dropna().iloc[0]
        return default

    def _safe_mean(group, cols):
        import numpy as _np
        for c in cols:
            if c in group and group[c].notna().any():
                v = _np.asarray(group[c].values, dtype=float)
                v = v[_np.isfinite(v)]
                if v.size:
                    return float(_np.mean(v))
        return float("nan")

    def _safe_sum(group, cols):
        import numpy as _np
        for c in cols:
            if c in group and group[c].notna().any():
                v = _np.asarray(group[c].values, dtype=float)
                v = v[_np.isfinite(v)]
                if v.size:
                    return float(_np.sum(v))
        return float("nan")

    # ---- Ranking (simple, used to compute per_run_summary only) ----------------------------
    def _rank_from_combined_simple(df_all: pd.DataFrame):
        import numpy as _np
        df = df_all.copy()
        if "strategy_return" not in df.columns and "cstrategy" in df.columns:
            df["strategy_return"] = df["cstrategy"] - 1.0
        df = df.sort_values(["model_type", "rep", "test_end"])

        def _agg_one(g: pd.DataFrame) -> pd.Series:
            n = g.shape[0]
            if "equity_strategy" in g.columns and not g["equity_strategy"].dropna().empty:
                final_eq = float(g["equity_strategy"].dropna().iloc[-1])
                eq_series = g["equity_strategy"].astype(float).values
            else:
                eq_series = _np.cumprod(1.0 + g["strategy_return"].astype(float).values)
                final_eq = float(eq_series[-1]) if n else _np.nan
            mret = g["strategy_return"].astype(float).values
            ann_ret = (final_eq ** (12.0 / n) - 1.0) if n > 0 else _np.nan
            m_active = mret[_np.abs(mret) > 1e-12]
            if m_active.size < 3:
                ann_vol = _np.nan; sharpe = _np.nan
            else:
                vol_m = float(_np.std(m_active, ddof=1))
                ann_vol = float(vol_m * _np.sqrt(12.0))
                sharpe = float(ann_ret / ann_vol) if _np.isfinite(ann_vol) and ann_vol > 0 else _np.nan
            dd = float(_np.min(eq_series / _np.maximum.accumulate(eq_series) - 1.0)) if n else _np.nan
            calmar = (ann_ret / abs(dd)) if (isinstance(dd, float) and dd < 0) else _np.nan
            win_rate = float((mret > 0).mean()) if n else _np.nan
            return pd.Series({
                "months": n, "final_equity": final_eq, "ann_return": ann_ret,
                "ann_vol": ann_vol, "sharpe": sharpe, "calmar": calmar, "win_rate": win_rate
            })

        per_run = (
            df.groupby(["model_type","rep","run_seed"], dropna=False)
            .apply(_agg_one, include_groups=False)
            .reset_index()
        )

        # keep ranking_df available for CSV saving (not printed)
        ranking = (
            per_run.groupby("model_type", as_index=False)
                .agg(runs=("rep","nunique"),
                     months_mean=("months","mean"),
                     sharpe_mean=("sharpe","mean"),
                     calmar_mean=("calmar","mean"),
                     ann_return_mean=("ann_return","mean"),
                     final_equity_median=("final_equity","median"))
        )
        ranking["composite"] = (
            0.50*ranking["sharpe_mean"] + 0.30*ranking["calmar_mean"] + 0.20*ranking["ann_return_mean"]
        )
        ranking = ranking.sort_values(["composite","final_equity_median"], ascending=[False, False]).reset_index(drop=True)
        ranking["rank"] = _np.arange(1, len(ranking) + 1)
        return per_run, ranking

    per_run_summary, ranking_df = _rank_from_combined_simple(combined_all)

    # === Wide per-run detailed table (ONLY table we print) ===================================
    if PRINT_EVAL_TABLES:
        try:
            # Build a rich, one-row-per-(model,rep,seed) table
            gcols = [c for c in ["model_type","rep","run_seed"] if c in combined_all.columns]
            if not gcols:
                print("⚠️ Cannot render per-run table: missing model/rep/seed columns.")
            else:
                rows = []
                # stable order: sort by sharpe desc then model
                _order = (per_run_summary.sort_values(["sharpe","model_type","rep"], ascending=[False, True, True])
                                    if {"sharpe","model_type","rep"} <= set(per_run_summary.columns)
                                    else per_run_summary)
                seen = set()
                for _, rr in _order.iterrows():
                    key = (rr.get("model_type",""), int(rr.get("rep",0)),
                           int(rr.get("run_seed",0) if pd.notna(rr.get("run_seed",None)) else 0))
                    if key in seen:
                        continue
                    seen.add(key)
                    model, rep, seed = key
                    grp = combined_all[(combined_all.get("model_type")==model) &
                                       (combined_all.get("rep")==rep) &
                                       (combined_all.get("run_seed")==seed)]
                    if grp.empty:
                        continue

                    test_start = _safe_first(grp, ["test_start","val_start","start_ts"])
                    test_end   = _safe_first(grp.iloc[[-1]], ["test_end","val_end","end_ts"])
                    months     = int(rr.get("months", grp.shape[0]) if pd.notna(rr.get("months", None)) else grp.shape[0])

                    final_eq   = float(rr.get("final_equity", float("nan")))
                    ann_ret    = float(rr.get("ann_return", float("nan")))
                    ann_vol    = float(rr.get("ann_vol", float("nan")))
                    sharpe     = float(rr.get("sharpe", float("nan")))
                    calmar     = float(rr.get("calmar", float("nan")))
                    win_rate   = float(rr.get("win_rate", float("nan")))

                    trades     = _safe_sum(grp, ["trades","n_trades","positions_opened"])
                    active     = _safe_mean(grp, ["active_rate","coverage","coverage_rate"])
                    da         = _safe_mean(grp, ["directional_accuracy","hit_rate","win_rate"])
                    prec       = _safe_mean(grp, ["precision_macro","precision"])
                    f1_        = _safe_mean(grp, ["f1_macro","f1"])
                    pph        = _safe_mean(grp, ["profit_per_hit","avg_profit_per_trade"])

                    label_thr  = _safe_first(grp, ["label_threshold","thr","threshold"])

                    lags_used  = _safe_first(grp, ["lags","lags_range"])
                    
                    # Effective confidence threshold (per-run summary):
                    # Use median across months. Prefer canonical column name(s).
                    eff_conf = float("nan")
                    try:
                        if "confidence_threshold" in grp.columns:
                            eff_conf = float(pd.to_numeric(grp["confidence_threshold"], errors="coerce").median())
                        if (not np.isfinite(eff_conf)) and ("confidence_threshold_used" in grp.columns):
                            eff_conf = float(pd.to_numeric(grp["confidence_threshold_used"], errors="coerce").median())
                        # Backward compat for older schemas (best-effort)
                        if (not np.isfinite(eff_conf)):
                            for _c in ["eff_conf", "confidence_used", "conf_threshold_used", "conf_threshold"]:
                                if _c in grp.columns:
                                    eff_conf = float(pd.to_numeric(grp[_c], errors="coerce").median())
                                    if np.isfinite(eff_conf):
                                        break
                    except Exception:
                        pass

                    rows.append([
                        str(model),
                        int(rep),
                        (int(seed) if pd.notna(seed) else ""),
                        str(test_start).split("+")[0] if test_start is not None else "",
                        str(test_end).split("+")[0]   if test_end   is not None else "",
                        months,
                        (f"{trades:.0f}"  if pd.notna(trades)   else "—"),
                        (f"{active:.5f}"  if pd.notna(active)   else "—"),
                        (f"{sharpe:.3f}"  if pd.notna(sharpe)   else "—"),
                        (f"{calmar:.3f}"  if pd.notna(calmar)   else "—"),
                        (f"{ann_ret:.5f}" if pd.notna(ann_ret)  else "—"),
                        (f"{final_eq:.5f}"if pd.notna(final_eq) else "—"),
                        (f"{da:.5f}"      if pd.notna(da)       else "—"),
                        (f"{prec:.5f}"    if pd.notna(prec)     else "—"),
                        (f"{f1_:.5f}"     if pd.notna(f1_)      else "—"),
                        (f"{pph:.6f}"     if pd.notna(pph)      else "—"),
                        (f"{float(label_thr):.6f}" if pd.notna(label_thr) else "—"),
                        (f"{float(eff_conf):.3f}"  if pd.notna(eff_conf)  else "—"),
                        (int(lags_used) if pd.notna(lags_used) else "—"),
                    ])
                    if len(rows) >= EVAL_TABLE_MAX_ROWS:
                        break

                _fmt_table_ascii(
                    ["model","rep","seed","test_start","test_end","months","trades","active",
                     "SR","Calmar","AnnRet","FinalEq","DA","Prec","F1",
                     "Profit/Hit","LabelThr","EffConf","lags"],
                    rows,
                    title="📈 Real-trading per-run summary (detailed, precise, top by Sharpe)"
                )
        except Exception as _e:
            print(f"[tables] detailed per-run table skipped: {_e}")

if __name__ == "__main__":
    main()