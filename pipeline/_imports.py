"""
Shared imports for all pipeline modules.

Heavy dependencies (TF, XGBoost, Optuna, model builders) are loaded lazily
via _LazyModule proxy — the real import only fires on first attribute access.
This saves 15-30 seconds on startup when only running light tasks.
"""
from __future__ import annotations

# ── Standard Library (fast, always needed) ──
import gc as _gc
import glob
import hashlib
import importlib
import json
import math
from math import sqrt
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Tuple

# ── Fast third-party (always needed) ──
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

# scikit-learn (fast to import)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

# ── Logging ──
from logging_config import log_print

# ── Project-local (lightweight) ──
from utilsNoWFO import (
    set_global_determinism,
    TRAIN_TEST_MONTHS, N_METRICS, METRIC_NAMES,
    ensure_metric_tuple, validate_metrics_shape,
    make_results_run_dir, ensure_model_dirs, comparison_dirs, month_dir_path,
    save_model_bar_comparison_outputs,
    save_model_underwater_outputs,
    save_model_rolling_performance_outputs,
    save_group_equity_curves,
    build_trade_log_from_df,
    save_feature_frequency_from_monthly_results,
    RollingStandardizer,
    add_cyclic_hour_features,
    build_features_from_params,
    realized_vol, bipower_variation, fracdiff, triple_barrier_labels,
    attach_macro_features,
    calibrate_prefit_and_predict_proba, ConformalClassifier, sanitize_proba,
    print_feature_stats, print_conf_stats,
    fit_temperature_from_proba, apply_temperature_to_proba,
    fit_coverage_threshold_on_calibration, freeze_confidence_threshold,
    is_coverage_intent, enforce_target_coverage_policy, target_coverage_policy,
    _build_bar_compare_dict,
    compute_full_evaluation_metrics, combine_block_scores,
    enforce_day1_eval_anchor, first_tradable_test_bar,
    compute_required_test_warmup_bars,
    init_study_tree, model_category,
)

# ── Runtime knobs ──
from pipeline.runtime import SAFE_CORES, CPU_TOTAL


# ═══════════════════════════════════════════════════════════════════
# Lazy imports — heavy modules loaded only on first use
# ═══════════════════════════════════════════════════════════════════

class _LazyModule:
    """Proxy that defers `import X` until first attribute access."""
    _UNSET = object()  # sentinel

    def __init__(self, name: str, package: str | None = None):
        self.__name__ = name
        self.__package__ = package
        self._mod = self._UNSET

    def _resolve(self):
        if self._mod is self._UNSET:
            mod = importlib.import_module(self.__name__, self.__package__)
            self._mod = mod
            # Replace in caller modules so future lookups skip proxy
            globals()[self.__name__.split(".")[-1]] = mod
        return self._mod

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)

    def __repr__(self):
        if self._mod is self._UNSET:
            return f"<LazyModule '{self.__name__}' (not yet loaded)>"
        return repr(self._mod)

    def __dir__(self):
        return dir(self._resolve())


# Heavy ML frameworks — loaded on first use
tf = _LazyModule("tensorflow")
keras = _LazyModule("tensorflow.keras")
xgb = _LazyModule("xgboost")
optuna = _LazyModule("optuna")

# Lazy Keras aliases — resolved on first access
Callback = _LazyModule("tensorflow.keras.callbacks").Callback
mixed_precision = _LazyModule("tensorflow.keras.mixed_precision")

# XGBoost classifier — lazy
XGBClassifier = _LazyModule("xgboost").XGBClassifier

# Lazy model builders (they import TF internally)
build_cnn = _LazyModule("models.cnn").build_cnn
build_lstm = _LazyModule("models.lstm").build_lstm
build_transformer = _LazyModule("models.transformer").build_transformer

# Lazy RL imports
TradingEnv = _LazyModule("rl.environment").TradingEnv
DQNAgent = _LazyModule("rl.dqn_agent").DQNAgent
filter_dqn_config = _LazyModule("rl.dqn_agent").filter_dqn_config

# Lazy ensemble imports
EnsembleCNNLSTMXGBoost = _LazyModule("models.ensemble_cnn_lstm_xgboost").EnsembleCNNLSTMXGBoost
AdaptiveRegimeStrategy = _LazyModule("models.ensemble_adaptive_regime").AdaptiveRegimeStrategy


# ── Optional deps ──
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env", override=False)
except Exception:
    pass

try:
    import pyarrow  # noqa: F401
    _CSV_ENGINE = "pyarrow"
except Exception:
    _CSV_ENGINE = "c"

try:
    from threadpoolctl import threadpool_limits as _tp_limits
except Exception:
    _tp_limits = None

try:
    import matplotlib.pyplot as _plt
except Exception:
    _plt = None

try:
    import ta  # noqa: F401
except Exception:
    ta = None

# Keep aliases
CSV_ENGINE = _CSV_ENGINE
_os = os
_np = np