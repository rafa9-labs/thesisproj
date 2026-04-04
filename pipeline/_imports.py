"""
Shared imports for all pipeline modules.

Centralizes all common imports so individual modules don't need to
duplicate them.  Each pipeline module does:  from pipeline._imports import *
"""
from __future__ import annotations

# ── Standard Library ──
import gc as _gc
import glob
import hashlib
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

# ── Third-Party ──
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

# Keras
Callback = keras.callbacks.Callback
mixed_precision = keras.mixed_precision

# Optional deps
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

# ── Logging ──
from logging_config import log_print

# ── Project-local ──
from models.cnn import build_cnn
from models.lstm import build_lstm
from models.transformer import build_transformer
from rl.environment import TradingEnv
from rl.dqn_agent import DQNAgent, filter_dqn_config
from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
from models.ensemble_adaptive_regime import AdaptiveRegimeStrategy
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

# Keep aliases
CSV_ENGINE = _CSV_ENGINE
_os = os
_np = np