# config.py — Centralized configuration for FX MLBacktester
#
# Single source of truth for all runtime settings.
# Reads from: .env file → environment variables → JSON configs → Python defaults.
#
# Usage:
#   from config import Settings, get_settings
#   cfg = get_settings()  # singleton
#   print(cfg.csv_base_path)

"""
Centralized configuration module.

Replaces the scattered os.environ.setdefault() calls across 5+ files
with a single, typed, testable configuration object.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal

PeriodUnit = Literal["months", "weeks", "days"]
WindowType = Literal["rolling", "expanding"]


def period_offset(count: int, unit: PeriodUnit = "months") -> "pd.DateOffset":
    """Create a pd.DateOffset for the given count and unit.

    Replaces all hardcoded ``pd.DateOffset(months=...)`` call sites.
    """
    import pandas as pd
    if unit == "months":
        return pd.DateOffset(months=count)
    if unit == "weeks":
        return pd.DateOffset(weeks=count)
    if unit == "days":
        return pd.DateOffset(days=count)
    raise ValueError(f"Unknown period_unit: {unit!r}")


def periods_between(a, b, unit: PeriodUnit = "months") -> int:
    """Number of complete periods between two timestamps (floor).

    Replaces the old ``months_between()`` which only handled months.
    """
    delta = b - a
    total_days = delta.days
    if unit == "months":
        return (b.year - a.year) * 12 + (b.month - a.month)
    if unit == "weeks":
        return total_days // 7
    if unit == "days":
        return total_days
    raise ValueError(f"Unknown period_unit: {unit!r}")


def to_period_freq(unit: PeriodUnit) -> str:
    """Map PeriodUnit to pandas period frequency string."""
    return {"months": "M", "weeks": "W", "days": "D"}[unit]


def convert_month_count_to_periods(months: int, unit: PeriodUnit) -> int:
    """Convert a month count to the equivalent count in the target unit.

    Approximate: 1 month ≈ 4 weeks ≈ 30 days.
    """
    if unit == "months":
        return months
    if unit == "weeks":
        return max(1, months * 4)
    if unit == "days":
        return max(1, months * 30)
    raise ValueError(f"Unknown period_unit: {unit!r}")

# ---------------------------------------------------------------------------
# Project root (one level up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# .env loading (safe, non-clobbering)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    """Read an environment variable, return as string."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    """Read an environment variable, return as int."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable, return as float."""
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable, return as bool."""
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _load_json(path: Path) -> dict:
    """Load a JSON config file, return empty dict on failure."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Paths to CSV data files."""
    csv_1h: Path = field(default_factory=lambda: PROJECT_ROOT / "csv_data" / "EURUSD_10_years_H1_OANDA.csv")
    csv_4h: Path = field(default_factory=lambda: PROJECT_ROOT / "csv_data" / "EURUSD_10_years_H4_OANDA.csv")
    csv_15min: Path = field(default_factory=lambda: PROJECT_ROOT / "csv_data" / "EURUSD_10_years_M15_OANDA.csv")
    csv_30min: Path = field(default_factory=lambda: PROJECT_ROOT / "csv_data" / "EURUSD_10_years_M30_OANDA.csv")
    base_timeframe: str = "M30"

    @property
    def base_csv(self) -> Path:
        """Return the CSV path for the base timeframe."""
        tf_map = {
            "M15": self.csv_15min,
            "M30": self.csv_30min,
            "H1": self.csv_1h,
            "H4": self.csv_4h,
        }
        return tf_map.get(self.base_timeframe, self.csv_30min)


# ---------------------------------------------------------------------------
# Timeframe Hierarchy — maps each base timeframe to its MTF timeframes + cadence
# ---------------------------------------------------------------------------
TIMEFRAME_HIERARCHY = {
    "M15": {"bars_per_day": 96, "annual_bars": 24192, "mtf_fast": "M30", "mtf_slow": "H1"},
    "M30": {"bars_per_day": 48, "annual_bars": 12096, "mtf_fast": "H1",  "mtf_slow": "H4"},
    "H1":  {"bars_per_day": 24, "annual_bars": 6048,  "mtf_fast": "H4",  "mtf_slow": "D1"},
    "H4":  {"bars_per_day": 6,  "annual_bars": 1512,  "mtf_fast": "D1",  "mtf_slow": "W1"},
}
DEFAULT_BASE_TIMEFRAME = "M30"


# ---------------------------------------------------------------------------
# Compute / Threading
# ---------------------------------------------------------------------------
@dataclass
class ComputeConfig:
    """CPU/GPU thread budgets and parallelism settings."""
    cpu_total: int = 0
    safe_cores: int = 0       # legacy, kept for backward compat
    blas_threads: int = 0     # per-operation BLAS parallelism (OMP/MKL)
    cv_n_jobs: int = 0        # fold-level parallelism (joblib CV)
    force_cpu: bool = False
    log_level_tf: str = "3"  # TF_CPP_MIN_LOG_LEVEL: 0=all, 1=INFO, 2=WARNING, 3=ERROR
    gpu_allow_growth: bool = True

    def __post_init__(self):
        if self.cpu_total <= 0:
            self.cpu_total = os.cpu_count() or 8

        _blas_env = _env("BLAS_THREADS_PER_TRIAL", "").strip()
        _cv_env = _env("CV_JOBS", "").strip()
        _mlb = _env_int("MLB_THREADS", 0)

        if _blas_env or _cv_env or _mlb:
            if _blas_env:
                self.blas_threads = max(1, int(_blas_env))
            if _cv_env:
                self.cv_n_jobs = max(1, int(_cv_env))
            if _mlb:
                self.blas_threads = self.blas_threads or _mlb
            if self.blas_threads <= 0:
                self.blas_threads = max(1, self.cpu_total - 2)
            if self.cv_n_jobs <= 0:
                self.cv_n_jobs = self.blas_threads
        else:
            try:
                from pipeline.resource_budget import get_resource_budget
                budget = get_resource_budget()
                self.blas_threads = budget.blas_threads
                self.cv_n_jobs = budget.cv_n_jobs
            except Exception:
                self.blas_threads = max(1, self.cpu_total - 2)
                self.cv_n_jobs = self.blas_threads

        if self.safe_cores <= 0:
            self.safe_cores = self.blas_threads


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LogConfig:
    """Logging behavior."""
    log_mode: str = "COMPACT"       # COMPACT, DEBUG, QUIET
    skip_plots: bool = False
    csv_engine: str = "pyarrow"     # or "c"


# ---------------------------------------------------------------------------
# HPO (Hyperparameter Optimization)
# ---------------------------------------------------------------------------
@dataclass
class HPOConfig:
    """Optuna / HPO settings."""
    config_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "hpo")
    ta_mode: str = "tuned"          # legacy, fixed, tuned
    disable_pruning: bool = False
    save_trial_feature_freq: bool = False


# ---------------------------------------------------------------------------
# Optuna Search Space (industry-standard ranges)
# ---------------------------------------------------------------------------
# Centralised hyperparameter ranges per model type.
# References:
#   - Logistic: Pedregosa et al. (scikit-learn), C ∈ [1e-2, 1e2]
#   - XGBoost:  Chen & Guestrin (2016), depth 3-8, lr [0.01, 0.3]
#   - SVM:      Hsu et al. (2010), C ∈ [1e-2, 1e2], γ ∈ [1e-4, 1e1]
#   - RF:       Breiman (2001), Biau (2012), fix bootstrap, tune depth/leaf
#   - LSTM:     Hochreiter & Schmidhuber (1997), 32-128 units, dropout 0.2-0.5
#   - CNN:      1D-CNN for time series, 32-96 filters, kernel 3-5
#   - Transformer: Vaswani et al. (2017), d_model ∈ {32,64,128}, heads ∈ {4,8}
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # -- Logistic Regression --
    "logistic": {
        "solver": "lbfgs",           # fixed: most stable for multinomial
        "penalty": "l2",             # fixed: lbfgs only supports l2
        "C": (1e-2, 1e2, True),      # (low, high, log_scale)
        "max_iter": 1000,            # fixed: convergence, not a tuning knob
        "tol": 1e-4,                 # fixed: convergence precision
        "class_weight": [None, "balanced"],
    },
    # -- XGBoost --
    "xgboost": {
        "n_estimators": (200, 800, 100),   # (low, high, step)
        "max_depth": (3, 8, 1),              # narrowed from 10
        "learning_rate": (0.01, 0.3, True), # (low, high, log_scale)
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        # Fixed to defaults: gamma=0, min_child_weight=1, lambda=1, alpha=0
        "warm_start": [True, False],
        "cold_restart_interval": [2, 3, 4, 6],
    },
    # -- SVM --
    "svm": {
        "C": (1e-2, 1e2, True),            # narrowed from [1e-3, 1e3]
        "gamma": [1e-4, 1e-3, 1e-2, 0.05],    # categorical, capped at 0.05 (0.5 caused O(n^3) hangs)
        "kernel": "rbf",                    # fixed: standard for FX
        "class_weight": "balanced",         # fixed: standard for imbalanced FX
    },
    # -- Random Forest --
    "random_forest": {
        "n_estimators": (300, 1000, 100),
        "max_depth": [8, 12, 16, 20],         # removed None (unconstrained -> perfect IS fit -> coverage rejection)
        "min_samples_leaf": (1, 10, 1),
        "max_features": ["sqrt", 0.33, 0.5],
        # Fixed: bootstrap=True, class_weight=None, n_jobs=-1
    },
    # -- Decision Tree --
    "decision_tree": {
        "max_depth": (3, 15, 1),             # (low, high, step)
        "min_samples_leaf": (1, 20, 1),
        "max_features": ["sqrt", "log2", None],
        "ccp_alpha": (0.0, 0.01),
        # Fixed: class_weight="balanced"
    },
    # -- LSTM --
    "lstm": {
        "units": [32, 64, 128],             # categorical, not int range
        "num_layers": [1, 2],
        "dropout_rate": (0.2, 0.5),
        "learning_rate": (1e-4, 5e-3, True), # narrowed from [1e-5, ...]
        # Fixed: dense_units=64, bidirectional=False, clipnorm=1.0,
        #        batch_size=256, use_seq_windows=False
        "warm_start": [True, False],
        "cold_restart_interval": [2, 3, 4, 6],
        "warm_start_lr_multiplier": (0.05, 0.3),
    },
    # -- CNN --
    "cnn": {
        "filters1": [32, 64, 96],           # conv layer 1
        "filters2": [32, 64, 96],           # conv layer 2
        "kernel_size": [3, 5],
        "learning_rate": (1e-4, 5e-3, True),
        # Fixed: dropout=0.3, dense_units=64, batch_size=256,
        #        use_seq_windows=False
        "warm_start": [True, False],
        "cold_restart_interval": [2, 3, 4, 6],
        "warm_start_lr_multiplier": (0.05, 0.3),
    },
    # -- Transformer --
    "transformer": {
        "d_model": [32, 64, 128],
        "num_heads": [4, 8],
        "dropout_rate": (0.1, 0.4),
        "learning_rate": (1e-4, 5e-3, True),
        # Fixed: num_blocks=1, ff_multiple=2, dense_units=128,
        #        pooling="cls", use_time2vec=False, batch_size=256
        "warm_start": [True, False],
        "cold_restart_interval": [2, 3, 4, 6],
        "warm_start_lr_multiplier": (0.05, 0.3),
    },
    # -- LightGBM (Microsoft histogram GBDT) --
    "lightgbm": {
        "n_estimators": (200, 800, 100),       # (low, high, step)
        "max_depth": (3, 8, 1),
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": (0.01, 0.3, True),    # (low, high, log_scale)
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_lambda": (0.0, 10.0),
        # Fixed: boosting_type=gbdt, min_child_samples=20
        "warm_start": [True, False],
        "cold_restart_interval": [2, 3, 4, 6],
    },
    # -- CatBoost (Yandex ordered boosting) --
    "catboost": {
        "iterations": (200, 800, 100),
        "depth": (3, 8, 1),
        "learning_rate": (0.01, 0.3, True),
        "subsample": (0.6, 1.0),
        "l2_leaf_reg": (1.0, 10.0),
        # Fixed: border_count=128, loss_function=MultiClass
        "warm_start": [True, False],
        "cold_restart_interval": [2, 3, 4, 6],
    },
    # -- GRU (Gated Recurrent Unit) --
    "gru": {
        "units": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout_rate": (0.2, 0.5),
        "learning_rate": (1e-4, 5e-3, True),
        # Fixed: dense_units=64, bidirectional=False, clipnorm=1.0, batch_size=256
    },
    # -- GRU-LSTM Hybrid --
    "gru_lstm": {
        "gru_units": [32, 64, 128],
        "lstm_units": [32, 64, 128],
        "dropout_rate": (0.2, 0.5),
        "learning_rate": (1e-4, 5e-3, True),
        # Fixed: dense_units=64, batch_size=256
    },
    # -- Stacking Ensemble (OOF meta-learner) --
    "stacking_ensemble": {
        "stack_cv": [3, 5, 8],
        "stack_method": ["auto", "predict_proba"],
        # sub-models selected by user in frontend, not tuned by HPO
    },
    # -- Meta Ensemble (Signal Committee) --
    "meta_ensemble": {
        "meta_combination_method": ["majority", "soft", "weighted"],
        # sub-models selected by user in frontend, not tuned by HPO
    },
    # -- Ensemble: Adaptive Regime --
    "ensemble_adaptive_regime": {
        "lstm_units": [32, 64, 128],
        "lstm_num_layers": (1, 3, 1),
        "lstm_dropout_rate": (0.1, 0.5),
        "lstm_learning_rate": (1e-4, 1e-2, True),
        "rf_n_estimators": (100, 500, 100),
        "rf_max_depth": [8, 12, 16, 20],
        "logit_C": (1e-2, 1e2, True),
        "logit_solver": ["lbfgs", "saga"],
        "adx_thresh": (15, 30, 1),
        "vol_thresh": (0.005, 0.02),
        "adx_thresh_q": (0.5, 0.9),
        # Fixed: adx_col, vol_col set at init; ensemble_method="hard"
    },
    # -- Ensemble: CNN-LSTM-XGBoost Fusion --
    "ensemble_cnn_lstm_xgboost": {
        "cnn_filters1": [32, 64, 96],
        "cnn_filters2": [32, 64, 96],
        "cnn_kernel_size": (2, 5, 1),
        "cnn_learning_rate": (1e-4, 5e-3, True),
        "lstm_units": [32, 64, 128],
        "lstm_learning_rate": (1e-4, 5e-3, True),
        "xgb_n_estimators": (200, 800, 100),
        "xgb_learning_rate": (0.005, 0.2, True),
        "xgb_max_depth": (3, 12, 1),
    },
}


# ---------------------------------------------------------------------------
# CV geometry search space — sampled per trial to let Optuna discover the
# optimal number of mini-blocks and validation fraction per model type.
# ---------------------------------------------------------------------------
CV_SEARCH_SPACE = {
    "cv_blocks": [3, 5, 7, 10],
    "cv_val_frac": [0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
}


# ---------------------------------------------------------------------------
# Pipeline protocol constants (single source of truth for .get() fallbacks)
# ---------------------------------------------------------------------------
# These values are referenced by backtester mixins via .get(key, DEFAULT).
# Kept in sync with CLASS_DEFAULTS in pipeline/metrics_tuples.py.
# ---------------------------------------------------------------------------
PIPELINE_CONSTANTS = {
    # Walk-forward window contract (P1: unified toggle)
    "window_type": "rolling",

    # Warm-start engine (P2: incremental learning)
    "warm_start": False,
    "cold_restart_interval": 3,
    "warm_start_lr_multiplier": 0.1,
    "warm_start_lr_floor": 1e-6,

    # Volatility & cost regime
    "vol_window_bars": 96,
    "high_vol_q": 0.85,
    "high_vol_conf_bump": 0.0,

    # Slippage normalization
    "slip_norm_bps": 0.25,
    "min_slip_norm_bps": 0.05,
    "gamma_slip_norm": 0.004,
    "beta_spread_norm": 0.0008,
    "alpha_vol_z": 0.004,

    # Spread / cost caps
    "vol_z_cap": 6.0,
    "spread_norm_cap": 5.0,
    "slip_ratio_cap": 6.0,

    # Coverage calibration
    "runtime_coverage_window": 48,
    "runtime_active_band_margin": 0.02,
    "runtime_conf_nudge": 0.015,
    "target_active_rate": 0.15,

    # Confidence thresholds
    "min_conf_thr": 0.33,
    "max_conf_thr": 0.90,
    "confidence_threshold": 0.50,

    # Base timeframe (one of M15, M30, H1, H4)
    "base_timeframe": "M30",

    # Execution cadence
    "bars_per_day": 48,
    "annual_bars": 12096,
    "spread_cap": 0.00040,
    "slippage_factor": 1.0,
    "impact_eta": 0.0,

    # Deep model training
    "deep_cv_batch_size": 256,
    "deep_cv_patience": 6,
    "deep_cv_max_epochs": 8,

    # News & sentiment
    "use_news": True,
    "news_sentiment_backend": "vader",
    "news_volume_windows": [6, 24],
    "news_event_flags": True,

    # Live trading news blending (post-prediction signal adjustment)
    "live_news_blend_enabled": False,
    "live_news_blend_weight": 0.10,       # 0.0 = model-only, 0.30 max recommended
    "live_news_cache_hours": 6.0,

    # LLM sentiment
    "llm_sentiment_enabled": False,    # off by default — only useful for live trading, not backtesting
    "llm_backend": "ollama",
    "llm_model": "llama3",
    "llm_api_key": "",
    "llm_weight": 0.7,
    "llm_batch_size": 10,
    "llm_cache_ttl_hours": 720,
    "llm_ollama_url": "http://localhost:11434",

    # Phase 1: BorutaSHAP feature selection
    "use_boruta_shap": True,
    "boruta_percentile": 90,
    "boruta_max_iter": 20,

    # Phase 6: UCB1 proposer
    "ucb1_exploration_factor": 2.0,
    "ucb1_llm_refresh_interval": 5,
}

# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    """DQN model paths and defaults."""
    model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "DQNSavedModels" / "dqn_model.keras")
    grid_config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "configs" / "dqn_grid_config.json")
    agent_config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "DQNSavedModels" / "dqn_model_config.json")
    grid_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.grid_config = _load_json(self.grid_config_path)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "configs" / "feature_config.json")
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.config = _load_json(self.config_path)


# ---------------------------------------------------------------------------
# Memory / RAM
# ---------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    """Memory management settings."""
    low_ram_trigger_gb: float = 1.25
    low_ram_force: bool = False
    ram_limit_gb: Optional[float] = None

    def __post_init__(self):
        self.low_ram_trigger_gb = _env_float("LOW_RAM_TRIGGER_GB", 1.25)
        self.low_ram_force = _env_bool("MLB_LOW_RAM", False)
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            default_limit = min(0.85 * total_gb, total_gb - 2)
            self.ram_limit_gb = _env_float("RAM_LIMIT_GB", default_limit)
        except ImportError:
            self.ram_limit_gb = None


# ---------------------------------------------------------------------------
# Execution Models (Sprint 2)
# ---------------------------------------------------------------------------
@dataclass
class ExecutionConfig:
    """Position sizing, stop-loss, trailing-stop, and risk-management settings."""
    sizing_method: str = "fixed"
    risk_fraction: float = 0.02
    kelly_fraction: float = 0.5
    kelly_min_trades: int = 10
    atr_risk_pct: float = 0.02
    atr_sl_mult: float = 2.0
    initial_equity: float = 10_000.0
    max_leverage: float = 5.0
    contract_size: float = 100_000.0

    def __post_init__(self):
        self.sizing_method = _env("SIZING_METHOD", self.sizing_method).strip().lower()
        self.risk_fraction = _env_float("SIZING_RISK_FRACTION", self.risk_fraction)
        self.kelly_fraction = _env_float("SIZING_KELLY_FRACTION", self.kelly_fraction)
        self.kelly_min_trades = _env_int("SIZING_KELLY_MIN_TRADES", self.kelly_min_trades)
        self.atr_risk_pct = _env_float("SIZING_ATR_RISK_PCT", self.atr_risk_pct)
        self.atr_sl_mult = _env_float("SIZING_ATR_SL_MULT", self.atr_sl_mult)
        self.initial_equity = _env_float("SIZING_INITIAL_EQUITY", self.initial_equity)
        self.max_leverage = _env_float("SIZING_MAX_LEVERAGE", self.max_leverage)


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    """Root configuration object — single source of truth."""
    data: DataConfig = field(default_factory=DataConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Experimental defaults (was CLASS_DEFAULTS in MLBacktesterNoWFO)
    experiment: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Populate logging from env
        self.logging.log_mode = _env("LOG_MODE", "COMPACT").upper()
        self.logging.skip_plots = _env_bool("SKIP_PLOTS", False) or _env_bool("MLB_SKIP_PLOTS", False)

        # Compute from env
        self.compute.force_cpu = _env_bool("TF_FORCE_CPU", False)
        self.compute.log_level_tf = _env("TF_CPP_MIN_LOG_LEVEL", "3")
        self.compute.gpu_allow_growth = _env_bool("TF_FORCE_GPU_ALLOW_GROWTH", True)

        # HPO from env
        self.hpo.config_dir = Path(_env("MLB_HPO_DIR", str(PROJECT_ROOT / "hpo")))
        self.hpo.ta_mode = _env("MLB_TA_MODE", "tuned").strip().lower()
        self.hpo.disable_pruning = _env_bool("MLB_DISABLE_OPTUNA_PRUNING", False)
        self.hpo.save_trial_feature_freq = _env_bool("SAVE_TRIAL_FEATURE_FREQ", False)

        # Data from env
        _csv_15 = _env("CSV_15MIN", "")
        _csv_30 = _env("CSV_30MIN", "")
        if _csv_15:
            self.data.csv_15min = Path(_csv_15)
        if _csv_30:
            self.data.csv_30min = Path(_csv_30)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton (lazy-initialized)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reset_settings() -> None:
    """Reset the singleton (useful for tests)."""
    global _settings_instance
    _settings_instance = None


# ---------------------------------------------------------------------------
# Apply environment-level side effects (once)
# ---------------------------------------------------------------------------
def apply_global_env(settings: Settings) -> None:
    """
    Apply process-wide environment variable settings from the config.

    This replaces the scattered os.environ.setdefault() calls.
    Called once at program startup, NOT at import time.
    """
    # TF logging
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", settings.compute.log_level_tf)
    if settings.compute.gpu_allow_growth:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Thread budgets: split BLAS (per-op) from CV_JOBS (per-fold)
    blas = settings.compute.blas_threads
    cv = settings.compute.cv_n_jobs

    blas_vars = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS", "SKLEARN_JOBS", "XGB_JOBS",
        "RF_JOBS",
    )
    for var in blas_vars:
        os.environ.setdefault(var, str(blas))
    os.environ.setdefault("CV_JOBS", str(cv))
    os.environ.setdefault("MLB_THREADS", str(blas))
    os.environ["BLAS_THREADS_PER_TRIAL"] = str(blas)


# ---------------------------------------------------------------------------
# Licensing constants
# ---------------------------------------------------------------------------
FREE_MODELS = frozenset({"logistic", "xgboost", "random_forest"})
FREE_EXECUTION_TYPES = frozenset({"fixed_lot"})
LICENSE_TRIAL_DAYS = 14
LICENSE_GRACE_PERIOD_DAYS = 7