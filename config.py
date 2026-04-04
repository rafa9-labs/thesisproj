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
from typing import Optional, List, Dict, Any

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
# Compute / Threading
# ---------------------------------------------------------------------------
@dataclass
class ComputeConfig:
    """CPU/GPU thread budgets and parallelism settings."""
    cpu_total: int = 0
    safe_cores: int = 0
    force_cpu: bool = False
    log_level_tf: str = "3"  # TF_CPP_MIN_LOG_LEVEL: 0=all, 1=INFO, 2=WARNING, 3=ERROR
    gpu_allow_growth: bool = True

    def __post_init__(self):
        if self.cpu_total <= 0:
            self.cpu_total = os.cpu_count() or 8
        if self.safe_cores <= 0:
            _blas_env = _env("BLAS_THREADS_PER_TRIAL", "").strip()
            if _blas_env:
                self.safe_cores = max(1, int(_blas_env))
            else:
                _mlb_threads = _env_int("MLB_THREADS", 0)
                self.safe_cores = _mlb_threads if _mlb_threads > 0 else max(1, self.cpu_total - 2)


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

    # Thread budgets
    sc = settings.compute.safe_cores
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS", "SKLEARN_JOBS", "XGB_JOBS",
        "RF_JOBS", "CV_JOBS", "BLAS_THREADS_PER_TRIAL",
    ):
        os.environ.setdefault(var, str(sc))
    os.environ["BLAS_THREADS_PER_TRIAL"] = str(sc)