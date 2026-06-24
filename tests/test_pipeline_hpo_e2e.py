"""Pipeline HPO + Real Trading — end-to-end integration tests.

Validates the full HPO-optimized backtesting pipeline for all 8 model types:
  - 10 Optuna trials per model
  - 36 months of HPO training window
  - 4 months of walk-forward real trading simulation

Runtime: ~30-60 minutes for the full suite (deep models dominate).
  - Shallow models (logistic, svm, rf, decision_tree, xgboost): ~2-5 min each
  - Deep models (cnn, lstm, transformer): ~5-15 min each

Usage:
    pytest tests/test_pipeline_hpo_e2e.py -v                          # all models
    pytest tests/test_pipeline_hpo_e2e.py -v -k "logistic"            # one model
    pytest tests/test_pipeline_hpo_e2e.py -v -k "shallow"             # shallow only
    pytest tests/test_pipeline_hpo_e2e.py -v -k "deep"                # deep only
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

tf_available = False
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    pass

from pipeline.backtester.composed import MLBacktester

# ── Constants ───────────────────────────────────────────────────────────

_HPO_TRIALS = 5
_HPO_MONTHS = 36
_REAL_TRADING_MONTHS = 2

_SHALLOW_MODELS = ["logistic", "svm", "random_forest", "decision_tree", "xgboost"]
_DEEP_MODELS = ["cnn", "lstm", "transformer"]

ALL_MODEL_TYPES = _SHALLOW_MODELS + _DEEP_MODELS


# ── Helpers ────────────────────────────────────────────────────────────

def _minimal_features_config():
    cfg = {
        "lags": 10,
        "lag_depth": 1,
        "roll_windows": [5],
        "include_raw_lags": True,
        "include_hour": True,
        "use_sma": True,
        "use_ema": False,
        "use_rsi": False,
        "use_macd": False,
        "use_bbands": False,
        "use_atr": True,
        "use_adx": False,
        "use_stoch": False,
        "use_mtf_ma": False,
        "feat_cache_enabled": False,
        "slice_cache_enabled": False,
    }
    return cfg


def _hpo_bt_config(model_type: str):
    return {
        "model_type": model_type,
        "rep": 1,
        "n_trials": _HPO_TRIALS,
        "n_startup_trials": 5,
        "_run_dir": tempfile.mkdtemp(),
    }


def _skip_if_no_db():
    from pipeline.data.data_sqlite import DataStore
    store = DataStore(os.path.join(_project_root, "data", "forex.db"))
    tfs = store.list_timeframes("EURUSD")
    if not {"M30", "H1", "H4"}.issubset(set(tfs)):
        pytest.skip("EURUSD timeframes (M30,H1,H4) not found in DB. Run download first.")


def _assert_smoke_result(df_sim, model_type: str):
    import numpy as np
    import pandas as pd

    assert isinstance(df_sim, pd.DataFrame), f"[{model_type}] result not a DataFrame"
    assert len(df_sim) >= 1, f"[{model_type}] empty result — no months completed"

    expected_cols = {"sharpe", "drawdown", "trades", "win_rate", "directional_accuracy"}
    missing = expected_cols - set(df_sim.columns)
    assert not missing, f"[{model_type}] missing columns: {missing}"

    for col in expected_cols:
        vals = df_sim[col].dropna()
        if len(vals) > 0:
            assert not np.isinf(vals.iloc[-1]), f"[{model_type}] inf in {col}: {vals.iloc[-1]}"
            assert np.isfinite(vals.iloc[-1]) or np.isnan(vals.iloc[-1]), f"[{model_type}] bad {col}: {vals.iloc[-1]}"

    sharpe_vals = df_sim["sharpe"].dropna()
    dd_vals = df_sim["drawdown"].dropna()

    if len(sharpe_vals) == 0 and len(dd_vals) == 0:
        return

    if len(sharpe_vals) > 0:
        assert not np.isinf(float(sharpe_vals.iloc[-1])), f"[{model_type}] inf Sharpe"
    if len(dd_vals) > 0:
        assert not np.isinf(float(dd_vals.iloc[-1])), f"[{model_type}] inf drawdown"


# ── Run helper — shared by all model tests ──────────────────────────────

def _run_hpo_e2e(model_type: str, db_path: str):
    os.environ.setdefault("CV_JOBS", "1")
    os.environ.setdefault("SMOKE_TEST", "0")

    bt = MLBacktester(
        symbol="EURUSD",
        start="2020-02-01",
        end="2023-03-03",
        trading_costs=False,
        features_config=_minimal_features_config(),
        db_path=db_path,
    )

    config = _hpo_bt_config(model_type)
    df_sim = bt.real_trading_simulation(
        config,
        models_to_test=[model_type],
        months=_REAL_TRADING_MONTHS,
    )
    _assert_smoke_result(df_sim, model_type)
    return df_sim


# ═════════════════════════════════════════════════════════════════════════
# SHALLOW MODELS
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestHpoE2EShallow:
    """Full HPO + real trading pipeline for all shallow models."""

    def test_hpo_logistic(self, seed_placeholder_db):
        _run_hpo_e2e("logistic", seed_placeholder_db)

    def test_hpo_svm(self, seed_placeholder_db):
        _run_hpo_e2e("svm", seed_placeholder_db)

    def test_hpo_random_forest(self, seed_placeholder_db):
        _run_hpo_e2e("random_forest", seed_placeholder_db)

    def test_hpo_decision_tree(self, seed_placeholder_db):
        _run_hpo_e2e("decision_tree", seed_placeholder_db)

    def test_hpo_xgboost(self, seed_placeholder_db):
        _run_hpo_e2e("xgboost", seed_placeholder_db)


# ═════════════════════════════════════════════════════════════════════════
# DEEP MODELS (skip if no TensorFlow)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.skipif(not tf_available, reason="TensorFlow not installed")
class TestHpoE2EDeep:
    """Full HPO + real trading pipeline for all deep models (TF required)."""

    def test_hpo_cnn(self, seed_placeholder_db):
        _run_hpo_e2e("cnn", seed_placeholder_db)

    def test_hpo_lstm(self, seed_placeholder_db):
        _run_hpo_e2e("lstm", seed_placeholder_db)

    def test_hpo_transformer(self, seed_placeholder_db):
        _run_hpo_e2e("transformer", seed_placeholder_db)
