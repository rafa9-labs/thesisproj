"""Pipeline validation — comprehensive end-to-end + unit tests.

Validates the full backtesting pipeline with:
  - Logistic (shallow) + CNN (TensorFlow/deep)
  - All execution models (position sizing, stops, trailing, risk manager)
  - Feature toggle combinations including news
  - Edge cases

Runtime: ~5-6 minutes total.
  - Class 1 (execution loop): ~2s
  - Class 2 (features): ~10s
  - Class 3 (logistic E2E): ~30s
  - Class 4 (CNN E2E): ~3-4 min
  - Class 5 (edge cases): ~30s

Usage:
    pytest tests/test_pipeline_validation.py -v
    pytest tests/test_pipeline_validation.py -v -m "not slow"       # skip E2E
    pytest tests/test_pipeline_validation.py -v -k "TestExecution"  # fast only
"""

from __future__ import annotations

import os
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

tf_available = False
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    pass

from pipeline.backtester.execution_patches import PatchConfig, LoopResult, run_execution_loop
from pipeline.backtester.composed import MLBacktester
from pipeline.metrics_tuples import CLASS_DEFAULTS


# ── Shared helpers ───────────────────────────────────────────────────

def _make_synthetic_ohlc(n=500, freq="1h"):
    """Synthetic OHLC data for execution loop tests."""
    dates = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
    high = close + np.abs(np.random.randn(n)) * 0.0005
    low = close - np.abs(np.random.randn(n)) * 0.0005
    spread = np.full(n, 0.00015)
    atr = np.full(n, 0.005)
    df = pd.DataFrame({
        "price": close,
        "high": high,
        "low": low,
        "close": close,
        "spread": spread,
        "atr_14": atr,
        "returns": np.concatenate([[0.0], np.diff(np.log(close))]),
    }, index=dates)
    return df


def _make_inputs(n=500):
    """Synthetic inputs for run_execution_loop."""
    df = _make_synthetic_ohlc(n)
    np.random.seed(42)
    pred = np.random.choice([-1.0, 0.0, 1.0], size=n)
    rets = df["returns"].values.copy()
    bar_vol = np.full(n, 0.001)
    gap = np.zeros(n, dtype=bool)
    regime = np.zeros(n, dtype=int)
    return df, pred, rets, bar_vol, gap, regime


def _base_config():
    """Base PatchConfig with defaults."""
    return PatchConfig()


def _run_loop(cfg=None, n=500, **overrides):
    """Run execution loop with given config, return LoopResult."""
    df, pred, rets, bar_vol, gap, regime = _make_inputs(n)
    if cfg is None:
        cfg = _base_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return run_execution_loop(
        df, pred, rets, bar_vol, gap, regime,
        cfg=cfg, trading_costs=False, slippage_factor=0.0,
    )


def _smoke_features_config():
    """Minimal features config for smoke tests (1 trial)."""
    cfg = deepcopy(CLASS_DEFAULTS.get("features", {}))
    cfg.setdefault("lags", 10)
    cfg.setdefault("lag_depth", 1)
    cfg.setdefault("roll_windows", [5])
    cfg.setdefault("include_raw_lags", True)
    cfg.setdefault("include_hour", True)
    cfg.setdefault("use_sma", True)
    cfg.setdefault("use_ema", False)
    cfg.setdefault("use_rsi", False)
    cfg.setdefault("use_macd", False)
    cfg.setdefault("use_bbands", False)
    cfg.setdefault("use_atr", True)
    cfg.setdefault("use_adx", False)
    cfg.setdefault("use_stoch", False)
    cfg.setdefault("use_mtf_ma", False)
    cfg["feat_cache_enabled"] = False
    cfg["slice_cache_enabled"] = False
    return cfg


def _smoke_bt_config(model_type="logistic"):
    """Config for real_trading_simulation smoke runs."""
    return {
        "model_type": model_type,
        "rep": 1,
        "n_trials": 1,
        "n_startup_trials": 1,
        "_run_dir": tempfile.mkdtemp(),
    }


def _validate_df_months(df_months):
    """Assert df_months is a valid non-empty result DataFrame with sane metrics."""
    assert isinstance(df_months, pd.DataFrame)
    assert len(df_months) > 0
    required = ["sharpe", "drawdown", "trades", "win_rate", "directional_accuracy"]
    for col in required:
        assert col in df_months.columns, f"Missing column: {col}"
    sharpe = df_months["sharpe"].iloc[0]
    assert np.isfinite(sharpe) or sharpe == 0.0


def _skip_if_no_csv():
    csv_path = os.path.join(_project_root, "csv_data", "EURUSD_10_years_H1_OANDA.csv")
    if not os.path.exists(csv_path):
        pytest.skip("EURUSD H1 CSV not found")


# ═══════════════════════════════════════════════════════════════════════
# CLASS 1: Execution Loop (unit, fast)
# ═══════════════════════════════════════════════════════════════════════

class TestExecutionLoop:
    """Unit tests for run_execution_loop with all execution model combos."""

    def test_baseline_fixed_sizing(self):
        result = _run_loop()
        assert isinstance(result, LoopResult)
        assert len(result.pos_actual) > 0
        assert len(result.strat) > 0

    def test_fractional_sizing(self):
        result = _run_loop(cfg=PatchConfig(sizing_method="fixed_fractional"))
        assert result.sizing_method_used == "fixed_fractional"
        assert result.final_equity > 0

    def test_kelly_sizing(self):
        result = _run_loop(cfg=PatchConfig(sizing_method="kelly"))
        assert result.sizing_method_used == "kelly"
        assert result.final_equity > 0

    def test_atr_sizing(self):
        result = _run_loop(cfg=PatchConfig(sizing_method="atr"))
        assert result.sizing_method_used == "atr"

    def test_vol_target_sizing(self):
        result = _run_loop(cfg=PatchConfig(
            sizing_method="vol_target",
            use_vol_target=True,
            target_bar=0.001,
        ))
        assert result.sizing_method_used == "vol_target"

    def test_fixed_pip_stop_long(self):
        np.random.seed(42)
        result = _run_loop(cfg=PatchConfig(stop_method="fixed_pip"))
        assert result.stop_method_used == "fixed_pip"

    def test_fixed_pip_stop_short(self):
        np.random.seed(99)
        result = _run_loop(cfg=PatchConfig(stop_method="fixed_pip"))
        assert result.stop_method_used == "fixed_pip"

    def test_atr_stop(self):
        result = _run_loop(cfg=PatchConfig(stop_method="atr"))
        assert result.stop_method_used == "atr"

    def test_sigma_stop(self):
        result = _run_loop(cfg=PatchConfig(stop_method="sigma"))
        assert result.stop_method_used == "sigma"

    def test_breakeven_stop(self):
        result = _run_loop(cfg=PatchConfig(
            stop_method="fixed_pip",
            stop_use_be=True,
            stop_be_trigger_pips=15.0,
        ))
        assert result.stop_method_used == "fixed_pip"

    def test_partial_close(self):
        result = _run_loop(cfg=PatchConfig(
            stop_method="fixed_pip",
            stop_use_partial_close=True,
            stop_tp1_pips=20.0,
            stop_tp1_ratio=0.5,
        ))
        assert result.stop_method_used == "fixed_pip"

    def test_trailing_fixed_pips(self):
        result = _run_loop(cfg=PatchConfig(trailing_method="fixed_pips"))
        assert result.trailing_method_used == "fixed_pips"

    def test_trailing_atr(self):
        result = _run_loop(cfg=PatchConfig(trailing_method="atr"))
        assert result.trailing_method_used == "atr"

    def test_chandelier_exit(self):
        result = _run_loop(cfg=PatchConfig(trailing_method="chandelier"))
        assert result.trailing_method_used == "chandelier"

    def test_risk_dd_breaker(self):
        result = _run_loop(cfg=PatchConfig(
            risk_use_dd_breaker=True,
            risk_max_drawdown_pct=0.01,
        ))
        assert result.risk_manager_active

    def test_risk_consec_loss(self):
        result = _run_loop(cfg=PatchConfig(
            risk_use_consec_loss=True,
            risk_max_consecutive_losses=2,
        ))
        assert result.risk_manager_active

    def test_risk_daily_loss(self):
        result = _run_loop(cfg=PatchConfig(
            risk_use_daily_loss=True,
            risk_max_daily_loss_pct=0.001,
        ))
        assert result.risk_manager_active

    def test_combined_kelly_atr_stop_trailing(self):
        result = _run_loop(cfg=PatchConfig(
            sizing_method="kelly",
            stop_method="atr",
            trailing_method="fixed_pips",
            risk_use_dd_breaker=True,
            risk_max_drawdown_pct=0.15,
        ))
        assert result.sizing_method_used == "kelly"
        assert result.stop_method_used == "atr"
        assert result.trailing_method_used == "fixed_pips"
        assert result.risk_manager_active

    def test_kill_switch_pct(self):
        result = _run_loop(cfg=PatchConfig(
            use_kill=True,
            kill_mode="pct",
            kill_pct=0.005,
        ))
        assert isinstance(result.kills_triggered, int)


# ═══════════════════════════════════════════════════════════════════════
# CLASS 2: Feature Configurations (unit, fast)
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureConfigurations:
    """Tests MLBacktester.prepare_features() with various toggle combos."""

    @pytest.fixture(autouse=True)
    def _setup_bt(self):
        _skip_if_no_csv()
        cfg = _smoke_features_config()
        self.bt = MLBacktester(
            symbol="EURUSD",
            start="2024-01-01",
            end="2024-02-01",
            trading_costs=False,
            features_config=cfg,
        )
        self.df = self.bt.data.copy()

    def test_all_features_enabled(self):
        cfg = _smoke_features_config()
        for key in ["use_sma", "use_ema", "use_rsi", "use_macd", "use_bbands", "use_atr", "use_adx", "use_stoch"]:
            cfg[key] = True
        self.bt.features_config = cfg
        df_out, features = self.bt.prepare_features(self.df, lags=10)
        assert len(df_out) > 0
        assert len(features) > 0

    def test_minimal_features_sma_only(self):
        cfg = _smoke_features_config()
        for key in ["use_ema", "use_rsi", "use_macd", "use_bbands", "use_atr", "use_adx", "use_stoch", "use_mtf_ma"]:
            cfg[key] = False
        self.bt.features_config = cfg
        df_out, features = self.bt.prepare_features(self.df, lags=10)
        assert len(df_out) > 0
        assert len(features) > 0

    def test_no_technical_indicators(self):
        cfg = _smoke_features_config()
        for key in ["use_sma", "use_ema", "use_rsi", "use_macd", "use_bbands", "use_atr", "use_adx", "use_stoch", "use_mtf_ma"]:
            cfg[key] = False
        self.bt.features_config = cfg
        df_out, features = self.bt.prepare_features(self.df, lags=10)
        assert len(df_out) > 0
        assert len(features) > 0

    def test_news_features_enabled(self):
        cfg = _smoke_features_config()
        cfg["use_news"] = True
        cfg["news_event_flags"] = True
        from news.features import merge_news_features
        from news.scraper import NewsScraper
        events = NewsScraper.economic_calendar_events(2024)
        news_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-05", periods=5, freq="1D", tz="UTC"),
            "sentiment_score": [0.3, -0.1, 0.5, 0.0, 0.2],
            "sentiment_magnitude": [0.3, 0.1, 0.5, 0.0, 0.2],
            "news_volume": [2.0, 1.0, 3.0, 0.0, 1.0],
            "sent_pos": [0.5, 0.1, 0.7, 0.3, 0.4],
            "sent_neg": [0.5, 0.5, 0.1, 0.3, 0.2],
            "sent_neu": [0.3, 0.4, 0.2, 0.4, 0.4],
        })
        df_test = self.df.copy()
        df_test = merge_news_features(df_test, news_df, events=events, config={
            "use_news": True, "news_event_flags": True,
        })
        assert any("sentiment" in c for c in df_test.columns)
        assert any("event_flag" in c for c in df_test.columns)

    def test_news_features_disabled(self):
        cfg = _smoke_features_config()
        cfg["use_news"] = False
        self.bt.features_config = cfg
        df_out, features = self.bt.prepare_features(self.df, lags=10)
        assert len(df_out) > 0
        sentiment_cols = [c for c in df_out.columns if "sentiment" in c]
        assert len(sentiment_cols) == 0

    def test_indicator_states_toggle(self):
        cfg = _smoke_features_config()
        cfg["use_indicator_states"] = True
        self.bt.features_config = cfg
        df_out, features = self.bt.prepare_features(self.df, lags=10)
        assert len(df_out) > 0
        assert len(features) > 0


# ═══════════════════════════════════════════════════════════════════════
# CLASS 3: End-to-End Logistic (slow)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestEndToEndLogistic:
    """Full pipeline smoke tests with logistic regression."""

    def _run_logistic(self, features_overrides=None):
        _skip_if_no_csv()
        cfg = _smoke_features_config()
        if features_overrides:
            cfg.update(features_overrides)
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-07-01",
            trading_costs=False,
            features_config=cfg,
        )
        config = _smoke_bt_config("logistic")
        df_sim = bt.real_trading_simulation(config, models_to_test=["logistic"], months=1)
        return df_sim

    def test_baseline_logistic(self):
        df_sim = self._run_logistic()
        _validate_df_months(df_sim)

    def test_logistic_with_kelly_sizing(self):
        df_sim = self._run_logistic({
            "sizing_method": "kelly",
            "sizing_risk_fraction": 0.02,
        })
        _validate_df_months(df_sim)

    def test_logistic_with_fixed_stops(self):
        df_sim = self._run_logistic({
            "stop_method": "fixed_pip",
            "stop_sl_pips": 30.0,
            "stop_tp_pips": 60.0,
        })
        _validate_df_months(df_sim)

    def test_logistic_with_trailing(self):
        df_sim = self._run_logistic({
            "trailing_method": "fixed_pips",
            "trailing_pips": 25.0,
            "trailing_activation_pips": 10.0,
        })
        _validate_df_months(df_sim)

    def test_logistic_with_risk_manager(self):
        df_sim = self._run_logistic({
            "risk_use_dd_breaker": True,
            "risk_max_drawdown_pct": 0.10,
            "risk_use_consec_loss": True,
            "risk_max_consecutive_losses": 5,
        })
        _validate_df_months(df_sim)

    def test_logistic_with_news(self):
        df_sim = self._run_logistic({
            "use_news": True,
            "news_event_flags": True,
        })
        _validate_df_months(df_sim)


# ═══════════════════════════════════════════════════════════════════════
# CLASS 4: End-to-End CNN (slow, requires TensorFlow)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.skipif(not tf_available, reason="TensorFlow not installed")
class TestEndToEndCNN:
    """Full pipeline smoke tests with CNN (TensorFlow deep model)."""

    def _run_cnn(self, features_overrides=None):
        _skip_if_no_csv()
        cfg = _smoke_features_config()
        if features_overrides:
            cfg.update(features_overrides)
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-07-01",
            trading_costs=False,
            features_config=cfg,
        )
        config = _smoke_bt_config("cnn")
        df_sim = bt.real_trading_simulation(config, models_to_test=["cnn"], months=1)
        return df_sim

    def test_baseline_cnn(self):
        df_sim = self._run_cnn()
        _validate_df_months(df_sim)

    def test_cnn_with_atr_sizing(self):
        df_sim = self._run_cnn({
            "sizing_method": "atr",
            "sizing_atr_risk_pct": 0.02,
        })
        _validate_df_months(df_sim)

    def test_cnn_with_stops_and_trailing(self):
        df_sim = self._run_cnn({
            "stop_method": "fixed_pip",
            "stop_sl_pips": 30.0,
            "stop_tp_pips": 60.0,
            "trailing_method": "fixed_pips",
            "trailing_pips": 25.0,
        })
        _validate_df_months(df_sim)

    def test_cnn_with_risk_manager(self):
        df_sim = self._run_cnn({
            "risk_use_dd_breaker": True,
            "risk_max_drawdown_pct": 0.15,
            "risk_use_consec_loss": True,
            "risk_max_consecutive_losses": 5,
        })
        _validate_df_months(df_sim)


# ═══════════════════════════════════════════════════════════════════════
# CLASS 5: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case validation — no crashes on unusual inputs."""

    def test_flat_predictions_no_trades(self):
        df, _, rets, bar_vol, gap, regime = _make_inputs(200)
        pred = np.zeros(200)
        result = run_execution_loop(
            df, pred, rets, bar_vol, gap, regime,
            cfg=PatchConfig(), trading_costs=False, slippage_factor=0.0,
        )
        assert isinstance(result, LoopResult)
        assert np.all(result.pos_actual == 0.0)

    def test_single_feature_column(self):
        _skip_if_no_csv()
        cfg = _smoke_features_config()
        for key in ["use_sma", "use_ema", "use_rsi", "use_macd", "use_bbands",
                     "use_atr", "use_adx", "use_stoch", "use_mtf_ma"]:
            cfg[key] = False
        cfg["include_hour"] = False
        cfg["lags"] = 1
        cfg["include_raw_lags"] = True
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-07-01",
            trading_costs=False,
            features_config=cfg,
        )
        df_out, features = bt.prepare_features(bt.data.copy(), lags=1)
        assert len(df_out) > 0

    def test_empty_news_data(self):
        _skip_if_no_csv()
        cfg = _smoke_features_config()
        cfg["use_news"] = True
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-07-01",
            trading_costs=False,
            features_config=cfg,
        )
        bt._news_aggregated = pd.DataFrame()
        bt._news_economic_events = []
        df_out, features = bt.prepare_features(bt.data.copy(), lags=10)
        assert len(df_out) > 0

    def test_very_short_date_range(self):
        _skip_if_no_csv()
        cfg = _smoke_features_config()
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-06-03",
            trading_costs=False,
            features_config=cfg,
        )
        assert bt.data is not None
        assert len(bt.data) > 0
