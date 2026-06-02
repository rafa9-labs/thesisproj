"""End-to-end integration tests for the Racecar pipeline (Phases A-E).

Validates that all phases chain together:
  A. Regime taxonomy -> 7-class detection on synthetic OHLC
  B/C. Committee builder -> construct config from regime performance data
  D. Committee backtester -> WFO evaluation with per-bar regime routing
  E. Live deployment -> mock data feed + streaming runner + health monitoring
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.committee_builder import (
    CommitteeBuilder,
    CommitteeConfig,
    RegimeAssignment,
)
from pipeline.committee_backtester import CommitteeBacktester
from pipeline.expert_profiler import FoldResult, RegimeModelMatrix
from pipeline.regime_utils import (
    detect_regimes,
    RegimeConfig,
    _REGIME_NAMES,
)

_RNG = np.random.default_rng(42)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _make_ohlc_with_regimes(n_bars: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2023-01-01", periods=n_bars, freq="1h", tz="UTC")
    base = 1.1000
    sec = max(100, n_bars // 5)
    price = np.zeros(n_bars)

    # Section 1: trend_up
    s, e = 0, sec
    trend = np.linspace(base, base + 0.02, e - s)
    price[s:e] = trend + rng.normal(0, 0.0005, e - s)

    # Section 2: mean_reverting
    s, e = sec, 2 * sec
    t_arr = np.arange(e - s)
    price[s:e] = base + 0.005 + 0.003 * np.sin(t_arr * 0.05) + rng.normal(0, 0.0003, e - s)

    # Section 3: trend_down
    s, e = 2 * sec, 3 * sec
    trend = np.linspace(base, base - 0.015, e - s)
    price[s:e] = trend + rng.normal(0, 0.0005, e - s)

    # Section 4: high_volatile
    s, e = 3 * sec, 4 * sec
    price[s:e] = base - 0.005 + rng.normal(0, 0.002, e - s)

    # Section 5: sideways
    s, e = 4 * sec, n_bars
    price[s:e] = base + rng.normal(0, 0.0002, e - s)

    df = pd.DataFrame({
        "mid_o": np.roll(price, 1),
        "mid_h": price + np.abs(rng.normal(0, 0.001, n_bars)),
        "mid_l": price - np.abs(rng.normal(0, 0.001, n_bars)),
        "mid_c": price,
        "spread": np.full(n_bars, 0.00015),
    }, index=dt)
    df["returns"] = df["mid_c"].pct_change().fillna(0.0)
    df.loc[df.index[0], "mid_o"] = price[0] - 0.0002
    return df


def _make_simple_config():
    return CommitteeConfig(
        regimes={
            "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
            "trend_down": RegimeAssignment(models=["logistic"], weights=[1.0]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


def _make_dummy_model(n_features: int = 15):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", random_state=42)
    X = _RNG.standard_normal((100, n_features)).astype(np.float32)
    y = _RNG.integers(0, 3, size=100).astype(np.int32)
    m.fit(X, y)
    return m


def _make_feature_names() -> list:
    return [
        "mid_c", "mid_h", "mid_l",
        "sma_20", "ema_20", "rv_48", "rolling_std_20",
        "rsi_14", "macd_diff",
        "bb_upper", "bb_lower", "bb_pct", "bbw",
        "atr_14", "adx_14",
    ]


# ════════════════════════════════════════════════════════════════════
# Phase A + D: Regime detection -> Committee backtest
# ════════════════════════════════════════════════════════════════════

class TestRegimeToBacktest:
    def test_regime_detection_produces_7_class_labels(self):
        df = _make_ohlc_with_regimes(3000)
        regime_ids = detect_regimes(df, config=RegimeConfig())
        assert len(regime_ids) == len(df)
        assert set(np.unique(regime_ids)).issubset(range(7))
        # At minimum, every bar gets a valid regime label

    def test_committee_backtest_runs_with_regime_data(self):
        df = _make_ohlc_with_regimes(4000)
        cfg = _make_simple_config()
        bt = CommitteeBacktester(cfg, confidence_threshold=0.5)
        result = bt.run_wfo(df, train_months=4, test_months=1)
        assert result is not None
        assert result.total_folds >= 1

    def test_committee_config_serialization_roundtrip(self):
        cfg = _make_simple_config()
        d = cfg.to_dict()
        assert d["version"] == 1
        assert "trend_up" in d["regimes"]
        # Rebuild from dict
        cfg2 = CommitteeConfig(
            regimes={
                rname: RegimeAssignment(
                    models=rdat["models"], weights=rdat["weights"])
                for rname, rdat in d["regimes"].items()
            },
            fallback=RegimeAssignment(
                models=d["fallback"]["models"],
                weights=d["fallback"]["weights"]) if d["fallback"] else None,
        )
        assert cfg2.regimes["trend_up"].models == ["logistic"]


# ════════════════════════════════════════════════════════════════════
# Phase B/C: Committee builder from regime performance data
# ════════════════════════════════════════════════════════════════════

class TestCommitteeBuilderIntegration:
    def test_builder_constructs_from_matrix(self):
        matrix = RegimeModelMatrix(
            regimes=["trend_up", "trend_down", "sideways"],
            models=["logistic", "random_forest"],
            sharpe_matrix=np.array([
                [1.2, -0.3, 0.1],
                [0.8, 0.5, -0.2],
            ], dtype=float),
            trade_matrix=np.array([
                [30, 10, 5],
                [25, 15, 8],
            ], dtype=int),
            hitrate_matrix=np.array([
                [0.6, 0.4, 0.5],
                [0.55, 0.5, 0.45],
            ], dtype=float),
            fold_counts=np.array([[3], [3]], dtype=int),
            raw_folds=[
                FoldResult(model="logistic", fold_idx=0, train_start=0, train_end=1,
                          test_start=2, test_end=3, sharpe=1.2, trades=30,
                          active_rate=0.6, win_rate=0.6, performance=0.5,
                          return_val=0.02, drawdown=0.01, geo_mean_ann=0.1,
                          directional_accuracy=0.55, f1_macro=0.5),
                FoldResult(model="logistic", fold_idx=1, train_start=0, train_end=1,
                          test_start=2, test_end=3, sharpe=-0.3, trades=10,
                          active_rate=0.3, win_rate=0.4, performance=-0.1,
                          return_val=-0.01, drawdown=0.02, geo_mean_ann=-0.05,
                          directional_accuracy=0.45, f1_macro=0.4),
                FoldResult(model="random_forest", fold_idx=0, train_start=0, train_end=1,
                          test_start=2, test_end=3, sharpe=0.8, trades=25,
                          active_rate=0.5, win_rate=0.55, performance=0.3,
                          return_val=0.01, drawdown=0.01, geo_mean_ann=0.05,
                          directional_accuracy=0.5, f1_macro=0.45),
                FoldResult(model="random_forest", fold_idx=1, train_start=0, train_end=1,
                          test_start=2, test_end=3, sharpe=-0.2, trades=8,
                          active_rate=0.2, win_rate=0.45, performance=-0.05,
                          return_val=-0.005, drawdown=0.01, geo_mean_ann=-0.02,
                          directional_accuracy=0.48, f1_macro=0.42),
            ],
        )

        builder = CommitteeBuilder()
        config = builder.build(matrix, constraints={"max_models_per_regime": 2,
                               "min_trades": 1})
        assert isinstance(config, CommitteeConfig)
        assert config.regimes, "Expected non-empty regimes in config"


# ════════════════════════════════════════════════════════════════════
# Phase D + E: Backtest -> Mock live session
# ════════════════════════════════════════════════════════════════════

class TestBacktestToLive:
    def test_mock_feed_generates_valid_bars(self):
        from trading.mock_live_data import MockLiveFeed, MockDataConfig

        cfg = MockDataConfig(n_bars=100, seed=42)
        feed = MockLiveFeed(cfg)
        bars = list(feed.generate_bars())
        assert len(bars) == 100
        for bar in bars:
            assert "mid_c" in bar
            assert bar["mid_h"] >= bar["mid_c"]
            assert bar["mid_l"] <= bar["mid_c"]

    def test_mock_feed_to_dataframe(self):
        from trading.mock_live_data import MockLiveFeed, MockDataConfig

        cfg = MockDataConfig(n_bars=200, seed=42)
        feed = MockLiveFeed(cfg)
        df = feed.to_dataframe()
        assert len(df) == 200
        assert "mid_c" in df.columns

    def test_live_runner_basic_signals(self):
        from trading.live_committee_runner import LiveCommitteeRunner
        from trading.mock_live_data import MockLiveFeed, MockDataConfig

        config = _make_simple_config()
        feat_names = _make_feature_names()
        models = {"logistic": _make_dummy_model(n_features=len(feat_names))}

        runner = LiveCommitteeRunner(
            config=config,
            models=models,
            feature_names=feat_names,
            confidence_threshold=0.4,
            lookback_bars=50,
        )
        runner.start()

        feed = MockLiveFeed(MockDataConfig(n_bars=100, seed=99))
        signals = []
        for bar in feed.generate_bars():
            signal = runner.process_bar(bar)
            signals.append(signal)
            # Signal is None until buffer fills (first lookback_bars)
            if signal is not None:
                assert signal.signal in (-1, 0, 1)

        runner.stop()
        # At least some bars should produce non-None signals
        non_none = [s for s in signals if s is not None]
        assert len(non_none) > 0

    def test_simulate_session_produces_metrics(self):
        from trading.live_committee_runner import LiveCommitteeRunner
        from trading.mock_live_data import (
            MockLiveFeed, MockDataConfig, simulate_session,
        )

        config = _make_simple_config()
        feat_names = _make_feature_names()
        models = {"logistic": _make_dummy_model(n_features=len(feat_names))}

        runner = LiveCommitteeRunner(
            config=config,
            models=models,
            feature_names=feat_names,
            confidence_threshold=0.4,
            lookback_bars=50,
        )
        runner.start()
        feed = MockLiveFeed(MockDataConfig(n_bars=100, seed=42))

        result = simulate_session(runner, feed, verbose=False)
        assert "summary" in result
        assert result["summary"]["bars_processed"] == 100

    def test_backtest_and_live_consistency(self):
        """Both backtest and mock live produce valid non-NaN results."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from trading.mock_live_data import (
            MockLiveFeed, MockDataConfig, simulate_session,
        )

        config = _make_simple_config()
        df = _make_ohlc_with_regimes(4000)

        # Phase D: committee backtest
        bt = CommitteeBacktester(config, confidence_threshold=0.5)
        bt_result = bt.run_wfo(df, train_months=4, test_months=1)
        assert bt_result is not None
        assert bt_result.total_folds >= 1
        assert np.isfinite(bt_result.avg_sharpe)

        # Phase E: mock live session
        feat_names = _make_feature_names()
        models = {"logistic": _make_dummy_model(n_features=len(feat_names))}
        runner = LiveCommitteeRunner(
            config=config, models=models, feature_names=feat_names,
            confidence_threshold=0.5, lookback_bars=50,
        )
        runner.start()
        feed = MockLiveFeed(MockDataConfig(n_bars=200, seed=42))
        live_result = simulate_session(runner, feed, verbose=False)

        assert live_result["summary"]["bars_processed"] == 200
        assert isinstance(live_result["returns"], list)


# ════════════════════════════════════════════════════════════════════
# Full pipeline: A -> C -> D -> E
# ════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_full_pipeline_chained(self):
        """All phases chain together without errors."""
        df = _make_ohlc_with_regimes(4000)

        # Phase A: regime detection (validates labeling works on real-ish data)
        regime_ids = detect_regimes(df)
        assert len(regime_ids) == len(df)

        # Build committee config covering detected regimes
        unique_regimes = set(map(int, regime_ids))
        detected_names = {
            _REGIME_NAMES.get(r, "sideways") for r in unique_regimes
        }
        regime_assignments = {}
        for rname in detected_names:
            regime_assignments[rname] = RegimeAssignment(
                models=["logistic"], weights=[1.0])
        config = CommitteeConfig(
            regimes=regime_assignments,
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )

        # Phase D: committee backtester
        bt = CommitteeBacktester(config, confidence_threshold=0.5)
        bt_result = bt.run_wfo(df, train_months=4, test_months=1)
        assert bt_result is not None
        assert bt_result.total_folds >= 1

        # Phase E: mock live session
        from trading.live_committee_runner import LiveCommitteeRunner
        from trading.mock_live_data import (
            MockLiveFeed, MockDataConfig, simulate_session,
        )

        feat_names = _make_feature_names()
        models = {"logistic": _make_dummy_model(n_features=len(feat_names))}
        runner = LiveCommitteeRunner(
            config=config, models=models, feature_names=feat_names,
            confidence_threshold=0.5, lookback_bars=50,
        )
        feed = MockLiveFeed(MockDataConfig(n_bars=200, seed=99))
        runner.start()
        live_result = simulate_session(runner, feed, verbose=False)

        assert live_result["summary"]["bars_processed"] == 200
        # Returns should be a list (possibly empty if no trades)
        assert isinstance(live_result["returns"], list)
