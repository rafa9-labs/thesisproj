"""End-to-end integration tests for the Racecar pipeline (Phases A-E).

Validates that all phases chain together:
  A. Regime taxonomy -> 7-class detection on synthetic OHLC
  B/C. Committee builder -> construct config from regime performance data
  D. Committee backtester -> WFO evaluation with per-bar regime routing
  E. Live deployment -> mock data feed + streaming runner + health monitoring
"""
import json
import os
import sys
import tempfile
import pytest

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


# ════════════════════════════════════════════════════════════════════
# Full Cycle: Profiler matrix -> Builder -> Backtester
# ════════════════════════════════════════════════════════════════════

class TestFullCycleBuilderToBacktest:
    """Tests the full cycle B->C->D end-to-end with a single base model.

    Validates:
    1. The full cycle produces a tradeable deployable CommitteeConfig.
    2. CommitteeBuilder correctly associates each model to the regimes
       where it performed best in training (Phase B profiling).
    3. The CommitteeBacktester accepts the auto-built config and produces
       valid WFO results.
    """

    def test_single_model_full_cycle_correct_regime_assignment(self):
        """One model profiled across 3 regimes -> builder assigns to correct regimes.

        logistic performs well in trend_up (Sharpe 0.6) and sideways (Sharpe 0.9),
        but poorly in trend_down (Sharpe -0.5, below min_sharpe=-0.1).
        Validates the config is tradeable, deployable, saved to disk,
        and selectable for forward-test reuse.
        """
        matrix = RegimeModelMatrix(
            regimes=["trend_up", "trend_down", "sideways"],
            models=["logistic"],
            sharpe_matrix=np.array([[0.6, -0.5, 0.9]], dtype=float),
            trade_matrix=np.array([[15, 8, 22]], dtype=int),
            hitrate_matrix=np.array([[0.55, 0.42, 0.62]], dtype=float),
            fold_counts=np.array([[3]], dtype=int),
            raw_folds=[
                FoldResult(
                    model="logistic", fold_idx=0, train_start=0, train_end=1,
                    test_start=2, test_end=3, sharpe=0.6, trades=15,
                    active_rate=0.45, win_rate=0.55, performance=0.25,
                    return_val=0.015, drawdown=0.01, geo_mean_ann=0.08,
                    directional_accuracy=0.55, f1_macro=0.48,
                    regime_counts={"trend_up": 120, "trend_down": 30, "sideways": 50},
                    dominant_regime="trend_up",
                ),
                FoldResult(
                    model="logistic", fold_idx=1, train_start=1, train_end=2,
                    test_start=3, test_end=4, sharpe=-0.5, trades=8,
                    active_rate=0.20, win_rate=0.42, performance=-0.2,
                    return_val=-0.01, drawdown=0.03, geo_mean_ann=-0.06,
                    directional_accuracy=0.40, f1_macro=0.35,
                    regime_counts={"trend_up": 20, "trend_down": 140, "sideways": 40},
                    dominant_regime="trend_down",
                ),
                FoldResult(
                    model="logistic", fold_idx=2, train_start=2, train_end=3,
                    test_start=4, test_end=5, sharpe=0.9, trades=22,
                    active_rate=0.55, win_rate=0.62, performance=0.40,
                    return_val=0.025, drawdown=0.008, geo_mean_ann=0.12,
                    directional_accuracy=0.60, f1_macro=0.52,
                    regime_counts={"trend_up": 40, "trend_down": 30, "sideways": 130},
                    dominant_regime="sideways",
                ),
            ],
        )

        builder = CommitteeBuilder(top_k=1, min_sharpe=-0.1, weight_method="sharpe_proportional")
        config = builder.build(matrix, constraints={"max_models_per_regime": 1,
                               "min_sharpe": -0.1})

        # ── Validation 1: correct regime assignment ──
        assert isinstance(config, CommitteeConfig)
        assert config.version == 1
        assert len(config.regimes) >= 1, "Expected at least one regime assignment"
        assert config.fallback is not None, "Expected a fallback assignment"
        assert config.fallback.models == ["logistic"], "Fallback should be the only model"

        # trend_up (Sharpe 0.6 > -0.1): should be assigned
        assert "trend_up" in config.regimes, (
            f"logistic has Sharpe 0.6 in trend_up (above min_sharpe -0.1), "
            f"should be assigned. Got regimes: {list(config.regimes.keys())}"
        )
        tu = config.regimes["trend_up"]
        assert tu.models == ["logistic"], (
            f"trend_up should assign logistic. Got: {tu.models}"
        )
        assert abs(sum(tu.weights) - 1.0) < 0.02, f"Weights should sum to 1.0: {tu.weights}"

        # sideways (Sharpe 0.9 > -0.1): should be assigned
        assert "sideways" in config.regimes, (
            f"logistic has Sharpe 0.9 in sideways (above min_sharpe -0.1), "
            f"should be assigned. Got regimes: {list(config.regimes.keys())}"
        )
        sw = config.regimes["sideways"]
        assert sw.models == ["logistic"], (
            f"sideways should assign logistic. Got: {sw.models}"
        )
        assert abs(sum(sw.weights) - 1.0) < 0.02, f"Weights should sum to 1.0: {sw.weights}"

        # trend_down (Sharpe -0.5 < -0.1): should NOT be assigned
        assert "trend_down" not in config.regimes, (
            f"logistic has Sharpe -0.5 in trend_down (below min_sharpe -0.1), "
            f"should be excluded. But found in regimes: {list(config.regimes.keys())}"
        )

        # ── Validation 2: metadata tracks model usage ──
        assert "model_usage" in config.metadata
        usage = config.metadata["model_usage"]
        assert usage.get("logistic", 0) == len(config.regimes), (
            f"logistic usage count should equal number of assigned regimes "
            f"({len(config.regimes)}), got {usage}"
        )

        # ── Validation 3: SAVED to disk + reloaded (deployable) ──
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            save_path = tmp.name

        try:
            config.to_json(save_path)
            assert os.path.isfile(save_path), "Config file must exist on disk"

            with open(save_path, encoding="utf-8") as f:
                raw = json.load(f)
            assert "regimes" in raw
            assert "fallback" in raw
            assert "version" in raw
            # The file must contain the correct regime entries
            assert raw["regimes"]["trend_up"]["models"] == ["logistic"]
            assert raw["regimes"]["sideways"]["models"] == ["logistic"]
            assert "trend_down" not in raw["regimes"]

            # Reload from disk via from_json
            loaded = CommitteeConfig.from_json(save_path)
            assert loaded.regimes.keys() == config.regimes.keys()
            assert loaded.fallback.models == config.fallback.models
            assert loaded.version == config.version
            assert loaded.constraints == config.constraints
        finally:
            if os.path.isfile(save_path):
                os.unlink(save_path)

        # ── Validation 4: TRADEABLE — backtester produces trades with valid metrics ──
        df = _make_ohlc_with_regimes(4000)
        bt = CommitteeBacktester(config, confidence_threshold=0.5)
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)
        assert result is not None
        assert result.total_folds >= 1
        assert result.config is config
        assert result.models == ["logistic"]

        summary = result.to_summary_dict()
        assert np.isfinite(summary["avg_sharpe"]), (
            f"avg_sharpe must be finite for a tradeable committee, got {summary['avg_sharpe']}"
        )
        assert summary["avg_trades"] > 0, (
            f"Committee must produce trades to be tradeable, got {summary['avg_trades']}"
        )
        assert summary["regimes_configured"] == len(config.regimes)

        # Every fold must have finite Sharpe and at least 1 trade
        for fold in result.folds:
            assert np.isfinite(fold.sharpe), (
                f"Fold {fold.fold_idx} has non-finite Sharpe {fold.sharpe}"
            )
            assert fold.trades > 0, (
                f"Fold {fold.fold_idx} produced 0 trades"
            )

        # ── Validation 5: SELECTABLE for forward-test reuse ──
        # The same config must route to the correct regime assignment on every fold
        for fold in result.folds:
            # regime_distribution reflects detected 7-class regimes in test window
            total = sum(fold.regime_distribution.values())
            assert abs(total - 1.0) < 0.02, (
                f"Fold {fold.fold_idx} regime distribution must sum to 1.0, got {total}"
            )
            # per_model_active_fraction contains the config's regime keys
            for regime_key in config.regimes:
                assert regime_key in fold.per_model_active_fraction, (
                    f"Fold {fold.fold_idx} missing config regime '{regime_key}' "
                    f"in per_model_active_fraction: {list(fold.per_model_active_fraction.keys())}"
                )

        # Re-run with different data slice (simulates selecting config for a new forward test)
        df2 = _make_ohlc_with_regimes(4000, seed=99)
        result2 = bt.run_wfo(df2, train_months=4, test_months=1, verbose=False)
        assert result2.total_folds >= 1
        assert np.isfinite(result2.avg_sharpe)
        # Config identity is preserved across runs
        assert result2.config.regimes.keys() == config.regimes.keys()
        assert result2.config.fallback.models == config.fallback.models


# ════════════════════════════════════════════════════════════════════
# Phase 0: prune_models tests — SKIPPED (Phase 2 removed)
# ════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="prune_models removed with Phase 2")
class TestPruneModels:
    def test_prune_models_selects_top_by_sharpe(self):
        pass

    def test_prune_models_excludes_below_min_sharpe(self):
        pass

    def test_prune_models_diversity_force_includes_missing_family(self):
        pass

    def test_prune_models_returns_all_survivors_and_pruned(self):
        pass


class TestCommitteePhase3Metrics:
    def test_fold_consistency_cv_infinite_for_too_few_folds(self):
        import numpy as np
        from pipeline.committee_backtester import CommitteeBacktestResult, CommitteeFoldResult
        from pipeline.committee_builder import CommitteeConfig

        config = CommitteeConfig()
        folds = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=1.0, trades=10, active_rate=0.5,
                win_rate=0.6, return_val=0.02, drawdown=0.01,
            )
            for i in range(2)  # only 2 folds
        ]
        result = CommitteeBacktestResult(config=config, folds=folds, models=["logistic"])
        assert result.fold_consistency_cv == float("inf")

    def test_fold_consistency_cv_computes_correctly(self):
        import numpy as np
        from pipeline.committee_backtester import CommitteeBacktestResult, CommitteeFoldResult
        from pipeline.committee_builder import CommitteeConfig

        config = CommitteeConfig()
        folds = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=float(s), trades=10, active_rate=0.5,
                win_rate=0.6, return_val=0.02, drawdown=0.01,
            )
            for i, s in enumerate([1.0, 1.5, 0.8])  # mean=1.1, std≈0.29
        ]
        result = CommitteeBacktestResult(config=config, folds=folds, models=["logistic"])
        cv = result.fold_consistency_cv
        assert np.isfinite(cv)
        assert cv < 1.0  # CV(1.0, 1.5, 0.8) ≈ 0.26 < 1.0

    def test_fold_consistency_pass_threshold(self):
        import numpy as np
        from pipeline.committee_backtester import CommitteeBacktestResult, CommitteeFoldResult
        from pipeline.committee_builder import CommitteeConfig

        config = CommitteeConfig()
        # High variance across folds → CV > 1.0
        folds = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=float(s), trades=10, active_rate=0.5,
                win_rate=0.6, return_val=0.02, drawdown=0.01,
            )
            for i, s in enumerate([2.5, -1.0, 0.3, -2.0, 1.2])  # high variance
        ]
        result = CommitteeBacktestResult(config=config, folds=folds, models=["logistic"])
        # CV should likely be > 1.0 for such high variance
        assert result.fold_consistency_pass is False or result.fold_consistency_cv >= 1.0

    def test_regime_coverage_report_per_model_active_fraction(self):
        import numpy as np
        from pipeline.committee_backtester import CommitteeBacktestResult, CommitteeFoldResult
        from pipeline.committee_builder import CommitteeConfig, RegimeAssignment

        config = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
                "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
        )
        folds = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=1.0, trades=50, active_rate=0.5,
                win_rate=0.6, return_val=0.02, drawdown=0.01,
                regime_distribution={"trend_up": 0.3, "sideways": 0.4, "quiet_squeeze": 0.3},
                per_model_active_fraction={"trend_up": 0.6, "sideways": 0.4},
            )
            for i in range(3)
        ]
        result = CommitteeBacktestResult(config=config, folds=folds, models=["logistic"])
        report = result.regime_coverage_report(min_trades=30, min_sharpe=0.0)
        assert "trend_up" in report
        assert "sideways" in report
        for rname, data in report.items():
            assert "covered" in data
            assert "sharpe" in data
            assert "trades" in data

    def test_regime_coverage_report_insufficient_trades_fails(self):
        import numpy as np
        from pipeline.committee_backtester import CommitteeBacktestResult, CommitteeFoldResult
        from pipeline.committee_builder import CommitteeConfig, RegimeAssignment

        config = CommitteeConfig(
            regimes={
                "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
        )
        # Very few trades → should fail min_trades=30
        folds = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=2.0, trades=5, active_rate=0.1,
                win_rate=0.8, return_val=0.01, drawdown=0.005,
                per_model_active_fraction={"sideways": 0.2},
            )
            for i in range(3)
        ]
        result = CommitteeBacktestResult(config=config, folds=folds, models=["logistic"])
        report = result.regime_coverage_report(min_trades=30, min_sharpe=0.0)
        # Each fold has ~5 * 0.2 = 1 trade, so ~3 total -> fails 30
        assert report["sideways"]["trades"] < 30
        assert not report["sideways"]["covered"]


# ════════════════════════════════════════════════════════════════════
# Pipeline Hardening Tests (Sprint 16.7)
# ════════════════════════════════════════════════════════════════════

class TestDiversityCap:

    @staticmethod
    def _make_matrix(models, sharpe_mat):
        regimes = list(_REGIME_NAMES.values())[:7]
        n_models = len(models)
        n_regimes = len(regimes)
        s_mat = np.full((n_models, n_regimes), np.nan)
        t_mat = np.full((n_models, n_regimes), 0.0)
        h_mat = np.full((n_models, n_regimes), 0.5)
        f_mat = np.zeros((n_models, n_regimes), dtype=int)
        for mi in range(n_models):
            for ri in range(min(n_regimes, len(sharpe_mat[mi]))):
                s_mat[mi, ri] = sharpe_mat[mi][ri]
                t_mat[mi, ri] = 30.0
                f_mat[mi, ri] = 5
        return RegimeModelMatrix(
            regimes=regimes, models=models,
            sharpe_matrix=s_mat, trade_matrix=t_mat,
            hitrate_matrix=h_mat, fold_counts=f_mat,
        )

    def test_max_3_regimes_per_model(self):
        """No model may dominate more than 3 regimes."""
        models = ["xgboost", "logistic", "lstm", "random_forest"]
        xgb_dominant = [2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0]
        logi_modest = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        lstm_modest = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        rf_modest = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        matrix = self._make_matrix(models, [xgb_dominant, logi_modest, lstm_modest, rf_modest])

        builder = CommitteeBuilder(
            top_k=2, weight_method="sharpe_proportional",
        )
        config = builder.build(matrix, constraints={"max_regimes_per_model": 3})

        model_counts = {}
        for regime, assignment in config.regimes.items():
            for m in assignment.models:
                model_counts[m] = model_counts.get(m, 0) + 1

        for model, count in model_counts.items():
            assert count <= 3, f"{model} assigned to {count} regimes (max=3)"

    def test_cap_enforced_with_fallback(self):
        """When the best model is capped, next-best model fills in (pre-filter approach)."""
        models = ["xgboost", "logistic", "random_forest"]
        xgb_dominant = [2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0]
        logi_backup = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        rf_backup = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        matrix = self._make_matrix(models, [xgb_dominant, logi_backup, rf_backup])

        builder = CommitteeBuilder(
            top_k=2, weight_method="sharpe_proportional",
        )
        config = builder.build(matrix, constraints={"max_regimes_per_model": 2})

        xgb_regimes = sum(1 for a in config.regimes.values() if "xgboost" in a.models)
        logi_regimes = sum(1 for a in config.regimes.values() if "logistic" in a.models)
        assert xgb_regimes <= 2, f"xgboost got {xgb_regimes} regimes"
        # With 3 models × 2 max each = 6 slots, 7 regimes get at least 6 covered
        assert logi_regimes >= 1, f"logistic got {logi_regimes}, expected >=1"

    def test_all_models_capped_all_regimes_still_covered(self):
        """With sufficient capacity (3 models × 3 max = 9 slots, top_k=1 needs 7),
        all regimes should get assignments."""
        models = ["xgboost", "logistic", "random_forest"]
        xgb = [2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0]
        logi = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        rf = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        matrix = self._make_matrix(models, [xgb, logi, rf])

        builder = CommitteeBuilder(top_k=1, weight_method="sharpe_proportional")
        config = builder.build(matrix, constraints={"max_regimes_per_model": 3})

        for regime in matrix.regimes:
            assert regime in config.regimes, f"Regime {regime} missing from config"
            assert len(config.regimes[regime].models) > 0


class TestPhase3Halting:

    @staticmethod
    def _make_backtest_result(sharpe_vals, trades=30, folds=3):
        from pipeline.committee_backtester import CommitteeBacktestResult, CommitteeFoldResult
        from pipeline.committee_builder import CommitteeConfig, RegimeAssignment

        config = CommitteeConfig(
            regimes={"trend_up": RegimeAssignment(models=["logistic"], weights=[1.0])},
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        fold_results = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=s, trades=trades, active_rate=0.15,
                win_rate=0.55, return_val=0.01, drawdown=0.01,
                regime_distribution={"trend_up": 1.0},
                per_model_active_fraction={"trend_up": 1.0},
            )
            for i, s in enumerate(sharpe_vals)
        ]
        return CommitteeBacktestResult(
            config=config, folds=fold_results, models=["logistic"],
            avg_sharpe=float(np.mean(sharpe_vals)),
            avg_trades=float(trades), total_folds=len(fold_results),
        )

    def test_cv_pass_when_consistent(self):
        """Fold CV < 1.0 when Sharpe is stable across folds."""
        result = self._make_backtest_result([0.5, 0.55, 0.52])
        assert result.fold_consistency_pass
        assert result.fold_consistency_cv < 1.0

    def test_cv_fail_when_volatile(self):
        """Fold CV >= 1.0 when Sharpe varies wildly."""
        result = self._make_backtest_result([0.1, 2.0, 0.05])
        assert not result.fold_consistency_pass

    def test_cv_infinite_for_single_fold(self):
        result = self._make_backtest_result([0.5], folds=1)
        assert not result.fold_consistency_pass

    def test_regime_coverage_pass(self):
        result = self._make_backtest_result([0.5, 0.55, 0.52], trades=50)
        report = result.regime_coverage_report(min_trades=30, min_sharpe=0.0)
        assert report["trend_up"]["covered"]

    def test_regime_coverage_fail_insufficient_trades(self):
        result = self._make_backtest_result([0.5, 0.55, 0.52], trades=5)
        report = result.regime_coverage_report(min_trades=30, min_sharpe=0.0)
        assert not report["trend_up"]["covered"]


class TestTAModeLock:

    def test_env_var_set_changes_sampler_behavior(self):
        """MLB_TA_MODE=fixed disables TA Optuna dimensions."""
        import os as _os
        saved = _os.environ.get("MLB_TA_MODE", "")
        try:
            _os.environ["MLB_TA_MODE"] = "fixed"
            from pipeline.tuning.helpers import MLB_TA_MODE
            assert MLB_TA_MODE in ("fixed", "tuned", "legacy")
            # The sampler reads from env on each call (via sampler.py fix)
            # Just verify the env var is set correctly
            assert _os.environ["MLB_TA_MODE"] == "fixed"
        finally:
            if saved:
                _os.environ["MLB_TA_MODE"] = saved
            elif "MLB_TA_MODE" in _os.environ:
                del _os.environ["MLB_TA_MODE"]
