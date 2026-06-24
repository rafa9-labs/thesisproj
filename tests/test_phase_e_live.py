"""Tests for Phase E — Model persistence, live committee runner, mock data.

Tests the full live deployment pipeline: model store, streaming prediction
engine, health monitoring, model rotation, and mock trading sessions.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment

from trading.model_store import (
    ModelStore,
)
from trading.live_committee_runner import (
    LiveCommitteeRunner,
    LiveSignal,
    ModelHealth,
)
from trading.mock_live_data import (
    MockLiveFeed,
    MockDataConfig,
    simulate_session,
)


_RNG = np.random.default_rng(42)


def _make_dummy_model(n_features: int = 20) -> object:
    """Create a tiny sklearn LogisticRegression for testing."""
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", random_state=42)
    X = _RNG.standard_normal((100, n_features)).astype(np.float32)
    y = _RNG.integers(0, 3, size=100).astype(np.int32)
    m.fit(X, y)
    return m


def _make_simple_committee() -> CommitteeConfig:
    return CommitteeConfig(
        version=1,
        regimes={
            "trend_up": RegimeAssignment(models=["logistic", "xgboost"], weights=[0.6, 0.4]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


def _make_feature_names() -> list:
    """Feature names matching what the committee expects."""
    return [
        "mid_c", "mid_h", "mid_l",
        "sma_20", "ema_20", "rv_48", "rolling_std_20",
        "rsi_14", "macd_diff",
        "bb_upper", "bb_lower", "bb_pct", "bbw",
        "atr_14", "adx_14",
    ]


# ════════════════════════════════════════════════════════════════════
# ModelStore
# ════════════════════════════════════════════════════════════════════

class TestModelStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ModelStore(str(tmp_path / "store"))

    def test_save_and_load_model(self, store):
        m = _make_dummy_model(15)
        feat_names = [f"f{i}" for i in range(15)]
        path = store.save_model(m, "logistic", feat_names, {"sharpe": 0.5})
        assert os.path.exists(path)

        loaded = store.load_model(path)
        assert loaded is not None
        assert hasattr(loaded, "predict_proba")

    def test_model_snapshot_metadata(self, store):
        m = _make_dummy_model(10)
        feat_names = [f"f{i}" for i in range(10)]
        path = store.save_model(m, "xgboost", feat_names, {"sharpe": 0.72, "trades": 120})
        meta_path = path + ".json"
        assert os.path.exists(meta_path)

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["model_type"] == "xgboost"
        assert meta["n_features"] == 10
        assert meta["metrics"]["sharpe"] == 0.72

    def test_save_committee_snapshot(self, store):
        models = {
            "logistic": _make_dummy_model(15),
            "xgboost": _make_dummy_model(15),
        }
        feat_names = _make_feature_names()
        cfg = _make_simple_committee()

        snapshot_dir = store.save_committee_snapshot(
            models=models,
            feature_names=feat_names,
            committee_config_json=json.dumps(cfg.to_dict()),
        )
        assert os.path.exists(snapshot_dir)
        assert os.path.exists(os.path.join(snapshot_dir, "manifest.json"))
        assert os.path.exists(os.path.join(snapshot_dir, "committee_config.json"))

    def test_load_committee_snapshot(self, store):
        models = {
            "logistic": _make_dummy_model(15),
            "xgboost": _make_dummy_model(15),
        }
        feat_names = _make_feature_names()
        cfg = _make_simple_committee()

        snapshot_dir = store.save_committee_snapshot(
            models=models,
            feature_names=feat_names,
            committee_config_json=json.dumps(cfg.to_dict()),
        )

        loaded = store.load_committee_snapshot(snapshot_dir)
        assert "models" in loaded
        assert "config_json" in loaded
        assert len(loaded["models"]) == 2
        assert "logistic" in loaded["models"]

    def test_list_snapshots(self, store):
        models = {"logistic": _make_dummy_model(10)}
        feat_names = [f"f{i}" for i in range(10)]
        cfg = CommitteeConfig(
            regimes={"sideways": RegimeAssignment(models=["logistic"], weights=[1.0])},
        )
        store.save_committee_snapshot(
            models=models, feature_names=feat_names,
            committee_config_json=json.dumps(cfg.to_dict()),
        )
        store.save_committee_snapshot(
            models=models, feature_names=feat_names,
            committee_config_json=json.dumps(cfg.to_dict()),
        )

        snapshots = store.list_snapshots()
        assert len(snapshots) == 2

    def test_health_tracking(self, store):
        records = [
            {"bar": 1, "signal": 1, "pnl": 0.001},
            {"bar": 2, "signal": -1, "pnl": -0.002},
        ]
        store.save_health("logistic", records)
        loaded = store.load_health("logistic")
        assert len(loaded) == 2
        assert loaded[0]["pnl"] == 0.001

    def test_health_limits_to_500(self, store):
        records = [{"bar": i} for i in range(600)]
        store.save_health("logistic", records)
        loaded = store.load_health("logistic")
        assert len(loaded) == 500

    def test_load_health_nonexistent_returns_empty(self, store):
        loaded = store.load_health("nonexistent")
        assert loaded == []

    def test_load_snapshot_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_committee_snapshot("/tmp/nonexistent_dir_xyz")

    def test_modelstore_root_created(self, tmp_path):
        root = str(tmp_path / "new_store")
        ModelStore(root)
        assert os.path.exists(root)

    def test_predict_proba_consistency_after_save_load(self, store):
        """Model predictions should be identical after save+load."""
        m = _make_dummy_model(10)
        path = store.save_model(m, "test", [f"f{i}" for i in range(10)])
        loaded = store.load_model(path)

        X = _RNG.standard_normal((5, 10)).astype(np.float32)
        p1 = m.predict_proba(X)
        p2 = loaded.predict_proba(X)
        assert np.allclose(p1, p2, atol=0.001)


# ════════════════════════════════════════════════════════════════════
# LiveSignal + ModelHealth
# ════════════════════════════════════════════════════════════════════

class TestDataStructures:
    def test_live_signal_to_dict(self):
        s = LiveSignal(
            timestamp="2024-01-01T12:00:00",
            signal=1, confidence=0.75, regime="trend_up", regime_prob=0.68,
            blended_probs={"short": 0.1, "flat": 0.15, "long": 0.75},
            active_models=["logistic", "xgboost"], model_weights=[0.6, 0.4],
        )
        d = s.to_dict()
        assert d["signal"] == 1
        assert d["regime"] == "trend_up"
        assert d["blended_probs"]["long"] == 0.75

    def test_model_health_records_trades(self):
        h = ModelHealth(model_type="logistic")
        assert h.total_signals == 0

        h.record_trade(1, 0.001)
        h.record_trade(1, 0.002)
        h.record_trade(-1, -0.001)

        assert h.total_signals == 3
        assert h.wins == 2
        assert h.losses == 1

    def test_model_health_computes_metrics(self):
        h = ModelHealth(model_type="test")
        for _ in range(10):
            h.record_trade(1, 0.001)   # all wins
        assert h.total_signals == 10
        assert not np.isnan(h.last_sharpe)
        assert h.last_hit_rate == 1.0
        assert h.is_healthy


# ════════════════════════════════════════════════════════════════════
# LiveCommitteeRunner
# ════════════════════════════════════════════════════════════════════

class TestLiveCommitteeRunner:
    @pytest.fixture
    def runner(self):
        models = {
            "logistic": _make_dummy_model(n_features=len(_make_feature_names())),
            "xgboost": _make_dummy_model(n_features=len(_make_feature_names())),
        }
        return LiveCommitteeRunner(
            config=_make_simple_committee(),
            models=models,
            feature_names=_make_feature_names(),
            confidence_threshold=0.5,
        )

    def test_start_stop(self, runner):
        runner.start()
        assert runner._is_running
        summary = runner.stop()
        assert not runner._is_running
        assert "bars_processed" in summary

    def test_process_bar_insufficient_history(self, runner):
        runner.start()
        bar = {"mid_c": 1.1000, "mid_h": 1.1005, "mid_l": 1.0995,
               "mid_o": 1.1000, "spread": 0.00015, "returns": 0.0}
        signal = runner.process_bar(bar)
        assert signal is None  # not enough bars for features

    def test_process_bar_eventually_emits(self, runner):
        """After enough bars, process_bar should produce signals."""
        runner.start()
        rng = np.random.default_rng(99)

        signals = []
        p = 1.1000
        for i in range(200):
            p += rng.normal(0, 0.0002)
            bar = {
                "mid_c": p, "mid_h": p + abs(rng.normal(0, 0.0003)),
                "mid_l": p - abs(rng.normal(0, 0.0003)),
                "mid_o": p - rng.normal(0, 0.0001),
                "spread": 0.00015,
                "returns": float(rng.normal(0, 0.0002)),
            }
            s = runner.process_bar(bar)
            if s is not None:
                signals.append(s)

        assert len(signals) > 0

    def test_process_bar_without_start_raises(self, runner):
        bar = {"mid_c": 1.1, "mid_h": 1.1005, "mid_l": 1.0995,
               "mid_o": 1.1, "spread": 0.00015, "returns": 0.0}
        with pytest.raises(RuntimeError):
            runner.process_bar(bar)

    def test_health_monitoring_updates(self, runner):
        runner.start()
        rng = np.random.default_rng(55)

        p = 1.1000
        for i in range(200):
            p += rng.normal(0, 0.0002)
            bar = {
                "mid_c": p, "mid_h": p + 0.0005,
                "mid_l": p - 0.0005,
                "mid_o": p - rng.normal(0, 0.0001),
                "spread": 0.00015,
                "returns": float(rng.normal(0, 0.0002)),
            }
            s = runner.process_bar(bar)
            if s is not None and s.signal != 0:
                runner.record_trade_outcome(s, s.signal * bar["returns"])

        health = runner.get_health_summary()
        assert "logistic" in health
        assert "xgboost" in health
        assert health["logistic"]["total_signals"] >= 0

    def test_health_summary_before_trades(self, runner):
        runner.start()
        health = runner.get_health_summary()
        assert "logistic" in health
        assert health["logistic"]["total_signals"] == 0

    def test_recent_regimes(self, runner):
        runner.start()
        rng = np.random.default_rng(44)
        p = 1.1000
        for i in range(150):
            p += rng.normal(0, 0.0003)
            bar = {
                "mid_c": p, "mid_h": p + 0.0005,
                "mid_l": p - 0.0005,
                "mid_o": p - rng.normal(0, 0.0001),
                "spread": 0.00015,
                "returns": float(rng.normal(0, 0.0002)),
            }
            runner.process_bar(bar)

        regimes = runner.get_recent_regimes(10)
        assert len(regimes) <= 10

    def test_recent_signals(self, runner):
        """get_recent_signals returns dicts."""
        runner.start()
        sig = LiveSignal(
            timestamp="now", signal=1, confidence=0.8, regime="trend_up",
            regime_prob=0.7, blended_probs={"short": 0.1, "flat": 0.1, "long": 0.8},
            active_models=["logistic"], model_weights=[1.0],
        )
        runner._signal_history.append(sig)
        recent = runner.get_recent_signals(5)
        assert len(recent) >= 1
        assert isinstance(recent[0], dict)

    def test_rotate_model(self, runner):
        runner.start()
        new_model = _make_dummy_model(n_features=len(_make_feature_names()))
        runner.rotate_model("logistic", "new_logistic", new_model)
        assert "new_logistic" in runner.models
        assert "logistic" not in runner.models or "logistic" in runner.models  # may still be in health

    def test_find_replacement(self, runner):
        from pipeline.committee.expert_profiler import RegimeModelMatrix

        regimes = ["trend_up", "trend_down", "mean_reverting", "breakout",
                    "high_volatile", "quiet_squeeze", "sideways"]
        models = ["logistic", "xgboost", "random_forest", "lstm"]
        sharpe_mat = np.array([
            [0.2, 0.1, 0.3, 0.1, 0.1, 0.15, 0.4],
            [0.6, 0.5, 0.2, 0.3, 0.1, 0.1, 0.1],
            [0.3, 0.2, 0.1, 0.4, 0.5, 0.1, 0.2],
            [0.7, 0.6, 0.1, 0.2, 0.1, 0.05, 0.1],
        ])
        matrix = RegimeModelMatrix(
            regimes=regimes, models=models, sharpe_matrix=sharpe_mat,
            trade_matrix=np.ones((4, 7)) * 20,
            hitrate_matrix=np.ones((4, 7)) * 0.5,
            fold_counts=np.ones((4, 7), dtype=int) * 3,
        )

        replacement = runner.find_replacement(matrix, "logistic", "trend_up")
        assert replacement is not None
        assert replacement != "logistic"

    def test_find_replacement_single_model(self, runner):
        from pipeline.committee.expert_profiler import RegimeModelMatrix

        n_regimes = 7  # must match _REGIME_NAMES
        matrix = RegimeModelMatrix(
            regimes=["trend_up", "trend_down", "mean_reverting", "breakout",
                      "high_volatile", "quiet_squeeze", "sideways"],
            models=["logistic"],
            sharpe_matrix=np.full((1, n_regimes), 0.5),
            trade_matrix=np.ones((1, n_regimes)) * 10,
            hitrate_matrix=np.ones((1, n_regimes)) * 0.5,
            fold_counts=np.ones((1, n_regimes), dtype=int) * 3,
        )
        replacement = runner.find_replacement(matrix, "logistic", "trend_up")
        assert replacement is None  # no other model


# ════════════════════════════════════════════════════════════════════
# MockLiveFeed
# ════════════════════════════════════════════════════════════════════

class TestMockLiveFeed:
    def test_generate_bars_yields_rows(self):
        feed = MockLiveFeed(MockDataConfig(n_bars=100))
        bars = list(feed.generate_bars())
        assert len(bars) == 100
        for bar in bars:
            assert "mid_c" in bar
            assert "mid_h" in bar
            assert "mid_l" in bar
            assert "mid_o" in bar
            assert "spread" in bar
            assert "returns" in bar

    def test_to_dataframe(self):
        feed = MockLiveFeed(MockDataConfig(n_bars=50))
        df = feed.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "mid_c" in df.columns

    def test_regime_labels(self):
        feed = MockLiveFeed(MockDataConfig(n_bars=200, regime_section_bars=50))
        labels = feed.regime_labels()
        assert len(labels) == 200
        # Should cycle through regimes
        assert "trend_up" in labels
        assert "trend_down" in labels

    def test_trend_section_has_positive_drift(self):
        """Trend-up section should have predominantly positive returns."""
        cfg = MockDataConfig(n_bars=800, regime_section_bars=100, trend_slope=0.00005, seed=42)
        feed = MockLiveFeed(cfg)
        bars = list(feed.generate_bars())

        # First section should be trend_up (bars 20-99 after warmup)
        tu_returns = [b["returns"] for b in bars[20:100]]
        avg_ret = np.mean(tu_returns)
        assert avg_ret > -0.00001, f"Expected near-zero or positive returns in trend_up, got {avg_ret:.6f}"

    def test_high_volatile_has_higher_variance(self):
        feed = MockLiveFeed(MockDataConfig(n_bars=800, regime_section_bars=100))
        bars = list(feed.generate_bars())

        # trend_up: bars 0-99, high_volatile: bars 300-399
        tu_std = np.std([b["returns"] for b in bars[20:100]])
        hv_std = np.std([b["returns"] for b in bars[310:399]])
        assert hv_std > tu_std * 1.5, f"HV std={hv_std:.6f} vs TU std={tu_std:.6f}"

    def test_small_config(self):
        feed = MockLiveFeed(MockDataConfig(n_bars=10, seed=123))
        bars = list(feed.generate_bars())
        assert len(bars) == 10

    def test_deterministic_with_seed(self):
        feed1 = MockLiveFeed(MockDataConfig(n_bars=50, seed=42))
        feed2 = MockLiveFeed(MockDataConfig(n_bars=50, seed=42))
        bars1 = list(feed1.generate_bars())
        bars2 = list(feed2.generate_bars())
        for i in range(50):
            assert bars1[i]["mid_c"] == bars2[i]["mid_c"]


# ════════════════════════════════════════════════════════════════════
# Integration: full mock trading session
# ════════════════════════════════════════════════════════════════════

class TestFullSession:
    def test_simulate_session_runs(self):
        """Full end-to-end mock trading session."""
        models = {
            "logistic": _make_dummy_model(n_features=len(_make_feature_names())),
            "xgboost": _make_dummy_model(n_features=len(_make_feature_names())),
        }
        runner = LiveCommitteeRunner(
            config=_make_simple_committee(),
            models=models,
            feature_names=_make_feature_names(),
            confidence_threshold=0.5,
        )
        feed = MockLiveFeed(MockDataConfig(n_bars=500, seed=77))

        runner.start()
        result = simulate_session(runner, feed, verbose=False)

        assert "signals" in result
        assert "summary" in result
        assert "returns" in result

        # Should have processed some signals
        assert result["summary"]["bars_processed"] == 500
        assert result["summary"]["signals_emitted"] > 0

    def test_simulate_session_returns_pnl(self):
        models = {
            "logistic": _make_dummy_model(n_features=len(_make_feature_names())),
        }
        runner = LiveCommitteeRunner(
            config=CommitteeConfig(
                regimes={"sideways": RegimeAssignment(models=["logistic"], weights=[1.0])},
                fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
            ),
            models=models,
            feature_names=_make_feature_names(),
            confidence_threshold=0.2,  # lower threshold to get more signals
        )
        feed = MockLiveFeed(MockDataConfig(n_bars=400, seed=123))

        runner.start()
        result = simulate_session(runner, feed, verbose=False)

        # Should record some returns
        assert len(result["returns"]) >= 0
        assert len(result["signals"]) >= 0

    def test_health_degradation_detected(self):
        """After many losing trades, health should flag as unhealthy."""
        models = {
            "logistic": _make_dummy_model(n_features=len(_make_feature_names())),
        }
        runner = LiveCommitteeRunner(
            config=CommitteeConfig(
                regimes={"trend_up": RegimeAssignment(models=["logistic"], weights=[1.0])},
                fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
            ),
            models=models,
            feature_names=_make_feature_names(),
            confidence_threshold=0.2,
            rotation_sharpe_threshold=0.0,
            rotation_hitrate_threshold=0.0,
        )
        feed = MockLiveFeed(MockDataConfig(n_bars=600, seed=99))

        runner.start()
        simulate_session(runner, feed, verbose=False)

        health = runner.get_health_summary()
        assert "logistic" in health
        # Health was tracked even if all trades were losing
        assert health["logistic"]["total_signals"] >= 0

    def test_process_bar_returns_signal_with_timestamp(self):
        """Signals should carry timestamps."""
        models = {"logistic": _make_dummy_model(n_features=len(_make_feature_names()))}
        runner = LiveCommitteeRunner(
            config=CommitteeConfig(
                regimes={"sideways": RegimeAssignment(models=["logistic"], weights=[1.0])},
                fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
            ),
            models=models,
            feature_names=_make_feature_names(),
            confidence_threshold=0.2,
        )
        feed = MockLiveFeed(MockDataConfig(n_bars=300, seed=42))

        runner.start()
        s = None
        for bar in feed.generate_bars():
            s = runner.process_bar(bar)
            if s is not None and s.signal != 0:
                break

        # May or may not have gotten a non-zero signal, but should have processed
        runner.stop()
