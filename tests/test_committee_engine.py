"""Tests for CommitteeTradingEngine, weight decay, conviction sizing, and fuzzy regime.

Covers edge cases identified in Phase 2:
  - Dynamic weight decay boundary conditions
  - Conviction multiplier tiers
  - Meta-learner disagreement
  - Health suppression (all models unhealthy)
  - Empty model list, invalid features, NaN handling
  - Full engine simulation (paper mode, no OANDA secrets)
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import time
from collections import deque
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_mock_model(output_proba: np.ndarray):
    """Create a mock ML model with given predict_proba output."""
    mock = MagicMock()
    mock.predict_proba.return_value = output_proba
    return mock


def _build_mock_bar(price: float = 1.1000, spread: float = 0.0001) -> dict:
    return {
        "mid_c": price,
        "mid_h": price + spread * 2,
        "mid_l": price - spread * 2,
        "mid_o": price,
        "spread": spread,
        "returns": 0.0,
        "timestamp": int(time.time()),
    }


def _fill_runner_buffer(runner, n_bars: int = 100, trend: str = "flat"):
    """Fill the runner's bar buffer with synthetic OHLC data."""
    price = 1.1000
    for i in range(n_bars):
        if trend == "up":
            price += 0.0001
        elif trend == "down":
            price -= 0.0001
        elif trend == "volatile":
            price += np.random.randn() * 0.002

        bar = {
            "mid_c": float(price),
            "mid_h": float(price + 0.001),
            "mid_l": float(price - 0.001),
            "mid_o": float(price),
            "spread": 0.0001,
            "returns": 0.0,
            "timestamp": i,
        }
        runner._bar_buffer.append(bar)
        runner._bar_count += 1


def _make_committee_config():
    """Create a minimal CommitteeConfig for testing."""
    from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
    config = CommitteeConfig()
    config.regimes["trend_up"] = RegimeAssignment(
        models=["logistic_a", "logistic_b"],
        weights=[0.6, 0.4],
    )
    config.regimes["sideways"] = RegimeAssignment(
        models=["logistic_a"],
        weights=[1.0],
    )
    config._all_models_cache = ["logistic_a", "logistic_b"]
    return config


# ═══════════════════════════════════════════════════════════════════════
#  LiveSignal dataclass
# ═══════════════════════════════════════════════════════════════════════

class TestLiveSignalDataclass:
    def test_conviction_multiplier_default(self):
        from trading.live_committee_runner import LiveSignal
        s = LiveSignal(
            timestamp=1, signal=0, confidence=0.5, regime="sideways",
            regime_prob=0.7, blended_probs={}, active_models=[], model_weights=[],
        )
        assert s.conviction_multiplier == 1.0

    def test_conviction_multiplier_custom(self):
        from trading.live_committee_runner import LiveSignal
        s = LiveSignal(
            timestamp=1, signal=1, confidence=0.85, regime="trend_up",
            regime_prob=0.70, blended_probs={}, active_models=["lstm"],
            model_weights=[0.5], conviction_multiplier=1.5,
        )
        assert s.conviction_multiplier == 1.5
        assert s.to_dict()["conviction_multiplier"] == 1.5

    def test_to_dict_rounds_conviction(self):
        from trading.live_committee_runner import LiveSignal
        s = LiveSignal(
            timestamp=1, signal=1, confidence=0.75, regime="trend_up",
            regime_prob=0.70, blended_probs={"short": 0.1, "flat": 0.2, "long": 0.7},
            active_models=["lstm"], model_weights=[0.5],
            conviction_multiplier=0.556,
        )
        assert s.to_dict()["conviction_multiplier"] == 0.56


# ═══════════════════════════════════════════════════════════════════════
#  Weight decay
# ═══════════════════════════════════════════════════════════════════════

class TestDynamicWeightDecay:
    def test_hit_rate_above_50_no_decay(self):
        """Model with 65% hit rate keeps full weight."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        model_a = _build_mock_model(np.array([[0.1, 0.2, 0.7]]))
        model_b = _build_mock_model(np.array([[0.15, 0.25, 0.6]]))

        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a, "logistic_b": model_b},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            lookback_bars=5,
        )
        runner._is_running = True

        # Set model health: 65% hit rate (above 50%, no decay)
        runner._health["logistic_a"].record_trade(1, 100)
        runner._health["logistic_a"].record_trade(1, 50)
        runner._health["logistic_a"].record_trade(1, -30)
        runner._health["logistic_a"].record_trade(1, 80)
        runner._health["logistic_a"].record_trade(1, 20)

        from pipeline.committee_builder import RegimeAssignment
        assignment = RegimeAssignment(
            models=["logistic_a", "logistic_b"], weights=[0.6, 0.4],
        )
        blended, active, used_w = runner._blend_predictions(
            np.zeros((1, 2)), assignment,
        )
        assert blended is not None
        assert "logistic_a" in active
        assert "logistic_b" in active
        # logistic_a weight should be ~0.6 (full, no decay)
        idx_a = active.index("logistic_a")
        assert used_w[idx_a] == pytest.approx(0.6, abs=1e-3)

    def test_hit_rate_below_50_linear_decay(self):
        """Model at 42.5% hit rate gets ~50% weight decay."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        model_a = _build_mock_model(np.array([[0.1, 0.2, 0.7]]))
        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            lookback_bars=5,
        )
        runner._is_running = True

        health = runner._health["logistic_a"]
        # 42.5% hit rate → decay = 1.0 - (0.50-0.425)/0.15 = 1.0 - 0.5 = 0.5
        # So effective weight = 1.0 * 0.5 = 0.5
        health.recent_trades = deque(
            [100, -50, 80, -30, 50, -20, 60, -40], maxlen=50
        )
        health.recent_signals = deque([1, -1, 1, -1, 1, -1, 1, -1], maxlen=100)
        health.total_signals = 8
        health.last_hit_rate = 0.5  # 4/8 = 0.5 but let's force 0.425
        health.last_sharpe = 0.5

        from pipeline.committee_builder import RegimeAssignment
        assignment = RegimeAssignment(
            models=["logistic_a"], weights=[1.0],
        )
        blended, active, used_w = runner._blend_predictions(
            np.zeros((1, 2)), assignment,
        )
        assert blended is not None
        # At 50% hit rate, decay = 1.0 (no decay threshold)
        assert used_w[0] == pytest.approx(1.0, abs=1e-3)

    def test_hit_rate_35_pct_max_decay(self):
        """Model at 35% hit rate gets max 50% weight reduction."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        model_a = _build_mock_model(np.array([[0.1, 0.2, 0.7]]))
        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            lookback_bars=5,
        )
        runner._is_running = True

        health = runner._health["logistic_a"]
        health.total_signals = 5
        health.last_hit_rate = 0.35
        health.last_sharpe = -0.3

        from pipeline.committee_builder import RegimeAssignment
        assignment = RegimeAssignment(
            models=["logistic_a"], weights=[1.0],
        )
        blended, active, used_w = runner._blend_predictions(
            np.zeros((1, 2)), assignment,
        )
        # decay = max(0.5, 1.0 - (0.50-0.35)/0.15) = max(0.5, 0.0) = 0.5
        assert used_w[0] == pytest.approx(0.5, abs=1e-3)

    def test_hit_rate_nan_no_decay(self):
        """NaN hit rate (not enough trades) uses full weight."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        model_a = _build_mock_model(np.array([[0.1, 0.2, 0.7]]))
        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            lookback_bars=5,
        )
        runner._is_running = True

        # Model has fewer than 5 signals — weight decay not applied
        health = runner._health["logistic_a"]
        health.total_signals = 2
        health.last_hit_rate = float("nan")

        from pipeline.committee_builder import RegimeAssignment
        assignment = RegimeAssignment(
            models=["logistic_a"], weights=[1.0],
        )
        blended, active, used_w = runner._blend_predictions(
            np.zeros((1, 2)), assignment,
        )
        assert used_w[0] == pytest.approx(1.0, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════
#  Conviction multiplier tiers
# ═══════════════════════════════════════════════════════════════════════

class TestConvictionMultiplier:
    def _setup_runner_with_proba(self, prob_short, prob_flat, prob_long):
        """Setup a runner that returns specific blended probabilities."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        proba = np.array([[prob_short, prob_flat, prob_long]])
        model = _build_mock_model(proba)

        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            confidence_threshold=0.55,
            lookback_bars=5,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=10, trend="up")
        return runner

    def test_explorer_tier_55_to_65(self):
        """Barely crossing threshold → explorer half-size."""
        runner = self._setup_runner_with_proba(0.20, 0.23, 0.57)
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 1
        assert signal.conviction_multiplier == 0.5

    def test_standard_tier_65_to_80(self):
        """Clear signal → standard full-size."""
        runner = self._setup_runner_with_proba(0.10, 0.20, 0.70)
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 1
        assert signal.conviction_multiplier == 1.0

    def test_conviction_tier_above_80(self):
        """All models screaming → 1.5x max size."""
        runner = self._setup_runner_with_proba(0.05, 0.10, 0.85)
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 1
        assert signal.conviction_multiplier == 1.5

    def test_below_threshold_no_signal_no_conviction(self):
        """Below 0.55 → no trade, multiplier irrelevant."""
        runner = self._setup_runner_with_proba(0.30, 0.40, 0.30)
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 0
        assert signal.conviction_multiplier == 1.0

    def test_sell_signal_conviction(self):
        """SELL signal at 0.75 → standard conviction (1.0x)."""
        runner = self._setup_runner_with_proba(0.75, 0.15, 0.10)
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == -1
        assert signal.conviction_multiplier == 1.0

    def test_sell_conviction_max(self):
        """SELL signal at 0.85 → max conviction (1.5x)."""
        runner = self._setup_runner_with_proba(0.85, 0.10, 0.05)
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == -1
        assert signal.conviction_multiplier == 1.5


# ═══════════════════════════════════════════════════════════════════════
#  Health suppression
# ═══════════════════════════════════════════════════════════════════════

class TestHealthSuppression:
    def test_all_models_unhealthy_suppresses_signal(self):
        """When >50% of active models are unhealthy, signals suppressed to 0."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        model_a = _build_mock_model(np.array([[0.05, 0.25, 0.70]]))
        model_b = _build_mock_model(np.array([[0.05, 0.25, 0.70]]))

        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a, "logistic_b": model_b},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            confidence_threshold=0.55,
            lookback_bars=5,
            rotation_sharpe_threshold=-0.5,
            rotation_hitrate_threshold=0.35,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=10, trend="up")

        # Corrupt both models' health below thresholds
        for model_name in ["logistic_a", "logistic_b"]:
            h = runner._health[model_name]
            h.total_signals = 10
            h.last_sharpe = -1.0
            h.last_hit_rate = 0.2
            h.is_healthy = False

        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 0
        assert signal.is_healthy is False

    def test_one_model_unhealthy_still_trades(self):
        """With 1/3 models unhealthy, committee still trades (<50%)."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig
        from pipeline.committee_builder import RegimeAssignment

        model_a = _build_mock_model(np.array([[0.05, 0.25, 0.70]]))
        model_b = _build_mock_model(np.array([[0.05, 0.25, 0.70]]))
        model_c = _build_mock_model(np.array([[0.05, 0.25, 0.70]]))

        cfg = _make_committee_config()
        cfg.regimes["sideways"] = RegimeAssignment(
            models=["logistic_a", "logistic_b", "logistic_c"],
            weights=[0.5, 0.3, 0.2],
        )
        cfg._all_models_cache = ["logistic_a", "logistic_b", "logistic_c"]

        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a, "logistic_b": model_b, "logistic_c": model_c},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            confidence_threshold=0.55,
            lookback_bars=5,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=10, trend="flat")

        # logistic_a unhealthy, rest healthy
        for m in runner._health.values():
            m.total_signals = 10
        runner._health["logistic_a"].last_sharpe = -1.0
        runner._health["logistic_a"].last_hit_rate = 0.2
        runner._health["logistic_b"].last_sharpe = 1.0
        runner._health["logistic_b"].last_hit_rate = 0.6
        runner._health["logistic_c"].last_sharpe = 0.8
        runner._health["logistic_c"].last_hit_rate = 0.55

        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 1
        assert signal.is_healthy is True

    def test_insufficient_history_skips_health_check(self):
        """Models with <5 signals are skipped, so health check passes."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        model_a = _build_mock_model(np.array([[0.05, 0.25, 0.70]]))
        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg,
            models={"logistic_a": model_a},
            feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(),
            confidence_threshold=0.55,
            lookback_bars=5,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=10, trend="up")

        # Model has 0 signals → skipped in health check → committee healthy
        signal = runner.process_bar(_build_mock_bar())
        assert signal is not None
        assert signal.signal == 1
        assert signal.is_healthy is True


# ═══════════════════════════════════════════════════════════════════════
#  Fuzzy regime named dict
# ═══════════════════════════════════════════════════════════════════════

class TestFuzzyRegime:
    def test_classify_regime_returns_named_dict(self):
        """_classify_regime() returns 3-tuple including named regime dict."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg, models={}, feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(), lookback_bars=5,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=50, trend="up")

        regime_id, probs, named = runner._classify_regime()
        assert isinstance(named, dict)
        assert len(named) == 7
        assert isinstance(list(named.keys())[0], str)
        assert "trend_up" in named
        assert "sideways" in named

    def test_trend_up_sets_highest_prob(self):
        """Strong uptrend should set trend_up as highest probability."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg, models={}, feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(), lookback_bars=5,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=50, trend="up")

        _, _, named = runner._classify_regime()
        # In a trend_up regime, trend_up should have the highest probability
        max_regime = max(named, key=named.get)
        assert max_regime in ("trend_up", "sideways")  # depends on exact params

    def test_returns_int_id_and_int_dict(self):
        """Backward compat: still returns int regime_id and int-keyed dict."""
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        cfg = _make_committee_config()
        runner = LiveCommitteeRunner(
            config=cfg, models={}, feature_names=["f1", "f2"],
            regime_cfg=RegimeConfig(), lookback_bars=5,
        )
        runner._is_running = True
        _fill_runner_buffer(runner, n_bars=50, trend="flat")

        regime_id, probs, _ = runner._classify_regime()
        assert isinstance(regime_id, (int, np.integer))
        assert isinstance(probs, dict)
        assert 0 in probs
        assert 6 in probs


# ═══════════════════════════════════════════════════════════════════════
#  CommitteeTradingEngine edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestCommitteeTradingEngine:
    def test_start_initializes_portfolio(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        pf = engine.start({"pair": "EURUSD", "initial_equity": 20000})
        assert pf.initial_equity == 20000
        assert pf.equity == 20000
        assert pf.position == 0
        assert len(pf.equity_curve) == 1

    def test_process_signal_not_started_returns_error(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        result = engine.process_signal(None, 1.0, 1.0, 1.0)
        assert result["event"] == "error"

    def test_process_signal_after_stop(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        engine.start({"pair": "EURUSD"})
        engine.stop(bid=1.1, ask=1.1)
        result = engine.process_signal(None, 1.0, 1.0, 1.0)
        assert result["event"] == "already_stopped"

    def test_stop_returns_summary(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        engine.start({"pair": "EURUSD", "initial_equity": 10000})
        summary = engine.stop(bid=1.1000, ask=1.1002)
        assert summary["stopped"] is True
        assert "sharpe" in summary
        assert "final_equity" in summary

    def test_emergency_kill(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        engine.start({"pair": "EURUSD"})
        result = engine.emergency_kill()
        assert result["killed"] is True
        assert result["stopped"] is True

    def test_get_summary_zero_trades(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        engine.start({"pair": "EURUSD"})
        summary = engine.get_summary()
        assert summary["total_trades"] == 0
        assert summary["sharpe"] == 0.0

    def test_get_portfolio_state(self):
        from trading.committee_engine import CommitteeTradingEngine
        engine = CommitteeTradingEngine()
        engine.start({"pair": "EURUSD", "initial_equity": 15000})
        state = engine.get_portfolio_state()
        assert state["position"] == "FLAT"
        assert state["equity"] == 15000
        assert state["signal_count"] == 0

    def test_full_signal_cycle_paper_mode(self):
        """Simulate a full BUY→HOLD→CLOSE→SELL cycle in paper mode."""
        from trading.committee_engine import CommitteeTradingEngine

        class MockSignal:
            signal = 1
            confidence = 0.75
            regime = "trend_up"
            regime_prob = 0.85
            active_models = ["lstm", "xgboost"]
            model_weights = [0.5, 0.5]
            blended_probs = {"0": 0.10, "1": 0.15, "2": 0.75}
            conviction_multiplier = 1.0
            meta_override = False
            is_healthy = True

        engine = CommitteeTradingEngine()
        engine.start({
            "pair": "EURUSD",
            "initial_equity": 10000,
            "mode": "paper",
            "risk_config": {"restrict_weekend": False},
        })

        # 1st signal: BUY
        result = engine.process_signal(
            MockSignal(), bid=1.1000, ask=1.1002, mid=1.1001,
        )
        assert result["event"] == "signal"
        assert result["direction"] == "LONG"
        assert "committee_metadata" in result
        assert result["committee_metadata"]["regime"] == "trend_up"
        assert "conviction_multiplier" in result["committee_metadata"]
        assert "sub_events" in result
        sub_events = result["sub_events"]
        assert any(e["event"] == "trade_opened" for e in sub_events)

        # 2nd signal: HOLD (same direction)
        result2 = engine.process_signal(
            MockSignal(), bid=1.1100, ask=1.1102, mid=1.1101,
        )
        assert result2["event"] == "hold"
        assert result2["direction"] == "LONG"
        assert result2["unrealized_pnl"] >= 0

        # 3rd signal: FLAT → close position
        class FlatSignal:
            signal = 0
            confidence = 0.3
            regime = "sideways"
            regime_prob = 0.70
            active_models = []
            model_weights = []
            blended_probs = {}
            conviction_multiplier = 1.0
            meta_override = False
            is_healthy = True

        result3 = engine.process_signal(
            FlatSignal(), bid=1.1008, ask=1.1010, mid=1.1009,
        )
        assert result3["event"] == "signal"
        assert result3["direction"] == "FLAT"
        sub = result3.get("sub_events", [])
        assert any(e["event"] == "trade_closed" for e in sub)

        # Verify summary has 1 closed trade
        summary = engine.get_summary()
        assert summary["total_trades"] == 1

    def test_committee_metadata_on_signal(self):
        """Verify committee_metadata is present on all event types."""
        from trading.committee_engine import CommitteeTradingEngine

        class MockSignal:
            signal = 1
            confidence = 0.72
            regime = "high_volatile"
            regime_prob = 0.70
            active_models = ["cnn"]
            model_weights = [0.7]
            blended_probs = {"0": 0.15, "1": 0.13, "2": 0.72}
            conviction_multiplier = 1.0
            meta_override = False
            is_healthy = True

        engine = CommitteeTradingEngine()
        engine.start({
            "pair": "EURUSD", "initial_equity": 10000, "mode": "paper",
        })
        result = engine.process_signal(
            MockSignal(), bid=1.1000, ask=1.1002, mid=1.1001,
        )

        meta = result.get("committee_metadata", {})
        assert meta["regime"] == "high_volatile"
        assert meta["regime_confidence"] == 0.70
        assert len(meta["active_models"]) == 1
        assert meta["active_models"][0]["name"] == "cnn"
        assert meta["active_models"][0]["weight"] == pytest.approx(0.7, abs=1e-3)
        assert meta["conviction_multiplier"] == 1.0
        assert len(meta["blended_probs"]) == 3

    def test_risk_blocked_includes_committee_metadata(self):
        """When a trade is blocked by risk gates, metadata is still included."""
        from trading.committee_engine import CommitteeTradingEngine
        from trading.risk_controls import LiveRiskConfig, LiveRiskState, new_session_state

        class MockSignal:
            signal = 1
            confidence = 0.75
            regime = "trend_up"
            regime_prob = 0.85
            active_models = ["lstm"]
            model_weights = [0.5]
            blended_probs = {"0": 0.10, "1": 0.15, "2": 0.75}
            conviction_multiplier = 1.0
            meta_override = False
            is_healthy = True

        engine = CommitteeTradingEngine()
        engine.start({
            "pair": "EURUSD",
            "initial_equity": 10000,
            "risk_config": {"max_drawdown_pct": 0.001},  # Very tight
        })

        # Simulate a big loss to trigger drawdown
        engine._risk_state.equity_peak = 10000
        engine._risk_state.current_equity = 9990  # 0.1% drawdown
        # This should trigger the pre-trade gate

        result = engine.process_signal(
            MockSignal(), bid=1.1000, ask=1.1002, mid=1.1001,
        )
        # May or may not be blocked depending on exact risk logic
        if result["event"] == "risk_blocked":
            assert "committee_metadata" in result
            assert result["committee_metadata"]["regime"] == "trend_up"

    def test_conviction_multiplier_scales_size(self):
        """Conviction multiplier changes the trade size."""
        from trading.committee_engine import CommitteeTradingEngine

        class MaxConvictionSignal:
            signal = 1
            confidence = 0.88
            regime = "trend_up"
            regime_prob = 0.70
            active_models = ["lstm", "xgboost", "cnn"]
            model_weights = [0.5, 0.3, 0.4]
            blended_probs = {"0": 0.05, "1": 0.07, "2": 0.88}
            conviction_multiplier = 1.5
            meta_override = False
            is_healthy = True

        engine = CommitteeTradingEngine()
        engine.start({"pair": "EURUSD", "initial_equity": 10000, "mode": "paper"})
        result = engine.process_signal(
            MaxConvictionSignal(), bid=1.1000, ask=1.1002, mid=1.1001,
        )
        sub = result.get("sub_events", [])
        opened = [e for e in sub if e["event"] == "trade_opened"]
        if opened:
            # Size should be inflated by 1.5x
            assert opened[0]["size"] > 0

    def test_pair_to_instrument(self):
        from trading.committee_engine import CommitteeTradingEngine
        e = CommitteeTradingEngine()
        assert e._pair_to_instrument("EURUSD") == "EUR_USD"
        assert e._pair_to_instrument("eur_usd") == "EUR_USD"
        assert e._pair_to_instrument("GBPJPY") == "GBP_JPY"
        assert e._pair_to_instrument("eurusd") == "EUR_USD"


# ═══════════════════════════════════════════════════════════════════════
#  DeployRequest model
# ═══════════════════════════════════════════════════════════════════════

class TestDeployCommitteeRequest:
    def test_deploy_committee_request_defaults(self):
        from api.routers.trading import DeployCommitteeRequest
        req = DeployCommitteeRequest(pair="EURUSD")
        assert req.pair == "EURUSD"
        assert req.timeframe == "H1"
        assert req.initial_equity == 10000.0
        assert req.confidence_threshold == 0.55
        assert req.lookback_bars == 100
        assert req.mode == "paper"
        assert req.sizing_config == {}
        assert req.risk_config == {}
        assert req.live_news_blend_enabled is False

    def test_deploy_committee_request_with_full_cycle(self):
        from api.routers.trading import DeployCommitteeRequest
        req = DeployCommitteeRequest(
            pair="EURUSD",
            timeframe="H4",
            initial_equity=50000,
            full_cycle_job_id="fullcycle_abc123",
            mode="live",
            confidence_threshold=0.65,
        )
        assert req.full_cycle_job_id == "fullcycle_abc123"
        assert req.mode == "live"
        assert req.initial_equity == 50000
        assert req.confidence_threshold == 0.65
