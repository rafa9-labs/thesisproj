"""Tests for trading/risk_controls.py — all 4 layers of risk gates."""
import time
import pytest
from collections import deque


@pytest.fixture
def default_config():
    from trading.risk_controls import LiveRiskConfig
    return LiveRiskConfig(initial_equity=10000.0)


@pytest.fixture
def fresh_state(default_config):
    from trading.risk_controls import new_session_state
    return new_session_state(default_config)


@pytest.fixture
def long_signal():
    return {"direction": "LONG", "confidence": 85.0}


@pytest.fixture
def weak_signal():
    return {"direction": "LONG", "confidence": 55.0}


# ═══════════════════════════════════════
#  Layer 1 — Pre-Trade Gates
# ═══════════════════════════════════════

class TestLayer1Gates:
    def test_g1_position_size_allowed(self, fresh_state, default_config):
        from trading.risk_controls import gate_max_position_size
        allowed, reason = gate_max_position_size(fresh_state, default_config, 2000.0)
        assert allowed is True
        assert reason == ""

    def test_g1_position_size_blocked(self, fresh_state, default_config):
        from trading.risk_controls import gate_max_position_size
        allowed, reason = gate_max_position_size(fresh_state, default_config, 5000.0)
        assert allowed is False
        assert "25pct" in reason

    def test_g2_daily_trades_allowed(self, fresh_state, default_config):
        from trading.risk_controls import gate_max_daily_trades
        allowed, _ = gate_max_daily_trades(fresh_state, default_config)
        assert allowed is True

    def test_g2_daily_trades_blocked(self, fresh_state, default_config):
        from trading.risk_controls import gate_max_daily_trades
        fresh_state.daily_trades = 25
        allowed, reason = gate_max_daily_trades(fresh_state, default_config)
        assert allowed is False
        assert "20" in reason

    def test_g3_hourly_trades_allowed(self, fresh_state, default_config):
        from trading.risk_controls import gate_max_hourly_trades
        allowed, _ = gate_max_hourly_trades(fresh_state, default_config)
        assert allowed is True

    def test_g3_hourly_trades_blocked(self, fresh_state, default_config):
        from trading.risk_controls import gate_max_hourly_trades
        fresh_state.hourly_trades = 8
        allowed, reason = gate_max_hourly_trades(fresh_state, default_config)
        assert allowed is False
        assert "5" in reason

    def test_g4_confidence_allowed(self, fresh_state, default_config, long_signal):
        from trading.risk_controls import gate_min_confidence
        allowed, _ = gate_min_confidence(fresh_state, default_config, long_signal["confidence"])
        assert allowed is True

    def test_g4_confidence_blocked(self, fresh_state, default_config, weak_signal):
        from trading.risk_controls import gate_min_confidence
        allowed, reason = gate_min_confidence(fresh_state, default_config, weak_signal["confidence"])
        assert allowed is False
        assert "65" in reason
        assert fresh_state.confidence_rejections == 1

    def test_g5_cooldown_first_trade(self, fresh_state, default_config):
        from trading.risk_controls import gate_trade_cooldown
        allowed, _ = gate_trade_cooldown(fresh_state, default_config, "LONG", "FLAT")
        assert allowed is True

    def test_g5_cooldown_same_direction_too_soon(self, fresh_state, default_config):
        from trading.risk_controls import gate_trade_cooldown
        fresh_state.last_trade_time = time.time() - 30
        allowed, reason = gate_trade_cooldown(fresh_state, default_config, "LONG", "LONG")
        assert allowed is False
        assert "cooldown" in reason

    def test_g5_cooldown_reversal_too_soon(self, fresh_state, default_config):
        from trading.risk_controls import gate_trade_cooldown
        fresh_state.last_trade_time = time.time() - 30
        allowed, reason = gate_trade_cooldown(fresh_state, default_config, "SHORT", "LONG")
        assert allowed is False
        assert "reversal" in reason

    def test_g6_schedule_weekday_allowed(self, default_config):
        from trading.risk_controls import gate_trading_schedule
        default_config.restrict_weekend = True
        default_config.weekend_close_utc_hour = 21
        default_config.weekend_close_utc_day = 4
        default_config.weekend_open_utc_hour = 21
        default_config.weekend_open_utc_day = 6
        allowed, _ = gate_trading_schedule(default_config)
        assert allowed is True

    def test_g7_margin_allowed(self, fresh_state, default_config):
        from trading.risk_controls import gate_margin_check
        allowed, _ = gate_margin_check(fresh_state, default_config, 0.30)
        assert allowed is True

    def test_g7_margin_blocked(self, fresh_state, default_config):
        from trading.risk_controls import gate_margin_check
        allowed, reason = gate_margin_check(fresh_state, default_config, 0.60)
        assert allowed is False
        assert "margin" in reason

    def test_all_pre_trade_gates_pass(self, fresh_state, default_config, long_signal):
        from trading.risk_controls import check_all_pre_trade_gates
        passed, _, blocked = check_all_pre_trade_gates(
            fresh_state, default_config, long_signal, 1000.0, "FLAT", 0.0
        )
        assert passed is True
        assert blocked == []

    def test_all_pre_trade_gates_blocked_by_confidence(self, fresh_state, default_config, weak_signal):
        from trading.risk_controls import check_all_pre_trade_gates
        passed, first_reason, blocked = check_all_pre_trade_gates(
            fresh_state, default_config, weak_signal, 1000.0, "FLAT", 0.0
        )
        assert passed is False
        assert "G4" in first_reason
        assert len(blocked) == 1

    def test_disabled_config_bypasses_all(self, fresh_state, default_config, weak_signal):
        from trading.risk_controls import check_all_pre_trade_gates
        default_config.enabled = False
        passed, _, _ = check_all_pre_trade_gates(
            fresh_state, default_config, weak_signal, 5000.0, "LONG", 0.9
        )
        assert passed is True


# ═══════════════════════════════════════
#  Layer 2 — Post-Trade Monitoring
# ═══════════════════════════════════════

class TestLayer2PostTrade:
    def test_drawdown_not_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_drawdown_kill
        fresh_state.current_equity = 9500
        triggered, _ = check_drawdown_kill(fresh_state, default_config)
        assert triggered is False

    def test_drawdown_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_drawdown_kill
        fresh_state.current_equity = 8000
        triggered, reason = check_drawdown_kill(fresh_state, default_config)
        assert triggered is True
        assert "drawdown" in reason

    def test_daily_loss_not_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_daily_loss
        triggered, _ = check_daily_loss(fresh_state, default_config)
        assert triggered is False

    def test_daily_loss_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_daily_loss
        fresh_state.current_equity = 9000
        triggered, reason = check_daily_loss(fresh_state, default_config)
        assert triggered is True
        assert "daily_loss" in reason

    def test_consecutive_losses_not_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_consecutive_losses
        fresh_state.consecutive_losses = 3
        triggered, _ = check_consecutive_losses(fresh_state, default_config)
        assert triggered is False

    def test_consecutive_losses_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_consecutive_losses
        fresh_state.consecutive_losses = 8
        triggered, reason = check_consecutive_losses(fresh_state, default_config)
        assert triggered is True
        assert "consecutive" in reason

    def test_rolling_sharpe_empty_not_triggered(self, fresh_state, default_config):
        from trading.risk_controls import check_rolling_sharpe
        triggered, _ = check_rolling_sharpe(fresh_state, default_config)
        assert triggered is False

    def test_rolling_sharpe_below_min(self, fresh_state, default_config):
        from trading.risk_controls import check_rolling_sharpe
        for _ in range(25):
            fresh_state.trade_pnl_window.append(-5.0)
        triggered, reason = check_rolling_sharpe(fresh_state, default_config)
        assert triggered is True
        assert "sharpe" in reason

    def test_win_rate_floor_below_min(self, fresh_state, default_config):
        from trading.risk_controls import check_win_rate_floor
        pnls = [-5, -3, -2, -4, -1, -6, -2, -3, -5, -4,
                -2, -1, -3, -5, -2, -4, -1, -3, -2, -5]
        for p in pnls:
            fresh_state.trade_pnl_window.append(p)
        triggered, reason = check_win_rate_floor(fresh_state, default_config)
        assert triggered is True
        assert "win_rate" in reason

    def test_profit_factor_below_min(self, fresh_state, default_config):
        from trading.risk_controls import check_profit_factor
        pnls = [-5, -3, -5, -4, 1, -6, -2, -3, -5, -4,
                -3, -1, -3, -5, -2, -4, -1, -3, -2, -5]
        for p in pnls:
            fresh_state.trade_pnl_window.append(p)
        triggered, reason = check_profit_factor(fresh_state, default_config)
        assert triggered is True
        assert "profit_factor" in reason

    def test_all_post_trade_gates_pass_on_win(self, fresh_state, default_config):
        from trading.risk_controls import check_all_post_trade_gates
        killed, reason, level = check_all_post_trade_gates(
            fresh_state, default_config, 50.0, True
        )
        assert killed is False
        assert fresh_state.consecutive_losses == 0

    def test_all_post_trade_gates_consec_loss_update(self, fresh_state, default_config):
        from trading.risk_controls import check_all_post_trade_gates
        for _ in range(3):
            check_all_post_trade_gates(fresh_state, default_config, -10.0, False)
        assert fresh_state.consecutive_losses == 3


# ═══════════════════════════════════════
#  Layer 3 — Infrastructure Guards
# ═══════════════════════════════════════

class TestLayer3Infra:
    def test_signal_staleness_allowed(self, fresh_state, default_config):
        from trading.risk_controls import check_signal_staleness
        allowed, _ = check_signal_staleness(fresh_state, default_config, time.time())
        assert allowed is True

    def test_signal_staleness_blocked(self, fresh_state, default_config):
        from trading.risk_controls import check_signal_staleness
        allowed, reason = check_signal_staleness(fresh_state, default_config, time.time() - 60)
        assert allowed is False
        assert "stale" in reason

    def test_heartbeat_allowed(self, fresh_state, default_config):
        from trading.risk_controls import check_heartbeat
        fresh_state.last_heartbeat = time.time()
        triggered, _ = check_heartbeat(fresh_state, default_config)
        assert triggered is False

    def test_heartbeat_lost(self, fresh_state, default_config):
        from trading.risk_controls import check_heartbeat
        fresh_state.last_heartbeat = time.time() - 400
        triggered, reason = check_heartbeat(fresh_state, default_config)
        assert triggered is True
        assert "heartbeat" in reason

    def test_session_lifetime_allowed(self, fresh_state, default_config):
        from trading.risk_controls import check_session_lifetime
        fresh_state.session_start = time.time()
        allowed, _ = check_session_lifetime(fresh_state, default_config)
        assert allowed is True

    def test_consecutive_api_errors(self, fresh_state, default_config):
        from trading.risk_controls import check_consecutive_api_errors
        fresh_state.consecutive_api_errors = 8
        triggered, reason = check_consecutive_api_errors(fresh_state, default_config)
        assert triggered is True
        assert "api" in reason

    def test_min_equity_allowed(self, fresh_state, default_config):
        from trading.risk_controls import check_min_equity
        triggered, _ = check_min_equity(fresh_state, default_config)
        assert triggered is False

    def test_min_equity_breached(self, fresh_state, default_config):
        from trading.risk_controls import check_min_equity
        fresh_state.current_equity = 3000
        triggered, reason = check_min_equity(fresh_state, default_config)
        assert triggered is True
        assert "equity" in reason


# ═══════════════════════════════════════
#  Layer 4 — Emergency Kill Switch
# ═══════════════════════════════════════

class TestLayer4Kill:
    def test_manual_kill(self, fresh_state):
        from trading.risk_controls import manual_kill, is_killed
        manual_kill(fresh_state, "test")
        assert is_killed(fresh_state) is True
        assert fresh_state.kill_level == "K1"

    def test_equity_floor_auto_kill(self, fresh_state, default_config):
        from trading.risk_controls import auto_kill_equity_floor
        fresh_state.current_equity = 2000
        result = auto_kill_equity_floor(fresh_state, default_config)
        assert result is True
        assert fresh_state.kill_level == "K3"

    def test_equity_floor_not_triggered(self, fresh_state, default_config):
        from trading.risk_controls import auto_kill_equity_floor
        result = auto_kill_equity_floor(fresh_state, default_config)
        assert result is False


# ═══════════════════════════════════════
#  State Management
# ═══════════════════════════════════════

class TestStateManagement:
    def test_new_session_state_initial_values(self, default_config):
        from trading.risk_controls import new_session_state
        state = new_session_state(default_config)
        assert state.equity_peak == 10000
        assert state.current_equity == 10000
        assert state.consecutive_losses == 0
        assert state.daily_trades == 0

    def test_record_api_error(self, fresh_state):
        from trading.risk_controls import record_api_error
        record_api_error(fresh_state)
        record_api_error(fresh_state)
        assert fresh_state.consecutive_api_errors == 2

    def test_record_api_success_resets(self, fresh_state):
        from trading.risk_controls import record_api_error, record_api_success
        record_api_error(fresh_state)
        record_api_error(fresh_state)
        record_api_success(fresh_state)
        assert fresh_state.consecutive_api_errors == 0

    def test_record_trade_submission(self, fresh_state):
        from trading.risk_controls import record_trade_submission
        record_trade_submission(fresh_state, "LONG")
        assert fresh_state.daily_trades == 1
        assert fresh_state.hourly_trades == 1

    def test_reset_daily_counters(self, fresh_state):
        from trading.risk_controls import record_trade_submission, reset_daily_counters
        record_trade_submission(fresh_state, "LONG")
        record_trade_submission(fresh_state, "SHORT")
        assert fresh_state.daily_trades == 2
        reset_daily_counters(fresh_state)
        assert fresh_state.daily_trades == 0
        assert fresh_state.hourly_trades == 0


# ═══════════════════════════════════════
#  API Schemas
# ═══════════════════════════════════════

class TestLiveAPISchemas:
    def test_deploy_live_request_defaults(self):
        from api.routers.trading import DeployLiveRequest
        req = DeployLiveRequest(pair="EURUSD")
        assert req.pair == "EURUSD"
        assert req.mode == "demo"
        assert req.risk_config == {}

    def test_live_session_info(self):
        from api.routers.trading import LiveSessionInfo
        info = LiveSessionInfo(
            session_id="live1",
            pair="EURUSD",
            model_type="xgboost",
            timeframe="H1",
            mode="demo",
            status="running",
            equity=10500,
            position="LONG",
            unrealized_pnl=50.0,
            signal_count=10,
            killed=False,
            kill_reason="",
        )
        assert info.session_id == "live1"
        assert info.mode == "demo"
        assert info.killed is False
