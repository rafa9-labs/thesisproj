"""
Tests for pipeline.execution.risk_manager — S2.4 Risk Management Framework.

Covers:
  - Drawdown breaker: trigger threshold, session_end resume, cooloff resume
  - Daily loss limit: pct mode, sigma mode, session_end resume
  - Consecutive losses: trigger after N losses, reset on win
  - should_suppress_entry: True when any safeguard paused
  - reset_daily: clears session-based pauses
  - tick_cooloffs: decrement and auto-resume
  - Multiple safeguards active simultaneously
  - get_pause_reason: correct comma-separated output
"""
import pytest

from pipeline.execution.risk_manager import (
    RiskConfig,
    RiskState,
    should_suppress_entry,
    check_drawdown,
    check_daily_loss,
    update_after_trade,
    tick_cooloffs,
    reset_daily,
    get_pause_reason,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dd_cfg(**overrides):
    defaults = dict(
        risk_use_dd_breaker=True,
        risk_max_drawdown_pct=0.20,
        risk_dd_resume="session_end",
        risk_dd_cooloff_bars=48,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _daily_cfg(**overrides):
    defaults = dict(
        risk_use_daily_loss=True,
        risk_max_daily_loss_pct=0.03,
        risk_max_daily_loss_sigma=3.0,
        risk_daily_loss_mode="pct",
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _consec_cfg(**overrides):
    defaults = dict(
        risk_use_consec_loss=True,
        risk_max_consecutive_losses=3,
        risk_consec_resume="session_end",
        risk_consec_cooloff_bars=10,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _fresh_state(equity_peak=10_000.0):
    return RiskState(equity_peak=equity_peak)


# ===================================================================
# Drawdown breaker
# ===================================================================

class TestDrawdownBreaker:

    def test_no_trigger_below_threshold(self):
        cfg = _dd_cfg(risk_max_drawdown_pct=0.20)
        state = _fresh_state()
        equity = 9_000.0
        assert check_drawdown(cfg, state, equity) is False
        assert not state.dd_paused

    def test_triggers_at_threshold(self):
        cfg = _dd_cfg(risk_max_drawdown_pct=0.20)
        state = _fresh_state()
        equity = 7_900.0
        assert check_drawdown(cfg, state, equity) is True
        assert state.dd_paused
        assert state.dd_breaches == 1

    def test_no_double_trigger(self):
        cfg = _dd_cfg(risk_max_drawdown_pct=0.20)
        state = _fresh_state()
        check_drawdown(cfg, state, 7_900.0)
        assert check_drawdown(cfg, state, 7_000.0) is False
        assert state.dd_breaches == 1

    def test_equity_peak_updates(self):
        cfg = _dd_cfg(risk_max_drawdown_pct=0.20)
        state = _fresh_state()
        check_drawdown(cfg, state, 11_000.0)
        assert state.equity_peak == 11_000.0
        check_drawdown(cfg, state, 12_000.0)
        assert state.equity_peak == 12_000.0

    def test_disabled(self):
        cfg = RiskConfig(risk_use_dd_breaker=False)
        state = _fresh_state()
        assert check_drawdown(cfg, state, 1.0) is False
        assert not state.dd_paused

    def test_cooloff_mode_sets_counter(self):
        cfg = _dd_cfg(risk_dd_resume="cooloff", risk_dd_cooloff_bars=20)
        state = _fresh_state()
        check_drawdown(cfg, state, 7_900.0)
        assert state.dd_paused
        assert state.dd_cooloff_remaining == 20

    def test_session_end_mode_no_cooloff_counter(self):
        cfg = _dd_cfg(risk_dd_resume="session_end")
        state = _fresh_state()
        check_drawdown(cfg, state, 7_900.0)
        assert state.dd_paused
        assert state.dd_cooloff_remaining == 0

    def test_zero_equity_peak_no_crash(self):
        cfg = _dd_cfg()
        state = RiskState(equity_peak=0.0)
        assert check_drawdown(cfg, state, 100.0) is False


# ===================================================================
# Daily loss limit
# ===================================================================

class TestDailyLossLimit:

    def test_pct_mode_trigger(self):
        cfg = _daily_cfg(risk_max_daily_loss_pct=0.03, risk_daily_loss_mode="pct")
        state = _fresh_state()
        bar_pnl = -0.04
        assert check_daily_loss(cfg, state, bar_pnl) is True
        assert state.daily_paused
        assert state.daily_loss_breaches == 1

    def test_pct_mode_no_trigger(self):
        cfg = _daily_cfg(risk_max_daily_loss_pct=0.03, risk_daily_loss_mode="pct")
        state = _fresh_state()
        bar_pnl = -0.01
        assert check_daily_loss(cfg, state, bar_pnl) is False
        assert not state.daily_paused

    def test_sigma_mode_trigger(self):
        cfg = _daily_cfg(risk_daily_loss_mode="sigma", risk_max_daily_loss_sigma=2.0)
        state = _fresh_state()
        bar_vol = 0.001
        bars_per_day = 48
        day_sigma = bar_vol * (bars_per_day ** 0.5)
        loss_limit = 2.0 * day_sigma
        bar_pnl = -(loss_limit + 0.001)
        assert check_daily_loss(cfg, state, bar_pnl, bar_vol, bars_per_day) is True
        assert state.daily_paused

    def test_sigma_mode_no_trigger(self):
        cfg = _daily_cfg(risk_daily_loss_mode="sigma", risk_max_daily_loss_sigma=3.0)
        state = _fresh_state()
        assert check_daily_loss(cfg, state, -0.001, 0.001, 48) is False
        assert not state.daily_paused

    def test_cumulative_daily_pnl(self):
        cfg = _daily_cfg(risk_max_daily_loss_pct=0.03, risk_daily_loss_mode="pct")
        state = _fresh_state()
        check_daily_loss(cfg, state, -0.015)
        assert not state.daily_paused
        check_daily_loss(cfg, state, -0.02)
        assert state.daily_paused
        assert state.daily_loss_breaches == 1

    def test_disabled(self):
        cfg = RiskConfig(risk_use_daily_loss=False)
        state = _fresh_state()
        assert check_daily_loss(cfg, state, -9999.0) is False

    def test_no_double_trigger(self):
        cfg = _daily_cfg(risk_max_daily_loss_pct=0.03, risk_daily_loss_mode="pct")
        state = _fresh_state()
        check_daily_loss(cfg, state, -400.0)
        assert check_daily_loss(cfg, state, -100.0) is False
        assert state.daily_loss_breaches == 1

    def test_already_paused_does_not_retrigger(self):
        cfg = _daily_cfg()
        state = _fresh_state()
        state.daily_paused = True
        assert check_daily_loss(cfg, state, -500.0) is False


# ===================================================================
# Consecutive losses
# ===================================================================

class TestConsecutiveLosses:

    def test_trigger_after_n_losses(self):
        cfg = _consec_cfg(risk_max_consecutive_losses=3)
        state = _fresh_state()
        update_after_trade(cfg, state, -10.0, False)
        assert not state.consec_paused
        update_after_trade(cfg, state, -10.0, False)
        assert not state.consec_paused
        update_after_trade(cfg, state, -10.0, False)
        assert state.consec_paused
        assert state.consec_loss_breaches == 1

    def test_win_resets_counter(self):
        cfg = _consec_cfg(risk_max_consecutive_losses=3)
        state = _fresh_state()
        update_after_trade(cfg, state, -10.0, False)
        update_after_trade(cfg, state, -10.0, False)
        update_after_trade(cfg, state, 10.0, True)
        assert state.consecutive_losses == 0
        assert not state.consec_paused

    def test_cooloff_mode(self):
        cfg = _consec_cfg(
            risk_max_consecutive_losses=2,
            risk_consec_resume="cooloff",
            risk_consec_cooloff_bars=15,
        )
        state = _fresh_state()
        update_after_trade(cfg, state, -10.0, False)
        update_after_trade(cfg, state, -10.0, False)
        assert state.consec_paused
        assert state.consec_cooloff_remaining == 15

    def test_disabled(self):
        cfg = RiskConfig(risk_use_consec_loss=False)
        state = _fresh_state()
        assert update_after_trade(cfg, state, -10.0, False) is False
        assert state.consecutive_losses == 0

    def test_no_double_trigger(self):
        cfg = _consec_cfg(risk_max_consecutive_losses=2)
        state = _fresh_state()
        update_after_trade(cfg, state, -10.0, False)
        update_after_trade(cfg, state, -10.0, False)
        assert state.consec_loss_breaches == 1
        update_after_trade(cfg, state, -10.0, False)
        assert state.consec_loss_breaches == 1

    def test_tracks_total_trades_and_wins(self):
        cfg = _consec_cfg(risk_max_consecutive_losses=100)
        state = _fresh_state()
        update_after_trade(cfg, state, 5.0, True)
        update_after_trade(cfg, state, -3.0, False)
        update_after_trade(cfg, state, 7.0, True)
        assert state.total_trades == 3
        assert state.total_wins == 2
        assert state.consecutive_losses == 0


# ===================================================================
# should_suppress_entry
# ===================================================================

class TestShouldSuppressEntry:

    def test_none_active(self):
        state = _fresh_state()
        assert should_suppress_entry(state) is False

    def test_dd_paused(self):
        state = _fresh_state()
        state.dd_paused = True
        assert should_suppress_entry(state) is True

    def test_daily_paused(self):
        state = _fresh_state()
        state.daily_paused = True
        assert should_suppress_entry(state) is True

    def test_consec_paused(self):
        state = _fresh_state()
        state.consec_paused = True
        assert should_suppress_entry(state) is True

    def test_multiple_paused(self):
        state = _fresh_state()
        state.dd_paused = True
        state.consec_paused = True
        assert should_suppress_entry(state) is True


# ===================================================================
# tick_cooloffs
# ===================================================================

class TestTickCooloffs:

    def test_decrement_dd_cooloff(self):
        cfg = _dd_cfg(risk_dd_resume="cooloff", risk_dd_cooloff_bars=3)
        state = _fresh_state()
        state.dd_paused = True
        state.dd_cooloff_remaining = 3
        tick_cooloffs(cfg, state)
        assert state.dd_cooloff_remaining == 2
        assert state.dd_paused
        assert state.bars_paused == 1

    def test_auto_resume_dd_at_zero(self):
        cfg = _dd_cfg(risk_dd_resume="cooloff", risk_dd_cooloff_bars=1)
        state = _fresh_state()
        state.dd_paused = True
        state.dd_cooloff_remaining = 1
        tick_cooloffs(cfg, state)
        assert not state.dd_paused
        assert state.dd_cooloff_remaining == 0

    def test_decrement_consec_cooloff(self):
        cfg = _consec_cfg(risk_consec_resume="cooloff", risk_consec_cooloff_bars=5)
        state = _fresh_state()
        state.consec_paused = True
        state.consec_cooloff_remaining = 5
        tick_cooloffs(cfg, state)
        assert state.consec_cooloff_remaining == 4
        assert state.consec_paused

    def test_auto_resume_consec_at_zero(self):
        cfg = _consec_cfg(risk_consec_resume="cooloff", risk_consec_cooloff_bars=1)
        state = _fresh_state()
        state.consec_paused = True
        state.consec_cooloff_remaining = 1
        tick_cooloffs(cfg, state)
        assert not state.consec_paused

    def test_session_end_mode_no_decrement(self):
        cfg = _dd_cfg(risk_dd_resume="session_end")
        state = _fresh_state()
        state.dd_paused = True
        state.dd_cooloff_remaining = 0
        tick_cooloffs(cfg, state)
        assert state.dd_paused

    def test_no_increment_when_nothing_paused(self):
        cfg = RiskConfig()
        state = _fresh_state()
        tick_cooloffs(cfg, state)
        assert state.bars_paused == 0


# ===================================================================
# reset_daily
# ===================================================================

class TestResetDaily:

    def test_clears_daily_pnl(self):
        state = _fresh_state()
        state.daily_pnl = -500.0
        reset_daily(state)
        assert state.daily_pnl == 0.0

    def test_unpauses_daily(self):
        state = _fresh_state()
        state.daily_paused = True
        reset_daily(state)
        assert not state.daily_paused

    def test_unpauses_dd_session_end(self):
        state = _fresh_state()
        state.dd_paused = True
        reset_daily(state)
        assert not state.dd_paused

    def test_unpauses_consec_session_end(self):
        state = _fresh_state()
        state.consec_paused = True
        reset_daily(state)
        assert not state.consec_paused

    def test_clears_cooloff_counters(self):
        state = _fresh_state()
        state.dd_cooloff_remaining = 10
        state.consec_cooloff_remaining = 5
        reset_daily(state)
        assert state.dd_cooloff_remaining == 0
        assert state.consec_cooloff_remaining == 0

    def test_preserves_equity_peak(self):
        state = _fresh_state()
        state.equity_peak = 15_000.0
        reset_daily(state)
        assert state.equity_peak == 15_000.0

    def test_preserves_breach_counters(self):
        state = _fresh_state()
        state.dd_breaches = 3
        state.daily_loss_breaches = 2
        state.consec_loss_breaches = 1
        reset_daily(state)
        assert state.dd_breaches == 3
        assert state.daily_loss_breaches == 2
        assert state.consec_loss_breaches == 1


# ===================================================================
# get_pause_reason
# ===================================================================

class TestGetPauseReason:

    def test_no_reason(self):
        state = _fresh_state()
        assert get_pause_reason(state) == ""

    def test_dd_only(self):
        state = _fresh_state()
        state.dd_paused = True
        assert get_pause_reason(state) == "drawdown"

    def test_daily_only(self):
        state = _fresh_state()
        state.daily_paused = True
        assert get_pause_reason(state) == "daily_loss"

    def test_consec_only(self):
        state = _fresh_state()
        state.consec_paused = True
        assert get_pause_reason(state) == "consecutive_loss"

    def test_multiple(self):
        state = _fresh_state()
        state.dd_paused = True
        state.consec_paused = True
        reasons = get_pause_reason(state)
        assert "drawdown" in reasons
        assert "consecutive_loss" in reasons


# ===================================================================
# Integration: multiple safeguards simultaneously
# ===================================================================

class TestMultipleSafeguards:

    def test_all_three_active(self):
        cfg = RiskConfig(
            risk_use_dd_breaker=True,
            risk_max_drawdown_pct=0.10,
            risk_use_daily_loss=True,
            risk_max_daily_loss_pct=0.02,
            risk_daily_loss_mode="pct",
            risk_use_consec_loss=True,
            risk_max_consecutive_losses=2,
            risk_initial_equity=10_000.0,
        )
        state = RiskState(equity_peak=10_000.0)

        update_after_trade(cfg, state, -10.0, False)
        update_after_trade(cfg, state, -10.0, False)
        assert state.consec_paused

        check_daily_loss(cfg, state, -300.0)
        assert state.daily_paused

        check_drawdown(cfg, state, 8_500.0)
        assert state.dd_paused

        assert should_suppress_entry(state) is True
        reasons = get_pause_reason(state)
        assert "drawdown" in reasons
        assert "daily_loss" in reasons
        assert "consecutive_loss" in reasons

    def test_reset_daily_clears_all_session_pauses(self):
        cfg = RiskConfig(
            risk_use_dd_breaker=True,
            risk_max_drawdown_pct=0.10,
            risk_use_daily_loss=True,
            risk_max_daily_loss_pct=0.01,
            risk_daily_loss_mode="pct",
            risk_use_consec_loss=True,
            risk_max_consecutive_losses=2,
        )
        state = RiskState(equity_peak=10_000.0)
        state.dd_paused = True
        state.daily_paused = True
        state.consec_paused = True

        reset_daily(state)
        assert not state.dd_paused
        assert not state.daily_paused
        assert not state.consec_paused
        assert should_suppress_entry(state) is False

    def test_cooloff_countdown_independent_of_reset(self):
        cfg = RiskConfig(
            risk_use_dd_breaker=True,
            risk_max_drawdown_pct=0.10,
            risk_dd_resume="cooloff",
            risk_dd_cooloff_bars=3,
            risk_use_consec_loss=True,
            risk_max_consecutive_losses=2,
            risk_consec_resume="cooloff",
            risk_consec_cooloff_bars=5,
        )
        state = RiskState(equity_peak=10_000.0)
        state.dd_paused = True
        state.dd_cooloff_remaining = 3
        state.consec_paused = True
        state.consec_cooloff_remaining = 5

        reset_daily(state)
        assert not state.dd_paused
        assert state.dd_cooloff_remaining == 0
        assert not state.consec_paused
        assert state.consec_cooloff_remaining == 0

    def test_cooloff_full_countdown_then_resume(self):
        cfg = _dd_cfg(risk_dd_resume="cooloff", risk_dd_cooloff_bars=2)
        state = _fresh_state()
        state.dd_paused = True
        state.dd_cooloff_remaining = 2

        tick_cooloffs(cfg, state)
        assert state.dd_paused
        assert state.bars_paused == 1

        tick_cooloffs(cfg, state)
        assert not state.dd_paused
        assert state.bars_paused == 2

        tick_cooloffs(cfg, state)
        assert state.bars_paused == 2


# ===================================================================
# Import test
# ===================================================================

def test_pipeline_execution_package_reexports():
    from pipeline.execution import (
        RiskConfig,
        RiskState,
        should_suppress_entry,
        get_pause_reason,
        check_drawdown,
        check_daily_loss,
        update_after_trade,
        tick_cooloffs,
        reset_daily,
    )
    assert RiskConfig is not None
    assert RiskState is not None
