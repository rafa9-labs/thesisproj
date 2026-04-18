"""Regression tests for execution engine bug fixes.

Each test targets a specific bug identified in the edge-case audit
and verifies the fix works correctly.

Sprint A fixes:
  E7.1 — Trailing indentation: position state no longer zeroed every bar
  E7.6/E1.4 — Equity not double-counted between bar-by-bar and trade-close
  E7.7 — Array length mismatch raises ValueError
  E2.1 — pip_value=0 does not cause immediate stop-out
  E3.1/E3.2 — Trailing HWM/LWM initialized from entry price
  E4.1 — reset_daily respects cooloff resume mode
  E4.2 — Sigma-mode daily loss skips when bar_vol=0 (no silent fallback to pct)
  E1.1 — Negative equity returns zero size
  E1.2 — Kelly criterion capped by equity ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.backtester.execution_patches import PatchConfig, LoopResult, run_execution_loop
from pipeline.execution.position_sizing import (
    SizingState, SizingConfig, SizingMethod, compute_size, update_state,
)
from pipeline.execution.stops import StopConfig, StopMethod, compute_stop_levels
from pipeline.execution.trailing import TrailingConfig, TrailingState, TrailingMethod
from pipeline.execution.risk_manager import (
    RiskConfig, RiskState, reset_daily, check_daily_loss,
)


def _make_inputs(n=500):
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
    high = close + np.abs(np.random.randn(n)) * 0.0005
    low = close - np.abs(np.random.randn(n)) * 0.0005
    df = pd.DataFrame({
        "price": close, "high": high, "low": low, "close": close,
        "spread": np.full(n, 0.00015), "atr_14": np.full(n, 0.005),
        "returns": np.concatenate([[0.0], np.diff(np.log(close))]),
    }, index=dates)
    np.random.seed(42)
    pred = np.random.choice([-1.0, 0.0, 1.0], size=n)
    rets = df["returns"].values.copy()
    bar_vol = np.full(n, 0.001)
    gap = np.zeros(n, dtype=bool)
    regime = np.zeros(n, dtype=int)
    return df, pred, rets, bar_vol, gap, regime


def _run_loop(cfg=None, n=500, **overrides):
    df, pred, rets, bar_vol, gap, regime = _make_inputs(n)
    if cfg is None:
        cfg = PatchConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return run_execution_loop(
        df, pred, rets, bar_vol, gap, regime,
        cfg=cfg, trading_costs=False, slippage_factor=0.0,
    )


# ── E7.1: Trailing indentation fix ──────────────────────────────────

class TestE71TrailingIndentation:
    def test_trailing_position_not_zeroed_while_holding(self):
        """Position state should NOT be zeroed on bars where we continue holding."""
        np.random.seed(42)
        cfg = PatchConfig(trailing_method="fixed_pips")
        df, pred, rets, bar_vol, gap, regime = _make_inputs(200)
        pred[:] = 1.0
        result = _run_loop(cfg=cfg, n=200)
        nonzero_positions = np.sum(result.pos_actual != 0.0)
        assert nonzero_positions > 50, (
            f"Only {nonzero_positions} non-zero positions — trailing path may be zeroing state"
        )

    def test_trailing_produces_nonzero_strat(self):
        np.random.seed(42)
        result = _run_loop(cfg=PatchConfig(trailing_method="atr"), n=300)
        assert np.any(result.strat != 0.0), "Strategy returns are all zero — trailing broken"


# ── E7.6/E1.4: Equity double-counting fix ───────────────────────────

class TestE76EquityDoubleCount:
    def test_equity_not_double_counted(self):
        """update_state should NOT add trade_pnl to equity (bar-by-bar already does it)."""
        state = SizingState(equity=10000.0)
        update_state(state, trade_pnl=100.0, is_win=True)
        assert state.equity == 10000.0, f"Equity should stay at 10000, got {state.equity}"
        assert state.trade_count == 1
        assert state.win_count == 1
        assert state.total_win == 100.0

    def test_equity_after_multiple_updates(self):
        state = SizingState(equity=10000.0)
        update_state(state, trade_pnl=50.0, is_win=True)
        update_state(state, trade_pnl=-30.0, is_win=False)
        assert state.equity == 10000.0, "Equity must not drift from trade-close updates"
        assert state.trade_count == 2
        assert state.win_count == 1


# ── E7.7: Array length validation ───────────────────────────────────

class TestE77ArrayLengthValidation:
    def test_mismatched_pred_length_raises(self):
        df, _, rets, bar_vol, gap, regime = _make_inputs(100)
        wrong_pred = np.zeros(50)
        with pytest.raises(ValueError, match="Array length mismatch"):
            run_execution_loop(
                df, wrong_pred, rets, bar_vol, gap, regime,
                cfg=PatchConfig(), trading_costs=False, slippage_factor=0.0,
            )

    def test_mismatched_rets_length_raises(self):
        df, pred, _, bar_vol, gap, regime = _make_inputs(100)
        wrong_rets = np.zeros(50)
        with pytest.raises(ValueError, match="Array length mismatch"):
            run_execution_loop(
                df, pred, wrong_rets, bar_vol, gap, regime,
                cfg=PatchConfig(), trading_costs=False, slippage_factor=0.0,
            )

    def test_matching_lengths_succeed(self):
        df, pred, rets, bar_vol, gap, regime = _make_inputs(100)
        result = run_execution_loop(
            df, pred, rets, bar_vol, gap, regime,
            cfg=PatchConfig(), trading_costs=False, slippage_factor=0.0,
        )
        assert isinstance(result, LoopResult)


# ── E2.1: pip_value=0 guard ─────────────────────────────────────────

class TestE21PipValueZero:
    def test_zero_pip_value_returns_empty_levels(self):
        cfg = StopConfig(method=StopMethod.FIXED_PIPS, pip_value=0.0)
        levels = compute_stop_levels(cfg, entry_price=1.1000, direction=1.0)
        assert levels.sl_price == 0.0
        assert levels.tp_price == 0.0

    def test_positive_pip_value_returns_levels(self):
        cfg = StopConfig(method=StopMethod.FIXED_PIPS, pip_value=0.0001)
        levels = compute_stop_levels(cfg, entry_price=1.1000, direction=1.0)
        assert levels.sl_price > 0.0
        assert levels.tp_price > 0.0


# ── E3.1/E3.2: Trailing HWM/LWM initialization ─────────────────────

class TestE31TrailingInit:
    def test_reset_with_entry_price(self):
        state = TrailingState()
        state.reset(lookback=22, entry_price=1.1000)
        assert state.high_water_mark == 1.1000
        assert state.low_water_mark == 1.1000

    def test_reset_without_entry_price(self):
        state = TrailingState()
        state.reset(lookback=22)
        assert state.high_water_mark == 0.0
        assert state.low_water_mark == float("inf")

    def test_hwm_updates_from_initial_price(self):
        state = TrailingState()
        state.reset(lookback=22, entry_price=1.1000)
        cfg = TrailingConfig(method=TrailingMethod.FIXED_PIPS)
        from pipeline.execution.trailing import update_trailing_state
        update_trailing_state(cfg, state, high=1.1050, low=1.0980)
        assert state.high_water_mark == 1.1050
        assert state.low_water_mark == 1.0980


# ── E4.1: reset_daily cooloff contract ──────────────────────────────

class TestE41ResetDailyCooloff:
    def _make_config(self, dd_resume="cooloff", consec_resume="cooloff"):
        return RiskConfig(
            risk_use_dd_breaker=True,
            risk_dd_resume=dd_resume,
            risk_use_consec_loss=True,
            risk_consec_resume=consec_resume,
        )

    def test_cooloff_mode_preserves_pause(self):
        """Cooloff-based pauses should NOT be cleared by reset_daily."""
        config = self._make_config(dd_resume="cooloff")
        state = RiskState()
        state.dd_paused = True
        state.dd_cooloff_remaining = 10
        reset_daily(state, config)
        assert state.dd_paused is True
        assert state.dd_cooloff_remaining == 10

    def test_session_end_mode_clears_pause(self):
        """Session-end pauses SHOULD be cleared by reset_daily."""
        config = self._make_config(dd_resume="session_end")
        state = RiskState()
        state.dd_paused = True
        state.dd_cooloff_remaining = 10
        reset_daily(state, config)
        assert state.dd_paused is False
        assert state.dd_cooloff_remaining == 0

    def test_daily_always_cleared(self):
        config = self._make_config()
        state = RiskState()
        state.daily_paused = True
        state.daily_pnl = -500.0
        reset_daily(state, config)
        assert state.daily_paused is False
        assert state.daily_pnl == 0.0

    def test_no_config_backward_compat(self):
        """Without config, all pauses cleared (backward compatible)."""
        state = RiskState()
        state.dd_paused = True
        state.consec_paused = True
        state.dd_cooloff_remaining = 10
        reset_daily(state, config=None)
        assert state.dd_paused is False
        assert state.consec_paused is False

    def test_mixed_resume_modes(self):
        """DD cooloff preserved, consec session_end cleared."""
        config = self._make_config(dd_resume="cooloff", consec_resume="session_end")
        state = RiskState()
        state.dd_paused = True
        state.dd_cooloff_remaining = 5
        state.consec_paused = True
        state.consec_cooloff_remaining = 3
        reset_daily(state, config)
        assert state.dd_paused is True
        assert state.dd_cooloff_remaining == 5
        assert state.consec_paused is False
        assert state.consec_cooloff_remaining == 0


# ── E4.2: Sigma-mode bar_vol=0 skip ─────────────────────────────────

class TestE42SigmaModeSkip:
    def test_sigma_mode_skips_when_bar_vol_zero(self):
        config = RiskConfig(
            risk_use_daily_loss=True,
            risk_daily_loss_mode="sigma",
            risk_max_daily_loss_sigma=2.0,
        )
        state = RiskState(daily_pnl=-0.05)
        result = check_daily_loss(config, state, bar_pnl=-0.002, bar_vol=0.0, bars_per_day=24)
        assert result is False
        assert state.daily_paused is False

    def test_sigma_mode_triggers_with_valid_bar_vol(self):
        config = RiskConfig(
            risk_use_daily_loss=True,
            risk_daily_loss_mode="sigma",
            risk_max_daily_loss_sigma=2.0,
        )
        state = RiskState(daily_pnl=-1.0)
        result = check_daily_loss(config, state, bar_pnl=-0.1, bar_vol=0.01, bars_per_day=24)
        assert result is True
        assert state.daily_paused is True


# ── E1.1: Negative equity guard ─────────────────────────────────────

class TestE11NegativeEquity:
    def test_negative_equity_returns_zero_size(self):
        state = SizingState(equity=-100.0)
        config = SizingConfig(method=SizingMethod.FIXED_FRACTIONAL, initial_equity=10000.0)
        size = compute_size(state, bar_vol=0.001, atr=0.005, config=config)
        assert size == 0.0

    def test_zero_equity_returns_zero_size(self):
        state = SizingState(equity=0.0)
        config = SizingConfig(method=SizingMethod.KELLY, initial_equity=10000.0)
        size = compute_size(state, bar_vol=0.001, atr=0.005, config=config)
        assert size == 0.0

    def test_positive_equity_returns_nonzero(self):
        state = SizingState(equity=10000.0)
        config = SizingConfig(method=SizingMethod.FIXED_FRACTIONAL, initial_equity=10000.0)
        size = compute_size(state, bar_vol=0.001, atr=0.005, config=config)
        assert size > 0.0


# ── E1.2: Kelly equity cap ──────────────────────────────────────────

class TestE12KellyEquityCap:
    def test_kelly_capped_when_equity_below_initial(self):
        state = SizingState(
            equity=5000.0,
            trade_count=100, win_count=60, total_win=600.0, total_loss=400.0,
        )
        config = SizingConfig(method=SizingMethod.KELLY, kelly_min_trades=10, initial_equity=10000.0)
        size = compute_size(state, bar_vol=0.001, atr=0.005, config=config)
        assert size >= 0.0
        assert size <= 1.0

    def test_kelly_zero_equity(self):
        state = SizingState(
            equity=0.0,
            trade_count=50, win_count=30, total_win=300.0, total_loss=200.0,
        )
        config = SizingConfig(method=SizingMethod.KELLY, kelly_min_trades=10, initial_equity=10000.0)
        size = compute_size(state, bar_vol=0.001, atr=0.005, config=config)
        assert size == 0.0
