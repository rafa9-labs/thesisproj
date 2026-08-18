"""
Risk management framework for the FX ML backtester.

Provides three independent circuit breakers that pause trading when
account-level risk thresholds are breached.  Each safeguard has its own
toggle and configurable resume behaviour.

Safeguards:
  - Drawdown breaker   -- pause when drawdown from equity peak exceeds N%
  - Daily loss limit    -- pause when cumulative daily PnL exceeds a loss threshold
  - Consecutive losses  -- pause after N consecutive losing trades

All safeguards operate at the **account level** (not per-trade) and
coexist independently with the existing kill switch (Patch #5).

Resume behaviour (per safeguard):
  - ``"session_end"`` -- paused until new calendar day / session gap
  - ``"cooloff"``     -- paused for N bars, then auto-resumes

Usage::

    from pipeline.execution.risk_manager import (
        RiskConfig, RiskState,
        check_drawdown, check_daily_loss, update_after_trade,
        should_suppress_entry, tick_cooloffs, reset_daily, get_pause_reason,
    )

    cfg = RiskConfig(
        risk_use_dd_breaker=True, risk_max_drawdown_pct=0.20,
        risk_use_daily_loss=True, risk_max_daily_loss_pct=0.03,
        risk_use_consec_loss=True, risk_max_consecutive_losses=5,
    )
    state = RiskState(equity_peak=10_000.0)

    for i in range(n):
        if new_day:
            reset_daily(state)
        if should_suppress_entry(state):
            tick_cooloffs(cfg, state)
            # suppress signal ...
        # after trade closes:
        update_after_trade(cfg, state, trade_pnl, is_win)
        # after each bar:
        check_drawdown(cfg, state, current_equity)
        check_daily_loss(cfg, state, bar_pnl)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RiskConfig:
    """Configuration for the risk management framework.

    Each safeguard is independently toggleable with its own threshold
    and resume behaviour.
    """

    # --- Drawdown circuit breaker ---
    # Thresholds mirror trading/risk_controls.LiveRiskConfig (M1/M2) so a
    # backtest with risk enabled sees the same limits as live trading.
    risk_use_dd_breaker: bool = False
    risk_max_drawdown_pct: float = 0.15
    risk_dd_resume: str = "session_end"
    risk_dd_cooloff_bars: int = 48

    # --- Daily loss limit ---
    risk_use_daily_loss: bool = False
    risk_max_daily_loss_pct: float = 0.05
    risk_max_daily_loss_sigma: float = 3.0
    risk_daily_loss_mode: str = "pct"

    # --- Consecutive losses ---
    risk_use_consec_loss: bool = False
    risk_max_consecutive_losses: int = 5
    risk_consec_resume: str = "session_end"
    risk_consec_cooloff_bars: int = 48

    # --- Common ---
    risk_initial_equity: float = 10_000.0
    risk_max_open_positions: int = 1


@dataclass
class RiskState:
    """Mutable state for the risk management framework.

    Instantiated once at loop start and updated bar-by-bar.
    """

    equity_peak: float = 10_000.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    total_trades: int = 0
    total_wins: int = 0

    dd_paused: bool = False
    dd_cooloff_remaining: int = 0

    daily_paused: bool = False

    consec_paused: bool = False
    consec_cooloff_remaining: int = 0

    dd_breaches: int = 0
    daily_loss_breaches: int = 0
    consec_loss_breaches: int = 0
    bars_paused: int = 0


def should_suppress_entry(state: RiskState) -> bool:
    """Check if any risk safeguard is currently pausing trading.

    Parameters
    ----------
    state : RiskState
        Current risk state.

    Returns
    -------
    bool
        ``True`` if the strategy should be forced flat.
    """
    return state.dd_paused or state.daily_paused or state.consec_paused


def get_pause_reason(state: RiskState) -> str:
    """Return which safeguard(s) are currently active.

    Parameters
    ----------
    state : RiskState
        Current risk state.

    Returns
    -------
    str
        Comma-separated list of active safeguards (e.g. ``"drawdown,consecutive"``).
    """
    reasons = []
    if state.dd_paused:
        reasons.append("drawdown")
    if state.daily_paused:
        reasons.append("daily_loss")
    if state.consec_paused:
        reasons.append("consecutive_loss")
    return ",".join(reasons) if reasons else ""


def check_drawdown(
    config: RiskConfig,
    state: RiskState,
    current_equity: float,
) -> bool:
    """Check if drawdown from equity peak exceeds the threshold.

    If breached, activates the drawdown circuit breaker (pauses trading).
    Updates ``equity_peak`` if current equity is a new high.

    Parameters
    ----------
    config : RiskConfig
        Risk configuration (needs ``risk_use_dd_breaker``, ``risk_max_drawdown_pct``).
    state : RiskState
        Mutable risk state.
    current_equity : float
        Current account equity.

    Returns
    -------
    bool
        ``True`` if the drawdown breaker was just triggered.
    """
    if not config.risk_use_dd_breaker:
        return False

    if current_equity > state.equity_peak:
        state.equity_peak = current_equity

    if state.equity_peak <= 0:
        return False

    drawdown = (state.equity_peak - current_equity) / state.equity_peak

    if drawdown >= config.risk_max_drawdown_pct and not state.dd_paused:
        state.dd_paused = True
        state.dd_breaches += 1
        if config.risk_dd_resume == "cooloff":
            state.dd_cooloff_remaining = config.risk_dd_cooloff_bars
        return True

    return False


def check_daily_loss(
    config: RiskConfig,
    state: RiskState,
    bar_pnl: float,
    bar_vol: float = 0.0,
    bars_per_day: int = 48,
) -> bool:
    """Check if cumulative daily loss exceeds the threshold.

    Called after each bar's PnL is booked.

    Parameters
    ----------
    config : RiskConfig
        Risk configuration.
    state : RiskState
        Mutable risk state (``daily_pnl`` is updated).
    bar_pnl : float
        Per-bar PnL (strat[i]).
    bar_vol : float
        Bar volatility (for sigma-based limit).
    bars_per_day : int
        Bars per day (for sigma scaling).

    Returns
    -------
    bool
        ``True`` if the daily loss breaker was just triggered.
    """
    if not config.risk_use_daily_loss:
        return False

    state.daily_pnl += bar_pnl

    if state.daily_paused:
        return False

    if config.risk_daily_loss_mode == "sigma":
        if bar_vol <= 0:
            return False
        day_sigma_est = bar_vol * (bars_per_day ** 0.5)
        loss_limit = config.risk_max_daily_loss_sigma * day_sigma_est
    else:
        loss_limit = config.risk_max_daily_loss_pct

    if state.daily_pnl <= -abs(loss_limit):
        state.daily_paused = True
        state.daily_loss_breaches += 1
        return True

    return False


def update_after_trade(
    config: RiskConfig,
    state: RiskState,
    trade_pnl: float,
    is_win: bool,
) -> bool:
    """Update risk state after a completed trade.

    Tracks consecutive losses and triggers the breaker if threshold met.

    Parameters
    ----------
    config : RiskConfig
        Risk configuration.
    state : RiskState
        Mutable risk state.
    trade_pnl : float
        Realised PnL of the closed trade.
    is_win : bool
        Whether the trade was profitable.

    Returns
    -------
    bool
        ``True`` if the consecutive-loss breaker was just triggered.
    """
    if not config.risk_use_consec_loss:
        return False

    state.total_trades += 1
    if is_win:
        state.total_wins += 1
        state.consecutive_losses = 0
    else:
        state.consecutive_losses += 1

    if (
        config.risk_max_consecutive_losses > 0
        and state.consecutive_losses >= config.risk_max_consecutive_losses
        and not state.consec_paused
    ):
        state.consec_paused = True
        state.consec_loss_breaches += 1
        if config.risk_consec_resume == "cooloff":
            state.consec_cooloff_remaining = config.risk_consec_cooloff_bars
        return True

    return False


def tick_cooloffs(config: RiskConfig, state: RiskState) -> None:
    """Decrement cooloff counters for all paused safeguards.

    Called each bar while any safeguard is active.  Automatically
    unpauses when the counter reaches zero (for ``"cooloff"`` resume mode).

    Parameters
    ----------
    config : RiskConfig
        Risk configuration (needed for resume mode).
    state : RiskState
        Mutable risk state.
    """
    any_paused = False

    if state.dd_paused:
        any_paused = True
        if config.risk_dd_resume == "cooloff" and state.dd_cooloff_remaining > 0:
            state.dd_cooloff_remaining -= 1
            if state.dd_cooloff_remaining <= 0:
                state.dd_paused = False

    if state.consec_paused:
        any_paused = True
        if config.risk_consec_resume == "cooloff" and state.consec_cooloff_remaining > 0:
            state.consec_cooloff_remaining -= 1
            if state.consec_cooloff_remaining <= 0:
                state.consec_paused = False

    if any_paused:
        state.bars_paused += 1


def reset_daily(state: RiskState, config: RiskConfig | None = None) -> None:
    """Reset daily PnL and session-based pauses for a new trading day.

    Called on each new calendar day or session gap.
    Only resets safeguards with ``"session_end"`` resume mode --
    cooloff-based safeguards continue counting down via :meth:`tick_cooloffs`.

    Parameters
    ----------
    state : RiskState
        Mutable risk state.
    config : RiskConfig or None
        If provided, used to check resume mode for each safeguard.
        If None, all pauses are cleared (backward-compatible).
    """
    state.daily_pnl = 0.0
    state.daily_paused = False

    if config is None:
        state.dd_paused = False
        state.consec_paused = False
        state.dd_cooloff_remaining = 0
        state.consec_cooloff_remaining = 0
        return

    if config.risk_dd_resume == "session_end":
        state.dd_paused = False
        state.dd_cooloff_remaining = 0

    if config.risk_consec_resume == "session_end":
        state.consec_paused = False
        state.consec_cooloff_remaining = 0
