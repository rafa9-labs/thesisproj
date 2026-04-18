"""
Position sizing models for the FX ML backtester.

Provides five sizing methods dispatched by ``compute_size()``:
  - FIXED            — constant 1.0 lot (baseline, legacy default)
  - FIXED_FRACTIONAL — proportional to account equity
  - KELLY            — Kelly criterion with configurable half-Kelly fraction
  - ATR              — volatility-adjusted via Average True Range
  - VOL_TARGET       — inverse-volatility target (migrated from execution_patches)

All sizers are **pure scalar functions** — no allocations, safe for the
tight bar-by-bar execution loop.

Usage::

    from pipeline.execution.position_sizing import (
        SizingMethod, SizingConfig, SizingState,
        compute_size, update_state,
    )

    cfg  = SizingConfig(method=SizingMethod.KELLY)
    st   = SizingState(equity=10_000.0)

    for i in range(n):
        size = compute_size(st, bar_vol[i], atr[i], cfg)
        # ... execute bar ...
        if trade_closed:
            update_state(st, pnl, is_win)
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SizingMethod:
    """String constants identifying each sizing model."""

    FIXED = "fixed"
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    ATR = "atr"
    VOL_TARGET = "vol_target"


@dataclass
class SizingConfig:
    """Immutable configuration for position sizing.

    Attributes
    ----------
    method : str
        One of :class:`SizingMethod` constants.
    risk_fraction : float
        Fraction of equity risked per trade (fixed-fractional & Kelly fallback).
    kelly_fraction : float
        Kelly scaling factor (0.5 = half-Kelly, conservative default).
    kelly_min_trades : int
        Minimum completed trades before Kelly is used; falls back to
        fixed-fractional below this threshold.
    atr_risk_pct : float
        Fraction of equity risked per trade (ATR sizing).
    atr_sl_mult : float
        Stop-loss distance as a multiple of ATR.
    initial_equity : float
        Starting account balance.
    max_leverage : float
        Hard cap on position size across all methods.
    contract_size : float
        Notional per standard lot (100 000 for EUR/USD).
    target_bar : float
        Target per-bar volatility (vol-target method).
    vol_floor : float
        Minimum bar-volatility denominator (avoids division by zero).
    max_lev : float
        Alias kept for vol-target backward compatibility.
    """

    method: str = SizingMethod.FIXED

    risk_fraction: float = 0.02
    kelly_fraction: float = 0.5
    kelly_min_trades: int = 10

    atr_risk_pct: float = 0.02
    atr_sl_mult: float = 2.0

    initial_equity: float = 10_000.0
    max_leverage: float = 5.0
    contract_size: float = 100_000.0

    target_bar: float = 0.0
    vol_floor: float = 1e-6
    max_lev: float = 1.5


@dataclass
class SizingState:
    """Mutable state tracked across bars for adaptive sizers.

    Instantiated once at the start of an execution loop and updated
    after every completed trade via :func:`update_state`.
    """

    equity: float = 10_000.0
    trade_count: int = 0
    win_count: int = 0
    total_win: float = 0.0
    total_loss: float = 0.0


def compute_size(
    state: SizingState,
    bar_vol: float,
    atr: float,
    config: SizingConfig,
) -> float:
    """Dispatch to the correct sizing model and return a lot size in ``[0, max_leverage]``.

    Parameters
    ----------
    state : SizingState
        Running account state (equity, trade stats).
    bar_vol : float
        Per-bar volatility (rolling std of returns) at the current bar.
    atr : float
        Average True Range at the current bar (0.0 if unavailable).
    config : SizingConfig
        Sizing model selection and parameters.

    Returns
    -------
    float
        Position size (lot units), clamped to ``[0, max_leverage]``.
    """
    m = config.method

    if state.equity <= 0:
        return 0.0

    if m == SizingMethod.FIXED:
        return 1.0

    if m == SizingMethod.FIXED_FRACTIONAL:
        return _clamp(_fixed_fractional(state, config), config.max_leverage)

    if m == SizingMethod.KELLY:
        return _clamp(_kelly(state, config), config.max_leverage)

    if m == SizingMethod.ATR:
        return _clamp(_atr_sizing(state, atr, config), config.max_leverage)

    if m == SizingMethod.VOL_TARGET:
        return _clamp(_vol_target(bar_vol, config), config.max_leverage)

    return 1.0


def update_state(
    state: SizingState,
    trade_pnl: float,
    is_win: bool,
) -> None:
    """Update running state after a completed trade.

    Parameters
    ----------
    state : SizingState
        Mutable state to update in-place.
    trade_pnl : float
        Realised PnL of the closed trade.
    is_win : bool
        ``True`` if the trade was profitable.
    """
    state.trade_count += 1
    if is_win:
        state.win_count += 1
        state.total_win += trade_pnl
    else:
        state.total_loss += abs(trade_pnl)


def _clamp(value: float, cap: float) -> float:
    return max(0.0, min(cap, value))


def _fixed_fractional(state: SizingState, config: SizingConfig) -> float:
    if config.initial_equity <= 0:
        return 1.0
    return config.risk_fraction * state.equity / config.initial_equity


def _kelly(state: SizingState, config: SizingConfig) -> float:
    if state.equity <= 0:
        return 0.0
    if state.trade_count < config.kelly_min_trades:
        return _fixed_fractional(state, config)

    n_wins = state.win_count
    n_losses = state.trade_count - n_wins
    if n_losses == 0 and n_wins == 0:
        return _fixed_fractional(state, config)

    p = n_wins / max(state.trade_count, 1)
    q = 1.0 - p

    avg_win = state.total_win / max(n_wins, 1)
    avg_loss = state.total_loss / max(n_losses, 1)
    b = avg_win / max(avg_loss, 1e-8)

    kelly_f = (p * b - q) / max(b, 1e-8)
    kelly_f = max(0.0, kelly_f)
    equity_ratio = state.equity / max(config.initial_equity, 1e-8)
    return config.kelly_fraction * kelly_f * min(equity_ratio, 1.0)


def _atr_sizing(state: SizingState, atr: float, config: SizingConfig) -> float:
    if atr <= 0:
        return _fixed_fractional(state, config)

    dollar_risk = config.atr_risk_pct * state.equity
    sl_distance = config.atr_sl_mult * atr
    lots = dollar_risk / max(sl_distance * config.contract_size, 1e-8)

    return lots


def _vol_target(bar_vol: float, config: SizingConfig) -> float:
    denom = max(bar_vol, config.vol_floor)
    ml = getattr(config, "max_lev", config.max_leverage)
    return min(ml, config.target_bar / denom)
