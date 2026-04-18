"""
Trailing stop implementations for the FX ML backtester.

Provides three trailing-stop methods that dynamically update the SL price
as the trade moves favourably.  Trailing stops operate in **price space**
and integrate with the S2.2 stop/TP system — they overwrite the static SL
once activated, and the existing stop check catches the exit.

Trailing methods:
  - FIXED_PIPS   — fixed pip distance from best price seen
  - ATR           — ATR × multiplier distance from best price
  - CHANDELIER    — highest_high(N) − multiplier × ATR (long)

All trailing stops share:
  - **Activation threshold** — trailing only kicks in after unrealised
    profit exceeds *activation_pips*; before that, the static SL from
    the stop/TP module stays active.
  - **Ratchet-only** — the trailing SL only moves toward current price,
    never away from it.

Usage::

    from pipeline.execution.trailing import (
        TrailingMethod, TrailingConfig, TrailingState,
        update_trailing_state, is_activated, compute_trailing_sl,
    )

    cfg = TrailingConfig(method=TrailingMethod.ATR, trail_atr_mult=3.0)
    state = TrailingState()

    for i in range(n):
        update_trailing_state(cfg, state, high=highs[i], low=lows[i])
        if is_activated(cfg, state, entry, close[i], direction):
            new_sl = compute_trailing_sl(cfg, state, direction, atr[i])
            # update active_stop_levels.sl_price = new_sl
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List


class TrailingMethod:
    """String constants identifying each trailing-stop model."""

    NONE = "none"
    FIXED_PIPS = "fixed_pips"
    ATR = "atr"
    CHANDELIER = "chandelier"


@dataclass
class TrailingConfig:
    """Configuration for trailing stop management.

    Attributes
    ----------
    method : str
        One of :class:`TrailingMethod` constants.
    trail_pips : float
        Trailing distance in pips (FIXED_PIPS method).
    trail_atr_mult : float
        Trailing distance as ATR multiplier (ATR method).
    chandelier_atr_mult : float
        ATR multiplier for Chandelier exit.
    chandelier_lookback : int
        Rolling window (bars) for highest-high / lowest-low.
    activation_pips : float
        Minimum unrealised profit (pips) before trailing activates.
    pip_value : float
        Pip value for the pair (0.0001 for EURUSD-like, 0.01 for USDJPY-like).
    """

    method: str = TrailingMethod.NONE

    trail_pips: float = 30.0
    trail_atr_mult: float = 3.0

    chandelier_atr_mult: float = 3.0
    chandelier_lookback: int = 22

    activation_pips: float = 10.0
    pip_value: float = 0.0001


@dataclass
class TrailingState:
    """Mutable state tracked across bars for the trailing stop.

    Instantiated once per trade and updated bar-by-bar.
    """

    activated: bool = False
    high_water_mark: float = 0.0
    low_water_mark: float = float("inf")
    recent_highs: deque = field(default_factory=lambda: deque(maxlen=22))
    recent_lows: deque = field(default_factory=lambda: deque(maxlen=22))

    def reset(self, lookback: int = 22, entry_price: float = 0.0) -> None:
        self.activated = False
        if entry_price > 0:
            self.high_water_mark = entry_price
            self.low_water_mark = entry_price
        else:
            self.high_water_mark = 0.0
            self.low_water_mark = float("inf")
        self.recent_highs = deque(maxlen=lookback)
        self.recent_lows = deque(maxlen=lookback)


def update_trailing_state(
    config: TrailingConfig,
    state: TrailingState,
    high: float,
    low: float,
) -> None:
    """Update HWM/LWM and rolling high/low buffers for the current bar.

    Parameters
    ----------
    config : TrailingConfig
        Needed for ``chandelier_lookback`` to size the rolling buffer.
    state : TrailingState
        Mutable state to update in-place.
    high : float
        Bar high price.
    low : float
        Bar low price.
    """
    if high > 0:
        state.high_water_mark = max(state.high_water_mark, high)
        state.recent_highs.append(high)

    if low > 0:
        if state.low_water_mark == float("inf"):
            state.low_water_mark = low
        else:
            state.low_water_mark = min(state.low_water_mark, low)
        state.recent_lows.append(low)

    lookback = config.chandelier_lookback
    if len(state.recent_highs) > lookback:
        while len(state.recent_highs) > lookback:
            state.recent_highs.popleft()
    if len(state.recent_lows) > lookback:
        while len(state.recent_lows) > lookback:
            state.recent_lows.popleft()


def is_activated(
    config: TrailingConfig,
    state: TrailingState,
    entry_price: float,
    current_price: float,
    direction: float,
) -> bool:
    """Check if the trailing stop should activate.

    Once activated, it stays activated for the rest of the trade.

    Parameters
    ----------
    config : TrailingConfig
        Needs ``activation_pips`` and ``pip_value``.
    state : TrailingState
        Mutable state (``activated`` is set to True if threshold met).
    entry_price : float
        Trade entry price.
    current_price : float
        Current bar close price.
    direction : float
        +1.0 for long, -1.0 for short.

    Returns
    -------
    bool
        Whether the trailing stop is now active.
    """
    if state.activated:
        return True

    threshold = config.activation_pips * config.pip_value

    if direction > 0:
        unrealised = current_price - entry_price
    else:
        unrealised = entry_price - current_price

    if unrealised >= threshold:
        state.activated = True

    return state.activated


def compute_trailing_sl(
    config: TrailingConfig,
    state: TrailingState,
    direction: float,
    atr: float = 0.0,
) -> float:
    """Compute the trailing stop-loss price for the current bar.

    The returned SL is the **best possible** (tightest) trailing level.
    The caller is responsible for ensuring it only ratchets toward the
    current price (never moves away from it).

    Parameters
    ----------
    config : TrailingConfig
        Trailing model configuration.
    state : TrailingState
        Current trailing state (HWM/LWM, rolling buffers).
    direction : float
        +1.0 for long, -1.0 for short.
    atr : float
        ATR at the current bar (0.0 if unavailable).

    Returns
    -------
    float
        New trailing SL price.  Returns 0.0 if method is NONE or
        insufficient data.
    """
    m = config.method
    pv = config.pip_value

    if m == TrailingMethod.NONE:
        return 0.0

    if m == TrailingMethod.FIXED_PIPS:
        dist = config.trail_pips * pv
        if direction > 0:
            return state.high_water_mark - dist
        else:
            return state.low_water_mark + dist

    if m == TrailingMethod.ATR:
        dist = config.trail_atr_mult * max(atr, 0.0)
        if direction > 0:
            return state.high_water_mark - dist
        else:
            return state.low_water_mark + dist

    if m == TrailingMethod.CHANDELIER:
        mult = config.chandelier_atr_mult
        atr_val = max(atr, 0.0)
        if direction > 0:
            hh = max(state.recent_highs) if state.recent_highs else state.high_water_mark
            return hh - mult * atr_val
        else:
            ll = min(state.recent_lows) if state.recent_lows else state.low_water_mark
            return ll + mult * atr_val

    return 0.0
