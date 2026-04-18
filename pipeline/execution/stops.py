"""
Stop-loss and take-profit management for the FX ML backtester.

Provides configurable stop/TP models that integrate into the
bar-by-bar execution loop in ``execution_patches.py``.

Stop/TP methods:
  - FIXED_PIPS    — static SL/TP in pip units
  - ATR           — dynamic SL/TP as multiples of Average True Range
  - SIGMA         — SL/TP as multiples of rolling bar volatility
  - NONE          — no explicit SL/TP (signal-only exits)

Additional features:
  - Breakeven stop management (move SL to entry after threshold)
  - Partial close / scale-out at configurable TP levels

Usage::

    from pipeline.execution.stops import (
        StopMethod, StopConfig, StopLevels,
        compute_stop_levels, check_stop_hit,
    )

    cfg = StopConfig(method=StopMethod.ATR, sl_atr_mult=2.0, tp_atr_mult=3.0)
    levels = compute_stop_levels(cfg, entry_price=1.0850, direction=1.0,
                                 atr=0.0030, bar_vol=0.0025)
    hit, hit_type = check_stop_hit(levels, current_price=1.0820)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


class StopMethod:
    """String constants identifying each stop/TP model."""

    NONE = "none"
    FIXED_PIPS = "fixed_pips"
    ATR = "atr"
    SIGMA = "sigma"


@dataclass
class StopConfig:
    """Configuration for stop-loss and take-profit management.

    Attributes
    ----------
    method : str
        One of :class:`StopMethod` constants.
    sl_pips : float
        Stop-loss distance in pips (FIXED_PIPS method).
    tp_pips : float
        Take-profit distance in pips (FIXED_PIPS method).
    sl_atr_mult : float
        SL as ATR multiplier (ATR method).
    tp_atr_mult : float
        TP as ATR multiplier (ATR method).
    sl_sigma_mult : float
        SL as bar-volatility multiplier (SIGMA method).
    tp_sigma_mult : float
        TP as bar-volatility multiplier (SIGMA method).
    pip_value : float
        Pip value for the pair (0.0001 for EURUSD-like, 0.01 for USDJPY-like).
    use_be : bool
        Enable breakeven stop management.
    be_trigger_pips : float
        Move SL to entry once unrealised profit exceeds this many pips.
    use_partial_close : bool
        Enable partial close (scale-out) at TP levels.
    tp1_ratio : float
        Fraction of position to close at TP1 (0.5 = close half).
    tp1_pips : float
        TP1 distance in pips (partial close level).
    tp2_pips : float
        TP2 distance in pips (full close level, 0 = no TP2).
    """

    method: str = StopMethod.NONE

    sl_pips: float = 30.0
    tp_pips: float = 60.0

    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0

    sl_sigma_mult: float = 2.0
    tp_sigma_mult: float = 3.0

    pip_value: float = 0.0001

    use_be: bool = False
    be_trigger_pips: float = 20.0

    use_partial_close: bool = False
    tp1_ratio: float = 0.5
    tp1_pips: float = 30.0
    tp2_pips: float = 0.0


@dataclass
class StopLevels:
    """Computed stop-loss and take-profit price levels for a single trade.

    All levels are in **price units** (not pips). A level of 0.0 means
    "not set" (no stop at that side).
    """

    sl_price: float = 0.0
    tp_price: float = 0.0
    be_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0


def compute_stop_levels(
    config: StopConfig,
    entry_price: float,
    direction: float,
    atr: float = 0.0,
    bar_vol: float = 0.0,
) -> StopLevels:
    """Compute SL/TP price levels for a new trade entry.

    Parameters
    ----------
    config : StopConfig
        Stop model configuration.
    entry_price : float
        Price at trade entry.
    direction : float
        +1.0 for long, -1.0 for short.
    atr : float
        ATR at the entry bar (0.0 if unavailable).
    bar_vol : float
        Rolling bar volatility at entry (0.0 if unavailable).

    Returns
    -------
    StopLevels
        Price levels for SL, TP, BE, TP1, TP2.
    """
    m = config.method
    levels = StopLevels()
    pv = config.pip_value

    if pv <= 0 and m in (StopMethod.FIXED_PIPS,):
        import logging
        logging.getLogger(__name__).warning(
            "pip_value=%.6f is non-positive — stop computation skipped to avoid "
            "immediate stop-out at entry price", pv
        )
        return levels

    if m == StopMethod.NONE:
        return levels

    if m == StopMethod.FIXED_PIPS:
        sl_dist = config.sl_pips * pv
        tp_dist = config.tp_pips * pv

    elif m == StopMethod.ATR:
        sl_dist = config.sl_atr_mult * max(atr, 0.0)
        tp_dist = config.tp_atr_mult * max(atr, 0.0)

    elif m == StopMethod.SIGMA:
        sl_dist = config.sl_sigma_mult * max(bar_vol, 0.0)
        tp_dist = config.tp_sigma_mult * max(bar_vol, 0.0)

    else:
        return levels

    if direction > 0:
        levels.sl_price = entry_price - sl_dist
        levels.tp_price = entry_price + tp_dist
    else:
        levels.sl_price = entry_price + sl_dist
        levels.tp_price = entry_price - tp_dist

    if config.use_be:
        levels.be_price = entry_price

    if config.use_partial_close:
        tp1_dist = config.tp1_pips * pv
        tp2_dist = config.tp2_pips * pv if config.tp2_pips > 0 else 0.0
        if direction > 0:
            levels.tp1_price = entry_price + tp1_dist
            levels.tp2_price = entry_price + tp2_dist if tp2_dist > 0 else 0.0
        else:
            levels.tp1_price = entry_price - tp1_dist
            levels.tp2_price = entry_price - tp2_dist if tp2_dist > 0 else 0.0

    return levels


def check_stop_hit(
    levels: StopLevels,
    current_price: float,
    direction: float,
    high_price: float = 0.0,
    low_price: float = 0.0,
) -> Tuple[bool, str]:
    """Check if SL or TP was hit during the current bar.

    Uses high/low prices for intra-bar detection when available,
    otherwise uses the close price.

    Parameters
    ----------
    levels : StopLevels
        Active stop levels.
    current_price : float
        Close price of the current bar.
    direction : float
        +1.0 for long, -1.0 for short.
    high_price : float
        High price of the current bar (0.0 = use close only).
    low_price : float
        Low price of the current bar (0.0 = use close only).

    Returns
    -------
    Tuple[bool, str]
        (hit, hit_type) where hit_type is "sl", "tp", "tp1", "tp2", or "".
    """
    bar_high = high_price if high_price > 0 else current_price
    bar_low = low_price if low_price > 0 else current_price

    if direction > 0:
        if levels.sl_price > 0 and bar_low <= levels.sl_price:
            return True, "sl"
        if levels.tp_price > 0 and bar_high >= levels.tp_price:
            return True, "tp"
        if levels.tp2_price > 0 and bar_high >= levels.tp2_price:
            return True, "tp2"
        if levels.tp1_price > 0 and bar_high >= levels.tp1_price:
            return True, "tp1"
    else:
        if levels.sl_price > 0 and bar_high >= levels.sl_price:
            return True, "sl"
        if levels.tp_price > 0 and bar_low <= levels.tp_price:
            return True, "tp"
        if levels.tp2_price > 0 and bar_low <= levels.tp2_price:
            return True, "tp2"
        if levels.tp1_price > 0 and bar_low <= levels.tp1_price:
            return True, "tp1"

    return False, ""


def check_breakeven(
    config: StopConfig,
    levels: StopLevels,
    entry_price: float,
    direction: float,
    current_price: float,
) -> StopLevels:
    """Check if breakeven stop should be activated, updating SL if so.

    Once the unrealised profit exceeds ``be_trigger_pips``, the SL is
    moved to the entry price (breakeven). This is irreversible — once
    the SL is at breakeven, it stays there.

    Parameters
    ----------
    config : StopConfig
        Stop configuration (needs ``use_be`` and ``be_trigger_pips``).
    levels : StopLevels
        Current stop levels (modified in-place and returned).
    entry_price : float
        Original trade entry price.
    direction : float
        +1.0 for long, -1.0 for short.
    current_price : float
        Current bar close price.

    Returns
    -------
    StopLevels
        Updated levels (SL moved to entry if triggered).
    """
    if not config.use_be:
        return levels

    pv = config.pip_value
    be_trigger_dist = config.be_trigger_pips * pv

    if direction > 0:
        unrealised_profit = current_price - entry_price
        if unrealised_profit >= be_trigger_dist and levels.sl_price < entry_price:
            levels.sl_price = entry_price
    else:
        unrealised_profit = entry_price - current_price
        if unrealised_profit >= be_trigger_dist and levels.sl_price > entry_price:
            levels.sl_price = entry_price

    return levels
