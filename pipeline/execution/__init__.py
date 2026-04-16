"""
pipeline.execution — Advanced execution models for the FX backtester.

Submodules:
  position_sizing  — Fixed, fractional, Kelly, ATR, vol-target sizers
  stops            — Fixed-pips, ATR, sigma SL/TP + breakeven + partial close
  trailing         — Fixed-pips, ATR, Chandelier trailing stops
"""

from pipeline.execution.position_sizing import (
    SizingMethod,
    SizingConfig,
    SizingState,
    compute_size,
    update_state,
)

from pipeline.execution.stops import (
    StopMethod,
    StopConfig,
    StopLevels,
    compute_stop_levels,
    check_stop_hit,
    check_breakeven,
)

from pipeline.execution.trailing import (
    TrailingMethod,
    TrailingConfig,
    TrailingState,
    update_trailing_state,
    is_activated,
    compute_trailing_sl,
)

__all__ = [
    "SizingMethod",
    "SizingConfig",
    "SizingState",
    "compute_size",
    "update_state",
    "StopMethod",
    "StopConfig",
    "StopLevels",
    "compute_stop_levels",
    "check_stop_hit",
    "check_breakeven",
    "TrailingMethod",
    "TrailingConfig",
    "TrailingState",
    "update_trailing_state",
    "is_activated",
    "compute_trailing_sl",
]
