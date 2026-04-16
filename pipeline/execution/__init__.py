"""
pipeline.execution — Advanced execution models for the FX backtester.

Submodules:
  position_sizing  — Fixed, fractional, Kelly, ATR, vol-target sizers
"""

from pipeline.execution.position_sizing import (
    SizingMethod,
    SizingConfig,
    SizingState,
    compute_size,
    update_state,
)

__all__ = [
    "SizingMethod",
    "SizingConfig",
    "SizingState",
    "compute_size",
    "update_state",
]
