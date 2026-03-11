"""
Execution and backtesting engine: trade simulation, risk management, and performance tracking
"""

from .simulator import TradeSimulator
from .risk import StaticStopLoss, TrailingStop, PositionSizer
from .metrics import PerformanceEvaluator, METRIC_NAMES
from .engine import BacktestEngine

__all__ = [
    "TradeSimulator",
    "StaticStopLoss",
    "TrailingStop",
    "PositionSizer",
    "PerformanceEvaluator",
    "METRIC_NAMES",
    "BacktestEngine",
]
