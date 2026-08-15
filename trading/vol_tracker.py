"""
Lightweight price-history tracker for live engines.

Maintains a ring buffer of mid prices and exposes the two volatility inputs
the backtest sizing/stops use: rolling bar volatility (sigma) and an ATR
proxy. This keeps live compute_size() calls aligned with the backtest
execution loop (which passes real bar_vol/ATR instead of 0.0).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class VolTracker:
    """Rolling volatility / ATR proxy from a stream of mid prices."""

    def __init__(self, vol_window: int = 48, atr_window: int = 14) -> None:
        self._prices: deque = deque(maxlen=max(vol_window, atr_window) + 1)
        self._ret_abs: deque = deque(maxlen=atr_window)
        self.vol_window = int(vol_window)
        self.atr_window = int(atr_window)

    def update(self, mid: float) -> tuple[float, float]:
        """Feed one mid price; return (bar_vol, atr).

        bar_vol = rolling std of log returns (vol_window bars)
        atr     = mean |log return| * price (atr_window bars), price units
        """
        try:
            mid = float(mid)
        except (TypeError, ValueError):
            return (0.0, 0.0)
        if mid <= 0:
            return (0.0, 0.0)

        bar_vol = 0.0
        atr = 0.0

        if self._prices:
            prev = self._prices[-1]
            if prev > 0:
                lr = float(np.log(mid / prev))
                self._ret_abs.append(abs(lr))
        self._prices.append(mid)

        if len(self._prices) >= max(3, self.vol_window // 4):
            p = np.asarray(self._prices, dtype=float)
            lrs = np.diff(np.log(p))
            if len(lrs) >= 2:
                bar_vol = float(np.std(lrs, ddof=1))

        if len(self._ret_abs) >= 2:
            atr = float(np.mean(self._ret_abs)) * mid

        return (bar_vol, atr)

    def reset(self) -> None:
        self._prices.clear()
        self._ret_abs.clear()
