"""
Mock Live Data Simulator — generates streaming OHLC bars for testing.

Phase E of the Multi-Agent Autonomous Exploration Engine.
Simulates a live feed of market data with configurable regime patterns,
spread, and volatility. Used to test the LiveCommitteeRunner without
a real OANDA connection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MockDataConfig:
    """Configuration for mock live data generation."""
    n_bars: int = 2000
    base_price: float = 1.1000
    spread_bps: float = 1.5
    trend_slope: float = 0.00001     # per-bar drift in trend sections
    vol_scale: float = 0.0003        # base volatility
    vol_spike: float = 0.0012        # volatility in high-volatile sections
    squeeze_vol: float = 0.00004      # volatility in quiet/sideways sections
    regime_section_bars: int = 400    # bars per regime section
    seed: int = 42

    # Regime rotation sequence (will cycle through these)
    regime_sequence: List[str] = field(default_factory=lambda: [
        "trend_up", "trend_down", "mean_reverting",
        "high_volatile", "sideways", "breakout",
        "trend_up",
    ])


class MockLiveFeed:
    """Generator that yields OHLC bars one at a time, simulating streaming data.

    Parameters
    ----------
    config : MockDataConfig
        Bar generation parameters.
    start_time : str or datetime-like
        Start timestamp for the bars.
    bar_freq : str
        Pandas frequency string (default "1h").
    """

    def __init__(
        self,
        config: Optional[MockDataConfig] = None,
        start_time: str = "2024-01-01",
        bar_freq: str = "1h",
    ):
        self.config = config or MockDataConfig()
        self.start_time = pd.Timestamp(start_time, tz="UTC") if start_time else pd.Timestamp("2024-01-01", tz="UTC")
        self.bar_freq = bar_freq
        self._rng = np.random.default_rng(self.config.seed)

    def generate_bars(self) -> Generator[Dict[str, float], None, None]:
        """Yield one bar at a time."""
        n = self.config.n_bars
        cfg = self.config

        # Pre-generate all prices
        prices = np.zeros(n, dtype=np.float64)
        prices[0] = cfg.base_price

        regime_seq = cfg.regime_sequence
        section_len = cfg.regime_section_bars

        for i in range(n - 1):
            regime_idx = (i // section_len) % len(regime_seq)
            regime = regime_seq[regime_idx]
            section_pos = i % section_len

            if regime == "trend_up":
                drift = cfg.trend_slope
                vol = cfg.vol_scale
            elif regime == "trend_down":
                drift = -cfg.trend_slope
                vol = cfg.vol_scale
            elif regime == "high_volatile":
                drift = 0.0
                vol = cfg.vol_spike
            elif regime == "breakout":
                drift = cfg.trend_slope * 3 if section_pos < 50 else cfg.trend_slope * 0.5
                vol = cfg.vol_scale * 2
            elif regime == "sideways":
                drift = 0.0
                vol = cfg.squeeze_vol
            elif regime == "mean_reverting":
                phase = section_pos / section_len * 2 * np.pi * 3
                drift = -0.0003 * np.sin(phase)  # oscillation
                vol = cfg.vol_scale * 0.7
            else:
                drift = 0.0
                vol = cfg.vol_scale

            shock = self._rng.normal(drift, vol)
            prices[i + 1] = prices[i] + shock

        # Generate bars
        timestamps = pd.date_range(self.start_time, periods=n, freq=self.bar_freq, tz="UTC")

        for i in range(n):
            vol = cfg.vol_scale
            regime_idx = (i // section_len) % len(regime_seq)
            if regime_seq[regime_idx] == "high_volatile":
                vol = cfg.vol_spike * 0.5

            bar_vol = abs(self._rng.normal(0, vol)) + 0.00005
            mid_c = prices[i]
            mid_h = mid_c + bar_vol
            mid_l = mid_c - bar_vol
            mid_o = prices[i - 1] if i > 0 else mid_c - self._rng.normal(0, vol)

            # Returns
            prev_close = prices[i - 1] if i > 0 else mid_c
            ret = np.log(mid_c / prev_close) if prev_close > 0 else 0.0

            yield {
                "timestamp": timestamps[i],
                "mid_c": float(mid_c),
                "mid_h": float(mid_h),
                "mid_l": float(mid_l),
                "mid_o": float(mid_o),
                "spread": float(cfg.spread_bps / 10000.0),
                "returns": float(ret),
            }

    def to_dataframe(self) -> pd.DataFrame:
        """Generate all bars as a DataFrame (for non-streaming use)."""
        bars = list(self.generate_bars())
        return pd.DataFrame(bars).set_index("timestamp")

    def regime_labels(self) -> List[str]:
        """Return the ground-truth regime label for each bar."""
        n = self.config.n_bars
        regime_seq = self.config.regime_sequence
        section_len = self.config.regime_section_bars
        return [regime_seq[(i // section_len) % len(regime_seq)] for i in range(n)]


def simulate_session(
    runner: Any,
    feed: MockLiveFeed,
    record_pnl: bool = True,
    verbose: bool = False,
) -> Dict:
    """Run a complete simulated trading session.

    Parameters
    ----------
    runner : LiveCommitteeRunner
        Initialized and started runner.
    feed : MockLiveFeed
        Bar generator.
    record_pnl : bool
        If True, compute next-bar PnL after each signal.
    verbose : bool
        Print progress every 200 bars.

    Returns
    -------
    dict with keys: signals (list), summary (dict), returns (list).
    """
    signals: List[Dict] = []
    returns: List[float] = []

    bars_processed = 0
    for bar in feed.generate_bars():
        bars_processed += 1

        signal = runner.process_bar(bar)

        if signal is not None:
            signals.append(signal.to_dict())

            if record_pnl and signal.signal != 0:
                # Compute PnL: use the next bar's return (simulating 1-bar execution delay)
                r = bar["returns"] * signal.signal
                returns.append(r)
                runner.record_trade_outcome(signal, r)

        if verbose and bars_processed % 200 == 0:
            print(f"  [MOCK] Bar {bars_processed}: {len(signals)} signals so far")

    summary = runner.stop()
    summary["bars_processed"] = bars_processed

    return {
        "signals": signals,
        "summary": summary,
        "returns": returns,
    }
