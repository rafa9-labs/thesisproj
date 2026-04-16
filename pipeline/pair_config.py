"""
Currency pair configuration registry.

Provides per-pair metadata (pip value, lot size, OANDA instrument name)
and helpers for resolving CSV data paths.

Usage::

    from pipeline.pair_config import get_pair_config, resolve_csv_paths

    cfg = get_pair_config("GBPUSD")
    print(cfg.pip_value)  # 0.0001

    paths = resolve_csv_paths("GBPUSD")
    # {"M30": "csv_data/GBPUSD_10_years_M30_OANDA.csv", ...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PairConfig:
    """Static configuration for a single currency pair."""

    symbol: str
    oanda_name: str
    pip_value: float
    lot_size: float = 100_000.0
    base_currency: str = ""
    quote_currency: str = ""
    typical_spread_bps: float = 1.0


PAIR_REGISTRY: Dict[str, PairConfig] = {
    "EURUSD": PairConfig(
        symbol="EURUSD",
        oanda_name="EUR_USD",
        pip_value=0.0001,
        lot_size=100_000.0,
        base_currency="EUR",
        quote_currency="USD",
        typical_spread_bps=0.8,
    ),
    "GBPUSD": PairConfig(
        symbol="GBPUSD",
        oanda_name="GBP_USD",
        pip_value=0.0001,
        lot_size=100_000.0,
        base_currency="GBP",
        quote_currency="USD",
        typical_spread_bps=1.2,
    ),
    "USDJPY": PairConfig(
        symbol="USDJPY",
        oanda_name="USD_JPY",
        pip_value=0.01,
        lot_size=100_000.0,
        base_currency="USD",
        quote_currency="JPY",
        typical_spread_bps=0.9,
    ),
    "AUDUSD": PairConfig(
        symbol="AUDUSD",
        oanda_name="AUD_USD",
        pip_value=0.0001,
        lot_size=100_000.0,
        base_currency="AUD",
        quote_currency="USD",
        typical_spread_bps=1.0,
    ),
    "USDCAD": PairConfig(
        symbol="USDCAD",
        oanda_name="USD_CAD",
        pip_value=0.0001,
        lot_size=100_000.0,
        base_currency="USD",
        quote_currency="CAD",
        typical_spread_bps=1.5,
    ),
    "GBPJPY": PairConfig(
        symbol="GBPJPY",
        oanda_name="GBP_JPY",
        pip_value=0.01,
        lot_size=100_000.0,
        base_currency="GBP",
        quote_currency="JPY",
        typical_spread_bps=2.0,
    ),
}

VALID_PAIRS: List[str] = list(PAIR_REGISTRY.keys())


def get_pair_config(symbol: str) -> PairConfig:
    """Look up pair configuration by symbol (e.g. ``"GBPUSD"``).

    Parameters
    ----------
    symbol : str
        Currency pair symbol (case-insensitive).

    Returns
    -------
    PairConfig

    Raises
    ------
    ValueError
        If the symbol is not in the registry.
    """
    key = symbol.upper().replace("_", "").replace("-", "").replace("/", "")
    cfg = PAIR_REGISTRY.get(key)
    if cfg is None:
        raise ValueError(
            f"Unknown pair '{symbol}'. "
            f"Available: {', '.join(VALID_PAIRS)}"
        )
    return cfg


def resolve_csv_paths(
    symbol: str,
    years: int = 10,
    data_dir: str = "csv_data",
) -> Dict[str, str]:
    """Resolve CSV file paths for a given pair and timeframe set.

    Parameters
    ----------
    symbol : str
        Currency pair symbol (e.g. ``"GBPUSD"``).
    years : int
        Number of years in the data file name.
    data_dir : str
        Directory containing CSV files.

    Returns
    -------
    dict[str, str]
        Keys: ``"base"`` (M30), ``"H1"``, ``"H4"``.
        Values: full relative paths to CSV files.
    """
    cfg = get_pair_config(symbol)
    s = cfg.symbol
    return {
        "base": os.path.join(data_dir, f"{s}_{years}_years_M30_OANDA.csv"),
        "H1": os.path.join(data_dir, f"{s}_{years}_years_H1_OANDA.csv"),
        "H4": os.path.join(data_dir, f"{s}_{years}_years_H4_OANDA.csv"),
    }


def find_best_csv_path(
    symbol: str,
    granularity: str,
    data_dir: str = "csv_data",
) -> Optional[str]:
    """Find a matching CSV file for a pair, trying multiple year spans.

    Checks for 10-year files first, then falls back to any matching file.

    Parameters
    ----------
    symbol : str
        Currency pair symbol.
    granularity : str
        Candle granularity (e.g. ``"M30"``, ``"H1"``, ``"H4"``).
    data_dir : str
        Directory containing CSV files.

    Returns
    -------
    str or None
        Path to the CSV file, or None if not found.
    """
    cfg = get_pair_config(symbol)
    s = cfg.symbol

    for years in (10, 5, 3, 1):
        path = os.path.join(data_dir, f"{s}_{years}_years_{granularity}_OANDA.csv")
        if os.path.isfile(path):
            return path

    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir), reverse=True):
            if fname.startswith(f"{s}_") and fname.endswith(f"_{granularity}_OANDA.csv"):
                return os.path.join(data_dir, fname)

    return None
