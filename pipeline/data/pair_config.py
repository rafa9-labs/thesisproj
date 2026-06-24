"""
Currency pair configuration registry.

Provides per-pair metadata (pip value, lot size, OANDA instrument name)
and helpers for resolving CSV data paths.

Usage::

    from pipeline.data.pair_config import get_pair_config, resolve_csv_paths

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


def register_custom_pair(
    symbol: str,
    pip_value: float,
    decimal_places: int,
    store = None,
) -> PairConfig:
    """Register a new currency pair that is not in the built-in registry.

    Parameters
    ----------
    symbol : str
        Currency pair symbol (e.g. ``"NZDUSD"``).
    pip_value : float
        Value of one pip (e.g. 0.0001 for most XXX/USD pairs).
    decimal_places : int
        Number of decimal places in the quote (4 for most pairs, 2 for JPY pairs).
    store : DataStore, optional
        If provided, persists the pair to the ``pairs`` SQLite table.

    Returns
    -------
    PairConfig

    Raises
    ------
    ValueError
        If symbol length is not 6 or pip_value is not positive.
    """
    key = symbol.upper().replace("_", "").replace("-", "").replace("/", "")
    if len(key) != 6:
        raise ValueError("Symbol must be a 6-character currency pair code (e.g. EURUSD).")
    if pip_value <= 0:
        raise ValueError("pip_value must be positive.")

    base = key[:3]
    quote = key[3:]
    oanda_name = f"{base}_{quote}"
    lot_size = 100000.0
    typical_spread_bps = 1.0

    cfg = PairConfig(
        symbol=key,
        oanda_name=oanda_name,
        pip_value=pip_value,
        lot_size=lot_size,
        base_currency=base,
        quote_currency=quote,
        typical_spread_bps=typical_spread_bps,
    )

    if store is not None:
        store.insert_pairs([{
            "symbol": cfg.symbol,
            "oanda_name": cfg.oanda_name,
            "pip_value": cfg.pip_value,
            "lot_size": cfg.lot_size,
            "base_currency": cfg.base_currency,
            "quote_currency": cfg.quote_currency,
            "typical_spread_bps": cfg.typical_spread_bps,
        }])

    return cfg
