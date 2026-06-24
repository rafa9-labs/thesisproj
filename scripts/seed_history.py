"""Seed the SQLite database with historical OHLCV data from Yahoo Finance.

Fetches free historical data for EURUSD, GBPUSD, USDJPY across
M1, M5, M15, M30, H1, H4, D1 timeframes and inserts into the
existing ``candles`` table using INSERT OR IGNORE to preserve
any live-streamed bars.

Usage::

    cd thesisproj
    python scripts/seed_history.py
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_history")

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "forex.db"

PAIRS = [
    {"symbol": "EURUSD", "ticker": "EURUSD=X", "pip": 0.0001, "oanda": "EUR_USD"},
    {"symbol": "GBPUSD", "ticker": "GBPUSD=X", "pip": 0.0001, "oanda": "GBP_USD"},
    {"symbol": "USDJPY", "ticker": "USDJPY=X", "pip": 0.01,   "oanda": "USD_JPY"},
]

TIMEFRAME_CONFIG = {
    "M1":  {"interval": "1m",  "period": "7d"},
    "M5":  {"interval": "5m",  "period": "60d"},
    "M15": {"interval": "15m", "period": "60d"},
    "M30": {"interval": "30m", "period": "60d"},
    "H1":  {"interval": "1h",  "period": "730d"},
    "H4":  {"interval": "1h",  "period": "730d", "resample": "4h"},
    "D1":  {"interval": "1d",  "period": "5y"},
}

INSERT_CANDLES_SQL = """
    INSERT OR IGNORE INTO candles
        (pair, timeframe, ts, mid_open, mid_high, mid_low, mid_close,
         bid_open, bid_close, ask_open, ask_close, spread, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_PAIR_SQL = """
    INSERT OR IGNORE INTO pairs
        (symbol, oanda_name, pip_value, base_currency, quote_currency)
    VALUES (?, ?, ?, ?, ?)
"""

BATCH_SIZE = 2_000
FETCH_DELAY = 0.75  # seconds between API calls to avoid rate-limiting

# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _ensure_pairs(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for p in PAIRS:
        base = p["symbol"][:3]
        quote = p["symbol"][3:]
        cur.execute(INSERT_PAIR_SQL, (p["symbol"], p["oanda"], p["pip"], base, quote))
    conn.commit()
    logger.info("Ensured %d pairs in DB", len(PAIRS))


def _format_ts(ts: pd.Timestamp) -> str:
    """Convert a pandas Timestamp to the DB string format."""
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S+00:00")


def _df_to_rows(df: pd.DataFrame, symbol: str, timeframe: str) -> list[tuple]:
    """Convert a yfinance DataFrame to (pair, tf, ts, o, h, l, c, ...) tuples."""
    rows: list[tuple] = []
    for idx, row in df.iterrows():
        if not isinstance(idx, pd.Timestamp):
            idx = pd.Timestamp(idx)
        ts_str = _format_ts(idx)
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])
        vol = int(row.get("Volume", 0) or 0)
        rows.append((symbol, timeframe, ts_str, o, h, l, c, None, None, None, None, None, vol))
    return rows


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    if df.empty:
        return df
    ohlc = df["Close"].resample(rule).ohlc()
    ohlc.columns = ["Open", "High", "Low", "Close"]
    vol = df["Volume"].resample(rule).sum()
    ohlc["Volume"] = vol
    ohlc = ohlc.dropna()
    return ohlc


def _insert_batch(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    cur = conn.cursor()
    cur.executemany(INSERT_CANDLES_SQL, rows)
    conn.commit()


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    if not DB_PATH.exists():
        logger.error("DB not found at %s — run the API first to create the schema", DB_PATH)
        return

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_pairs(conn)

    total_inserted = 0

    for pair_cfg in PAIRS:
        symbol = pair_cfg["symbol"]
        ticker = pair_cfg["ticker"]
        logger.info("═" * 50)
        logger.info("Processing %s (%s)", symbol, ticker)

        yt = yf.Ticker(ticker)

        for tf, cfg in TIMEFRAME_CONFIG.items():
            interval = cfg["interval"]
            period = cfg["period"]
            resample_to = cfg.get("resample")

            logger.info("  %s/%s  interval=%s period=%s  fetching...",
                         symbol, tf, interval, period)

            try:
                df = yt.history(period=period, interval=interval)
                time.sleep(FETCH_DELAY)
            except Exception as exc:
                logger.warning("  Fetch failed for %s/%s: %s", symbol, tf, exc)
                continue

            if df.empty:
                logger.warning("  No data returned for %s/%s", symbol, tf)
                continue

            if resample_to:
                logger.info("  Resampling %s -> %s (%d raw bars)", tf, tf, len(df))
                df = _resample_ohlc(df, resample_to)

            rows = _df_to_rows(df, symbol, tf)
            logger.info("  Prepared %d rows for %s/%s", len(rows), symbol, tf)

            inserted = 0
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                before = conn.execute(
                    "SELECT COUNT(*) FROM candles WHERE pair = ? AND timeframe = ?",
                    (symbol, tf),
                ).fetchone()[0]
                _insert_batch(conn, batch)
                after = conn.execute(
                    "SELECT COUNT(*) FROM candles WHERE pair = ? AND timeframe = ?",
                    (symbol, tf),
                ).fetchone()[0]
                batch_inserted = after - before
                inserted += batch_inserted

            total_inserted += inserted
            if inserted > 0:
                logger.info("  Saved %d new bars for %s/%s", inserted, symbol, tf)
            else:
                logger.info("  No new bars for %s/%s (already up to date)", symbol, tf)

    conn.close()
    logger.info("═" * 50)
    logger.info("Done. %d total bars inserted.", total_inserted)


if __name__ == "__main__":
    main()
