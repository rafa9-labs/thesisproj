"""
OANDA candle data downloader.
Refactored from CSVDownloadOanda.py into a reusable module.
Reads credentials from .env file (never from committed config).
Inserts directly into SQLite via DataStore -- no intermediate CSV files.

Usage as module::

    from pipeline.data_downloader import download_pair
    download_pair("GBP_USD", granularities=["M30", "H1", "H4"])

Usage as CLI::

    python -m pipeline.data_downloader --instrument GBP_USD --years 10
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dateutil.parser import parse as parse_datetime
from dateutil.relativedelta import relativedelta


def _load_credentials() -> tuple[str, str]:
    """Load OANDA credentials from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    token = os.environ.get("OANDA_ACCESS_TOKEN", "")
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "")

    if not token or not account_id:
        raise RuntimeError(
            "OANDA credentials not found. Set OANDA_ACCESS_TOKEN and "
            "OANDA_ACCOUNT_ID in .env file or environment variables."
        )
    return token, account_id


def _align_to_grid(dt: datetime, granularity: str) -> datetime:
    """Snap a datetime to the candle grid boundary."""
    if granularity.startswith("M"):
        n = int(granularity[1:])
        return dt.replace(minute=(dt.minute // n) * n, second=0, microsecond=0)
    if granularity.endswith("H"):
        n = int(granularity[:-1])
        return dt.replace(hour=(dt.hour // n) * n, minute=0, second=0, microsecond=0)
    return dt


def _fetch_candles(
    client,
    instrument: str,
    start: datetime,
    end: datetime,
    granularity: str,
) -> pd.DataFrame:
    """Fetch candles from OANDA API with pagination."""
    from oandapyV20.endpoints import instruments

    start = _align_to_grid(start, granularity)
    all_data: list[dict] = []
    current_time = start
    previous_last_time: Optional[datetime] = None

    while current_time < end:
        params = {
            "from": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
            "count": 5000,
            "price": "MBA",
        }

        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            client.request(r)
            candles = r.response.get("candles")
        except Exception as e:
            print(f"  Error fetching data: {e}")
            break

        if not candles:
            break

        batch: list[dict] = []
        for candle in candles:
            if not candle["complete"]:
                continue

            time_str = candle["time"]
            candle_time = parse_datetime(time_str)

            if candle_time >= end:
                break

            mid = candle.get("mid", {})
            bid = candle.get("bid", {})
            ask = candle.get("ask", {})

            try:
                spread = float(ask["c"]) - float(bid["c"])
            except (KeyError, TypeError):
                spread = None

            batch.append({
                "time": time_str,
                "mid_open": mid.get("o"),
                "mid_high": mid.get("h"),
                "mid_low": mid.get("l"),
                "mid_close": mid.get("c"),
                "bid_open": bid.get("o"),
                "bid_close": bid.get("c"),
                "ask_open": ask.get("o"),
                "ask_close": ask.get("c"),
                "spread": spread,
                "volume": candle.get("volume"),
            })

        if not batch:
            break

        last_time = parse_datetime(batch[-1]["time"])

        if previous_last_time and last_time == previous_last_time:
            print(f"  Last candle repeated at {last_time}. Breaking.")
            break

        all_data.extend(batch)
        previous_last_time = last_time
        current_time = last_time + timedelta(seconds=1)

        print(f"  Fetched {len(batch)} candles ending at {last_time.isoformat()}")
        time.sleep(1)

    return pd.DataFrame(all_data)


def _df_to_rows(df: pd.DataFrame, pair: str, timeframe: str) -> list[tuple]:
    """Convert a fetched DataFrame into (pair, tf, ts, o, h, l, c, ...) tuples."""
    rows = []
    for _, r in df.iterrows():
        rows.append((
            pair,
            timeframe,
            str(r["time"]),
            float(r.get("mid_open", 0) or 0),
            float(r.get("mid_high", 0) or 0),
            float(r.get("mid_low", 0) or 0),
            float(r.get("mid_close", 0) or 0),
            float(r.get("bid_open", 0) or 0),
            float(r.get("bid_close", 0) or 0),
            float(r.get("ask_open", 0) or 0),
            float(r.get("ask_close", 0) or 0),
            float(r.get("spread", 0) or 0),
            int(r.get("volume", 0) or 0),
        ))
    return rows


BATCH_SIZE = 50_000


def _resolve_granularities(
    base_timeframe: str = "M30",
    granularities: Optional[List[str]] = None,
) -> List[str]:
    """Compute the set of timeframes needed given a base timeframe.

    Always includes the base + its MTF fast/slow from TIMEFRAME_HIERARCHY.
    Additional granularities are merged in without duplication.
    """
    from config import TIMEFRAME_HIERARCHY, DEFAULT_BASE_TIMEFRAME
    tf_h = TIMEFRAME_HIERARCHY.get(base_timeframe, TIMEFRAME_HIERARCHY[DEFAULT_BASE_TIMEFRAME])
    required = [base_timeframe, tf_h["mtf_fast"], tf_h["mtf_slow"]]
    if granularities:
        for g in granularities:
            if g not in required:
                required.append(g)
    return required


def download_pair(
    instrument: str,
    store = None,
    granularities: Optional[List[str]] = None,
    years: int = 10,
    output_dir: str = "csv_data",
    end_date: Optional[datetime] = None,
    pair_symbol: Optional[str] = None,
    base_timeframe: str = "M30",
) -> dict[str, int]:
    """Download candle data for a single OANDA instrument across multiple timeframes.

    Inserts directly into SQLite via DataStore.

    Parameters
    ----------
    instrument : str
        OANDA instrument name, e.g. ``"GBP_USD"``.
    store : DataStore
        DataStore instance for direct SQLite insert.
    granularities : list[str], optional
        Additional timeframes to download. Defaults to those needed for
        ``base_timeframe`` (base + MTF fast + MTF slow).
    years : int
        How many years of history to fetch.
    output_dir : str
        Unused -- kept for backward compatibility.
    end_date : datetime, optional
        End date for the data. Defaults to now.
    pair_symbol : str, optional
        Symbol like ``"EURUSD"``. When ``store`` is provided, used directly.
        Defaults to ``instrument.replace("_","")``.
    base_timeframe : str
        Primary trading timeframe (M15, M30, H1, H4). Determines which
        additional MTF timeframes are required.

    Returns
    -------
    dict[str, int]
        Mapping of granularity -> number of rows inserted.
    """
    from oandapyV20 import API

    granularities = _resolve_granularities(
        base_timeframe=base_timeframe,
        granularities=granularities,
    )

    token, _ = _load_credentials()
    client = API(access_token=token)

    if end_date is None:
        end_date = datetime.now(timezone.utc)

    start_date = end_date - relativedelta(years=years)

    if pair_symbol is None:
        pair_symbol = instrument.replace("_", "")

    saved: dict[str, int] = {}

    for gran in granularities:
        print(f"\n{'='*60}")
        print(f"Downloading {instrument} {gran} -- {start_date.date()} to {end_date.date()}")
        print(f"{'='*60}")

        df = _fetch_candles(client, instrument, start_date, end_date, gran)

        if df.empty:
            print(f"  No data retrieved for {instrument} {gran}.")
            continue

        if "complete" in df.columns:
            df = df.drop(columns=["complete"])

        cols = [
            "time", "mid_open", "mid_high", "mid_low", "mid_close",
            "bid_open", "bid_close", "ask_open", "ask_close", "spread", "volume",
        ]
        df = df[[c for c in cols if c in df.columns]]

        df["time"] = pd.to_datetime(df["time"])
        df = df.drop_duplicates("time")
        df = df.sort_values("time")
        df = df.reset_index(drop=True)

        rows = _df_to_rows(df, pair_symbol, gran)
        print(f"  Prepared {len(rows)} rows for {pair_symbol} {gran}")

        if store is not None:
            t0 = time.time()
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                store.insert_candles_batch(batch)
            t_ins = time.time() - t0
            rate = len(rows) / max(t_ins, 0.001)
            print(f"  Inserted {len(rows)} rows into DB in {t_ins:.1f}s ({rate:.0f} rows/s)")

        saved[gran] = len(rows)

    return saved


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Download OANDA candle data")
    parser.add_argument("--instrument", default="EUR_USD", help="OANDA instrument name (e.g. GBP_USD)")
    parser.add_argument("--pair", default=None, help="Pair symbol for DB (e.g. EURUSD). Defaults to instrument without _")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--granularities", nargs="+", default=["M30", "H1", "H4"])
    parser.add_argument("--output-dir", default="csv_data", help="(unused, kept for compat)")
    parser.add_argument("--db", default="data/forex.db", help="SQLite database path")
    args = parser.parse_args()

    from pipeline.data_sqlite import DataStore
    store = DataStore(args.db)
    pair_sym = args.pair or args.instrument.replace("_", "")
    download_pair(
        instrument=args.instrument,
        store=store,
        granularities=args.granularities,
        years=args.years,
        pair_symbol=pair_sym,
    )
