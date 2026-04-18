"""
SQLite data layer for the FX ML pipeline.

Provides indexed candle storage and retrieval, replacing raw CSV access.
Thread-safe via connection pooling. Designed for single-user desktop use.

Usage::

    from pipeline.data_sqlite import DataStore

    store = DataStore("data/forex.db")
    df = store.get_candles("EURUSD", "M30", "2024-01-01", "2024-06-01")
    print(store.list_pairs())
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    pair        TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    mid_open    REAL,
    mid_high    REAL,
    mid_low     REAL,
    mid_close   REAL,
    bid_open    REAL,
    bid_close   REAL,
    ask_open    REAL,
    ask_close   REAL,
    spread      REAL,
    volume      INTEGER,
    PRIMARY KEY (pair, timeframe, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_pair_tf_ts
    ON candles (pair, timeframe, ts);

CREATE TABLE IF NOT EXISTS pairs (
    symbol          TEXT PRIMARY KEY,
    oanda_name      TEXT    NOT NULL,
    pip_value       REAL    NOT NULL,
    lot_size        REAL    DEFAULT 100000.0,
    base_currency   TEXT    DEFAULT '',
    quote_currency  TEXT    DEFAULT '',
    typical_spread_bps REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    config      TEXT,
    result      TEXT,
    error       TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""


class DataStore:
    """SQLite-backed market data store."""

    def __init__(self, db_path: str = "data/forex.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._connect()
        try:
            yield conn, conn.cursor()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._cursor() as (conn, cur):
            cur.executescript(SCHEMA_SQL)

    def _normalize_ts(self, ts: str) -> str:
        """Normalize a timestamp string for SQLite comparison.

        OANDA CSVs store timestamps like '2024-06-01 00:00:00+00:00'.
        If the input lacks a timezone suffix, append '+00:00' so
        lexicographic comparison still works.
        """
        ts = ts.strip()
        if "+" not in ts and "-" not in ts[11:]:
            ts = ts + "+00:00"
        return ts

    def insert_pairs(self, pairs: List[Dict]):
        with self._cursor() as (conn, cur):
            cur.executemany(
                """INSERT OR REPLACE INTO pairs
                   (symbol, oanda_name, pip_value, lot_size, base_currency, quote_currency, typical_spread_bps)
                   VALUES (:symbol, :oanda_name, :pip_value, :lot_size, :base_currency, :quote_currency, :typical_spread_bps)""",
                pairs,
            )

    def get_pair(self, symbol: str) -> Optional[Dict]:
        with self._cursor() as (conn, cur):
            cur.execute("SELECT * FROM pairs WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def list_pairs(self) -> List[Dict]:
        with self._cursor() as (conn, cur):
            cur.execute("SELECT * FROM pairs ORDER BY symbol")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def insert_candles_batch(self, rows: List[tuple]):
        with self._cursor() as (conn, cur):
            cur.executemany(
                """INSERT OR REPLACE INTO candles
                   (pair, timeframe, ts, mid_open, mid_high, mid_low, mid_close,
                    bid_open, bid_close, ask_open, ask_close, spread, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_candles(
        self,
        pair: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve candles as a DataFrame matching the CSV format.

        Parameters
        ----------
        pair : str
            Symbol like ``"EURUSD"``.
        timeframe : str
            Granularity like ``"M30"``, ``"H1"``, ``"H4"``.
        start, end : str, optional
            ISO date strings for range filtering.

        Returns
        -------
        pd.DataFrame with columns:
            time, mid_open, mid_high, mid_low, mid_close,
            bid_open, bid_close, ask_open, ask_close, spread, volume
        """
        sql = """SELECT ts, mid_open, mid_high, mid_low, mid_close,
                        bid_open, bid_close, ask_open, ask_close, spread, volume
                 FROM candles
                 WHERE pair = ? AND timeframe = ?"""
        params: list = [pair, timeframe]

        if start is not None:
            sql += " AND ts >= ?"
            params.append(self._normalize_ts(start))
        if end is not None:
            sql += " AND ts <= ?"
            params.append(self._normalize_ts(end))

        sql += " ORDER BY ts"

        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if df.empty:
            return df

        df.rename(columns={"ts": "time"}, inplace=True)

        try:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        except Exception:
            pass

        for col in ("mid_open", "mid_high", "mid_low", "mid_close",
                     "bid_open", "bid_close", "ask_open", "ask_close", "spread"):
            if col in df.columns:
                df[col] = df[col].astype("float32")
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype("int32")
        return df

    def get_candle_count(self, pair: str, timeframe: str) -> int:
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT COUNT(*) FROM candles WHERE pair = ? AND timeframe = ?",
                (pair, timeframe),
            )
            return cur.fetchone()[0]

    def get_date_range(self, pair: str, timeframe: str) -> Optional[Tuple[str, str]]:
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT MIN(ts), MAX(ts) FROM candles WHERE pair = ? AND timeframe = ?",
                (pair, timeframe),
            )
            row = cur.fetchone()
            if row and row[0]:
                return (row[0], row[1])
        return None

    def list_timeframes(self, pair: str) -> List[str]:
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT DISTINCT timeframe FROM candles WHERE pair = ? ORDER BY timeframe",
                (pair,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_pair_summary(self) -> List[Dict]:
        """Get summary of all pairs: row counts and date ranges per timeframe."""
        with self._cursor() as (conn, cur):
            cur.execute("""
                SELECT pair, timeframe, COUNT(*) as rows,
                       MIN(ts) as start_date, MAX(ts) as end_date
                FROM candles
                GROUP BY pair, timeframe
                ORDER BY pair, timeframe
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def vacuum(self):
        with self._connect() as conn:
            conn.execute("VACUUM")
