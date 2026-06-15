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

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Serialize write access across threads so that SQLite's WAL writer never
# times out under concurrent writers. Readers still proceed in parallel.
_WRITE_LOCK = threading.Lock()


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
    parent_job_id TEXT,
    study_meta  TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS job_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT    NOT NULL,
    event_index   INTEGER NOT NULL,
    event_data    TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id
    ON job_events (job_id);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id_event_index
    ON job_events (job_id, event_index);

CREATE TABLE IF NOT EXISTS deployed_models (
    id              TEXT PRIMARY KEY,
    model_type      TEXT    NOT NULL,
    snapshot_path   TEXT    NOT NULL,
    best_sharpe     REAL,
    best_return     REAL,
    created_at      TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'inactive',
    tags            TEXT    DEFAULT '[]',
    parent_job_id   TEXT
);

CREATE TABLE IF NOT EXISTS live_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    model_id        TEXT    NOT NULL,
    pair            TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    predicted_class INTEGER NOT NULL,
    confidence      REAL    NOT NULL,
    signal_used     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS saved_committees (
    id                TEXT PRIMARY KEY,
    name              TEXT    NOT NULL,
    full_cycle_job_id TEXT,
    pair              TEXT    DEFAULT 'EURUSD',
    timeframe         TEXT    DEFAULT 'H1',
    config_json       TEXT    NOT NULL,
    trust_score       REAL,
    avg_sharpe        REAL,
    is_active         INTEGER NOT NULL DEFAULT 0,
    tags              TEXT    DEFAULT '[]',
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_type
    ON jobs (status, type);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at
    ON jobs (created_at);

CREATE INDEX IF NOT EXISTS idx_deployed_models_status
    ON deployed_models (status);

CREATE INDEX IF NOT EXISTS idx_saved_committees_active
    ON saved_committees (is_active);
"""



class DataNotAvailableError(Exception):
    """Raised when required timeframe data is missing for a pair."""


class DataStore:
    """SQLite-backed market data store."""

    def __init__(self, db_path: str = "data/forex.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a fresh, standalone connection (kept for vacuum/one-off use)."""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local, long-lived SQLite connection.

        Opening/closing sqlite3 connections on every API request is the
        dominant I/O cost for read-heavy endpoints. This pool keeps one
        connection per worker thread alive and reuses it across calls.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self):
        """Read-capable cursor using the thread-local connection."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            yield conn, cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def _write_cursor(self):
        """Write cursor serialized by a module-level lock.

        SQLite WAL allows many readers but only one writer. Serializing
        write attempts in-process prevents busy-timeout errors when the
        pipeline and API threads write concurrently.
        """
        with _WRITE_LOCK:
            conn = self._get_connection()
            cur = conn.cursor()
            try:
                yield conn, cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close_current_connection(self) -> None:
        """Close the thread-local connection for the calling thread."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _ensure_schema(self):
        with self._write_cursor() as (conn, cur):
            expected = {"candles", "pairs", "jobs", "job_events"}
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing = {row[0] for row in cur.fetchall()}
            missing = expected - existing
            if missing:
                cur.executescript(SCHEMA_SQL)
            else:
                try:
                    cur.execute("ALTER TABLE jobs ADD COLUMN parent_job_id TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    cur.execute("ALTER TABLE jobs ADD COLUMN study_meta TEXT")
                except sqlite3.OperationalError:
                    pass
                try:
                    cur.executescript("""
                        CREATE TABLE IF NOT EXISTS deployed_models (
                            id              TEXT PRIMARY KEY,
                            model_type      TEXT    NOT NULL,
                            snapshot_path   TEXT    NOT NULL,
                            best_sharpe     REAL,
                            best_return     REAL,
                            created_at      TEXT    NOT NULL,
                            status          TEXT    NOT NULL DEFAULT 'inactive',
                            tags            TEXT    DEFAULT '[]',
                            parent_job_id   TEXT
                        );
                        CREATE TABLE IF NOT EXISTS live_predictions (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp       TEXT    NOT NULL,
                            model_id        TEXT    NOT NULL,
                            pair            TEXT    NOT NULL,
                            timeframe       TEXT    NOT NULL,
                            predicted_class INTEGER NOT NULL,
                            confidence      REAL    NOT NULL,
                            signal_used     INTEGER NOT NULL DEFAULT 0
                        );
                    """)
                except sqlite3.OperationalError:
                    pass
                try:
                    cur.executescript("""
                        CREATE TABLE IF NOT EXISTS saved_committees (
                            id                TEXT PRIMARY KEY,
                            name              TEXT    NOT NULL,
                            full_cycle_job_id TEXT,
                            pair              TEXT    DEFAULT 'EURUSD',
                            timeframe         TEXT    DEFAULT 'H1',
                            config_json       TEXT    NOT NULL,
                            trust_score       REAL,
                            avg_sharpe        REAL,
                            is_active         INTEGER NOT NULL DEFAULT 0,
                            tags              TEXT    DEFAULT '[]',
                            created_at        TEXT    NOT NULL,
                            updated_at        TEXT    NOT NULL
                        );
                    """)
                except sqlite3.OperationalError:
                    pass
                try:
                    cur.executescript("""
                        CREATE INDEX IF NOT EXISTS idx_jobs_status_type ON jobs (status, type);
                        CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at);
                        CREATE INDEX IF NOT EXISTS idx_deployed_models_status ON deployed_models (status);
                        CREATE INDEX IF NOT EXISTS idx_saved_committees_active ON saved_committees (is_active);
                    """)
                except sqlite3.OperationalError:
                    pass
            conn.commit()

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
        with self._write_cursor() as (conn, cur):
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
        with self._write_cursor() as (conn, cur):
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

        conn = self._get_connection()
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

    def get_latest_candles(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Return the most recent N candles for a pair+timeframe, oldest first."""
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT ts, mid_open, mid_high, mid_low, mid_close, "
                "bid_open, bid_close, ask_open, ask_close, spread, volume "
                "FROM candles WHERE pair=? AND timeframe=? ORDER BY ts DESC LIMIT ?",
                (pair.upper(), timeframe, limit),
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=cols)
            df = pd.DataFrame(rows, columns=cols)
            df.rename(columns={"ts": "time"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            for c in ("mid_open","mid_high","mid_low","mid_close","bid_open","bid_close","ask_open","ask_close","spread"):
                if c in df.columns:
                    df[c] = df[c].astype("float32")
            return df.sort_values("time").reset_index(drop=True)

    def append_job_event(self, job_id: str, event_data: str) -> int:
        with self._write_cursor() as (conn, cur):
            cur.execute(
                "INSERT INTO job_events (job_id, event_index, event_data, created_at) "
                "VALUES (?, COALESCE((SELECT MAX(event_index) FROM job_events WHERE job_id = ?), -1) + 1, ?, ?)",
                (job_id, job_id, event_data, datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def get_job_events(self, job_id: str, after: int = 0) -> List[Dict]:
        with self._cursor() as (conn, cur):
            cur.execute(
                "SELECT event_index, event_data FROM job_events WHERE job_id = ? AND event_index >= ? ORDER BY event_index",
                (job_id, after),
            )
            rows = cur.fetchall()
            return [{"_idx": row[0], **json.loads(row[1])} for row in rows]

    def get_job_event_count(self, job_id: str) -> int:
        with self._cursor() as (conn, cur):
            cur.execute("SELECT COUNT(*) FROM job_events WHERE job_id = ?", (job_id,))
            return cur.fetchone()[0]

    def clear_job_events(self, job_id: str) -> None:
        with self._write_cursor() as (conn, cur):
            cur.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))

    def trim_job_events(self, job_id: str, max_count: int) -> None:
        with self._write_cursor() as (conn, cur):
            cur.execute("SELECT COUNT(*) FROM job_events WHERE job_id = ?", (job_id,))
            count = cur.fetchone()[0]
            to_delete = count - max_count
            if to_delete > 0:
                cur.execute(
                    "DELETE FROM job_events WHERE job_id = ? AND rowid IN ("
                    "  SELECT rowid FROM job_events WHERE job_id = ? ORDER BY event_index ASC LIMIT ?"
                    ")",
                    (job_id, job_id, to_delete),
                )

    def prune_job_events(self, hours: int = 24) -> int:
        from datetime import timedelta
        with self._write_cursor() as (conn, cur):
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            cur.execute("DELETE FROM job_events WHERE created_at < ?", (cutoff,))
            return cur.rowcount
