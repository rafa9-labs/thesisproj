"""
CSV → SQLite migration tool.

Migrates all OANDA CSV files from csv_data/ into a single indexed SQLite database.
Uses batched executemany for high throughput (~500k rows/min).

Usage::

    python -m pipeline.data_migrator
    python -m pipeline.data_migrator --db data/forex.db
    python -m pipeline.data_migrator --pair EURUSD
    python -m pipeline.data_migrator --force
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline.data_sqlite import DataStore
from pipeline.pair_config import PAIR_REGISTRY, VALID_PAIRS

DATA_DIR = "csv_data"
DEFAULT_DB = "data/forex.db"
BATCH_SIZE = 50_000

_CSV_PATTERN = re.compile(
    r"^([A-Z]{6})_(\d+)_years_(M\d+|H\d+)_OANDA\.csv$"
)


def _discover_csvs(data_dir: str = DATA_DIR) -> list[dict]:
    """Find all OANDA CSV files and parse their names."""
    csvs = []
    if not os.path.isdir(data_dir):
        return csvs
    for fname in sorted(os.listdir(data_dir)):
        m = _CSV_PATTERN.match(fname)
        if not m:
            continue
        csvs.append({
            "pair": m.group(1),
            "years": int(m.group(2)),
            "timeframe": m.group(3),
            "path": os.path.join(data_dir, fname),
        })
    return csvs


def _csv_to_rows(csv_path: str, pair: str, timeframe: str) -> list[tuple]:
    """Load a CSV and convert to candle row tuples."""
    df = pd.read_csv(csv_path)
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


def migrate_pair(
    store: DataStore,
    csv_path: str,
    pair: str,
    timeframe: str,
    force: bool = False,
) -> int:
    """Migrate a single CSV file. Returns rows inserted."""
    existing = store.get_candle_count(pair, timeframe)
    if existing > 0 and not force:
        return 0

    print(f"  Reading {csv_path} ...", end=" ", flush=True)
    t0 = time.time()
    rows = _csv_to_rows(csv_path, pair, timeframe)
    t_read = time.time() - t0
    print(f"{len(rows)} rows ({t_read:.1f}s)")

    print(f"  Inserting {pair} {timeframe} ...", end=" ", flush=True)
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        store.insert_candles_batch(batch)
    t_ins = time.time() - t0
    rate = len(rows) / max(t_ins, 0.001)
    print(f"{len(rows)} rows in {t_ins:.1f}s ({rate:.0f} rows/s)")

    return len(rows)


def migrate_all(
    db_path: str = DEFAULT_DB,
    data_dir: str = DATA_DIR,
    pair_filter: str | None = None,
    force: bool = False,
):
    """Run the full migration."""
    print(f"CSV -> SQLite Migration")
    print(f"  Source: {data_dir}/")
    print(f"  Target: {db_path}")
    print()

    store = DataStore(db_path)

    # Insert pair metadata
    pairs_data = []
    for cfg in PAIR_REGISTRY.values():
        d = {
            "symbol": cfg.symbol,
            "oanda_name": cfg.oanda_name,
            "pip_value": cfg.pip_value,
            "lot_size": cfg.lot_size,
            "base_currency": cfg.base_currency,
            "quote_currency": cfg.quote_currency,
            "typical_spread_bps": cfg.typical_spread_bps,
        }
        if pair_filter and cfg.symbol != pair_filter.upper():
            continue
        pairs_data.append(d)
    store.insert_pairs(pairs_data)
    print(f"Inserted {len(pairs_data)} pair configs")

    # Discover and migrate CSVs
    csvs = _discover_csvs(data_dir)
    if pair_filter:
        csvs = [c for c in csvs if c["pair"] == pair_filter.upper()]

    if not csvs:
        print("No CSV files found.")
        return

    print(f"\nFound {len(csvs)} CSV files to migrate:\n")

    total_rows = 0
    t_start = time.time()
    for csv_info in csvs:
        pair = csv_info["pair"]
        tf = csv_info["timeframe"]
        path = csv_info["path"]

        existing = store.get_candle_count(pair, tf)
        if existing > 0 and not force:
            print(f"  SKIP {pair} {tf} — {existing} rows already in DB (use --force to overwrite)")
            continue

        n = migrate_pair(store, path, pair, tf, force=force)
        total_rows += n

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*50}")
    print(f"Migration complete: {total_rows:,} rows in {elapsed:.1f}s")

    summary = store.get_pair_summary()
    print(f"\n{'Pair':<8} {'TF':<5} {'Rows':>10} {'Start':<12} {'End':<12}")
    print("-" * 50)
    for s in summary:
        print(f"{s['pair']:<8} {s['timeframe']:<5} {s['rows']:>10,} {s['start_date'][:10]:<12} {s['end_date'][:10]:<12}")

    db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
    print(f"\nDatabase size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate CSV data to SQLite")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--data-dir", default=DATA_DIR, help="CSV directory")
    parser.add_argument("--pair", default=None, help="Migrate only this pair (e.g. EURUSD)")
    parser.add_argument("--force", action="store_true", help="Re-migrate even if data exists")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    migrate_all(
        db_path=args.db,
        data_dir=args.data_dir,
        pair_filter=args.pair,
        force=args.force,
    )
