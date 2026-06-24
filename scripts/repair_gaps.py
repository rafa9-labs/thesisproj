"""Find and repair gaps in the candles SQLite DB by fetching from OANDA.

Skips normal weekend gaps (Fri 22:00 - Sun 22:00 UTC) and fills only
weekday gaps caused by syncer downtime.

Usage: python scripts/repair_gaps.py
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from oandapyV20 import API

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data.data_downloader import _fetch_candles
from pipeline.data.data_sqlite import DataStore
from pipeline.data.candle_syncer import _resolve_credentials, _normalize_ts

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAMES = ["M30", "H1", "H4"]
TF_SECONDS = {"M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}


def _is_weekend_gap(start, end):
    """True if the gap is a normal Fri 22:00 - Sun 22:00 weekend closure."""
    Fri = 4
    Mon = 0
    Sun = 6
    dow_start = start.weekday()
    dow_end = end.weekday()
    # Gap starts Fri evening or Saturday and ends Sunday evening or Monday
    if dow_start == Fri and start.hour >= 22 and dow_end in (Sun, Mon):
        return True
    if dow_start == Sat and dow_end in (Sun, Mon):
        return True
    return False


def find_gaps(store, pair, tf, min_gap_hours=1.5):
    """Find weekday gaps in the candles table for a pair+tf."""
    with store._cursor() as (conn, cur):
        cur.execute(
            "SELECT ts FROM candles WHERE pair=? AND timeframe=? ORDER BY ts",
            (pair.upper(), tf),
        )
        rows = [r[0] for r in cur.fetchall()]

    if len(rows) < 2:
        return []

    period = TF_SECONDS.get(tf, 1800)
    gaps = []
    for i in range(1, len(rows)):
        try:
            t1 = pd.Timestamp(rows[i - 1])
            t2 = pd.Timestamp(rows[i])
            diff = (t2 - t1).total_seconds()
            if diff > period * min_gap_hours and not _is_weekend_gap(t1, t2):
                gaps.append({
                    "start": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": t2.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "hours": diff / 3600,
                })
        except Exception:
            continue

    return gaps


def repair_gap(client, store, pair, tf, gap_start, gap_end):
    """Fetch candles from OANDA to fill a gap."""
    oanda_name = f"{pair[:3]}_{pair[3:]}"

    start_dt = pd.Timestamp(gap_start) - pd.Timedelta(hours=1)
    end_dt = pd.Timestamp(gap_end) + pd.Timedelta(hours=1)

    df = _fetch_candles(
        client, oanda_name,
        start_dt.to_pydatetime(), end_dt.to_pydatetime(),
        tf, include_incomplete=True,
    )

    if df.empty:
        print("    OANDA returned no data for %s -> %s" % (gap_start, gap_end))
        return 0

    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates("time")
    df = df.sort_values("time")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            pair, tf,
            _normalize_ts(str(r["time"])),
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

    if rows:
        store.insert_candles_batch(rows)
    return len(rows)


def main():
    creds = _resolve_credentials()
    if not creds:
        print("ERROR: No OANDA credentials found.")
        sys.exit(1)

    token, account_id = creds
    client = API(access_token=token, environment="practice")

    store = DataStore("data/forex.db")
    total_fixed = 0

    for pair in PAIRS:
        for tf in TIMEFRAMES:
            print("\n%s/%s: scanning for weekday gaps..." % (pair, tf))
            gaps = find_gaps(store, pair, tf)

            if not gaps:
                print("  No weekday gaps found")
                continue

            for g in gaps:
                print("  Gap: %s -> %s (%.1fh)" % (g["start"], g["end"], g["hours"]))
                n = repair_gap(client, store, pair, tf, g["start"], g["end"])
                if n:
                    print("    -> Filled %d candles" % n)
                    total_fixed += n
                else:
                    print("    -> No candles fetched")
                time.sleep(0.5)

    print("\n" + "=" * 60)
    print("Repair complete - %d candles inserted" % total_fixed)
    print("\nVerifying:")
    for pair in PAIRS:
        for tf in TIMEFRAMES:
            gaps = find_gaps(store, pair, tf)
            if gaps:
                print("  %s/%s: STILL has %d gap(s)" % (pair, tf, len(gaps)))
            else:
                print("  %s/%s: no weekday gaps (OK)" % (pair, tf))


if __name__ == "__main__":
    main()
