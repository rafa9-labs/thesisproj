"""Normalize mixed-format timestamps in the candles table to consistent format."""
import sqlite3
import pandas as pd

DB = "data/forex.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Find rows with OANDA RFC3339 timestamps (containing 'T')
cur.execute("SELECT pair, timeframe, ts FROM candles WHERE ts LIKE '%T%'")
broken = cur.fetchall()
print(f"Found {len(broken)} rows with OANDA timestamp format")

for pair, tf, ts in broken:
    try:
        t = pd.Timestamp(ts)
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        new_ts = t.strftime("%Y-%m-%d %H:%M:%S+00:00")
        cur.execute(
            "UPDATE candles SET ts = ? WHERE pair = ? AND timeframe = ? AND ts = ?",
            (new_ts, pair, tf, ts),
        )
    except Exception as e:
        print(f"  skip {pair}/{tf}/{ts}: {e}")

conn.commit()
conn.close()
print("Done")
