import sqlite3
conn = sqlite3.connect("data/forex.db")
cur = conn.cursor()
cur.execute("SELECT ts, mid_open, mid_high, mid_low, mid_close FROM candles WHERE pair='EURUSD' AND timeframe='M30' AND ts='2026-06-16 17:30:00+00:00'")
rows = cur.fetchall()
print(f"Found {len(rows)} rows at 2026-06-16 17:30:00+00:00")
for r in rows:
    print(f"  {r}")
