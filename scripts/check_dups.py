import sqlite3
conn = sqlite3.connect("data/forex.db")
cur = conn.cursor()

cur.execute("SELECT ts, COUNT(1) as cnt FROM candles WHERE pair='EURUSD' AND timeframe='M30' GROUP BY ts HAVING cnt > 1 LIMIT 10")
dups = cur.fetchall()
print(f"Duplicate timestamps:")
for ts, cnt in dups:
    print(f"  {ts}: {cnt} rows")

cur.execute("SELECT COUNT(1) FROM (SELECT ts FROM candles WHERE pair='EURUSD' AND timeframe='M30' GROUP BY ts HAVING COUNT(1) > 1)")
total = cur.fetchone()[0]
print(f"Total duplicate timestamps: {total}")
