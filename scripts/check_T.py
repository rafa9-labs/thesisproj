import sqlite3
conn = sqlite3.connect("data/forex.db")
cur = conn.cursor()
cur.execute("SELECT ts FROM candles WHERE pair='EURUSD' AND timeframe='M30' AND ts LIKE '%T%'")
rows = cur.fetchall()
print(f"Rows still with T in timestamp: {len(rows)}")
for r in rows:
    print(f"  {r[0]}")
conn.close()
