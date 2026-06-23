import sqlite3
conn = sqlite3.connect("data/forex.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(1) FROM candles WHERE ts LIKE '%T%'")
count = cur.fetchone()[0]
print(f"Rows with T in timestamp: {count}")
cur.execute("DELETE FROM candles WHERE ts LIKE '%T%'")
print(f"Deleted {cur.rowcount} rows")
conn.commit()
conn.close()
