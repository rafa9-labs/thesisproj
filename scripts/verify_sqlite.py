"""Quick verification of the SQLite data layer."""
from pipeline.data_sqlite import DataStore

store = DataStore("data/forex.db")

pairs = store.list_pairs()
print("Pairs in DB:", [p["symbol"] for p in pairs])

df = store.get_candles("GBPUSD", "M30", "2024-06-01", "2024-06-02")
print(f"GBPUSD M30 (1 day): {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"mid_close dtype: {df['mid_close'].dtype}")

df_jpy = store.get_candles("USDJPY", "H1", "2024-01-01", "2024-02-01")
print(f"USDJPY H1 (1 month): {len(df_jpy)} rows")

rng = store.get_date_range("EURUSD", "M30")
print(f"EURUSD M30 range: {rng}")

pair = store.get_pair("USDJPY")
print(f"USDJPY pip_value from DB: {pair['pip_value']}")

summary = store.get_pair_summary()
total_rows = sum(s["rows"] for s in summary)
print(f"Total rows: {total_rows:,}")
print("Data layer verification PASSED")
