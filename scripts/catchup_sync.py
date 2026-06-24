"""One-shot catch-up sync for all active pairs — fills gap between last DB candle and now."""
import asyncio
import sys
sys.path.insert(0, ".")

from pipeline.data.data_sqlite import DataStore
from pipeline.data.candle_syncer import CandleSyncer, _DEFAULT_TIMEFRAMES


async def main():
    pairs = [p.strip() for p in sys.argv[1].split(",")] if len(sys.argv) > 1 else ["EURUSD", "GBPUSD", "USDJPY"]
    timeframes = _DEFAULT_TIMEFRAMES

    store = DataStore("data/forex.db")
    syncer = CandleSyncer(store, active_pairs=pairs, timeframes=timeframes)
    await syncer.start()

    print(f"Syncing {len(pairs)} pairs x {len(timeframes)} timeframes...")
    total = 0
    for pair in pairs:
        for tf in timeframes:
            try:
                n = await syncer.sync_pair(pair, tf)
                if n:
                    print(f"  {pair}/{tf}: +{n} candles")
                total += n
            except Exception as e:
                print(f"  {pair}/{tf}: FAILED ({e})")

    await syncer.stop()
    print(f"Done. {total} candles synced.")


if __name__ == "__main__":
    asyncio.run(main())
