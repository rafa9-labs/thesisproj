"""
Batch download all currency pairs for the FX ML pipeline.

Downloads M30, H1, and H4 data for each pair via OANDA API.
Skips pairs that already have data (checks for existing CSVs).

Usage::

    python scripts/download_all_pairs.py
    python scripts/download_all_pairs.py --pairs GBPUSD USDJPY
    python scripts/download_all_pairs.py --force
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data.data_downloader import download_pair


PAIRS = [
    ("EUR_USD", "EURUSD"),
    ("GBP_USD", "GBPUSD"),
    ("USD_JPY", "USDJPY"),
    ("AUD_USD", "AUDUSD"),
    ("USD_CAD", "USDCAD"),
    ("GBP_JPY", "GBPJPY"),
]

GRANULARITIES = ["M30", "H1", "H4"]
DEFAULT_YEARS = 10
DATA_DIR = "csv_data"


def _csv_exists(pair_symbol: str, gran: str, years: int) -> bool:
    path = os.path.join(DATA_DIR, f"{pair_symbol}_{years}_years_{gran}_OANDA.csv")
    return os.path.isfile(path)


def main():
    parser = argparse.ArgumentParser(description="Batch download FX pair data")
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Specific pairs to download (e.g. GBPUSD USDJPY). Default: all.",
    )
    parser.add_argument(
        "--years", type=int, default=DEFAULT_YEARS,
        help=f"Years of history (default: {DEFAULT_YEARS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if CSVs already exist",
    )
    args = parser.parse_args()

    pairs_to_download = PAIRS
    if args.pairs:
        requested = set(args.pairs)
        pairs_to_download = [(o, s) for o, s in PAIRS if s in requested]
        missing = requested - {s for _, s in PAIRS}
        if missing:
            print(f"Warning: unknown pairs ignored: {missing}")

    print(f"Will download {len(pairs_to_download)} pair(s) x {len(GRANULARITIES)} timeframes")
    print(f"Years: {args.years} | Output: {DATA_DIR}/")
    print()

    total_files = 0
    skipped = 0

    for oanda_name, symbol in pairs_to_download:
        for gran in GRANULARITIES:
            if not args.force and _csv_exists(symbol, gran, args.years):
                print(f"SKIP {symbol} {gran} — CSV already exists")
                skipped += 1
                continue

        if skipped == len(GRANULARITIES) and not args.force:
            skipped = 0
            continue
        skipped = 0

        try:
            saved = download_pair(
                instrument=oanda_name,
                granularities=GRANULARITIES,
                years=args.years,
                output_dir=DATA_DIR,
            )
            total_files += len(saved)
        except Exception as e:
            print(f"ERROR downloading {oanda_name}: {e}")
            continue

    print(f"\nDone. Downloaded {total_files} file(s).")


if __name__ == "__main__":
    main()
