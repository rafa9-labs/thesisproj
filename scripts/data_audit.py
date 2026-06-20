"""
Seed Data Audit & Clean-up Script

Scans csv_data/ for V1-compatible currency pair data, reports:
  - Missing files per (pair, timeframe) combination
  - Data gaps, start/end timestamps, row counts
  - Flags non-core files for removal

V1 Core Parameters:
  Pairs:     EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
  Timeframes: M30, H1, H4

Usage:
  python scripts/data_audit.py           # Audit only (dry-run)
  python scripts/data_audit.py --clean   # Audit + remove non-core files
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── V1 Core Parameters ──────────────────────────────────────────
V1_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
V1_TIMEFRAMES = ["M30", "H1", "H4"]
CSV_DIR = Path("csv_data")

MIN_EXPECTED_BARS: Dict[str, int] = {
    "M30": 50_000,
    "H1": 25_000,
    "H4": 6_000,
}


def _parse_csv_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Extract (pair, timeframe) from OANDA-named CSV files."""
    m = re.match(r"^([A-Z]{6})_(\d+)_years_([MH]\d+)_OANDA\.csv$", filename)
    if m:
        return m.group(1), m.group(3)
    return None


def audit_csv_files() -> Dict[str, any]:
    """Scan csv_data/ and produce a structured audit report."""
    if not CSV_DIR.is_dir():
        return {"error": f"Directory not found: {CSV_DIR.resolve()}"}

    csv_files: List[Path] = sorted(CSV_DIR.glob("*.csv"))
    existing: Dict[Tuple[str, str], Path] = {}
    non_core: List[Path] = []
    unrecognized: List[Path] = []

    for f in csv_files:
        parsed = _parse_csv_filename(f.name)
        if parsed is None:
            unrecognized.append(f)
            continue
        pair, tf = parsed
        if pair not in V1_PAIRS or tf not in V1_TIMEFRAMES:
            non_core.append(f)
        else:
            existing[(pair, tf)] = f

    # Build grid: which combos exist
    combo_grid: Dict[str, Dict[str, bool]] = {}
    for pair in V1_PAIRS:
        combo_grid[pair] = {}
        for tf in V1_TIMEFRAMES:
            combo_grid[pair][tf] = (pair, tf) in existing

    # Missing combos
    missing: List[str] = []
    for pair in V1_PAIRS:
        for tf in V1_TIMEFRAMES:
            if (pair, tf) not in existing:
                missing.append(f"{pair}_{tf}")

    # Analyze existing core files for data quality
    file_details: List[Dict[str, any]] = []
    for (pair, tf), path in sorted(existing.items()):
        try:
            df = pd.read_csv(path, nrows=5)
            time_col = "time" if "time" in df.columns else df.columns[0]
        except Exception:
            file_details.append({
                "pair": pair, "timeframe": tf, "path": str(path),
                "error": "Failed to read CSV header",
            })
            continue

        try:
            df_full = pd.read_csv(path, usecols=[time_col])
            start = str(df_full[time_col].iloc[0])
            end = str(df_full[time_col].iloc[-1])
            rows = len(df_full)
            min_expected = MIN_EXPECTED_BARS.get(tf, 5000)
            adequate = rows >= min_expected

            file_details.append({
                "pair": pair,
                "timeframe": tf,
                "path": str(path),
                "rows": rows,
                "start_date": start,
                "end_date": end,
                "adequate": adequate,
                "min_expected": min_expected,
            })
        except Exception as e:
            file_details.append({
                "pair": pair, "timeframe": tf, "path": str(path),
                "error": str(e),
            })

    return {
        "csv_dir": str(CSV_DIR.resolve()),
        "v1_pairs": V1_PAIRS,
        "v1_timeframes": V1_TIMEFRAMES,
        "total_csv_files": len(csv_files),
        "core_files": len(existing),
        "non_core_files": len(non_core),
        "unrecognized_files": len(unrecognized),
        "missing_combos": missing,
        "combo_grid": combo_grid,
        "file_details": file_details,
        "non_core_list": [str(f) for f in non_core],
        "unrecognized_list": [str(f) for f in unrecognized],
    }


def print_report(report: Dict[str, any]) -> None:
    """Pretty-print the audit report."""
    if "error" in report:
        print(f"[ERROR] {report['error']}")
        sys.exit(1)

    print("=" * 78)
    print("  V1 Seed Data Audit Report")
    print("=" * 78)
    print(f"  Directory       : {report['csv_dir']}")
    print(f"  Total CSV files : {report['total_csv_files']}")
    print(f"  Core (V1) files : {report['core_files']}")
    print(f"  Non-core files  : {report['non_core_files']}")
    print(f"  Unrecognized    : {report['unrecognized_files']}")
    print()

    # Combo grid
    print("  Coverage Grid (V1 pairs x timeframes):")
    print(f"  {'':>10}" + "".join(f"  {tf:>6}" for tf in V1_TIMEFRAMES))
    print(f"  {'':>10}" + "".join(" " + "-" * 6 for _ in V1_TIMEFRAMES))
    for pair in V1_PAIRS:
        row = f"  {pair:<10}"
        for tf in V1_TIMEFRAMES:
            status = "[ OK ]" if report["combo_grid"][pair][tf] else "[MISS]"
            row += f"  {status:>6}"
        print(row)
    print()

    # Missing
    if report["missing_combos"]:
        print(f"  MISSING ({len(report['missing_combos'])}):")
        for m in report["missing_combos"]:
            print(f"    - {m}")
    else:
        print("  All V1 pair x timeframe combinations present.")
    print()

    # File details
    print(f"  Core File Inventory ({len(report['file_details'])} files):")
    print(f"  {'Pair':<10} {'TF':>5} {'Rows':>9} {'Start':<26} {'End':<26} {'Status'}")
    print(f"  {'-' * 10} {'-' * 5} {'-' * 9} {'-' * 26} {'-' * 26} {'-' * 10}")
    for fd in report["file_details"]:
        if "error" in fd:
            print(f"  {fd['pair']:<10} {fd['timeframe']:>5} {'ERROR':>9}  {fd['error']}")
        else:
            status = "OK" if fd["adequate"] else f"LOW (<{fd['min_expected']:,})"
            color = "" if fd["adequate"] else " **"
            print(
                f"  {fd['pair']:<10} {fd['timeframe']:>5} {fd['rows']:>9,}  "
                f"{fd['start_date']:<26} {fd['end_date']:<26} {status}{color}"
            )
    print()

    # Non-core files
    if report["non_core_list"]:
        print(f"  Non-core files (flagged for removal):")
        for f in report["non_core_list"]:
            print(f"    {f}")
        print()
    else:
        print("  No non-core files found.")
        print()

    if report["unrecognized_list"]:
        print(f"  Unrecognized CSV files (cannot parse filename):")
        for f in report["unrecognized_list"]:
            print(f"    {f}")
        print()

    print("=" * 78)


def clean_non_core(report: Dict[str, any], dry_run: bool = False) -> List[str]:
    """Remove non-core CSV files. Returns list of removed paths."""
    removed: List[str] = []
    for filepath in report.get("non_core_list", []):
        path = Path(filepath)
        if not path.exists():
            print(f"  [SKIP] Already removed: {path.name}")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would remove: {path.name}")
            removed.append(str(path))
        else:
            try:
                os.remove(path)
                removed.append(str(path))
                print(f"  [REMOVED] {path.name}")
            except OSError as e:
                print(f"  [ERROR] Failed to remove {path.name}: {e}")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1 Seed Data Audit & Clean-up",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove non-core CSV files (non-V1 pairs/timeframes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be removed without deleting",
    )
    args = parser.parse_args()

    report = audit_csv_files()
    print_report(report)

    if args.clean or args.dry_run:
        print("  Clean-up:")
        clean_non_core(report, dry_run=args.dry_run)
        print()

        # Re-audit after clean
        if not args.dry_run:
            print("  Re-audit after clean-up:")
            report2 = audit_csv_files()
            print_report(report2)


if __name__ == "__main__":
    main()
