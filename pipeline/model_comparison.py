"""
Model Comparison & Leaderboard — post-process pipeline results.

Provides a clean API for:
  - Scanning the latest results directory
  - Building a ranked leaderboard across models
  - Loading equity curves for overlay plots
  - Statistical significance testing (paired t-test on monthly returns)

Usage (CLI):
    python -m pipeline.model_comparison
    python -m pipeline.model_comparison --results-dir results/15_04_26__16_15
    python -m pipeline.model_comparison --quick  # only scan, no heavy stats
"""

from __future__ import annotations

import os
import sys
import glob
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_results_dir(root: str = "results") -> Optional[str]:
    """Return the path to the most recent results directory, or None."""
    if not os.path.isdir(root):
        return None
    dirs = sorted(glob.glob(os.path.join(root, "*")), key=os.path.getmtime, reverse=True)
    for d in dirs:
        if os.path.isdir(d):
            return d
    return None


def find_all_results_dirs(root: str = "results") -> list[str]:
    """Return all results directories sorted newest-first."""
    if not os.path.isdir(root):
        return []
    dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    return sorted(dirs, key=os.path.getmtime, reverse=True)


def load_ranking(results_dir: str) -> Optional[pd.DataFrame]:
    """Load the FINAL ranking CSV from a results directory."""
    csv_path = os.path.join(results_dir, "csv_ranking_FINAL.csv")
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    # Fallback: look for any ranking CSV
    candidates = sorted(glob.glob(os.path.join(results_dir, "**", "*ranking*.csv"), recursive=True))
    if candidates:
        return pd.read_csv(candidates[-1])
    return None


def load_combined_monthly(results_dir: str) -> Optional[pd.DataFrame]:
    """Load the combined monthly results CSV."""
    csv_path = os.path.join(results_dir, "combined_monthly_all.csv")
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    return None


def load_equity_curves(results_dir: str) -> dict[str, pd.DataFrame]:
    """Load per-bar equity curves from bar comparison CSVs.

    Returns dict mapping model_name -> DataFrame with columns:
        [timestamp, equity, model]
    """
    curves = {}
    # Look for bar comparison CSVs across all repetitions
    bar_csvs = sorted(glob.glob(os.path.join(results_dir, "**", "bar_compare_models_rep*.csv"), recursive=True))
    if not bar_csvs:
        bar_csvs = sorted(glob.glob(os.path.join(results_dir, "**", "model_bar_compare.csv"), recursive=True))

    for csv_path in bar_csvs:
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            for col in df.columns:
                if col.endswith("_equity") or col in ("cstrategy_cont", "creturns_cont"):
                    name = col.replace("_equity", "").replace("cstrategy_cont", "").replace("creturns_cont", "")
                    if col == "creturns_cont":
                        name = "Buy&Hold"
                    if name and name not in curves:
                        curves[name] = df[[col]].rename(columns={col: "equity"})
                        curves[name]["model"] = name
        except Exception:
            continue

    # Alternative: load from per-model monthly results and compound
    if not curves:
        combined = load_combined_monthly(results_dir)
        if combined is not None and not combined.empty:
            model_col = combined.get("model_type", combined.get("model"))
            if model_col is not None:
                for model_name in model_col.unique():
                    mask = model_col == model_name
                    model_df = combined[mask].sort_values("test_end")
                    if "equity_strategy" in model_df.columns:
                        eq = model_df["equity_strategy"].dropna()
                        if not eq.empty:
                            curves[model_name] = pd.DataFrame({
                                "equity": eq.values,
                                "model": model_name,
                            }, index=range(len(eq)))

    return curves


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

# Columns to display in the leaderboard (in order)
LEADERBOARD_DISPLAY = [
    "rank", "model", "months", "trades", "SR", "PSR", "DSR",
    "Calmar", "AnnRet", "FinalEq", "DA", "Prec", "F1",
    "active", "Profit/Hit", "EffConf",
]

# Friendly names for display
LEADERBOARD_LABELS = {
    "rank": "#",
    "model": "Model",
    "months": "Months",
    "trades": "Trades",
    "SR": "Sharpe",
    "PSR": "PSR",
    "DSR": "DSR",
    "Calmar": "Calmar",
    "AnnRet": "Ann Ret",
    "FinalEq": "Final Eq",
    "DA": "Dir Acc",
    "Prec": "Precision",
    "F1": "F1",
    "active": "Active%",
    "Profit/Hit": "Profit/Hit",
    "EffConf": "Eff Conf",
}


def build_leaderboard(results_dir: str, sort_by: str = "SR") -> pd.DataFrame:
    """Build a clean leaderboard DataFrame from a results directory.

    Args:
        results_dir: Path to a results/<date> directory.
        sort_by: Column to sort by (default: 'SR' = Sharpe Ratio).

    Returns:
        DataFrame with leaderboard columns, sorted descending by sort_by.
    """
    ranking = load_ranking(results_dir)
    if ranking is None or ranking.empty:
        return pd.DataFrame()

    df = ranking.copy()

    # Ensure rank column exists
    if "rank" not in df.columns:
        df["rank"] = range(1, len(df) + 1)

    # Ensure model column has consistent name
    if "model" not in df.columns and "model_type" in df.columns:
        df.rename(columns={"model_type": "model"}, inplace=True)

    # Select only available display columns
    available = [c for c in LEADERBOARD_DISPLAY if c in df.columns]
    df = df[available].copy()

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

    return df


def leaderboard_to_ascii(df: pd.DataFrame, max_width: int = 120) -> str:
    """Render leaderboard as a formatted ASCII table."""
    if df.empty:
        return "(no results)"

    # Format numeric columns
    display_df = df.copy()
    for col in display_df.columns:
        if col in ("rank", "months", "trades"):
            display_df[col] = display_df[col].apply(
                lambda x: str(int(x)) if pd.notna(x) else "—"
            )
        elif col == "model":
            continue
        else:
            display_df[col] = display_df[col].apply(
                lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
            )

    # Build table
    lines = []
    header = " 🏆 MODEL LEADERBOARD ".center(max_width, "=")
    lines.append(header)

    # Column widths
    col_widths = {}
    friendly_names = {}
    for col in display_df.columns:
        friendly = LEADERBOARD_LABELS.get(col, col)
        friendly_names[col] = friendly
        col_widths[col] = max(len(friendly), display_df[col].astype(str).str.len().max())

    # Header row
    header_parts = []
    for col in display_df.columns:
        header_parts.append(friendly_names[col].rjust(col_widths[col]))
    lines.append("  ".join(header_parts))
    lines.append("-" * len("  ".join(header_parts)))

    # Data rows
    for _, row in display_df.iterrows():
        parts = []
        for col in display_df.columns:
            parts.append(str(row[col]).rjust(col_widths[col]))
        lines.append("  ".join(parts))

    lines.append("=" * max_width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical Significance
# ---------------------------------------------------------------------------

def paired_significance_test(results_dir: str, metric: str = "strategy_return") -> Optional[pd.DataFrame]:
    """Run paired t-tests on monthly returns between all model pairs.

    Returns a DataFrame with columns: [model_a, model_b, t_stat, p_value, significant_5pct].
    """
    from scipy.stats import ttest_rel

    combined = load_combined_monthly(results_dir)
    if combined is None or combined.empty:
        return None

    model_col = combined.get("model_type", combined.get("model"))
    if model_col is None or "test_end" not in combined.columns:
        return None

    if metric not in combined.columns:
        # Try to compute from cstrategy
        if "cstrategy" in combined.columns:
            combined["strategy_return"] = combined["cstrategy"] - 1.0
            metric = "strategy_return"
        else:
            return None

    # Pivot: rows = test_end (month), columns = model, values = metric
    pivot = combined.pivot_table(index="test_end", columns=model_col, values=metric, aggfunc="first")
    pivot = pivot.dropna(how="all")

    models = list(pivot.columns)
    n_models = len(models)
    if n_models < 2:
        return None

    rows = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a, b = models[i], models[j]
            paired = pivot[[a, b]].dropna()
            if len(paired) < 3:
                continue
            t_stat, p_value = ttest_rel(paired[a], paired[b])
            rows.append({
                "model_a": a,
                "model_b": b,
                "t_stat": round(t_stat, 4),
                "p_value": round(p_value, 4),
                "significant_5pct": p_value < 0.05,
                "n_months": len(paired),
            })

    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Summary Export
# ---------------------------------------------------------------------------

def export_comparison_report(results_dir: str, output_dir: Optional[str] = None) -> str:
    """Export a complete comparison report (leaderboard + significance + curves).

    Returns the path to the output directory.
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "comparison_report")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Leaderboard
    lb = build_leaderboard(results_dir)
    if not lb.empty:
        lb.to_csv(os.path.join(output_dir, "leaderboard.csv"), index=False)
        lb.to_json(os.path.join(output_dir, "leaderboard.json"), orient="records", indent=2)

    # 2. Equity curves (combined)
    curves = load_equity_curves(results_dir)
    if curves:
        combined_curves = pd.DataFrame()
        for name, df in curves.items():
            s = df["equity"].copy()
            s.name = name
            if combined_curves.empty:
                combined_curves = pd.DataFrame(s)
            else:
                combined_curves = combined_curves.join(s, how="outer")
        combined_curves.to_csv(os.path.join(output_dir, "equity_curves.csv"))

    # 3. Significance tests
    sig = paired_significance_test(results_dir)
    if sig is not None and not sig.empty:
        sig.to_csv(os.path.join(output_dir, "significance_tests.csv"), index=False)

    # 4. Metadata
    meta = {
        "results_dir": results_dir,
        "n_models": len(lb) if not lb.empty else 0,
        "models": lb["model"].tolist() if not lb.empty and "model" in lb.columns else [],
    }
    with open(os.path.join(output_dir, "report_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for model comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Comparison & Leaderboard")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to specific results directory (default: latest)")
    parser.add_argument("--list", action="store_true",
                        help="List all available results directories")
    parser.add_argument("--export", action="store_true",
                        help="Export full comparison report")
    parser.add_argument("--significance", action="store_true",
                        help="Run paired significance tests")
    parser.add_argument("--quick", action="store_true",
                        help="Only show leaderboard, skip heavy computations")
    args = parser.parse_args()

    # List mode
    if args.list:
        dirs = find_all_results_dirs()
        if not dirs:
            print("No results directories found.")
            return
        print("\n📁 Available Results Directories:\n")
        for i, d in enumerate(dirs, 1):
            ranking = os.path.join(d, "csv_ranking_FINAL.csv")
            has_ranking = "✅" if os.path.isfile(ranking) else "❌"
            mod_time = os.path.getmtime(d)
            from datetime import datetime
            ts = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            print(f"  {i}. {os.path.basename(d)} ({ts}) {has_ranking}")
        return

    # Find results directory
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = find_latest_results_dir()
        if results_dir is None:
            print("❌ No results directories found. Run some backtests first.")
            sys.exit(1)

    if not os.path.isdir(results_dir):
        print(f"❌ Directory not found: {results_dir}")
        sys.exit(1)

    print(f"\n📊 Analyzing results: {results_dir}\n")

    # Leaderboard
    lb = build_leaderboard(results_dir)
    if lb.empty:
        print("❌ No ranking data found in this results directory.")
        sys.exit(1)

    print(leaderboard_to_ascii(lb))

    # Significance tests
    if not args.quick:
        if args.significance:
            print("\n🔬 Paired Significance Tests (monthly returns):\n")
            sig = paired_significance_test(results_dir)
            if sig is not None and not sig.empty:
                print(sig.to_string(index=False))
            else:
                print("  Not enough paired data for significance tests.")

    # Export
    if args.export:
        out = export_comparison_report(results_dir)
        print(f"\n✅ Comparison report exported to: {out}")


if __name__ == "__main__":
    main()