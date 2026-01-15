#!/usr/bin/env python3
"""
Re-plot per-bar model vs Buy&Hold using:
  (1) Original price CSV (e.g., OANDA EURUSD M30) to compute canonical BH from mid_close
  (2) A saved model_bar_compare.csv (from your experiment) to reuse the model equity curve + timestamp grid

This avoids re-running Optuna / training. It only rebuilds BH and reuses the model equity curve.

Typical usage (from your project root where utilsNoWFO.py is importable):
  python replot_bar_compare.py \
    --price-csv /path/EURUSD_10_years_M30_OANDA.csv \
    --compare-csv /path/model_bar_compare.csv \
    --model random_forest \
    --out-dir /path/out

If utilsNoWFO.py isn't importable from your current working directory:
  python replot_bar_compare.py ... --project-root /home/benji/projects/thesisproj

This script can output two thesis-valid BH baselines:
  - Contract BH (Tradeable Grid): BH on the same bar grid as the engine evaluation.
  - Context BH (Full Calendar): BH on the full price timeline for the month span,
    with model equity forward-filled onto that full timeline.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np


def _add_project_root(project_root: str | None):
    if project_root:
        project_root = os.path.abspath(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)


def _load_compare(compare_csv: str, index_col: str | None):
    df = pd.read_csv(compare_csv)

    # Your saved compare CSV often uses "Unnamed: 0" as time index.
    if index_col is None:
        if "Unnamed: 0" in df.columns:
            index_col = "Unnamed: 0"
        elif "time" in df.columns:
            index_col = "time"
        else:
            index_col = df.columns[0]

    t = pd.to_datetime(df[index_col], utc=True, errors="coerce")
    if t.isna().all():
        raise ValueError(
            f"Could not parse datetime from compare CSV column '{index_col}'. "
            f"Columns={list(df.columns)}"
        )
    df = df.set_index(t).sort_index()
    return df, index_col


def _pick_equity_column(compare_df: pd.DataFrame, model: str | None, equity_col: str | None):
    if equity_col and equity_col in compare_df.columns:
        return equity_col

    if model:
        cand = f"{model}_equity"
        if cand in compare_df.columns:
            return cand

    equity_cols = [c for c in compare_df.columns if c.endswith("_equity")]
    if equity_cols:
        return equity_cols[0]

    exclude = {"BH", "BH_fixed", "time"}
    numeric = [
        c for c in compare_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(compare_df[c])
    ]
    if numeric:
        return numeric[0]

    raise ValueError(f"Could not find an equity column. Available columns={list(compare_df.columns)}")


def _rebase_to_one(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.isna().any():
        s = s.ffill().bfill()
    first = s.dropna().iloc[0] if s.dropna().size else np.nan
    if np.isfinite(first) and first != 0.0:
        return s / first
    return s


def _compute_bh_aligned(price_csv: str, target_index: pd.DatetimeIndex,
                        time_col: str, close_col: str):
    """
    Compute BH from price CSV, then align it onto the provided target_index using merge_asof.
    This is useful when the compare CSV index is a filtered/tradeable grid.
    """
    px = pd.read_csv(price_csv)
    if time_col not in px.columns:
        raise ValueError(f"Price CSV missing time column '{time_col}'. Columns={list(px.columns)}")
    if close_col not in px.columns:
        raise ValueError(f"Price CSV missing close column '{close_col}'. Columns={list(px.columns)}")

    px[time_col] = pd.to_datetime(px[time_col], utc=True, errors="coerce")
    px = px.dropna(subset=[time_col]).sort_values(time_col)

    close = pd.to_numeric(px[close_col], errors="coerce").astype(float)
    px = px.assign(_close=close).dropna(subset=["_close"])
    ret = px["_close"].pct_change().fillna(0.0)
    px["bh_true"] = (1.0 + ret).cumprod()

    aligned = pd.merge_asof(
        pd.DataFrame({"time": target_index}),
        px[[time_col, "bh_true"]].rename(columns={time_col: "time"}),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("time")["bh_true"].astype(float)

    if aligned.isna().any():
        aligned = aligned.ffill().bfill()

    aligned = _rebase_to_one(aligned)
    if not np.isfinite(aligned.iloc[0]) or aligned.iloc[0] == 0.0:
        raise ValueError("Aligned BH has invalid first value; check timestamps/columns.")
    return aligned


def _compute_bh_full_calendar(
    price_csv: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_col: str,
    close_col: str,
) -> pd.Series:
    """Buy&Hold on the *full price timeline* between [start, end] (context baseline)."""
    px = pd.read_csv(price_csv)
    if time_col not in px.columns:
        raise ValueError(f"Price CSV missing time column '{time_col}'. Columns={list(px.columns)}")
    if close_col not in px.columns:
        raise ValueError(f"Price CSV missing close column '{close_col}'. Columns={list(px.columns)}")

    px[time_col] = pd.to_datetime(px[time_col], utc=True, errors="coerce")
    px = px.dropna(subset=[time_col]).sort_values(time_col)
    px = px[(px[time_col] >= start) & (px[time_col] <= end)]
    if px.empty:
        raise ValueError("No price rows inside requested [start, end] range.")

    close = pd.to_numeric(px[close_col], errors="coerce").astype(float)
    px = px.assign(_close=close).dropna(subset=["_close"])

    ret = px["_close"].pct_change().fillna(0.0)
    bh = (1.0 + ret).cumprod()

    bh.index = pd.to_datetime(px[time_col].values, utc=True)
    bh = bh.sort_index().astype(float)
    bh = _rebase_to_one(bh)

    if not np.isfinite(bh.iloc[0]) or bh.iloc[0] == 0.0:
        raise ValueError("Full-calendar BH has invalid first value; check price data.")
    return bh


def main():
    ap = argparse.ArgumentParser(
        description="Re-plot model vs canonical BH using your existing plotting function."
    )
    ap.add_argument("--price-csv", required=True, help="Path to original price CSV (e.g., OANDA).")
    ap.add_argument("--compare-csv", required=True, help="Path to model_bar_compare.csv generated by experiment.")
    ap.add_argument("--project-root", default=None, help="Project root to add to PYTHONPATH (so utilsNoWFO imports).")
    ap.add_argument("--model", default=None, help="Model name (e.g., random_forest). Used to pick {model}_equity.")
    ap.add_argument("--equity-col", default=None, help="Explicit equity column name in compare CSV (overrides --model).")
    ap.add_argument("--compare-index-col", default=None, help="Datetime column in compare CSV (default auto).")
    ap.add_argument("--price-time-col", default="time", help="Datetime column in price CSV (default: time).")
    ap.add_argument("--price-close-col", default="mid_close", help="Close column in price CSV (default: mid_close).")

    ap.add_argument("--out-dir", default=None, help="Output directory for both CSV and PNG.")
    ap.add_argument("--csv-dir", default=None, help="CSV output directory (optional).")
    ap.add_argument("--png-dir", default=None, help="PNG output directory (optional).")
    ap.add_argument("--out-prefix", default="results/model_bar_compare_replot",
                    help="Prefix if no dirs supplied.")

    ap.add_argument("--style", default="nature")
    ap.add_argument("--palette", default="okabe_ito_no_black")
    ap.add_argument("--bh-color", default="#666666")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--line-width", type=float, default=1.0)
    ap.add_argument("--n-time-parts", type=int, default=10)
    ap.add_argument("--overlap-mode", default="intersection", choices=["intersection", "union_rebase"])
    ap.add_argument("--annotate-coverage", action="store_true",
                    help="Show overlap % in legend labels.")
    ap.add_argument("--no-save-csv", action="store_true", help="Do not write CSV output.")

    ap.add_argument(
        "--bh-baselines",
        default="both",
        choices=["contract", "calendar", "both"],
        help=(
            "Which BH baseline(s) to plot: "
            "contract = BH on your tradeable/eval grid (system contract), "
            "calendar = BH on full price timeline for the month span (context), "
            "both = generate two PNGs."
        ),
    )

    args = ap.parse_args()

    _add_project_root(args.project_root)

    try:
        from utilsNoWFO import save_model_bar_comparison_outputs
    except Exception as e:
        raise SystemExit(
            "Could not import save_model_bar_comparison_outputs from utilsNoWFO.py.\n"
            "Run from your project root or pass --project-root <path>.\n"
            f"Import error: {e}"
        )

    cmp, used_idx_col = _load_compare(args.compare_csv, args.compare_index_col)
    equity_col = _pick_equity_column(cmp, args.model, args.equity_col)

    # --- model equity: always normalized to start at 1.0 for plotting consistency
    eq = _rebase_to_one(cmp[equity_col])

    models_arg = [args.model] if args.model else None
    png_paths = []

    # --- (1) Contract BH: use compare CSV BH if available; otherwise recompute aligned
    if args.bh_baselines in ("contract", "both"):
        if "BH" in cmp.columns and pd.api.types.is_numeric_dtype(cmp["BH"]):
            bh_contract = _rebase_to_one(cmp["BH"])
        else:
            bh_contract = _compute_bh_aligned(
                price_csv=args.price_csv,
                target_index=cmp.index,
                time_col=args.price_time_col,
                close_col=args.price_close_col,
            )

        bt_plot_dict_contract = {
            "BH": bh_contract,
            equity_col: eq,
        }

        png_paths.append(
            (
                "Buy-and-Hold (Tradeable Grid / Contract BH)",
                save_model_bar_comparison_outputs(
                    bt_plot_dict_contract,
                    models=models_arg,
                    out_prefix=f"{args.out_prefix}__bh_contract",
                    style=args.style,
                    palette=args.palette,
                    bh_color=args.bh_color,
                    n_time_parts=args.n_time_parts,
                    dpi=args.dpi,
                    line_width=args.line_width,
                    out_dir=args.out_dir,
                    csv_dir=args.csv_dir,
                    png_dir=args.png_dir,
                    overlap_mode=args.overlap_mode,
                    annotate_coverage=args.annotate_coverage,
                    save_csv=(not args.no_save_csv),
                ),
            )
        )

    # --- (2) Context BH: full price timeline for the month span; model equity forward-filled
    if args.bh_baselines in ("calendar", "both"):
        start_cmp = cmp.index.min()
        end_cmp = cmp.index.max()

        # Month-start in UTC for context baseline (matches your thesis intent: show full month context)
        start_month = start_cmp.normalize().replace(day=1)

        bh_calendar = _compute_bh_full_calendar(
            args.price_csv,
            start=start_month,
            end=end_cmp,
            time_col=args.price_time_col,
            close_col=args.price_close_col,
        )

        # Forward-fill model equity onto the full calendar timeline.
        # Before the first compare timestamp, keep neutral (flat at 1.0).
        eq_calendar = eq.reindex(bh_calendar.index, method="ffill")
        eq_calendar = eq_calendar.fillna(1.0)
        eq_calendar = _rebase_to_one(eq_calendar)

        bt_plot_dict_calendar = {
            "BH": bh_calendar,
            equity_col: eq_calendar,
        }

        png_paths.append(
            (
                "Buy-and-Hold (Full Calendar / Context BH)",
                save_model_bar_comparison_outputs(
                    bt_plot_dict_calendar,
                    models=models_arg,
                    out_prefix=f"{args.out_prefix}__bh_calendar",
                    style=args.style,
                    palette=args.palette,
                    bh_color=args.bh_color,
                    n_time_parts=args.n_time_parts,
                    dpi=args.dpi,
                    line_width=args.line_width,
                    out_dir=args.out_dir,
                    csv_dir=args.csv_dir,
                    png_dir=args.png_dir,
                    overlap_mode=args.overlap_mode,
                    annotate_coverage=args.annotate_coverage,
                    save_csv=(not args.no_save_csv),
                ),
            )
        )

    print("\nDone.")
    print("Compare index column used:", used_idx_col)
    print("Equity column used:", equity_col)
    for tag, p in png_paths:
        print(f"{tag} PNG:", p)


if __name__ == "__main__":
    main()
