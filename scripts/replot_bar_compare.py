#!/usr/bin/env python3
"""
Re-plot per-bar model vs Buy&Hold using:
  (1) Original price CSV (e.g., OANDA EURUSD M30) to compute Buy&Hold baselines
  (2) A saved model_bar_compare.csv (from your experiment) to reuse the model equity curve + timestamp grid

It produces two BH variants that are BOTH thesis-acceptable, but answer different questions:

1) BH (Tradeable Grid / Contract BH)
   - Only accrues returns on the same timestamp grid the model is evaluated on.
   - If your evaluation grid skips bars (e.g., session-only trading), gaps are treated as FLAT
     (no exposure), so overnight/weekend gaps do NOT contribute to BH.

2) BH (Full Calendar / Context BH)
   - Standard continuous Buy&Hold on the full price timeline between the same start/end dates.

This avoids re-running Optuna / training. It only rebuilds BH and reuses the saved model equity curve.

Typical usage:
  python replot_bar_compare.py \
    --price-csv EURUSD_10_years_M30_OANDA.csv \
    --compare-csv model_bar_compare.csv \
    --model decision_tree \
    --bh-baselines both \
    --overlap-mode union_rebase \
    --out-prefix ./decision_tree_bar_compare

If utilsNoWFO.py isn't importable from your current working directory:
  python replot_bar_compare.py ... --project-root /home/benji/projects/thesisproj
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


def _load_price(price_csv: str, time_col: str, close_col: str) -> pd.DataFrame:
    px = pd.read_csv(price_csv)
    if time_col not in px.columns:
        raise ValueError(f"Price CSV missing time column '{time_col}'. Columns={list(px.columns)}")
    if close_col not in px.columns:
        raise ValueError(f"Price CSV missing close column '{close_col}'. Columns={list(px.columns)}")

    px[time_col] = pd.to_datetime(px[time_col], utc=True, errors="coerce")
    px = px.dropna(subset=[time_col]).sort_values(time_col)
    px[close_col] = px[close_col].astype(float)
    return px[[time_col, close_col]].copy()


def _compute_bh_calendar_full(
    price_csv: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_col: str,
    close_col: str,
) -> pd.Series:
    """Standard continuous Buy&Hold on the *full price timeline* between [start, end]."""
    px = _load_price(price_csv, time_col, close_col)
    px = px[(px[time_col] >= start) & (px[time_col] <= end)]
    if px.empty:
        raise ValueError("No price rows inside requested [start, end] range.")

    close = px[close_col]
    ret = close.pct_change().fillna(0.0)
    bh = (1.0 + ret).cumprod()
    bh.index = pd.to_datetime(px[time_col].values, utc=True)
    bh = bh.sort_index()

    first = bh.iloc[0]
    if not np.isfinite(first) or first == 0.0:
        raise ValueError("Full-calendar BH has invalid first value; check price data.")
    return (bh / first).astype(float)


def _compute_bh_tradeable_contract(
    price_csv: str,
    *,
    contract_index: pd.DatetimeIndex,
    time_col: str,
    close_col: str,
    gap_multiple: float = 1.5,
) -> pd.Series:
    """
    Buy&Hold on the model's tradeable grid (contract_index):

    - We align prices to contract_index (merge_asof/backward).
    - Compute pct_change on THAT aligned series.
    - If there is a time gap larger than ~1.5x the typical bar interval,
      we set the return on the first bar after the gap to 0 (flat exposure over the gap).

    This matches the idea: "BH under the same decision grid / session contract as the model".
    """
    if contract_index is None or len(contract_index) < 2:
        raise ValueError("contract_index is empty/too short for contract BH computation.")

    start = contract_index.min()
    end = contract_index.max()

    px = _load_price(price_csv, time_col, close_col)
    px = px[(px[time_col] >= start) & (px[time_col] <= end)]
    if px.empty:
        raise ValueError("No price rows inside requested [start, end] range for contract BH.")

    # Align close to contract timestamps
    aligned = pd.merge_asof(
        pd.DataFrame({"time": contract_index}),
        px.rename(columns={time_col: "time"}),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("time")[close_col].astype(float)

    aligned = aligned.ffill().bfill()

    # Returns only between consecutive contract bars
    ret = aligned.pct_change().fillna(0.0)

    # Identify gaps (session breaks / weekends) and force those returns to 0
    dt = contract_index.to_series().diff()
    # Typical bar step (median of positive diffs)
    typical = dt[dt.notna()]
    typical = typical[typical > pd.Timedelta(0)]
    if len(typical) > 0:
        step = typical.median()
        gap_mask = dt > (step * gap_multiple)
        # gap affects the *first bar after the gap* (i.e., current bar's return)
        ret.loc[gap_mask.index[gap_mask.fillna(False)]] = 0.0

    bh = (1.0 + ret).cumprod()
    first = bh.iloc[0]
    if not np.isfinite(first) or first == 0.0:
        raise ValueError("Contract BH has invalid first value; check timestamps/columns.")
    return (bh / first).astype(float)


def main():
    ap = argparse.ArgumentParser(
        description="Re-plot model vs BH (contract and/or calendar) using your existing plotting helper."
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
            "contract = BH on tradeable grid (session/decision grid, gaps flat), "
            "calendar = continuous BH on full timeline (context), "
            "both = generate two PNGs."
        ),
    )

    ap.add_argument(
        "--gap-multiple",
        type=float,
        default=1.5,
        help="Gap detector: treat dt > gap_multiple * typical_step as a 'flat exposure' gap for contract BH.",
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

    # --- model equity (normalized to start at 1.0) --------------------------
    eq = cmp[equity_col].astype(float)
    if eq.isna().any():
        eq = eq.ffill().bfill()
    first_eq = eq.iloc[0]
    if np.isfinite(first_eq) and first_eq != 0.0:
        eq = eq / first_eq

    models_arg = [args.model] if args.model else None
    png_paths = []

    # --- (1) Contract BH: same decision grid as the model -------------------
    if args.bh_baselines in ("contract", "both"):
        bh_contract = _compute_bh_tradeable_contract(
            price_csv=args.price_csv,
            contract_index=cmp.index,
            time_col=args.price_time_col,
            close_col=args.price_close_col,
            gap_multiple=args.gap_multiple,
        )

        bt_plot_dict_contract = {
            "BH": bh_contract,
            equity_col: eq,
        }

        png_paths.append(
            (
                "BH_CONTRACT",
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

    # --- (2) Calendar BH: continuous BH, model ffilled across missing bars ---
    if args.bh_baselines in ("calendar", "both"):
        start = cmp.index.min()
        end = cmp.index.max()
        bh_calendar = _compute_bh_calendar_full(
            args.price_csv,
            start=start,
            end=end,
            time_col=args.price_time_col,
            close_col=args.price_close_col,
        )

        # Reindex model equity onto full calendar index.
        # Outside compare timestamps (e.g., outside your trading session), it stays flat (carry-forward).
        eq_calendar = eq.reindex(bh_calendar.index, method="ffill").fillna(1.0)

        bt_plot_dict_calendar = {
            "BH": bh_calendar,
            equity_col: eq_calendar,
        }

        png_paths.append(
            (
                "BH_CALENDAR",
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
