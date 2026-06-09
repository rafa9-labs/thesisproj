"""Compare BorutaSHAP feature selection with and without the Economic Noise Floor.

Runs boruta_sweep_features() twice on a sampled subset of H1 data:
  1. floor=0.00 (baseline — original behavior, shadow_max only)
  2. floor=0.02 (economic floor — must be >=2% of top feature's SHAP)

Uses a reduced sample size and fewer folds to keep the comparison fast
(<5 min). The relative delta (features lost) is what matters, not exact counts.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from pipeline.feature_sweep import expand_features
from pipeline.boruta_sweep import boruta_sweep_features, _make_labels

DATA = os.path.join(
    os.path.dirname(__file__), "..", "csv_data", "EURUSD_10_years_H1_OANDA.csv"
)

# Use last N bars for a fast comparison (full 53k bars takes 5-10 min per run)
N_SAMPLE = 15000


def run(label: str, floor_pct: float) -> dict:
    print(f"\n{'='*64}")
    print(f"  {label}: economic_floor_pct = {floor_pct}")
    print(f"{'='*64}")

    df = pd.read_csv(DATA)
    df = df.iloc[-N_SAMPLE:].reset_index(drop=True)
    df_feat = expand_features(df)
    labels = _make_labels(df_feat, threshold=0.0001)

    # Drop hold (neutral) bars
    valid = labels != -1
    df_feat = df_feat.loc[valid].copy()
    labels = labels[valid]

    # Exclude non-feature columns
    exclude = {
        "returns", "time", "timestamp", "label",
        "mid_h", "mid_l", "mid_c", "mid_o",
        "mid_high", "mid_low", "mid_close", "mid_open",
        "bid_open", "bid_close", "ask_open", "ask_close",
        "spread", "volume",
    }
    numeric_cols = [
        c for c in df_feat.columns
        if c not in exclude and np.issubdtype(df_feat[c].dtype, np.number)
    ]
    df_feat = df_feat[numeric_cols].copy()
    df_feat = df_feat.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X = df_feat.to_numpy(np.float32)
    y = labels

    print(f"  Sample: {len(X)} bars, {len(numeric_cols)} features")

    t0 = time.perf_counter()
    locked, scores, report = boruta_sweep_features(
        df,
        label_threshold=0.0001,
        n_folds=3,
        max_iter=5,
        random_state=42,
        economic_floor_pct=floor_pct,
    )
    elapsed = time.perf_counter() - t0

    n_confirmed = report.get("features_confirmed", "?")
    n_rejected = report.get("features_rejected", "?")
    n_locked = report.get("features_locked", "?")
    iterations = report.get("boruta_iterations", "?")

    print(f"  Confirmed: {n_confirmed}  Rejected: {n_rejected}  Locked: {n_locked}")
    print(f"  Iterations: {iterations}  Elapsed: {elapsed:.1f}s")
    print(f"  Locked features: {locked[:12]}{'...' if len(locked) > 12 else ''}")

    return {
        "floor_pct": floor_pct,
        "confirmed": n_confirmed,
        "rejected": n_rejected,
        "locked": locked,
        "locked_set": set(locked),
        "elapsed": elapsed,
        "n_features": len(numeric_cols),
    }


def main():
    print("BorutaSHAP Economic Noise Floor — Before/After Comparison")
    print(f"Data: {DATA}")

    # Step 1 — baseline (no economic floor)
    baseline = run("BASELINE (no floor)", floor_pct=0.0)

    # Step 2 — economic floor at 2%
    economic = run("ECONOMIC FLOOR (2% of top feature)", floor_pct=0.02)

    # --- Delta Report ---
    print(f"\n{'='*64}")
    print("  DELTA REPORT")
    print(f"{'='*64}")

    kept = baseline["locked_set"] & economic["locked_set"]
    lost = baseline["locked_set"] - economic["locked_set"]
    gained = economic["locked_set"] - baseline["locked_set"]

    print(f"  Baseline floor=0.00   -> confirmed={baseline['confirmed']}, locked={len(baseline['locked_set'])}")
    print(f"  Economic floor=0.02   -> confirmed={economic['confirmed']}, locked={len(economic['locked_set'])}")
    print(f"  Features kept:  {len(kept)}")
    print(f"  Features lost:  {len(lost)}")
    print(f"  Features gained: {len(gained)}")

    if lost:
        print(f"\n  LOST (dropped by 2% floor):")
        for f in sorted(lost):
            print(f"    - {f}")

    if gained:
        print(f"\n  GAINED (promoted by 2% floor):")
        for f in sorted(gained):
            print(f"    + {f}")

    # Summary verdict
    n_feat = baseline["n_features"]
    pct_confirmed_before = baseline["confirmed"] / n_feat * 100 if isinstance(baseline["confirmed"], int) else None
    pct_confirmed_after = economic["confirmed"] / n_feat * 100 if isinstance(economic["confirmed"], int) else None

    print(f"\n  Verdict:")
    if isinstance(baseline["confirmed"], int) and isinstance(economic["confirmed"], int):
        reduction = baseline["confirmed"] - economic["confirmed"]
        print(f"    Confirmation rate: {baseline['confirmed']}/{n_feat} ({pct_confirmed_before:.1f}%) -> {economic['confirmed']}/{n_feat} ({pct_confirmed_after:.1f}%)")
        print(f"    Features pruned: {reduction}")

        if 25 <= economic["confirmed"] <= 45:
            print("    THRESHOLD TUNED CORRECTLY — matches institutional-grade Boruta range (30-50 confirmed)")
        elif economic["confirmed"] < 10:
            print("    WARNING: Threshold too aggressive (<10 confirmed). Raise economic_floor_pct.")
        elif economic["confirmed"] > 55:
            print("    WARNING: Threshold too lenient (>55 confirmed). Lower economic_floor_pct.")
    else:
        print("    Could not compute confirmation rates.")

    print(f"  Elapsed: baseline={baseline['elapsed']:.1f}s, economic={economic['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
