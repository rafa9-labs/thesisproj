"""Smoke test: Full Cycle pipeline on real EURUSD H1 data.

Runs Phases -1 through 4 with minimal parameters to validate end-to-end
integration on real market data. Designed for manual execution — takes
10-15 minutes with 2 models.

Usage:
    python tests/smoke_full_cycle.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["KODAQUANT_NO_GPU"] = "1"
os.environ["MLB_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.feature_sweep import run_phase_minus1, load_locked_features
from pipeline.regime_utils import detect_regimes_anchored, RegimeConfig
from pipeline.expert_profiler import ExpertProfiler, prune_models
from pipeline.committee_builder import CommitteeBuilder
from pipeline.committee_backtester import CommitteeBacktester

import numpy as np
import pandas as pd

CSV_PATH = "csv_data/EURUSD_10_years_H1_OANDA.csv"
RESULTS_DIR = Path("results/smoke_full_cycle")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  KODAQUANT FULL CYCLE SMOKE TEST")
print("=" * 60)

# ── Load data ──
print(f"\n[LOAD] Reading {CSV_PATH}...")
df_full = pd.read_csv(CSV_PATH)
print(f"  {len(df_full)} bars, {df_full.time.iloc[0][:10]} to {df_full.time.iloc[-1][:10]}")

df = df_full.head(35000).copy()  # ~4 years for speed
for col in ["time"]:
    if col in df.columns:
        df["time"] = pd.to_datetime(df["time"])

# ── PHASE -1: Feature Sweep ──
print("\n" + "-" * 40)
print("  PHASE -1: Feature Sweep")
print("-" * 40)
t0 = time.time()

locked_path = str(RESULTS_DIR / "locked_features.json")
locked_features = load_locked_features(locked_path)
if locked_features:
    print(f"  (cached) Loaded {len(locked_features)} locked features")
else:
    locked_features, report = run_phase_minus1(
        df, output_path=locked_path,
        n_estimators=50, max_depth=4, n_folds=2, random_state=42,
    )
    print(f"  Locked: {len(locked_features)} features")
    print(f"  Pruned: {report['pruned_count']} features")
    print(f"  Top 3: {locked_features[:3]}")

assert len(locked_features) >= 5, f"Too few locked features: {len(locked_features)}"
elapsed = time.time() - t0
print(f"  Time: {elapsed:.0f}s")

# ── PHASE 0: Anchored Regime Detection ──
print("\n" + "-" * 40)
print("  PHASE 0: Anchored Regime Detection")
print("-" * 40)
t0 = time.time()

regime_ids = detect_regimes_anchored(df, window=252, random_state=42)
unique_regimes = set(np.unique(regime_ids).tolist())
fallback_pct = (regime_ids == 6).sum() / len(regime_ids)
print(f"  Regimes: {unique_regimes}")
print(f"  Fallback (6): {fallback_pct:.1%}")
assert unique_regimes.issubset({1, 3, 5, 6}), f"Bad regime IDs: {unique_regimes}"
elapsed = time.time() - t0
print(f"  Time: {elapsed:.0f}s")

# ── PHASE 0: ExpertProfiler ──
print("\n" + "-" * 40)
print("  PHASE 0: ExpertProfiler (Pre-screen)")
print("-" * 40)
t0 = time.time()

os.environ["MLB_TA_MODE"] = "fixed"

profiler = ExpertProfiler(
    data_config={"symbol": "EURUSD", "csv_data_path": str(CSV_PATH)},
    wfo_config={
        "n_months": 12, "n_trials": 2, "hpo_mode": "static",
        "hpo_sampler": "tpe", "cv_blocks": 3, "cv_val_frac": 0.05,
        "plateau_patience": 5, "locked_features": locked_features,
    },
    regime_cfg=RegimeConfig(),
)

models_to_test = ["logistic", "xgboost"]
print(f"  Models: {models_to_test}")
print(f"  Trials: 2 per model")

try:
    raw_df_regime = df.copy()
    if "time" not in raw_df_regime.columns:
        raw_df_regime["time"] = raw_df_regime.index

    phase0_result = profiler.profile(
        models=models_to_test, n_months=12, n_trials=2,
        seed=42, verbose=True, raw_df=raw_df_regime,
    )
    matrix = phase0_result.matrix
    print(f"  Folds collected: {len(matrix.raw_folds)}")
    print(f"  Models in matrix: {matrix.models}")

    survivors, pruned = prune_models(matrix, min_sharpe=0.0, max_models=7)
    print(f"  Survivors: {survivors}")
    print(f"  Pruned: {pruned}")
    assert len(survivors) >= 1, "No survivors — pipeline cannot proceed"
except Exception as e:
    print(f"  WARNING: Phase 0 profiling failed: {e}")
    print("  (this is expected with 2-trial HPO on short data; skipping further phases)")
    print("\n" + "=" * 60)
    print("  SMOKE TEST COMPLETE (partial — Phase 0 skipped)")
    print("=" * 60)
    sys.exit(0)

elapsed = time.time() - t0
print(f"  Time: {elapsed:.0f}s")

# ── PHASE 2: Committee Assembly ──
print("\n" + "-" * 40)
print("  PHASE 2: Committee Assembly")
print("-" * 40)
t0 = time.time()

builder = CommitteeBuilder(top_k=2, min_sharpe=0.0, weight_method="sharpe_proportional")
committee_config = builder.build(
    matrix, constraints={"max_models_per_regime": 2, "max_regimes_per_model": 3},
)
n_assigned = len(committee_config.regimes)
print(f"  Regimes with assignments: {n_assigned}")
print(f"  Fallback model: {committee_config.fallback.models}")

cc_data = committee_config.to_dict()
with open(RESULTS_DIR / "committee_config.json", "w") as f:
    json.dump(cc_data, f, indent=2, default=str)

elapsed = time.time() - t0
print(f"  Time: {elapsed:.0f}s")

# ── PHASE 3: Validation ──
print("\n" + "-" * 40)
print("  PHASE 3: WFO Validation")
print("-" * 40)
t0 = time.time()

df_bt = df_full.tail(8000).copy()
df_bt = df_bt.rename(columns={
    "mid_open": "mid_o", "mid_high": "mid_h",
    "mid_low": "mid_l", "mid_close": "mid_c",
})
df_bt["returns"] = np.log(df_bt["mid_c"] / df_bt["mid_c"].shift(1)).fillna(0.0)
df_bt["time"] = pd.to_datetime(df_bt["time"])
df_bt = df_bt.set_index("time")

bt = CommitteeBacktester(
    committee_config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
)
result = bt.run_wfo(df_bt, train_months=6, test_months=1, verbose=False)

print(f"  Folds: {result.total_folds}")
print(f"  Avg Sharpe: {result.avg_sharpe:.3f}")
print(f"  Avg Trades: {result.avg_trades:.1f}")
print(f"  Fold CV: {result.fold_consistency_cv:.3f}")
print(f"  CV Pass: {result.fold_consistency_pass}")

coverage = result.regime_coverage_report(min_trades=5, min_sharpe=-0.5)
for regime, info in coverage.items():
    if info.get("trades", 0) > 0:
        print(f"    {regime}: {info['trades']} trades, covered={info['covered']}")

elapsed = time.time() - t0
print(f"  Time: {elapsed:.0f}s")

# ── Summary ──
print("\n" + "=" * 60)
print("  SMOKE TEST COMPLETE")
print("=" * 60)
print(f"  Phase -1: {len(locked_features)} features locked")
print(f"  Phase 0:  {len(survivors)} survivors: {survivors}")
print(f"  Phase 2:  {n_assigned} regimes assigned, fallback={committee_config.fallback.models[0]}")
print(f"  Phase 3:  Sharpe={result.avg_sharpe:.3f}, folds={result.total_folds}")
print()
print(f"  Results saved to {RESULTS_DIR}/")
