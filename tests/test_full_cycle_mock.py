"""Pipeline validation tests: full-cycle phases on synthetic mock data.

Each test uses mock_ohlc_df (1000-bar synthetic OHLC) and exercises the
same code paths as the production pipeline — no mocks, no stubs.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KODAQUANT_NO_GPU"] = "1"
os.environ["MLB_THREADS"] = "1"
os.environ["MLB_DISABLE_OPTUNA_PRUNING"] = "1"
os.environ["SKLEARN_JOBS"] = "1"

MOCK_MODELS = ["logistic", "random_forest"]


def _assert_no_nan_in_list(lst, label):
    for i, v in enumerate(lst):
        assert not (isinstance(v, float) and np.isnan(v)), f"{label}[{i}] is NaN"


# ════════════════════════════════════════════════════════════════════
# Phase 1: Feature Sweep
# ════════════════════════════════════════════════════════════════════

def test_phase1_feature_sweep(mock_ohlc_df):
    """Phase 1 on mock data: returns locked features list and sweep report."""
    from pipeline.features.feature_sweep import sweep_features

    locked, scores, report = sweep_features(
        mock_ohlc_df,
        label_threshold=0.0001,
        n_estimators=20,
        max_depth=3,
        n_folds=2,
        n_repeats=3,
        random_state=42,
    )

    assert isinstance(locked, list), f"Expected list, got {type(locked).__name__}"
    assert len(locked) >= 3, f"Expected 3+ features, got {len(locked)}"
    assert all(isinstance(f, str) for f in locked), "All locked features must be strings"

    assert isinstance(scores, dict)
    assert isinstance(report, dict)
    assert report.get("pruned_count", 0) >= 0
    assert len(report.get("fold_reports", [])) == 2

    top_5 = locked[:5]
    sensible_prefixes = (
        "returns_lag", "price_sma", "price_ema",
        "rsi_", "adx_", "macd_", "bb_pct_", "atr_", "rv_",
    )
    sensible = sum(1 for f in top_5 if any(f.startswith(p) for p in sensible_prefixes))
    assert sensible >= 1, f"Top features should include known indicators, got {top_5}"


# ── Phase 2 tests: SKIPPED — Phase 2 removed (dead code) ──

@pytest.mark.skip(reason="Phase 2 removed — profiling on default params is dead code")
def test_phase2_profiling(mock_ohlc_df):
    pass


@pytest.mark.skip(reason="Phase 2 removed — Phase 1→2 integration no longer exists")
def test_phase1_to_phase2_integration(mock_ohlc_df):
    pass


# ════════════════════════════════════════════════════════════════════
# Full Pipeline: direct call to _run_full_cycle with mock data
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_full_pipeline_direct(mock_ohlc_df, tmp_path):
    """Exercise the exact _run_full_cycle code path with mock data.

    Monkey-patches _load_csv_for_committee to return the mock DataFrame,
    creates a temporary job directory, and runs all enabled phases.
    Verifies results.json is saved with correct structure.
    """
    from unittest.mock import patch, PropertyMock
    from api.routers.committee import (
        _run_full_cycle, _load_csv_for_committee, FullCycleRequest,
        _FULL_CYCLE_DIR,
    )

    job_id = "test_full_pipeline_direct"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Write mock CSV so _load_csv_for_committee can use it
    csv_path = tmp_path / "mock_data.csv"
    df_out = mock_ohlc_df.reset_index()
    df_out.to_csv(str(csv_path), index=False)

    req = FullCycleRequest(
        models=MOCK_MODELS,
        pair="EURUSD",
        timeframe="H1",
        sweep_n_estimators=20,
        sweep_max_depth=3,
        skip_feature_sweep=False,
        debug_mode=True,
        enable_phase3=False,
        enable_phase4=False,
        enable_phase5=False,
        enable_phase6=False,
        committee_top_k=3,
        train_months=3,
        test_months=1,
        hpo_trials={"logistic": 2, "random_forest": 2},
        hpo_startup_trials={"logistic": 1, "random_forest": 1},
    )

    def _mock_load_csv(pair, timeframe):
        import pandas as _pd
        _df = _pd.read_csv(csv_path)
        _df["time"] = _pd.to_datetime(_df["time"])
        _df = _df.set_index("time")
        rename_map = {
            "mid_open": "mid_o", "mid_high": "mid_h",
            "mid_low": "mid_l", "mid_close": "mid_c",
        }
        _df = _df.rename(columns=rename_map)
        if "returns" not in _df.columns:
            _df["returns"] = _df["mid_c"].pct_change().fillna(0.0)
        return csv_path, _df

    t0 = time.time()
    with patch(
        "api.routers.committee._load_csv_for_committee",
        side_effect=_mock_load_csv,
    ):
        _run_full_cycle(job_dir, job_id, req, "2026-01-01T00:00:00")
    elapsed = time.time() - t0

    # Verify results.json was saved
    results_path = job_dir / "results.json"
    assert results_path.exists(), "results.json was not created"

    with open(results_path) as f:
        results = json.load(f)

    assert results["job_id"] == job_id
    assert results["status"] == "completed"
    assert results["locked_features_count"] > 0, "Phase 1 did not produce features"
    assert results.get("pruned_features_count", 0) >= 0
    assert len(results.get("phase0_survivors", [])) == 2
    # Phase 2 pre-screening was removed; the racecar profile matrix only exists
    # when Phase 3+ runs (all phases disabled in this fast smoke).
    assert results.get("racecar_profile_matrix") is None

    # Verify lock file is job-scoped
    lock_path = job_dir / "locked_features.json"
    assert lock_path.exists(), "locked_features.json not saved to job dir"

    with open(lock_path) as f:
        lock_data = json.load(f)
    if isinstance(lock_data, dict):
        lock_data = lock_data.get("locked_features", [])
    assert isinstance(lock_data, list)
    assert len(lock_data) >= 3

    assert elapsed < 120, f"Full cycle took {elapsed:.1f}s, expected <120s"
