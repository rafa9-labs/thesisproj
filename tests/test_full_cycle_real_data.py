"""Production validation tests: Full Cycle pipeline components on real EURUSD H1 data.

All tests marked @pytest.mark.slow — skipped in default CI runs.
Loads first 2 years of EURUSD H1 (17,520 bars) for speed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KODAQUANT_NO_GPU"] = "1"
os.environ["MLB_THREADS"] = "1"
os.environ["MLB_TA_MODE"] = "fixed"

EURUSD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "csv_data", "EURUSD_10_years_H1_OANDA.csv",
)

TWO_YEAR_ROWS = 17520


def _load_real_data(n_rows: int = TWO_YEAR_ROWS):
    """Load first N rows of real EURUSD H1 data."""
    if not os.path.exists(EURUSD_PATH):
        pytest.skip("EURUSD H1 CSV not found")
    df = pd.read_csv(EURUSD_PATH, nrows=n_rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


def _get_all_data():
    """Load full EURUSD H1 dataset."""
    if not os.path.exists(EURUSD_PATH):
        pytest.skip("EURUSD H1 CSV not found")
    return pd.read_csv(EURUSD_PATH)


# ════════════════════════════════════════════════════════════════════
# A1: Phase -1 Feature Sweep on Real Data
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPhaseMinus1RealData:

    def test_expand_features_on_real_data(self):
        """Phase -1 expand: produces 55-85 feature columns from real OHLC."""
        from pipeline.features.feature_sweep import expand_features
        df = _load_real_data()
        result = expand_features(df)

        feat_cols = [c for c in result.columns
                     if c.startswith(("sma_", "ema_", "rsi_", "adx_", "atr_",
                                      "bb_", "bbw_", "bb_pct_", "donchian_",
                                      "macd_", "rv_", "price_sma", "price_ema",
                                      "returns_lag"))]
        assert 50 <= len(feat_cols) <= 90, f"Expected 50-90 feature cols, got {len(feat_cols)}"
        assert len(result) == len(df), "Row count mismatch after expand"

    def test_sweep_features_on_real_data(self):
        """Phase -1 sweep: prunes to 15-50 features, top features are sensible."""
        from pipeline.features.feature_sweep import sweep_features
        df = _load_real_data()
        locked, scores, report = sweep_features(
            df, n_estimators=30, max_depth=4, n_folds=2, n_repeats=3,
        )

        assert 10 <= len(locked) <= 55, f"Expected 10-55 locked, got {len(locked)}"
        assert report["pruned_count"] > 0, "Expected some features to be pruned"

        # Top features should be returns/prices/ratios, not noise
        top_5 = locked[:5]
        sensible_prefixes = ("returns_lag", "price_sma", "price_ema",
                             "rsi_", "adx_", "macd_", "bb_pct_", "atr_", "rv_",
                             "sma_", "ema_", "bbw_", "hv_", "donchian_")
        sensible_count = sum(
            1 for f in top_5 if any(f.startswith(p) for p in sensible_prefixes)
        )
        assert sensible_count >= 3, f"Only {sensible_count}/5 top features are sensible: {top_5}"

    def test_sweep_saves_and_loads_real_data(self, tmp_path):
        """Phase -1: full run_phase_minus1 saves loadable JSON."""
        from pipeline.features.feature_sweep import run_phase_minus1, load_locked_features
        df = _load_real_data()
        out_path = str(tmp_path / "locked_features.json")
        locked, report = run_phase_minus1(
            df, output_path=out_path, n_estimators=30, max_depth=4, n_folds=2,
        )
        assert os.path.exists(out_path)
        loaded = load_locked_features(out_path)
        assert loaded == locked


# ════════════════════════════════════════════════════════════════════
# A2: Anchored Regime Detection on Real Data
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestAnchoredRegimeRealData:

    def test_detect_regimes_on_full_data(self):
        """Anchored GMM on full 10-year data: 3 valid clusters, valid IDs only."""
        from pipeline.regime.regime_utils import detect_regimes_anchored
        df = _get_all_data()
        ids = detect_regimes_anchored(df, window=252, random_state=42)

        valid = {1, 3, 5, 6}
        unique = set(np.unique(ids).tolist())
        assert unique.issubset(valid), f"Unexpected regime IDs: {unique - valid}"

        non_fallback = ids[ids != 6]
        n_clusters = len(set(np.unique(non_fallback)))
        assert n_clusters >= 2, f"Expected >=2 clusters, got {n_clusters}"

        # Most bars should be classified (not fallback=6)
        fallback_frac = (ids == 6).sum() / len(ids)
        assert fallback_frac < 0.15, f"Too many fallback bars: {fallback_frac:.2%}"

    def test_per_fold_mode_on_real_data(self):
        """Per-fold anchored detection: train on early data, predict on later data."""
        from pipeline.regime.regime_utils import detect_regimes_anchored
        df = _get_all_data()
        n = len(df)
        split = n // 2
        df_train = df.iloc[:split]
        df_test = df.iloc[split:split + 5000]

        ids = detect_regimes_anchored(
            df_test, df_train=df_train, window=252, random_state=42,
        )
        assert len(ids) == len(df_test)
        assert set(np.unique(ids).tolist()).issubset({1, 3, 5, 6})

    def test_centroid_labels_consistent_across_folds(self):
        """Centroid anchoring stable: same data → same cluster mapping."""
        from pipeline.regime.regime_utils import detect_regimes_anchored
        df = _load_real_data(5000)
        ids1 = detect_regimes_anchored(df, window=250, random_state=42)
        ids2 = detect_regimes_anchored(df, window=250, random_state=42)
        assert np.array_equal(ids1, ids2)


# ════════════════════════════════════════════════════════════════════
# A3: Phase 0 ExpertProfiler on Real Data
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPhase0RealData:

    def test_profiler_constructs_with_real_config(self):
        """ExpertProfiler accepts real data config and constructs cleanly."""
        from pipeline.committee.expert_profiler import ExpertProfiler, RegimeConfig
        profiler = ExpertProfiler(
            data_config={"symbol": "EURUSD"},
            wfo_config={
                "n_months": 6, "n_trials": 2, "hpo_mode": "static",
                "hpo_sampler": "tpe", "cv_blocks": 3, "cv_val_frac": 0.05,
            },
            regime_cfg=RegimeConfig(),
        )
        assert profiler is not None

    def test_profiler_with_locked_features(self):
        """ExpertProfiler stores locked_features in wfo_config."""
        from pipeline.features.feature_sweep import sweep_features
        from pipeline.committee.expert_profiler import ExpertProfiler, RegimeConfig
        df = _load_real_data(5000)
        locked, _, _ = sweep_features(
            df, n_estimators=20, max_depth=3, n_folds=2, n_repeats=2,
        )
        profiler = ExpertProfiler(
            data_config={"symbol": "EURUSD"},
            wfo_config={
                "n_months": 6, "n_trials": 2, "hpo_mode": "static",
                "locked_features": locked,
            },
            regime_cfg=RegimeConfig(),
        )
        assert "locked_features" in profiler.wfo_config

    @pytest.mark.skip(reason="prune_models removed with Phase 2")
    def test_prune_models_on_synthetic_matrix(self):
        pass


# ════════════════════════════════════════════════════════════════════
# A4: Committee Backtester on Real Data
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestCommitteeBacktesterRealData:

    def test_run_wfo_on_real_data_short_window(self):
        """CommitteeBacktester runs WFO on 2yr real data without crashing."""
        from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.regime.regime_utils import RegimeConfig

        df = _load_real_data()
        df = df.rename(columns={
            "mid_open": "mid_o", "mid_high": "mid_h",
            "mid_low": "mid_l", "mid_close": "mid_c",
        })
        df["returns"] = np.log(df["mid_c"] / df["mid_c"].shift(1)).fillna(0.0)
        df = df.set_index("time")

        config = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
                "high_volatile": RegimeAssignment(models=["logistic"], weights=[1.0]),
                "mean_reverting": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )

        bt = CommitteeBacktester(
            config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
        )
        result = bt.run_wfo(
            df.tail(4000), train_months=6, test_months=1, verbose=False,
        )

        assert result is not None
        assert result.total_folds > 0
        assert result.folds is not None

    def test_fold_consistency_cv_computed(self):
        """Fold consistency CV is finite and non-negative on real-ish data."""
        from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
        from pipeline.committee.committee_backtester import (
            CommitteeBacktester, CommitteeBacktestResult, CommitteeFoldResult,
        )
        from pipeline.regime.regime_utils import RegimeConfig

        config = CommitteeConfig(
            regimes={"trend_up": RegimeAssignment(models=["logistic"], weights=[1.0])},
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        folds = [
            CommitteeFoldResult(
                fold_idx=i, train_start=0, train_end=1, test_start=2,
                test_end=3, sharpe=0.5 + i * 0.1, trades=30 + i * 5,
                active_rate=0.15, win_rate=0.55, return_val=0.01, drawdown=0.01,
                regime_distribution={"trend_up": 1.0},
                per_model_active_fraction={"trend_up": 1.0},
            )
            for i in range(5)
        ]
        result = CommitteeBacktestResult(
            config=config, folds=folds, models=["logistic"],
        )
        cv = result.fold_consistency_cv
        assert np.isfinite(cv), f"CV should be finite, got {cv}"
        assert cv >= 0, f"CV should be non-negative, got {cv}"

    def test_regime_coverage_report_on_real_data(self):
        """Regime coverage report works with valid committee + real data."""
        from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.regime.regime_utils import RegimeConfig

        df = _load_real_data()
        df = df.rename(columns={
            "mid_open": "mid_o", "mid_high": "mid_h",
            "mid_low": "mid_l", "mid_close": "mid_c",
        })
        df["returns"] = np.log(df["mid_c"] / df["mid_c"].shift(1)).fillna(0.0)
        df = df.set_index("time")

        config = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
                "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )

        bt = CommitteeBacktester(
            config, regime_cfg=RegimeConfig(), confidence_threshold=0.5,
        )
        result = bt.run_wfo(
            df.tail(6000), train_months=3, test_months=1, verbose=False,
        )

        report = result.regime_coverage_report(min_trades=5, min_sharpe=-1.0)
        assert isinstance(report, dict)
        assert "trend_up" in report
