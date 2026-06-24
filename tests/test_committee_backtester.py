"""Tests for CommitteeBacktester — Phase D.

Tests per-bar regime routing, model selection, weight blending,
WFO evaluation, and edge cases using synthetic OHLC data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.committee.committee_builder import (
    CommitteeConfig,
    RegimeAssignment,
)
from pipeline.committee.committee_backtester import (
    CommitteeBacktester,
    CommitteeFoldResult,
    CommitteeBacktestResult,
)
from pipeline.regime.regime_utils import RegimeConfig


_RNG = np.random.default_rng(42)


def _make_synthetic_ohlc(n_bars: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC data with distinct regime sections.

    Sections (equal split):
      1st fifth:   trend_up (rising, moderate vol)
      2nd fifth:   mean_reverting (oscillating, RSI extremes)
      3rd fifth:   trend_down (declining)
      4th fifth:   high_volatile (choppy, wide ranges)
      5th fifth:   sideways (flat, low vol)
    """
    rng = np.random.default_rng(seed)
    base = 1.1000
    dt_index = pd.date_range("2023-01-01", periods=n_bars, freq="1h", tz="UTC")

    sec = max(50, n_bars // 5)
    boundaries = [0, sec, 2 * sec, 3 * sec, 4 * sec, n_bars]
    n = n_bars
    price = np.zeros(n)

    # Section 1: trend_up
    s, e = boundaries[0], boundaries[1]
    price[s:e] = base + np.linspace(0, 0.008, e - s) + rng.normal(0, 0.0002, e - s).cumsum()

    # Section 2: mean_reverting
    s, e = boundaries[1], boundaries[2]
    t2 = np.linspace(0, 10 * np.pi, e - s)
    price[s:e] = base + 0.008 + 0.003 * np.sin(t2) + rng.normal(0, 0.00015, e - s)

    # Section 3: trend_down
    s, e = boundaries[2], boundaries[3]
    price[s:e] = base + 0.006 + np.linspace(0, -0.007, e - s) + rng.normal(0, 0.0002, e - s).cumsum()

    # Section 4: high_volatile
    s, e = boundaries[3], boundaries[4]
    noise = rng.normal(0, 0.0006, e - s).cumsum()
    price[s:e] = base + 0.001 + noise

    # Section 5: sideways
    s, e = boundaries[4], boundaries[5]
    price[s:e] = base + rng.normal(0, 0.00008, e - s).cumsum()

    # MID_H/MID_L with proportional spread
    bar_vol = np.abs(rng.normal(0, 0.0003, n))
    bar_vol[boundaries[3]:boundaries[4]] *= 3.0   # amplified for volatile
    bar_vol[boundaries[4]:boundaries[5]] *= 0.3    # reduced for sideways

    df = pd.DataFrame({
        "mid_c": price,
        "mid_h": price + bar_vol,
        "mid_l": price - bar_vol,
        "mid_o": np.roll(price, 1),
        "spread": np.full(n, 0.00015),
    }, index=dt_index)

    df.iloc[0, df.columns.get_loc("mid_o")] = df.iloc[0, df.columns.get_loc("mid_c")]

    # Add returns
    df["returns"] = np.log(df["mid_c"] / df["mid_c"].shift(1)).astype(np.float32)
    df.iloc[0, df.columns.get_loc("returns")] = 0.0

    return df


def _make_simple_committee() -> CommitteeConfig:
    """Create a simple 2-model committee for testing.

    trend_up → xgboost
    trend_down → xgboost
    sideways → logistic
    fallback → logistic
    """
    return CommitteeConfig(
        version=1,
        regimes={
            "trend_up": RegimeAssignment(models=["xgboost"], weights=[1.0]),
            "trend_down": RegimeAssignment(models=["xgboost"], weights=[1.0]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


def _make_multi_model_committee() -> CommitteeConfig:
    """Create a committee with multiple models per regime."""
    return CommitteeConfig(
        version=1,
        regimes={
            "trend_up": RegimeAssignment(models=["xgboost", "random_forest"], weights=[0.6, 0.4]),
            "trend_down": RegimeAssignment(models=["xgboost", "logistic"], weights=[0.7, 0.3]),
            "mean_reverting": RegimeAssignment(models=["logistic", "xgboost"], weights=[0.55, 0.45]),
            "high_volatile": RegimeAssignment(models=["random_forest"], weights=[1.0]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["xgboost"], weights=[1.0]),
    )


# ════════════════════════════════════════════════════════════════════
# Data generation
# ════════════════════════════════════════════════════════════════════

class TestSyntheticData:
    def test_ohlc_has_correct_shape(self):
        df = _make_synthetic_ohlc(4000)
        assert len(df) == 4000
        for col in ["mid_c", "mid_h", "mid_l", "mid_o", "spread", "returns"]:
            assert col in df.columns

    def test_ohlc_has_datetime_index(self):
        df = _make_synthetic_ohlc(500)
        assert isinstance(df.index, pd.DatetimeIndex) or hasattr(df.index, "tz")

    def test_sections_have_distinct_patterns(self):
        """Trend-up section should have rising prices, sideways flat, etc."""
        df = _make_synthetic_ohlc(4000)
        sec = 4000 // 5  # = 800

        # Trend-up (first section): mean return should be positive
        tu_ret = df["returns"].iloc[100:sec - 1].mean()
        assert tu_ret > 0, f"Expected positive returns in trend_up, got {tu_ret:.6f}"

        # Trend-down (third section): mean return should be negative
        td_start = 2 * sec
        td_end = 3 * sec - 1
        td_ret = df["returns"].iloc[td_start + 100:td_end].mean()
        assert td_ret < 0, f"Expected negative returns in trend_down, got {td_ret:.6f}"

        # Volatile vs sideways: higher std in volatile
        vol_start = 3 * sec
        vol_end = 4 * sec - 1
        sw_start = 4 * sec
        sw_end = 5 * sec - 1
        vol_std = df["returns"].iloc[vol_start + 50:vol_end].std()
        sw_std = df["returns"].iloc[sw_start + 50:sw_end].std()
        assert vol_std > sw_std * 1.5, f"Volatile std={vol_std:.6f} vs sideways std={sw_std:.6f}"


# ════════════════════════════════════════════════════════════════════
# CommitteeBacktester — Core construction
# ════════════════════════════════════════════════════════════════════

class TestCommitteeBacktesterConstruct:
    def test_construct(self):
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        assert bt.config is cfg
        assert bt.confidence_threshold == 0.6
        assert len(bt._trained_models) == 0

    def test_construct_with_params(self):
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(
            cfg,
            regime_cfg=RegimeConfig(adx_thresh=25.0),
            confidence_threshold=0.55,
            label_threshold=0.0002,
        )
        assert bt.regime_cfg.adx_thresh == 25.0
        assert bt.confidence_threshold == 0.55


# ════════════════════════════════════════════════════════════════════
# Feature preparation
# ════════════════════════════════════════════════════════════════════

class TestFeaturePreparation:
    def test_prepare_features_adds_indicators(self):
        df = _make_synthetic_ohlc(1000)
        bt = CommitteeBacktester(_make_simple_committee())
        out = bt._prepare_features(df)

        for col in ["adx_14", "rsi_14", "ema_20", "bbw", "atr_14", "rv_48", "macd_diff"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_prepare_features_no_nan_in_new_cols(self):
        """New indicator columns should be mostly valid after warmup."""
        df = _make_synthetic_ohlc(500)
        bt = CommitteeBacktester(_make_simple_committee())
        out = bt._prepare_features(df)

        # After warmup (bar 100+), indicators should be finite
        for col in ["ema_20", "adx_14"]:
            valid_frac = out[col].iloc[100:].notna().mean()
            assert valid_frac > 0.8, f"{col}: only {valid_frac:.1%} valid after warmup"


# ════════════════════════════════════════════════════════════════════
# Model building
# ════════════════════════════════════════════════════════════════════

class TestModelBuilding:
    def test_build_logistic(self):
        bt = CommitteeBacktester(_make_simple_committee())
        m = bt._build_model("logistic", n_features=10)
        assert m is not None
        assert hasattr(m, "fit")
        assert hasattr(m, "predict_proba")

    def test_build_all_known_types(self):
        bt = CommitteeBacktester(_make_simple_committee())
        for mtype in ["logistic", "random_forest", "decision_tree", "svm"]:
            m = bt._build_model(mtype, n_features=15)
            assert m is not None
            assert hasattr(m, "predict_proba")

    def test_build_unknown_fallback(self):
        bt = CommitteeBacktester(_make_simple_committee())
        m = bt._build_model("imaginary_model_xyz", n_features=10)
        assert m is not None  # falls back to logistic
        assert hasattr(m, "predict_proba")


# ════════════════════════════════════════════════════════════════════
# Labeling
# ════════════════════════════════════════════════════════════════════

class TestLabeling:
    def test_labels_are_3_class(self):
        df = _make_synthetic_ohlc(500)
        bt = CommitteeBacktester(_make_simple_committee())
        bt._prepare_features(df)
        labels = bt._make_labels(df)
        assert set(np.unique(labels.dropna())).issubset({0, 1, 2})
        assert len(labels) == 500

    def test_labels_near_tail_are_nan(self):
        """Last bar should have NaN label (no next return)."""
        df = _make_synthetic_ohlc(200)
        bt = CommitteeBacktester(_make_simple_committee())
        bt._prepare_features(df)
        labels = bt._make_labels(df)
        assert pd.isna(labels.iloc[-1])

    def test_labels_mirror_return_direction(self):
        """Positive return → label 2 (long), negative → label 0 (short)."""
        df = _make_synthetic_ohlc(500)
        df["returns"] = pd.Series([0.001] * 499 + [0.0], index=df.index)  # all positive
        bt = CommitteeBacktester(_make_simple_committee())
        bt._prepare_features(df)
        labels = bt._make_labels(df)
        valid = labels.dropna()
        assert (valid == 2).mean() > 0.5  # most bars should be long


# ════════════════════════════════════════════════════════════════════
# Regime prediction
# ════════════════════════════════════════════════════════════════════

class TestRegimePrediction:
    def test_predict_regimes_returns_valid_ids(self):
        df = _make_synthetic_ohlc(800)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        reg_ids = bt._predict_regimes(df)
        assert len(reg_ids) == len(df)
        assert reg_ids.dtype == np.int8
        assert set(np.unique(reg_ids)).issubset({0, 1, 2, 3, 4, 5, 6})

    def test_regimes_differ_across_sections(self):
        """Trend-up section should have different dominant regime than sideways."""
        df = _make_synthetic_ohlc(4000)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        reg_ids = bt._predict_regimes(df)
        sec = 4000 // 5

        tu_reg = reg_ids[200:sec - 1]
        sw_reg = reg_ids[4 * sec + 100:5 * sec - 1]

        tu_mode = int(np.bincount(tu_reg.astype(int)).argmax())
        sw_mode = int(np.bincount(sw_reg.astype(int)).argmax())
        assert tu_mode != sw_mode, f"trend_up mode={tu_mode}, sideways mode={sw_mode}"


# ════════════════════════════════════════════════════════════════════
# Prediction blending
# ════════════════════════════════════════════════════════════════════

class TestBlendPredictions:
    def test_blend_returns_3col_proba(self):
        df = _make_synthetic_ohlc(1000)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        # Train models manually
        exclude = {"regime_7class", "regime_name", "regime_id",
                    "regime_trend", "regime_sideways", "regime_volatile",
                    "time", "returns", "spread", "label"}
        feat_cols = [c for c in df.columns
                     if c not in exclude and df[c].dtype in (np.float32, np.float64, np.int32, np.int64, np.int8)]
        X = df[feat_cols].fillna(0.0).to_numpy(np.float32)
        y = bt._make_labels(df).fillna(1).astype(np.float64).to_numpy(np.int32)

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_train = X[valid][:len(X) // 2]
        y_train = y[valid][:len(X) // 2]

        bt._trained_models["logistic"] = bt._build_model("logistic", n_features=X.shape[1])
        bt._trained_models["logistic"].fit(X_train, y_train)
        bt._trained_models["xgboost"] = bt._build_model("xgboost", n_features=X.shape[1])
        bt._trained_models["xgboost"].fit(X_train, y_train)

        # Train regime classifier
        bt._regime_clf = bt._train_regime_classifier(df)

        # Predict
        X_test = X[valid][len(X) // 2:]
        regime_ids = bt._predict_regimes(df.iloc[valid][len(X) // 2:])

        blended = bt._blend_predictions(X_test, regime_ids)
        assert blended.shape == (len(X_test), 3)
        assert (blended >= 0).all()
        assert (blended <= 1).all()

    def test_proba_to_trade(self):
        bt = CommitteeBacktester(_make_simple_committee(), confidence_threshold=0.6)
        proba = np.array([
            [0.3, 0.2, 0.5],  # max=0.5 < 0.6 → no trade
            [0.1, 0.2, 0.7],  # long
            [0.8, 0.1, 0.1],  # short
            [0.2, 0.6, 0.2],  # flat (max is flat class)
        ])
        trades = bt._proba_to_trade(proba)
        assert trades[0] == 0.0
        assert trades[1] == 1.0
        assert trades[2] == -1.0
        assert trades[3] == 0.0  # flat class max → no trade

    def test_proba_to_trade_edge_threshold(self):
        bt = CommitteeBacktester(_make_simple_committee(), confidence_threshold=0.5)
        proba = np.array([[0.3, 0.2, 0.5]])
        trades = bt._proba_to_trade(proba)
        assert trades[0] == 1.0  # exactly at threshold → long

    def test_blend_single_model_same_as_direct(self):
        """When committee has 1 model per regime, blended should match that model."""
        df = _make_synthetic_ohlc(400)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        feat_cols = [c for c in df.columns
                     if c not in {"time", "returns", "spread", "label",
                                  "regime_7class", "regime_name", "regime_id",
                                  "regime_trend", "regime_sideways", "regime_volatile"}
                     and df[c].dtype in (np.float32, np.float64, np.int32, np.int64)]

        X = df[feat_cols].fillna(0.0).to_numpy(np.float32)
        y = bt._make_labels(df).fillna(1).to_numpy(np.int32)

        # Train logistic
        bt._trained_models["logistic"] = bt._build_model("logistic", n_features=X.shape[1])
        bt._trained_models["logistic"].fit(X, y)

        # Predict regimes — force all as sideways
        regime_ids = np.full(len(df), 6, dtype=np.int8)

        blended = bt._blend_predictions(X, regime_ids)
        direct = bt._trained_models["logistic"].predict_proba(X)

        # When single model & weight=1.0, blended should match direct
        assert np.allclose(blended, direct, atol=0.01)


# ════════════════════════════════════════════════════════════════════
# Single fold evaluation
# ════════════════════════════════════════════════════════════════════

class TestSingleFold:
    def test_evaluate_fold_returns_result(self):
        df = _make_synthetic_ohlc(1500)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        train = df.iloc[:1000]
        test = df.iloc[1000:1450]

        result = bt._evaluate_fold(0, train, test)
        assert result is not None
        assert result.fold_idx == 0
        assert result.trades >= 0
        assert 0.0 <= result.active_rate <= 1.0

    def test_evaluate_fold_with_multi_model(self):
        df = _make_synthetic_ohlc(2000)
        bt = CommitteeBacktester(_make_multi_model_committee())
        df = bt._prepare_features(df)

        train = df.iloc[:1400]
        test = df.iloc[1400:1900]

        result = bt._evaluate_fold(0, train, test)
        assert result is not None
        assert len(bt._trained_models) >= 2
        assert result.sharpe is not None

    def test_evaluate_fold_train_too_short(self):
        df = _make_synthetic_ohlc(200)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        train = df.iloc[:30]
        test = df.iloc[30:150]

        result = bt._evaluate_fold(0, train, test)
        assert result is None

    def test_evaluate_fold_regime_distribution(self):
        df = _make_synthetic_ohlc(2000)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        train = df.iloc[:1400]
        test = df.iloc[1400:1900]

        result = bt._evaluate_fold(0, train, test)
        assert result is not None
        assert len(result.regime_distribution) > 0
        assert np.isclose(sum(result.regime_distribution.values()), 1.0, atol=0.05)

    def test_evaluate_fold_no_nan_predictions(self):
        """Most predictions should be non-NaN."""
        df = _make_synthetic_ohlc(2000)
        bt = CommitteeBacktester(_make_simple_committee())
        df = bt._prepare_features(df)

        train = df.iloc[:1400]
        test = df.iloc[1400:1900]

        result = bt._evaluate_fold(0, train, test)
        assert result is not None
        assert result.num_nan_predictions < len(test) * 0.3  # < 30% NaN


# ════════════════════════════════════════════════════════════════════
# Full WFO run
# ════════════════════════════════════════════════════════════════════

class TestWFO:
    def test_run_wfo_simple_committee(self):
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(_make_simple_committee())
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)

        assert result.total_folds > 0
        assert len(result.models) >= 2
        assert result.avg_sharpe is not None
        assert not np.isnan(result.avg_sharpe)

    def test_run_wfo_multi_model(self):
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(_make_multi_model_committee())
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)

        assert result.total_folds > 0
        assert len(result.models) >= 3  # logistic, xgboost, random_forest
        for f in result.folds:
            assert f.trades >= 0

    def test_run_wfo_produces_per_fold_metrics(self):
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(_make_simple_committee())
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)

        for f in result.folds:
            assert f.trades >= 0
            assert 0.0 <= f.active_rate <= 1.0
            assert len(f.regime_distribution) > 0

    def test_run_wfo_with_custom_regime_config(self):
        df = _make_synthetic_ohlc(6000)
        cfg = _make_simple_committee()
        rc = RegimeConfig(adx_thresh=30.0, rsi_high=75, rsi_low=25)
        bt = CommitteeBacktester(cfg, regime_cfg=rc)
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)
        assert result.total_folds > 0

    def test_run_wfo_all_regimes_covered(self):
        """With multi-model committee, the backtester should exercise all regime routes."""
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(_make_multi_model_committee())
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)

        regimes_seen = set()
        for f in result.folds:
            regimes_seen.update(f.per_model_active_fraction.keys())
        # At least a few different regimes should appear
        assert len(regimes_seen) >= 2

    def test_run_wfo_empty_df_raises(self):
        bt = CommitteeBacktester(_make_simple_committee())
        empty = pd.DataFrame()
        with pytest.raises(Exception):
            bt.run_wfo(empty, verbose=False)


# ════════════════════════════════════════════════════════════════════
# Metrics computation
# ════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_compute_metrics_no_trades(self):
        df = pd.DataFrame({
            "returns": np.random.randn(100) * 0.0001,
            "pred": np.zeros(100),
        })
        bt = CommitteeBacktester(_make_simple_committee())
        result = bt._compute_metrics(df)
        assert result is not None
        sharpe, trades, active_rate, _, _, _ = result
        assert trades < 5
        assert active_rate < 0.1

    def test_compute_metrics_with_trades(self):
        """Active predictions should produce reasonable metrics."""
        n = 500
        rets = np.random.randn(n) * 0.0003
        preds = np.where(rets > 0, 1.0, -1.0)
        df = pd.DataFrame({"returns": rets, "pred": preds})
        bt = CommitteeBacktester(_make_simple_committee())
        sharpe, trades, active_rate, win_rate, _, _ = bt._compute_metrics(df)
        assert trades > 0
        # With 1-bar delay, perfect foresight still has noise — but should be decent
        assert win_rate > 0.35

    def test_compute_metrics_handles_nan_returns(self):
        df = pd.DataFrame({
            "returns": [0.001, np.nan, -0.001, 0.002],
            "pred": [1.0, 1.0, -1.0, 1.0],
        })
        bt = CommitteeBacktester(_make_simple_committee())
        result = bt._compute_metrics(df)
        assert result is not None
        _, trades, _, _, _, _ = result
        assert trades >= 0


# ════════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_end_to_end_simple(self):
        """Full flow: OHLC → features → regime → committee → WFO → metrics."""
        df = _make_synthetic_ohlc(6000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)

        assert result.total_folds >= 2
        assert len(result.models) == 2

        # Avg Sharpe should be somewhat reasonable for synthetic data
        assert -5.0 < result.avg_sharpe < 10.0

    def test_print_summary_does_not_crash(self, capsys):
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(_make_simple_committee())
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)
        bt.print_summary(result)
        out = capsys.readouterr().out
        assert "COMMITTEE BACKTESTER" in out

    def test_to_summary_dict(self):
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(_make_simple_committee())
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)
        d = result.to_summary_dict()
        for key in ["models", "folds", "avg_sharpe", "avg_trades", "execution_time_s"]:
            assert key in d

    def test_committee_with_fallback_only(self):
        """Committee with only fallback and no regime assignments."""
        cfg = CommitteeConfig(
            version=1,
            regimes={},
            fallback=RegimeAssignment(models=["logistic", "xgboost"], weights=[0.5, 0.5]),
        )
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(cfg)
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)
        assert result.total_folds > 0
        for f in result.folds:
            assert f.trades >= 0

    def test_committee_with_single_model(self):
        """Trivial committee: one model for everything."""
        cfg = CommitteeConfig(
            version=1,
            regimes={
                "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        df = _make_synthetic_ohlc(6000)
        bt = CommitteeBacktester(cfg)
        result = bt.run_wfo(df, train_months=4, test_months=1, verbose=False)
        assert result.total_folds > 0
        assert len(result.models) == 1


# ════════════════════════════════════════════════════════════════════
# CommitteeFoldResult and CommitteeBacktestResult
# ════════════════════════════════════════════════════════════════════

class TestResultDataclasses:
    def test_fold_result_defaults(self):
        fr = CommitteeFoldResult(
            fold_idx=0,
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-12-31"),
            test_start=pd.Timestamp("2024-01-01"),
            test_end=pd.Timestamp("2024-02-01"),
            sharpe=0.5, trades=25, active_rate=0.12, win_rate=0.55,
            return_val=0.03, drawdown=0.02,
        )
        assert fr.sharpe == 0.5
        assert fr.trades == 25
        assert fr.regime_distribution == {}
        assert fr.per_model_active_fraction == {}

    def test_backtest_result_to_summary(self):
        cfg = _make_simple_committee()
        result = CommitteeBacktestResult(
            config=cfg, folds=[], models=["a", "b"],
            avg_sharpe=0.42, avg_trades=30.0,
            total_folds=5, execution_time_s=10.0,
        )
        s = result.to_summary_dict()
        assert s["models"] == ["a", "b"]
        assert s["folds"] == 5
        assert s["avg_sharpe"] == 0.42


# ════════════════════════════════════════════════════════════════════
# Sequence feature building (no TF dependency)
# ════════════════════════════════════════════════════════════════════

class TestSequenceFeatures:
    def test_build_sequences_shape(self):
        bt = CommitteeBacktester(_make_simple_committee())
        df = _make_synthetic_ohlc(300)
        # Add returns column (needed by _build_sequences)
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        seq = bt._build_sequences(df, seq_len=30)
        expected_rows = 300 - 30 + 1
        assert seq.shape[0] == expected_rows
        assert seq.shape[1] == 30
        assert seq.shape[2] == len(CommitteeBacktester._SEQ_FEATURE_COLS)
        assert seq.dtype == np.float32

    def test_build_sequences_too_short(self):
        bt = CommitteeBacktester(_make_simple_committee())
        df = pd.DataFrame({
            "mid_o": np.arange(10, dtype=np.float32),
            "mid_h": np.arange(10, dtype=np.float32) + 0.01,
            "mid_l": np.arange(10, dtype=np.float32) - 0.01,
            "mid_c": np.arange(10, dtype=np.float32),
            "returns": np.zeros(10, dtype=np.float32),
        })

        seq = bt._build_sequences(df, seq_len=30)
        assert seq.shape[0] == 0
        assert seq.shape[1] == 30

    def test_build_sequences_missing_columns(self):
        bt = CommitteeBacktester(_make_simple_committee())
        df = pd.DataFrame({"arbitrary": np.arange(50, dtype=np.float32)})
        seq = bt._build_sequences(df, seq_len=5)
        assert seq.shape[0] == 0

    def test_build_sequences_normalization(self):
        bt = CommitteeBacktester(_make_simple_committee())
        df = _make_synthetic_ohlc(500)
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        seq = bt._build_sequences(df, seq_len=20)
        assert seq.shape[0] > 0
        # Normalized data should have mean ~0 (not raw prices ~1.1)
        all_vals = seq.ravel()
        assert abs(all_vals.mean()) < 0.5, "mean should be near zero after normalization"
        # Range should be reasonable for z-scored data (not raw FX prices)
        assert abs(all_vals.max() - all_vals.min()) < 100.0

    def test_seq_len_configurable(self):
        bt = CommitteeBacktester(_make_simple_committee(), seq_len=10)
        df = _make_synthetic_ohlc(300)
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        seq = bt._build_sequences(df, bt.seq_len)
        assert seq.shape[1] == 10
        assert seq.shape[0] == 300 - 10 + 1


# ════════════════════════════════════════════════════════════════════
# Deep model integration tests (TF required)
# ════════════════════════════════════════════════════════════════════

def _tf_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


TF_REASON = "TensorFlow not available"


class TestCommitteeDeepModelDispatch:
    """Tests that deep models route to correct fit/predict paths."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        # Silence TF oneDNN/log spam during tests
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    def test_deep_model_types_are_class_constants(self):
        assert "cnn" in CommitteeBacktester._DEEP_MODEL_TYPES
        assert "lstm" in CommitteeBacktester._DEEP_MODEL_TYPES
        assert "gru" in CommitteeBacktester._DEEP_MODEL_TYPES
        assert "gru_lstm" in CommitteeBacktester._DEEP_MODEL_TYPES
        assert "transformer" in CommitteeBacktester._DEEP_MODEL_TYPES
        assert "logistic" not in CommitteeBacktester._DEEP_MODEL_TYPES
        assert "xgboost" not in CommitteeBacktester._DEEP_MODEL_TYPES
        assert CommitteeBacktester._ENSEMBLE_ADAPTIVE == "ensemble_adaptive_regime"

    @pytest.mark.skipif(not _tf_available(), reason=TF_REASON)
    def test_lstm_model_gets_sequence_input(self):
        cfg = CommitteeConfig(
            regimes={"trend_up": RegimeAssignment(models=["lstm"], weights=[1.0])},
            fallback=RegimeAssignment(models=["lstm"], weights=[1.0]),
        )
        bt = CommitteeBacktester(cfg, seq_len=5, model_params={
            "lstm": {"units": 4, "epochs": 1, "dropout": 0.0,
                     "use_early_stopping": False},
        })
        df = _make_synthetic_ohlc(4000)
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        result = bt.run_wfo(df, train_months=4, test_months=1)
        assert result is not None
        assert result.total_folds >= 1

    @pytest.mark.skipif(not _tf_available(), reason=TF_REASON)
    def test_hybrid_committee_sklearn_and_lstm(self):
        """Committee with both sklearn and LSTM members routes correctly."""
        cfg = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(
                    models=["lstm", "logistic"], weights=[0.6, 0.4]),
                "sideways": RegimeAssignment(
                    models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        bt = CommitteeBacktester(cfg, seq_len=5, model_params={
            "lstm": {"units": 4, "epochs": 1, "dropout": 0.0,
                     "use_early_stopping": False},
        })
        df = _make_synthetic_ohlc(4000)
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        result = bt.run_wfo(df, train_months=4, test_months=1)
        assert result is not None
        assert result.total_folds >= 1

    @pytest.mark.skipif(not _tf_available(), reason=TF_REASON)
    def test_ensemble_adaptive_regime_as_member(self):
        """ensemble_adaptive_regime takes (X_seq, X_flat) for predict_proba."""
        cfg = CommitteeConfig(
            regimes={"trend_up": RegimeAssignment(
                models=["ensemble_adaptive_regime"], weights=[1.0])},
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        bt = CommitteeBacktester(cfg, seq_len=5, model_params={
            "ensemble_adaptive_regime": {
                "units": 4, "epochs": 1,
                "dropout": 0.0, "use_early_stopping": False,
            },
        })
        df = _make_synthetic_ohlc(4000)
        df["returns"] = df["mid_c"].pct_change().fillna(0.0)

        result = bt.run_wfo(df, train_months=4, test_months=1)
        assert result is not None
        assert result.total_folds >= 1
