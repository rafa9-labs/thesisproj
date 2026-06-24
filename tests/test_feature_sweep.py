"""Tests for Phase -1 feature sweep (grid expansion + shallow RF + permutation importance)."""
import sys
import os
import json

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.features.feature_sweep import (
    expand_features,
    sweep_features,
    save_locked_features,
    load_locked_features,
    run_phase_minus1,
    INDICATOR_GRID,
    RETURNS_LAGS,
)

from pipeline.regime.regime_utils import detect_regimes_anchored


class TestExpandFeatures:

    @staticmethod
    def _make_ohlc_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = np.cumsum(rng.normal(0, 0.0008, n)) + 1.10
        close = np.maximum(close, 0.5)
        high = close + np.abs(rng.normal(0, 0.0003, n))
        low = close - np.abs(rng.normal(0, 0.0003, n))
        idx = pd.date_range("2018-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "time": idx,
            "mid_high": high,
            "mid_low": low,
            "mid_close": close,
        })

    def test_expand_adds_indicator_columns(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        required = ["returns", "sma_20", "ema_20", "rsi_14", "adx_14",
                    "atr_14", "bb_upper_20", "macd_diff"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_expand_all_sma_windows(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        for w in INDICATOR_GRID["sma"]:
            assert f"sma_{w}" in result.columns

    def test_expand_all_ema_windows(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        for w in INDICATOR_GRID["ema"]:
            assert f"ema_{w}" in result.columns

    def test_expand_all_rsi_windows(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        for w in INDICATOR_GRID["rsi"]:
            assert f"rsi_{w}" in result.columns

    def test_expand_all_bbands_variants(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        for w in INDICATOR_GRID["bbands"]:
            for variant in ["bb_upper", "bb_lower", "bbw", "bb_pct"]:
                assert f"{variant}_{w}" in result.columns

    def test_expand_returns_lags(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        for lag in RETURNS_LAGS:
            assert f"returns_lag{lag}" in result.columns

    def test_expand_adds_donchian(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        assert "donchian_up_20" in result.columns
        assert "donchian_break_up_20" in result.columns

    def test_expand_realized_vol(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        assert "rv_48" in result.columns
        assert "rv_240" in result.columns

    def test_expand_normalizes_column_names(self):
        df = self._make_ohlc_df(2000)
        df = df.rename(columns={"mid_high": "mid_h", "mid_low": "mid_l",
                                "mid_close": "mid_c"})
        result = expand_features(df)
        assert "sma_20" in result.columns

    def test_total_feature_count_reasonable(self):
        df = self._make_ohlc_df(2000)
        result = expand_features(df)
        feat_cols = [c for c in result.columns if c.startswith(("sma_", "ema_", "rsi_",
                        "adx_", "atr_", "bb_", "bbw_", "bb_pct_", "donchian_",
                        "macd_", "rv_", "price_sma", "price_ema", "returns_lag"))]
        assert len(feat_cols) > 30, f"Expected >30 feature columns, got {len(feat_cols)}"
        assert len(feat_cols) < 150, f"Expected <150 feature columns, got {len(feat_cols)}"


class TestSweepFeatures:

    @staticmethod
    def _make_ohlc_df(n: int = 3000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        tr = rng.normal(0, 0.0008, n)
        # Inject a directional signal: lagged returns slightly predict next
        signal = 0.6 * np.roll(tr, 1) + 0.4 * rng.normal(0, 0.0005, n)
        signal[:5] = tr[:5]
        close = np.cumsum(signal) + 1.10
        close = np.maximum(close, 0.5)
        high = close + np.abs(rng.normal(0, 0.0003, n))
        low = close - np.abs(rng.normal(0, 0.0003, n))
        idx = pd.date_range("2019-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "time": idx,
            "mid_high": high,
            "mid_low": low,
            "mid_close": close,
        })

    def test_sweep_returns_locked_features(self):
        df = self._make_ohlc_df(2000)
        locked, scores, report = sweep_features(
            df, n_estimators=50, max_depth=4, n_folds=2, n_repeats=3,
        )
        assert isinstance(locked, list)
        assert len(locked) > 0
        assert isinstance(scores, dict)

    def test_sweep_prunes_some_features(self):
        df = self._make_ohlc_df(2000)
        locked, scores, report = sweep_features(
            df, n_estimators=50, max_depth=4, n_folds=2, n_repeats=3,
        )
        assert report["pruned_count"] > 0, "Expected some features to be pruned"

    def test_locked_features_subset_of_total(self):
        df = self._make_ohlc_df(2000)
        locked, scores, report = sweep_features(
            df, n_estimators=50, max_depth=4, n_folds=2, n_repeats=3,
        )
        assert report["locked_count"] <= report["total_features"]
        assert report["locked_count"] + report["pruned_count"] == report["total_features"]

    def test_sweep_deterministic(self):
        df = self._make_ohlc_df(2000)
        locked1, _, _ = sweep_features(df, n_estimators=50, max_depth=4,
                                       n_folds=2, n_repeats=2, random_state=42)
        locked2, _, _ = sweep_features(df, n_estimators=50, max_depth=4,
                                       n_folds=2, n_repeats=2, random_state=42)
        assert locked1 == locked2

    def test_importance_scores_sorted(self):
        df = self._make_ohlc_df(2000)
        locked, scores, report = sweep_features(
            df, n_estimators=50, max_depth=4, n_folds=2, n_repeats=3,
        )
        locked_imps = [scores.get(f, 0) for f in locked]
        for i in range(1, len(locked_imps)):
            assert locked_imps[i - 1] >= locked_imps[i], "Locked features not sorted descending"


class TestSaveLoadFeatures:

    def test_save_and_load_roundtrip(self, tmp_path):
        features = ["sma_20", "ema_50", "rsi_14", "adx_28"]
        path = str(tmp_path / "test_locked.json")
        save_locked_features(features, path)
        loaded = load_locked_features(path)
        assert loaded == features

    def test_load_missing_file_returns_none(self):
        result = load_locked_features("nonexistent_path.json")
        assert result is None


class TestRunPhaseMinus1:

    @staticmethod
    def _make_ohlc_df(n: int = 3000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        tr = rng.normal(0, 0.0008, n)
        close = np.cumsum(tr) + 1.10
        close = np.maximum(close, 0.5)
        high = close + np.abs(rng.normal(0, 0.0003, n))
        low = close - np.abs(rng.normal(0, 0.0003, n))
        idx = pd.date_range("2019-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "time": idx,
            "mid_high": high,
            "mid_low": low,
            "mid_close": close,
        })

    def test_run_phase_minus1_saves_json(self, tmp_path):
        df = self._make_ohlc_df(2000)
        out_path = str(tmp_path / "locked_features.json")
        locked, report = run_phase_minus1(
            df, output_path=out_path, n_estimators=50,
            max_depth=4, n_folds=2,
        )
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "locked_features" in data
        assert len(data["locked_features"]) == len(locked)

    def test_run_phase_minus1_saves_report(self, tmp_path):
        df = self._make_ohlc_df(2000)
        out_path = str(tmp_path / "locked_features.json")
        locked, report = run_phase_minus1(
            df, output_path=out_path, n_estimators=50,
            max_depth=4, n_folds=2,
        )
        report_path = out_path.replace(".json", "_report.json")
        assert os.path.exists(report_path)


class TestIntegrationWithRegimeDetection:

    @staticmethod
    def _make_ohlc_df(n: int = 3000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        tr = rng.normal(0, 0.0008, n)
        close = np.cumsum(tr) + 1.10
        close = np.maximum(close, 0.5)
        high = close + np.abs(rng.normal(0, 0.0003, n))
        low = close - np.abs(rng.normal(0, 0.0003, n))
        idx = pd.date_range("2019-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "time": idx,
            "mid_high": high,
            "mid_low": low,
            "mid_close": close,
        })

    def test_sweep_then_detect_regimes(self):
        """Feature sweep should not interfere with anchored regime detection."""
        df = self._make_ohlc_df(2000)
        locked, _, _ = sweep_features(df, n_estimators=50, max_depth=4,
                                      n_folds=2, n_repeats=2)
        assert len(locked) > 0
        regime_ids = detect_regimes_anchored(df, window=250, random_state=42)
        assert set(np.unique(regime_ids).tolist()).issubset({1, 3, 5, 6})
