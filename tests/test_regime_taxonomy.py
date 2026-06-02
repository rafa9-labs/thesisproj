"""Tests for 7-class regime detection and RegimeClassifier model.

Phase A of the Multi-Agent Autonomous Exploration Engine.
Covers: rule-based detection, DataFrame augmentation, RF training/inference,
edge cases, and legacy compatibility.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.regime_utils import (
    detect_regimes,
    attach_regime_columns,
    regime_counts,
    regime_transition_matrix,
    to_legacy,
    recommended_models_for_regime,
    RegimeConfig,
    _REGIME_NAMES,
    _LEGACY_MAP,
    _resolve_col,
    _COLUMN_FALLBACKS,
)


_RNG = np.random.default_rng(42)
_N = 500


def _make_ohlc(n: int = _N) -> pd.DataFrame:
    """Generate realistic synthetic OHLC data with strong regime signatures."""
    base = 1.1000
    sec = n // 7

    trend_up  = np.linspace(base, base + 0.010, sec)
    trend_dn  = np.linspace(base + 0.010, base + 0.005, sec)
    volatile  = base + 0.005 + np.cumsum(np.random.randn(sec) * 0.0010)
    squeeze   = np.full(sec, base + 0.005) + np.random.randn(sec) * 0.00005
    breakout  = np.linspace(base + 0.005, base + 0.012, sec * 2)
    mean_rev  = base + 0.008 + 0.003 * np.sin(np.linspace(0, 8 * np.pi, sec))

    mid_c = np.concatenate([trend_up, trend_dn, volatile, squeeze, breakout, mean_rev])
    mid_c = mid_c[:n]
    if len(mid_c) < n:
        pad = np.full(n - len(mid_c), base + 0.005 + np.random.randn() * 0.0003)
        mid_c = np.concatenate([mid_c, pad])

    noise = np.random.randn(n) * 0.0002
    mid_c += noise

    df = pd.DataFrame({
        "mid_c": mid_c,
        "mid_h": mid_c + np.abs(np.random.randn(n)) * 0.0005,
        "mid_l": mid_c - np.abs(np.random.randn(n)) * 0.0005,
        "mid_o": np.roll(mid_c, 1),
    })
    df.loc[0, "mid_o"] = df.loc[0, "mid_c"]

    # Compute indicators that reflect the regimes
    # ADX — trending proxy
    returns = df["mid_c"].diff()
    df["adx_14"] = _compute_synthetic_adx(df["mid_c"], n)

    # EMA
    df["ema_20"] = df["mid_c"].ewm(span=20, adjust=False).mean()

    # RSI
    df["rsi_14"] = _compute_synthetic_rsi(df["mid_c"], n)

    # Bollinger Bands
    bb_sma = df["mid_c"].rolling(20).mean()
    bb_std = df["mid_c"].rolling(20).std()
    df["bb_upper"] = bb_sma + 2.0 * bb_std
    df["bb_lower"] = bb_sma - 2.0 * bb_std
    df["bbw"] = (df["bb_upper"] - df["bb_lower"]) / bb_sma.replace(0, np.nan)
    df["bb_pct"] = (df["mid_c"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # ATR
    df["atr_14"] = df["mid_h"].sub(df["mid_l"]).rolling(14).mean()

    # Realized Vol
    df["rv_48"] = returns.pow(2).rolling(48).sum().pow(0.5)

    # MACD
    ema12 = df["mid_c"].ewm(span=12, adjust=False).mean()
    ema26 = df["mid_c"].ewm(span=26, adjust=False).mean()
    df["macd_diff"] = ema12 - ema26

    # Donchian
    df["donchian_up_20"] = df["mid_h"].rolling(20).max()
    df["donchian_dn_20"] = df["mid_l"].rolling(20).min()
    df["donchian_break_up_20"] = (df["mid_c"] > df["donchian_up_20"].shift(1)).astype(int)
    df["donchian_break_dn_20"] = (df["mid_c"] < df["donchian_dn_20"].shift(1)).astype(int)

    return df


def _compute_synthetic_adx(price: np.ndarray, n: int) -> np.ndarray:
    """Rough synthetic ADX — high in trend sections, low in squeeze/volatile."""
    adx = np.full(n, 15.0)
    section_len = n // 7
    # trend_up: high ADX
    adx[:section_len] = np.linspace(28, 35, section_len)
    # trend_dn: high ADX
    adx[section_len:2*section_len] = 30.0 + _RNG.uniform(0, 5, section_len)
    # volatile: mixed ADX
    adx[2*section_len:3*section_len] = 18.0 + _RNG.uniform(-5, 5, section_len)
    # squeeze: low ADX
    adx[3*section_len:4*section_len] = 10.0 + _RNG.uniform(0, 3, section_len)
    # breakout: rising ADX
    adx[4*section_len:min(6*section_len, n)] = np.linspace(15, 35, min(2*section_len, n - 4*section_len))
    # mean_reverting: moderate ADX
    adx[6*section_len:] = 18.0 + _RNG.uniform(-3, 3, n - 6*section_len if n > 6*section_len else 0)
    return adx


def _compute_synthetic_rsi(price: np.ndarray, n: int) -> np.ndarray:
    """Rough synthetic RSI — extreme in mean_reverting section."""
    rsi = np.full(n, 50.0)
    section_len = n // 7
    # trend sections: moderate-high RSI
    rsi[:section_len] = 55.0 + _RNG.uniform(0, 10, section_len)
    rsi[section_len:2*section_len] = 35.0 + _RNG.uniform(0, 10, section_len)
    # volatile: wide range
    rsi[2*section_len:3*section_len] = 50.0 + _RNG.uniform(-20, 20, section_len)
    # squeeze: centered
    rsi[3*section_len:4*section_len] = 50.0 + _RNG.uniform(-5, 5, section_len)
    # breakout: strong momentum
    rsi[4*section_len:min(6*section_len, n)] = 65.0 + _RNG.uniform(-5, 15, min(2*section_len, n - 4*section_len))
    # mean_reverting: extreme oscillates
    remaining = n - 6*section_len if n > 6*section_len else 0
    if remaining > 0:
        rsi[6*section_len:] = 50.0 + 35.0 * np.sin(np.linspace(0, 6 * np.pi, remaining))
    return rsi


# ════════════════════════════════════════════════════════════════════
# Module-level imports
# ════════════════════════════════════════════════════════════════════


class TestRegimeUtilsImports:
    def test_import(self):
        assert detect_regimes is not None
        assert attach_regime_columns is not None

    def test_constants(self):
        assert len(_REGIME_NAMES) == 7
        assert _REGIME_NAMES[0] == "quiet_squeeze"
        assert _REGIME_NAMES[6] == "sideways"
        assert len(_LEGACY_MAP) == 7

    def test_column_fallbacks(self):
        assert "adx" in _COLUMN_FALLBACKS
        assert _COLUMN_FALLBACKS["adx"] == ("adx_14", "adx_28", "adx")


# ════════════════════════════════════════════════════════════════════
# Rule-based detection tests
# ════════════════════════════════════════════════════════════════════

class TestDetectRegimes:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_ohlc()

    def test_output_shape(self, df):
        regimes = detect_regimes(df)
        assert isinstance(regimes, np.ndarray)
        assert regimes.dtype == np.int8
        assert len(regimes) == len(df)

    def test_all_valid_classes(self, df):
        regimes = detect_regimes(df)
        unique = set(np.unique(regimes))
        assert unique.issubset({0, 1, 2, 3, 4, 5, 6}), f"Unexpected classes: {unique}"
        # With our synthetic data, we should see multiple classes
        assert len(unique) >= 3, f"Expected >=3 distinct regimes, got {len(unique)}: {unique}"

    def test_no_nan_output(self, df):
        regimes = detect_regimes(df)
        assert not np.isnan(regimes).any()

    def test_default_is_sideways(self, df):
        """SIDEWAYS (6) should be the fallback for missing features."""
        empty = pd.DataFrame({"mid_c": df["mid_c"]})
        regimes = detect_regimes(empty)
        assert np.all(regimes == 6)

    def test_missing_columns_graceful(self, df):
        """Should not crash with partial columns."""
        partial = df[["mid_c", "adx_14", "ema_20"]].copy()
        regimes = detect_regimes(partial)
        assert len(regimes) == len(df)
        assert regimes.dtype == np.int8

    def test_single_row(self, df):
        regimes = detect_regimes(df.iloc[:1])
        assert len(regimes) == 1
        assert regimes[0] in (0, 1, 2, 3, 4, 5, 6)

    def test_all_nan_data(self):
        df_nan = pd.DataFrame({
            "mid_c": [np.nan] * 10,
            "adx_14": [np.nan] * 10,
            "ema_20": [np.nan] * 10,
        })
        regimes = detect_regimes(df_nan)
        assert np.all(regimes == 6)  # everything defaults to sideways

    def test_config_override(self, df):
        cfg = RegimeConfig(adx_thresh=50.0, rsi_high=95, rsi_low=5)
        regimes_strict = detect_regimes(df, cfg)
        regimes_default = detect_regimes(df)
        # Strict thresholds should produce more sideways (6)
        strict_sideways = (regimes_strict == 6).sum()
        default_sideways = (regimes_default == 6).sum()
        assert strict_sideways >= default_sideways

    def test_regime_counts_dict(self, df):
        regimes = detect_regimes(df)
        counts = regime_counts(regimes)
        assert isinstance(counts, dict)
        assert sum(counts.values()) == len(df)
        for name in _REGIME_NAMES.values():
            assert name in counts

    def test_transition_matrix_shape(self, df):
        regimes = detect_regimes(df)
        mat = regime_transition_matrix(regimes)
        assert mat.shape == (7, 7)
        assert mat.dtype == np.int64
        assert mat.sum() <= len(df) - 1

    def test_to_legacy(self, df):
        regimes = detect_regimes(df)
        legacy = to_legacy(regimes)
        assert legacy.dtype == np.int8
        assert set(np.unique(legacy)).issubset({0, 1, 2})

    def test_recommended_models(self):
        models_0 = recommended_models_for_regime(0)  # quiet_squeeze → empty/has some
        models_1 = recommended_models_for_regime(1)  # trend_up → lstm, transformer, xgb...
        assert isinstance(models_0, list)
        assert isinstance(models_1, list)
        assert len(models_1) > 0

    def test_large_dataset(self):
        n_large = 5000
        price = np.cumsum(_RNG.standard_normal(n_large) * 0.0005) + 1.1000
        df = pd.DataFrame({
            "mid_c": price,
            "adx_14": np.full(n_large, 30.0),
            "ema_20": pd.Series(price).ewm(span=20, adjust=False).mean(),
            "rsi_14": np.full(n_large, 50.0),
            "bb_pct": np.full(n_large, 0.5),
            "bbw": np.full(n_large, 0.01),
            "atr_14": np.full(n_large, 0.002),
            "rv_48": np.full(n_large, 0.0005),
            "macd_diff": np.full(n_large, 0.0),
        })
        regimes = detect_regimes(df)
        assert len(regimes) == n_large


# ════════════════════════════════════════════════════════════════════
# attach_regime_columns tests
# ════════════════════════════════════════════════════════════════════

class TestAttachRegimeColumns:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_ohlc()

    def test_adds_7class_column(self, df):
        out = attach_regime_columns(df)
        assert "regime_7class" in out.columns
        assert out["regime_7class"].dtype == np.int8

    def test_adds_regime_name_column(self, df):
        out = attach_regime_columns(df)
        assert "regime_name" in out.columns
        unique_names = set(out["regime_name"].unique())
        assert unique_names.issubset(set(_REGIME_NAMES.values()))

    def test_legacy_columns(self, df):
        out = attach_regime_columns(df, include_legacy=True)
        for col in ["regime_id", "regime_trend", "regime_sideways", "regime_volatile"]:
            assert col in out.columns

    def test_no_legacy_columns(self, df):
        out = attach_regime_columns(df, include_legacy=False)
        assert "regime_id" not in out.columns
        assert "regime_trend" not in out.columns

    def test_original_columns_preserved(self, df):
        for col in df.columns:
            assert col in attach_regime_columns(df).columns

    def test_legacy_consistency(self, df):
        """7-class → 3-class mapping should be consistent."""
        out = attach_regime_columns(df)
        for row in range(len(out)):
            k7 = out.loc[row, "regime_7class"]
            k3 = out.loc[row, "regime_id"]
            expected = _LEGACY_MAP.get(int(k7), 0)
            assert k3 == expected, f"Row {row}: 7class={k7} → 3class={k3}, expected={expected}"


# ════════════════════════════════════════════════════════════════════
# RegimeClassifier model tests
# ════════════════════════════════════════════════════════════════════

class TestRegimeClassifier:
    @pytest.fixture(scope="class")
    def df(self):
        return _make_ohlc()

    @pytest.fixture
    def labeled_df(self, df):
        """Generate a DataFrame with rule-based labels for training."""
        out = attach_regime_columns(df)
        return out

    def test_import(self):
        from models.regime_classifier import RegimeClassifier
        assert RegimeClassifier is not None

    def test_construct(self):
        from models.regime_classifier import RegimeClassifier
        clf = RegimeClassifier(n_estimators=20, max_depth=4)
        assert clf.model_type == "regime_classifier"
        assert clf.is_deep is False
        assert clf.supports_proba is True
        assert not clf.is_fitted

    def test_fit_predict(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        y = labeled_df["regime_7class"].to_numpy(dtype=np.int32)
        clf = RegimeClassifier(n_estimators=30, max_depth=6, min_samples_leaf=10, random_state=42)
        clf.fit(labeled_df, y)

        assert clf.is_fitted
        preds = clf.predict(labeled_df)
        assert len(preds) == len(labeled_df)
        assert preds.dtype == np.int8
        assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4, 5, 6})

    def test_predict_proba(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        y = labeled_df["regime_7class"].to_numpy(dtype=np.int32)
        clf = RegimeClassifier(n_estimators=30, max_depth=6, min_samples_leaf=10, random_state=42)
        clf.fit(labeled_df, y)

        n_test = 20
        proba = clf.predict_proba(labeled_df.iloc[:n_test])
        assert proba.shape == (n_test, 7)  # always 7 cols (zeros for unseen classes)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_without_fit_raises(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(labeled_df)

    def test_auto_target_from_column(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier(n_estimators=20, max_depth=4, random_state=42)
        clf.fit(labeled_df)
        assert clf.is_fitted
        preds = clf.predict(labeled_df.head(5))
        assert len(preds) == 5

    def test_valid_non_nan_fit(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        # Insert some NaN rows
        corrupt = labeled_df.copy()
        corrupt.iloc[10] = np.nan

        y = corrupt["regime_7class"].to_numpy(dtype=np.int32)
        clf = RegimeClassifier(n_estimators=20, random_state=42)
        clf.fit(corrupt, y)
        assert clf.is_fitted

    def test_all_nan_fit_raises(self):
        from models.regime_classifier import RegimeClassifier

        bad = pd.DataFrame({
            "adx_14": [np.nan] * 5,
            "rsi_14": [np.nan] * 5,
            "regime_7class": [0] * 5,
        })
        clf = RegimeClassifier(n_estimators=10)
        with pytest.raises(ValueError, match="No valid"):
            clf.fit(bad)

    def test_feature_importances(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        y = labeled_df["regime_7class"].to_numpy(dtype=np.int32)
        clf = RegimeClassifier(n_estimators=30, max_depth=6, random_state=42)
        clf.fit(labeled_df, y)

        imps = clf.feature_importances()
        assert imps is not None
        assert len(imps) > 0
        assert all(isinstance(v, float) for v in imps.values())
        # Should sum to ~1.0
        assert 0.99 < sum(imps.values()) < 1.01

    def test_feature_importances_before_fit_is_none(self):
        from models.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        assert clf.feature_importances() is None

    def test_recommend_models(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        y = labeled_df["regime_7class"].to_numpy(dtype=np.int32)
        clf = RegimeClassifier(n_estimators=30, max_depth=6, random_state=42)
        clf.fit(labeled_df, y)

        recs = clf.recommend_models(labeled_df.head(10), top_k=3, min_prob=0.05)
        assert len(recs) == 10
        for rec in recs:
            assert "regime_name" in rec
            assert "regime_prob" in rec
            assert "recommended_models" in rec
            assert "all_probs" in rec
            assert len(rec["recommended_models"]) <= 3
            assert isinstance(rec["regime_prob"], float)

    def test_update_model_regime_map(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier(n_estimators=10, random_state=42)
        clf.fit(labeled_df)

        new_map = {"xgboost": [0, 1, 2, 3, 4, 5, 6]}
        clf.update_model_regime_map(new_map)
        recs = clf.recommend_models(labeled_df.head(3), top_k=5)
        for rec in recs:
            # With the new map, xgboost should appear everywhere
            assert "xgboost" in rec["recommended_models"]

    def test_get_params(self):
        from models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier(n_estimators=77, max_depth=5)
        params = clf.get_params()
        assert params["n_estimators"] == 77
        assert params["max_depth"] == 5

    def test_free_clears_state(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier(n_estimators=10, random_state=42)
        clf.fit(labeled_df)
        assert clf.is_fitted

        clf.free()
        assert not clf.is_fitted
        with pytest.raises(RuntimeError):
            clf.predict(labeled_df.head(3))

    def test_missing_feature_columns_raises(self, labeled_df):
        from models.regime_classifier import RegimeClassifier

        y = labeled_df["regime_7class"].to_numpy(dtype=np.int32)
        clf = RegimeClassifier(
            n_estimators=10,
            feature_columns=["adx_14", "rsi_14", "imaginary_col_xyz"],
            random_state=42,
        )
        clf.fit(labeled_df, y)

        # predict with partial columns should raise
        bad_df = pd.DataFrame({"adx_14": labeled_df["adx_14"].head(10)})
        with pytest.raises(ValueError, match="Missing"):
            clf.predict(bad_df)

    def test_one_class_y(self, labeled_df):
        """Fitting on single-class labels should still work."""
        from models.regime_classifier import RegimeClassifier

        y_all_sideways = np.full(len(labeled_df), 6, dtype=np.int32)
        clf = RegimeClassifier(n_estimators=10, max_depth=4, random_state=42)
        clf.fit(labeled_df, y_all_sideways)
        assert clf.is_fitted

        preds = clf.predict(labeled_df.head(10))
        assert np.all(preds == 6)

    def test_base_model_inheritance(self):
        from models.regime_classifier import RegimeClassifier
        from models.base_model import BaseModel

        clf = RegimeClassifier()
        assert isinstance(clf, BaseModel)


# ════════════════════════════════════════════════════════════════════
# RegimeConfig tests
# ════════════════════════════════════════════════════════════════════

class TestRegimeConfig:
    def test_defaults(self):
        cfg = RegimeConfig()
        assert cfg.adx_thresh == 20.0
        assert cfg.rsi_high == 70.0
        assert cfg.rsi_low == 30.0

    def test_overrides(self):
        cfg = RegimeConfig(adx_thresh=25, rsi_high=80, bbpct_low=0.05)
        assert cfg.adx_thresh == 25.0
        assert cfg.rsi_high == 80.0
        assert cfg.bbpct_low == 0.05

    def test_unknown_kwargs_ignored(self):
        cfg = RegimeConfig(fake_param=999)
        assert hasattr(cfg, "adx_thresh")


# ════════════════════════════════════════════════════════════════════
# Column resolver edge cases
# ════════════════════════════════════════════════════════════════════

class TestResolveCol:
    def test_exact_match(self):
        df = pd.DataFrame({"adx_14": [1, 2, 3]})
        col = _resolve_col(df, "adx")
        assert col == "adx_14"

    def test_no_match(self):
        df = pd.DataFrame({"x": [1]})
        col = _resolve_col(df, "adx")
        assert col is None

    def test_prefers_first_in_fallback(self):
        df = pd.DataFrame({"adx_14": [1], "adx": [2]})
        col = _resolve_col(df, "adx")
        assert col == "adx_14"
