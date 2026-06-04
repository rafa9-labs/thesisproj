"""Verify that deployment feature computation matches profiling feature computation."""

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_sweep import (
    expand_features,
    compute_feature_matrix,
    FEATURE_NAMES,
    INDICATOR_GRID,
)
from pipeline.committee_builder import CommitteeConfig, RegimeAssignment


@pytest.fixture
def ohlc_df():
    """Small synthetic OHLC DataFrame."""
    n = 200
    rng = np.random.default_rng(42)
    price = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
    df = pd.DataFrame({
        "mid_h": price * 1.002,
        "mid_l": price * 0.998,
        "mid_c": price,
        "spread": np.full(n, 0.0001),
    })
    return df


def test_feature_names_is_generated():
    """FEATURE_NAMES contains all expected indicator families."""
    assert len(FEATURE_NAMES) > 50, f"Expected 65+ features, got {len(FEATURE_NAMES)}"
    assert "sma_20" in FEATURE_NAMES
    assert "ema_20" in FEATURE_NAMES
    assert "rsi_14" in FEATURE_NAMES
    assert "adx_14" in FEATURE_NAMES
    assert "atr_14" in FEATURE_NAMES
    assert "bb_upper_20" in FEATURE_NAMES
    assert "bb_lower_20" in FEATURE_NAMES
    assert "macd_diff" in FEATURE_NAMES
    assert "rv_48" in FEATURE_NAMES
    assert "returns_lag1" in FEATURE_NAMES


def test_feature_names_is_sorted():
    """FEATURE_NAMES is sorted for deterministic column ordering."""
    assert FEATURE_NAMES == sorted(FEATURE_NAMES)


def test_compute_feature_matrix_shape(ohlc_df):
    """compute_feature_matrix returns correct shape."""
    result = compute_feature_matrix(ohlc_df)
    assert result.shape[0] == len(ohlc_df)
    assert result.shape[1] >= len(FEATURE_NAMES)  # + OHLC columns if requested


def test_compute_feature_matrix_no_nan(ohlc_df):
    """compute_feature_matrix returns no NaN/inf values."""
    result = compute_feature_matrix(ohlc_df, include_ohlc=False)
    assert not result.isna().any().any()
    assert not result.isin([np.inf, -np.inf]).any().any()


def test_compute_feature_matrix_subset(ohlc_df):
    """compute_feature_matrix filters to requested feature_names."""
    subset = ["sma_20", "ema_20", "rsi_14"]
    result = compute_feature_matrix(ohlc_df, feature_names=subset, include_ohlc=False)
    assert list(result.columns) == subset
    assert result.shape == (len(ohlc_df), 3)


def test_compute_feature_matrix_includes_ohlc(ohlc_df):
    """compute_feature_matrix includes OHLC when requested."""
    result = compute_feature_matrix(ohlc_df, feature_names=["rsi_14"], include_ohlc=True)
    assert "mid_c" in result.columns
    assert "mid_h" in result.columns
    assert "mid_l" in result.columns
    assert "rsi_14" in result.columns


def test_compute_feature_matrix_float32(ohlc_df):
    """compute_feature_matrix returns float32 dtype."""
    result = compute_feature_matrix(ohlc_df, include_ohlc=False)
    assert result.dtypes.iloc[0] == np.float32


def test_expand_and_compute_are_consistent(ohlc_df):
    """compute_feature_matrix returns same values as expand_features for common cols."""
    expanded = expand_features(ohlc_df)
    computed = compute_feature_matrix(ohlc_df, include_ohlc=False)

    common = [c for c in computed.columns if c in expanded.columns]
    assert len(common) > 0

    for col in common[:5]:  # spot-check first 5
        np.testing.assert_array_almost_equal(
            computed[col].fillna(0.0).values,
            expanded[col].fillna(0.0).values,
            decimal=4,
            err_msg=f"Column {col} differs between expand_features and compute_feature_matrix",
        )


def test_runner_feature_names_subset_of_grid():
    """Indicator features in _LIVE_FEATURE_NAMES are a subset of FEATURE_NAMES."""
    from api.routers.live import _LIVE_FEATURE_NAMES
    ohlc_cols = {"mid_c", "mid_h", "mid_l"}
    for name in _LIVE_FEATURE_NAMES:
        if name in ohlc_cols:
            continue
        assert name in FEATURE_NAMES, \
            f"Live feature {name} not in FEATURE_NAMES. Must add to expand_features() or update live.py."


def test_compute_feature_matrix_missing_cols(ohlc_df):
    """compute_feature_matrix gracefully ignores non-existent columns."""
    result = compute_feature_matrix(
        ohlc_df,
        feature_names=["sma_20", "nonexistent_feature", "rsi_14"],
        include_ohlc=False,
    )
    assert "nonexistent_feature" not in result.columns
    assert "sma_20" in result.columns
    assert "rsi_14" in result.columns


def test_committee_config_roundtrips_model_params():
    """CommitteeConfig.to_dict() → from_dict() preserves model_params."""
    config = CommitteeConfig(
        regimes={
            "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        model_params={"logistic": {"C": 0.5, "max_iter": 200}},
    )
    d = config.to_dict()
    assert "model_params" in d
    assert d["model_params"]["logistic"]["C"] == 0.5

    config2 = CommitteeConfig.from_dict(d)
    assert config2.model_params == {"logistic": {"C": 0.5, "max_iter": 200}}


def test_expand_features_produces_all_families(ohlc_df):
    """expand_features produces all 7 indicator families."""
    expanded = expand_features(ohlc_df)

    families = {
        "sma": any(c.startswith("sma_") for c in expanded.columns),
        "ema": any(c.startswith("ema_") for c in expanded.columns),
        "rsi": any(c.startswith("rsi_") for c in expanded.columns),
        "adx": any(c.startswith("adx_") for c in expanded.columns),
        "atr": any(c.startswith("atr_") for c in expanded.columns),
        "bb_upper": any(c.startswith("bb_upper_") for c in expanded.columns),
        "donchian": any(c.startswith("donchian_") for c in expanded.columns),
        "macd": "macd_diff" in expanded.columns,
        "rv": "rv_48" in expanded.columns,
    }
    for family, present in families.items():
        assert present, f"Missing indicator family: {family}"
