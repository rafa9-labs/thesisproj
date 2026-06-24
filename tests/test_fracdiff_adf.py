"""Tests for find_min_stationary_d — P4 ADF floor for fracdiff."""
import numpy as np
import pandas as pd
import pytest

from pipeline.features.feature_utils import find_min_stationary_d, fracdiff


def _make_price_series(n=500, seed=42):
    """Generate a price series that becomes stationary after mild differencing."""
    rng = np.random.default_rng(seed)
    # Random walk (I(1), non-stationary) + mean-reverting noise (I(0))
    rw = np.cumsum(rng.normal(0, 0.001, n)) + 1.0
    mr = rng.normal(0, 0.0005, n)
    price = rw + mr
    return pd.Series(price, name="mid_c")


def _make_pure_random_walk(n=500):
    """Pure random walk — needs d close to 1 for stationarity."""
    rng = np.random.default_rng(99)
    rw = np.cumsum(rng.normal(0, 0.001, n)) + 1.0
    return pd.Series(rw, name="mid_c")


class TestFindMinStationaryD:
    def test_returns_reasonable_d(self):
        price = _make_price_series(500)
        d = find_min_stationary_d(price)
        assert 0.0 < d <= 0.95

    def test_higher_d_for_pure_rw(self):
        """Pure random walk should need higher d than mixed series."""
        rw = _make_pure_random_walk(500)
        mixed = _make_price_series(500)
        d_rw = find_min_stationary_d(rw)
        d_mixed = find_min_stationary_d(mixed)
        # RW should need >= d than mixed
        assert d_rw >= d_mixed

    def test_short_series_returns_default(self):
        """Too few bars should return the default 0.4."""
        price = _make_price_series(50)
        d = find_min_stationary_d(price)
        assert d == 0.4

    def test_output_is_stationary(self):
        """The returned d should actually produce a stationary series."""
        price = _make_price_series(500)
        d = find_min_stationary_d(price)
        from statsmodels.tsa.stattools import adfuller
        fd = fracdiff(price, d=d).dropna()
        if len(fd) > 50:
            _, p_val, *_ = adfuller(fd.values, maxlag=30, autolag="AIC")
            assert p_val < 0.05, f"d={d} produced non-stationary series (p={p_val})"

    def test_deterministic_output(self):
        """Same input should produce same output."""
        rng = np.random.default_rng(123)
        price1 = pd.Series(np.cumsum(rng.normal(0, 0.001, 500)) + 1.0)
        rng2 = np.random.default_rng(123)
        price2 = pd.Series(np.cumsum(rng2.normal(0, 0.001, 500)) + 1.0)
        assert find_min_stationary_d(price1) == find_min_stationary_d(price2)

    def test_all_stationary_returns_lowest_d(self):
        """If series is already stationary at d=0.05, return 0.05."""
        rng = np.random.default_rng(777)
        stationary = pd.Series(rng.normal(0, 0.001, 500), name="mid_c")
        d = find_min_stationary_d(stationary)
        assert d <= 0.15  # should find a low d
