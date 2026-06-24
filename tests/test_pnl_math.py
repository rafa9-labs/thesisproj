"""Unit tests for pure PnL and metrics math functions.

Tests compute_metrics, compute_geometric_mean_annualized, hac_std,
estimate_frequency_per_year, and direction coercion — all pure, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.metrics.metrics_eval import (
    _coerce_direction_labels,
    compute_metrics,
    compute_geometric_mean_annualized,
    estimate_frequency_per_year,
    hac_std,
    _macro_prec_f1_from_confusion,
)


def _make_returns(values, freq="D"):
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=index)


def _make_positions(values, freq="D"):
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=index)


class TestSharpeRatio:
    """compute_metrics Sharpe ratio tests."""

    def test_sharpe_known_monthly_positive(self):
        rng = np.random.default_rng(42)
        rets = _make_returns(0.01 + rng.normal(0, 0.002, 36), freq="ME")
        sharpe, _, _ = compute_metrics(rets, None, frequency_per_year=12,
                                        min_active_obs=10)
        assert sharpe >= 1.0

    def test_sharpe_zero_volatility(self):
        rets = _make_returns([0.0] * 20, freq="D")
        sharpe, _, _ = compute_metrics(rets, None, frequency_per_year=252)
        assert sharpe == 0.0

    def test_sharpe_negative_returns(self):
        rng = np.random.default_rng(43)
        rets = _make_returns(-0.01 + rng.normal(0, 0.002, 36), freq="ME")
        sharpe, _, _ = compute_metrics(rets, None, frequency_per_year=12,
                                        min_active_obs=10)
        assert sharpe < 0.0

    def test_sharpe_all_zeros_except_one(self):
        rets = _make_returns([0.0] * 99 + [0.01], freq="D")
        sharpe, _, _ = compute_metrics(rets, None, frequency_per_year=252)
        assert np.isfinite(sharpe)

    def test_sharpe_insufficient_observations(self):
        rets = _make_returns([0.01, -0.005], freq="ME")
        sharpe, _, _ = compute_metrics(rets, None, frequency_per_year=12,
                                        min_active_obs=100)
        assert sharpe == 0.0


class TestMaxDrawdown:
    """Max drawdown computed within compute_metrics."""

    def test_max_drawdown_simple(self):
        rets = _make_returns([-0.20, 0.05, 0.02, 0.03] * 10, freq="D")
        _, dd, _ = compute_metrics(rets, None, frequency_per_year=252,
                                    min_active_obs=2)
        assert dd <= -0.15

    def test_max_drawdown_no_loss(self):
        rets = _make_returns([0.01, 0.02, 0.015, 0.005], freq="D")
        _, dd, _ = compute_metrics(rets, None, frequency_per_year=252)
        assert dd == 0.0

    def test_max_drawdown_from_peak(self):
        rets = _make_returns([0.10, -0.05, -0.05, 0.02], freq="D")
        _, dd, _ = compute_metrics(rets, None, frequency_per_year=252)
        assert dd < 0.0
        assert dd >= -0.10


class TestTradeCounting:
    """Trade count from position changes."""

    def test_trades_simple(self):
        pos = _make_positions([0, 1, 1, -1, -1, 0, 1])
        _, _, trades = compute_metrics(_make_returns([0.0] * 7), pos,
                                        frequency_per_year=252)
        assert trades == 5

    def test_trades_none_positions(self):
        _, _, trades = compute_metrics(
            _make_returns([0.01, 0.02]), None, frequency_per_year=252,
        )
        assert trades == 0

    def test_trades_single_position(self):
        _, _, trades = compute_metrics(
            _make_returns([0.01]),
            _make_positions([1]),
            frequency_per_year=252,
        )
        assert trades == 0

    def test_trades_constantly_flat(self):
        pos = _make_positions([0, 0, 0, 0, 0])
        _, _, trades = compute_metrics(
            _make_returns([0.01] * 5), pos, frequency_per_year=252,
        )
        assert trades == 0

    def test_trades_nan_in_positions(self):
        pos = pd.Series([0, 1, np.nan, -1, 0])
        _, _, trades = compute_metrics(
            _make_returns([0.01] * 5), pos, frequency_per_year=252,
        )
        assert trades == 4


class TestGeoMean:
    """compute_geometric_mean_annualized tests."""

    def test_geo_mean_one_percent_monthly(self):
        rets = _make_returns([0.01] * 12, freq="ME")
        g = compute_geometric_mean_annualized(rets)
        assert g > 0.0
        assert np.isfinite(g)

    def test_geo_mean_empty(self):
        g = compute_geometric_mean_annualized(pd.Series([], dtype=float))
        assert np.isnan(g)

    def test_geo_mean_single_point(self):
        rets = _make_returns([0.01])
        g = compute_geometric_mean_annualized(rets)
        assert np.isfinite(g)


class TestHACStd:
    """Newey-West HAC standard deviation."""

    def test_hac_std_positive(self):
        x = np.random.default_rng(42).normal(0.001, 0.01, 500)
        s = hac_std(x)
        assert s > 0

    def test_hac_std_single_value(self):
        s = hac_std(np.array([0.5]))
        assert s == 0.0

    def test_hac_std_all_same(self):
        s = hac_std(np.array([1.0] * 100))
        assert s == 0.0

    def test_hac_std_with_nans(self):
        x = np.array([0.01, np.nan, 0.02, 0.03, np.nan])
        s = hac_std(x)
        assert s > 0


class TestFrequencyEstimation:
    """estimate_frequency_per_year tests."""

    def test_frequency_h1_weekday(self):
        index = pd.date_range("2024-01-01", "2024-03-31", freq="h")
        freq = estimate_frequency_per_year(index)
        assert 5000 < freq < 9000

    def test_frequency_daily(self):
        index = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        freq = estimate_frequency_per_year(index)
        assert 350 < freq < 370

    def test_frequency_empty(self):
        index = pd.DatetimeIndex([], freq="D")
        freq = estimate_frequency_per_year(index)
        assert freq == 252.0

    def test_frequency_short_index(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        freq = estimate_frequency_per_year(index)
        assert freq == 252.0


class TestDirectionCoerce:
    """_coerce_direction_labels tests."""

    def test_coerce_float_to_int(self):
        result = _coerce_direction_labels(np.array([0.9, -0.9, 0.1, -0.1]))
        assert list(result) == [1, -1, 0, 0]

    def test_coerce_already_int(self):
        result = _coerce_direction_labels(np.array([1, -1, 0, 1]))
        assert list(result) == [1, -1, 0, 1]

    def test_coerce_empty(self):
        result = _coerce_direction_labels(np.array([]))
        assert result.size == 0

    def test_coerce_out_of_range(self):
        result = _coerce_direction_labels(np.array([5, -5, 0]))
        assert list(result) == [0, 0, 0]


class TestPrecisionF1:
    """_macro_prec_f1_from_confusion tests."""

    def test_perfect_prediction(self):
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, -1, 0, 1])
        prec, f1, _ = _macro_prec_f1_from_confusion(y_true, y_pred)
        assert prec == 1.0
        assert f1 == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([-1, -1, -1])
        prec, f1, _ = _macro_prec_f1_from_confusion(y_true, y_pred)
        assert prec == 0.0
        assert f1 == 0.0

    def test_mixed(self):
        y_true = np.array([-1, -1, 0, 0, 1, 1])
        y_pred = np.array([-1, 0, 0, 1, 1, -1])
        prec, f1, _ = _macro_prec_f1_from_confusion(y_true, y_pred)
        assert 0.0 < prec < 1.0
        assert 0.0 < f1 < 1.0
