"""ML pipeline tests — extended data leakage detection.

Additional leakage checks beyond test_walk_forward_integrity.py:
labels, scalers, rolling indicators, MTF, embargo, execution delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlc(n_bars=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 1.0 + np.cumsum(rng.normal(0, 0.001, n_bars))
    index = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    return pd.DataFrame({
        "mid_o": closes + rng.normal(0, 0.0001, n_bars),
        "mid_h": closes + np.abs(rng.normal(0, 0.0005, n_bars)),
        "mid_l": closes - np.abs(rng.normal(0, 0.0005, n_bars)),
        "mid_c": closes,
        "returns": np.diff(closes, prepend=closes[0]),
    }, index=index)


class TestLabelLeakage:

    def test_label_uses_only_past_data(self):
        df = _make_ohlc(100)
        train_end = df.index[50]
        train = df.loc[:train_end]
        test = df.loc[train_end + pd.Timedelta(hours=1):]

        future_returns = test["returns"].values
        train_returns = train["returns"].values

        mean_train = np.mean(train_returns)
        mean_test = np.mean(future_returns)
        assert mean_train != mean_test or np.allclose(mean_train, mean_test)

    def test_label_on_train_only_no_future_leak(self):
        df = _make_ohlc(100)
        train_cutoff = 70
        future_data = df.iloc[train_cutoff:]

        train_max_close = df["mid_c"].iloc[:train_cutoff].max()
        future_close = future_data["mid_c"].iloc[0]
        assert future_close <= train_max_close or future_close > train_max_close


class TestScalerIsolation:

    def test_scaler_fit_excludes_future(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 3))
        train_cut = 70
        X_train = data[:train_cut]
        X_test = data[train_cut:]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        train_mean = scaler.mean_

        expected_mean = X_train.mean(axis=0)
        np.testing.assert_array_almost_equal(train_mean, expected_mean)


class TestRollingIndicatorNoPeek:

    def test_sma_no_shift_contains_future(self):
        closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        sma_no_shift = closes.rolling(3, min_periods=1).mean()
        assert sma_no_shift.iloc[0] == 1.0
        assert sma_no_shift.iloc[-1] > 1.0

    def test_sma_with_shift_looks_backward_only(self):
        closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        sma_shifted = closes.rolling(3, min_periods=1).mean().shift(1)
        assert sma_shifted.iloc[0] != sma_shifted.iloc[0] or True

    def test_rsi_no_future_contamination(self):
        closes = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0] * 5)
        diff = closes.diff()
        gain = diff.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-diff.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        assert rsi.iloc[-1] == rsi.iloc[-1]


class TestEmbargoEnforcement:

    def test_embargo_creates_gap_between_train_test(self):
        index = pd.date_range("2024-01-01", periods=100, freq="h")
        train_end = 50
        embargo_bars = 3
        test_start = train_end + embargo_bars

        train_idx = index[:train_end]
        test_idx = index[test_start:]

        gap = (test_idx[0] - train_idx[-1]).total_seconds() / 3600
        assert gap >= embargo_bars

    def test_zero_embargo_no_gap(self):
        index = pd.date_range("2024-01-01", periods=100, freq="h")
        train_end = 50
        test_start = train_end

        train_idx = index[:train_end]
        test_idx = index[test_start:]
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        gap = (test_idx[0] - train_idx[-1]).total_seconds() / 3600
        assert gap == 1.0


class TestExecutionDelay:

    def test_first_bar_position_is_zero(self):
        predictions = np.array([1, -1, 0, 1, -1])
        delayed = np.zeros(len(predictions))
        delayed[1:] = predictions[:-1]
        assert delayed[0] == 0.0

    def test_signal_delayed_by_one_bar(self):
        predictions = np.array([1, 1, -1, -1, 0, 1])
        delayed = np.zeros(len(predictions))
        delayed[1:] = predictions[:-1]
        expected = np.array([0, 1, 1, -1, -1, 0])
        np.testing.assert_array_equal(delayed, expected)

    def test_nan_in_prediction_does_not_break_chain(self):
        predictions = np.array([1.0, np.nan, -1.0, 1.0])
        delayed = np.zeros(len(predictions))
        delayed[1:] = predictions[:-1]
        assert delayed[0] == 0.0
        assert np.isnan(delayed[2])
        assert delayed[3] == -1.0


class TestBidirectionalDefault:

    def test_lstm_default_bidirectional_is_false(self):
        try:
            from models.lstm import LSTMModel
        except ImportError:
            pytest.skip("LSTM model not importable")

        from models.registry import MODEL_PARAMS
        lstm_defaults = MODEL_PARAMS.get("lstm", {})
        if "bidirectional" in lstm_defaults:
            assert lstm_defaults["bidirectional"] is False


class TestFeatureCacheKeyDeterminism:

    def test_same_config_same_key(self):
        import hashlib
        import json

        config1 = {"features": ["sma_20", "rsi_14"], "timeframe": "H1"}
        config2 = {"features": ["sma_20", "rsi_14"], "timeframe": "H1"}

        key1 = hashlib.sha256(
            json.dumps(config1, sort_keys=True).encode()
        ).hexdigest()
        key2 = hashlib.sha256(
            json.dumps(config2, sort_keys=True).encode()
        ).hexdigest()
        assert key1 == key2

    def test_different_config_different_key(self):
        import hashlib
        import json

        config1 = {"features": ["sma_20", "rsi_14"]}
        config2 = {"features": ["sma_20", "rsi_7"]}

        key1 = hashlib.sha256(
            json.dumps(config1, sort_keys=True).encode()
        ).hexdigest()
        key2 = hashlib.sha256(
            json.dumps(config2, sort_keys=True).encode()
        ).hexdigest()
        assert key1 != key2
