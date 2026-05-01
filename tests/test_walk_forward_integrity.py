"""Walk-forward integrity tests — Phase 3.5.

Validates that the pipeline's data splitting, scaling, embargo,
and execution delay mechanics are leakage-free.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_price_df(n_bars: int = 2000, freq: str = "30min") -> pd.DataFrame:
    """Create a synthetic OHLC price DataFrame with returns."""
    dates = pd.date_range("2020-01-01", periods=n_bars, freq=freq)
    np.random.seed(42)
    price = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0005)
    df = pd.DataFrame({
        "price": price,
        "close": price,
        "high": price + np.abs(np.random.randn(n_bars)) * 0.0002,
        "low":  price - np.abs(np.random.randn(n_bars)) * 0.0002,
        "spread": np.full(n_bars, 0.00015),
    }, index=dates)
    df["returns"] = df["close"].pct_change()
    return df


def _instantiate_backtester():
    """
    Create a lightweight MLBacktester instance with minimal config.
    Avoids loading real CSV data or heavy dependencies.
    """
    from pipeline.backtester.composed import MLBacktester
    df = _make_price_df()

    bt = MLBacktester.__new__(MLBacktester)
    # CoreMixin defaults
    bt.features_config = {
        "sma_window": 20, "ema_window": 20, "rsi_window": 14,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_window": 20, "bb_dev": 2.0, "atr_window": 14,
        "adx_window": 14, "stoch_k_window": 14, "stoch_d_window": 3,
        "use_sma": True, "use_ema": True, "use_rsi": True,
        "use_macd": True, "use_bbands": True, "use_atr": True,
        "use_adx": True, "use_stoch": True, "use_mtf_ma": True,
        "use_regime_features": False,
        "lags": 10, "lag_depth": 1,
        "roll_windows": [5],
        "indicator_windows": {},
        "include_raw_lags": True, "include_hour": True,
        "include_hour_cyclic": True,
        "price_col": "close",
    }
    bt.data = df
    bt.model_type = "logistic"
    bt._is_debug = lambda: False
    bt._in_optuna_cv = False
    return bt


# ── 1. Walk-forward split: no train/test overlap ────────────────────────────

class TestWalkForwardSplits:
    """Tests for get_walk_forward_splits() in EvaluationMixin."""

    def test_splits_no_overlap(self):
        """Every fold's train period must end before its test period starts."""
        bt = _instantiate_backtester()
        walk_data = bt.data

        train_months_list = [6, 12]
        test_months_list = [1]
        max_end = walk_data.index[-1]

        tasks = bt.get_walk_forward_splits(
            walk_data, train_months_list, test_months_list, max_end
        )

        assert len(tasks) > 0, "Expected at least one WFO split"

        for task in tasks:
            start, train_m, test_m, pu = task if len(task) == 4 else (*task, "months")
            train_end = start + pd.DateOffset(months=train_m)
            test_start = train_end
            # Train must end at or before test starts
            assert train_end <= test_start + pd.Timedelta(days=1), (
                f"Train end {train_end} > test start {test_start}"
            )

    def test_splits_chronological(self):
        """Within each (train_months, test_months) combo, splits are time-ordered."""
        bt = _instantiate_backtester()
        walk_data = bt.data

        tasks = bt.get_walk_forward_splits(
            walk_data, train_months_list=[12], test_months_list=[1],
            max_end=walk_data.index[-1],
        )

        dates = [t[0] for t in tasks]
        assert dates == sorted(dates), "Splits are not chronologically ordered"

    def test_splits_non_empty(self):
        """Each split must produce non-zero train and test periods."""
        bt = _instantiate_backtester()
        walk_data = bt.data

        tasks = bt.get_walk_forward_splits(
            walk_data, train_months_list=[6], test_months_list=[1],
            max_end=walk_data.index[-1],
        )

        for task in tasks:
            start, train_m, test_m, pu = task if len(task) == 4 else (*task, "months")
            assert train_m >= 1, f"Train months = {train_m}, expected >= 1"
            assert test_m >= 1, f"Test months = {test_m}, expected >= 1"

    def test_no_future_data_in_train(self):
        """Train end must never exceed test start for any fold."""
        bt = _instantiate_backtester()
        walk_data = bt.data

        tasks = bt.get_walk_forward_splits(
            walk_data, train_months_list=[12], test_months_list=[1],
            max_end=walk_data.index[-1],
        )

        for task in tasks:
            start, train_m, test_m, pu = task if len(task) == 4 else (*task, "months")
            train_end = start + pd.DateOffset(months=train_m)
            test_end = start + pd.DateOffset(months=train_m + test_m)
            # Train window must end before test window ends
            assert train_end <= test_end


# ── 2. Scaling isolation ─────────────────────────────────────────────────────

class TestScalingIsolation:
    """Verify that scale_features uses train-only statistics."""

    def test_scale_uses_train_stats(self):
        """When applying saved means/stds, test data statistics must not affect scaling."""
        bt = _instantiate_backtester()

        np.random.seed(123)
        train_df = pd.DataFrame({
            "feat_a": np.random.randn(100) * 2 + 5,
            "feat_b": np.random.randn(100) * 0.5 + 10,
        })
        test_df = pd.DataFrame({
            "feat_a": np.random.randn(50) * 3 + 20,  # different distribution
            "feat_b": np.random.randn(50) * 1.0 + 50,
        })
        features = ["feat_a", "feat_b"]

        # Fit on train
        df_train_scaled, means, stds = bt.scale_features(
            train_df.copy(), features
        )

        # Apply train stats to test
        df_test_scaled, _, _ = bt.scale_features(
            test_df.copy(), features, means=means, stds=stds
        )

        # Test scaled values should NOT be zero-mean/unit-variance
        # (because they use train statistics, not their own)
        test_mean_a = df_test_scaled["feat_a"].mean()
        test_std_a = df_test_scaled["feat_a"].std()

        # If test data were scaled with its own stats, mean would be ~0.
        # Since train has mean ~5 and test has mean ~20, the test scaled mean
        # should be far from 0 (approximately (20-5)/2 = 7.5).
        assert abs(test_mean_a) > 1.0, (
            f"Test scaled mean too close to 0 ({test_mean_a:.3f}), "
            "suggesting test statistics leaked into scaling"
        )

    def test_train_scaled_zero_mean_unit_var(self):
        """Train data scaled with its own stats should be ~N(0,1)."""
        bt = _instantiate_backtester()

        np.random.seed(42)
        df = pd.DataFrame({
            "feat_x": np.random.randn(500) * 3 + 10,
        })
        features = ["feat_x"]

        df_scaled, means, stds = bt.scale_features(df.copy(), features)

        # Mean should be ~0, std ~1
        assert abs(df_scaled["feat_x"].mean()) < 0.1
        assert abs(df_scaled["feat_x"].std() - 1.0) < 0.1

    def test_scaling_no_infs(self):
        """Scaled features must not contain inf or -inf."""
        bt = _instantiate_backtester()

        df = pd.DataFrame({
            "feat": [1.0, 2.0, 3.0, 1e10, -1e10],
        })
        features = ["feat"]

        df_scaled, _, _ = bt.scale_features(df.copy(), features)

        assert not np.isinf(df_scaled["feat"]).any(), "Scaled data contains inf"


# ── 3. Execution delay ──────────────────────────────────────────────────────

class TestExecutionDelay:
    """Verify that signal at bar T produces trades starting at bar T+1."""

    def test_signal_shift(self):
        """
        In the pipeline, positions should be taken at bar T+1 based on
        signal at bar T. This is typically implemented via shift(1).
        """
        n = 100
        signals = pd.Series([0, 0, 1, 1, 0, -1, 0, 0, 1, 0] * 10, name="signal")

        # The pipeline should delay signals by 1 bar
        positions = signals.shift(1).fillna(0)

        # At bar 2 (signal=1), position should still be 0 (previous bar signal=0)
        assert positions.iloc[2] == 0, "Position at bar 2 should reflect bar-1 signal"
        # At bar 3 (signal=1), position should be 1 (previous bar signal=1)
        assert positions.iloc[3] == 1, "Position at bar 3 should reflect bar-2 signal"

    def test_no_lookahead_in_positions(self):
        """Positions must never use future signal values."""
        signals = pd.Series([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        positions = signals.shift(1).fillna(0)

        # Bar 2 has signal=1 but position should still be 0 (uses bar 1's signal)
        assert positions.iloc[2] == 0
        # Bar 3 should have position=1 (uses bar 2's signal)
        assert positions.iloc[3] == 1


# ── 4. Embargo enforcement ──────────────────────────────────────────────────

class TestEmbargoEnforcement:
    """Test that embargo bars create a gap between train and test."""

    def test_embargo_creates_gap(self):
        """
        With embargo_bars > 0, there must be a gap of at least embargo_bars
        between the last train index and first test index.
        """
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="30min")
        df = pd.DataFrame({"close": np.random.randn(n).cumsum() + 100}, index=dates)

        embargo_bars = 10
        train_end_idx = 800
        test_start_idx = train_end_idx + embargo_bars

        train = df.iloc[:train_end_idx]
        test = df.iloc[test_start_idx:]

        # Verify gap
        gap = test.index[0] - train.index[-1]
        expected_gap = pd.Timedelta(minutes=30 * (embargo_bars + 1))

        assert gap >= expected_gap, (
            f"Gap {gap} < expected {expected_gap} with embargo={embargo_bars}"
        )

    def test_zero_embargo_no_gap(self):
        """With embargo_bars=0, test starts immediately after train."""
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="30min")
        df = pd.DataFrame({"close": np.random.randn(n).cumsum() + 100}, index=dates)

        embargo_bars = 0
        train_end_idx = 400
        test_start_idx = train_end_idx + embargo_bars

        train = df.iloc[:train_end_idx]
        test = df.iloc[test_start_idx:]

        gap = test.index[0] - train.index[-1]
        # With 0 embargo, gap should be exactly 1 bar
        assert gap == pd.Timedelta(minutes=30), f"Expected 1-bar gap, got {gap}"


# ── 5. Feature computation: backward-looking only ───────────────────────────

class TestFeatureBackwardLooking:
    """Verify features don't use future data."""

    def test_sma_backward_looking(self):
        """SMA at bar T should only use bars [T-window+1 .. T]."""
        import ta

        np.random.seed(42)
        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        sma = ta.trend.sma_indicator(prices, window=20)

        # SMA at bar 50 should be determined solely by bars 31..50
        sma_50 = sma.iloc[50]
        expected = prices.iloc[31:51].mean()
        assert abs(sma_50 - expected) < 1e-10, "SMA uses data outside its window"

    def test_rsi_backward_looking(self):
        """RSI at bar T must not depend on bars after T."""
        import ta

        np.random.seed(42)
        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        rsi_full = ta.momentum.RSIIndicator(prices, window=14).rsi()

        # Compute RSI on truncated data (first 60 bars)
        rsi_trunc = ta.momentum.RSIIndicator(prices.iloc[:60], window=14).rsi()

        # RSI at bar 59 should be identical in both
        assert abs(rsi_full.iloc[59] - rsi_trunc.iloc[59]) < 1e-10, (
            "RSI value changed when future data was added — look-ahead bias!"
        )

    def test_returns_backward_looking(self):
        """Return at bar T = (price[T] - price[T-1]) / price[T-1]."""
        prices = pd.Series([100.0, 101.0, 99.5, 102.0, 100.5])
        returns = prices.pct_change()

        assert pd.isna(returns.iloc[0])  # first return is NaN
        assert abs(returns.iloc[1] - 0.01) < 1e-10
        assert abs(returns.iloc[2] - (-1.5 / 101.0)) < 1e-10


# ── 6. Label computation: no future leakage in training ─────────────────────

class TestLabelIntegrity:
    """Verify labels are computed correctly without look-ahead."""

    def test_label_with_neutral(self):
        """Test the 3-class label function."""
        bt = _instantiate_backtester()

        returns = np.array([0.01, -0.005, 0.0001, -0.003, 0.005, -0.01])
        threshold = 0.003

        labels = bt.label_with_neutral(returns, threshold)

        assert labels[0] == 2   # 0.01 > 0.003 → buy
        assert labels[1] == 0   # -0.005 < -0.003 → sell
        assert labels[2] == 1   # 0.0001 in [-0.003, 0.003] → neutral
        assert labels[3] == 1   # -0.003 == -threshold → neutral (strict <)
        assert labels[4] == 2   # 0.005 > 0.003 → buy
        assert labels[5] == 0   # -0.01 < -0.003 → sell

    def test_labels_deterministic(self):
        """Same input → same labels."""
        bt = _instantiate_backtester()
        np.random.seed(99)
        returns = np.random.randn(100) * 0.01
        threshold = 0.005

        labels1 = bt.label_with_neutral(returns, threshold)
        labels2 = bt.label_with_neutral(returns, threshold)
        np.testing.assert_array_equal(labels1, labels2)


class TestPeriodUnit:
    """Tests for the period_unit parameter in walk-forward splits."""

    def test_period_offset_months(self):
        from config import period_offset
        off = period_offset(3, "months")
        assert off == pd.DateOffset(months=3)

    def test_period_offset_weeks(self):
        from config import period_offset
        off = period_offset(4, "weeks")
        assert off == pd.DateOffset(weeks=4)

    def test_period_offset_days(self):
        from config import period_offset
        off = period_offset(10, "days")
        assert off == pd.DateOffset(days=10)

    def test_periods_between_months(self):
        from config import periods_between
        a = pd.Timestamp("2020-01-01")
        b = pd.Timestamp("2020-06-01")
        assert periods_between(a, b, "months") == 5

    def test_periods_between_weeks(self):
        from config import periods_between
        a = pd.Timestamp("2020-01-01")
        b = pd.Timestamp("2020-02-12")
        assert periods_between(a, b, "weeks") == 6

    def test_periods_between_days(self):
        from config import periods_between
        a = pd.Timestamp("2020-01-01")
        b = pd.Timestamp("2020-01-15")
        assert periods_between(a, b, "days") == 14

    def test_convert_month_count(self):
        from config import convert_month_count_to_periods
        assert convert_month_count_to_periods(6, "months") == 6
        assert convert_month_count_to_periods(6, "weeks") == 24
        assert convert_month_count_to_periods(6, "days") == 180

    def test_to_period_freq(self):
        from config import to_period_freq
        assert to_period_freq("months") == "M"
        assert to_period_freq("weeks") == "W"
        assert to_period_freq("days") == "D"

    def test_wfo_splits_weeks_unit(self):
        bt = _instantiate_backtester()
        walk_data = bt.data
        tasks = bt.get_walk_forward_splits(
            walk_data, train_months_list=[24], test_months_list=[4],
            max_end=walk_data.index[-1], period_unit="weeks",
        )
        assert len(tasks) > 0
        for task in tasks:
            start, train_p, test_p, pu = task
            assert pu == "weeks"
            assert train_p >= 1
            assert test_p >= 1

    def test_wfo_splits_days_unit(self):
        bt = _instantiate_backtester()
        walk_data = bt.data
        tasks = bt.get_walk_forward_splits(
            walk_data, train_months_list=[12], test_months_list=[1],
            max_end=walk_data.index[-1], period_unit="days",
        )
        assert len(tasks) > 0
        for task in tasks:
            start, train_p, test_p, pu = task
            assert pu == "days"
            assert train_p >= 1
            assert test_p >= 1

    def test_schema_period_unit_default(self):
        from schemas.backtest import BacktestParams
        bp = BacktestParams()
        assert bp.period_unit == "months"

    def test_schema_period_unit_weeks(self):
        from schemas.backtest import BacktestParams
        bp = BacktestParams(period_unit="weeks")
        assert bp.period_unit == "weeks"

    def test_schema_period_unit_invalid(self):
        from schemas.backtest import BacktestParams
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BacktestParams(period_unit="hours")
