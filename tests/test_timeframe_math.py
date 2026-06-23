"""Unit tests for timeframe math and period conversion functions.

Tests period_offset, periods_between, to_period_freq, convert_month_count_to_periods,
and TIMEFRAME_HIERARCHY validation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from config import (
    period_offset,
    periods_between,
    to_period_freq,
    convert_month_count_to_periods,
    TIMEFRAME_HIERARCHY,
)


class TestPeriodOffset:

    def test_months_offset(self):
        offset = period_offset(3, "months")
        assert offset == pd.DateOffset(months=3)

    def test_weeks_offset(self):
        offset = period_offset(2, "weeks")
        assert offset == pd.DateOffset(weeks=2)

    def test_days_offset(self):
        offset = period_offset(10, "days")
        assert offset == pd.DateOffset(days=10)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown period_unit"):
            period_offset(1, "years")


class TestPeriodsBetween:

    def test_months_between(self):
        a = pd.Timestamp("2024-01-15")
        b = pd.Timestamp("2024-04-20")
        assert periods_between(a, b, "months") == 3

    def test_months_between_same_month(self):
        a = pd.Timestamp("2024-03-01")
        b = pd.Timestamp("2024-03-31")
        assert periods_between(a, b, "months") == 0

    def test_weeks_between(self):
        a = pd.Timestamp("2024-01-01")
        b = pd.Timestamp("2024-01-22")
        assert periods_between(a, b, "weeks") == 3

    def test_days_between(self):
        a = pd.Timestamp("2024-01-01")
        b = pd.Timestamp("2024-01-11")
        assert periods_between(a, b, "days") == 10

    def test_same_timestamp(self):
        ts = pd.Timestamp("2024-06-15")
        assert periods_between(ts, ts, "months") == 0
        assert periods_between(ts, ts, "weeks") == 0
        assert periods_between(ts, ts, "days") == 0

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown period_unit"):
            periods_between(pd.Timestamp("2024-01-01"),
                            pd.Timestamp("2024-02-01"), "years")


class TestToPeriodFreq:

    def test_months_to_M(self):
        assert to_period_freq("months") == "M"

    def test_weeks_to_W(self):
        assert to_period_freq("weeks") == "W"

    def test_days_to_D(self):
        assert to_period_freq("days") == "D"


class TestConvertMonthCount:

    def test_months_to_months(self):
        assert convert_month_count_to_periods(6, "months") == 6

    def test_months_to_weeks(self):
        assert convert_month_count_to_periods(3, "weeks") == 12

    def test_months_to_days(self):
        assert convert_month_count_to_periods(2, "days") == 60

    def test_minimum_one(self):
        assert convert_month_count_to_periods(0, "weeks") == 1
        assert convert_month_count_to_periods(0, "days") == 1

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown period_unit"):
            convert_month_count_to_periods(3, "years")


class TestTimeframeHierarchy:

    def test_m30_bars_per_day(self):
        assert TIMEFRAME_HIERARCHY["M30"]["bars_per_day"] == 48

    def test_h1_bars_per_day(self):
        assert TIMEFRAME_HIERARCHY["H1"]["bars_per_day"] == 24

    def test_h4_bars_per_day(self):
        assert TIMEFRAME_HIERARCHY["H4"]["bars_per_day"] == 6

    def test_m30_mtf_fast_slow(self):
        assert TIMEFRAME_HIERARCHY["M30"]["mtf_fast"] == "H1"
        assert TIMEFRAME_HIERARCHY["M30"]["mtf_slow"] == "H4"

    def test_h1_mtf_fast_slow(self):
        assert TIMEFRAME_HIERARCHY["H1"]["mtf_fast"] == "H4"
        assert TIMEFRAME_HIERARCHY["H1"]["mtf_slow"] == "D1"

    def test_all_keys_have_required_fields(self):
        required = {"bars_per_day", "mtf_fast", "mtf_slow"}
        for tf, cfg in TIMEFRAME_HIERARCHY.items():
            assert required <= set(cfg.keys()), f"{tf} missing fields"
