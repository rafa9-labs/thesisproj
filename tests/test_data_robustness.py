"""Regression tests for Sprint B data/feature robustness fixes.

  E5.1 — tz_localize(None) crash on already-tz-aware event timestamps
  E8.1/E8.2 — Empty DataFrame after dropna returns early with warning
  E8.5 — Hour extraction on non-DatetimeIndex does not crash
  E6.5 — Cache load failures logged at WARNING level
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from news.features import merge_news_features, _add_event_flags
from news.scraper import NewsScraper, NewsArticle


# ── E5.1: tz crash on already-aware event timestamps ─────────────────

class TestE51TzCrashEventFlags:
    def _make_df(self, tz="UTC", n=50):
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz=tz)
        return pd.DataFrame({"close": [1.1] * n}, index=dates)

    def test_tz_aware_events_with_tz_aware_df(self):
        """Events with tz-aware timestamps should work with tz-aware OHLC data."""
        df = self._make_df(tz="UTC")
        events = [
            {"date": pd.Timestamp("2025-01-01 12:00", tz="UTC"), "event": "NFP", "impact": 3},
        ]
        result = _add_event_flags(df, events, proximity_bars=3)
        assert "event_flag_nfp" in result.columns
        assert result["event_flag_nfp"].sum() > 0

    def test_tz_naive_events_with_tz_aware_df(self):
        """Naive event timestamps should be localized to df's timezone."""
        df = self._make_df(tz="UTC")
        events = [
            {"date": datetime(2025, 1, 1, 12, 0), "event": "FOMC", "impact": 3},
        ]
        result = _add_event_flags(df, events, proximity_bars=3)
        assert "event_flag_fomc" in result.columns
        assert result["event_flag_fomc"].sum() > 0

    def test_tz_aware_events_with_tz_naive_df(self):
        """tz-aware events with tz-naive df should not crash."""
        df = self._make_df(tz=None)
        events = [
            {"date": pd.Timestamp("2025-01-01 12:00", tz="UTC"), "event": "CPI", "impact": 2},
        ]
        result = _add_event_flags(df, events, proximity_bars=3)
        assert "event_flag_cpi" in result.columns

    def test_mixed_tz_events_no_crash(self):
        df = self._make_df(tz="UTC")
        events = [
            {"date": pd.Timestamp("2025-01-01 12:00", tz="UTC"), "event": "NFP", "impact": 3},
            {"date": datetime(2025, 1, 2, 10, 0), "event": "FOMC", "impact": 3},
            {"date": pd.Timestamp("2025-01-03 08:00", tz="US/Eastern"), "event": "CPI", "impact": 2},
        ]
        result = _add_event_flags(df, events, proximity_bars=3)
        assert "event_flag_nfp" in result.columns
        assert "event_flag_fomc" in result.columns
        assert "event_flag_cpi" in result.columns


# ── E8.1/E8.2: Empty DataFrame guard ────────────────────────────────

class TestE81EmptyDataFrameGuard:
    def test_all_nan_features_returns_gracefully(self):
        """DataFrame with NaN features after dropna should return empty, not crash."""
        dates = pd.date_range("2024-06-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "price": [1.1] * 100,
            "high": [1.1] * 100,
            "low": [1.1] * 100,
            "close": [1.1] * 100,
            "spread": [0.0001] * 100,
            "returns": [0.0] * 100,
        }, index=dates)
        from pipeline.backtester.composed import MLBacktester
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-07-01",
            trading_costs=False,
            features_config={"lags": 5, "use_sma": True, "include_hour": True},
        )
        df_out, features = bt.prepare_features(df, lags=5)
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(features, list)

    def test_constant_prices_no_crash(self):
        """Constant price column should not crash the pipeline."""
        from pipeline.backtester.composed import MLBacktester
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "csv_data", "EURUSD_10_years_H1_OANDA.csv"
        )
        if not os.path.exists(csv_path):
            pytest.skip("EURUSD H1 CSV not found")
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-06-15",
            trading_costs=False,
            features_config={"lags": 10, "use_sma": True, "include_hour": True},
        )
        if len(bt.data) < 5:
            pytest.skip("Not enough data")
        df_copy = bt.data.copy()
        df_copy["price"] = 1.1000
        df_copy["close"] = 1.1000
        df_copy["returns"] = 0.0
        df_out, features = bt.prepare_features(df_copy, lags=10)
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(features, list)


# ── E8.5: Hour extraction on non-DatetimeIndex ──────────────────────

class TestE85HourNonDatetimeIndex:
    def test_string_index_no_crash(self):
        """String index should skip hour features without crashing."""
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "csv_data", "EURUSD_10_years_H1_OANDA.csv"
        )
        if not os.path.exists(csv_path):
            pytest.skip("EURUSD H1 CSV not found")
        from pipeline.backtester.composed import MLBacktester
        bt = MLBacktester(
            symbol="EURUSD",
            start="2024-06-01",
            end="2024-07-01",
            trading_costs=False,
            features_config={"lags": 5, "use_sma": True, "include_hour": True},
        )
        df_copy = bt.data.copy()
        df_copy.index = [f"bar_{i}" for i in range(len(df_copy))]
        df_out, features = bt.prepare_features(df_copy, lags=5)
        assert isinstance(df_out, pd.DataFrame)
        assert "hour" not in df_out.columns


# ── E6.5: Cache load WARNING logging ────────────────────────────────

class TestE65CacheLoadWarningLog:
    def test_corrupt_cache_logs_warning(self, caplog):
        """Cache load failure should be logged at WARNING level."""
        with tempfile.TemporaryDirectory() as td:
            scraper = NewsScraper(cache_dir=td)
            corrupt_path = os.path.join(td, "articles.parquet")
            with open(corrupt_path, "w") as f:
                f.write("this is not a parquet file")
            with caplog.at_level(logging.WARNING):
                loaded = scraper.load_cache(label="articles")
            assert loaded == []
            assert any("Cache load failed" in r.message for r in caplog.records)
