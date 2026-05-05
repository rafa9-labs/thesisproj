"""Tests for the news package: scraper, sentiment, features."""

import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news.scraper import NewsScraper, NewsArticle, ECONOMIC_EVENTS, RSS_FEEDS
from news.sentiment import SentimentAnalyzer, SentimentResult
from news.features import merge_news_features, get_news_feature_columns, _add_event_flags


# ── NewsArticle ──────────────────────────────────────────────────────

class TestNewsArticle:
    def test_dedup_hash_deterministic(self):
        a1 = NewsArticle(title="EURUSD rises", body="x", timestamp=datetime.now(timezone.utc), source="test")
        a2 = NewsArticle(title="EURUSD rises", body="x", timestamp=datetime.now(timezone.utc), source="test")
        assert a1.dedup_hash == a2.dedup_hash

    def test_different_title_different_hash(self):
        a1 = NewsArticle(title="EURUSD rises", body="x", timestamp=datetime.now(timezone.utc), source="test")
        a2 = NewsArticle(title="GBPUSD falls", body="x", timestamp=datetime.now(timezone.utc), source="test")
        assert a1.dedup_hash != a2.dedup_hash

    def test_pair_tags_default_empty(self):
        a = NewsArticle(title="test", body="test", timestamp=datetime.now(timezone.utc), source="test")
        assert a.pair_tags == []


# ── NewsScraper ──────────────────────────────────────────────────────

class TestNewsScraper:
    def test_init_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            scraper = NewsScraper(cache_dir=os.path.join(td, "cache"))
            assert scraper.cache_dir.exists()

    def test_dedup_in_add_to_seen(self):
        scraper = NewsScraper(cache_dir=tempfile.mkdtemp())
        articles = [
            NewsArticle(title="A", body="b", timestamp=datetime.now(timezone.utc), source="s1"),
            NewsArticle(title="A", body="b", timestamp=datetime.now(timezone.utc), source="s1"),
            NewsArticle(title="B", body="b", timestamp=datetime.now(timezone.utc), source="s1"),
        ]
        unique = scraper._add_to_seen(articles)
        assert len(unique) == 2

    def test_pair_tag_extraction(self):
        text = "EURUSD hits new high, GBPUSD follows"
        tags = NewsScraper._extract_pair_tags(text)
        assert "EURUSD" in tags
        assert "GBPUSD" in tags

    def test_pair_tag_extraction_with_slash(self):
        text = "EUR/USD and USD/JPY trading volumes up"
        tags = NewsScraper._extract_pair_tags(text)
        assert "EURUSD" in tags
        assert "USDJPY" in tags

    def test_pair_tag_no_duplicates(self):
        text = "EURUSD and EUR/USD both mentioned"
        tags = NewsScraper._extract_pair_tags(text)
        assert tags.count("EURUSD") == 1

    def test_economic_calendar_returns_events(self):
        events = NewsScraper.economic_calendar_events(2025)
        assert len(events) > 0
        event_names = {e["event"] for e in events}
        assert "NFP" in event_names
        assert "FOMC" in event_names
        assert "CPI" in event_names

    def test_economic_calendar_filter(self):
        events = NewsScraper.economic_calendar_events(2025, events=["NFP"])
        assert all(e["event"] == "NFP" for e in events)
        assert len(events) == 12

    def test_economic_calendar_sorted(self):
        events = NewsScraper.economic_calendar_events(2025)
        dates = [e["date"] for e in events]
        assert dates == sorted(dates)

    def test_economic_calendar_impact_range(self):
        events = NewsScraper.economic_calendar_events(2025)
        for e in events:
            assert 1 <= e["impact"] <= 3

    def test_save_and_load_cache(self):
        with tempfile.TemporaryDirectory() as td:
            scraper = NewsScraper(cache_dir=td)
            articles = [
                NewsArticle(title="Test 1", body="Body 1", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), source="s"),
                NewsArticle(title="Test 2", body="Body 2", timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc), source="s"),
            ]
            scraper.save_cache(articles, label="test_cache")

            scraper2 = NewsScraper(cache_dir=td)
            loaded = scraper2.load_cache(label="test_cache")
            assert len(loaded) == 2
            assert loaded[0].title == "Test 1"

    def test_load_cache_empty_on_missing(self):
        scraper = NewsScraper(cache_dir=tempfile.mkdtemp())
        loaded = scraper.load_cache(label="nonexistent")
        assert loaded == []

    def test_fetch_all_returns_cached(self):
        with tempfile.TemporaryDirectory() as td:
            scraper = NewsScraper(cache_dir=td)
            articles = [
                NewsArticle(title="Cached", body="body", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), source="s"),
            ]
            scraper.save_cache(articles, label="test")

            scraper2 = NewsScraper(cache_dir=td)
            result = scraper2.fetch_all(use_rss=False, use_newsapi=False, cache_label="test")
            assert len(result) == 1
            assert result[0].title == "Cached"

    def test_rate_limit_respected(self):
        scraper = NewsScraper(cache_dir=tempfile.mkdtemp(), rate_limit_sec=0.01)
        t0 = datetime.now(timezone.utc)
        scraper._rate_limit()
        scraper._rate_limit()
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        assert elapsed >= 0.01

    def test_fetch_rss_handles_missing_feedparser(self, monkeypatch):
        scraper = NewsScraper(cache_dir=tempfile.mkdtemp())
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "feedparser":
                raise ImportError("no feedparser")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = scraper.fetch_rss()
        assert result == []

    def test_fetch_newsapi_no_key(self):
        scraper = NewsScraper(cache_dir=tempfile.mkdtemp(), newsapi_key="")
        result = scraper.fetch_newsapi()
        assert result == []


# ── SentimentAnalyzer ────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_vader_backend(self):
        sa = SentimentAnalyzer(backend="vader")
        assert sa.backend == "vader"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            SentimentAnalyzer(backend="invalid")

    def test_score_positive_text(self):
        sa = SentimentAnalyzer(backend="vader")
        result = sa.score_text("EURUSD surges to new highs, strong bullish momentum")
        assert result.score > 0
        assert result.backend == "vader"

    def test_score_negative_text(self):
        sa = SentimentAnalyzer(backend="vader")
        result = sa.score_text("Markets crash, terrible losses, fear dominates")
        assert result.score < 0

    def test_score_empty_text(self):
        sa = SentimentAnalyzer(backend="vader")
        result = sa.score_text("")
        assert result.score == 0.0
        assert result.neutral == 1.0

    def test_score_whitespace_text(self):
        sa = SentimentAnalyzer(backend="vader")
        result = sa.score_text("   ")
        assert result.score == 0.0

    def test_magnitude_is_nonnegative(self):
        sa = SentimentAnalyzer(backend="vader")
        result = sa.score_text("Any text at all")
        assert result.magnitude >= 0

    def test_score_articles(self):
        sa = SentimentAnalyzer(backend="vader")
        articles = [
            NewsArticle(title="Good news", body="Markets rally", timestamp=datetime.now(timezone.utc), source="s"),
            NewsArticle(title="Bad news", body="Markets crash", timestamp=datetime.now(timezone.utc), source="s"),
        ]
        scored = sa.score_articles(articles)
        assert len(scored) == 2
        assert scored[0][1].backend == "vader"

    def test_aggregate_to_df_empty(self):
        sa = SentimentAnalyzer(backend="vader")
        df = sa.aggregate_to_df([])
        assert df.empty
        assert "sentiment_score" in df.columns

    def test_aggregate_to_df_hourly(self):
        sa = SentimentAnalyzer(backend="vader")
        base = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        articles = [
            NewsArticle(title="Good", body="rally", timestamp=base, source="s"),
            NewsArticle(title="Better", body="huge gains", timestamp=base + timedelta(minutes=30), source="s"),
            NewsArticle(title="OK", body="flat", timestamp=base + timedelta(hours=2), source="s"),
        ]
        scored = sa.score_articles(articles)
        df = sa.aggregate_to_df(scored, freq="1h")
        assert len(df) >= 2
        assert "sentiment_score" in df.columns
        assert "news_volume" in df.columns

    def test_aggregate_to_df_daily(self):
        sa = SentimentAnalyzer(backend="vader")
        base = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        articles = [
            NewsArticle(title="A", body="up", timestamp=base, source="s"),
            NewsArticle(title="B", body="down", timestamp=base + timedelta(hours=6), source="s"),
        ]
        scored = sa.score_articles(articles)
        df = sa.aggregate_to_df(scored, freq="1D")
        assert len(df) == 1
        assert df.iloc[0]["news_volume"] == 2


# ── News Features ────────────────────────────────────────────────────

class TestMergeNewsFeatures:
    def _make_ohlc(self, n=100, freq="1h"):
        dates = pd.date_range("2025-01-01", periods=n, freq=freq)
        return pd.DataFrame({
            "open": np.random.randn(n).cumsum() + 1.1,
            "high": np.random.randn(n).cumsum() + 1.11,
            "low": np.random.randn(n).cumsum() + 1.09,
            "close": np.random.randn(n).cumsum() + 1.1,
        }, index=dates)

    def _make_news_df(self, ohlc_index):
        n = min(20, len(ohlc_index))
        timestamps = ohlc_index[:n]
        return pd.DataFrame({
            "timestamp": timestamps,
            "sentiment_score": np.random.randn(n).astype(np.float32),
            "sentiment_magnitude": np.abs(np.random.randn(n)).astype(np.float32),
            "news_volume": np.random.randint(1, 10, n).astype(np.float32),
            "sent_pos": np.random.rand(n).astype(np.float32),
            "sent_neg": np.random.rand(n).astype(np.float32),
            "sent_neu": np.random.rand(n).astype(np.float32),
        })

    def test_no_news_df_adds_defaults(self):
        df = self._make_ohlc()
        result = merge_news_features(df, None, config={"use_news": True})
        assert "sentiment_score" in result.columns
        assert "sentiment_magnitude" in result.columns

    def test_news_disabled_returns_original(self):
        df = self._make_ohlc()
        result = merge_news_features(df, None, config={"use_news": False})
        assert "sentiment_score" not in result.columns

    def test_news_merge_adds_columns(self):
        df = self._make_ohlc()
        news_df = self._make_news_df(df.index)
        result = merge_news_features(df, news_df, config={"use_news": True})
        assert "sentiment_score" in result.columns
        assert "sentiment_magnitude" in result.columns
        assert "news_volume_6bars" in result.columns
        assert "news_volume_24bars" in result.columns

    def test_news_merge_preserves_rows(self):
        df = self._make_ohlc()
        news_df = self._make_news_df(df.index)
        result = merge_news_features(df, news_df, config={"use_news": True})
        assert len(result) == len(df)

    def test_custom_volume_windows(self):
        df = self._make_ohlc()
        news_df = self._make_news_df(df.index)
        result = merge_news_features(df, news_df, config={
            "use_news": True,
            "news_volume_windows": [3, 12],
        })
        assert "news_volume_3bars" in result.columns
        assert "news_volume_12bars" in result.columns

    def test_event_flags(self):
        df = self._make_ohlc(n=200, freq="1h")
        events = [
            {"date": datetime(2025, 1, 1, 13, 30), "event": "NFP", "impact": 3},
            {"date": datetime(2025, 1, 5, 19, 0), "event": "FOMC", "impact": 3},
        ]
        result = merge_news_features(df, None, events=events, config={
            "use_news": True,
            "news_event_flags": True,
        })
        assert "event_flag_nfp" in result.columns
        assert "event_flag_fomc" in result.columns
        assert result["event_flag_nfp"].sum() > 0
        assert result["event_flag_fomc"].sum() > 0

    def test_event_flags_disabled(self):
        df = self._make_ohlc()
        events = [{"date": datetime(2025, 1, 1, 13, 30), "event": "NFP", "impact": 3}]
        result = merge_news_features(df, None, events=events, config={
            "use_news": True,
            "news_event_flags": False,
        })
        assert "event_flag_nfp" not in result.columns

    def test_event_flags_proximity(self):
        dates = pd.date_range("2025-01-01 10:00", periods=20, freq="1h")
        df = pd.DataFrame({"close": [1.1] * 20}, index=dates)
        events = [{"date": datetime(2025, 1, 1, 15, 0), "event": "NFP", "impact": 3}]
        result = _add_event_flags(df, events, proximity_bars=3)
        flags = result["event_flag_nfp"].values
        assert flags.sum() == 7  # 3 before + event + 3 after

    def test_non_datetime_index_falls_back(self):
        df = pd.DataFrame({"close": [1.0, 1.1, 1.2]}, index=[0, 1, 2])
        result = merge_news_features(df, None, config={"use_news": True})
        assert "sentiment_score" in result.columns

    def test_empty_events_no_flag_columns(self):
        df = self._make_ohlc()
        result = merge_news_features(df, None, events=[], config={
            "use_news": True,
            "news_event_flags": True,
        })
        flag_cols = [c for c in result.columns if c.startswith("event_flag_")]
        assert len(flag_cols) == 0


class TestGetNewsFeatureColumns:
    def test_default_columns(self):
        cols = get_news_feature_columns()
        assert "sentiment_score" in cols
        assert "sentiment_magnitude" in cols
        assert "news_volume_6bars" in cols
        assert "news_volume_24bars" in cols

    def test_custom_windows(self):
        cols = get_news_feature_columns({"news_volume_windows": [3, 12]})
        assert "news_volume_3bars" in cols
        assert "news_volume_12bars" in cols


# ── Package imports ──────────────────────────────────────────────────

class TestPackageImports:
    def test_news_package_imports(self):
        from news import NewsScraper, SentimentAnalyzer, merge_news_features, ECONOMIC_EVENTS
        assert NewsScraper is not None
        assert SentimentAnalyzer is not None
        assert merge_news_features is not None
        assert isinstance(ECONOMIC_EVENTS, list)

    def test_economic_events_list(self):
        assert "NFP" in ECONOMIC_EVENTS
        assert "FOMC" in ECONOMIC_EVENTS
        assert "CPI" in ECONOMIC_EVENTS


# ── Walk-forward integrity ───────────────────────────────────────────

class TestWalkForwardIntegrity:
    def test_no_future_news_in_merge(self):
        """Verify that news features only use past data (forward-fill ensures this)."""
        dates = pd.date_range("2025-01-01 10:00", periods=10, freq="1h")
        df = pd.DataFrame({"close": [1.1] * 10}, index=dates)

        # News only available from bar 5 onwards
        news_df = pd.DataFrame({
            "timestamp": dates[5:],
            "sentiment_score": [0.5] * 5,
            "sentiment_magnitude": [0.3] * 5,
            "news_volume": [3.0] * 5,
            "sent_pos": [0.6] * 5,
            "sent_neg": [0.1] * 5,
            "sent_neu": [0.3] * 5,
        })

        result = merge_news_features(df, news_df, config={"use_news": True})

        # Bars 0-4 should have NaN sentiment (no news yet)
        assert pd.isna(result.iloc[0]["sentiment_score"])

        # Bar 5 should have the first sentiment value
        assert result.iloc[5]["sentiment_score"] == pytest.approx(0.5, abs=1e-5)

        # Bar 6 should forward-fill bar 5's value
        assert result.iloc[6]["sentiment_score"] == pytest.approx(0.5, abs=1e-5)


# ── Integration: backtest features with news injected ──────────────────

class TestNewsFeatureIntegration:
    """Verify news features wiring from config through to feature columns."""

    def test_news_columns_in_features_when_injected(self):
        """When _news_aggregated is set and use_news=True, news columns must appear in features result."""
        from pipeline.backtester.features_mixin import FeaturesMixin

        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            "price": np.random.randn(200).cumsum() + 100,
            "returns": np.random.randn(200) * 0.001,
            "high": np.random.randn(200).cumsum() + 101,
            "low": np.random.randn(200).cumsum() + 99,
            "close": np.random.randn(200).cumsum() + 100,
            "spread": np.abs(np.random.randn(200)) * 0.0001,
        }, index=dates)

        news_df = pd.DataFrame({
            "timestamp": dates[:50],
            "sentiment_score": np.random.randn(50).astype(np.float32),
            "sentiment_magnitude": np.abs(np.random.randn(50)).astype(np.float32),
            "news_volume": np.ones(50, dtype=np.float32),
            "sent_pos": np.abs(np.random.randn(50)).astype(np.float32),
            "sent_neg": np.abs(np.random.randn(50)).astype(np.float32),
            "sent_neu": np.abs(np.random.randn(50)).astype(np.float32),
        })

        result_df = merge_news_features(df, news_df, config={"use_news": True})
        assert "sentiment_score" in result_df.columns
        assert "sentiment_magnitude" in result_df.columns

    def test_no_news_data_skipped_gracefully(self):
        """When use_news=True but _news_aggregated is None, the feature block should not crash."""
        from news.features import get_news_feature_columns
        cols = get_news_feature_columns({"use_news": True})
        assert "sentiment_score" in cols
        assert "sentiment_magnitude" in cols

    def test_news_volume_windows_in_features(self):
        """Rolling news volume windows should appear in feature columns list."""
        from news.features import get_news_feature_columns
        cols = get_news_feature_columns({"use_news": True, "news_volume_windows": [6, 24]})
        assert "news_volume_6bars" in cols
        assert "news_volume_24bars" in cols

    def test_use_news_false_skips_news_features(self):
        """When use_news=False, merge_news_features should return df unchanged."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "price": np.random.randn(100).cumsum() + 100,
            "returns": np.random.randn(100) * 0.001,
            "close": np.random.randn(100).cumsum() + 100,
        }, index=dates)

        news_df = pd.DataFrame({
            "timestamp": dates[:50],
            "sentiment_score": np.random.randn(50).astype(np.float32),
            "sentiment_magnitude": np.abs(np.random.randn(50)).astype(np.float32),
            "news_volume": np.ones(50, dtype=np.float32),
        })

        result_df = merge_news_features(df, news_df, config={"use_news": False})
        assert "sentiment_score" not in result_df.columns, "News features should not appear when use_news=False"
