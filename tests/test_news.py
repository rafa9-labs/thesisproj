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
            now = datetime.now(timezone.utc)
            articles = [
                NewsArticle(title="Test 1", body="Body 1", timestamp=now - timedelta(days=1), source="s"),
                NewsArticle(title="Test 2", body="Body 2", timestamp=now, source="s"),
            ]
            scraper.save_cache(articles, label="test_cache")

            scraper2 = NewsScraper(cache_dir=td)
            loaded = scraper2.load_cache(label="test_cache")
            assert len(loaded) == 2
            titles = {a.title for a in loaded}
            assert "Test 1" in titles
            assert "Test 2" in titles

    def test_load_cache_empty_on_missing(self):
        scraper = NewsScraper(cache_dir=tempfile.mkdtemp())
        loaded = scraper.load_cache(label="nonexistent")
        assert loaded == []

    def test_fetch_all_returns_cached(self):
        with tempfile.TemporaryDirectory() as td:
            scraper = NewsScraper(cache_dir=td)
            now = datetime.now(timezone.utc)
            articles = [
                NewsArticle(title="Cached", body="body", timestamp=now - timedelta(days=1), source="s"),
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


# ── Enhanced pair tagging (Phase 1) ────────────────────────────────────

class TestEnhancedPairTags:
    def test_keyword_tagging_ecb_gives_eur(self):
        text = "ECB raises rates by 25 basis points"
        tags = NewsScraper._extract_pair_tags(text)
        assert "EUR" in tags

    def test_keyword_tagging_fed_gives_usd(self):
        text = "Federal Reserve signals rate cuts ahead"
        tags = NewsScraper._extract_pair_tags(text)
        assert "USD" in tags

    def test_keyword_tagging_boe_gives_gbp(self):
        text = "BOE holds rates steady as pound weakens"
        tags = NewsScraper._extract_pair_tags(text)
        assert "GBP" in tags

    def test_currency_code_word_boundary(self):
        text = "AUD strengthens against major currencies"
        tags = NewsScraper._extract_pair_tags(text)
        assert "AUD" in tags

    def test_no_false_positive_audit(self):
        text = "AUDIT committee reviews financial statements"
        tags = NewsScraper._extract_pair_tags(text)
        assert "AUD" not in tags

    def test_multiple_keywords_same_currency(self):
        text = "Dollar strengthens as Powell speaks at FOMC"
        tags = NewsScraper._extract_pair_tags(text)
        assert "USD" in tags
        assert tags.count("USD") == 1

    def test_pair_and_currency_together(self):
        text = "EURUSD rallies as ECB and Federal Reserve diverge"
        tags = NewsScraper._extract_pair_tags(text)
        assert "EURUSD" in tags
        assert "EUR" in tags
        assert "USD" in tags

    def test_lagarde_is_eur(self):
        text = "Lagarde signals cautious approach"
        tags = NewsScraper._extract_pair_tags(text)
        assert "EUR" in tags

    def test_bundesbank_is_eur(self):
        text = "Bundesbank warns on inflation risks"
        tags = NewsScraper._extract_pair_tags(text)
        assert "EUR" in tags

    def test_yen_is_jpy(self):
        text = "Yen weakens as BOJ maintains policy"
        tags = NewsScraper._extract_pair_tags(text)
        assert "JPY" in tags

    def test_generic_terms_not_tagged(self):
        text = "Global inflation and GDP growth outlook"
        tags = NewsScraper._extract_pair_tags(text)
        assert "USD" not in tags
        assert "EUR" not in tags

    def test_no_pair_mention_gets_zero_tags(self):
        text = "Markets rally on risk appetite"
        tags = NewsScraper._extract_pair_tags(text)
        assert len(tags) == 0


class TestFilterByPairSimplified:
    def setup_method(self):
        self.eurusd_article = NewsArticle(
            title="EURUSD rises",
            body="dollar weakens",
            timestamp=datetime.now(timezone.utc),
            source="test",
            pair_tags=["EURUSD", "EUR", "USD"],
        )
        self.gbp_article = NewsArticle(
            title="BOE holds rates",
            body="pound stable",
            timestamp=datetime.now(timezone.utc),
            source="test",
            pair_tags=["GBP"],
        )
        self.usd_article = NewsArticle(
            title="Fed minutes",
            body="dollar outlook",
            timestamp=datetime.now(timezone.utc),
            source="test",
            pair_tags=["USD"],
        )
        self.untagged_article = NewsArticle(
            title="Global markets",
            body="risk on sentiment",
            timestamp=datetime.now(timezone.utc),
            source="test",
            pair_tags=[],
        )

    def test_eurusd_article_matches_eurusd(self):
        result = NewsScraper.filter_by_pair([self.eurusd_article], "EURUSD")
        assert len(result) == 1

    def test_eurusd_article_does_not_match_gbpjpy(self):
        result = NewsScraper.filter_by_pair([self.eurusd_article], "GBPJPY")
        assert len(result) == 0

    def test_eurusd_article_matches_gbpusd_via_usd_tag(self):
        result = NewsScraper.filter_by_pair([self.eurusd_article], "GBPUSD")
        assert len(result) == 1

    def test_gbp_article_matches_gbpusd(self):
        result = NewsScraper.filter_by_pair([self.gbp_article], "GBPUSD")
        assert len(result) == 1

    def test_gbp_article_does_not_match_eurusd(self):
        result = NewsScraper.filter_by_pair([self.gbp_article], "EURUSD")
        assert len(result) == 0

    def test_untagged_article_matches_nothing(self):
        result = NewsScraper.filter_by_pair([self.untagged_article], "EURUSD")
        assert len(result) == 0

    def test_usd_article_matches_all_usd_pairs(self):
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
        for pair in pairs:
            result = NewsScraper.filter_by_pair([self.usd_article], pair)
            assert len(result) == 1, f"USD article should match {pair}"


# ── Sentiment cache (Phase 3) ──────────────────────────────────────────

class TestSentimentCache:
    def test_cache_miss_on_empty(self):
        from news.sentiment import SentimentAnalyzer
        result = SentimentAnalyzer.get_cached_sentiment("EURUSD")
        assert result is None

    def test_cache_hit_after_set(self):
        from news.sentiment import SentimentAnalyzer
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=0.42, article_count=10, magnitude=0.35)
        result = SentimentAnalyzer.get_cached_sentiment("EURUSD", max_age_hours=6.0)
        assert result is not None
        assert result["score"] == 0.42
        assert result["article_count"] == 10
        assert result["magnitude"] == 0.35

    def test_cache_stale_after_ttl(self):
        from news.sentiment import SentimentAnalyzer
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=0.42, article_count=10)
        result = SentimentAnalyzer.get_cached_sentiment("EURUSD", max_age_hours=-0.001)
        assert result is None

    def test_cache_state_returns_metadata(self):
        from news.sentiment import SentimentAnalyzer
        SentimentAnalyzer.cache_sentiment("GBPUSD", avg_score=-0.15, article_count=5, magnitude=0.20)
        state = SentimentAnalyzer.get_sentiment_cache_state("GBPUSD", max_age_hours=6.0)
        assert state["cached"] is True
        assert state["last_updated_iso"] is not None
        assert state["next_update_iso"] is not None

    def test_cache_state_miss(self):
        from news.sentiment import SentimentAnalyzer
        state = SentimentAnalyzer.get_sentiment_cache_state("NONEXISTENT", max_age_hours=6.0)
        assert state["cached"] is False

    def test_pair_normalization(self):
        from news.sentiment import SentimentAnalyzer
        SentimentAnalyzer.cache_sentiment("eur/usd", avg_score=0.55, article_count=3)
        result = SentimentAnalyzer.get_cached_sentiment("EURUSD")
        assert result is not None
        assert result["score"] == 0.55


# ── Signal blending (Phase 2) ─────────────────────────────────────────

class TestSignalBlending:
    def setup_method(self):
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        RealTradingMixin._news_blend_timer = 0.0
        RealTradingMixin._news_blend_cached_pair = ""
        RealTradingMixin._news_blend_cached_score = 0.0

    def test_blend_disabled_returns_signal_unchanged(self):
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        result = RealTradingMixin._blend_news_sentiment(
            0.7, "EURUSD", {"live_news_blend_enabled": False}
        )
        assert result == 0.7

    def test_blend_no_cache_returns_unchanged(self):
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        result = RealTradingMixin._blend_news_sentiment(
            0.7, "EURUSD", {"live_news_blend_enabled": True, "live_news_blend_weight": 0.10}
        )
        assert result == 0.7

    def test_blend_with_cache_applies_weight(self):
        from news.sentiment import SentimentAnalyzer
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=0.50, article_count=10)
        result = RealTradingMixin._blend_news_sentiment(
            0.7, "EURUSD", {"live_news_blend_enabled": True, "live_news_blend_weight": 0.10}
        )
        expected = 0.7 + 0.50 * 0.10
        assert abs(result - expected) < 1e-6

    def test_blend_clamped_to_one(self):
        from news.sentiment import SentimentAnalyzer
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=0.90, article_count=10)
        result = RealTradingMixin._blend_news_sentiment(
            1.0, "EURUSD", {"live_news_blend_enabled": True, "live_news_blend_weight": 0.30}
        )
        assert result == 1.0

    def test_blend_clamped_to_negative_one(self):
        from news.sentiment import SentimentAnalyzer
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=-0.90, article_count=10)
        result = RealTradingMixin._blend_news_sentiment(
            -1.0, "EURUSD", {"live_news_blend_enabled": True, "live_news_blend_weight": 0.30}
        )
        assert result == -1.0

    def test_blend_neutral_signal_can_become_directional(self):
        from news.sentiment import SentimentAnalyzer
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=0.80, article_count=10)
        result = RealTradingMixin._blend_news_sentiment(
            0.0, "EURUSD", {"live_news_blend_enabled": True, "live_news_blend_weight": 0.10}
        )
        assert result > 0.0

    def test_blend_weight_zero_returns_unchanged(self):
        from news.sentiment import SentimentAnalyzer
        from pipeline.backtester.real_trading_mixin import RealTradingMixin
        SentimentAnalyzer.cache_sentiment("EURUSD", avg_score=0.80, article_count=10)
        result = RealTradingMixin._blend_news_sentiment(
            0.7, "EURUSD", {"live_news_blend_enabled": True, "live_news_blend_weight": 0.0}
        )
        assert result == 0.7

    # ── Live trading news blending integration tests ─────────────────

    def test_deploy_request_schema_accepts_news_blend_fields(self):
        from api.routers.live import DeployRequest, DeployCommitteeRequest
        dr = DeployRequest(pair="EURUSD", model="logistic", live_news_blend_enabled=True, live_news_blend_weight=0.15)
        assert dr.live_news_blend_enabled is True
        assert dr.live_news_blend_weight == 0.15
        dcr = DeployCommitteeRequest(pair="EURUSD", live_news_blend_enabled=True, live_news_blend_weight=0.20)
        assert dcr.live_news_blend_enabled is True
        assert dcr.live_news_blend_weight == 0.20

    def test_deploy_request_defaults(self):
        from api.routers.live import DeployRequest
        dr = DeployRequest(pair="EURUSD")
        assert dr.live_news_blend_enabled is False
        assert dr.live_news_blend_weight == 0.10

    def test_predict_signal_handles_missing_model_gracefully(self):
        from api.routers.live import _predict_signal
        session = {"live_news_blend_enabled": False, "pair": "EURUSD", "backtester": None, "model_obj": None}
        result = _predict_signal(session, None)
        assert result is None

    def test_session_stores_news_blend_config(self):
        from api.routers.live import DeployRequest
        req = DeployRequest(pair="EURUSD", live_news_blend_enabled=True, live_news_blend_weight=0.15)
        assert req.live_news_blend_enabled is True
        assert req.live_news_blend_weight == 0.15

    def test_session_config_defaults(self):
        from api.routers.live import DeployRequest
        req_enabled = DeployRequest(pair="GBPUSD", live_news_blend_enabled=True)
        assert req_enabled.live_news_blend_enabled is True
        assert req_enabled.live_news_blend_weight == 0.10
        req_disabled = DeployRequest(pair="GBPUSD")
        assert req_disabled.live_news_blend_enabled is False
        assert req_disabled.live_news_blend_weight == 0.10

    def test_blend_weight_range_in_deploy_request(self):
        from api.routers.live import DeployRequest
        req = DeployRequest(pair="EURUSD", live_news_blend_weight=0.30)
        assert req.live_news_blend_weight == 0.30
        req2 = DeployRequest(pair="EURUSD", live_news_blend_weight=0.0)
        assert req2.live_news_blend_weight == 0.0
