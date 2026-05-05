"""Tests for the LLM sentiment engine: backends, caching, blending, features."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys_path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path_root not in os.sys.path:
    os.sys.path.insert(0, sys_path_root)

from pipeline.llm.sentiment import (
    LLMSentimentEngine,
    OllamaBackend,
    OpenAIBackend,
    AnthropicBackend,
    VADERFallback,
    _parse_llm_json,
    _default_scores,
    _article_hash,
)


# ── _parse_llm_json ───────────────────────────────────────────────────

class TestParseLLMJson:
    def test_valid_json(self):
        raw = '{"direction": 0.5, "confidence": 0.8, "volatility": 0.3, "currencies_affected": ["USD"]}'
        result = _parse_llm_json(raw)
        assert result["direction"] == pytest.approx(0.5)
        assert result["confidence"] == pytest.approx(0.8)
        assert result["volatility"] == pytest.approx(0.3)
        assert result["currencies_affected"] == ["USD"]

    def test_json_with_markdown_fence(self):
        raw = '```json\n{"direction": 0.5, "confidence": 0.8, "volatility": 0.3, "currencies_affected": []}\n```'
        result = _parse_llm_json(raw)
        assert result["direction"] == pytest.approx(0.5)

    def test_json_with_surrounding_text(self):
        raw = 'Here is the analysis: {"direction": -0.2, "confidence": 0.6, "volatility": 0.7, "currencies_affected": ["EUR"]} End.'
        result = _parse_llm_json(raw)
        assert result["direction"] == pytest.approx(-0.2)

    def test_invalid_json_returns_default(self):
        result = _parse_llm_json("not json at all")
        assert result == _default_scores()

    def test_direction_clipped(self):
        raw = '{"direction": 2.0, "confidence": 0.5, "volatility": 0.3}'
        result = _parse_llm_json(raw)
        assert result["direction"] == 1.0

    def test_negative_direction_clipped(self):
        raw = '{"direction": -3.0, "confidence": 0.5, "volatility": 0.3}'
        result = _parse_llm_json(raw)
        assert result["direction"] == -1.0


# ── _article_hash ─────────────────────────────────────────────────────

class TestArticleHash:
    def test_deterministic(self):
        h1 = _article_hash("EUR rises", "body text", "EURUSD")
        h2 = _article_hash("EUR rises", "body text", "EURUSD")
        assert h1 == h2

    def test_different_title_different_hash(self):
        h1 = _article_hash("EUR rises", "body", "EURUSD")
        h2 = _article_hash("GBP falls", "body", "EURUSD")
        assert h1 != h2

    def test_different_pair_different_hash(self):
        h1 = _article_hash("Same title", "body", "EURUSD")
        h2 = _article_hash("Same title", "body", "GBPUSD")
        assert h1 != h2


# ── VADERFallback ─────────────────────────────────────────────────────

class TestVADERFallback:
    def test_returns_direction(self):
        fb = VADERFallback()
        result = fb.analyze("Euro rises strongly against dollar", "EURUSD")
        assert -1.0 <= result["direction"] <= 1.0
        assert "fallback" in result
        assert result["fallback"] is True


# ── OllamaBackend (mocked) ────────────────────────────────────────────

class TestOllamaBackendMock:
    @patch("requests.post")
    def test_ollama_analyze(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": '{"direction": 0.4, "confidence": 0.7, "volatility": 0.5, "currencies_affected": ["EUR"]}'
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        backend = OllamaBackend(model="llama3")
        result = backend.analyze("EUR rises on strong data", "EURUSD")
        assert result["direction"] == pytest.approx(0.4)
        assert result["confidence"] == pytest.approx(0.7)

    @patch("requests.post")
    def test_ollama_connection_failure(self, mock_post):
        mock_post.side_effect = ConnectionError("Ollama not running")
        backend = OllamaBackend(model="llama3")
        with pytest.raises(ConnectionError):
            backend.analyze("test", "EURUSD")


# ── OpenAIBackend (mocked) ───────────────────────────────────────────

class TestOpenAIBackendMock:
    @patch("requests.post")
    def test_openai_analyze(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"direction": -0.3, "confidence": 0.9, "volatility": 0.6, "currencies_affected": ["USD"]}'}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        backend = OpenAIBackend(model="gpt-4o-mini", api_key="sk-test")
        result = backend.analyze("Dollar weakens", "EURUSD")
        assert result["direction"] == pytest.approx(-0.3)


# ── LLMSentimentEngine ────────────────────────────────────────────────

class TestLLMSentimentEngine:
    def test_engine_created_with_defaults(self):
        engine = LLMSentimentEngine()
        assert engine.backend_name == "ollama"
        assert engine.weight == 0.7
        assert engine.enabled is True

    def test_engine_created_with_custom_config(self):
        cfg = {"llm_backend": "openai", "llm_weight": 0.9, "llm_model": "gpt-4o-mini"}
        engine = LLMSentimentEngine(config=cfg)
        assert engine.backend_name == "openai"
        assert engine.weight == 0.9

    def test_disabled_engine_uses_fallback(self):
        engine = LLMSentimentEngine(config={"llm_sentiment_enabled": False})
        result = engine.score_article("EUR rises", "body", "EURUSD")
        assert result.get("fallback") is True

    def test_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.db")
            engine = LLMSentimentEngine(config={"llm_sentiment_enabled": False})
            engine._init_cache_db(db_path)

            ahash = _article_hash("test title", "test body", "EURUSD")
            scores = {"direction": 0.5, "confidence": 0.8, "volatility": 0.3, "currencies_affected": []}
            engine._cache_score(ahash, scores, "EURUSD")

            cached = engine._cached_score(ahash)
            assert cached is not None
            assert cached["direction"] == pytest.approx(0.5)
            assert cached["confidence"] == pytest.approx(0.8)

            engine.close()

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.db")
            engine = LLMSentimentEngine(config={"llm_sentiment_enabled": False})
            engine._init_cache_db(db_path)

            cached = engine._cached_score("nonexistent_hash")
            assert cached is None
            engine.close()

    def test_blending_formula(self):
        engine = LLMSentimentEngine(config={"llm_weight": 0.7})
        llm_scores = [{"direction": 0.8, "confidence": 0.9, "volatility": 0.5}]
        vader_scores = [{"direction": 0.2, "confidence": 0.5, "volatility": 0.3}]
        blended = engine.merge_with_vader(llm_scores, vader_scores)
        assert len(blended) == 1
        assert blended[0]["direction"] == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)
        assert blended[0]["llm_weight_used"] == 0.7

    def test_blending_fallback_zero_weight(self):
        engine = LLMSentimentEngine(config={"llm_weight": 0.7})
        llm_scores = [{"direction": 0.8, "confidence": 0.9, "volatility": 0.5, "fallback": True}]
        vader_scores = [{"direction": 0.2, "confidence": 0.5, "volatility": 0.3}]
        blended = engine.merge_with_vader(llm_scores, vader_scores)
        assert blended[0]["direction"] == pytest.approx(0.2)
        assert blended[0]["llm_weight_used"] == 0.0

    def test_aggregate_to_df_empty(self):
        engine = LLMSentimentEngine()
        result = engine.aggregate_to_df([], freq="1h")
        assert len(result) == 0

    def test_aggregate_to_df_with_articles(self):
        article_mock = MagicMock()
        article_mock.timestamp = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
        scores = {"direction": 0.5, "confidence": 0.8, "volatility": 0.3}

        engine = LLMSentimentEngine()
        result = engine.aggregate_to_df([(article_mock, scores)], freq="1h")
        assert len(result) == 1
        assert result.iloc[0]["llm_sentiment"] == pytest.approx(0.5)
        assert result.iloc[0]["llm_confidence"] == pytest.approx(0.8)


# ── Feature integration ───────────────────────────────────────────────

class TestLLMFeatureIntegration:
    def test_merge_llm_features_adds_columns(self):
        from news.features import merge_llm_features
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "sentiment_score": np.random.randn(100).astype(np.float32) * 0.1,
        }, index=dates)

        llm_df = pd.DataFrame({
            "timestamp": dates[:50],
            "llm_sentiment": np.random.randn(50).astype(np.float32),
            "llm_confidence": np.abs(np.random.randn(50)).astype(np.float32) * 0.5,
            "llm_volatility": np.abs(np.random.randn(50)).astype(np.float32) * 0.3,
        })

        result = merge_llm_features(df, llm_df, config={"llm_sentiment_enabled": True})
        assert "llm_sentiment" in result.columns
        assert "llm_confidence" in result.columns
        assert "llm_volatility" in result.columns
        assert "blended_sentiment" in result.columns
        assert "llm_sentiment_ma_6" in result.columns
        assert "llm_sentiment_ma_24" in result.columns

    def test_merge_llm_features_disabled(self):
        from news.features import merge_llm_features
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100}, index=dates)
        result = merge_llm_features(df, pd.DataFrame(), config={"llm_sentiment_enabled": False})
        assert "llm_sentiment" not in result.columns

    def test_get_llm_feature_columns(self):
        from news.features import get_llm_feature_columns
        cols = get_llm_feature_columns()
        assert "llm_sentiment" in cols
        assert "blended_sentiment" in cols
        assert "llm_sentiment_ma_6" in cols

    def test_blended_sentiment_formula(self):
        from news.features import merge_llm_features
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        vader_scores = np.full(50, 0.2, dtype=np.float32)
        df = pd.DataFrame({
            "close": np.random.randn(50).cumsum() + 100,
            "sentiment_score": vader_scores,
        }, index=dates)

        llm_df = pd.DataFrame({
            "timestamp": dates[:25],
            "llm_sentiment": np.full(25, 0.8, dtype=np.float32),
            "llm_confidence": np.full(25, 0.9, dtype=np.float32),
            "llm_volatility": np.full(25, 0.3, dtype=np.float32),
        })

        result = merge_llm_features(df, llm_df, config={"llm_sentiment_enabled": True, "llm_weight": 0.7})
        expected = 0.7 * 0.8 + 0.3 * 0.2
        assert result["blended_sentiment"].iloc[24] == pytest.approx(expected, abs=0.01)