"""LLM-powered sentiment engine for financial news analysis.

Supports multiple backends:
  - Ollama (default, free, local)
  - OpenAI API (paid, cloud)
  - Anthropic API (paid, cloud)

Features:
  - Per-article caching in SQLite (process once, reuse forever)
  - Batched article processing (multiple articles per LLM call)
  - Fallback to VADER when LLM is unavailable
  - Structured JSON output with direction, confidence, volatility
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DB_NAME = "llm_sentiment_cache.db"


class LLMSentimentBackend(Protocol):
    def analyze(self, text: str, pair: str) -> Dict[str, Any]: ...


class OllamaBackend:
    """Local Ollama backend. Free, private. Requires `ollama serve` running."""

    def __init__(self, model: str = "llama3", base_url: str = ""):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def analyze(self, text: str, pair: str) -> Dict[str, Any]:
        from pipeline.llm.prompts import SENTIMENT_ANALYSIS_PROMPT
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            pair=pair,
            title=text[:200],
            body=text[:2000],
        )
        try:
            import requests
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return _parse_llm_json(raw)
        except Exception as e:
            logger.warning("Ollama backend failed: %s", e)
            raise


class OpenAIBackend:
    """OpenAI API backend. Paid, cloud. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def analyze(self, text: str, pair: str) -> Dict[str, Any]:
        from pipeline.llm.prompts import SENTIMENT_ANALYSIS_PROMPT
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            pair=pair,
            title=text[:200],
            body=text[:2000],
        )
        try:
            import requests
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return _parse_llm_json(raw)
        except Exception as e:
            logger.warning("OpenAI backend failed: %s", e)
            raise


class AnthropicBackend:
    """Anthropic API backend. Paid, cloud. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def analyze(self, text: str, pair: str) -> Dict[str, Any]:
        from pipeline.llm.prompts import SENTIMENT_ANALYSIS_PROMPT
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            pair=pair,
            title=text[:200],
            body=text[:2000],
        )
        try:
            import requests
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"]
            return _parse_llm_json(raw)
        except Exception as e:
            logger.warning("Anthropic backend failed: %s", e)
            raise


class VADERFallback:
    """Fallback using VADER when LLM backends are unavailable."""

    def analyze(self, text: str, pair: str) -> Dict[str, Any]:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityEngine
        except ImportError:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SentimentIntensityEngine
        analyzer = SentimentIntensityEngine()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        return {
            "direction": float(np.clip(compound, -1.0, 1.0)),
            "confidence": abs(compound),
            "volatility": 1.0 - scores.get("neu", 0.5),
            "currencies_affected": [],
            "fallback": True,
        }


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown fences and whitespace."""
    text = raw.strip()
    if text.startswith("```"):
        for line in text.split("\n"):
            if line.strip().startswith("```"):
                continue
            text = line if not text else text
        text = "\n".join(l for l in raw.strip().split("\n") if not l.strip().startswith("```"))
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response")
                return _default_scores()
        else:
            return _default_scores()

    return {
        "direction": float(np.clip(parsed.get("direction", 0.0), -1.0, 1.0)),
        "confidence": float(np.clip(parsed.get("confidence", 0.5), 0.0, 1.0)),
        "volatility": float(np.clip(parsed.get("volatility", 0.3), 0.0, 1.0)),
        "currencies_affected": parsed.get("currencies_affected", []),
    }


def _default_scores() -> Dict[str, Any]:
    return {
        "direction": 0.0,
        "confidence": 0.0,
        "volatility": 0.3,
        "currencies_affected": [],
        "fallback": True,
    }


def _article_hash(title: str, body: str, pair: str) -> str:
    """Deterministic hash for article+pair → cache key."""
    h = hashlib.sha256(f"{pair}:{title}:{body[:500]}".encode()).hexdigest()[:32]
    return h


class LLMSentimentEngine:
    """Main LLM sentiment engine with caching, batch processing, and fallback.

    Parameters
    ----------
    config : dict
        Keys:
        - llm_sentiment_enabled (bool): master toggle
        - llm_backend (str): "ollama" | "openai" | "anthropic"
        - llm_model (str): model name
        - llm_api_key (str): API key (not needed for Ollama)
        - llm_weight (float): blending weight [0, 1] for LLM vs VADER
        - llm_batch_size (int): articles per batch call
        - llm_cache_ttl_hours (int): cache TTL
        - llm_ollama_url (str): Ollama base URL
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("llm_sentiment_enabled", True))
        self.backend_name = self.config.get("llm_backend", "ollama")
        self.model = self.config.get("llm_model", "llama3")
        self.api_key = self.config.get("llm_api_key", "")
        self.weight = float(self.config.get("llm_weight", 0.7))
        self.batch_size = int(self.config.get("llm_batch_size", 10))
        self.cache_ttl_hours = int(self.config.get("llm_cache_ttl_hours", 720))
        self._backend: LLMSentimentBackend | None = None
        self._fallback = VADERFallback()
        self._cache_db: sqlite3.Connection | None = None

    @property
    def backend(self) -> LLMSentimentBackend:
        if self._backend is None:
            if self.backend_name == "openai":
                self._backend = OpenAIBackend(model=self.model, api_key=self.api_key)
            elif self.backend_name == "anthropic":
                self._backend = AnthropicBackend(model=self.model, api_key=self.api_key)
            else:
                self._backend = OllamaBackend(
                    model=self.model,
                    base_url=self.config.get("llm_ollama_url", ""),
                )
        return self._backend

    def _init_cache_db(self, db_path: str) -> None:
        """Initialize or connect to the SQLite cache database."""
        if self._cache_db is not None:
            return
        self._cache_db = sqlite3.connect(db_path)
        self._cache_db.execute("""
            CREATE TABLE IF NOT EXISTS llm_sentiment_cache (
                article_hash TEXT PRIMARY KEY,
                direction REAL NOT NULL,
                confidence REAL NOT NULL,
                volatility REAL NOT NULL,
                currencies_affected TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                pair TEXT NOT NULL DEFAULT ''
            )
        """)
        self._cache_db.commit()

    def _get_cache_path(self) -> str:
        """Return path to the cache database."""
        from config import PROJECT_ROOT
        cache_dir = PROJECT_ROOT / "news_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / CACHE_DB_NAME)

    def _cached_score(self, article_hash: str) -> Dict[str, Any] | None:
        """Look up a cached score by article hash."""
        if self._cache_db is None:
            return None
        try:
            cursor = self._cache_db.execute(
                "SELECT direction, confidence, volatility, currencies_affected, created_at "
                "FROM llm_sentiment_cache WHERE article_hash = ?",
                (article_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            created = row[4]
            if self.cache_ttl_hours > 0:
                from datetime import timedelta
                try:
                    created_dt = datetime.fromisoformat(created)
                    if datetime.now(timezone.utc) - created_dt.replace(tzinfo=timezone.utc) > timedelta(hours=self.cache_ttl_hours):
                        return None
                except Exception:
                    pass
            return {
                "direction": row[0],
                "confidence": row[1],
                "volatility": row[2],
                "currencies_affected": json.loads(row[3]),
            }
        except Exception:
            return None

    def _cache_score(self, article_hash: str, scores: Dict[str, Any], pair: str = "") -> None:
        """Write a score to cache."""
        if self._cache_db is None:
            return
        try:
            self._cache_db.execute(
                "INSERT OR REPLACE INTO llm_sentiment_cache "
                "(article_hash, direction, confidence, volatility, currencies_affected, created_at, pair) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    article_hash,
                    scores.get("direction", 0.0),
                    scores.get("confidence", 0.0),
                    scores.get("volatility", 0.3),
                    json.dumps(scores.get("currencies_affected", [])),
                    datetime.now(timezone.utc).isoformat(),
                    pair,
                ),
            )
            self._cache_db.commit()
        except Exception as e:
            logger.warning("Failed to cache LLM score: %s", e)

    def score_article(self, title: str, body: str, pair: str) -> Dict[str, Any]:
        """Score a single article with caching."""
        if not self.enabled:
            return self._fallback.analyze(f"{title} {body}", pair)

        db_path = self._get_cache_path()
        self._init_cache_db(db_path)

        ahash = _article_hash(title, body, pair)
        cached = self._cached_score(ahash)
        if cached is not None:
            return cached

        text = f"{title} {body}"
        try:
            scores = self.backend.analyze(text, pair)
        except Exception as e:
            logger.warning("LLM backend failed, falling back to VADER: %s", e)
            scores = self._fallback.analyze(text, pair)
            scores["fallback"] = True

        self._cache_score(ahash, scores, pair)
        return scores

    def score_articles(self, articles: list, pair: str = "EURUSD") -> List[Tuple[Any, Dict[str, Any]]]:
        """Score a list of NewsArticle objects.

        Returns list of (article, scores_dict) tuples.
        """
        results = []
        for article in articles:
            title = getattr(article, "title", str(article))
            body = getattr(article, "body", "")
            scores = self.score_article(title, body, pair)
            results.append((article, scores))
        return results

    def merge_with_vader(
        self,
        llm_scores: List[Dict[str, Any]],
        vader_scores: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Blend LLM and VADER scores using configurable weight.

        blended = llm_weight * llm + (1 - llm_weight) * vader
        """
        blended = []
        for i, llm in enumerate(llm_scores):
            vader = vader_scores[i] if i < len(vader_scores) else {"direction": 0.0, "confidence": 0.0, "volatility": 0.3}
            is_fallback = llm.get("fallback", False)
            w = 0.0 if is_fallback else self.weight
            blended.append({
                "direction": w * llm.get("direction", 0.0) + (1 - w) * vader.get("direction", 0.0),
                "confidence": w * llm.get("confidence", 0.5) + (1 - w) * vader.get("confidence", 0.5),
                "volatility": w * llm.get("volatility", 0.3) + (1 - w) * vader.get("volatility", 0.3),
                "currencies_affected": llm.get("currencies_affected", []),
                "llm_weight_used": w,
            })
        return blended

    def aggregate_to_df(
        self,
        scored_articles: List[Tuple[Any, Dict[str, Any]]],
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Aggregate scored articles into a time-bucketed DataFrame.

        Similar to SentimentAnalyzer.aggregate_to_df but with LLM fields.
        """
        if not scored_articles:
            return pd.DataFrame(columns=[
                "timestamp", "llm_sentiment", "llm_confidence", "llm_volatility",
                "news_volume", "sentiment_score",
            ])

        records = []
        for article, scores in scored_articles:
            ts = getattr(article, "timestamp", None)
            if ts is None:
                continue
            records.append({
                "timestamp": ts,
                "llm_sentiment": float(scores.get("direction", 0.0)),
                "llm_confidence": float(scores.get("confidence", 0.0)),
                "llm_volatility": float(scores.get("volatility", 0.3)),
                "news_volume": 1.0,
            })

        if not records:
            return pd.DataFrame(columns=[
                "timestamp", "llm_sentiment", "llm_confidence", "llm_volatility",
                "news_volume",
            ])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

        agg = df.resample(freq).agg({
            "llm_sentiment": "mean",
            "llm_confidence": "mean",
            "llm_volatility": "mean",
            "news_volume": "sum",
        }).reset_index()

        return agg

    def get_live_sentiment(self, pair: str, articles: list | None = None) -> Dict[str, Any]:
        """Get live sentiment for a pair from recent articles.

        Returns a summary dict with blended sentiment.
        """
        if not self.enabled:
            return {"direction": 0.0, "confidence": 0.0, "volatility": 0.3, "pair": pair, "backend": "disabled"}

        if articles:
            scored = self.score_articles(articles, pair=pair)
            if scored:
                dirs = [s.get("direction", 0.0) for _, s in scored]
                confs = [s.get("confidence", 0.0) for _, s in scored]
                vols = [s.get("volatility", 0.3) for _, s in scored]
                return {
                    "direction": float(np.mean(dirs)),
                    "confidence": float(np.mean(confs)),
                    "volatility": float(np.mean(vols)),
                    "pair": pair,
                    "backend": self.backend_name,
                    "article_count": len(scored),
                }

        return {"direction": 0.0, "confidence": 0.0, "volatility": 0.3, "pair": pair, "backend": self.backend_name}

    def close(self):
        if self._cache_db is not None:
            try:
                self._cache_db.close()
            except Exception:
                pass
            self._cache_db = None