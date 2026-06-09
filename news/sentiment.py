"""Sentiment analysis for news articles.

Provides two backends:
- **VADER** (default): fast, rule-based, zero heavy dependencies.
- **finBERT** (opt-in): financial-domain BERT via HuggingFace transformers.

Per-article scoring is aggregated into time-bucketed DataFrames
(hourly or daily) suitable for merging with OHLC bars.

Sentiment results are cached per pair with a configurable TTL
(default 6 hours) to avoid re-scoring on every bar in live trading.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SENTIMENT_CACHE: Dict[str, Tuple[float, int, float, float]] = {}
"""Per-pair cache: {pair: (avg_score, article_count, timestamp, magnitude)}"""


@dataclass
class SentimentResult:
    score: float
    magnitude: float
    positive: float
    negative: float
    neutral: float
    backend: str


class SentimentAnalyzer:
    """Score news articles using VADER or finBERT.

    Parameters
    ----------
    backend : str
        ``"vader"`` (default) or ``"finbert"``.
    """

    def __init__(self, backend: str = "vader"):
        backend = backend.lower().strip()
        if backend not in ("vader", "finbert"):
            raise ValueError(f"Unknown sentiment backend: {backend!r} (use 'vader' or 'finbert')")
        self.backend = backend
        self._vader = None
        self._finbert_pipeline = None

    def _init_vader(self):
        if self._vader is not None:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError(
                "vaderSentiment is required for VADER sentiment. "
                "Install with: pip install vaderSentiment"
            )

    def _init_finbert(self):
        if self._finbert_pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
            )
        except ImportError:
            raise ImportError(
                "transformers + torch are required for finBERT. "
                "Install with: pip install transformers torch"
            )

    def score_text(self, text: str) -> SentimentResult:
        """Score a single text string.

        Parameters
        ----------
        text : str

        Returns
        -------
        SentimentResult
        """
        if not text or not text.strip():
            return SentimentResult(
                score=0.0, magnitude=0.0,
                positive=0.0, negative=0.0, neutral=1.0,
                backend=self.backend,
            )

        if self.backend == "vader":
            return self._score_vader(text)
        else:
            return self._score_finbert(text)

    def _score_vader(self, text: str) -> SentimentResult:
        self._init_vader()
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        return SentimentResult(
            score=compound,
            magnitude=abs(compound),
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            backend="vader",
        )

    def _score_finbert(self, text: str) -> SentimentResult:
        self._init_finbert()
        text_chunk = text[:512]
        results = self._finbert_pipeline(text_chunk)
        label_map = {}
        for r in results:
            label_map[r["label"].lower()] = r["score"]

        pos = label_map.get("positive", 0.0)
        neg = label_map.get("negative", 0.0)
        neu = label_map.get("neutral", 1.0)
        score = pos - neg

        return SentimentResult(
            score=score,
            magnitude=max(pos, neg, neu),
            positive=pos,
            negative=neg,
            neutral=neu,
            backend="finbert",
        )

    @staticmethod
    def get_cached_sentiment(
        pair: str,
        max_age_hours: float = 6.0,
    ) -> Optional[Dict]:
        """Retrieve cached sentiment for a pair if still fresh.

        Parameters
        ----------
        pair : str
            Currency pair (e.g. EURUSD).
        max_age_hours : float
            Maximum age in hours before cache is considered stale.

        Returns
        -------
        dict or None
            ``{score, article_count, magnitude, last_updated_iso}``
            or None if cache miss or stale.
        """
        pair_key = pair.upper().replace("/", "").replace("-", "")
        entry = _SENTIMENT_CACHE.get(pair_key)
        if entry is None:
            return None
        avg_score, article_count, cached_ts, magnitude = entry
        age_hours = (time.time() - cached_ts) / 3600.0
        if age_hours > max_age_hours:
            return None
        last_updated = datetime.fromtimestamp(cached_ts, tz=timezone.utc).isoformat()
        return {
            "score": float(avg_score),
            "article_count": int(article_count),
            "magnitude": float(magnitude),
            "last_updated_iso": last_updated,
            "cache_age_hours": round(age_hours, 2),
        }

    @staticmethod
    def cache_sentiment(
        pair: str,
        avg_score: float,
        article_count: int,
        magnitude: float = 0.0,
    ) -> None:
        """Store sentiment result in cache."""
        pair_key = pair.upper().replace("/", "").replace("-", "")
        _SENTIMENT_CACHE[pair_key] = (avg_score, article_count, time.time(), magnitude)

    @staticmethod
    def recency_weight(timestamp, half_life_days: float = 3.0) -> float:
        """Exponential decay weight based on article age.

        ``weight = 2^(-age_days / half_life_days)``
        Today's article → 1.0, half_life days ago → 0.5, 2× half_life → 0.25.
        """
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return 0.0
        ts = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        age_days = max(0.0, age_hours / 24.0)
        return 2.0 ** (-age_days / half_life_days)

    @staticmethod
    def get_sentiment_cache_state(pair: str, max_age_hours: float = 6.0) -> Dict:
        """Return cache metadata including freshness for UI display."""
        pair_key = pair.upper().replace("/", "").replace("-", "")
        entry = _SENTIMENT_CACHE.get(pair_key)
        if entry is None:
            return {"cached": False, "last_updated_iso": None, "next_update_iso": None, "age_hours": None}
        _, _, cached_ts, _ = entry
        age_hours = (time.time() - cached_ts) / 3600.0
        next_ts = cached_ts + max_age_hours * 3600.0
        return {
            "cached": True,
            "last_updated_iso": datetime.fromtimestamp(cached_ts, tz=timezone.utc).isoformat(),
            "next_update_iso": datetime.fromtimestamp(next_ts, tz=timezone.utc).isoformat(),
            "age_hours": round(age_hours, 2),
        }

    def score_articles(self, articles: list) -> List[tuple]:
        """Score a list of ``NewsArticle`` objects.

        Parameters
        ----------
        articles : list[NewsArticle]

        Returns
        -------
        list[tuple]
            Each tuple: ``(article, SentimentResult)``
        """
        scored = []
        for article in articles:
            text = f"{article.title} {article.body}"
            result = self.score_text(text)
            scored.append((article, result))
        return scored

    def aggregate_to_df(
        self,
        scored_articles: List[tuple],
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Aggregate scored articles into time buckets.

        Parameters
        ----------
        scored_articles : list[tuple]
            Output of :meth:`score_articles`.
        freq : str
            Pandas frequency string. ``"1h"`` (hourly) or ``"1D"`` (daily).

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (bucket start), ``sentiment_score``,
            ``sentiment_magnitude``, ``news_volume``, ``sent_pos``,
            ``sent_neg``, ``sent_neu``.
        """
        if not scored_articles:
            return pd.DataFrame(columns=[
                "timestamp", "sentiment_score", "sentiment_magnitude",
                "news_volume", "sent_pos", "sent_neg", "sent_neu",
            ])

        rows = []
        for article, result in scored_articles:
            ts = article.timestamp
            if not isinstance(ts, datetime):
                try:
                    ts = pd.Timestamp(ts).to_pydatetime()
                except Exception:
                    continue
            rows.append({
                "timestamp": ts,
                "sentiment_score": result.score,
                "sentiment_magnitude": result.magnitude,
                "sent_pos": result.positive,
                "sent_neg": result.negative,
                "sent_neu": result.neutral,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "sentiment_score", "sentiment_magnitude",
                "news_volume", "sent_pos", "sent_neg", "sent_neu",
            ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

        agg = df.groupby(pd.Grouper(freq=freq)).agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_magnitude=("sentiment_magnitude", "mean"),
            news_volume=("sentiment_score", "size"),
            sent_pos=("sent_pos", "mean"),
            sent_neg=("sent_neg", "mean"),
            sent_neu=("sent_neu", "mean"),
        ).reset_index()

        agg = agg[agg["news_volume"] > 0].reset_index(drop=True)
        return agg
