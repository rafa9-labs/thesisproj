"""Sentiment analysis for news articles.

Provides two backends:
- **VADER** (default): fast, rule-based, zero heavy dependencies.
- **finBERT** (opt-in): financial-domain BERT via HuggingFace transformers.

Per-article scoring is aggregated into time-bucketed DataFrames
(hourly or daily) suitable for merging with OHLC bars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
