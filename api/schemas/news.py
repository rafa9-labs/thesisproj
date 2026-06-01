"""News event schemas."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class NewsEventItem(BaseModel):
    time: int
    event: str
    currency: str
    impact: str  # "high", "medium", "low"


class NewsEventsResponse(BaseModel):
    events: List[NewsEventItem]
    count: int


class NewsArticleFull(BaseModel):
    title: str
    body: str
    source: str
    url: str
    timestamp: str
    pair_tags: List[str] = []
    sentiment_score: float = 0.0
    summary: str = ""
    bias: str = "neutral"  # "long", "short", "neutral"


class NewsArticlesResponse(BaseModel):
    articles: List[NewsArticleFull]
    total: int
    pair: str = ""