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