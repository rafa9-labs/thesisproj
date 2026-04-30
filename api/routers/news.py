"""News & sentiment status + events endpoints."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

from api.schemas.news import NewsEventItem, NewsEventsResponse
from news.scraper import NewsScraper

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/status")
def news_status():
    cache_dir = Path(os.environ.get("NEWS_CACHE_DIR", "news_cache"))
    articles = 0
    events = []
    sentiment_backend = "vader"

    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            try:
                import pyarrow.parquet as pq
                articles += pq.read_metadata(str(f)).num_rows
            except Exception:
                pass

    return {
        "sentiment_backend": sentiment_backend,
        "cached_articles": articles,
        "event_types": ["NFP", "FOMC", "CPI", "GDP", "Retail_Sales", "PMI", "ECB_Rate", "BOE_Rate"],
        "features": {
            "vader_compound": True,
            "event_flags": True,
            "news_volume_windows": [6, 24],
        },
        "finbert_available": False,
    }


@router.get("/events", response_model=NewsEventsResponse)
def get_news_events(
    start: Optional[int] = Query(None, description="Start unix timestamp (seconds)"),
    end: Optional[int] = Query(None, description="End unix timestamp (seconds)"),
    impact: Optional[str] = Query(None, description="Comma-separated impact levels: high,medium,low"),
):
    impact_levels = set()
    if impact:
        impact_levels = {i.strip().lower() for i in impact.split(",") if i.strip()}

    start_dt = datetime.fromtimestamp(start, tz=timezone.utc) if start else None
    end_dt = datetime.fromtimestamp(end, tz=timezone.utc) if end else None

    years_to_check = set()
    if start_dt:
        years_to_check.add(start_dt.year)
    if end_dt:
        years_to_check.add(end_dt.year)
    if not years_to_check:
        years_to_check = {datetime.now(tz=timezone.utc).year}

    all_events = []
    for year in sorted(years_to_check):
        raw = NewsScraper.economic_calendar_events(year)
        for ev in raw:
            ev_dt = ev["date"]
            if not isinstance(ev_dt, datetime):
                continue

            ev_time = int(ev_dt.replace(tzinfo=timezone.utc).timestamp())

            if start_dt and ev_dt < start_dt.replace(tzinfo=None):
                continue
            if end_dt and ev_dt > end_dt.replace(tzinfo=None):
                continue

            impact_val = {3: "high", 2: "medium", 1: "low"}.get(ev.get("impact", 1), "low")
            if impact_levels and impact_val not in impact_levels:
                continue

            all_events.append(
                NewsEventItem(
                    time=ev_time,
                    event=ev["event"],
                    currency="USD",
                    impact=impact_val,
                )
            )

    all_events.sort(key=lambda e: e.time)
    return NewsEventsResponse(events=all_events, count=len(all_events))