"""News & sentiment status + events endpoints."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

from api.schemas.news import NewsArticleFull, NewsArticlesResponse, NewsEventItem, NewsEventsResponse
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


@router.get("/articles", response_model=NewsArticlesResponse)
def get_news_articles(
    pair: str = Query("", description="Optional currency pair filter (e.g. EURUSD)"),
    days: int = Query(30, description="Max age of articles in days"),
):
    """Return all cached articles with sentiment scores.

    Articles are sorted by sentiment magnitude (most opinionated first).
    Filter by pair optionally; returns all articles if pair is empty.
    """
    try:
        from news.scraper import NewsScraper
        from news.sentiment import SentimentAnalyzer
        scraper = NewsScraper()
        articles = scraper.fetch_all()

        if pair and pair.strip():
            articles = NewsScraper.filter_by_pair(articles, pair.strip().upper())

        now = datetime.now(timezone.utc)
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if days > 0:
            from datetime import timedelta
            cutoff = now - timedelta(days=days)
        articles = [a for a in articles if a.timestamp.replace(tzinfo=timezone.utc) >= cutoff]

        vader_analyzer = SentimentAnalyzer(backend="vader")
        scored = vader_analyzer.score_articles(articles)

        result = []
        for article, s in sorted(scored, key=lambda x: abs(x[1].score), reverse=True):
            score = round(s.score, 4)
            result.append(NewsArticleFull(
                title=article.title[:200],
                body=(article.body or "")[:500],
                source=article.source,
                url=article.url or "",
                timestamp=article.timestamp.isoformat() if hasattr(article.timestamp, "isoformat") else str(article.timestamp),
                pair_tags=article.pair_tags or [],
                sentiment_score=score,
                summary=article.summary,
                bias=NewsArticle.bias_label(score),
            ))

        return NewsArticlesResponse(articles=result, total=len(result), pair=pair or "all")
    except Exception as e:
        return NewsArticlesResponse(articles=[], total=0, pair=pair or "all")


@router.get("/sentiment/live")
def get_live_sentiment(
    pair: str = Query("EURUSD", description="Currency pair for sentiment analysis"),
):
    """Get live sentiment data for a currency pair.

    Aggregates VADER sentiment from cached news articles and optionally
    LLM-sentiment if configured.
    """
    try:
        from news.scraper import NewsScraper
        from news.sentiment import SentimentAnalyzer
        scraper = NewsScraper()
        articles = scraper.fetch_all()
        filtered = NewsScraper.filter_by_pair(articles, pair)

        vader_analyzer = SentimentAnalyzer(backend="vader")
        scored_vader = vader_analyzer.score_articles(filtered)

        vader_directions = [s.score for _, s in scored_vader]
        vader_magnitudes = [s.magnitude for _, s in scored_vader]

        vader_avg = sum(vader_directions) / max(len(vader_directions), 1)
        vader_mag = sum(vader_magnitudes) / max(len(vader_magnitudes), 1)

        top_articles = []
        for article, s in sorted(scored_vader, key=lambda x: abs(x[1].score), reverse=True)[:5]:
            score = round(s.score, 4)
            top_articles.append({
                "title": article.title[:120],
                "body": (article.body or "")[:250],
                "source": article.source,
                "url": article.url or "",
                "pair_tags": article.pair_tags or [],
                "sentiment_score": score,
                "summary": article.summary,
                "bias": NewsArticle.bias_label(score),
                "timestamp": article.timestamp.isoformat() if hasattr(article.timestamp, "isoformat") else str(article.timestamp),
            })

        # Continuous position mapping
        blended = round(vader_avg, 4)
        recommended_position = max(-1.0, min(1.0, blended))
        position_confidence = abs(recommended_position)

        result = {
            "pairs": {
                pair: {
                    "vader_sentiment": round(vader_avg, 4),
                    "vader_magnitude": round(vader_mag, 4),
                    "blended_sentiment": blended,
                    "article_count": len(filtered),
                    "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                    "recommended_position": round(recommended_position, 4),
                    "position_confidence": round(position_confidence, 4),
                }
            },
            "top_articles": top_articles,
            "backend": "vader",
            "model": "vader",
        }

        try:
            from pipeline.llm.sentiment import LLMSentimentEngine
            from config import PIPELINE_CONSTANTS
            llm_config = {
                "llm_sentiment_enabled": True,
                "llm_backend": PIPELINE_CONSTANTS.get("llm_backend", "ollama"),
                "llm_model": PIPELINE_CONSTANTS.get("llm_model", "llama3"),
                "llm_api_key": PIPELINE_CONSTANTS.get("llm_api_key", ""),
                "llm_weight": PIPELINE_CONSTANTS.get("llm_weight", 0.7),
                "llm_ollama_url": PIPELINE_CONSTANTS.get("llm_ollama_url", "http://localhost:11434"),
            }
            engine = LLMSentimentEngine(config=llm_config)
            llm_articles = filtered[:10] if len(filtered) >= 10 else filtered
            scored_llm = engine.score_articles(llm_articles, pair=pair)
            live = engine.get_live_sentiment(pair, llm_articles)
            engine.close()

            llm_w = llm_config["llm_weight"]
            llm_dir = live.get("direction", 0.0)
            blended_llm = llm_w * llm_dir + (1 - llm_w) * vader_avg
            blended_llm = round(blended_llm, 4)
            recommended_position_llm = max(-1.0, min(1.0, blended_llm))

            result["pairs"][pair].update({
                "llm_sentiment": round(llm_dir, 4),
                "llm_confidence": round(live.get("confidence", 0.0), 4),
                "llm_volatility": round(live.get("volatility", 0.3), 4),
                "blended_sentiment": blended_llm,
                "llm_weight": llm_w,
                "currencies_affected": live.get("currencies_affected", []),
                "recommended_position": round(recommended_position_llm, 4),
                "position_confidence": round(abs(recommended_position_llm), 4),
            })
            result["backend"] = llm_config["llm_backend"]
            result["model"] = llm_config["llm_model"]
        except Exception:
            result["pairs"][pair]["llm_sentiment"] = None

        return result
    except Exception as e:
        return {
            "pairs": {},
            "error": str(e)[:200],
            "backend": "unavailable",
            "model": "none",
        }