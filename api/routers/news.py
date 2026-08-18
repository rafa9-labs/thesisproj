"""News & sentiment status + events endpoints."""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, Response

from api.schemas.news import NewsArticleFull, NewsArticlesResponse, NewsEventItem, NewsEventsResponse
from news.scraper import NewsScraper, NewsArticle, _strip_html, _highlight_sentiment_phrases

logger = logging.getLogger(__name__)

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

    finbert_available = False
    try:
        from transformers import pipeline
        finbert_available = True
    except ImportError:
        pass

    return {
        "sentiment_backend": sentiment_backend,
        "cached_articles": articles,
        "event_types": ["NFP", "FOMC", "CPI", "GDP", "Retail_Sales", "PMI", "ECB_Rate", "BOE_Rate", "BOJ_Rate"],
        "features": {
            "vader_compound": True,
            "event_flags": True,
            "news_volume_windows": [6, 24],
        },
        "finbert_available": finbert_available,
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
                    currency=ev.get("currency", "USD"),
                    impact=impact_val,
                )
            )

    all_events.sort(key=lambda e: e.time)
    return NewsEventsResponse(events=all_events, count=len(all_events))


@router.get("/articles", response_model=NewsArticlesResponse)
def get_news_articles(
    pair: str = Query("", description="Optional currency pair filter (e.g. EURUSD)"),
    days: int = Query(30, description="Max age of articles in days"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Articles per page"),
):
    """Return cached articles with sentiment scores (paginated).

    Articles are sorted by sentiment magnitude (most opinionated first).
    Filter by pair optionally; returns all articles if pair is empty.
    """
    try:
        from news.sentiment import SentimentAnalyzer
        scraper = NewsScraper()
        articles = scraper.fetch_all()

        if not articles:
            return NewsArticlesResponse(articles=[], total=0, pair=pair or "all")

        if pair and pair.strip():
            articles = NewsScraper.filter_by_pair(articles, pair.strip().upper())

        now = datetime.now(timezone.utc)
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if days > 0:
            from datetime import timedelta
            cutoff = now - timedelta(days=days)
        articles = [a for a in articles if a.timestamp.replace(tzinfo=timezone.utc) >= cutoff]

        if not articles:
            return NewsArticlesResponse(articles=[], total=0, pair=pair or "all")

        vader_analyzer = SentimentAnalyzer(backend="vader")
        scored = vader_analyzer.score_articles(articles)

        result = []
        seen_urls: set = set()
        for article, s in sorted(scored, key=lambda x: abs(x[1].score), reverse=True):
            url_key = (article.url or "").strip().rstrip("/")
            if url_key and url_key in seen_urls:
                continue
            if url_key:
                seen_urls.add(url_key)

            score = round(s.score, 4)
            result.append(NewsArticleFull(
                title=article.title[:200],
                body=_strip_html(article.body)[:1200],
                source=article.source,
                url=article.url or "",
                timestamp=article.timestamp.isoformat() if hasattr(article.timestamp, "isoformat") else str(article.timestamp),
                pair_tags=article.pair_tags or [],
                sentiment_score=score,
                summary=article.summary,
                bias=NewsArticle.bias_label(score),
                highlighted_body=article.highlighted_body if hasattr(article, "highlighted_body") else None,
            ))

        total = len(result)
        offset = (page - 1) * page_size
        result = result[offset : offset + page_size]
        return NewsArticlesResponse(articles=result, total=total, pair=pair or "all")
    except Exception:
        logger.exception("Failed to fetch news articles")
        return NewsArticlesResponse(articles=[], total=0, pair=pair or "all")


MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]


def _compute_relevance_tier(article, pair: str) -> int:
    """Return 1 for exact pair match, 2 for single-currency match, 0 for other."""
    tags = [t.upper() for t in article.pair_tags]
    if pair in tags:
        return 1
    base = pair[:3]
    quote = pair[3:]
    if base in tags or quote in tags:
        return 2
    return 0


def _recency_weighted_stats(scored, analyzer):
    """Compute recency-weighted VADER average and magnitude from scored articles.

    Each article's weight = abs(score) * magnitude * recency_weight(timestamp),
    where half-life is 5 days for |score| >= 0.4, 3 days otherwise.

    Returns (weighted_avg_score, weighted_avg_magnitude, article_count).
    """
    weight_sum = 0.0
    score_sum = 0.0
    mag_sum = 0.0
    for article, result in scored:
        half_life = 5.0 if abs(result.score) >= 0.4 else 3.0
        rw = analyzer.recency_weight(article.timestamp, half_life)
        w = abs(result.score) * result.magnitude * rw
        weight_sum += w
        score_sum += result.score * w
        mag_sum += result.magnitude * w
    if weight_sum < 1e-9:
        return 0.0, 0.0, len(scored)
    return score_sum / weight_sum, mag_sum / weight_sum, len(scored)


def _build_pair_data(pair: str, score: float, mag: float, count: int, cache_state: dict):
    pos = max(-1.0, min(1.0, score))
    return {
        "vader_sentiment": round(score, 4),
        "vader_magnitude": round(mag, 4),
        "blended_sentiment": round(score, 4),
        "article_count": count,
        "last_updated": cache_state.get("last_updated_iso"),
        "next_update": cache_state.get("next_update_iso"),
        "cache_age_hours": cache_state.get("cache_age_hours", 0.0),
        "recommended_position": round(pos, 4),
        "position_confidence": round(abs(pos), 4),
    }


def _format_article(
    article,
    score: float,
    tier: int | None = None,
    llm_sentiment: float | None = None,
    llm_confidence: float | None = None,
) -> dict:
    hl_body = article.highlighted_body if hasattr(article, "highlighted_body") else None
    d = {
        "title": article.title[:120],
        "body": _strip_html(article.body)[:1200],
        "source": article.source,
        "url": article.url or "",
        "pair_tags": article.pair_tags or [],
        "sentiment_score": score,
        "summary": article.summary,
        "bias": NewsArticle.bias_label(score),
        "timestamp": article.timestamp.isoformat() if hasattr(article.timestamp, "isoformat") else str(article.timestamp)[:26],
        "highlighted_body": hl_body,
    }
    if tier is not None:
        d["relevance_tier"] = tier
    if llm_sentiment is not None:
        d["llm_sentiment"] = llm_sentiment
    if llm_confidence is not None:
        d["llm_confidence"] = llm_confidence
    return d


@router.get("/sentiment/live")
def get_live_sentiment(
    response: Response,
    pair: str = Query("EURUSD", description="Currency pair for sentiment analysis"),
):
    """Get live sentiment data for all major currency pairs.

    Sentiment is cached per pair for 6 hours. Returns cached scores when
    fresh, re-fetches and re-scores when stale.

    The ``pair`` parameter controls which pair's articles appear in
    ``top_articles``. The ``pairs`` dict always contains all 7 major
    pairs plus an ``OTHER`` entry for articles with no pair tags.

    Article ranking: relevance tier (1=exact pair, 2=partial, 0=other),
    then by composite impact (abs(score) * magnitude * recency_weight)
    within each tier. Limited to 20 articles.
    """
    pair_clean = pair.upper().replace("/", "").replace("-", "")
    cache_max_age = 12.0  # hours — increased from 6h for better UX

    try:
        from news.sentiment import SentimentAnalyzer
        from news.scraper import NewsScraper

        scraper = NewsScraper()
        articles = scraper.fetch_all()
        vader_analyzer = SentimentAnalyzer(backend="vader")

        pairs_data: dict = {}

        # ── Always get articles for the requested pair's top_articles list ──
        selected_filtered = NewsScraper.filter_by_pair(articles, pair_clean) or None

        # ── Per-pair aggregate: recency-weighted VADER ──
        for p in MAJOR_PAIRS:
            cached = SentimentAnalyzer.get_cached_sentiment(p, max_age_hours=cache_max_age)
            cache_state = SentimentAnalyzer.get_sentiment_cache_state(p, max_age_hours=cache_max_age)

            if cached is not None:
                pairs_data[p] = _build_pair_data(
                    p, cached["score"], cached["magnitude"],
                    cached["article_count"], cache_state,
                )
                if p == pair_clean:
                    pairs_data[p]["cache_age_hours"] = cached["cache_age_hours"]
                continue

            filtered = NewsScraper.filter_by_pair(articles, p)
            if not filtered:
                pairs_data[p] = _build_pair_data(p, 0.0, 0.0, 0, cache_state)
                continue

            scored = vader_analyzer.score_articles(filtered)
            avg, mag, cnt = _recency_weighted_stats(scored, SentimentAnalyzer)

            SentimentAnalyzer.cache_sentiment(p, avg_score=avg, article_count=cnt, magnitude=mag)
            fresh_state = SentimentAnalyzer.get_sentiment_cache_state(p, max_age_hours=cache_max_age)
            pairs_data[p] = _build_pair_data(p, avg, mag, cnt, fresh_state)
            if p == pair_clean:
                pairs_data[p]["cache_age_hours"] = 0.0

        # ―― OTHER: articles with zero pair_tags ――
        other_articles = [a for a in articles if not a.pair_tags]
        if other_articles:
            scored_other = vader_analyzer.score_articles(other_articles)
            other_avg, other_mag, other_cnt = _recency_weighted_stats(scored_other, SentimentAnalyzer)
            other_state = SentimentAnalyzer.get_sentiment_cache_state("OTHER", max_age_hours=cache_max_age)
            pairs_data["OTHER"] = _build_pair_data("OTHER", other_avg, other_mag, other_cnt, other_state)
        else:
            pairs_data["OTHER"] = _build_pair_data("OTHER", 0.0, 0.0, 0, {})

        # ―― LLM scoring (requested pair only, expensive) ――
        # Score ONCE and build per-article LLM lookup map for use in top_articles
        llm_available = False
        llm_score_map: dict = {}
        llm_dir = 0.0
        llm_conf = 0.0
        llm_vol = 0.3
        llm_currencies: list = []
        llm_scored_any = False
        llm_backend_name = "vader"
        llm_model_name = "vader"

        if selected_filtered is not None:
            try:
                from pipeline.llm.sentiment import LLMSentimentEngine
                from config import PIPELINE_CONSTANTS
                llm_config = {
                    "llm_sentiment_enabled": PIPELINE_CONSTANTS.get("llm_sentiment_enabled", False),
                    "llm_backend": PIPELINE_CONSTANTS.get("llm_backend", "ollama"),
                    "llm_model": PIPELINE_CONSTANTS.get("llm_model", "llama3"),
                    "llm_api_key": PIPELINE_CONSTANTS.get("llm_api_key", ""),
                    "llm_weight": 0.55,
                    "llm_ollama_url": PIPELINE_CONSTANTS.get("llm_ollama_url", "http://localhost:11434"),
                }
                llm_backend_name = llm_config["llm_backend"]
                llm_model_name = llm_config["llm_model"]
                engine = LLMSentimentEngine(config=llm_config)
                try:
                    llm_articles = selected_filtered[:10] if len(selected_filtered) >= 10 else selected_filtered
                    scored_llm = engine.score_articles(llm_articles, pair=pair_clean)

                    dirs = []
                    confs = []
                    vols = []
                    for article, scores in scored_llm:
                        ahash = article.dedup_hash
                        llm_score_map[ahash] = scores
                        d = scores.get("direction", 0.0)
                        c = scores.get("confidence", 0.0)
                        v = scores.get("volatility", 0.3)
                        dirs.append(d)
                        confs.append(c)
                        vols.append(v)
                        if not scores.get("fallback", False):
                            llm_scored_any = True

                    if dirs:
                        llm_dir = float(sum(dirs) / len(dirs))
                        llm_conf = float(sum(confs) / len(confs))
                        llm_vol = float(sum(vols) / len(vols))
                        llm_currencies = sorted(set(
                            c for _, s in scored_llm
                            for c in s.get("currencies_affected", [])
                        ))
                        llm_available = llm_scored_any
                finally:
                    engine.close()
            except Exception:
                logger.exception("LLM scoring failed")
                pairs_data[pair_clean]["llm_sentiment"] = None

        # ―― Build pair-level LLM stats ――
        if llm_score_map:
            llm_w = 0.55
            vader_avg = pairs_data[pair_clean]["vader_sentiment"]
            vader_contrib = round((1 - llm_w) * vader_avg, 4)
            llm_contrib = round(llm_w * llm_dir, 4)
            blended_llm = round(vader_contrib + llm_contrib, 4)
            recommended_position_llm = max(-1.0, min(1.0, blended_llm))

            pairs_data[pair_clean].update({
                "llm_sentiment": round(llm_dir, 4),
                "llm_confidence": round(llm_conf, 4),
                "llm_volatility": round(llm_vol, 4),
                "blended_sentiment": blended_llm,
                "llm_weight": llm_w,
                "vader_contribution": vader_contrib,
                "llm_contribution": llm_contrib,
                "currencies_affected": llm_currencies,
                "recommended_position": round(recommended_position_llm, 4),
                "position_confidence": round(abs(recommended_position_llm), 4),
            })

        # ―― Top articles: tier + impact sorted, limit 20 ――
        # LLM scores are attached per-article from the lookup map
        top_articles = []
        tier_counts = {"exact": 0, "partial": 0, "other": 0}
        if selected_filtered:
            scored_vader = vader_analyzer.score_articles(selected_filtered)
            ranked: list = []
            for article, result in scored_vader:
                tier = _compute_relevance_tier(article, pair_clean)
                if tier == 1:
                    tier_counts["exact"] += 1
                elif tier == 2:
                    tier_counts["partial"] += 1
                else:
                    tier_counts["other"] += 1
                half_life = 5.0 if abs(result.score) >= 0.4 else 3.0
                rw = SentimentAnalyzer.recency_weight(article.timestamp, half_life)
                impact = abs(result.score) * result.magnitude * rw
                llm_scores = llm_score_map.get(article.dedup_hash)
                ranked.append((
                    tier,
                    -impact,
                    _format_article(
                        article,
                        round(result.score, 4),
                        tier,
                        llm_sentiment=round(llm_scores.get("direction", 0.0), 4) if llm_scores else None,
                        llm_confidence=round(llm_scores.get("confidence", 0.0), 4) if llm_scores else None,
                    ),
                ))
            ranked.sort(key=lambda x: (x[0], x[1]))
            top_articles = [item[2] for item in ranked[:20]]

        article_count_by_tier = tier_counts if selected_filtered else {"exact": 0, "partial": 0, "other": 0}

        status = "ok" if selected_filtered else "no_articles"
        result = {
            "pairs": pairs_data,
            "top_articles": top_articles,
            "article_count_by_tier": article_count_by_tier,
            "backend": llm_backend_name if llm_score_map else "vader",
            "model": llm_model_name if llm_score_map else "vader",
            "from_cache": False,
            "status": status,
            "llm_available": llm_available,
        }

        response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"
        return result
    except Exception as e:
        logger.exception("Failed to compute live sentiment")
        return {
            "pairs": {},
            "top_articles": [],
            "error": str(e)[:200],
            "backend": "unavailable",
            "model": "none",
            "from_cache": False,
            "status": "error",
        }