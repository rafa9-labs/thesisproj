"""Background news ingestion worker.

Runs on the FastAPI event loop as an asyncio task:
- Every 5 min: fetch RSS, score with VADER for all major pairs, store in memory.
- Every 60 min: score with LLM (if enabled) for any pair with fresh articles.
- Broadcasts ``news_sync`` via WebSocket when new articles are ingested.

All scores are stored in a global in-memory cache so the HTTP
``/news/sentiment/live`` endpoint can respond in < 1 ms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]

SCRAPE_INTERVAL_S = 300
LLM_INTERVAL_S = 3600

# ── Global in-memory response cache ──────────────────────────────────

_sentiment_cache: Dict[str, Any] = {}
"""Keyed by pair (e.g. ``EURUSD``). Each value is the same dict shape
that ``GET /news/sentiment/live`` returns for that pair's ``pairs[p]`` entry."""

_top_articles_by_pair: Dict[str, list] = {}
"""Pre-formatted top_articles list per pair."""

_articles_cache: list = []
"""Full raw NewsArticle list from last fetch (for /articles endpoint)."""

_source_counts: Dict[str, int] = {}
"""Per-source article counts from last scrape cycle."""

_source_diversity: Dict[str, Any] = {}
"""Source diversity metadata for API responses."""

_last_scrape_time: float = 0.0
_last_llm_time: float = 0.0
_scrape_lock = asyncio.Lock()


def get_cached_sentiment(pair: str) -> Optional[dict]:
    pair = pair.upper().replace("/", "").replace("-", "")
    return _sentiment_cache.get(pair)


def get_cached_all_pairs() -> dict:
    return dict(_sentiment_cache)


def get_cached_top_articles(pair: str) -> list:
    pair = pair.upper().replace("/", "").replace("-", "")
    return _top_articles_by_pair.get(pair, [])


def get_cached_articles() -> list:
    return _articles_cache


def get_cache_meta() -> dict:
    return {
        "last_scrape_time": _last_scrape_time,
        "last_llm_time": _last_llm_time,
        "pairs_cached": list(_sentiment_cache.keys()),
        "article_count": len(_articles_cache),
        "source_counts": dict(_source_counts),
        "source_diversity": dict(_source_diversity),
    }


# ── Scoring helpers ──────────────────────────────────────────────────

def _score_articles_vader(articles, vader_analyzer) -> list:
    scored = vader_analyzer.score_articles(articles)
    return scored


def _recency_weighted_stats(scored, analyzer, use_weights=False):
    """Compute recency-weighted VADER average and magnitude from scored articles.

    Each article's weight = abs(score) * magnitude * recency_weight(timestamp),
    where half-life is 5 days for |score| >= 0.4, 3 days otherwise.

    Parameters
    ----------
    scored : list[tuple]
        (article, SentimentResult) or (article, SentimentResult, source_weight).
    analyzer : type
        SentimentAnalyzer class (for recency_weight static method).
    use_weights : bool
        If True, multiply weight by source_weight from 3-tuple.

    Returns
    -------
    tuple[float, float, int]
        (weighted_avg_score, weighted_avg_magnitude, article_count)
    """
    weight_sum = 0.0
    score_sum = 0.0
    mag_sum = 0.0
    for item in scored:
        if use_weights and len(item) == 3:
            article, result, src_weight = item
        else:
            article, result = item[0], item[1]
            src_weight = 1.0
        half_life = 5.0 if abs(result.score) >= 0.4 else 3.0
        rw = analyzer.recency_weight(article.timestamp, half_life)
        w = abs(result.score) * result.magnitude * rw * src_weight
        weight_sum += w
        score_sum += result.score * w
        mag_sum += result.magnitude * w
    if weight_sum < 1e-9:
        return 0.0, 0.0, len(scored)
    return score_sum / weight_sum, mag_sum / weight_sum, len(scored)


def _build_pair_data(pair: str, score: float, mag: float, count: int) -> dict:
    pos = max(-1.0, min(1.0, score))
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "vader_sentiment": round(score, 4),
        "vader_magnitude": round(mag, 4),
        "blended_sentiment": round(score, 4),
        "article_count": count,
        "last_updated": now_iso,
        "next_update": "",
        "cache_age_hours": 0.0,
        "recommended_position": round(pos, 4),
        "position_confidence": round(abs(pos), 4),
    }


def _compute_relevance_tier(article, pair: str) -> int:
    tags = [t.upper() for t in (article.pair_tags or [])]
    if pair in tags:
        return 1
    base = pair[:3]
    quote = pair[3:]
    if base in tags or quote in tags:
        return 2
    return 0


def _format_article(article, score: float, tier: int | None = None,
                    llm_sentiment: float | None = None,
                    llm_confidence: float | None = None) -> dict:
    from news.scraper import _strip_html, NewsArticle
    hl_body = article.highlighted_body if hasattr(article, "highlighted_body") else None
    d = {
        "title": (article.title or "")[:120],
        "body": _strip_html(article.body or "")[:1200],
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


async def _broadcast_news_sync(article_count: int):
    """Send news_sync to all connected WebSocket clients."""
    try:
        from api.routers.ws import broadcast_news_event
        await broadcast_news_event("news_sync", {
            "article_count": article_count,
            "cached": article_count > 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass


# ── Main scrape + score loop ─────────────────────────────────────────

def _run_scrape_vader() -> int:
    """Synchronous: fetch RSS, score all pairs with VADER, update global cache.

    Returns number of articles fetched.
    """
    from news.scraper import NewsScraper
    from news.sentiment import SentimentAnalyzer

    scraper = NewsScraper()
    articles = scraper.fetch_all()

    if not articles:
        return 0

    global _articles_cache, _source_counts, _source_diversity
    _articles_cache = articles

    # Track source distribution
    _source_counts = NewsScraper.get_source_counts(articles)
    total = len(articles)
    _source_diversity = {
        "total_articles": total,
        "active_sources": len(_source_counts),
        "source_pct": {
            src: round(count / total, 4) if total > 0 else 0
            for src, count in _source_counts.items()
        },
        "dominant_source_pct": round(
            max(_source_counts.values()) / total, 4
        ) if total > 0 and _source_counts else 0,
    }

    vader = SentimentAnalyzer(backend="vader")

    # Per-pair VADER scoring with source weights
    for p in MAJOR_PAIRS:
        filtered = NewsScraper.filter_by_pair(articles, p)
        if not filtered:
            _sentiment_cache[p] = _build_pair_data(p, 0.0, 0.0, 0)
            _top_articles_by_pair[p] = []
            continue
        scored = vader.score_articles_weighted(filtered)
        avg, mag, cnt = _recency_weighted_stats(scored, SentimentAnalyzer, use_weights=True)
        _sentiment_cache[p] = _build_pair_data(p, avg, mag, cnt)

        # Build top articles for this pair
        display = NewsScraper.sample_balanced(filtered)
        scored_display = vader.score_articles_weighted(display)
        ranked = []
        for article, result, src_weight in scored_display:
            tier = _compute_relevance_tier(article, p)
            half_life = 5.0 if abs(result.score) >= 0.4 else 3.0
            rw = SentimentAnalyzer.recency_weight(article.timestamp, half_life)
            impact = abs(result.score) * result.magnitude * rw * src_weight
            ranked.append((tier, -impact, _format_article(article, round(result.score, 4), tier)))
        ranked.sort(key=lambda x: (x[0], x[1]))
        _top_articles_by_pair[p] = [item[2] for item in ranked[:20]]

    # OTHER
    other_articles = [a for a in articles if not a.pair_tags]
    if other_articles:
        scored_other = vader.score_articles_weighted(other_articles)
        other_avg, other_mag, other_cnt = _recency_weighted_stats(
            scored_other, SentimentAnalyzer, use_weights=True
        )
        _sentiment_cache["OTHER"] = _build_pair_data("OTHER", other_avg, other_mag, other_cnt)
    else:
        _sentiment_cache["OTHER"] = _build_pair_data("OTHER", 0.0, 0.0, 0)

    return len(articles)


def _run_llm_scoring():
    """Synchronous: score articles for all major pairs with LLM. Updates global cache in-place."""
    from news.scraper import NewsScraper
    from pipeline.llm.sentiment import LLMSentimentEngine
    from config import PIPELINE_CONSTANTS

    if not PIPELINE_CONSTANTS.get("llm_sentiment_enabled", False):
        return

    articles = _articles_cache
    if not articles:
        return

    llm_config = {
        "llm_sentiment_enabled": PIPELINE_CONSTANTS.get("llm_sentiment_enabled", False),
        "llm_backend": PIPELINE_CONSTANTS.get("llm_backend", "ollama"),
        "llm_model": PIPELINE_CONSTANTS.get("llm_model", "llama3"),
        "llm_api_key": PIPELINE_CONSTANTS.get("llm_api_key", ""),
        "llm_weight": 0.55,
        "llm_ollama_url": PIPELINE_CONSTANTS.get("llm_ollama_url", "http://localhost:11434"),
    }

    try:
        engine = LLMSentimentEngine(config=llm_config)
        llm_weight = 0.55

        for pair in MAJOR_PAIRS:
            filtered = NewsScraper.filter_by_pair(articles, pair)
            if not filtered or len(filtered) < 3:
                continue

            try:
                llm_articles = filtered[:10] if len(filtered) >= 10 else filtered
                scored_llm = engine.score_articles(llm_articles, pair=pair)

                dirs, confs, vols = [], [], []
                llm_score_map = {}
                for article, scores in scored_llm:
                    ahash = article.dedup_hash
                    llm_score_map[ahash] = scores
                    dirs.append(scores.get("direction", 0.0))
                    confs.append(scores.get("confidence", 0.0))
                    vols.append(scores.get("volatility", 0.3))

                if dirs:
                    llm_dir = float(sum(dirs) / len(dirs))
                    llm_conf = float(sum(confs) / len(confs))
                    llm_vol = float(sum(vols) / len(vols))

                    vader_avg = _sentiment_cache.get(pair, {}).get("vader_sentiment", 0.0)
                    vader_contrib = round((1 - llm_weight) * vader_avg, 4)
                    llm_contrib = round(llm_weight * llm_dir, 4)
                    blended = round(vader_contrib + llm_contrib, 4)
                    rec_pos = max(-1.0, min(1.0, blended))

                    if pair in _sentiment_cache:
                        _sentiment_cache[pair].update({
                            "llm_sentiment": round(llm_dir, 4),
                            "llm_confidence": round(llm_conf, 4),
                            "llm_volatility": round(llm_vol, 4),
                            "blended_sentiment": blended,
                            "llm_weight": llm_weight,
                            "vader_contribution": vader_contrib,
                            "llm_contribution": llm_contrib,
                            "recommended_position": round(rec_pos, 4),
                            "position_confidence": round(abs(rec_pos), 4),
                        })

                    # Attach LLM scores to top articles
                    top = _top_articles_by_pair.get(pair, [])
                    for art in top:
                        pass  # LLM scores embedded in article format not yet; skip for now
            except Exception:
                logger.debug("LLM scoring failed for pair %s", pair)
        engine.close()
    except Exception:
        logger.exception("LLM scoring engine init failed")


def _run_scrape_cycle() -> int:
    """Run one complete scrape+score cycle. Thread-safe.

    Returns article count.
    """
    global _last_scrape_time
    count = _run_scrape_vader()
    _last_scrape_time = time.time()
    return count


def _run_llm_cycle():
    global _last_llm_time
    _run_llm_scoring()
    _last_llm_time = time.time()


# ── Async task loop ──────────────────────────────────────────────────

async def news_ingestion_loop():
    """Run on the FastAPI event loop. Gracefully shuts down on cancel."""
    logger.info("[NewsWorker] Starting background ingestion loop")
    while True:
        try:
            async with _scrape_lock:
                count = await asyncio.to_thread(_run_scrape_cycle)
                if count > 0:
                    logger.info("[NewsWorker] Scraped %d articles", count)

            if count > 0:
                await _broadcast_news_sync(count)

            # LLM scoring on slower cadence
            now = time.time()
            if now - _last_llm_time >= LLM_INTERVAL_S:
                async with _scrape_lock:
                    await asyncio.to_thread(_run_llm_cycle)
                logger.info("[NewsWorker] LLM scoring cycle complete")

        except asyncio.CancelledError:
            logger.info("[NewsWorker] Cancelled, shutting down")
            return
        except Exception:
            logger.exception("[NewsWorker] Scrape cycle failed")

        await asyncio.sleep(SCRAPE_INTERVAL_S)


async def run_initial_scrape():
    """Run one scrape immediately at startup. Non-fatal."""
    try:
        async with _scrape_lock:
            count = await asyncio.to_thread(_run_scrape_cycle)
            logger.info("[NewsWorker] Initial scrape: %d articles", count)
    except Exception:
        logger.exception("[NewsWorker] Initial scrape failed")
