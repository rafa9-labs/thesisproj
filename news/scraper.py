"""News scraper — RSS feeds, NewsAPI, and economic calendar data.

All articles are normalized to ``NewsArticle`` dataclasses, deduplicated
by title hash, and cached to disk (Parquet) so we never re-fetch on
subsequent runs.

Cache directory: ``news_cache/`` (gitignored)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = PROJECT_ROOT / "news_cache"

ECONOMIC_EVENTS = [
    "NFP",
    "FOMC",
    "CPI",
    "GDP",
    "Retail_Sales",
    "PMI",
    "ECB_Rate",
    "BOE_Rate",
]

RSS_FEEDS = {
    "reuters_fx": "https://www.reuters.com/rssFeed/currenciesNews",
    "reuters_markets": "https://www.reuters.com/rssFeed/marketsNews",
    "forexlive": "https://www.forexlive.com/feed",
    "investing_fx": "https://www.investing.com/rss/news_301.rss",
}


@dataclass
class NewsArticle:
    title: str
    body: str
    timestamp: datetime
    source: str
    url: str = ""
    pair_tags: List[str] = field(default_factory=list)
    _dedup_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self._dedup_hash:
            raw = f"{self.title}:{self.source}"
            self._dedup_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def dedup_hash(self) -> str:
        return self._dedup_hash


class NewsScraper:
    """Fetch and cache news articles from multiple sources.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for Parquet disk cache.
    rate_limit_sec : float
        Minimum seconds between HTTP requests.
    newsapi_key : str or None
        Optional NewsAPI.org API key (from .env ``NEWSAPI_KEY``).
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        rate_limit_sec: float = 1.0,
        newsapi_key: str | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sec = rate_limit_sec
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY", "")
        self._last_request_time = 0.0
        self._seen_hashes: set = set()

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        self._last_request_time = time.time()

    def _add_to_seen(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        unique = []
        for a in articles:
            if a.dedup_hash not in self._seen_hashes:
                self._seen_hashes.add(a.dedup_hash)
                unique.append(a)
        return unique

    # ── RSS ──────────────────────────────────────────────────────────

    def fetch_rss(self, feed_urls: Dict[str, str] | None = None) -> List[NewsArticle]:
        """Fetch articles from RSS feeds.

        Parameters
        ----------
        feed_urls : dict or None
            ``{name: url}`` mapping. Defaults to :data:`RSS_FEEDS`.

        Returns
        -------
        list[NewsArticle]
        """
        feeds = feed_urls or RSS_FEEDS
        articles: List[NewsArticle] = []

        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed — RSS scraping disabled")
            return articles

        for name, url in feeds.items():
            self._rate_limit()
            try:
                parsed = feedparser.parse(url)
                for entry in getattr(parsed, "entries", []):
                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue
                    body = getattr(entry, "summary", getattr(entry, "description", ""))
                    if isinstance(body, list):
                        body = " ".join(str(x) for x in body)
                    body = str(body).strip()

                    ts = self._parse_rss_date(entry)
                    pair_tags = self._extract_pair_tags(title + " " + body)

                    articles.append(NewsArticle(
                        title=title,
                        body=body,
                        timestamp=ts,
                        source=name,
                        url=getattr(entry, "link", ""),
                        pair_tags=pair_tags,
                    ))
            except Exception as exc:
                logger.debug("RSS fetch failed for %s: %s", name, exc)

        return self._add_to_seen(articles)

    # ── NewsAPI ──────────────────────────────────────────────────────

    def fetch_newsapi(
        self,
        query: str = "forex OR EURUSD OR GBPUSD OR USDJPY",
        from_date: str | None = None,
        to_date: str | None = None,
        page_size: int = 100,
    ) -> List[NewsArticle]:
        """Fetch articles from NewsAPI.org.

        Requires ``NEWSAPI_KEY`` in .env or passed explicitly.
        Returns empty list if no key is configured.
        """
        if not self.newsapi_key:
            logger.debug("NewsAPI key not configured — skipping")
            return []

        try:
            import requests
        except ImportError:
            from urllib import request as _urllib_req
            import requests as _req
            requests = _req

        articles: List[NewsArticle] = []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.newsapi_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        self._rate_limit()
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("articles", []):
                title = (item.get("title") or "").strip()
                if not title or title == "[Removed]":
                    continue
                body = (item.get("description") or "") + " " + (item.get("content") or "")
                body = body.strip()
                ts_str = item.get("publishedAt", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    ts = datetime.now(timezone.utc)
                pair_tags = self._extract_pair_tags(title + " " + body)
                articles.append(NewsArticle(
                    title=title,
                    body=body,
                    timestamp=ts,
                    source=item.get("source", {}).get("name", "newsapi"),
                    url=item.get("url", ""),
                    pair_tags=pair_tags,
                ))
        except Exception as exc:
            logger.debug("NewsAPI fetch failed: %s", exc)

        return self._add_to_seen(articles)

    # ── Economic Calendar ────────────────────────────────────────────

    @staticmethod
    def economic_calendar_events(
        year: int,
        events: List[str] | None = None,
    ) -> List[Dict]:
        """Return a hardcoded economic calendar for major FX events.

        This provides known NFP, FOMC, CPI release dates by year.
        For production use, replace with a live calendar API.

        Parameters
        ----------
        year : int
            Year to get events for.
        events : list[str] or None
            Filter to these event names (from :data:`ECONOMIC_EVENTS`).

        Returns
        -------
        list[dict]
            Each dict has keys: ``date``, ``event``, ``impact`` (1-3).
        """
        event_filter = set(events) if events else set(ECONOMIC_EVENTS)

        # NFP (first Friday of each month)
        nfp_dates = []
        for month in range(1, 13):
            try:
                first_day = datetime(year, month, 1)
                friday_count = 0
                day = 1
                while day <= 31:
                    try:
                        d = datetime(year, month, day)
                    except ValueError:
                        break
                    if d.weekday() == 4:
                        friday_count += 1
                        if friday_count == 1:
                            nfp_dates.append(d)
                            break
                    day += 1
            except Exception:
                pass

        # FOMC meetings (approximate — 8 per year, ~every 6 weeks)
        fomc_dates = []
        fomc_months = [1, 3, 5, 6, 7, 9, 10, 12]
        for month in fomc_months:
            try:
                fomc_dates.append(datetime(year, month, 15))
            except ValueError:
                pass

        # CPI (typically mid-month, ~12th-15th)
        cpi_dates = []
        for month in range(1, 13):
            try:
                cpi_dates.append(datetime(year, month, 13))
            except ValueError:
                pass

        result = []
        if "NFP" in event_filter:
            for d in nfp_dates:
                result.append({"date": d, "event": "NFP", "impact": 3})
        if "FOMC" in event_filter:
            for d in fomc_dates:
                result.append({"date": d, "event": "FOMC", "impact": 3})
        if "CPI" in event_filter:
            for d in cpi_dates:
                result.append({"date": d, "event": "CPI", "impact": 2})
        if "GDP" in event_filter:
            for month in [1, 4, 7, 10]:
                result.append({"date": datetime(year, month, 27), "event": "GDP", "impact": 2})
        if "Retail_Sales" in event_filter:
            for month in range(1, 13):
                result.append({"date": datetime(year, month, 14), "event": "Retail_Sales", "impact": 2})
        if "PMI" in event_filter:
            for month in range(1, 13):
                result.append({"date": datetime(year, month, 1), "event": "PMI", "impact": 1})
        if "ECB_Rate" in event_filter:
            for month in [1, 3, 4, 6, 7, 9, 10, 12]:
                result.append({"date": datetime(year, month, 20), "event": "ECB_Rate", "impact": 3})
        if "BOE_Rate" in event_filter:
            for month in [2, 3, 5, 6, 8, 9, 11, 12]:
                result.append({"date": datetime(year, month, 10), "event": "BOE_Rate", "impact": 3})

        result.sort(key=lambda x: x["date"])
        return result

    # ── Disk Cache ───────────────────────────────────────────────────

    def save_cache(self, articles: List[NewsArticle], label: str = "articles"):
        """Save articles to Parquet cache."""
        if not articles:
            return
        rows = []
        for a in articles:
            rows.append({
                "title": a.title,
                "body": a.body,
                "timestamp": a.timestamp.isoformat() if isinstance(a.timestamp, datetime) else str(a.timestamp),
                "source": a.source,
                "url": a.url,
                "pair_tags": json.dumps(a.pair_tags),
                "dedup_hash": a.dedup_hash,
            })
        df = pd.DataFrame(rows)
        path = self.cache_dir / f"{label}.parquet"
        df.to_parquet(path, index=False)
        logger.debug("Cached %d articles to %s", len(df), path)

    def load_cache(self, label: str = "articles") -> List[NewsArticle]:
        """Load articles from Parquet cache."""
        path = self.cache_dir / f"{label}.parquet"
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path)
            articles = []
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        ts = datetime.now(timezone.utc)
                pair_tags = row.get("pair_tags", "[]")
                if isinstance(pair_tags, str):
                    try:
                        pair_tags = json.loads(pair_tags)
                    except (json.JSONDecodeError, TypeError):
                        pair_tags = []
                articles.append(NewsArticle(
                    title=str(row["title"]),
                    body=str(row.get("body", "")),
                    timestamp=ts,
                    source=str(row.get("source", "")),
                    url=str(row.get("url", "")),
                    pair_tags=pair_tags if isinstance(pair_tags, list) else [],
                ))
            self._add_to_seen(articles)
            logger.debug("Loaded %d cached articles from %s", len(articles), path)
            return articles
        except Exception as exc:
            logger.debug("Cache load failed: %s", exc)
            return []

    def fetch_all(
        self,
        use_rss: bool = True,
        use_newsapi: bool = False,
        from_date: str | None = None,
        to_date: str | None = None,
        cache_label: str = "articles",
    ) -> List[NewsArticle]:
        """Fetch from all enabled sources, merging with cache.

        Parameters
        ----------
        use_rss : bool
            Fetch RSS feeds.
        use_newsapi : bool
            Fetch from NewsAPI.
        from_date, to_date : str or None
            Date range for NewsAPI.
        cache_label : str
            Cache file name (without extension).

        Returns
        -------
        list[NewsArticle]
            Deduplicated, chronologically sorted.
        """
        cached = self.load_cache(cache_label)
        new_articles: List[NewsArticle] = []

        if use_rss:
            new_articles.extend(self.fetch_rss())

        if use_newsapi:
            api_articles = self.fetch_newsapi(from_date=from_date, to_date=to_date)
            new_articles.extend(api_articles)

        all_articles = cached + new_articles
        seen: set = set()
        unique: List[NewsArticle] = []
        for a in all_articles:
            if a.dedup_hash not in seen:
                seen.add(a.dedup_hash)
                unique.append(a)

        unique.sort(key=lambda a: a.timestamp if isinstance(a.timestamp, datetime) else datetime.min)

        if new_articles:
            self.save_cache(unique, cache_label)

        return unique

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_rss_date(entry) -> datetime:
        for attr in ("published_parsed", "updated_parsed"):
            val = getattr(entry, attr, None)
            if val:
                try:
                    import time as _time
                    return datetime(*val[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
        for attr in ("published", "updated"):
            val = getattr(entry, attr, "")
            if val:
                try:
                    from email.utils import parsedate_to_datetime
                    return parsedate_to_datetime(val)
                except Exception:
                    pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _extract_pair_tags(text: str) -> List[str]:
        pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "NZDUSD", "USDCHF", "XAUUSD",
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
            "NZD/USD", "USD/CHF", "XAU/USD",
        ]
        text_upper = text.upper()
        found = []
        for p in pairs:
            if p in text_upper:
                tag = p.replace("/", "")
                if tag not in found:
                    found.append(tag)
        return found
