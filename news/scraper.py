"""News scraper -- RSS feeds, NewsAPI, and economic calendar data.

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
    "BOJ_Rate",
]

RSS_FEEDS = {
    "forexlive": "https://www.forexlive.com/feed",
    "investing_fx": "https://www.investing.com/rss/news_301.rss",
}


import re as _re

_HTML_TAG = _re.compile(r"<[^>]*>")
_MULTI_SPACE = _re.compile(r"\s+")

def _strip_html(text: str) -> str:
    return _MULTI_SPACE.sub(" ", _HTML_TAG.sub(" ", text or "")).strip()


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

    @property
    def summary(self) -> str:
        text = _strip_html(self.body)
        if len(text) <= 180:
            return text
        return text[:177].rsplit(" ", 1)[0] + "..."

    @property
    def highlighted_body(self) -> str:
        return _highlight_sentiment_phrases(_strip_html(self.body))

    @staticmethod
    def bias_label(score: float) -> str:
        if score > 0.05:
            return "long"
        if score < -0.05:
            return "short"
        return "neutral"


# ── Sentiment keyword highlighting for frontend display ──────────────

_BULLISH_PHRASES = [
    r"\b(bullish?|surge[ds]?|rall(?:y|ies|ied)|soar[esd]?|jump[esd]?|climb[esd]?|"
    r"boost[esd]?|rebound[esd]?|outperform[esd]?|beat[es]?\s+(?:forecast|estimate|expectation)|"
    r"upward|upbeat|rosy|optimis[mt]|dovish|doves?|"
    r"easing?|stimulus|accommodati(?:ve|on)|rate\s+cut|cut\s+rates?|"
    r"strong(?:er|ly)?\s+(?:growth|data|demand|recovery|jobs|employment|GDP)|"
    r"improve[ds]?\s+(?:outlook|sentiment|confidence)|"
    r"expansion|recovery|growing?|growth|positive|"
    r"resilien[ct]|stable|stability)\b",
]

_BEARISH_PHRASES = [
    r"\b(bearish?|plummet[esd]?|plung[esd]?|tumble[esd]?|slump[esd]?|sink[esd]?|"
    r"drop[peds]?|decline[ds]?|fall[es]?|fading?|weaken[esd]?|"
    r"underperform[esd]?|miss[es]?\s+(?:forecast|estimate|expectation)|"
    r"downward|gloomy|pessimis[mt]|hawkish|hawks?|"
    r"tightening|rate\s+hike|hik(?:e|ed|ing)\s+rates?|"
    r"weak(?:er)?\s+(?:growth|data|demand|recovery|jobs|employment|GDP)|"
    r"deteriorat(?:e|ing|ion)|worse|worries?|concern|fears?|"
    r"recession|contraction|slowdown|stagflation|crisis|turmoil|"
    r"risk-off|risk\s+off|sell-off|selloff|"
    r"uncertain|volatile|volatility|tension[es]?|conflict|"
    r"inflation(?:ary)?\s+(?:fears?|concern|pressure|risk|surge|spike|rise))\b",
]

_bullish_re = _re.compile("|".join(_BULLISH_PHRASES), _re.IGNORECASE)
_bearish_re = _re.compile("|".join(_BEARISH_PHRASES), _re.IGNORECASE)


def _highlight_sentiment_phrases(text: str) -> str:
    if not text:
        return ""

    spans = []
    for m in _bearish_re.finditer(text):
        spans.append((m.start(), m.end(), "bearish"))
    for m in _bullish_re.finditer(text):
        spans.append((m.start(), m.end(), "bullish"))
    if not spans:
        return text

    spans.sort(key=lambda x: (x[0], -x[1]))

    merged = []
    for s in spans:
        if merged and s[0] < merged[-1][1]:
            last = merged[-1]
            if last[2] == s[2]:
                merged[-1] = (last[0], max(last[1], s[1]), last[2])
            continue
        merged.append(s)

    result = []
    last_end = 0
    for start, end, sentiment in merged:
        result.append(text[last_end:start])
        cls = "text-emerald-400 bg-emerald-500/10 px-0.5 rounded font-semibold" if sentiment == "bullish" else "text-rose-400 bg-rose-500/10 px-0.5 rounded font-semibold"
        result.append(f'<span class="{cls}">{text[start:end]}</span>')
        last_end = end
    result.append(text[last_end:])
    return "".join(result)


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

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        self._last_request_time = time.time()

    # -- RSS ----------------------------------------------------------

    def fetch_rss(self, feed_urls: Dict[str, str] | None = None, cached_hashes: set | None = None) -> List[NewsArticle]:
        """Fetch articles from RSS feeds.

        Parameters
        ----------
        feed_urls : dict or None
            ``{name: url}`` mapping. Defaults to :data:`RSS_FEEDS`.
        cached_hashes : set or None
            Dedup hashes already in cache -- skip entries that match.

        Returns
        -------
        list[NewsArticle]
        """
        feeds = feed_urls or RSS_FEEDS
        articles: List[NewsArticle] = []
        hashes = cached_hashes or set()

        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed -- RSS scraping disabled")
            return articles

        for name, url in feeds.items():
            self._rate_limit()
            try:
                parsed = feedparser.parse(url)
                for entry in getattr(parsed, "entries", []):
                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue
                    raw = f"{title}:{name}"
                    entry_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
                    if entry_hash in hashes:
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

        return articles

    # -- NewsAPI ------------------------------------------------------

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
            logger.debug("NewsAPI key not configured -- skipping")
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

        return articles

    # -- Economic Calendar --------------------------------------------

    _EVENT_CURRENCY_MAP = {
        "NFP": "USD",
        "FOMC": "USD",
        "CPI": "USD",
        "GDP": "USD",
        "Retail_Sales": "USD",
        "PMI": "USD",
        "ECB_Rate": "EUR",
        "BOE_Rate": "GBP",
        "BOJ_Rate": "JPY",
    }

    @staticmethod
    def economic_calendar_events(
        year: int,
        events: List[str] | None = None,
    ) -> List[Dict]:
        """Return an economic calendar for major FX events.

        Uses hardcoded approximate dates per event type.
        Each dict has keys: ``date``, ``event``, ``impact`` (1-3), ``currency``.

        For production use, call :meth:`fetch_calendar_live` which tries
        online sources before falling back to this method.
        """
        event_filter = set(events) if events else set(ECONOMIC_EVENTS)

        # NFP (first Friday of each month)
        nfp_dates = []
        for month in range(1, 13):
            try:
                day = 1
                friday_count = 0
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

        # FOMC meetings (approximate -- 8 per year)
        fomc_dates = []
        for month in [1, 3, 5, 6, 7, 9, 10, 12]:
            try:
                fomc_dates.append(datetime(year, month, 15))
            except ValueError:
                pass

        # CPI (mid-month)
        cpi_dates = []
        for month in range(1, 13):
            try:
                cpi_dates.append(datetime(year, month, 13))
            except ValueError:
                pass

        result = []
        _cur = NewsScraper._EVENT_CURRENCY_MAP
        if "NFP" in event_filter:
            for d in nfp_dates:
                result.append({"date": d, "event": "NFP", "impact": 3, "currency": _cur.get("NFP", "USD")})
        if "FOMC" in event_filter:
            for d in fomc_dates:
                result.append({"date": d, "event": "FOMC", "impact": 3, "currency": _cur.get("FOMC", "USD")})
        if "CPI" in event_filter:
            for d in cpi_dates:
                result.append({"date": d, "event": "CPI", "impact": 2, "currency": _cur.get("CPI", "USD")})
        if "GDP" in event_filter:
            for month in [1, 4, 7, 10]:
                result.append({"date": datetime(year, month, 27), "event": "GDP", "impact": 2, "currency": _cur.get("GDP", "USD")})
        if "Retail_Sales" in event_filter:
            for month in range(1, 13):
                result.append({"date": datetime(year, month, 14), "event": "Retail_Sales", "impact": 2, "currency": _cur.get("Retail_Sales", "USD")})
        if "PMI" in event_filter:
            for month in range(1, 13):
                result.append({"date": datetime(year, month, 1), "event": "PMI", "impact": 1, "currency": _cur.get("PMI", "USD")})
        if "ECB_Rate" in event_filter:
            for month in [1, 3, 4, 6, 7, 9, 10, 12]:
                result.append({"date": datetime(year, month, 20), "event": "ECB_Rate", "impact": 3, "currency": _cur.get("ECB_Rate", "EUR")})
        if "BOE_Rate" in event_filter:
            for month in [2, 3, 5, 6, 8, 9, 11, 12]:
                result.append({"date": datetime(year, month, 10), "event": "BOE_Rate", "impact": 3, "currency": _cur.get("BOE_Rate", "GBP")})
        if "BOJ_Rate" in event_filter:
            for month in [1, 3, 4, 6, 7, 9, 10, 12]:
                result.append({"date": datetime(year, month, 20), "event": "BOJ_Rate", "impact": 3, "currency": _cur.get("BOJ_Rate", "JPY")})

        result.sort(key=lambda x: x["date"])
        return result

    @staticmethod
    def fetch_calendar_live(
        year: int | None = None,
    ) -> List[Dict]:
        """Try to fetch economic calendar from online sources.

        Falls back to hardcoded calendar if online sources are unavailable.
        Caches results to Parquet for fast subsequent access.

        Parameters
        ----------
        year : int or None
            Year to fetch. Defaults to current year.

        Returns
        -------
        list[dict]
            Each dict: ``date``, ``event``, ``impact``, ``currency``.
        """
        if year is None:
            year = datetime.now(timezone.utc).year

        # Try to load from Parquet cache first
        cached = _load_calendar_cache(year)
        if cached is not None:
            return cached

        # Try online source
        try:
            import requests
            current_year = datetime.now(timezone.utc).year
            if year == current_year:
                resp = requests.get(
                    "https://cdn.jsdelivr.net/gh/fawazahmed0/forex-calendar@main/calendar.json",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                events = _parse_online_calendar(data)
                if events:
                    _save_calendar_cache(year, events)
                    return events
        except Exception:
            logger.debug("Online calendar fetch failed, using hardcoded")

        # Fallback to hardcoded
        fallback = NewsScraper.economic_calendar_events(year)
        _save_calendar_cache(year, fallback)
        return fallback


    # -- Disk Cache ---------------------------------------------------

    def save_cache(self, articles: List[NewsArticle], label: str = "articles"):
        """Save articles to Parquet cache with atomic write and eviction."""
        if not articles:
            return

        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=60)
        articles = [a for a in articles
                    if (isinstance(a.timestamp, datetime) and a.timestamp.replace(tzinfo=timezone.utc) >= cutoff)
                    or (isinstance(a.timestamp, str) and a.timestamp >= cutoff.isoformat())]
        articles.sort(key=lambda a: a.timestamp if isinstance(a.timestamp, datetime) else datetime.min, reverse=True)
        articles = articles[:2000]

        rows = []
        for a in articles:
            rows.append({
                "title": a.title,
                "body": _strip_html(a.body),
                "timestamp": a.timestamp.isoformat() if isinstance(a.timestamp, datetime) else str(a.timestamp),
                "source": a.source,
                "url": a.url,
                "pair_tags": json.dumps(a.pair_tags),
                "dedup_hash": a.dedup_hash,
            })
        df = pd.DataFrame(rows)
        path = self.cache_dir / f"{label}.parquet"
        tmp_path = path.with_suffix(".tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
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
            logger.debug("Loaded %d cached articles from %s", len(articles), path)
            return articles
        except Exception as exc:
            logger.warning("Cache load failed for %s: %s", path, exc)
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
        cached_hashes = {a.dedup_hash for a in cached}

        if use_rss:
            new_articles.extend(self.fetch_rss(cached_hashes=cached_hashes))

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

    # -- Helpers -------------------------------------------------------

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

    _CURRENCY_PATTERNS = [
        ("EURUSD", r"\bEURUSD\b|\bEUR/USD\b"),
        ("GBPUSD", r"\bGBPUSD\b|\bGBP/USD\b"),
        ("USDJPY", r"\bUSDJPY\b|\bUSD/JPY\b"),
        ("AUDUSD", r"\bAUDUSD\b|\bAUD/USD\b"),
        ("USDCAD", r"\bUSDCAD\b|\bUSD/CAD\b"),
        ("NZDUSD", r"\bNZDUSD\b|\bNZD/USD\b"),
        ("USDCHF", r"\bUSDCHF\b|\bUSD/CHF\b"),
        ("XAUUSD", r"\bXAUUSD\b|\bXAU/USD\b"),
    ]

    _CURRENCY_CODE_PATTERNS = [
        ("EUR", r"\bEUR\b"),
        ("USD", r"\bUSD\b"),
        ("GBP", r"\bGBP\b"),
        ("JPY", r"\bJPY\b"),
        ("AUD", r"\bAUD\b"),
        ("CAD", r"\bCAD\b"),
        ("NZD", r"\bNZD\b"),
        ("CHF", r"\bCHF\b"),
    ]

    _CURRENCY_KEYWORDS = {
        "EUR": [r"\beuro(?:zone|pean)?\b", r"\becb\b", r"\bbundesbank\b", r"\blagarde\b",
                r"\bgermany\b", r"\bgerman\b", r"\bfrance\b", r"\bfrench\b", r"\bparis\b", r"\bberlin\b"],
        "USD": [r"\bdollar\b", r"\bfederal reserve\b", r"\bfomc\b", r"\bpowell\b", r"\bwall street\b"],
        "GBP": [r"\bpound\b", r"\bsterling\b", r"\bboe\b", r"\bcable\b", r"\bkingdom\b",
                r"\bbritain\b", r"\bbritish\b", r"\blondon\b", r"\buk\b"],
        "JPY": [r"\byen\b", r"\bboj\b", r"\bkuroda\b", r"\btokyo\b", r"\bjapan\b", r"\bjapanese\b"],
        "AUD": [r"\baussie\b", r"\brba\b", r"\bsydney\b", r"\baustralia\b", r"\baustralian\b"],
        "CAD": [r"\bloonie\b", r"\bboc\b", r"\bottawa\b", r"\bcanada\b", r"\bcanadian\b"],
        "NZD": [r"\bkiwi\b", r"\brbnz\b", r"\bwellington\b", r"\bnew zealand\b"],
        "CHF": [r"\bfranc\b", r"\bsnb\b", r"\bzurich\b", r"\bswiss\b", r"\bswitzerland\b"],
    }

    _pair_regex = _re.compile(
        "|".join(f"(?:{p})" for _, p in _CURRENCY_PATTERNS),
        _re.IGNORECASE,
    )
    _currency_code_regex = _re.compile(
        "|".join(f"(?:{p})" for _, p in _CURRENCY_CODE_PATTERNS),
        _re.IGNORECASE,
    )
    _keyword_regex_map = {
        cc: _re.compile("|".join(f"(?:{kw})" for kw in kws), _re.IGNORECASE)
        for cc, kws in _CURRENCY_KEYWORDS.items()
    }

    @staticmethod
    def _extract_pair_tags(text: str) -> List[str]:
        found: List[str] = []

        # Pass 1: full pair symbols (EURUSD, EUR/USD)
        for match in NewsScraper._pair_regex.finditer(text):
            tag = match.group(0).replace("/", "")
            if tag not in found:
                found.append(tag)

        # Pass 2: individual currency codes (EUR, USD, etc.)
        for match in NewsScraper._currency_code_regex.finditer(text):
            tag = match.group(0).upper()
            if tag not in found:
                found.append(tag)

        # Pass 3: keyword-derived currency tags (euro -> EUR, dollar -> USD, etc.)
        for cc, regex in NewsScraper._keyword_regex_map.items():
            if regex.search(text) and cc not in found:
                found.append(cc)

        return found

    @staticmethod
    def _add_to_seen(articles: List[NewsArticle]) -> List[NewsArticle]:
        seen: set = set()
        unique: List[NewsArticle] = []
        for a in articles:
            if a.dedup_hash not in seen:
                seen.add(a.dedup_hash)
                unique.append(a)
        return unique

    @staticmethod
    def filter_by_pair(articles: List[NewsArticle], pair: str) -> List[NewsArticle]:
        """Return articles relevant to a currency pair.

        Matches articles whose pair_tags include:
        - the full pair symbol (e.g. EURUSD)
        - either individual currency (e.g. EUR or USD)

        Tags are populated by _extract_pair_tags at ingestion time.
        No runtime text scanning or keyword fallback.
        """
        pair = pair.upper().replace("/", "").replace("-", "")
        if not pair or len(pair) < 6:
            return articles
        base = pair[:3]
        quote = pair[3:]

        relevant = []
        for article in articles:
            tags = [t.upper() for t in article.pair_tags]
            if pair in tags or base in tags or quote in tags:
                relevant.append(article)
        return relevant


# ── Calendar Cache Helpers (module-level) ────────────────────────────

def _calendar_cache_path(year: int) -> Path:
    return _CACHE_DIR / f"calendar_{year}.parquet"


def _load_calendar_cache(year: int) -> List[Dict] | None:
    path = _calendar_cache_path(year)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_days = (datetime.now(timezone.utc) - mtime).days
        if age_days > 1:
            return None
        return df.to_dict("records")
    except Exception:
        return None


def _save_calendar_cache(year: int, events: List[Dict]) -> None:
    if not events:
        return
    df = pd.DataFrame(events)
    for col in ["date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    path = _calendar_cache_path(year)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _parse_online_calendar(data: list) -> List[Dict]:
    events = []
    for item in data:
        try:
            date_str = item.get("date", "")
            event_name = item.get("name", item.get("event", "")).strip()
            if not event_name or not date_str:
                continue
            impact_map = {"High": 3, "Medium": 2, "Low": 1}
            impact = impact_map.get(item.get("impact", "medium"), 2)
            currency = item.get("currency", item.get("country", "USD")).upper()
            date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            events.append({
                "date": date,
                "event": event_name[:80],
                "impact": impact,
                "currency": currency[:3],
            })
        except Exception:
            continue
    return events
