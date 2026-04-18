"""News & Sentiment package for the FX ML Backtester.

Provides:
- NewsScraper: RSS + NewsAPI + economic calendar fetching with disk cache
- SentimentAnalyzer: VADER (default) and finBERT (opt-in) scoring
- merge_news_features: Walk-forward-safe feature engineering from news data
"""

from news.scraper import NewsScraper, NewsArticle, ECONOMIC_EVENTS
from news.sentiment import SentimentAnalyzer
from news.features import merge_news_features

__all__ = [
    "NewsScraper",
    "NewsArticle",
    "ECONOMIC_EVENTS",
    "SentimentAnalyzer",
    "merge_news_features",
]
