"""News & sentiment status endpoint."""
import os
from pathlib import Path
from fastapi import APIRouter

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
