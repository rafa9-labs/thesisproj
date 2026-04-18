"""Health check endpoint."""
from fastapi import APIRouter

from api.config import settings
from api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    redis_status = "unknown"
    try:
        import redis as _redis
        r = _redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {e}"

    db_rows = 0
    try:
        from api.dependencies import get_data_store
        store = get_data_store()
        summary = store.get_pair_summary()
        db_rows = sum(s["rows"] for s in summary)
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=settings.version,
        redis=redis_status,
        db_rows=db_rows,
    )
