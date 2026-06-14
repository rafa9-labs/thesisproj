"""KodaQuant — FastAPI application."""
from contextlib import asynccontextmanager
import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.config import settings
from api.middleware import install_security_middleware
from api.routers import backtest, committee, config, data, health, license, live, live_metrics, models, news, pairs, prices, trading, ws

IS_DESKTOP = os.environ.get("FX_APP_MODE", "") == "desktop"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.dependencies import get_data_store
    get_data_store()
    from api.config import settings
    from api.shutdown import startup_cleanup, wal_checkpoint
    start = startup_cleanup(settings.db_full_path)
    if start:
        print(f"[Shutdown] Startup: reset {start} stale running job(s) to pending")
    try:
        from pipeline.model_registry_disk import scan_and_repair
        result = scan_and_repair(settings.db_full_path)
        if any(v for v in result.values()):
            print(f"[Registry] scan_and_repair: registered={result['registered']} cleaned={result['cleaned']} skipped={result['skipped']}")
    except Exception:
        pass

    _stop_wal_timer = None
    async def _periodic_wal_checkpoint():
        while True:
            await asyncio.sleep(60)
            wal_checkpoint(settings.db_full_path)
    try:
        _stop_wal_timer = asyncio.create_task(_periodic_wal_checkpoint())
    except Exception:
        pass

    yield
    from api.shutdown import shutdown_cleanup
    shutdown_cleanup(settings.db_full_path)
    if _stop_wal_timer and not _stop_wal_timer.done():
        _stop_wal_timer.cancel()
    print("[Shutdown] Graceful shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

install_security_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:8001",
        "null",
    ] if not IS_DESKTOP else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(pairs.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(committee.router, prefix="/api/v1")
app.include_router(news.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(license.router, prefix="/api/v1")
app.include_router(ws.router, prefix="/api/v1")
app.include_router(prices.router, prefix="/api/v1")
app.include_router(live.router, prefix="/api/v1")
app.include_router(live_metrics.router, prefix="/api/v1")
app.include_router(trading.router, prefix="/api/v1")
app.include_router(trading.live_router, prefix="/api/v1")


if IS_DESKTOP:
    _frontend_dist = None
    for _candidate in [
        os.path.join(settings.project_root, "_internal", "frontend", "dist"),
        os.path.join(settings.project_root, "frontend", "dist"),
        os.path.join(settings.project_root, "dist"),
        os.path.join(os.path.dirname(settings.project_root), "frontend", "dist"),
    ]:
        if os.path.isdir(_candidate):
            _frontend_dist = _candidate
            break

    if os.path.isdir(_frontend_dist):
        app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="static-assets")

        @app.get("/favicon.svg")
        async def _favicon():
            p = os.path.join(_frontend_dist, "favicon.svg")
            if os.path.exists(p):
                return FileResponse(p, media_type="image/svg+xml")
            return FileResponse(os.path.join(_frontend_dist, "index.html"))

        @app.get("/{path:path}")
        async def _spa_fallback(path: str):
            file_path = os.path.join(_frontend_dist, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(os.path.join(_frontend_dist, "index.html"))
