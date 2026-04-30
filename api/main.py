"""FX ML Pipeline — FastAPI application."""
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.config import settings
from api.routers import backtest, config, data, health, models, news, pairs, ws

IS_DESKTOP = os.environ.get("FX_APP_MODE", "") == "desktop"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.dependencies import get_data_store
    get_data_store()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

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
app.include_router(news.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(ws.router, prefix="/api/v1")


if IS_DESKTOP:
    _frontend_dist = os.path.join(settings.project_root, "frontend", "dist")
    if not os.path.isdir(_frontend_dist):
        _frontend_dist = os.path.join(os.path.dirname(settings.project_root), "frontend", "dist")

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
