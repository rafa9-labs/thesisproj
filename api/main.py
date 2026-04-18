"""FX ML Pipeline — FastAPI application."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import backtest, data, health, models, pairs, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure DB schema exists
    from api.dependencies import get_data_store
    get_data_store()
    yield
    # Shutdown: nothing to clean


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(pairs.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(ws.router, prefix="/api/v1")
