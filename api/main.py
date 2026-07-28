"""KodaQuant — FastAPI application."""
from contextlib import asynccontextmanager
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.config import settings
from api.middleware import install_security_middleware
from api.routers import backtest, committee, config, data, health, hardware, license, live, live_metrics, models, news, pairs, prices, trading, ws

IS_DESKTOP = os.environ.get("FX_APP_MODE", "") == "desktop"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.dependencies import get_data_store
    get_data_store()
    from api.config import settings
    from api.shutdown import startup_cleanup, wal_checkpoint

    try:
        from api.hardware import discover_gpu_vram
        vram = discover_gpu_vram()
        if vram > 0:
            settings.gpu_total_vram_mb = vram
            print(f"[Hardware] GPU VRAM detected: {vram} MB")
            settings.gpu_enabled = True
        else:
            print("[Hardware] No GPU detected — VRAM gate disabled")
    except Exception:
        pass

    from api.config import load_persisted_execution_settings
    load_persisted_execution_settings()

    try:
        from api.process_manager import get_process_manager
        pm = get_process_manager()
        if not getattr(pm, "_initialized", False):
            pm.initialize()
    except Exception as exc:
        print(f"[ProcessManager] Init failed: {exc}")

    start = startup_cleanup(settings.db_full_path)
    if start:
        print(f"[Shutdown] Startup: reset {start} stale running job(s) to pending")
    try:
        store = get_data_store()
        store.mark_orphaned_sessions()
        db_path = store.db_path
        db_exists = "exists" if Path(db_path).exists() else "MISSING"
        print(f"[DB] Path: {db_path} ({db_exists})")
    except Exception:
        pass
    try:
        from pipeline.models.model_registry_disk import scan_and_repair
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

    _candle_syncer = None
    try:
        from pipeline.data.candle_syncer import CandleSyncer
        all_pairs = [p["symbol"] for p in store.list_pairs()]
        pairs = all_pairs or settings.sync_pairs
        if pairs:
            _candle_syncer = CandleSyncer(store, active_pairs=pairs)
            await _candle_syncer.start()
            short = ", ".join(pairs[:5])
            tail = "..." if len(pairs) > 5 else ""
            print(f"[CandleSyncer] Syncing {len(pairs)} pairs: {short}{tail}")
    except Exception:
        print("[CandleSyncer] Failed to start — sync disabled")

    # ── Startup gap-fill: sync missing candles BEFORE live streams ──
    if _candle_syncer is not None:
        print("[StartupSync] Filling candle gaps since last shutdown...")
        _sync_pairs = all_pairs if all_pairs else settings.sync_pairs
        _total = 0
        for _pair in _sync_pairs:
            for _tf in ("M30", "H1", "H4"):
                for _attempt in range(3):
                    try:
                        _n = await _candle_syncer.sync_pair(_pair, _tf)
                        if _n:
                            _total += _n
                        break
                    except Exception as _exc:
                        if _attempt < 2:
                            print(f"[StartupSync] {_pair}/{_tf}: attempt {_attempt + 1}/3 FAILED ({_exc}) — retrying in 5s")
                            await asyncio.sleep(5)
                        else:
                            print(f"[StartupSync] {_pair}/{_tf}: FAILED after 3 attempts ({_exc})")
        if _total:
            print(f"[StartupSync] Gap fill complete — {_total} candles synced")
        else:
            print("[StartupSync] No gaps detected")

    from api.routers.price_stream import PriceStreamManager
    psm = PriceStreamManager.get()
    _stream_pairs = all_pairs if all_pairs else settings.sync_pairs
    for pair in _stream_pairs:
        try:
            await psm.ensure_stream(pair)
        except Exception as e:
            print(f"[PriceStream] Failed to start stream for {pair}: {e}")
    if _stream_pairs:
        print(f"[PriceStream] Started streams for {len(_stream_pairs)} pairs")

    yield

    try:
        await PriceStreamManager.get().shutdown()
    except Exception:
        pass
    if _candle_syncer is not None:
        try:
            await _candle_syncer.stop()
        except Exception:
            pass
    from api.shutdown import shutdown_cleanup
    shutdown_cleanup(settings.db_full_path)
    try:
        from api.process_manager import get_process_manager
        get_process_manager().shutdown()
    except Exception:
        pass
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
app.include_router(hardware.router, prefix="/api/v1")
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
