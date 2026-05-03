"""FX ML Backtester -- PyInstaller entry point.

This file is the production entry point for the bundled application.
It starts the FastAPI backend, serving both the API and the React frontend.

Usage:
    python run_server.py [--host HOST] [--port PORT] [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _find_frontend_dist(project_root: str) -> str | None:
    candidates = [
        os.path.join(project_root, "_internal", "frontend", "dist"),
        os.path.join(project_root, "frontend", "dist"),
        os.path.join(project_root, "dist"),
        os.path.join(os.path.dirname(project_root), "frontend", "dist"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _setup_paths():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if _is_frozen():
        bundle_dir = sys._MEIPASS
        project_root = os.path.dirname(sys.executable)
        os.environ.setdefault("API_DB_PATH", os.path.join(project_root, "data", "forex.db"))
        os.environ.setdefault("FX_APP_MODE", "desktop")
        sys.path.insert(0, bundle_dir)
    else:
        project_root = str(Path(__file__).resolve().parent)
        os.environ.setdefault("FX_APP_MODE", "dev")
        sys.path.insert(0, project_root)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return project_root


def _serve_static(app, frontend_dist: str | None):
    if not frontend_dist or not os.path.isdir(frontend_dist):
        return

    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/favicon.svg")
    async def favicon():
        p = os.path.join(frontend_dist, "favicon.svg")
        if os.path.exists(p):
            return FileResponse(p, media_type="image/svg+xml")
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        from fastapi.responses import FileResponse as FR
        file_path = os.path.join(frontend_dist, path)
        if os.path.isfile(file_path):
            return FR(file_path)
        return FR(os.path.join(frontend_dist, "index.html"))


def main():
    parser = argparse.ArgumentParser(description="FX ML Backtester Server")
    parser.add_argument("--host", default=os.environ.get("API_HOST", "127.0.0.1"), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8001")), help="Bind port")
    parser.add_argument("--data-dir", default=None, help="Data directory override")
    args = parser.parse_args()

    project_root = _setup_paths()

    if args.data_dir:
        db_path = os.path.join(args.data_dir, "forex.db")
        os.environ["API_DB_PATH"] = db_path
        os.environ["API_CSV_DATA_DIR"] = os.path.join(args.data_dir, "csv_data")

    import uvicorn
    from api.main import app

    frontend_dist = _find_frontend_dist(project_root)

    _serve_static(app, frontend_dist)

    host = args.host
    port = args.port

    print(f"[FX Backtester] Starting server on {host}:{port}")
    print(f"[FX Backtester] Frontend dist: {frontend_dist}")
    print(f"[FX Backtester] Data dir: {os.environ.get('API_DB_PATH', 'default')}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()