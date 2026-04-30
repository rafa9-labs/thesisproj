# -*- mode: python ; coding: utf-8 -*-
"""FX ML Backtester — PyInstaller spec for single-directory bundle.

Build:
    pyinstaller forex_pipeline.spec
    
Output:
    dist/fx_backend/          (single-dir bundle)
    dist/fx_backend.exe        (entry point)

The bundle includes:
- Python runtime + all dependencies
- FastAPI, Celery, pandas, numpy, scikit-learn, xgboost, etc.
- Pipeline modules (api/, pipeline/, models/, news/, etc.)
- React frontend as static assets (served by FastAPI in desktop mode)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(SPECPATH).resolve())

frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
if not os.path.isdir(frontend_dist):
    frontend_dist = ""

block_cipher = None

datas = [
    (os.path.join(PROJECT_ROOT, "config.py"), "."),
    (os.path.join(PROJECT_ROOT, "hpo"), "hpo"),
    (os.path.join(PROJECT_ROOT, "csv_data"), "csv_data"),
]

if frontend_dist:
    datas.append((frontend_dist, os.path.join("frontend", "dist")))

hidden_imports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "fastapi.responses",
    "fastapi.staticfiles",
    "celery",
    "celery.app",
    "kombu",
    "redis",
    "sqlalchemy",
    "aiosqlite",
    "httpx",
    "websockets",
    "pydantic",
    "pydantic_settings",
    "pandas",
    "numpy",
    "sklearn",
    "sklearn.linear_model",
    "sklearn.svm",
    "sklearn.ensemble",
    "sklearn.calibration",
    "sklearn.model_selection",
    "xgboost",
    "lightgbm",
    "optuna",
    "feedparser",
    "vaderSentiment",
    "joblib",
    "scipy",
    "scipy.stats",
    "scipy.optimize",
]

excludes = [
    "matplotlib",
    "plotly",
    "tensorboard",
    "torch",
    "tensorflow",
    "keras",
    "tkinter",
    "unittest",
    "pytest",
    "IPython",
    "notebook",
    "jupyter",
    "sphinx",
]

a = Analysis(
    [os.path.join(PROJECT_ROOT, "run_server.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fx_backend",
    debug=False,
    bootloader_ignore_errno=False,
    strip=False,
    upx=True,
    console=True,
    icon=os.path.join(PROJECT_ROOT, "frontend", "public", "favicon.ico") if os.path.exists(os.path.join(PROJECT_ROOT, "frontend", "public", "favicon.ico")) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="fx_backend",
)