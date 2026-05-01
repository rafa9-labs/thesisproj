# -*- mode: python ; coding: utf-8 -*-
"""FX ML Backtester — PyInstaller spec for single-directory bundle.

Build:
    pyinstaller forex_pipeline.spec
    
Output:
    dist/fx_backend/          (single-dir bundle)
    dist/fx_backend.exe       (entry point)

The bundle includes:
- Python runtime + all dependencies
- FastAPI, Celery, pandas, numpy, scikit-learn, xgboost, TensorFlow, etc.
- Pipeline modules (api/, pipeline/, models/, news/, rl/, etc.)
- HPO configs and reference data (csv_data)
- React frontend as static assets (served by FastAPI in desktop mode)

User data (DB, results, cache) stored in %APPDATA%/FX ML Backtester/.
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
    (os.path.join(PROJECT_ROOT, "schemas"), "schemas"),
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
    "sklearn.tree",
    "sklearn.preprocessing",
    "sklearn.pipeline",
    "sklearn.utils",
    "sklearn.metrics",
    "xgboost",
    "optuna",
    "feedparser",
    "vaderSentiment",
    "joblib",
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    "ta",
    "models",
    "models.registry",
    "models.base_model",
    "models.logistic",
    "models.xgboost_model",
    "models.svm",
    "models.random_forest",
    "models.cnn",
    "models.lstm",
    "models.transformer",
    "models.ensemble_adaptive_regime",
    "models.ensemble_cnn_lstm_xgboost",
    "pipeline",
    "pipeline._imports",
    "pipeline.runtime",
    "pipeline.feature_cache",
    "pipeline.metrics",
    "pipeline.metrics_eval",
    "pipeline.metrics_extra",
    "pipeline.metrics_tuples",
    "pipeline.memory_utils",
    "pipeline.io_utils",
    "pipeline.model_utils",
    "pipeline.calibration",
    "pipeline.coverage",
    "pipeline.plotting",
    "pipeline.workers",
    "pipeline.logging_config",
    "pipeline.standalone_utils",
    "pipeline.dqn_config",
    "pipeline.hpo_persistence",
    "pipeline.optuna_utils",
    "pipeline.model_comparison",
    "pipeline.main_cli",
    "pipeline.data_downloader",
    "pipeline.backtester.composed",
    "pipeline.backtester.data_mixin",
    "pipeline.backtester.features_mixin",
    "pipeline.backtester.execution_patches",
    "pipeline.backtester.strategy_mixin",
    "pipeline.backtester.run_mixin",
    "pipeline.backtester.deep_mixin",
    "pipeline.backtester.dqn_mixin",
    "pipeline.backtester.model_factory_mixin",
    "pipeline.backtester.real_trading_mixin",
    "pipeline.backtester.ensemble_mixin",
    "pipeline.execution",
    "pipeline.execution.position_sizing",
    "pipeline.execution.stops",
    "pipeline.execution.trailing",
    "pipeline.execution.risk_manager",
    "pipeline.tuning",
    "pipeline.tuning.runner",
    "pipeline.tuning.sampler",
    "pipeline.tuning.objective",
    "pipeline.tuning.refit",
    "pipeline.tuning.helpers",
    "api",
    "api.main",
    "api.config",
    "api.tasks",
    "api.routers.backtest",
    "api.routers.models",
    "api.routers.pairs",
    "api.routers.news",
    "api.routers.data",
    "api.routers.ws",
    "api.schemas.backtest",
    "api.schemas.news",
    "api.services",
    "news",
    "news.scraper",
    "news.sentiment",
    "news.features",
    "rl",
    "rl.dqn_agent",
    "rl.environment",
    "rl.replay_buffer",
    "rl.wrappers",
]

try:
    import tensorflow
    hidden_imports += [
        "tensorflow",
        "keras",
        "keras.src",
        "keras.src.layers",
        "keras.src.models",
        "keras.src.optimizers",
    ]
except ImportError:
    pass

excludes = [
    "matplotlib",
    "plotly",
    "tensorboard",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
    "jupyter",
    "sphinx",
    "torch",
    "unittest",
]

icon_path = os.path.join(PROJECT_ROOT, "frontend", "public", "favicon.ico")
if not os.path.exists(icon_path):
    icon_path = None

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
    icon=icon_path,
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