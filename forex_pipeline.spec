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

NOTE: PyInstaller spec files run in a special namespace where
Analysis, PYZ, EXE, and COLLECT are injected by the bootloader.
IDE linters flag these as "undefined" — they are correct at runtime.
Similarly, SPECPATH is injected by PyInstaller and is not available
at static-analysis time. Suppress false positives with noqa comments.
"""

import importlib.util
import os
from pathlib import Path

# PyInstaller spec files are executed via `exec(code, spec_namespace)` where
# spec_namespace already contains SPECPATH, Analysis, PYZ, EXE, COLLECT.
# When a linter analyzes this file, those names are undefined.  We conditionally
# define stubs so the file is valid both for linters and for PyInstaller.
from types import SimpleNamespace as _NS

try:
    from PyInstaller.utils.hooks import collect_submodules  # noqa: F401
except ImportError:
    # Linter-only fallback — PyInstaller is always installed when building.
    def collect_submodules(package: str, filter=lambda _: True, on_error="warn once") -> list[str]: return []

if "SPECPATH" not in globals():
    SPECPATH: str = ""

if "Analysis" not in globals():
    def Analysis(*a, **kw) -> _NS:
        return _NS(pure=None, zipped_data=None, scripts=[], binaries=[], zipfiles=[], datas=[])

if "PYZ" not in globals():
    def PYZ(*a, **kw): ...

if "EXE" not in globals():
    def EXE(*a, **kw): ...

if "COLLECT" not in globals():
    def COLLECT(*a, **kw): ...

PROJECT_ROOT = str(Path(SPECPATH).resolve())

frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
if not os.path.isdir(frontend_dist):
    frontend_dist = ""

block_cipher = None
_KEY = "kodaquant-2026-protect"

_xgb_lib_dir: str = ""
_xgb_dir: str = ""
if importlib.util.find_spec("xgboost") is not None:
    import xgboost as _xgb  # noqa: F811 — intentional conditional import
    _xgb_lib_dir = os.path.join(os.path.dirname(_xgb.__file__), "lib")
    _xgb_dir = os.path.dirname(_xgb.__file__)

datas = [
    (os.path.join(PROJECT_ROOT, "config.py"), "."),
    (os.path.join(PROJECT_ROOT, "hpo"), "hpo"),
    (os.path.join(PROJECT_ROOT, "csv_data"), "csv_data"),
    (os.path.join(PROJECT_ROOT, "schemas"), "schemas"),
]
if _xgb_dir:
    datas.append((os.path.join(_xgb_dir, "VERSION"), "xgboost"))

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
    "logging_config",
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
    "api.licensing",
    "api.licensing.storage",
    "api.licensing.fingerprint",
    "api.licensing.paddle_client",
    "api.licensing.gates",
    "api.licensing.manager",
    "api.licensing.middleware",
    "api.middleware",
    "cryptography",
    "cryptography.fernet",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.hashes",
    "cryptography.hazmat.primitives.kdf",
    "cryptography.hazmat.primitives.kdf.hkdf",
]

if importlib.util.find_spec("tensorflow") is not None:
    hidden_imports += [
        "tensorflow",
        "keras",
        "keras.src",
        "keras.src.layers",
        "keras.src.models",
        "keras.src.optimizers",
    ]

hidden_imports += collect_submodules("celery")
hidden_imports += collect_submodules("kombu")

excludes = [
    "tensorboard",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
    "jupyter",
    "sphinx",
    "torch",
    "torchvision",
    "timm",
    "transformers",
    "bitsandbytes",
    "av",
    "cv2",
    "idlelib",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
]

icon_path = os.path.join(PROJECT_ROOT, "build", "icon.ico")
if not os.path.exists(icon_path):
    icon_path = os.path.join(PROJECT_ROOT, "frontend", "public", "favicon.ico")
if not os.path.exists(icon_path):
    icon_path = None

_xgb_binaries = []
if _xgb_lib_dir:
    _xgb_binaries.append((os.path.join(_xgb_lib_dir, "xgboost.dll"), "xgboost/lib"))

a = Analysis(
    [os.path.join(PROJECT_ROOT, "run_server.py")],
    pathex=[PROJECT_ROOT],
    binaries=_xgb_binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    key=_KEY,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher, key=_KEY)

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