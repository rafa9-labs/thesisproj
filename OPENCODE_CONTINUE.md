# OpenCode Continuation Prompt

> **Context for**: Starting a new opencode session to continue implementation
> **Date**: 2026-04-19
> **Branch**: `main`
> **Last commit**: `1d5557c` — feat(S8 polish): health check dots, multi-model tabs, PNG export, settings sync, news live data, chunk splitting

---

## What This Project Is

A **Forex ML Backtesting Pipeline** — a commercial-grade walk-forward FX backtesting platform with 8 ML models, Streamlit UI (interim), and one-command model comparison. **End goal**: commercial Electron desktop app sold via Paddle. The repo name "thesisproj" is misleading — this is NOT academic work. Refer to it as "forex pipeline" or "the pipeline". **Never use the word "thesis".**

## What's Done ✅

- **Phase 3**: Pipeline hardening — data leakage audit, feature disk cache (Parquet), Optuna search space simplification, named constants, walk-forward integrity tests (16/16 pass), import chain fixes, schema validation
- **Phase 4.5**: All 8 models verified end-to-end (logistic, xgboost, CNN, LSTM, Transformer, ensemble_cnn_lstm_xgboost, ensemble_adaptive_regime, DQN)
- **GPU detection** + warnings + smoke test launchers
- **Sprint 1**: Model Comparison & Leaderboard — `pipeline/model_comparison.py` + `run_comparison.bat`
- **Sprint 2**: Advanced Execution Models — COMPLETE
  - S2.1: Position sizing (5 methods) — `pipeline/execution/position_sizing.py`
  - S2.2: Stop-loss/take-profit — `pipeline/execution/stops.py`
  - S2.3: Trailing stops (fixed, ATR, Chandelier) — `pipeline/execution/trailing.py`
  - S2.4: Risk management framework — `pipeline/execution/risk_manager.py`
  - S2.5: Integration into execution_patches — **REMAINING**
- **Sprint A**: 9 execution engine bug fixes + 24 regression tests (commit `c4be08a`)
- **Sprint B**: Data/feature robustness — tz crash, empty df guard, hour guard, cache logging + 8 tests (commit `a998756`)
- **Sprint 3**: Multi-Currency Expansion — 6 pairs, data downloader, pair config, pipeline wiring
- **Sprint 4**: RAM optimization — float32 everywhere, clear caches, eliminate redundant copies
- **Sprint 6**: News & Sentiment Features — RSS scraper, VADER/finBERT, event flags, 44 tests
- **Sprint 7**: FastAPI Backend — COMPLETE
  - `api/` package: FastAPI app with CORS, lifespan, health check
  - 5 routers: health, pairs, models, backtest, data, WebSocket
  - Pydantic v2 request/response schemas
  - Celery tasks for backtest execution + data download
  - JobManager with SQLite-backed job tracking
  - WebSocket progress via Redis pub/sub
  - `pipeline/data_sqlite.py`: DataStore with WAL mode, batched inserts
  - `pipeline/data_migrator.py`: CSV→SQLite migration
- **Sprint 8**: React Frontend — IN PROGRESS (M4 scaffold)
  - S8.1: Vite + TS + React 18 + TailwindCSS + shadcn/ui + React Router ✅
  - S8.2: Layout shell (AppShell, TerminalPanel, dark/light mode) ✅
  - S8.3: Dashboard page (KPIs, recent jobs table) ✅
  - S8.4: Backtest config page (model selector, asset selector, features, labels, HPO, execution, progress, run summary) ✅ POLISHED
    - News/sentiment toggles, date pickers, logistic hyperparams, equity/leverage controls, atrSlMult, kellyMinTrades
  - S8.6: Compare page (equity overlay, leaderboard, significance matrix) ✅
  - S8.8: ErrorBoundary + shared components ✅
  - 5 frontend tests (dashboard-kpis, formatters, job-store, schemas, settings-store)
  - **Remaining**: S8.5 (Results & charts), S8.7 (Settings page)
- **Sprint 9**: Electron Desktop Shell — SCAFFOLDED
  - S9.1: Electron scaffold (main.ts, preload.ts, tsconfig) ✅
  - S9.2: Python backend lifecycle (python.ts, health.ts) ✅
  - S9.3: Native menus & tray (menu.ts, tray.ts) ✅
  - **Remaining**: S9.4 (PyInstaller integration), S9.5 (Electron build pipeline)
- **Test suite**: 436 tests all green (441 with frontend tests)

## What's Next

Remaining sprints in order of priority:

| Sprint | Topic | Est | Status |
|--------|-------|-----|--------|
| **S1** | Model Comparison & Leaderboard | 3-4h | ✅ DONE |
| **S2** | Advanced Execution Models | 6-8h | ✅ DONE |
| **S3** | Multi-Currency Expansion | 4-5h | ✅ DONE |
| **S4** | Docker + CI/CD | 3-4h | ✅ DONE (RAM optimization) |
| **S5** | Comprehensive Tests + Benchmarks | 4-6h | ⬜ TODO |
| **S6** | News & Sentiment Features | 6-8h | ⬜ TODO |
| **S7** | FastAPI Backend | 8-10h | ✅ DONE |
| **S8** | React Frontend | 20-25h | 🔄 IN PROGRESS (S8.1-4, 8.6, 8.8 done) |
| **S9** | Electron Desktop Shell | 10-12h | 🔄 SCAFFOLDED (S9.1-3 done) |
| **S10** | Security & Licensing (Paddle) | 12-15h | ⬜ TODO |
| **S11** | Installer & Auto-Update | 6-8h | ⬜ TODO |
| **S12** | Commercial Infrastructure | 8-10h | ⬜ TODO |
| **S13** | Beta & Launch | 6-8h | ⬜ TODO |

**Product target**: Commercial Electron desktop app (React + FastAPI + Python), sold via Paddle.
**Pricing**: Hybrid — one-time purchase + annual updates subscription.

## Key Files to Start With

Read these first to understand the codebase:

1. **`CLAUDE.md`** — Full project context, architecture map, conventions
2. **`ROADMAP.md`** — Complete sprint & phase details (restructured for desktop app)
3. **`api/main.py`** — FastAPI app entry point
4. **`api/routers/`** — API routers (backtest, models, pairs, data, health, ws)
5. **`api/schemas/`** — Pydantic v2 request/response models
6. **`api/services/`** — JobManager with SQLite-backed job tracking
7. **`pipeline/execution/`** — Execution models (position_sizing, stops, trailing, risk_manager)
8. **`config.py`** — Global config with `ExecutionConfig`
9. **`frontend/src/`** — React frontend (pages, components, hooks, stores, api client)
10. **`electron/`** — Electron shell (main, preload, python, tray, menu, health)

## Important Rules

1. **Terminal**: Windows PowerShell 5.1 — NEVER use `&&` (use `;`), NEVER use `cd /d`
2. **Git**: Always push to GitHub before ending a session
3. **Branch**: Always work on feature branches, never main
4. **Naming**: Never use "thesis" — always "forex pipeline" or "the pipeline"
5. **Patterns**: Use `_PC = config.PIPELINE_CONSTANTS` for named constants
6. **Testing**: After implementation, verify with `python -c "from pipeline.execution.X import ..."` etc.

## How to Verify Work

```powershell
# Test imports
python -c "from pipeline.execution.position_sizing import *; print('OK')"
python -c "from api.main import app; print('OK')"

# Run existing tests
.\run_all_tests.bat

# Run smoke test
.\run_smoke.bat

# Run model comparison
.\run_comparison.bat smoke

# Run API tests
pytest tests/test_api.py -v