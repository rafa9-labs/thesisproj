# KodaQuant — Project Status

> **Generated**: 2026-05-28
> **Method**: Verified via code audit against ROADMAP.md claims

---

## Core Engine (Phases 1-3, Sprints 1-3, 5-6, 14) — 100% DONE

| Component | Status | Evidence |
|-----------|--------|----------|
| Walk-forward backtesting engine | Done | `pipeline/backtester/composed.py` |
| 8 model types (logistic, xgboost, svm, rf, cnn, lstm, transformer, dqn) | Done | `models/` |
| 2 ensemble models (adaptive regime, cnn-lstm-xgb) | Done | `models/ensemble_*.py` |
| Feature engineering (50+ TA indicators, news features) | Done | `pipeline/backtester/features_mixin.py` |
| Feature disk cache (Parquet) | Done | `pipeline/feature_cache.py` |
| HPO tuning (Optuna) | Done | `pipeline/tuning/` |
| Position sizing (fixed, kelly, ATR, vol-target) | Done | `pipeline/execution/position_sizing.py` |
| Stop-loss / take-profit / trailing stops | Done | `pipeline/execution/stops.py`, `trailing.py` |
| Risk management (DD breaker, consecutive loss limit) | Done | `pipeline/execution/risk_manager.py` |
| Execution simulation (slippage, spread, costs) | Done | `pipeline/backtester/execution_patches.py` |
| Data leakage audit | Done | `tests/test_walk_forward_integrity.py` (16/16 pass) |
| Model comparison & leaderboard | Done | `pipeline/model_comparison.py` |
| Multi-currency (6 pairs, 3 timeframes) | Done | `pipeline/main_cli.py` |
| Probability calibration | Done | `pipeline/calibration.py` |
| Overfitting detection & diagnostics | Done | `pipeline/overfitting.py` |
| Walk-forward transparency panel | Done | `frontend/src/pages/Results/WalkForwardPanel.tsx` |
| Training diagnostics (feature importance, confusion matrices) | Done | `pipeline/backtester/real_trading_mixin.py` |
| Backtest summary generator | Done | `pipeline/summary_generator.py` |
| Parameter Intelligence (guide, explorer, fANOVA) | Done | `ParameterGuide.tsx`, `ParallelCoordinates.tsx`, etc. |
| Model persistence (snapshots, save/load) | Done | `pipeline/model_persistence.py` |
| Deployed models registry (CRUD, tags, scan) | Done | `pipeline/model_registry_disk.py` |
| MetaEnsemble (majority/soft/weighted voting) | Done | `models/meta_ensemble.py` |
| Live prediction bridge (predict, compare) | Done | `pipeline/model_persistence.py`, `api/routers/models.py` |
| 496+ tests | Done | `tests/` |

---

## Product Shell (Sprints 7-12, 15, 23) — 100% DONE

| Component | Status | Evidence |
|-----------|--------|----------|
| FastAPI backend | Done | `api/main.py` (all routers) |
| Celery task queue | Done | `api/tasks.py` |
| React + Vite + TypeScript frontend | Done | `frontend/src/` |
| Electron desktop shell | Done | `electron/main.ts` |
| Windows installer | Done | `electron-builder.yml` |
| Auto-update (electron-updater) | Done | `electron/updater.ts` |
| Paddle licensing (fingerprint, activation, gates) | Done | `api/licensing/` |
| KodaQuant branding | Done | 37 tests, icons regenerated |
| WebSocket progress streaming | Done | `api/routers/backtest.py`, `frontend/src/api/websocket.ts` |
| Structured HPO progress display | Done | `pipeline/printer.py` |
| Log noise reduction (98%) | Done | `pipeline/backtester/run_mixin.py` |

---

## What's Partially Done — With Remaining Work

### S4: Docker & CI/CD
- [x] Dockerfile.api
- [x] Dockerfile.frontend
- [x] docker-compose.yml
- [x] CI pipeline (.github/workflows/ci.yml — lint + tests)
- [ ] **Remaining**: GPU passthrough in Docker (nvidia-docker), multi-arch builds

### S13: Beta & Launch
- [x] Release pipeline scripts exist (`scripts/publish_release.bat`, `scripts/build_electron.bat`)
- [x] Installer + auto-update working
- [x] App is functionally complete for beta users
- [ ] **Remaining**: Run actual beta test with users, collect feedback, final QA pass, public launch

### S16.9: Forward Test & Live Trading Bridge
- [x] `pipeline/forward_test.py` — full engine (393 lines)
- [x] API endpoint `POST /models/{id}/forward-test`
- [x] Celery task `_run_forward_test_impl`
- [x] `model_id` param in `POST /live/deploy`
- [x] Tests exist (10 tests in `test_forward_test.py`)
- [ ] **Remaining**: Forward Test Tab UI (S16.9-P1.3), LiveTradingPage saved model selector (S16.9-P2.2)

### S18: Live News & Market Data
- [x] News scraper (RSS/NewsAPI, caching) — `news/scraper.py`
- [x] News sentiment (NLP scoring) — `news/sentiment.py`  
- [x] News features for backtesting — `news/features.py`
- [x] Live price endpoint — `api/routers/prices.py` (OANDA bid/ask, OHLC from SQLite)
- [ ] **Remaining**: Real-time RSS polling daemon (S18.1), live news-to-feature pipeline (S18.3), live market dashboard widgets (S18.4 — UI)

### S19: Ensemble Models & Extensibility
- [x] Ensemble CNN-LSTM-XGBoost — `models/ensemble_cnn_lstm_xgboost.py`
- [x] Ensemble Adaptive Regime — `models/ensemble_adaptive_regime.py`
- [ ] **Remaining**: VotingClassifier, Stacking ensemble, custom plugin system (`models/plugins/`), benchmark suite

### S20: LLM / AI Integration
- [x] LLM advisor (backtest analysis) — `pipeline/llm/advisor.py`
- [x] LLM sentiment (news + result analysis) — `pipeline/llm/sentiment.py`
- [x] Prompt templates — `pipeline/llm/prompts.py`
- [ ] **Remaining**: Market commentary generator (S20.1), strategy suggestion engine (S20.2), feature suggest via LLM (S20.3), model generation via LLM (S20.4)

### S21: Live Trading with OANDA
- [x] OANDA data downloader — `pipeline/data_downloader.py`
- [x] Live price API — `api/routers/prices.py`
- [x] Live session deploy + WebSocket signals — `api/routers/live.py` (398 lines, deploy/stop/status/sessions endpoints)
- [x] Signal loop: fetch live OANDA prices -> compute features -> predict -> stream
- [x] model_id support for deploying saved models instead of training fresh
- [ ] **Remaining**: OANDA position/order management client — `trading/oanda_client.py` (S21.1)
- [ ] **Remaining**: Paper trading engine — `trading/paper_engine.py` (S21.2)
- [ ] **Remaining**: Live execution with risk controls — `trading/live_engine.py`, `trading/risk_controls.py` (S21.3)
- [ ] **Remaining**: Trading dashboard UI — (S21.4 — UI)

### S22: Commercial Infrastructure
- [x] Licensing code (Paddle) — `api/licensing/` (from S10, NOT S22)
- [ ] **Remaining**: ALL commercial infra is undone —
  - S22.1: Paddle product & pricing setup in dashboard
  - S22.2: Landing page / marketing website (`website/` — does not exist)
  - S22.3: Documentation site (`docs/` — no MkDocs)
  - S22.4: Legal (ToS, Privacy, Disclaimer, EULA — no `legal/` directory)
  - S22.5: Analytics & monitoring middleware

---

## NOTHING Done — Fully Open

| Sprint | Description | Backend Files Missing |
|--------|-------------|-----------------------|
| S22 | Commercial Infrastructure | `website/`, `docs/`, `legal/`, `api/middleware/analytics.py` |

---

## UI-Only Work (excluded from current backend audit)

| Sprint | Items |
|--------|-------|
| S16.9 (P1.3, P2.2) | Forward Test Tab, LiveTradingPage saved model selector |
| S17 | Cmd+K search, tab navigation redesign, UI consistency, empty states |
| S18.4 | Live market dashboard widgets (ticker strip, sparklines, calendar) |
| S19 (partial) | ModelSelector updates for ensemble, plugin UI |
| S20 (partial) | Commentary panels, strategy suggest UI |
| S21.4 | Trading dashboard (position monitor, trade history, kill switch button) |

---

## Summary: What's Actually Left

| Area | Backend Hours | UI Hours | Priority |
|------|:---:|:---:|:---:|
| **S21: Live Trading (missing pieces)** | 6h | 4h | **HIGH** — last feature before launch |
| S22: Commercial Infrastructure | 8-10h | 0h | MEDIUM — needed for launch |
| S13: Beta & Launch | 2h | 4h | MEDIUM — release gate |
| S18: Live News (missing pieces) | 3h | 3h | LOW — nice to have |
| S19: More Ensemble Models | 4h | 2h | LOW — nice to have |
| S20: LLM Features | 8h | 4h | LOW — nice to have |
| **TOTAL** | **31-33h** | **17h** | **~48-50h to launch** |

---

## Recommended Next Step

**S21.1 — OANDA Trading Client (`trading/oanda_client.py`)**

This is the single biggest gap between "signal generator" and "trading platform":
- Account info, position list, trade list
- Market/limit/stop order submission with SL/TP
- Position management (close partial, modify SL/TP)
- Rate-limiting (120 req/s demo, 20 req/s live)

The live signal pipeline already exists (`api/routers/live.py` signal loop fetches prices from OANDA, runs predictions, streams via WebSocket). The missing piece is **execution** — actually placing and managing trades.
