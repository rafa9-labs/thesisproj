# KodaQuant — Product Roadmap

> **Last Updated**: 2026-05-14
> **Branch**: `feature/phase2-api-bridge`
> **Revenue Target**: £500–2K/month within 3 months of launch
> **Philosophy**: Optimize → Feature Parity → Polish → Secure → Deploy → Scale → Enrich → Automate

---

## Phase 3: Pipeline Hardening & Stability ✅ COMPLETE

> **Goal**: Bullet-proof the backtesting engine. No data leaks, no crashes, reproducible results.

- [x] **3.1** Data leakage audit ✅ DONE (2026-04-12)
  - Verify no future data bleeds in feature computation (all indicators look backward only) ✅ PASS
  - Verify labeling uses only past/current bar data ✅ PASS
  - Verify walk-forward split is strictly chronological (train ≤ test) ✅ PASS
  - Verify 1-bar execution delay enforced in every execution path ✅ PASS
  - **Files**: `pipeline/backtester/composed.py`, `pipeline/backtester/execution_patches.py`, `FINDINGS_data_leakage_audit.md`
  - **Est**: 4h

- [x] **3.2** Feature disk cache (Parquet) ✅ DONE (2026-04-14)
  - Cache computed features per month to avoid recomputation on reruns
  - Cache key = SHA256(data file + size + mtime + canonical feature config) → 16-char hex
  - Parquet + JSON sidecar in `.feature_cache/` (gitignored)
  - Load path: `features_mixin.py` lines 172-193 (before TA recomputation)
  - Save path: `features_mixin.py` lines 720-733 (after feature engineering)
  - 3-layer architecture: in-memory slice cache → disk cache → fresh computation
  - Saves 30-120s per model rerun on 100K+ row DataFrames
  - **Files**: `pipeline/feature_cache.py`, `pipeline/backtester/features_mixin.py`
  - **Est**: 3h

- [x] **3.3** Simplify Optuna search space ✅ DONE (2026-04-13)
  - Reduced all model HP ranges to industry-standard literature values
  - Added centralized `SEARCH_SPACE` dict in `config.py` with references
  - Logistic: C [1e-2,1e2], XGBoost: 5 dims, SVM: 2 dims, RF: 4 dims, LSTM/CNN/Transformer: 4 dims each
  - **Files**: `pipeline/tuning/sampler.py`, `config.py`
  - **Est**: 2h

- [x] **3.4** Replace magic numbers with named constants ✅ DONE (2026-04-13)
  - Added `PIPELINE_CONSTANTS` dict (26 named constants) in `config.py`
  - Replaced 51+ inconsistent hardcoded `.get()` fallbacks across 7 mixin files
  - Key constants: `vol_window_bars`, `high_vol_q`, `slip_norm_bps`, `gamma_slip_norm`, etc.
  - Before: `vol_window_bars` had 3 different defaults (48, 96, mixed) across files
  - After: single source of truth via `_PC["vol_window_bars"]` everywhere
  - **Files**: `config.py`, `pipeline/tuning/sampler.py`, `pipeline/backtester/{strategy,ensemble,dqn,deep,real_trading,run}_mixin.py`
  - **Est**: 2h

- [x] **3.5** Walk-forward integrity tests ✅ DONE (2026-04-12)
  - 16/16 tests pass: train/test overlap, execution delay, chronological ordering
  - **Files**: `tests/test_walk_forward_integrity.py`
  - **Est**: 3h

- [x] **3.6** Import chain fixes ✅ DONE (2026-04-15)
  - Fixed `save_monthly_model_stats` NameError in `real_trading_mixin.py` (was missing from `pipeline/_imports.py`)
  - Added `save_monthly_model_stats` to the `utilsNoWFO` import block in `pipeline/_imports.py`
  - Verified import resolves correctly at runtime
  - **Files**: `pipeline/_imports.py`
  - **Est**: 0.5h

- [x] **3.7** Schema validation module ✅ DONE (2026-04-15)
  - Created `schemas/` package with typed Pydantic-like validators for config, features, HPO, backtest settings
  - `schemas/backtest.py`, `schemas/features.py`, `schemas/hpo.py`, `schemas/settings.py`
  - Tests in `tests/test_schemas.py`
  - **Files**: `schemas/*.py`
  - **Est**: 2h

**Phase 3 complete when**: All data leakage tests pass, features cache to disk, Optuna runs complete without errors.

---

## Sprint 1: Model Comparison & Leaderboard ✅ DONE (2026-04-15)

> **Goal**: One-command multi-model comparison with ranked leaderboard and significance testing.

- [x] **S1.1** Model comparison module ✅ DONE
  - `pipeline/model_comparison.py` — post-process pipeline results into clean leaderboard
  - Scans latest `results/` directory, loads ranking CSVs, equity curves
  - ASCII leaderboard table with friendly column labels
  - Paired t-test significance testing between model pairs (monthly returns)
  - Export full comparison report (leaderboard CSV/JSON + equity curves + significance)
  - CLI: `python -m pipeline.model_comparison` (auto-finds latest results)
  - **Files**: `pipeline/model_comparison.py`
  - **Est**: 2h

- [x] **S1.2** Comparison launcher ✅ DONE
  - `run_comparison.bat` — one-command multi-model runner
  - Modes: `smoke` (all 8 models, 1 trial), `full` (all trials, 3 months), `quick` (logistic+xgboost), `analyze` (existing results only), `gpu` (WSL)
  - Automatically runs leaderboard + significance after backtests
  - **Files**: `run_comparison.bat`
  - **Est**: 0.5h

---

## Sprint 2: Advanced Execution Models ✅ COMPLETE

> **Goal**: Professional-grade trading simulation with position sizing, trailing stops, risk management.
> **Maps to**: Phase 4.2 (indicators) + Phase 4.3 (execution models)
> **Est**: 6-8h

- [x] **S2.1** Position sizing models ✅ DONE (commit `6f00605`)
  - Fixed fractional sizing (% of equity per trade)
  - Kelly criterion position sizing
  - Fixed lot sizing (current behavior, as baseline)
  - Volatility-adjusted sizing (ATR-based)
  - Configurable via `config.py` and UI controls
  - **Files**: `pipeline/execution/position_sizing.py`, `config.py`

- [x] **S2.2** Stop-loss / take-profit management ✅ DONE (commit `6e15059`)
  - Fixed SL/TP in pips
  - ATR-based dynamic SL/TP
  - Breakeven stop management
  - Partial close (scale-out) at TP levels
  - **Files**: `pipeline/execution/stops.py`

- [x] **S2.3** Trailing stop implementation ✅ DONE (commit `1d8768d`)
  - Standard trailing stop (fixed pips)
  - ATR trailing stop
  - Chandelier exit
  - Configurable activation threshold
  - **Files**: `pipeline/execution/trailing.py`

- [x] **S2.4** Risk management framework ✅ DONE (commit `3fdf551`)
  - Max drawdown circuit breaker (pause trading when DD > threshold)
  - Max consecutive losses limit
  - Daily loss limit
  - Correlation-aware position limits
  - **Files**: `pipeline/execution/risk_manager.py`

- [x] **S2.5** Execution model integration ✅ DONE (2026-04-20, branch `feature/phase2-api-bridge`)
  - Wired execution models into API task pipeline
  - Execution config passed through `config_overrides` from React frontend
  - Metrics breakdown: gross vs net, all KPI metrics extracted
  - **Files**: `api/tasks.py`, `frontend/src/stores/useBacktestStore.ts`
  - **Est**: 1h

---

## Sprint 3: Multi-Currency Expansion ✅ COMPLETE

> **Goal**: Support 5+ currency pairs across 3 timeframes (15 data configs).
> **Maps to**: New capability
> **Est**: 4-5h

- [x] **S3.1** Data download automation ✅ DONE (commit `8371310`)
  - OANDA API downloader for multiple pairs
  - Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD (+ XAUUSD gold)
  - Timeframes: M30, H1, H4
  - Rate-limited, resumable downloads
  - **Files**: `pipeline/data_downloader.py`

- [x] **S3.2** Multi-pair backtest runner ✅ DONE (commit `8371310`)
  - Extend `pipeline/main_cli.py` to iterate over pairs
  - Cross-pair leaderboard (normalized metrics for fair comparison)
  - Pair-specific HPO configs
  - **Files**: `pipeline/main_cli.py`, `pipeline/model_comparison.py`

- [x] **S3.3** Cross-pair comparison UI ✅ DONE
  - Pair selector dropdown in UI
  - Cross-pair equity curve overlay
  - Heatmap: model × pair Sharpe ratio grid
  - **Files**: `frontend/src/pages/Compare/CrossPairSection.tsx`, `frontend/src/components/charts/CrossPairOverlayChart.tsx`, `frontend/src/components/charts/SharpeHeatmap.tsx`, `frontend/src/pages/Dashboard/PerformanceHeatmapSection.tsx`, `api/routers/backtest.py` (heatmap + cross-pair-curves endpoints)
  - **Est**: 1h

---

## Sprint 4: Docker + CI/CD ✅ PARTIAL (RAM optimization done)

> **Goal**: One-click deploy, automated testing pipeline.
> **Maps to**: Phase 7.1 + 7.2
> **Est**: 3-4h

- [ ] **S4.1** Dockerfile + docker-compose
  - Multi-stage build (slim image)
  - GPU passthrough config for deep models
  - Streamlit on port 8501
  - Volume mounts for data and results
  - **Files**: new `Dockerfile`, `docker-compose.yml`
  - **Est**: 2h

- [ ] **S4.2** CI/CD pipeline (GitHub Actions)
  - Lint (ruff/flake8) on every PR
  - Unit tests on every push
  - Integration tests on merge to main
  - Auto-deploy to Streamlit Cloud on main merge
  - **Files**: new `.github/workflows/ci.yml`
  - **Est**: 1.5h

- [x] **S4.3** RAM optimization ✅ DONE (commit `0ffd725`)
  - float32 everywhere, clear caches, eliminate redundant copies
  - **Files**: `pipeline/backtester/*.py`

---

## Sprint 5: Comprehensive Tests + Benchmarks ✅ COMPLETE (2026-05-01)

> **Goal**: Bulletproof confidence in results. 496+ tests, model validation, benchmarks, golden regression.
> **Maps to**: Phase 11
> **Est**: 4-6h

- [x] **S5.1** AI-generated unit tests for pipeline/ ✅ DONE
  - 496+ tests pass (excl. slow/api), up from 396
  - Build validation, module importability, data file checks
  - **Files**: `tests/test_build_validation.py` (46 tests)

- [x] **S5.2** AI-generated unit tests for models/ ✅ DONE
  - All 10 registered models + ensemble_cnn_lstm_xgboost: build, fit, predict, shape, proba
  - Classical (5), Deep (3 Keras), DQN, Ensemble (2)
  - Edge cases: single sample, hyperparams, invalid input_shape
  - Fixed circular import in `pipeline/dqn_config.py` + `models/registry.py` (filter_dqn_config)
  - **Files**: `tests/test_models_train_predict.py` (40 tests)

- [x] **S5.3** Performance benchmarks ✅ DONE
  - Classical model build/train/predict time limits (5s threshold)
  - Memory profiling (100MB threshold)
  - Deep model benchmarks (60s threshold, marked @pytest.mark.slow)
  - **Files**: `tests/benchmarks/` (11 tests, 4 slow)

- [x] **S5.4** Golden output regression tests ✅ DONE
  - Deterministic predictions for logistic, SVM, RF, DT, XGBoost
  - Registry key stability test (detects added/removed models)
  - Search space key stability test
  - Auto-creates golden files on first run, verifies on subsequent runs
  - **Files**: `tests/golden/` (7 tests + golden .npz/.json files)

---

## Sprint 6: News & Sentiment Features ✅ COMPLETE

> **Goal**: News-derived features for smarter backtesting. Unique differentiator.
> **Maps to**: Phase 10
> **Est**: 6-8h

- [x] **S6.1** News scraper ✅ DONE
  - RSS feeds (Reuters, Bloomberg, ForexLive, Investing.com) via feedparser
  - NewsAPI integration (optional, key from .env NEWSAPI_KEY)
  - Economic calendar (NFP, FOMC, CPI, GDP, Retail_Sales, PMI, ECB_Rate, BOE_Rate)
  - Rate-limited, cached to Parquet (news_cache/), deduplicated by title hash
  - **Files**: `news/scraper.py`

- [x] **S6.2** Sentiment analysis ✅ DONE
  - VADER (default, zero-dep): fast rule-based scoring, compound [-1, 1]
  - finBERT (opt-in, via HuggingFace transformers): financial-domain BERT
  - Per-article scoring → hourly/daily aggregation
  - **Files**: `news/sentiment.py`

- [x] **S6.3** News-derived features in pipeline ✅ DONE
  - Sentiment score + magnitude as input features (float32)
  - News volume rolling windows (configurable, default 6 and 24 bars)
  - Event flags (NFP, FOMC, CPI proximity markers)
  - Walk-forward safe (forward-fill only, no future data)
  - Controlled by features_config toggles: use_news, news_event_flags, news_sentiment_backend
  - **Files**: `news/features.py`, `pipeline/backtester/features_mixin.py`

- [x] **S6.4** News overlay on charts ✅ DONE
  - Vertical event markers on equity curves (color-coded by impact level)
  - Legend entries for high/medium impact events
  - Toggle button (Events ON/OFF) on EquitySection
  - API: `GET /news/events?start=&end=&impact=high,medium`
  - **Files**: `api/routers/news.py`, `api/schemas/news.py`, `frontend/src/pages/Results/EquitySection.tsx`, `frontend/src/components/charts/EquityCurveChart.tsx`

---

## Sprint 7: FastAPI Backend ✅ COMPLETE

> **Goal**: Decouple pipeline engine from Streamlit into a proper REST API.
> **Est**: 8-10h

- [x] **S7.1** FastAPI project scaffold ✅ DONE (commits `b15a7eb`, `1fd4615`)
  - `api/` package with `main.py`, routers, middleware
  - CORS config for local React frontend
  - Lifecycle management (startup/shutdown events)
  - **Files**: `api/__init__.py`, `api/main.py`

- [x] **S7.2** Core API endpoints ✅ DONE
  - `POST /api/v1/backtest` — run backtest with config payload
  - `GET /api/v1/backtest/{id}/status` — poll progress
  - `GET /api/v1/backtest/{id}/results` — fetch results + metrics
  - `GET /api/v1/models` — list available models + registry
  - `GET /api/v1/config` — get/set pipeline configuration
  - **Files**: `api/routers/backtest.py`, `api/routers/models.py`, `api/routers/pairs.py`

- [x] **S7.3** WebSocket progress streaming ✅ DONE
  - `WS /api/v1/ws/{job_id}` — real-time progress events
  - Events: model training start/complete, epoch progress, fold complete, final metrics
  - **Files**: `api/routers/ws.py`

- [x] **S7.4** Data & results management ✅ DONE
  - `GET /api/v1/pairs` — list available currency pairs + timeframes
  - `GET /api/v1/data/` — data management endpoints
  - SQLite data layer with WAL mode, batched inserts
  - CSV→SQLite migration tool
  - **Files**: `api/routers/data.py`, `api/routers/pairs.py`, `pipeline/data_sqlite.py`, `pipeline/data_migrator.py`

- [x] **S7.5** Background task queue ✅ DONE
  - Celery tasks for long-running backtests
  - JobManager with SQLite-backed job tracking
  - Redis pub/sub for WebSocket progress
  - **Files**: `api/tasks.py`, `api/services/__init__.py`

---

## Sprint 8: React Frontend ✅ COMPLETE

> **Goal**: Professional desktop-grade React UI that replaces Streamlit as the user-facing product.
> **Architecture**: Vite + TypeScript + React 18 + TailwindCSS + shadcn/ui
> **Est**: 20-25h

- [x] **S8.1** React project scaffold ✅ DONE (commit `42a1629`, `8d46875`)
  - Vite + TypeScript + React 18
  - TailwindCSS + shadcn/ui component library
  - React Router for multi-page navigation
  - API client layer (axios + WebSocket hook)
  - **Files**: `frontend/` package
  - **Est**: 2h

- [x] **S8.2** Layout shell & navigation ✅ DONE (commit `8d46875`)
  - Sidebar nav (dashboard, backtest, results, compare, settings)
  - Dark/light mode toggle with system preference detection
  - Responsive layout (works in Electron window 1280x800+)
  - System tray integration stub
  - **Files**: `frontend/src/layout/`, `frontend/src/components/layout/AppShell.tsx`, `TerminalPanel.tsx`
  - **Est**: 2h

- [x] **S8.3** Dashboard page ✅ DONE
  - KPI cards (total backtests, best Sharpe, avg win rate, equity curve thumbnail)
  - Recent backtests table with equity thumbnails
  - Model performance heatmap (model x pair Sharpe grid) with metric selector
  - **Files**: `frontend/src/pages/Dashboard/` (DashboardPage, DashboardKPIs, RecentJobsTable, PerformanceHeatmapSection), `frontend/src/components/charts/SharpeHeatmap.tsx`, `frontend/src/components/charts/EquityThumbnail.tsx`
  - **Est**: 3h

- [x] **S8.4** Backtest configuration page ✅ DONE (commits `8d46875`, `96f4ab1`)
  - Model selector (multi-select for comparison)
  - Currency pair + timeframe dropdowns
  - Date range pickers (start/end)
  - Execution model config (position sizing, stops, risk manager, equity, leverage)
  - Feature toggles panel (core indicators, advanced, news/sentiment)
  - Labels & triple barrier controls
  - HPO settings + logistic hyperparameters (conditional)
  - "Run Backtest" button with progress bar via WebSocket
  - Pre-flight summary modal with validation
  - **Files**: `frontend/src/pages/Backtest/` (9 files)
  - **Est**: 4h

- [x] **S8.5** Results & charts page ✅ DONE (commit `8d46875`)
  - Equity curve (lightweight-charts — interactive, zoomable, crosshair) + buy-hold overlay + drawdown histogram
  - KPI cards grid (8 metrics: Sharpe, Sortino, max DD, total return, win rate, trades, profit factor, avg trade)
  - Monthly returns table + bar chart
  - Trade log table (ag-grid, sortable, custom cell renderers, row click detail panel)
  - HPO diagnostics (param importance + optimization trace)
  - Config viewer (collapsible JSON)
  - Multi-model tab switching (commit `1d5557c`)
  - PNG export for equity chart (commit `1d5557c`)
  - Export CSV + JSON
  - **Files**: `frontend/src/pages/Results/` (7 files), `frontend/src/components/charts/` (2 files)
  - **Est**: 5h

- [x] **S8.6** Model comparison page ✅ DONE (commit `8d46875`)
  - Side-by-side equity curve overlay
  - Leaderboard table (sortable by any metric)
  - Paired t-test significance indicators
  - Parameter sensitivity chart (Recharts ScatterChart with param selector)
  - Cross-pair comparison section (overlay equity curves across pairs)
  - **Files**: `frontend/src/pages/Compare/` (ComparePage, EquityOverlayChart, LeaderboardTable, SignificanceMatrix, CrossPairSection), `frontend/src/components/charts/ParameterSensitivityChart.tsx`, `frontend/src/components/charts/CrossPairOverlayChart.tsx`
  - **Est**: 3h

- [x] **S8.7** Settings page ✅ DONE (commit `8d46875`)

- [x] **S8.8** Error handling & loading states ✅ DONE (commit `8d46875`)
  - Global error boundary with user-friendly messages
  - Loading skeletons for all data-fetching components
  - Toast notifications for success/error/warning
  - Retry logic for failed API calls
  - **Files**: `frontend/src/components/shared/ErrorBoundary.tsx`, `frontend/src/hooks/`
  - **Est**: 2h

---

## Sprint 8B: Frontend ↔ API Integration & Bug Fixes ✅ COMPLETE (2026-04-20)

> **Goal**: Fix all React ↔ FastAPI communication bugs, remove Streamlit, add data validation.
> **Branch**: `feature/phase2-api-bridge`
> **Est**: 6-8h

- [x] **S8B.1** WebSocket port fix ✅ DONE
  - WebSocket was connecting to port 8000, REST API on 8001 — silently failing
  - Fixed: `frontend/src/api/websocket.ts` — default URL port 8000 → 8001
  - **Files**: `frontend/src/api/websocket.ts`

- [x] **S8B.2** React Query cache invalidation on WS events ✅ DONE
  - Job completion events now invalidate `["jobs"]` and `["job-results"]` query keys
  - Results page auto-updates when Celery job finishes
  - **Files**: `frontend/src/stores/useJobStore.ts`

- [x] **S8B.3** REST polling fallback for progress ✅ DONE
  - `BacktestProgress` now uses `useJobStatus` as polling fallback when WebSocket fails
  - Polls every 2s until job completes, then fires completion event
  - **Files**: `frontend/src/pages/Backtest/BacktestProgress.tsx`

- [x] **S8B.4** Auto-navigate to results after completion ✅ DONE
  - When backtest completes, auto-navigates to `/results/:jobId` after 800ms
  - **Files**: `frontend/src/pages/Backtest/BacktestPage.tsx`

- [x] **S8B.5** AppShell memory leak + StatusDot syntax fix ✅ DONE
  - `useState` used as effect (interval never cleaned up) → fixed to `useEffect`
  - Stray `}` in StatusDot color prop causing render crash
  - **Files**: `frontend/src/components/layout/AppShell.tsx`

- [x] **S8B.6** Timestamp + NaN serialization in Celery tasks ✅ DONE
  - `df_sim.to_dict()` produced `pd.Timestamp` objects → `json.dumps` crash
  - Fixed: reset_index, convert datetime cols to ISO strings, replace NaN with None
  - Added `_sanitize_metrics()` helper for numpy type → JSON conversion
  - **Files**: `api/tasks.py`

- [x] **S8B.7** Frontend null guards for metrics ✅ DONE
  - `results.metrics.length` crashed when API returned null metrics
  - Fixed in: ResultsPage, ComparePage, DashboardKPIs, useJobStore, LeaderboardTable, SignificanceMatrix
  - Pattern: `const metrics = results.metrics ?? []` before all accesses
  - **Files**: `frontend/src/pages/Results/ResultsPage.tsx`, `frontend/src/pages/Compare/ComparePage.tsx`, `frontend/src/pages/Compare/LeaderboardTable.tsx`, `frontend/src/pages/Compare/SignificanceMatrix.tsx`, `frontend/src/pages/Dashboard/DashboardKPIs.tsx`, `frontend/src/stores/useJobStore.ts`

- [x] **S8B.8** React hooks violation fix ✅ DONE
  - `useJobStore()` called inside ternary expressions — violated React hooks rules
  - When `activeJobId` changed from null to string, hook count changed between renders
  - Fixed: always call `useJobStore`, selector handles null internally
  - **Files**: `frontend/src/pages/Backtest/BacktestPage.tsx`, `frontend/src/pages/Backtest/BacktestProgress.tsx`

- [x] **S8B.9** Remove all Streamlit code ✅ DONE
  - Deleted: `app.py`, `ui/` directory (8 files + cache), `launch_ui.bat`, `test_ui_imports.py`
  - Streamlit was only a dev tool; React is the product frontend
  - **Files**: removed

- [x] **S8B.10** HPO_CONFIG_DIR bug fixes ✅ DONE
  - Missing import in `real_trading_mixin.py` → SVM crashed with NameError
  - Wrong default path in `dqn_config.py` (`pipeline/hpo/` → `hpo/`)
  - Graceful fallback when no cached HPO config (warns + uses defaults instead of crashing)
  - **Files**: `pipeline/backtester/real_trading_mixin.py`, `pipeline/dqn_config.py`

- [x] **S8B.11** Date range constraints ✅ DONE
  - Date pickers now use `min={dataMin}` / `max={dataMax}` from API (browser-level enforcement)
  - JS-level validation rejects out-of-range dates
  - Auto-clears stored dates when timeframe changes
  - Backend clamps `end` to actual data max date
  - Validation rules: start before data, end after data, start > end, range too short
  - Default: full data range, end clamped to last complete month
  - **Files**: `frontend/src/pages/Backtest/AssetSelector.tsx`, `frontend/src/hooks/useValidation.ts`, `api/tasks.py`, `api/schemas/backtest.py`

- [x] **S8B.12** FeaturesPanel redesign ✅ DONE
  - Vertical stacking with padded sections instead of grid layout
  - Section titles: Core Indicators, Transformations, Lag Features, Advanced Toggles, News & Sentiment
  - **Files**: `frontend/src/pages/Backtest/FeaturesPanel.tsx`

- [x] **S8B.13** MLBacktester None date handling ✅ DONE
  - `data_mixin.py`: explicit None handling (skip `.loc[]` slice, use full range)
  - Auto-fills `self.start`/`self.end` from actual data bounds after loading
  - `real_trading_mixin.py`: guards `pd.to_datetime(self.start)` with fallback to `self.data.index[0]`
  - **Files**: `pipeline/backtester/data_mixin.py`, `pipeline/backtester/real_trading_mixin.py`

---

## Sprint 14: Pipeline Enhancements ✅ COMPLETE (2026-05-01)

> **Goal**: Add flexible walk-forward periods and HPO duration control.
> **Est**: 5-8h

- [x] **S14.1** Daily/Weekly walk-forward periods ✅ DONE
  - Added `PeriodUnit` type + `period_offset()`, `periods_between()`, `convert_month_count_to_periods()`, `to_period_freq()` helpers to `config.py`
  - Added `period_unit` field to `BacktestParams` schema (Literal months/weeks/days)
  - Refactored `evaluation_mixin.py` `get_walk_forward_splits()` — returns 4-tuple, period-unit aware
  - Replaced all 15 `pd.DateOffset(months=...)` call sites across 3 pipeline files
  - Frontend HpoPanel: period unit selector dropdown (months/weeks/days)
  - 13 new `TestPeriodUnit` tests
  - **Files**: `config.py`, `schemas/backtest.py`, `pipeline/backtester/evaluation_mixin.py`, `pipeline/backtester/real_trading_mixin.py`, `pipeline/backtester/run_mixin.py`, `frontend/src/pages/Backtest/HpoPanel.tsx`, `tests/test_walk_forward_integrity.py`

- [x] **S14.2** HPO duration control ✅ DONE
  - Added `max_hpo_duration_minutes` to config and schema
  - Timeout callback in `run_optuna_tuning()` stops study when time budget exceeded
  - Threaded through both call sites in `run_mixin.py`
  - Frontend control in `Backtest/HpoPanel.tsx`
  - **Files**: `schemas/backtest.py`, `pipeline/tuning/runner.py`, `pipeline/backtester/run_mixin.py`, `frontend/src/pages/Backtest/HpoPanel.tsx`

---


> **Goal**: Wrap React + FastAPI into a native-feeling desktop application.
> **Architecture**: Electron main process + Python subprocess + React BrowserWindow
> **Est**: 10-12h

- [x] **S9.1** Electron scaffold ✅ DONE (commit `8d46875`)
  - `electron/` directory with main process
  - BrowserWindow config (min size 1280x800, frameless option)
  - Dev vs production mode detection
  - **Files**: `electron/main.ts`, `electron/preload.ts`, `electron/tsconfig.json`
  - **Est**: 2h

- [x] **S9.2** Python backend lifecycle management ✅ DONE (commit `8d46875`)
  - Spawn FastAPI subprocess from Electron main process
  - Port discovery (find available localhost port)
  - Health check polling until backend is ready
  - Graceful shutdown (SIGTERM → wait → SIGKILL)
  - Log forwarding from Python to Electron console
  - **Files**: `electron/python.ts`, `electron/health.ts`
  - **Est**: 3h

- [x] **S9.3** Native menus & system tray ✅ DONE (commit `8d46875`)
  - Application menu (File, Edit, View, Backtest, Help)
  - System tray icon with status (running, backtesting, error)
  - Tray context menu (show window, run backtest, quit)
  - **Files**: `electron/tray.ts`, `electron/menu.ts`
  - **Est**: 2h

- [x] **S9.4** PyInstaller integration ✅ DONE (2026-05-01)
  - `forex_pipeline.spec` — bundle Python + all dependencies
  - Removed TF/Keras from excludes (deep models need them)
  - Added 50+ hidden imports for all pipeline, models, API, news, RL modules
  - Conditional TF import (only in hidden_imports if TF installed)
  - Added `schemas/` to data files
  - Fixed `icon_path` handling (None fallback when favicon.ico missing)
  - `scripts/build_python.bat` — improved with TF check, size reporting
  - `tests/test_build_validation.py` — 46 tests verifying spec, imports, data, entry points
  - **Files**: `forex_pipeline.spec`, `scripts/build_python.bat`, `tests/test_build_validation.py`

- [x] **S9.5** Electron build pipeline ✅ DONE (2026-05-04)
  - `electron-builder.yml` — full config (appId, NSIS, extraResources, GitHub publish)
  - `scripts/build_electron.bat` — 7-step build pipeline (prereqs → React → TS → PyInstaller → icons → electron-builder)
  - `scripts/setup_code_signing.bat` — self-signed cert creation + signtool docs
  - `scripts/build_python.bat` — PyInstaller build with TF check + size reporting
  - Build verified end-to-end: PyInstaller (1.7GB) → React+TS (2.1MB) → NSIS installer (515MB)
  - 46/46 build validation tests passing
  - Code signing: disabled for dev (`signAndEditExecutable: false`), documented path for production signing
  - **Files**: `electron-builder.yml`, `scripts/build_electron.bat`, `scripts/build_python.bat`, `scripts/setup_code_signing.bat`, `forex_pipeline.spec`
  - **Est**: 2h

---

## Sprint 10: Security & Licensing ✅ COMPLETE (2026-05-04)

> **Goal**: Protect source code and implement Paddle-based license key validation.
> **Pricing model**: Hybrid — one-time purchase + annual updates subscription
> **Est**: 12-15h

- [x] **S10.1** Code protection — lightweight ✅ DONE (2026-05-04)
  - PyInstaller `--key` AES-256 bytecode encryption added to `forex_pipeline.spec`
  - Strip docstrings script: `scripts/strip_docstrings.py`
  - `.env` files excluded from bundle
  - `scripts/build_python.bat` updated with new steps
  - Anti-debugging checks in Electron (`electron/anti_debug.ts`): DevTools detection, debugger timing, context menu blocking
  - **Files**: `forex_pipeline.spec`, `scripts/strip_docstrings.py`, `scripts/build_python.bat`, `electron/anti_debug.ts`

- [x] **S10.2** Paddle SDK integration ✅ DONE (2026-05-04)
  - Paddle v3 API client for license verify/activate/deactivate
  - Offline grace period (7 days without re-verification)
  - Sandbox mode support
  - **Files**: `api/licensing/paddle_client.py`

- [x] **S10.3** Machine fingerprinting ✅ DONE (2026-05-04)
  - WMI-based hardware ID (CPU, motherboard, BIOS, MAC, disk serial)
  - Cross-platform fallbacks (macOS, Linux)
  - Partial match tolerance (3/5 components = same machine)
  - Machine transfer flow (deactivate → activate on new machine)
  - **Files**: `api/licensing/fingerprint.py`

- [x] **S10.4** License enforcement in app ✅ DONE (2026-05-04)
  - `api/licensing/manager.py` — LicenseManager singleton with status/activate/verify/deactivate/trial
  - `api/licensing/gates.py` — Feature gates (FREE_MODELS, LOCKED_FEATURES)
  - `api/licensing/middleware.py` — FastAPI dependencies (require_feature, require_paid_model, require_licensed)
  - `api/routers/license.py` — 7 endpoints: /status, /activate, /deactivate, /verify, /trial, /check, /features
  - `api/main.py` — License router registered + security middleware installed
  - `electron/license.ts` — License IPC channels (activate, start-trial, get-status)
  - `electron/main.ts` — License check on startup + IPC registration
  - `electron/preload.ts` — Exposed license IPC to renderer
  - `frontend/src/components/LicenseDialog/LicenseDialog.tsx` — Activation dialog
  - `frontend/src/components/LicenseDialog/TrialCountdown.tsx` — Trial countdown badge
  - `frontend/src/pages/Settings/SettingsPage.tsx` — Live license status section
  - `frontend/src/api/queries.ts` — useLicenseStatus, useActivateLicense, useDeactivateLicense, useStartTrial
  - `frontend/src/api/schemas.ts` — LicenseStatusResponse type
  - **Files**: `api/licensing/manager.py`, `api/licensing/gates.py`, `api/routers/license.py`, `electron/license.ts`, `frontend/src/components/LicenseDialog/`

- [x] **S10.5** Encrypted local storage ✅ DONE (2026-05-04)
  - Fernet symmetric encryption with HKDF key derivation (APP_SECRET + machine fingerprint)
  - Auto-generated APP_SECRET stored next to secure.db
  - Encrypted SQLite: licenses, api_keys, trial, kv_store tables
  - WAL mode for concurrent access
  - **Files**: `api/licensing/storage.py`

- [x] **S10.6** Security audit ✅ DONE (2026-05-04)
  - `api/middleware/__init__.py` — SecurityHeadersMiddleware + RateLimitMiddleware
  - Rate limiting: 60 req/min general, 5 req/min for license activation
  - CORS locked down for non-desktop mode
  - `.env.example` updated with PADDLE_* and APP_SECRET vars
  - `tests/test_security_audit.py` — 41 tests covering headers, input validation, encrypted storage, feature gates, fingerprint, Paddle client, build security, Electron security, API endpoint audit
  - `tests/test_licensing.py` — 29 tests covering storage, fingerprint, gates, manager
  - npm audit: 3 vulnerabilities (DOMPurify via monaco-editor, Electron) — low risk for desktop app
  - **Files**: `api/middleware/__init__.py`, `tests/test_security_audit.py`, `tests/test_licensing.py`

---

## Sprint 11: Installer & Auto-Update ✅ COMPLETE (2026-05-04)

> **Goal**: Professional Windows installer with automatic updates.
> **Est**: 6-8h

- [x] **S11.1a** Brand cleanup + get-app-version IPC ✅ DONE (2026-05-04)
  - `electron/main.ts`: title "KodaQuant", ipcMain.handle("get-app-version"), updater/sentry wiring
  - `electron/utils.ts`: user data dir renamed from "FX ML Backtester" to "KodaQuant"
  - `electron/preload.ts`: update IPC channels (checkForUpdates, downloadUpdate, installUpdate, etc.)
  - **Files**: `electron/main.ts`, `electron/utils.ts`, `electron/preload.ts`

- [x] **S11.2** Auto-update system ✅ DONE (2026-05-04)
  - `electron-updater` installed + `electron/updater.ts` — full autoUpdater lifecycle
  - IPC handlers: check-for-updates, download-update, install-update, is-update-downloaded
  - Help menu "Check for Updates..." always visible (dev + prod)
  - `frontend/src/components/UpdateNotification/` — 4 states (available, downloading, ready, checking)
  - Shell-only update strategy (Electron+React delta ~2MB; Python backend requires full reinstall)
  - **Files**: `electron/updater.ts`, `electron/menu.ts`, `frontend/src/components/UpdateNotification/`

- [x] **S11.3** Crash reporting ✅ DONE (2026-05-04)
  - `@sentry/electron` installed + `electron/sentry.ts` — opt-in, production-only, PII scrubbed
  - DSN from SENTRY_DSN env var
  - `captureException()` and `captureMessage()` helpers
  - **Files**: `electron/sentry.ts`

- [x] **S11.1c** Version management ✅ DONE (2026-05-04)
  - `scripts/set_version.py` — single-source-of-truth version setter (package.json + spec file)
  - `scripts/publish_release.bat` — full build+tag+push pipeline
  - `scripts/build_electron.bat` — reads version from package.json
  - **Files**: `scripts/set_version.py`, `scripts/publish_release.bat`, `scripts/build_electron.bat`

- [x] **S11.F1** Fix: Help menu "Check for Updates" ✅ DONE (2026-05-04)
  - Menu item was hidden in dev mode (app.isPackaged guard removed)
  - **Files**: `electron/menu.ts`

- [x] **S11.F2** Fix: WebSocket progress field name mismatch ✅ DONE (2026-05-04)
  - Backend sends `period`/`total_periods`, frontend expected `month`/`total_months` → simMonth always undefined
  - Backend sends `n_trials`, frontend expected `total_trials` → hpoTotalTrials always undefined
  - Frontend now reads both variants: `event.month ?? event.period`, `event.total_trials ?? event.n_trials`
  - **Files**: `frontend/src/api/schemas.ts`, `frontend/src/stores/useJobStore.ts`

- [x] **S11.F3** Fix: HeatmapCellData not exported ✅ DONE (2026-05-04)
  - `SharpeHeatmap.tsx` declared `interface HeatmapCellData` without `export`
  - PerformanceHeatmapSection.tsx import was failing at runtime
  - **Files**: `frontend/src/components/charts/SharpeHeatmap.tsx`

- [ ] **S11.1b** NSIS installer branding ✅ DONE (2026-05-04)
  - Generated `build/installer-header.bmp` (150×57) and `build/installer-sidebar.bmp` (162×314)
  - KodaQuant dark theme with accent cyan branding
  - Uncommented bitmap refs in `electron-builder.yml`
  - **Files**: `build/installer-header.bmp`, `build/installer-sidebar.bmp`, `electron-builder.yml`

- [ ] **S11.4** First-run experience ✅ DONE (2026-05-04)
  - 2-step WelcomeWizard: (1) pair + timeframe selection, (2) optional OANDA API key + data directory
  - First-launch detection via `localStorage` key `kodaquant-welcome-done`
  - Settings persistence: oandaApiKey, dataDir saved to settings store
  - Default pair/timeframe set in backtest store on completion
  - "Skip for now" option available
  - **Files**: `frontend/src/pages/Welcome/WelcomePage.tsx`, `frontend/src/App.tsx`, `frontend/src/stores/useSettingsStore.ts`

---

## Sprint 12: Product Intelligence & UX Overhaul ✅ COMPLETE

> **Goal**: Make news features actually work and matter for trading. Add LLM-driven sentiment as a first-class feature. Redesign Dashboard and Results into what a trader actually needs. Add live price monitor + candlestick charts.
> **Branch**: `feature/s12-product-intelligence`
> **Est**: 22-24h

- [x] **S12.1** Fix broken news pipeline wiring ✅ DONE (2026-05-05)
  - `api/tasks.py`: before `_run_backtest_impl()`, call `NewsScraper().fetch_all()` → `SentimentAnalyzer(backend=cfg["news_sentiment_backend"]).score_articles()` → `aggregate_to_df()` → inject `bt._news_aggregated` and `bt._news_economic_events`
  - `pipeline/backtester/features_mixin.py`: when `use_news=True` and no data injected, log warning instead of silently skipping; default ON when data available
  - `config.py`: change `use_news` default to `True`; thread `news_sentiment_backend` through to `SentimentAnalyzer`
  - Integration test: backtest with `use_news=True` asserting news feature columns appear in output
  - **Files**: `api/tasks.py`, `pipeline/backtester/features_mixin.py`, `config.py`, `news/features.py`
  - **Est**: 2h

- [x] **S12.2a** LLM sentiment engine — core module ✅ DONE (2026-05-05)
  - New `pipeline/llm/` package (`__init__.py`, `sentiment.py`, `prompts.py`)
  - Abstract `LLMSentimentBackend` with `OllamaBackend` (default), `OpenAIBackend`, `AnthropicBackend`
  - Structured prompt per article → JSON output: `{direction: float[-1,1], confidence: float[0,1], volatility: float[0,1], currencies: [str]}`
  - Per-article caching: SQLite table `llm_sentiment_cache` (article hash → scores, process once, reuse forever)
  - Fallback: if Ollama unavailable, fall back to VADER silently
  - Config keys: `llm_sentiment_enabled=True`, `llm_backend="ollama"`, `llm_model="llama3"`, `llm_api_key=""`, `llm_weight=0.7`, `llm_batch_size=10`, `llm_cache_ttl_hours=720`
  - **Files**: `pipeline/llm/__init__.py`, `pipeline/llm/sentiment.py`, `pipeline/llm/prompts.py`, `config.py`
  - **Est**: 3h

- [x] **S12.2b** LLM sentiment — feature integration + blending ✅ DONE (2026-05-05)
  - Extend `news/features.py`: new `merge_llm_features()` that left-joins LLM sentiment columns onto OHLC bars
  - New feature columns: `llm_sentiment`, `llm_confidence`, `llm_volatility`, `llm_sentiment_ma_6`, `llm_sentiment_ma_24`
  - Blending formula: `final_sentiment = llm_weight * llm_sentiment + (1 - llm_weight) * vader_sentiment`
  - Wire into `features_mixin.py`: if `llm_sentiment_enabled`, call `merge_llm_features()` after VADER/finBERT merge
  - Wire into `api/tasks.py`: call LLM analysis after news fetch, before backtester run
  - **Files**: `news/features.py`, `pipeline/backtester/features_mixin.py`, `api/tasks.py`
  - **Est**: 2h

- [x] **S12.2c** LLM sentiment — frontend config + API endpoints ✅ DONE (2026-05-05)
  - `frontend/src/pages/Backtest/FeaturesPanel.tsx`: add LLM toggle, backend dropdown (Ollama/OpenAI/Anthropic), model name input, weight slider (0-1)
  - `api/routers/news.py`: add `GET /news/sentiment/live` endpoint returning current aggregate sentiment per pair
  - `api/schemas/backtest.py`: add LLM config fields
  - `frontend/src/api/queries.ts`: add `useLiveSentiment()` hook
  - `frontend/src/lib/constants.ts`: add LLM defaults
  - **Files**: `frontend/src/pages/Backtest/FeaturesPanel.tsx`, `api/routers/news.py`, `api/schemas/backtest.py`, `frontend/src/api/queries.ts`, `frontend/src/lib/constants.ts`
  - **Est**: 2h

- [x] **S12.2d** LLM sentiment — tests ✅ DONE (2026-05-05)
  - `tests/test_llm_sentiment.py`: mock Ollama responses, test caching, test blending, test fallback to VADER
  - Test per-article scoring, batch aggregation, walk-forward integrity
  - **Files**: `tests/test_llm_sentiment.py`
  - **Est**: 1h

- [x] **S12.3** Results history browser ✅ DONE (2026-05-05)
  - New `ResultsHistoryPage` component at `/results` route: full-width table of ALL completed backtests
  - Columns: Date, Pair, Models, Sharpe, Return %, Win Rate, Max DD, Status, Actions
  - Sortable by any column, filterable (pair, model, status), searchable
  - Click row → `/results/:jobId` for detailed view
  - Checkbox select → bulk export (CSV + JSON of selected results)
  - API: `GET /backtest/results/summary` (lightweight, no equity curves/trades); pagination (`offset`/`limit`) on existing `GET /backtest`
  - Routing: `/results` → `ResultsHistoryPage`, `/results/:jobId` → existing `ResultsPage`
  - **Files**: new `frontend/src/pages/Results/ResultsHistoryPage.tsx`, `api/routers/backtest.py`, `api/schemas/backtest.py`, `frontend/src/App.tsx`, `frontend/src/api/queries.ts`
  - **Est**: 3h

- [x] **S12.4** Dashboard redesign — live command center ✅ DONE (2026-05-05)
  - New `QuickActions` component: 2-3 buttons (New Backtest, Re-run Last)
  - New `MarketPulsePanel` component: live sentiment gauge (VADER + LLM blended), real article feed with VADER scores, cache status
  - Pair selector dropdown: user picks active pair, MarketPulse + PriceTicker respond
  - Restructure `DashboardPage`: Quick Actions + Pair Selector → Price Ticker → Candlestick Chart → Market Pulse → KPIs → Recent Activity
  - Replaced KPIs: Avg Sharpe, Avg Win Rate, Profitable Months % (with per-card explanation text)
  - Moved Heatmap from Dashboard to `/results` page
  - Fixed VADER crash (`s.direction` → `s.score`), added `top_articles` to live sentiment API
  - Fixed heatmap 404 (route shadowing), Celery `bind=True` error, emit_event duplicate kwarg
  - **Files**: `frontend/src/pages/Dashboard/DashboardPage.tsx` (restructure), new `Dashboard/QuickActions.tsx`, new `Dashboard/MarketPulsePanel.tsx`, `api/routers/news.py`, `DashboardKPIs.tsx`, `api/routers/backtest.py`, `api/tasks.py`
  - **Est**: 3h (actual: included 3 bug fixes)

- [x] **S12.5** Live price monitor + candlestick charts + backtest visualization ✅ DONE (2026-05-06)
  - **Data strategy**: SQLite candles table for historical OHLCV (instant, offline), OANDA PricingInfo for live bid/ask (REST poll, 3s), user's own OANDA key from encrypted Settings storage
  - `pipeline/data_sqlite.py`: new `get_latest_candles(pair, timeframe, limit)` method — indexed query, no date range needed
  - New `api/routers/prices.py`: `GET /prices/live?pairs=EURUSD,GBPUSD,USDJPY&lookback_bars=50` (bid/ask/mid from OANDA + sparkline from SQLite) + `GET /candles/{pair}/{tf}?limit=200` (OHLC bars from SQLite)
  - `api/routers/backtest.py`: new `GET /backtest/{job_id}/trades/chart-data?model=...` — candle OHLCV + trade markers with entry/exit prices (joined from candles) + equity curve
  - `api/tasks.py`: enrich trade log with `entry_price`/`exit_price` by joining timestamps with candle closes
  - New `frontend/src/pages/Dashboard/PriceTicker.tsx`: 3-pair horizontal strip (bid/ask/mid/change%/sparkline), loading skeleton, "Add OANDA key" empty state
  - New `frontend/src/components/charts/CandlestickChart.tsx`: lightweight-charts CandlestickSeries + volume HistogramSeries, M15/M30/H1/H2 toggle, dark theme
  - New `frontend/src/pages/Results/BacktestChart.tsx`: CandlestickSeries + trade markers (▲ buy ▼ sell) + equity overlay + barrier labels
  - Frontend hooks: `useLivePrices()`, `useCandles()`, `useTradeChartData()`
  - One-time M15 data download for EURUSD/GBPUSD/USDJPY
  - **Files**: `pipeline/data_sqlite.py`, new `api/routers/prices.py`, `api/routers/backtest.py`, `api/tasks.py`, `api/main.py`, new `PriceTicker.tsx`, new `CandlestickChart.tsx`, new `BacktestChart.tsx`, `DashboardPage.tsx`, `ResultsPage.tsx`, `frontend/src/api/queries.ts`, `frontend/src/api/schemas.ts`, `requirements.txt`
  - **Est**: 7h

---

## Sprint 13: Beta & Launch ⬜ NOT STARTED

> **Goal**: Ship to real users, iterate, launch publicly.
> **Est**: 6-8h (+ 2 weeks beta period)

- [ ] **S13.1** Closed beta
  - Recruit 10-20 beta testers (forex forums, Reddit, Discord)
  - Beta build with telemetry + feedback button
  - Feedback collection form (in-app + Google Forms)
  - Known issues tracker
  - **Est**: 2h setup + 2 weeks beta period

- [ ] **S13.2** Performance optimization
  - Profile cold start time (target < 5 seconds to UI)
  - Optimize backtest memory for 8GB RAM machines
  - Reduce Electron bundle size (tree-shaking, lazy loading)
  - FastAPI startup optimization
  - **Est**: 3h

- [ ] **S13.3** Launch preparation
  - Product Hunt listing draft
  - Social media announcements (Twitter/X, LinkedIn, Reddit r/algotrading)
  - Email list setup (Mailchimp/ConvertKit)
  - Demo video recording (2-minute walkthrough)
  - Press kit (screenshots, logos, feature list)
  - **Est**: 2h

- [ ] **S13.4** Post-launch monitoring
  - Monitor Paddle sales dashboard
  - Sentry error rates
  - Customer support channel (email + Discord)
  - Weekly metrics review (sales, crashes, feature requests)
  - **Est**: ongoing

---

## Sprint 15: KodaQuant Branding ✅ COMPLETE (2026-05-10)

> **Goal**: Rename and rebrand from "FX ML Backtester / thesisproj" to "KodaQuant" for commercial launch.
> **Est**: 4-6h

- [x] **S15.1** Source code name cleanup ✅ DONE (2026-05-10)
  - 12 files updated: `electron/tray.ts`, `forex_pipeline.spec`, `scripts/dev.bat`, `run_smoke_gpu.bat`, `api/config.py`, `run_server.py`, `news/__init__.py`, `schemas/__init__.py`, `tests/test_full_system.py`, `tests/test_electron_build.py`, `CLAUDE.md`, `README.md`
  - Removed stale `ui/` __pycache__ directory
  - 25 automated tests verify zero stale "FX ML Backtester" references across all file types
  - **Files**: `tests/test_branding.py`

- [x] **S15.2** Icon set ✅ DONE (2026-05-10)
  - Redesigned `build/icon.svg`: geometric diamond on dark background with cyan (`#00E5FF`), zero text
  - New `scripts/generate_icons.py`: PIL ImageDraw renders PNG (1024x1024) and multi-res ICO (8 resolutions: 16-256px) via binary ICO format builder
  - Generated: `build/icon.png`, `build/icon.ico`, `frontend/public/favicon.ico`
  - Colors already defined in `index.css` as `--color-koda-*` CSS custom properties (Tailwind v4, no config file needed)
  - Dark mode already default
  - 12 automated tests verify icon content, colors, resolutions, and electron-builder references
  - **Files**: `build/icon.svg`, `scripts/generate_icons.py`, `tests/test_branding.py`

- [x] **S15.3** Splash screen & installer branding ✅ DONE
  - Splash screen (`electron/splash.ts`) already says "KodaQuant" with diamond logo + progress bar
  - Installer header/sidebar BMPs exist in `build/` (generated S11.1b with KodaQuant dark + cyan theme)
  - App icons (ico, png) generated via `scripts/generate_icons.py`
  - **Verification**: manual build + run required to confirm visual quality
  - **Files**: `electron/splash.ts`, `build/installer-header.bmp`, `build/installer-sidebar.bmp`, `build/icon.*`

- [x] **S15.4** Documentation & about screen ✅ DONE (2026-05-10)
  - `AboutDialog.tsx`: GitHub link updated from `thesisproj` to `kodaquant`
  - `SETUP.md`: full rewrite — KodaQuant branding, removed all Streamlit content, proper markdown tables
  - `PROJECT_PLAN.md`: marked as superseded, updated header to KodaQuant
  - `ROADMAP.md`: S15 status section updated (this section)
  - `CLAUDE.md`: updated S15 status
  - 4 automated tests verify docs
  - **Files**: `frontend/src/components/shared/AboutDialog.tsx`, `SETUP.md`, `PROJECT_PLAN.md`, `ROADMAP.md`

**Sprint 15 complete**: 37 automated branding tests pass. All source code, icons, docs use KodaQuant.
**Test file**: `tests/test_branding.py`

---

## Sprint 16: Trading Logic, Overfitting & Backtest Transparency ✅ COMPLETE

> **Goal**: Improve core backtest quality — better training logic, overfitting detection, and results that users can actually understand and trust.
> **Est**: 12-16h

- [x] **S16.1** Overfitting detection & reporting ✅ DONE
  - Train/test score divergence detection (e.g. Sharpe drops > 40% from train to OOS)
  - Bootstrap confidence intervals for key metrics (Sharpe, return, max DD)
  - Overfitting risk score per model in results (color-coded: green/yellow/red)
  - Cross-validation fold stability report (std of Sharpe across folds)
  - **Files**: `pipeline/backtester/evaluation_mixin.py`, `pipeline/metrics_eval.py`, new `pipeline/overfitting.py`

- [x] **S16.2** Walk-forward transparency panel ✅ DONE
  - Per-period breakdown: train Sharpe vs test Sharpe, train return vs test return
  - Visual timeline showing which months are train vs test (interactive)
  - Signal count per period (was the model trading or sitting out?)
  - Regime overlay: which market regime each period fell into
  - **Files**: `frontend/src/pages/Results/WalkForwardPanel.tsx`, new `api/routers/backtest.py` endpoint

- [x] **S16.3** Training diagnostics — what the model actually learned ✅ DONE
  - Feature importance heatmap per model (gain-based for tree models, permutation for others)
  - Prediction distribution: histogram of predicted probabilities vs actual outcomes
  - Confusion matrix per walk-forward period
  - "Was this model confident?" — filter trades by prediction confidence bands
  - **Files**: `pipeline/backtester/real_trading_mixin.py`, `frontend/src/pages/Results/TrainingDiagnostics.tsx`

- [x] **S16.4** Plain-English backtest summary generator ✅ DONE
  - Auto-generate a 3-sentence natural-language summary: "This backtest ran XGBoost on EURUSD H1 from Jan 2023 to Dec 2025. It achieved a Sharpe of 1.2 with 58% win rate and -8.3% max drawdown. The model was most confident during low-volatility trending periods."
  - Key findings bullets (best period, worst period, regime performance)
  - Copy-to-clipboard for sharing
  - **Files**: new `pipeline/summary_generator.py`, `frontend/src/pages/Results/BacktestSummary.tsx`

- [x] **S16.5** Better default training parameters ✅ DONE
  - Audit search space bounds — tighten ranges that allow degenerate configs
  - Add early stopping patience defaults (stop if 20 trials without improvement)
  - Minimum trade count filter (reject configs that trade < 10 times/month)
  - Default to `calibrate=True` for production runs
  - **Files**: `config.py` (SEARCH_SPACE), `pipeline/tuning/runner.py`

- [x] **S16.6** Backtest Setup UI redesign ✅ DONE
  - 6-tab layout: Quick Start, Asset & Model, Study & HPO, Features, Hyperparameters, Execution
  - ConfigSummaryBar with always-visible 5-column preview
  - Quick Start: categorized presets (Debug, Classical, Deep, Ensemble, RL)
  - ValidationBar: missing-config indicator, save preset, deploy button
  - **Files**: `frontend/src/pages/Backtest/BacktestPage.tsx`, `QuickStartTab.tsx`, `HyperparamsTab.tsx`, `ConfigSummaryBar.tsx`, `ValidationBar.tsx`

- [x] **S16.7** Parameter Intelligence (Guide + Explorer + fANOVA + LLM Advisor) ✅ DONE
  - Parameter Guide: dynamic warnings per model from store values
  - Parameter Explorer: ParallelCoordinates + ContourPlot on ResultsPage
  - fANOVA interaction effects in OverfittingPanel
  - LLM Advisor: "Analyze Results" button → AI suggests next study → "Apply Study" one-click
  - Hyperparameters tab: ModelHyperparamsPanel + ParameterGuideInline
  - **Files**: `ParameterGuide.tsx`, `ParallelCoordinates.tsx`, `ContourPlot.tsx`, `LLMAdvisor.tsx`, `pipeline/overfitting.py`, `pipeline/llm/advisor.py`

**Sprint 16 complete** when: All 7 sub-tasks complete. Overfitting detection works, summary is human-readable, training diagnostics are transparent, UI is tab-based with parameter intelligence. ✅ ALL DONE

---

## Sprint 16.8: Model Persistence, Deployment & Experiment Tracking ✅ COMPLETE

> **Goal**: Save trained models as deployable artifacts, track experiment lineage, and enable model sharing across KodaQuant installations.
> **Dependencies**: Zero. Everything uses existing `joblib`, SQLite, sklearn.
> **Est**: 15-19h
> **Branch**: `feature/sprint16.8-model-persistence`
> **Completed**: 2026-05-21 — All 4 phases (A-D) implemented + tested (64 fast tests pass)

### Phase A: Model Snapshot System ✅ DONE
- [x] **A.1-A.6**: `save_snapshot()`, `load_snapshot()`, metadata schema, pip_freeze, auto-save per walk-forward cycle, parent_job_id lineage
- **Files**: `pipeline/model_persistence.py`, `models/base_model.py`, `api/tasks.py`, `pipeline/data_sqlite.py`

### Phase B: Deployment Registry ✅ DONE
- [x] **B.1-B.6**: `deployed_models` table, `model_registry_disk.py` (scan/register/activate/delete/tags), API endpoints, DeployedModelsPage UI, TagEditor
- **Files**: `pipeline/model_registry_disk.py`, `api/routers/models.py`, `DeployedModelsPage.tsx`, `TagEditor.tsx`

### Phase C: Multi-Model Signal Engine ✅ DONE
- [x] **C.1-C.4**: `MetaEnsemble` (majority/soft/weighted voting), registry entry, search space, frontend ModelSelector
- **Files**: `models/meta_ensemble.py`, `models/registry.py`, `config.py`, `ModelSelector.tsx`

### Phase D: Live Prediction Bridge ✅ DONE
- [x] **D.1-D.3**: `.active` pointer, `/active/predict`, `/active/predict-with-data`, `/active/compare`
- **Files**: `pipeline/model_persistence.py`, `api/routers/models.py`, `pipeline/data_sqlite.py`

**Results**: 160 total tests (72 backend + 88 frontend), zero TS errors. All 4 phases deployed to `feature/sprint16.8-model-persistence`.

---

## Sprint 16.9: Saved Model — Forward Test & Live Trading Bridge

> **Goal**: Make saved models actually usable. Add a Forward Test tab for temporal testing and bridge deployed models into live trading.
> **Dependencies**: Sprint 16.8 (model persistence + registry).
> **Est**: 5-7h
> **Branch**: `feature/sprint16.8-model-persistence`

### Phase 1: Forward Test Engine + UI (3-4h)

- [ ] **P1.1** `pipeline/forward_test.py` — `run_forward_test(snapshot_path, pair, tf, start, end, sizing)`
  - Loads snapshot, creates MLBacktester, injects model, runs `real_trading_simulation()` with `skip_hpo=True, skip_training=True`
  - Returns full metrics/equity/trades (same schema as backtest results)
  - Also produces `generate_forecast_errors()` for future DM test (Phase 3 deferral)
  - **Files**: New `pipeline/forward_test.py`

- [ ] **P1.2** API endpoint — `POST /models/{id}/forward-test`
  - Validates model exists, launches Celery job via `_run_forward_test_impl`
  - Returns `job_id` → frontend navigates to Monitor → Results
  - **Files**: `api/routers/models.py`, `api/tasks.py`

- [ ] **P1.3** Frontend — "Forward Test" tab (7th tab on BacktestPage)
  - Model selector dropdown from `/models/deployed`
  - Date range picker, pair/timeframe selectors, position sizing dropdown
  - "Run Forward Test" button → creates job → navigates to Monitor
  - **Files**: New `ForwardTestTab.tsx`, modify `BacktestPage.tsx`

### Phase 2: Live Trading with Saved Models (1-2h)

- [ ] **P2.1** Modify `POST /live/deploy` — add `model_id` parameter
  - If provided: load snapshot via `load_snapshot(model_id)`, skip `_run_backtest_for_model()`
  - Signal loop unchanged (already uses `session["model_obj"]`)
  - **Files**: `api/routers/live.py`

- [ ] **P2.2** Frontend — LiveTradingPage saved model selector
  - Replace model type dropdown with deployed model list (from `/models/deployed?status=active`)
  - Pass `model_id` in deploy request
  - **Files**: `LiveTradingPage.tsx`

### Phase 3 (DEFERRED): Validation Gate with DM/SPA/Chow Tests
> Full statistical validation gate for auto-promotion. Deferred to Sprint 17+.

- [ ] DM Test (Diebold-Mariano) — compare forecast errors of candidate vs benchmark
- [ ] SPA Test (Hansen) — control for data snooping bias across K configurations  
- [ ] Chow Test — structural break detection before promotion
- [ ] Dual-lock gate: candidate must beat Active Model AND Naive Baseline

### Test Specifications (8 tests)

- [ ] `test_forward_test_engine` — load saved model, run on test range, verify metrics
- [ ] `test_forward_test_no_trades` — model that predicts flat → clean zero-trade result
- [ ] `test_forward_test_endpoint` — POST /models/{id}/forward-test returns valid job_id
- [ ] `test_live_deploy_with_model_id` — deploy saved model → signal loop starts
- [ ] `test_live_deploy_no_model_id` — backward compat: train-fresh path still works
- [ ] `test_forward_test_tab_renders` — tab shows model list + date picker
- [ ] `test_live_page_model_selector` — saved model dropdown renders active models
- [ ] `test_forward_test_results_page` — results page renders forward test output identically

---

## Sprint 17: UI Polish, Tabs Flow & Search

> **Goal**: Make the app feel professional — smooth page transitions, global search, consistent component patterns, and UI cleanup.
> **Est**: 8-10h

- [ ] **S17.1** Global search bar (Cmd+K / Ctrl+K)
  - Search across: backtest jobs, model names, currency pairs, settings
  - Fuzzy matching with keyboard navigation
  - Recent searches stored in localStorage
  - **Files**: `frontend/src/components/shared/CommandPalette.tsx`
  - **Est**: 3h

- [ ] **S17.2** Tab flow & navigation redesign
  - Results page: tab per model (not scroll) with persistent tab state
  - Compare page: side-by-side with tab anchors (Overview, Equity, Metrics, Trades, HPO)
  - Settings page: tab sections (General, Models, Execution, License, About)
  - Keyboard shortcuts for tab navigation (1-5 keys)
  - **Files**: `frontend/src/pages/Results/ResultsPage.tsx`, `frontend/src/pages/Compare/ComparePage.tsx`, `frontend/src/pages/Settings/SettingsPage.tsx`
  - **Est**: 2h

- [ ] **S17.3** UI consistency & cleanup pass
  - Standardize all section headers (typeface, size, tracking, margin)
  - Consistent skeleton loading states (replace ad-hoc spinners)
  - Error boundary per page (not just global)
  - Responsive breakpoints for 1280px / 1600px / 1920px
  - Remove unused imports, dead code, commented-out blocks across all pages
  - **Files**: all `frontend/src/pages/**/*.tsx`
  - **Est**: 2h

- [ ] **S17.4** Empty states & onboarding hints
  - No-data states for Dashboard, Results, Compare with call-to-action
  - Tooltip hints for first-time users (e.g. "Select a model to begin")
  - Progressive disclosure: hide advanced features until basic workflow completed
  - **Files**: `frontend/src/components/shared/EmptyState.tsx`, all page components
  - **Est**: 2h

---

## Sprint 18: Live News & Market Data Integration

> **Goal**: Connect real-time news sentiment and live price data to backtesting and (later) live trading. Make S6's news features actually impactful on trading signals.
> **Est**: 10-14h

- [ ] **S18.1** Live news feed with impact scoring
  - Real-time RSS aggregation (ForexLive, Reuters, Bloomberg) via background task
  - NLP impact scoring per currency pair (EURUSD, GBPUSD, etc.) in real time
  - News timeline overlay on backtest equity curves (already partially done in S6.4)
  - Live news panel in dashboard with auto-refresh
  - **Files**: `news/scraper.py`, `news/sentiment.py`, `frontend/src/pages/Dashboard/NewsFeedPanel.tsx`, new `api/routers/news.py` live endpoint
  - **Est**: 3h

- [ ] **S18.2** Live price data from OANDA API
  - OANDA streaming API integration for all configured pairs
  - Real-time candle subscriptions (M1, M5, M15, M30, H1, H4)
  - Price cache in Redis with TTL (latest 1000 candles per pair per timeframe)
  - Frontend WebSocket for live price ticks
  - **Files**: new `pipeline/live_feed.py`, `api/routers/prices.py`, `frontend/src/api/livePrices.ts`
  - **Est**: 4h

- [ ] **S18.3** News-impact feature engineering (live mode)
  - Score each incoming news event → generate real-time sentiment features
  - Map news events to affected pairs (e.g. NFP → all USD pairs, BOE rate → GBP pairs)
  - Feature pipeline: news event → sentiment score → rolling aggregates → model input
  - Backtest mode: replay historical news as if they arrived in real-time
  - **Files**: `news/features.py` (extend), `pipeline/backtester/features_mixin.py`
  - **Est**: 3h

- [ ] **S18.4** Live market dashboard
  - Price ticker strip at top of Dashboard (scrolling pair prices)
  - Mini sparkline charts per pair (last 24h)
  - Economic calendar widget with next-7-days events
  - Click pair → open backtest with that pair pre-selected
  - **Files**: `frontend/src/pages/Dashboard/MarketDashboard.tsx`, `frontend/src/components/charts/SparklineChart.tsx`
  - **Est**: 3h

---

## Sprint 19: Ensemble Models & Model Extensibility

> **Goal**: Implement all ensemble model types defined in the registry, add custom model support, and make the model layer truly extensible.
> **Est**: 8-10h

- [ ] **S19.1** Remaining ensemble implementations
  - Ensemble VotingClassifier (majority vote, soft vote, weighted)
  - Ensemble Stacking (meta-learner on top of base model predictions)
  - Ensemble Adaptive Regime — enhance with dynamic weight rebalancing per regime
  - Ensemble CNN-LSTM-XGBoost — verify end-to-end, add to UI model selector
  - Ensure all 10+ model types appear in frontend ModelSelector
  - **Files**: `models/ensemble_voting.py`, `models/ensemble_stacking.py`, `models/ensemble_adaptive_regime.py`, `models/ensemble_cnn_lstm_xgboost.py`, `frontend/src/pages/Backtest/ModelSelector.tsx`
  - **Est**: 4h

- [ ] **S19.2** Custom model plugin system
  - `models/plugins/` directory — drop-in Python files with `build()`, `train()`, `predict()` interface
  - Auto-discovery: scan plugins dir on startup, register with ModelRegistry
  - Frontend: "Custom Model" option in ModelSelector with upload textarea
  - Validation: run smoke test on plugin registration
  - **Files**: new `models/plugins/`, `models/registry.py` (extend)
  - **Est**: 3h

- [ ] **S19.3** Model comparison benchmark suite
  - Standardized benchmark dataset (fixed pair, timeframe, date range)
  - Run all models on same data with same HPO budget
  - Generate comparison report: Sharpe, return, win rate, avg trade, max DD, training time
  - Export as PDF/HTML for sharing
  - **Files**: `pipeline/model_comparison.py` (extend), `frontend/src/pages/Compare/BenchmarkReport.tsx`
  - **Est**: 2h

---

## Sprint 20: LLM / AI Integration & Intelligent Trading

> **Goal**: Leverage LLMs for market interpretation, strategy generation, and trade reasoning. Make the product genuinely AI-augmented, not just ML.
> **Est**: 10-14h

- [ ] **S20.1** LLM-powered market commentary
  - On-demand natural language analysis of current market conditions
  - Feed recent price action + news events + sentiment to LLM via structured prompt
  - Display in a "Market Insight" panel (not chat, just generated commentary)
  - Cache results (5-min TTL) to avoid excessive API calls
  - Support OpenAI, Anthropic, and local LLM (Ollama) backends
  - **Files**: new `pipeline/llm/commentary.py`, `api/routers/insights.py`, `frontend/src/pages/Dashboard/MarketInsightPanel.tsx`
  - **Est**: 3h

- [ ] **S20.2** Strategy suggestion engine
  - Given pair + timeframe + recent performance, suggest strategy adjustments
  - "Your XGBoost model on EURUSD H1 had 48% win rate last month. Consider: (1) tightening stop-loss to 15 pips, (2) switching to H4 timeframe, (3) adding ATR volatility filter."
  - Not auto-executing — purely advisory
  - **Files**: new `pipeline/llm/strategy_suggest.py`
  - **Est**: 2h

- [ ] **S20.3** LLM-augmented feature engineering
  - LLM generates candidate feature specifications from market context
  - Human-in-the-loop: suggest features → user approves → auto-generate code → run backtest
  - E.g. "Given the current NFP-sensitive regime, consider: EURUSD_H1_ATR_RATIO_14 = ATR(14) / ATR(50)"
  - **Files**: new `pipeline/llm/feature_suggest.py`, `pipeline/backtester/features_mixin.py` (extend)
  - **Est**: 3h

- [ ] **S20.4** Model insertion via LLM
  - Define a model spec in natural language → LLM generates the Python code
  - Auto-validate (build/train/predict smoke test)
  - Register as custom plugin (ties into S19.2)
  - Sandbox: run generated code in restricted environment
  - **Files**: new `pipeline/llm/model_generator.py`, `models/plugins/` (auto-generated)
  - **Est**: 3h

---

## Sprint 21: Live Trading with OANDA

> **Goal**: Enable real-time trading via OANDA API (demo first, then live). Paper trading with real data, then live execution with risk controls.
> **Est**: 12-16h

- [ ] **S21.1** OANDA API client (v20 REST)
  - Account info, position list, trade list, order submission
  - Market + limit + stop orders with SL/TP
  - Position management (close partial, modify SL/TP)
  - Demo and live account support (configurable API key + URL)
  - Rate limiting (OANDA: 120 req/sec for demo, 20 for live)
  - **Files**: new `trading/oanda_client.py`, `trading/oanda_models.py`
  - **Est**: 3h

- [ ] **S21.2** Paper trading engine
  - Run trained models against live OANDA price feed
  - Generate signals, execute paper trades (no real money)
  - Track paper portfolio: equity, positions, P&L in real time
  - Compare paper vs backtest performance for the same period
  - **Files**: new `trading/paper_engine.py`, `api/routers/trading.py`
  - **Est**: 4h

- [ ] **S21.3** Live trading engine with risk controls
  - Signal → execute via OANDA API with position sizing
  - Max position size limit (e.g. 2% of equity)
  - Max daily trades limit
  - Kill switch: stop all trading if drawdown > threshold
  - Confirmation dialog before live order submission
  - Trade journal: timestamp, signal, entry, exit, P&L, model confidence
  - **Files**: new `trading/live_engine.py`, `trading/risk_controls.py`, `frontend/src/pages/Trading/LiveTradingPage.tsx`
  - **Est**: 4h

- [ ] **S21.4** Trading dashboard UI
  - Live position monitor (open trades, P&L, time in trade)
  - Signal history with confidence scores
  - One-click "Start Paper Trading" / "Start Live Trading" with warnings
  - Emergency stop button (closes all positions)
  - Trade history with filtering and CSV export
  - **Files**: `frontend/src/pages/Trading/` (TradingPage, PositionMonitor, TradeHistory, RiskControls)
  - **Est**: 4h

---

## Superseded Phases

The following phases from the original roadmap are **superseded** by the desktop app strategy (Sprints 7-13):

| Old Phase | Status | Replaced By |
|-----------|--------|-------------|
| Phase 4 (Streamlit UI) | Removed (2026-04-20) | React frontend (Sprint 8) is the product |
| Phase 5 (Streamlit Cloud) | Cancelled | React desktop app is the product |
| Phase 6 (Auth & User DB) | Cancelled | S10 (licensing) — no multi-user DB needed |
| Phase 8 (Multi-User + Stripe) | Cancelled | S12 (Paddle) — desktop billing model |
| Phase 9 (Python SDK) | Deferred | Post-launch consideration |

Phase 4 (Feature Parity) is ~90% complete — only 4.1-4.4 remain (audit + backport from `init-proj`).
Phase 7 (Docker + CI/CD) is now Sprint 4.
Phase 10 (News & Sentiment) is now Sprint 6.
Phase 11 (Testing) is now Sprint 5.

---

## Quick Reference: How to Run

```bash
# Start Redis (required for Celery + WebSocket progress)
redis-server

# Start FastAPI backend (port 8002)
cd api && uvicorn main:app --host 127.0.0.1 --port 8002 --reload

# Start Celery worker (separate terminal)
celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery

# Start React frontend (port 5173)
cd frontend && npm run dev

# Then open: http://localhost:5173
```

## Key Files

| File | Role |
|------|------|
| `api/main.py` | FastAPI entry point (uvicorn) |
| `api/tasks.py` | Celery backtest tasks + result serialization |
| `api/routers/backtest.py` | Backtest + heatmap + cross-pair API endpoints |
| `api/routers/news.py` | News status + economic events API |
| `api/routers/license.py` | License management (7 endpoints) |
| `api/licensing/` | Paddle, fingerprint, storage, gates, middleware |
| `api/middleware/` | Security headers + rate limiting |
| `api/schemas/news.py` | News event schemas |
| `frontend/src/pages/Backtest/` | Backtest config UI (9 components) |
| `frontend/src/pages/Results/` | Results display + charts (7 components) |
| `frontend/src/pages/Compare/` | Model comparison + cross-pair (5 components) |
| `frontend/src/pages/Dashboard/` | Dashboard + KPIs + heatmap + thumbnails |
| `frontend/src/components/charts/` | All chart components (10 files) |
| `frontend/src/api/websocket.ts` | WebSocket manager for real-time progress |
| `frontend/src/api/queries.ts` | React Query hooks for all API endpoints |
| `frontend/src/stores/useJobStore.ts` | Job state + WebSocket progress tracking |
| `frontend/src/components/UpdateNotification/` | Auto-update UI notification |
| `electron/main.ts` | Electron main process (KodaQuant shell) |
| `electron/updater.ts` | Auto-update via electron-updater |
| `electron/sentry.ts` | Opt-in crash reporting |
| `electron/license.ts` | License IPC for Paddle |
| `pipeline/backtester/composed.py` | MLBacktester engine |
| `pipeline/backtester/data_mixin.py` | Data loading + date range handling |
| `pipeline/backtester/real_trading_mixin.py` | Real trading simulation + walk-forward |
| `pipeline/backtester/features_mixin.py` | Feature engineering (TA indicators) |
| `pipeline/runtime.py` | GPU detection, thread budgets, CUDA config |
| `config.py` | Global configuration |
| `models/registry.py` | Model registry |
| `pipeline/execution/position_sizing.py` | Position sizing models |
| `pipeline/model_comparison.py` | Model comparison & leaderboard |
| `scripts/set_version.py` | Single-source version management |
| `scripts/publish_release.bat` | Build + tag + push release pipeline |
| `scripts/build_electron.bat` | 7-step Electron + PyInstaller build |
| `electron-builder.yml` | Electron builder config (NSIS, GitHub publish) |

## Full Sprint Sequence

| Sprint | Topic | Est | Status |
|--------|-------|-----|--------|
| **S1** | Model Comparison & Leaderboard | 3-4h | DONE |
| **S2** | Advanced Execution Models | 6-8h | DONE |
| **S3** | Multi-Currency Expansion | 4-5h | DONE |
| **S4** | Docker + CI/CD | 3-4h | PARTIAL (RAM opt + CI done) |
| **S5** | Comprehensive Tests + Benchmarks | 4-6h | ✅ DONE (496+ tests) |
| **S6** | News & Sentiment Features | 6-8h | DONE |
| **S7** | FastAPI Backend | 8-10h | DONE |
| **S8** | React Frontend | 20-25h | DONE |
| **S8B** | Frontend ↔ API Integration & Bug Fixes | 6-8h | ✅ DONE (2026-04-20) |
| **S9** | Electron Desktop Shell | 10-12h | ✅ COMPLETE (S9.1-9.5 all done) |
| **S10** | Security & Licensing (Paddle) | 12-15h | ✅ DONE (2026-05-04) |
| **S11** | Installer & Auto-Update | 6-8h | ✅ COMPLETE (2026-05-04) |
| **S12** | Product Intelligence & UX Overhaul | 22-24h | ✅ COMPLETE (7 sub-tasks all done, 3 bug fixes applied) |
| **S13** | Beta & Launch | 6-8h | TODO |
| **S14** | Pipeline Enhancements (daily WF, HPO duration) | 5-8h | ✅ DONE |
| **S15** | KodaQuant Branding | 4-6h | ✅ COMPLETE (2026-05-10, 37 tests) |
| **S16** | Trading Logic, Overfitting & Backtest Transparency | 12-16h | ✅ DONE |
| **S16.8** | Model Persistence, Deployment & Experiment Tracking | 15-19h | ✅ COMPLETE (2026-05-21, 64 tests) |
| **S16.9** | Forward Test + Live Trading Bridge | 5-7h | IN PROGRESS |
| **S17** | UI Polish, Tabs Flow & Search | 8-10h | TODO |
| **S18** | Live News & Market Data Integration | 10-14h | TODO |
| **S19** | Ensemble Models & Model Extensibility | 8-10h | TODO |
| **S20** | LLM / AI Integration & Intelligent Trading | 10-14h | TODO |
| **S21** | Live Trading with OANDA | 12-16h | TODO |
| **S22** | Commercial Infrastructure (deferred from S12) | 8-10h | TODO |
| **S23** | Pipeline Stability & Live Monitor UX | 6-8h | ✅ DONE (2026-05-14) |

## Completion Criteria Summary

| Sprint | When it's done |
|--------|---------------|
| S1-S2 | Execution models complete, all sizing/stop/risk models work |
| S3-S4 | Multi-currency supported, Docker builds pass, CI green |
| S5-S6 | 200+ tests, >80% coverage, news features integrated | ✅ BOTH DONE — 496+ tests
| S7 | FastAPI serves all pipeline operations via REST + WebSocket | ✅ DONE
| S8 | React UI replaces Streamlit for all user interactions | ✅ DONE — all 8 sub-tasks complete |
| S9 | Electron wraps React + Python into desktop app | ✅ COMPLETE (S9.1-9.5 all done) |
| S10 | Code protected, Paddle licensing active, feature gating works | ✅ DONE
| S11 | Windows installer + auto-update functional | ✅ COMPLETE (2026-05-04) |
| S12 | News pipeline works, LLM sentiment is ML feature, Results shows all history, Dashboard has live price ticker + candlestick charts + Market Pulse, backtest trade visualization on Results detail, live monitor with multi-pair grid |
| S13 | Beta tested, publicly launched, first sales |
| S15 | All user-facing text says "KodaQuant", professional branding | ✅ DONE — 37 tests, 12 files cleaned, icons regenerated |
| S16 | Overfitting detection works, backtest summary is human-readable, training is transparent | ✅ DONE |
| S16.8 | Model persistence complete — snapshots save/load/export/import, deployed models registry with CRUD, MetaEnsemble signal committee, live prediction bridge with compare endpoint | ✅ DONE |
| S16.9 | Forward Test tab operational — saved models tested on any date range; Live Trading deploys saved models instead of training fresh |
| S17 | Cmd+K search, tab-style navigation, consistent UI, no dead code |
| S18 | Live news feed + live OANDA prices feed into backtesting + dashboard |
| S19 | All ensemble types working, plugin system for custom models |
| S20 | LLM commentary, strategy suggestions, AI-augmented features |
| S21 | Paper trading on OANDA demo, live trading with risk controls |
| S22 | Product listed on Paddle, landing page live, docs published, legal complete, analytics active |
| S23 | Log noise reduced 98%, live equity chart shows BH from origin with model grouping, no double-submit, no TypeError crash |

---

## Sprint 23: Pipeline Stability & Live Monitor UX ✅ COMPLETE (2026-05-14)

> **Goal**: Eliminate log noise, fix live monitor equity chart, prevent double-submission bugs.
> **Branch**: `feature/phase2-api-bridge`
> **Est**: 6-8h

- [x] **S23.1** Double-submission guard ✅ DONE (2026-05-14)
  - RunSummary deploy button was `disabled={errors > 0}` but didn't check `isSubmitting`
  - `isSubmitting` was never passed from BacktestPage to RunSummary (TS prop was added to interface but not destructured)
  - Fixed: pass `isSubmitting` prop + gate button + show "Submitting..." text
  - **Files**: `frontend/src/pages/Backtest/RunSummary.tsx`, `frontend/src/pages/Backtest/BacktestPage.tsx`

- [x] **S23.2** Structured HPO progress display ✅ DONE (2026-05-14)
  - New `pipeline/printer.py` — `HPOProgress` class replaces 550 lines of per-trial noise with ~10 structured lines
  - Progress bar with `\r` in TTY mode, compact fold summaries in Celery (non-TTY) mode
  - `KODAQUANT_VERBOSE=1` re-enables the full firehose for debugging
  - **Files**: `pipeline/printer.py`, `pipeline/backtester/run_mixin.py`

- [x] **S23.3** Per-fold/per-trial noise suppression ✅ DONE (2026-05-14)
  - Suppressed `print_pruned_block_summary` (8-line boxes × 6 call sites)
  - Suppressed `print_block_summary` (per-fold tables), `print_trial_header_table` (50+ param table)
  - Gated `[Eligibility]`, `[Calib][Coverage]`, `[DEBUG][Costs]` behind `KODAQUANT_VERBOSE`
  - Suppressed `[WARN] Skipping fold: Not enough samples` per-trial
  - Gated news/LLM warnings (`use_news=True but no data`) to fire once per instance
  - Suppressed Optuna default `[I ...] Trial N finished` log lines
  - **Files**: `pipeline/backtester/run_mixin.py`, `pipeline/backtester/strategy_mixin.py`, `pipeline/backtester/features_mixin.py`

- [x] **S23.4** `cv_config_override` TypeError fix ✅ DONE (2026-05-14)
  - `_progress_tracking_objective` wrapper didn't accept `cv_config_override` kwarg
  - Every HPO trial crashed with `TypeError: got an unexpected keyword argument 'cv_config_override'`
  - All 10 trials were pruned → "Global HPO failed: No completed Optuna trials"
  - Fixed: add `cv_config_override=None` parameter + forward it to the wrapped objective
  - **Files**: `pipeline/backtester/run_mixin.py`

- [x] **S23.5** Walk-forward equity chart fixes ✅ DONE (2026-05-14)
  - **Scale mismatch**: BH values stored as raw cumulative equity (1.05) while strategy lines converted to percentage (5) — plotted on different Y scales. Fixed: BH goes through same `toChartVal()` conversion.
  - **Zero-height SVG**: `minHeight: 320` doesn't give `ResponsiveContainer` a measurable height → SVG rendered at 0px. Fixed: explicit `height: 320` on wrapper + `height={320}` on `ResponsiveContainer`.
  - **Period-0 origin**: Chart now starts at 0% before period M1 (synthetic origin point added to chartData).
  - **Model grouping**: Per-month summary table now groups by model with 6px separators and model name in brand-colored first column. Added `model` field to `OosPeriodResult`.
  - **Stale-job 404 loop**: `useJobStatus` polled every 2s for stale job IDs from React Query cache. Fixed: return `false` from `refetchInterval` when error is 404.
  - **Files**: `frontend/src/pages/Monitor/EquityChart.tsx`, `frontend/src/api/schemas.ts`, `frontend/src/stores/useJobStore.ts`, `frontend/src/api/queries.ts`

**Sprint 23 complete**: All 127 tests pass (39 backend + 88 frontend). Log noise reduced ~98% during HPO. Live monitor chart shows BH line from period 0, grouped by model with separators.

---

## Sprint 22: Commercial Infrastructure ⬜ DEFERRED FROM S12

> **Goal**: Everything needed to sell and support the product via Paddle.
> **Pricing**: Free trial (14 days) → Pro (£149 one-time + £49/year updates) → Team (£299 + £99/year)
> **Est**: 8-10h
> **Note**: Originally S12, deferred to focus on product intelligence features first.

- [ ] **S22.1** Paddle product & pricing setup
  - Paddle product page: KodaQuant Pro
  - Pricing tiers configured in Paddle dashboard
  - Discount codes for launch promotion
  - License email templates (welcome, renewal reminder)
  - **Est**: 2h

- [ ] **S22.2** Landing page / marketing website
  - Next.js or Astro static site
  - Hero section with demo GIF/video
  - Features grid, pricing table, testimonials placeholder
  - "Buy Now" button → Paddle checkout
  - Blog section for SEO
  - **Files**: `website/` package
  - **Est**: 3h

- [ ] **S22.3** Documentation site
  - MkDocs Material theme
  - Getting started guide
  - API reference (auto-generated from FastAPI schemas)
  - Execution models guide
  - FAQ / troubleshooting
  - **Files**: `docs/` directory
  - **Est**: 2h

- [ ] **S22.4** Legal & compliance
  - Terms of Service (software license agreement)
  - Privacy Policy
  - Disclaimer (not financial advice)
  - EULA for commercial use
  - **Files**: `legal/` directory
  - **Est**: 1h

- [ ] **S22.5** Analytics & monitoring
  - Anonymous usage telemetry (model usage, feature adoption)
  - Download tracking via Paddle webhooks
  - Conversion funnel monitoring (trial → paid)
  - **Files**: `api/middleware/analytics.py`
  - **Est**: 1h