# FX ML Backtester — Product Roadmap

> **Last Updated**: 2026-04-20
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

## Sprint 14: Pipeline Enhancements

> **Goal**: Add flexible walk-forward periods and HPO duration control.
> **Est**: 5-8h

- [ ] **S14.1** Daily/Weekly walk-forward periods
  - Add `period_unit: Literal["months", "weeks", "days"]` to config/schemas
  - Create `_make_offset(count, unit)` helper replacing all `DateOffset(months=...)`
  - Generalize `months_between()` in `evaluation_mixin.py` to handle days/weeks
  - Convert 37-month warm-up offset proportionally (≈160 weeks or 1120 days)
  - Generalize `pd.to_period("M")` in CV fold builder
  - Update frontend schemas + UI controls for period unit selector
  - **Files**: `config.py`, `schemas/`, `pipeline/backtester/real_trading_mixin.py`, `pipeline/backtester/evaluation_mixin.py`, `pipeline/backtester/run_mixin.py`, `frontend/src/pages/Backtest/`
  - **Est**: 3-5h + 2-3h tests

- [ ] **S14.2** HPO duration control
  - Add `max_hpo_duration_minutes` config option
  - Implement early stopping in Optuna when time budget exceeded
  - API endpoint to set duration limit per job
  - Frontend control in Backtest config page
  - **Files**: `config.py`, `pipeline/tuning/runner.py`, `api/routers/backtest.py`, `frontend/src/pages/Backtest/`
  - **Est**: 2-3h

---

## Future Investigation: Daily/Weekly Walk-Forward Periods (MOVED TO S14.1)

> **Status**: Not currently supported. Pipeline is hardcoded to monthly periods.
> **Effort**: Medium refactor (3-5 hours core + 2-3 hours tests)

The walk-forward engine uses `pd.DateOffset(months=...)` in 7 call sites across 3 files:
- `pipeline/backtester/real_trading_mixin.py` (lines 762-765, 777, 793, 1022)
- `pipeline/backtester/evaluation_mixin.py` (lines 41, 45, 49, 52)
- `pipeline/backtester/run_mixin.py` (lines 113, 540, 3791-3792)

**What would be needed:**
1. Add `period_unit: Literal["months", "weeks", "days"]` to config/schemas
2. Create `_make_offset(count, unit)` helper replacing all `DateOffset(months=...)`
3. Generalize `months_between()` in `evaluation_mixin.py` to handle days/weeks
4. Convert 37-month warm-up offset proportionally (≈160 weeks or 1120 days)
5. Generalize `pd.to_period("M")` in CV fold builder
6. Update frontend schemas + UI controls

**Candidate for**: Sprint 14 or post-launch enhancement.

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

- [ ] **S9.5** Electron build pipeline
  - electron-builder config for Windows (.exe, .msi)
  - Code signing (self-signed for dev, proper cert for production)
  - Include bundled Python as extraResource
  - NSIS installer customization
  - **Files**: `electron-builder.yml`, `scripts/build_electron.bat`
  - **Est**: 2h

---

## Sprint 10: Security & Licensing ⬜ NOT STARTED

> **Goal**: Protect source code and implement Paddle-based license key validation.
> **Pricing model**: Hybrid — one-time purchase + annual updates subscription
> **Est**: 12-15h

- [ ] **S10.1** Code protection
  - Critical pipeline modules compiled with Cython (`.py` → `.pyd`)
  - PyInstaller with `--key` (AES encryption of bytecode)
  - Strip docstrings and debug symbols from production build
  - Anti-debugging checks in launcher
  - **Files**: `setup_cython.py`, `scripts/build_secure.bat`
  - **Est**: 4h

- [ ] **S10.2** Paddle SDK integration
  - Paddle seller account setup + product configuration
  - License key generation via Paddle API
  - License activation endpoint (online verification)
  - Offline grace period (7 days without internet)
  - **Files**: `api/licensing/paddle.py`, `api/licensing/__init__.py`
  - **Est**: 3h

- [ ] **S10.3** Machine fingerprinting
  - Hardware ID generation (CPU serial + MAC + disk serial + motherboard)
  - License binding to machine (one license = one machine, configurable)
  - Machine transfer flow (deactivate old, activate new)
  - **Files**: `api/licensing/fingerprint.py`
  - **Est**: 2h

- [ ] **S10.4** License enforcement in app
  - License check on startup (blocking — show activation dialog if unlicensed)
  - Trial mode (14-day full access, then restricted)
  - Feature gating (free: 3 models + basic execution; paid: all models + advanced execution)
  - License status in Settings page
  - **Files**: `electron/license.ts`, `frontend/src/components/LicenseDialog/`
  - **Est**: 2h

- [ ] **S10.5** Encrypted local storage
  - Encrypted SQLite for user settings, backtest history, license state
  - Key derivation from machine fingerprint + app secret
  - Secure credential storage (OANDA API key, etc.)
  - **Files**: `api/storage.py`
  - **Est**: 2h

- [ ] **S10.6** Security audit
  - Dependency vulnerability scan (`pip audit`, `npm audit`)
  - Input validation on all API endpoints
  - Rate limiting on API routes
  - No secrets in compiled binary
  - **Files**: `api/middleware/security.py`
  - **Est**: 2h

---

## Sprint 11: Installer & Auto-Update ⬜ NOT STARTED

> **Goal**: Professional Windows installer with automatic updates.
> **Est**: 6-8h

- [ ] **S11.1** Windows installer (Inno Setup / NSIS)
  - Custom installer wizard with branding
  - Start menu shortcut + desktop shortcut
  - File association (.fxbacktest for sharing configs)
  - Uninstaller with "keep data" option
  - Silent install option for enterprise deployment
  - **Files**: `installer/setup.iss`
  - **Est**: 3h

- [ ] **S11.2** Auto-update system
  - GitHub Releases as update source
  - Version check on startup
  - Differential updates (patch only changed files)
  - Update notification in UI (download + install button)
  - Rollback mechanism if update fails
  - **Files**: `electron/updater.ts`, `frontend/src/components/UpdateNotification/`
  - **Est**: 3h

- [ ] **S11.3** Crash reporting
  - Sentry integration (Electron + Python error capture)
  - User opt-in for anonymous crash reports
  - Breadcrumbs for debugging
  - **Files**: `electron/sentry.ts`, `api/middleware/error_tracking.py`
  - **Est**: 1h

- [ ] **S11.4** First-run experience
  - Welcome wizard (set data path, configure OANDA API key, choose default pair)
  - Quick start guide / interactive tutorial
  - Sample backtest with pre-loaded data (demo mode)
  - **Files**: `frontend/src/pages/Welcome/`
  - **Est**: 1h

---

## Sprint 12: Commercial Infrastructure ⬜ NOT STARTED

> **Goal**: Everything needed to sell and support the product via Paddle.
> **Pricing**: Free trial (14 days) → Pro (£149 one-time + £49/year updates) → Team (£299 + £99/year)
> **Est**: 8-10h

- [ ] **S12.1** Paddle product & pricing setup
  - Paddle product page: FX Backtester Pro
  - Pricing tiers configured in Paddle dashboard
  - Discount codes for launch promotion
  - License email templates (welcome, renewal reminder)
  - **Est**: 2h

- [ ] **S12.2** Landing page / marketing website
  - Next.js or Astro static site
  - Hero section with demo GIF/video
  - Features grid, pricing table, testimonials placeholder
  - "Buy Now" button → Paddle checkout
  - Blog section for SEO
  - **Files**: `website/` package
  - **Est**: 3h

- [ ] **S12.3** Documentation site
  - MkDocs Material theme
  - Getting started guide
  - API reference (auto-generated from FastAPI schemas)
  - Execution models guide
  - FAQ / troubleshooting
  - **Files**: `docs/` directory
  - **Est**: 2h

- [ ] **S12.4** Legal & compliance
  - Terms of Service (software license agreement)
  - Privacy Policy
  - Disclaimer (not financial advice)
  - EULA for commercial use
  - **Files**: `legal/` directory
  - **Est**: 1h

- [ ] **S12.5** Analytics & monitoring
  - Anonymous usage telemetry (model usage, feature adoption)
  - Download tracking via Paddle webhooks
  - Conversion funnel monitoring (trial → paid)
  - **Files**: `api/middleware/analytics.py`
  - **Est**: 1h

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

## Sprint 15: KodaQuant Branding

> **Goal**: Rename and rebrand from "FX ML Backtester / thesisproj" to "KodaQuant" for commercial launch.
> **Est**: 4-6h

- [ ] **S15.1** App name update
  - Rename product references from "FX ML Backtester" to "KodaQuant" in all user-facing places
  - Update: `electron-builder.yml` (productName, shortcutName), `forex_pipeline.spec`, `run_server.py` print messages, API startup banner, frontend title/meta
  - GitHub repo name stays as-is (internal)
  - **Est**: 1h

- [ ] **S15.2** UI theme & color scheme
  - Define KodaQuant color palette (primary, accent, background)
  - Update TailwindCSS theme config with KodaQuant branding colors
  - Update logo SVG and favicon
  - Dark mode as default (professional trading aesthetic)
  - **Files**: `frontend/tailwind.config.ts`, `frontend/src/index.css`, `frontend/public/favicon.svg`, `frontend/public/favicon.ico`
  - **Est**: 2h

- [ ] **S15.3** Splash screen & installer branding
  - Custom splash screen with KodaQuant logo + progress bar
  - NSIS installer welcome/finish pages with branding
  - App icon set (desktop, taskbar, installer)
  - **Files**: `electron/main.ts` (splash), `electron-builder.yml`, `build/icon.*`
  - **Est**: 1h

- [ ] **S15.4** Documentation & about screen
  - About dialog with version, license info, links
  - README update with KodaQuant branding
  - **Est**: 1h

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

# Start FastAPI backend (port 8001)
cd api && uvicorn main:app --host 127.0.0.1 --port 8001 --reload

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
| `api/schemas/news.py` | News event schemas |
| `frontend/src/pages/Backtest/` | Backtest config UI (9 components) |
| `frontend/src/pages/Results/` | Results display + charts (7 components) |
| `frontend/src/pages/Compare/` | Model comparison + cross-pair (5 components) |
| `frontend/src/pages/Dashboard/` | Dashboard + KPIs + heatmap + thumbnails |
| `frontend/src/components/charts/` | All chart components (10 files) |
| `frontend/src/api/websocket.ts` | WebSocket manager for real-time progress |
| `frontend/src/api/queries.ts` | React Query hooks for all API endpoints |
| `pipeline/backtester/composed.py` | MLBacktester engine |
| `pipeline/backtester/data_mixin.py` | Data loading + date range handling |
| `pipeline/backtester/real_trading_mixin.py` | Real trading simulation + walk-forward |
| `pipeline/runtime.py` | GPU detection, thread budgets, CUDA config |
| `config.py` | Global configuration |
| `models/registry.py` | Model registry |
| `pipeline/execution/position_sizing.py` | Position sizing models |
| `pipeline/model_comparison.py` | Model comparison & leaderboard |

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
| **S9** | Electron Desktop Shell | 10-12h | S9.1-4 ✅ DONE; S9.5 remaining |
| **S10** | Security & Licensing (Paddle) | 12-15h | TODO |
| **S11** | Installer & Auto-Update | 6-8h | TODO |
| **S12** | Commercial Infrastructure | 8-10h | TODO |
| **S13** | Beta & Launch | 6-8h | TODO |
| **S14** | Pipeline Enhancements (daily WF, HPO duration) | 5-8h | TODO |
| **S15** | KodaQuant Branding | 4-6h | TODO |

## Completion Criteria Summary

| Sprint | When it's done |
|--------|---------------|
| S1-S2 | Execution models complete, all sizing/stop/risk models work |
| S3-S4 | Multi-currency supported, Docker builds pass, CI green |
| S5-S6 | 200+ tests, >80% coverage, news features integrated | ✅ BOTH DONE — 496+ tests
| S7 | FastAPI serves all pipeline operations via REST + WebSocket | ✅ DONE
| S8 | React UI replaces Streamlit for all user interactions | ✅ DONE — all 8 sub-tasks complete |
| S9 | Electron wraps React + Python into desktop app | 🔄 S9.1-3 scaffolded; S9.4-5 remaining |
| S10 | Code protected, Paddle licensing active, feature gating works |
| S11 | Windows installer + auto-update functional |
| S12 | Product listed on Paddle, landing page live, docs published |
| S13 | Beta tested, publicly launched, first sales |