# FX ML Backtester — Product Roadmap

> **Last Updated**: 2026-04-17
> **Branch**: `feature/sprint2-execution-models`
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

- [ ] **S2.5** Execution model integration
  - Wire execution models into `execution_patches.py`
  - Add execution config to UI sidebar
  - Execution model selection dropdown in backtest config
  - Metrics breakdown: gross vs net, impact of each cost component
  - **Files**: `pipeline/backtester/execution_patches.py`, `ui/controls.py`
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

- [ ] **S3.3** Cross-pair comparison UI
  - Pair selector dropdown in UI
  - Cross-pair equity curve overlay
  - Heatmap: model × pair Sharpe ratio grid
  - **Files**: `ui/controls.py`, `ui/charts.py`
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

## Sprint 5: Comprehensive Tests + Benchmarks ⬜ NOT STARTED

> **Goal**: Bulletproof confidence in results. 200+ tests, mutation testing, benchmarks.
> **Maps to**: Phase 11
> **Est**: 4-6h

- [ ] **S5.1** AI-generated unit tests for pipeline/
  - Target >80% line coverage for all pipeline modules
  - **Files**: `tests/test_pipeline_*.py`
  - **Est**: 2h

- [ ] **S5.2** AI-generated unit tests for models/
  - Test all 8 model types: train, predict, shape validation
  - **Files**: `tests/test_models_*.py`
  - **Est**: 1h

- [ ] **S5.3** Performance benchmarks
  - Benchmark backtest duration per model
  - Detect speed regressions
  - Memory usage profiling
  - **Files**: new `tests/benchmarks/`
  - **Est**: 1h

- [ ] **S5.4** Golden output regression tests
  - Golden output files for known configs
  - Compare new results against golden
  - Alert on unexpected metric changes
  - **Files**: `tests/golden/`
  - **Est**: 1h

---

## Sprint 6: News & Sentiment Features ⬜ NOT STARTED

> **Goal**: News-derived features for smarter backtesting. Unique differentiator.
> **Maps to**: Phase 10
> **Est**: 6-8h

- [ ] **S6.1** News scraper
  - RSS feeds (Reuters, Bloomberg, FX-specific)
  - NewsAPI integration
  - Economic calendar (NFP, FOMC, CPI)
  - Rate-limited, cached, deduplicated
  - **Files**: new `news/scraper.py`
  - **Est**: 2h

- [ ] **S6.2** Sentiment analysis
  - finBERT integration for financial sentiment
  - VADER for quick scoring
  - Per-article sentiment score → daily/hourly aggregation
  - **Files**: new `news/sentiment.py`
  - **Est**: 2h

- [ ] **S6.3** News-derived features in pipeline
  - Sentiment score as input feature
  - News volume as volatility proxy
  - Event flags (pre-NFP, post-FOMC)
  - Walk-forward safe (only past news used)
  - **Files**: `pipeline/backtester/features_mixin.py`, `news/`
  - **Est**: 2h

- [ ] **S6.4** News overlay on charts
  - Mark major events on equity curve
  - Event impact analysis
  - **Files**: `ui/charts.py`
  - **Est**: 1h

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

## Sprint 8: React Frontend ⬜ NOT STARTED

> **Goal**: Professional desktop-grade React UI that replaces Streamlit as the user-facing product.
> **Architecture**: Vite + TypeScript + React 18 + TailwindCSS + shadcn/ui
> **Est**: 20-25h

- [ ] **S8.1** React project scaffold
  - Vite + TypeScript + React 18
  - TailwindCSS + shadcn/ui component library
  - React Router for multi-page navigation
  - API client layer (axios + WebSocket hook)
  - **Files**: `frontend/` package
  - **Est**: 2h

- [ ] **S8.2** Layout shell & navigation
  - Sidebar nav (dashboard, backtest, results, compare, settings)
  - Dark/light mode toggle with system preference detection
  - Responsive layout (works in Electron window 1280x800+)
  - System tray integration stub
  - **Files**: `frontend/src/layout/`
  - **Est**: 2h

- [ ] **S8.3** Dashboard page
  - KPI cards (total backtests, best Sharpe, avg win rate, equity curve thumbnail)
  - Recent backtests table with quick actions
  - Model performance heatmap (model x pair Sharpe grid)
  - **Files**: `frontend/src/pages/Dashboard/`
  - **Est**: 3h

- [ ] **S8.4** Backtest configuration page
  - Model selector (multi-select for comparison)
  - Currency pair + timeframe dropdowns
  - Execution model config (position sizing, stops, risk manager)
  - Feature toggles panel
  - HPO settings
  - "Run Backtest" button with progress bar via WebSocket
  - **Files**: `frontend/src/pages/Backtest/`
  - **Est**: 4h

- [ ] **S8.5** Results & charts page
  - Equity curve (Plotly.js — interactive, zoomable, crosshair)
  - KPI cards grid (Sharpe, Sortino, max DD, total return, win rate, profit factor)
  - Monthly returns heatmap
  - Trade log table (sortable, filterable)
  - Export buttons (CSV, PNG, JSON)
  - **Files**: `frontend/src/pages/Results/`
  - **Est**: 5h

- [ ] **S8.6** Model comparison page
  - Side-by-side equity curve overlay
  - Leaderboard table (sortable by any metric)
  - Paired t-test significance indicators
  - Parameter sensitivity chart
  - **Files**: `frontend/src/pages/Compare/`
  - **Est**: 3h

- [ ] **S8.7** Settings page
  - Pipeline config editor (JSON editor with validation)
  - GPU/compute settings
  - Data source management (OANDA API key, pair download)
  - License activation UI (for S10 integration)
  - **Files**: `frontend/src/pages/Settings/`
  - **Est**: 2h

- [ ] **S8.8** Error handling & loading states
  - Global error boundary with user-friendly messages
  - Loading skeletons for all data-fetching components
  - Toast notifications for success/error/warning
  - Retry logic for failed API calls
  - **Files**: `frontend/src/components/ErrorBoundary/`, `frontend/src/hooks/`
  - **Est**: 2h

---

## Sprint 9: Electron Desktop Shell ⬜ NOT STARTED

> **Goal**: Wrap React + FastAPI into a native-feeling desktop application.
> **Architecture**: Electron main process + Python subprocess + React BrowserWindow
> **Est**: 10-12h

- [ ] **S9.1** Electron scaffold
  - `electron/` directory with main process
  - BrowserWindow config (min size 1280x800, frameless option)
  - Dev vs production mode detection
  - **Files**: `electron/main.ts`, `electron/preload.ts`
  - **Est**: 2h

- [ ] **S9.2** Python backend lifecycle management
  - Spawn FastAPI subprocess from Electron main process
  - Port discovery (find available localhost port)
  - Health check polling until backend is ready
  - Graceful shutdown (SIGTERM → wait → SIGKILL)
  - Log forwarding from Python to Electron console
  - **Files**: `electron/python.ts`
  - **Est**: 3h

- [ ] **S9.3** Native menus & system tray
  - Application menu (File, Edit, View, Backtest, Help)
  - System tray icon with status (running, backtesting, error)
  - Tray context menu (show window, run backtest, quit)
  - **Files**: `electron/tray.ts`, `electron/menu.ts`
  - **Est**: 2h

- [ ] **S9.4** PyInstaller integration
  - `forex_pipeline.spec` — bundle Python + all dependencies
  - Include React build as static assets served by FastAPI
  - Single-directory bundle (not --onefile, for faster startup)
  - Test on clean Windows 10/11 VM
  - **Files**: `forex_pipeline.spec`, `scripts/build_python.bat`
  - **Est**: 3h

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

## Superseded Phases

The following phases from the original roadmap are **superseded** by the desktop app strategy (Sprints 7-13):

| Old Phase | Status | Replaced By |
|-----------|--------|-------------|
| Phase 5 (Streamlit Cloud) | Partial — Streamlit stays as dev tool | Not deployed to cloud; React desktop app is the product |
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
# Option 1: Double-click launch_ui.bat (Windows)

# Option 2: PowerShell
launch_ui.bat

# Option 3: Manual WSL
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_wsl.sh \
    python -m streamlit run app.py --server.headless true

# Then open: http://localhost:8501
```

## Key Files

| File | Role |
|------|------|
| `app.py` | Streamlit entry point (dev UI, interim) |
| `ui/controls.py` | Nav bar + 6-tab layout + GPU warnings |
| `ui/state.py` | AppState + backtest adapter |
| `ui/results.py` | Results display + export |
| `ui/dashboard.py` | Dashboard renderer |
| `ui/charts.py` | Plotly chart builders |
| `pipeline/backtester/composed.py` | MLBacktester engine |
| `pipeline/runtime.py` | GPU detection, thread budgets, CUDA config |
| `config.py` | Global configuration |
| `models/registry.py` | Model registry |
| `pipeline/execution/position_sizing.py` | Position sizing models (S2.1) |
| `pipeline/backtester/execution_patches.py` | Execution loop + patches |
| `run_smoke_gpu.bat` | GPU smoke test launcher (Windows → WSL) |
| `run_smoke_gpu.sh` | GPU smoke test script (WSL + CUDA) |
| `pipeline/model_comparison.py` | Model comparison & leaderboard module |
| `run_comparison.bat` | One-command multi-model comparison runner |

## Full Sprint Sequence

| Sprint | Topic | Est | Status |
|--------|-------|-----|--------|
| **S1** | Model Comparison & Leaderboard | 3-4h | DONE |
| **S2** | Advanced Execution Models | 6-8h | DONE |
| **S3** | Multi-Currency Expansion | 4-5h | DONE |
| **S4** | Docker + CI/CD | 3-4h | PARTIAL (RAM opt done) |
| **S5** | Comprehensive Tests + Benchmarks | 4-6h | TODO |
| **S6** | News & Sentiment Features | 6-8h | TODO |
| **S7** | FastAPI Backend | 8-10h | DONE |
| **S8** | React Frontend | 20-25h | TODO |
| **S9** | Electron Desktop Shell | 10-12h | TODO |
| **S10** | Security & Licensing (Paddle) | 12-15h | TODO |
| **S11** | Installer & Auto-Update | 6-8h | TODO |
| **S12** | Commercial Infrastructure | 8-10h | TODO |
| **S13** | Beta & Launch | 6-8h | TODO |

## Completion Criteria Summary

| Sprint | When it's done |
|--------|---------------|
| S1-S2 | Execution models complete, all sizing/stop/risk models work |
| S3-S4 | Multi-currency supported, Docker builds pass, CI green |
| S5-S6 | 200+ tests, >80% coverage, news features integrated |
| S7 | FastAPI serves all pipeline operations via REST + WebSocket |
| S8 | React UI replaces Streamlit for all user interactions |
| S9 | Electron wraps React + Python into desktop app |
| S10 | Code protected, Paddle licensing active, feature gating works |
| S11 | Windows installer + auto-update functional |
| S12 | Product listed on Paddle, landing page live, docs published |
| S13 | Beta tested, publicly launched, first sales |