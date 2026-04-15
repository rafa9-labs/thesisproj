# FX ML Backtester — Product Roadmap

> **Last Updated**: 2026-04-15
> **Branch**: `feature/phase4-streamlit-ui`
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

## Phase 4: Feature Parity with `init-proj` 🔄 IN PROGRESS

> **Goal**: Backport all features from the old `init-proj` codebase so nothing is lost.

- [ ] **4.1** Audit `init-proj` feature set
  - Document all indicators in `init-proj/src/features/*` vs current pipeline
  - Document all execution models in `init-proj/src/execution/*` vs current
  - Document all UI features in `init-proj/src/ui/*` vs current `ui/`
  - Create gap analysis spreadsheet
  - **Est**: 2h

- [ ] **4.2** Backport missing indicators
  - Vectorized indicator implementations from `init-proj/src/features/`
  - Any missing TA-Lib wrappers, custom indicators
  - Add toggle controls in UI Features tab
  - **Files**: `pipeline/backtester/deep_mixin.py`, `ui/controls.py`
  - **Est**: 4h

- [ ] **4.3** Backport execution models
  - Risk management from `init-proj/src/execution/`
  - Position sizing models (fixed fractional, Kelly, etc.)
  - Stop-loss / take-profit management
  - Trailing stops
  - **Files**: `pipeline/backtester/execution_patches.py`, new `pipeline/execution/`
  - **Est**: 6h

- [ ] **4.4** Backport missing UI features
  - Any missing dashboards, charts, comparison views
  - Model comparison (side-by-side equity curves)
  - Parameter sensitivity analysis
  - **Files**: `ui/charts.py`, `ui/results.py`, `ui/controls.py`
  - **Est**: 4h

- [ ] **4.5** Verify all model types work end-to-end 🔄 MOSTLY COMPLETE (2026-04-15)
  - ✅ Logistic — PASS (31s, 4 trades)
  - ✅ XGBoost — PASS (22s, 6 trades)
  - ✅ CNN — PASS (42s, 60 trades)
  - ✅ LSTM — PASS (76s, 60 trades)
  - ✅ Transformer — PASS (48s, 60 trades)
  - ✅ Ensemble (CNN-LSTM-XGBoost) — PASS (100s, 64 trades)
  - ✅ Ensemble (adaptive regime) — PASS (28s, 54 trades) — `cv='prefit'` fixed with `FrozenEstimator`
  - ✅ DQN — `RewardProcessWrapper`/`TradingEnv`/`CostAwareWrapper` imports fixed in `dqn_mixin.py`; import chain verified, training starts and runs correctly; very slow on CPU (~70+ min for 20 eps × 30K steps with walk-forward folds) — functional but recommended to run on GPU
  - Smoke test enhanced: detects silent failures (0 trades = FAIL, not PASS)
  - Each model: train → predict → trade → metrics → display
  - **Files**: `models/*.py`, `pipeline/_imports.py`, `pipeline/backtester/dqn_mixin.py`, `models/ensemble_adaptive_regime.py`, `tests/smoke_all_models.py`
  - **Est**: 4h ✅ DONE

**Phase 4 complete when**: All `init-proj` features are present, all model types run end-to-end without errors.

---

## Phase 5: UI Polish & Streamlit Cloud Deploy 🔄 ~75%

> **Goal**: Professional, elegant UI that works flawlessly. Ready for public access.

- [ ] **5.1** End-to-end smoke test 🔄 PARTIAL (2026-04-15)
  - ✅ UI loads in browser, sidebar renders, all 6 tabs functional (Puppeteer verified)
  - ✅ Logistic model runs end-to-end via smoke test
  - ✅ XGBoost model runs end-to-end via smoke test
  - ✅ CNN/LSTM/Transformer — all pass on CPU (42s/76s/48s)
  - ✅ Ensemble models (CNN-LSTM-XGBoost, adaptive regime) — both pass
  - ✅ DQN — import fix verified, training runs correctly (slow on CPU ~70+ min)
  - ⬜ Full backtest from UI with equity curves + KPI cards + monthly breakdown
  - ⬜ HPO diagnostics display verification
  - **Files**: `tests/smoke_all_models.py`
  - **Est**: 2h (0.5h remaining)

- [ ] **5.2** Cancellation support
  - Add "Stop" button during long backtests
  - Use `threading` + `st.stop()` pattern
  - Clean up resources on cancellation
  - **Files**: `app.py`, `ui/state.py`
  - **Est**: 2h

- [ ] **5.3** UI polish & error states
  - Loading spinners for all async operations
  - Graceful error messages (no raw tracebacks to user)
  - Responsive layout (mobile-friendly)
  - Consistent spacing, fonts, color scheme
  - Dark/light mode support
  - **Files**: `ui/*.py`, `app.py`
  - **Est**: 4h

- [ ] **5.4** Export verification
  - CSV download: metrics + monthly breakdown
  - PNG download: equity curve chart
  - JSON download: best configuration
  - PDF report generation (optional)
  - **Files**: `ui/results.py`
  - **Est**: 2h

- [ ] **5.5** Deploy to Streamlit Cloud
  - Create `.streamlit/secrets.toml` template
  - Create `packages.txt` for system dependencies
  - Push to Streamlit Cloud
  - Configure custom domain (optional)
  - **Est**: 1h

**Phase 5 complete when**: App is live on Streamlit Cloud, all features work in browser, export downloads work.

---

## Phase 6: Security, Auth & User Database ⬜ NOT STARTED

> **Goal**: User accounts, secure data handling, rate limiting. Production-ready security.

- [ ] **6.1** User authentication (streamlit-authenticator)
  - Registration flow (email + password)
  - Login/logout flow
  - Password hashing (bcrypt)
  - Session management with cookies
  - **Files**: new `ui/auth.py`, `ui/state.py`
  - **Est**: 4h

- [ ] **6.2** User database (SQLite → PostgreSQL)
  - SQLite for local/dev, PostgreSQL for production
  - Schema: users, backtest_results, user_configs
  - Migration scripts (Alembic)
  - **Files**: new `db/` package, `db/models.py`, `db/migrations/`
  - **Est**: 6h

- [ ] **6.3** Results persistence per user
  - Save backtest results to DB after each run
  - Load previous results from DB (not just filesystem)
  - User dashboard: history of all past runs
  - **Files**: `ui/results.py`, `ui/state.py`, `db/`
  - **Est**: 4h

- [ ] **6.4** Security audit
  - Input validation on all user inputs (SQL injection, XSS)
  - CSRF protection
  - Rate limiting (per-user backtest throttling)
  - `.env` secrets management (no hardcoded keys)
  - Dependency vulnerability scan (`pip audit`)
  - **Files**: `ui/validators.py`, `.env.example`, `requirements_freeze.txt`
  - **Est**: 4h

- [ ] **6.5** Free tier limits
  - 3 backtests/day for free users
  - 3 models max for free users
  - Usage tracking in DB
  - Upgrade prompt when limits hit
  - **Files**: `ui/controls.py`, `db/`
  - **Est**: 3h

**Phase 6 complete when**: Users can register, login, run backtests (with limits), results persist across sessions.

---

## Phase 7: Deployability & Infrastructure ⬜ NOT STARTED

> **Goal**: One-click deploy, CI/CD, monitoring. Ops-ready.

- [ ] **7.1** Dockerfile + docker-compose
  - Reproducible Python environment
  - GPU passthrough config for deep models
  - Multi-stage build (slim image)
  - **Files**: new `Dockerfile`, `docker-compose.yml`
  - **Est**: 3h

- [ ] **7.2** CI/CD pipeline (GitHub Actions)
  - Lint (ruff/flake8) on every PR
  - Unit tests on every push
  - Integration tests on merge to main
  - Auto-deploy to Streamlit Cloud / AWS
  - **Files**: new `.github/workflows/ci.yml`
  - **Est**: 3h

- [ ] **7.3** Health checks & monitoring
  - `/health` endpoint
  - Uptime monitoring (UptimeRobot or similar)
  - Error tracking (Sentry integration)
  - Performance metrics (backtest duration, memory usage)
  - **Files**: new `monitoring/`
  - **Est**: 2h

- [ ] **7.4** Backup strategy
  - Automated daily DB backups
  - Backup rotation (keep last 30 days)
  - Restore procedure documented
  - **Est**: 2h

- [ ] **7.5** SSL/TLS & domain setup
  - Custom domain with HTTPS
  - SSL certificate (Let's Encrypt)
  - DNS configuration
  - **Est**: 1h

- [ ] **7.6** Auto-scaling strategy
  - Document scaling plan for traffic spikes
  - Queue system for backtest jobs (Celery + Redis)
  - Horizontal scaling considerations
  - **Est**: 2h

**Phase 7 complete when**: `docker-compose up` launches the full stack, CI/CD runs green, monitoring alerts work.

---

## Phase 8: Multi-User Platform & Monetization ⬜ NOT STARTED

> **Goal**: Scale to many users, Stripe billing, team features.

- [ ] **8.1** FastAPI backend
  - Decouple backtest engine from Streamlit
  - REST API with versioned endpoints (`/api/v1/`)
  - API key authentication
  - Async job queue for long-running backtests
  - **Files**: new `api/` package
  - **Est**: 8h

- [ ] **8.2** React/Next.js frontend
  - Professional SPA replacing Streamlit for scale
  - Responsive design (mobile + desktop)
  - Real-time backtest progress (WebSocket)
  - **Files**: new `frontend/` package
  - **Est**: 2 weeks

- [ ] **8.3** Redis caching layer
  - Cache hot results (popular backtests)
  - Session store
  - Rate limiting backend
  - **Est**: 3h

- [ ] **8.4** User isolation & sandboxing
  - Per-user data isolation
  - Per-user resource limits (CPU, memory, time)
  - User workspace separation
  - **Est**: 4h

- [ ] **8.5** Team/org support
  - Shared configs within teams
  - Team dashboard with combined results
  - Role-based access (admin, member, viewer)
  - **Est**: 6h

- [ ] **8.6** Stripe billing integration
  - Free tier: 3 backtests/day, 3 models, no export
  - Pro tier: £19/mo — unlimited, all models, full export
  - Team tier: £49/mo — Pro + shared configs, team dashboard
  - API tier: £99/mo — REST API, webhooks, white-label
  - **Files**: new `billing/` package
  - **Est**: 6h

- [ ] **8.7** White-label / embed support
  - Embeddable widgets for partners
  - Custom branding options
  - API documentation portal
  - **Est**: 4h

**Phase 8 complete when**: Multiple users can run backtests simultaneously, billing works, teams can collaborate.

---

## Phase 9: Python SDK & Distribution ⬜ NOT STARTED

> **Goal**: `pip install fxbacktester` — secondary revenue stream.

- [ ] **9.1** Package structure
  - `pyproject.toml` with proper metadata
  - Clean public API: `MLBacktester`, `Config`, `Results`
  - **Files**: new `fxbacktester/` package
  - **Est**: 4h

- [ ] **9.2** Jupyter notebooks (5 examples)
  - Quick start: basic backtest
  - Multi-model comparison
  - Custom feature engineering
  - Walk-forward analysis deep dive
  - Exporting & reporting
  - **Files**: new `notebooks/`
  - **Est**: 4h

- [ ] **9.3** API documentation (MkDocs)
  - Full API reference
  - Getting started guide
  - Advanced usage docs
  - Deploy to ReadTheDocs
  - **Files**: new `docs/`
  - **Est**: 4h

- [ ] **9.4** PyPI publish
  - Test on TestPyPI first
  - Automated publishing via CI/CD
  - Version management (semver)
  - **Est**: 2h

- [ ] **9.5** License key gating
  - Premium models behind license key
  - License validation server
  - Graceful degradation for free users
  - **Est**: 4h

**Phase 9 complete when**: `pip install fxbacktester` works, docs are live, notebooks run.

---

## Phase 10: News & Sentiment Integration ⬜ NOT STARTED

> **Goal**: News-derived features for smarter backtesting. Event-driven analysis.

- [ ] **10.1** News scraper
  - RSS feeds (Reuters, Bloomberg, FX-specific)
  - NewsAPI integration
  - Economic calendar scraping (NFP, FOMC, CPI)
  - Rate-limited, cached, deduplicated
  - **Files**: new `news/` package
  - **Est**: 6h

- [ ] **10.2** Sentiment analysis
  - finBERT integration for financial sentiment
  - VADER for quick scoring
  - Per-article sentiment score
  - Aggregated daily/hourly sentiment features
  - **Files**: new `news/sentiment.py`
  - **Est**: 4h

- [ ] **10.3** News-derived features in pipeline
  - Sentiment score as input feature
  - News count/volume as volatility proxy
  - Event flags (pre-NFP, post-FOMC)
  - Integrate into walk-forward (only past news used)
  - **Files**: `pipeline/backtester/deep_mixin.py`, `news/`
  - **Est**: 4h

- [ ] **10.4** News overlay on charts
  - Mark major events on equity curve
  - News timeline alongside backtest results
  - Event impact analysis (how did NFP affect this strategy?)
  - **Files**: `ui/charts.py`, `ui/results.py`
  - **Est**: 3h

- [ ] **10.5** Event-driven backtesting
  - Filter backtest to only trade around major events
  - Event-strategy comparison
  - Calendar-aware walk-forward
  - **Est**: 4h

**Phase 10 complete when**: News features are available in the UI, sentiment scores improve model metrics.

---

## Phase 11: Automated AI Testing ⬜ NOT STARTED

> **Goal**: AI-generated comprehensive test suite. Mutation testing. Coverage enforcement.

- [ ] **11.1** AI-generated unit tests
  - Use AI to generate tests for all `pipeline/` modules
  - Use AI to generate tests for all `models/` modules
  - Use AI to generate tests for all `ui/` modules
  - Target: >80% line coverage
  - **Files**: `tests/test_*.py`
  - **Est**: 6h

- [ ] **11.2** Integration test harness
  - End-to-end: model selection → train → predict → trade → metrics → UI render
  - Test all model types in sequence
  - Test all data files (EURUSD H1, H4, M30)
  - **Files**: `tests/test_integration_full.py`
  - **Est**: 4h

- [ ] **11.3** Mutation testing
  - Use `mutmut` to verify tests catch real bugs
  - Target: >85% mutation kill rate
  - Fix weak tests that miss mutations
  - **Est**: 3h

- [ ] **11.4** Coverage enforcement in CI
  - Fail CI if coverage drops below 80%
  - Coverage badge in README
  - Per-module coverage reporting
  - **Files**: `.github/workflows/ci.yml`
  - **Est**: 2h

- [ ] **11.5** Performance benchmarks
  - Benchmark backtest duration per model
  - Detect speed regressions in CI
  - Memory usage profiling
  - **Files**: new `tests/benchmarks/`
  - **Est**: 3h

- [ ] **11.6** Regression test suite
  - Golden output files for known configs
  - Compare new results against golden
  - Alert on unexpected metric changes
  - **Files**: `tests/golden/`
  - **Est**: 3h

**Phase 11 complete when**: CI runs 200+ tests in <5min, coverage >80%, mutation kill rate >85%.

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
| `app.py` | Streamlit entry point |
| `ui/controls.py` | Nav bar + 6-tab layout |
| `ui/state.py` | AppState + backtest adapter |
| `ui/results.py` | Results display + export |
| `ui/dashboard.py` | Dashboard renderer |
| `ui/charts.py` | Plotly chart builders |
| `pipeline/backtester/composed.py` | MLBacktester engine |
| `config.py` | Global configuration |
| `models/registry.py` | Model registry |

## Completion Criteria Summary

| Phase | When it's done |
|-------|---------------|
| 3 | No data leaks, features cached, tests pass |
| 4 | All `init-proj` features present, all models work |
| 5 | App live on cloud, export works, UI polished |
| 6 | Users can register/login, results persist, rate-limited |
| 7 | Docker one-click deploy, CI green, monitoring active |
| 8 | Multi-user, Stripe billing, team features |
| 9 | `pip install fxbacktester` works |
| 10 | News features improve model metrics |
| 11 | 200+ tests, >80% coverage, mutation kill >85% |