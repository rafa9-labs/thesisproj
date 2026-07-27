# CLAUDE.md — Project Rules & Context for AI Assistants

> **Purpose**: Single source of truth for project identity, conventions, rules, and current state.
> **Used by**: OpenCode, Claude Code, or any AI coding assistant working on this project.

---

## Project Identity

- This is **KodaQuant** — a commercial-grade walk-forward FX backtesting platform.
- The repo name "thesisproj" is misleading — this has **NOTHING to do with any thesis or academic work**.
- Refer to it as "KodaQuant" or "the pipeline" in all conversations and documentation.
- **Never use the word "thesis"** in any context related to this project.

## Terminal & Shell Rules

- The default terminal is **Windows PowerShell 5.1**, NOT cmd.exe.
- **NEVER** use `&&` to chain commands — it causes a ParserError. Use `;` instead.
- **NEVER** use `cd /d` — that's cmd.exe syntax. In PowerShell, just use `cd C:\Path`.
- If you absolutely need cmd.exe syntax, wrap it: `cmd.exe /c "command1 && command2"`.
- Examples:
  - ❌ BAD: `cd /d c:\path && python script.py 2>&1`
  - ✅ GOOD: `cd C:\path ; python script.py 2>&1`

## Git Rules

- **ALWAYS** push all work to GitHub before ending a session.
- Before completing a task, check for uncommitted changes.
- Commit with descriptive messages. Push to `origin <current-branch>`.
- Current branch: `repo-cleanup`
- Remote: `https://github.com/rafa9-labs/thesisproj.git`

## Architecture Overview

### Entry Points
- `api/main.py` — FastAPI backend entry point (uvicorn)
- `frontend/` — React + Vite frontend (the product UI)
- `pipeline/main_cli.py` — CLI runner (headless backtesting)
- `pipeline/model_comparison.py` — Model comparison & leaderboard (CLI)

### Pipeline Engine (`pipeline/`)
- `pipeline/backtester/composed.py` — **MLBacktester** class (11 mixins)
- `pipeline/backtester/run_mixin.py` — HPO loop, Optuna study, walk-forward optimization
- `pipeline/backtester/real_trading_mixin.py` — Real trading simulation + equity curve
- `pipeline/backtester/features_mixin.py` — Feature engineering (TA indicators)
- `pipeline/backtester/execution_patches.py` — Execution simulation (slippage, costs)
- `pipeline/backtester/deep_mixin.py` — Deep learning model handling
- `pipeline/backtester/dqn_mixin.py` — DQN/RL model handling
- `pipeline/backtester/model_factory_mixin.py` — Model creation dispatch
- `pipeline/backtester/ensemble_mixin.py` — Ensemble model logic
- `pipeline/backtester/strategy_mixin.py` — Strategy signal generation
- `pipeline/backtester/evaluation_mixin.py` — CV evaluation + metrics
- `pipeline/backtester/data_mixin.py` — Data loading + date range
- `pipeline/execution/` — Position sizing, stops, trailing, risk management
- `pipeline/tuning/` — Optuna HPO (runner, objective, sampler, refit)
- `pipeline/metrics/` — 16-metric evaluation, overfitting detection, DSR, PBO
- `pipeline/features/` — Feature engineering, BorutaSHAP, caching
- `pipeline/committee/` — Multi-agent committee system (11 files)
- `pipeline/regime/` — HMM regime detection + utilities
- `pipeline/llm/` — LLM sentiment engine + advisor
- `pipeline/models/` — Model persistence, registry, fast retrain
- `pipeline/data/` — SQLite store, candle syncer, data downloader
- `pipeline/forward_test.py` — Forward testing for saved models
- `pipeline/main_cli.py` — CLI runner (headless backtesting)
- `pipeline/workers.py` — Multiprocessing worker pool
- `pipeline/hardware_profile.py` — CPU/GPU detection + VRAM measurement

### Models (`models/`)
- `models/registry.py` — Model registry with `@register_model` decorator
- `models/base_model.py` — Abstract base model
- `models/logistic.py`, `models/svm.py`, `models/random_forest.py`, `models/decision_tree.py`
- `models/xgboost_model.py`, `models/lightgbm_proxy.py`
- `models/cnn.py`, `models/lstm.py`, `models/gru.py`, `models/gru_lstm.py`, `models/transformer.py`
- `models/ensemble_cnn_lstm_xgboost.py`, `models/ensemble_adaptive_regime.py`
- `models/stacking_ensemble.py`, `models/meta_ensemble.py`
- `models/regime_classifier.py`
- **All 18 model types verified working end-to-end**

### Frontend (`frontend/`)
- React 19 + TypeScript 6 + TailwindCSS 4 + shadcn/ui
- 10 pages: Dashboard, Backtest, Results, Compare, Committee, Trading, Models, News, Monitor, Settings
- 60+ components (charts, panels, forms, tables)
- 7 Zustand stores + React Query cache
- API client: `frontend/src/api/` (REST + WebSocket)
- State: Zustand stores + React Query cache

### Trading (`trading/`)
- `trading/oanda_client.py` — OANDA v20 REST API client
- `trading/live_engine.py` — Live trading engine with 4-layer risk architecture
- `trading/paper_engine.py` — Paper trading engine with portfolio tracking
- `trading/committee_engine.py` — Committee-based live trading
- `trading/risk_controls.py` — 17 risk gates (pre-trade, post-trade, infra, kill switch)
- `trading/live_committee_runner.py` — Live committee runner with regime switching
- `trading/model_store.py` — Model artifact loading for live deployment
- `trading/rotation_scheduler.py` — Model rotation scheduling
- `trading/lean_bridge.py` — QuantConnect LEAN integration
- `trading/alerting.py` — Trade and risk event alerting

### Streamlit UI — REMOVED (2026-04-20)
- All Streamlit code deleted: `app.py`, `ui/` directory, `launch_ui.bat`
- React frontend is the product UI

### Config & Schemas
- `config.py` — Global config (`PIPELINE_CONSTANTS`, `SEARCH_SPACE`, 697 lines)
- `schemas/` — Pydantic v2 validators (backtest, features, hpo, settings)
- `pipeline/runtime.py` — GPU detection, thread budgets, CUDA config
- `pipeline/feature_cache.py` — Parquet disk cache for features

### Data (`csv_data/`)
- 24 CSV files: 6 pairs × 4 timeframes (M15, M30, H1, H4)
- `EURUSD_10_years_{M15,M30,H1,H4}_OANDA.csv`
- `GBPUSD_10_years_{M15,M30,H1,H4}_OANDA.csv`
- `USDJPY_10_years_{M15,M30,H1,H4}_OANDA.csv`
- `AUDUSD_10_years_{M15,M30,H1,H4}_OANDA.csv`
- `USDCAD_10_years_{M15,M30,H1,H4}_OANDA.csv`
- `GBPJPY_10_years_{M15,M30,H1,H4}_OANDA.csv`

### HPO Configs (`hpo/`)
- Best configs for: cnn, lstm, transformer, xgboost, logistic, ensemble_adaptive_regime, ensemble_cnn_lstm_xgboost

### RL (`rl/`)
- `rl/dqn_agent.py` — Dueling DQN agent
- `rl/environment.py` — Trading environment (gym-style)
- `rl/replay_buffer.py` — Experience replay
- `rl/wrappers.py` — Reward processing, cost-aware wrappers

### Tests (`tests/`)
- 2,028 tests covering pipeline, metrics, models, schemas, walk-forward integrity, build validation, committee, trading
- `tests/test_models_train_predict.py` — Build/train/predict for all 18 model types (40 tests)
- `tests/test_build_validation.py` — PyInstaller spec + hidden imports validation (46 tests)
- `tests/benchmarks/` — Model timing + memory benchmarks (11 tests, 4 slow)
- `tests/golden/` — Deterministic output regression tests (7 tests + golden data files)
- `tests/smoke_all_models.py` — Smoke test for all 18 model types
- `tests/conftest.py` — Shared fixtures

### Key Utilities
- `pipeline/metrics_eval.py` — Evaluation metrics (Sharpe, Sortino, max DD, etc.)
- `pipeline/metrics_tuples.py` — Metric tuple definitions
- `pipeline/optuna_utils.py` — Optuna HPO utilities
- `pipeline/tuning/` — Tuning sampler, runner, objective
- `pipeline/hpo_persistence.py` — HPO result persistence
- `pipeline/workers.py` — Multi-process worker pool
- `pipeline/plotting.py` — Chart generation (matplotlib/plotly)
- `pipeline/io_utils.py` — File I/O utilities
- `pipeline/model_utils.py` — Model utility functions
- `pipeline/memory_utils.py` — Memory management
- `pipeline/calibration.py` — Probability calibration
- `pipeline/coverage.py` — Model coverage analysis

## Key Patterns & Conventions

### Configuration
- Named constants via `_PC = config.PIPELINE_CONSTANTS` in mixin files
- Search space via `SEARCH_SPACE` dict in `config.py`
- Env vars for CLI: `MODEL_LIST`, `SEEDS`, `REPEATS`, `N_MONTHS`, `SMOKE_TEST`

### Walk-Forward Backtesting
- Monthly refit with strict chronological splits (train ≤ test)
- 1-bar execution delay enforced everywhere (anti-look-ahead)
- Triple-barrier labeling for advanced exit strategies
- Cost-aware execution: spread + slippage modeled

### Feature Engineering
- 3-layer cache: in-memory slice → Parquet disk cache → fresh computation
- Cache key: SHA256(data file + size + mtime + feature config)
- All indicators look backward only (no future data leakage)

### Model Registry
- Models registered via `models/registry.py`
- Each model: `build()`, `train()`, `predict()` interface
- Deep models: lazy TF/PyTorch init to avoid memory issues

## MCP Servers Available (from Cline session)

The following MCP servers were configured in Cline and may need to be replicated in opencode:

### 1. filesystem
```
npx -y @modelcontextprotocol/server-filesystem C:\Users\rafa\Projects
```
Tools: read_file, write_file, edit_file, search_files, directory_tree, etc.

### 2. github
```
npx -y @modelcontextprotocol/server-github
```
Tools: create_or_update_file, push_files, create_issue, create_pull_request, search_code, etc.
- Repo: `rafa9-labs/thesisproj`
- Branch: `feature/phase4-streamlit-ui`

### 3. postgres
```
npx -y @modelcontextprotocol/server-postgres "$DATABASE_URL"
```
Tools: query (read-only SQL)

### 4. context7
```
npx -y @upstash/context7-mcp@latest
```
Tools: resolve-library-id, query-docs (documentation lookup)

### 5. sequential-thinking
```
npx -y @modelcontextprotocol/server-sequential-thinking
```
Tools: sequentialthinking (structured problem-solving)

### 6. brave-search
```
npx -y @modelcontextprotocol/server-brave-search
```
Tools: brave_web_search, brave_local_search

### 7. puppeteer
```
npx -y @modelcontextprotocol/server-puppeteer
```
Tools: puppeteer_navigate, puppeteer_screenshot, puppeteer_click, puppeteer_fill, puppeteer_evaluate

### 8. fetch
```
npx -y mcp-fetch-server
```
Tools: fetch_html, fetch_markdown, fetch_txt, fetch_json, fetch_readable, fetch_youtube_transcript

## Current Sprint Status

| Sprint | Topic | Status | Est |
|--------|-------|--------|-----|
| **Sprint 1** | Model Comparison + Leaderboard | ✅ DONE | 3-4h |
| **Sprint 2** | Advanced Execution Models | ✅ DONE | 6-8h |
| **Sprint 3** | Multi-Currency Expansion | ✅ DONE | 4-5h |
| **Sprint 4** | Docker + CI/CD | 🔄 PARTIAL | 3-4h |
| **Sprint 5** | Comprehensive Tests + Benchmarks | ✅ DONE | 4-6h |
| **Sprint 6** | News & Sentiment Features | ✅ DONE | 6-8h |
| **Sprint 7** | FastAPI Backend | ✅ DONE | 8-10h |
| **Sprint 8** | React Frontend | ✅ DONE | 20-25h |
| **Sprint 8B** | Frontend ↔ API Integration | ✅ DONE | 6-8h |
| **Sprint 9** | Electron Desktop Shell | ✅ DONE | 10-12h |
| **Sprint 10** | Security & Licensing (Paddle) | ✅ DONE | 12-15h |
| **Sprint 11** | Installer & Auto-Update | ✅ DONE | 6-8h |
| **Sprint 12** | Product Intelligence & UX | ✅ DONE | 22-24h |
| **Sprint 13** | Beta & Launch | ⬜ TODO | 6-8h |
| **Sprint 14** | Pipeline Enhancements | ✅ DONE | 5-8h |
| **Sprint 15** | KodaQuant Branding | ✅ DONE | 4-6h |
| **Sprint 16** | Overfitting & Transparency | ✅ DONE | 12-16h |
| **Sprint 16.8** | Model Persistence & Deployment | ✅ DONE | 15-19h |
| **Sprint 16.9** | Forward Test + Live Trading Bridge | ✅ DONE | 5-7h |
| **Sprint 17** | UI Polish & Search | ⬜ TODO | 8-10h |
| **Sprint 18** | Live News & Market Data | 🔄 PARTIAL | 10-14h |
| **Sprint 19** | Ensemble Models & Extensibility | 🔄 PARTIAL | 8-10h |
| **Sprint 20** | LLM / AI Integration | 🔄 PARTIAL | 10-14h |
| **Sprint 21** | Live Trading (OANDA) | ✅ DONE | 12-16h |
| **Sprint 22** | Commercial Infrastructure | ⬜ TODO | 8-10h |
| **Sprint 23** | Pipeline Stability & Live Monitor UX | ✅ DONE | 6-8h |
| **Sprint 24** | Historical News as Backtest Features | ⬜ TODO | 6-8h |

**Product target**: Commercial Electron desktop app (React + FastAPI + Python), sold via Paddle.
**Pricing**: Hybrid — one-time purchase + annual updates subscription.
**Test suite**: 2,028 tests across 95+ test files.
**Models**: 18 registered (shallow, boosting, deep, RL, ensemble, meta).

**Next task**: Sprint 13 — Beta & Launch (S13.1 closed beta → S13.2 performance optimization → S13.3 launch preparation)

See `docs/ROADMAP.md` for full sprint details with sub-tasks and file references.

## How to Run

```powershell
# Start Redis
redis-server

# Start FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

# Start Celery worker (separate terminal)
celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery

# Start React frontend (separate terminal)
cd frontend; npm run dev

# Run smoke test (all 18 models, 1 trial)
.\run_smoke.bat

# Run model comparison
.\run_comparison.bat smoke

# Run all tests
.\run_all_tests.bat
```

### GPU Backtests (WSL2)

**Setup** (one-time):
- WSL2 Ubuntu-22.04 with RTX 3090 (CUDA 13.1 driver on Windows host)
- Python venv: `~/thesisproj-venv` in WSL home (NOT under /mnt/c/)
- TF 2.18 bundles CUDA 12.x libs — no separate CUDA toolkit needed in WSL
- All project pip deps installed in the venv (pandas, scikit-learn, optuna, xgboost, etc.)
- `results/` directory must be owned by `benji`, not root. If broken, delete from Windows and recreate: `wsl bash -c "mkdir -p /mnt/c/Users/rafa/ML_Trading/thesisproj/results"`

**Run from PowerShell**:  
```powershell
wsl bash -c "cd /mnt/c/Users/rafa/ML_Trading/thesisproj && export MLB_THREADS=1 && MODEL_LIST=lstm,cnn SEEDS=42 REPEATS=3 N_MONTHS=12 SMOKE_TEST=0 PYTHONPATH=. ~/thesisproj-venv/bin/python pipeline/main_cli.py 2>&1"
```

**Verify GPU**:  
```powershell
wsl bash -c "~/thesisproj-venv/bin/python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'" 2>&1
```

> Frontend, API, and Celery remain on Windows. Only the GPU-accelerated Python pipeline runs in WSL2. Project source stays on `/mnt/c/` (Windows filesystem).

## Key Files Quick Reference

| File | Role |
|------|------|
| `api/main.py` | FastAPI entry point (15 routers) |
| `api/tasks.py` | Celery background tasks |
| `config.py` | Global configuration + constants + ExecutionConfig + SEARCH_SPACE |
| `pipeline/backtester/composed.py` | MLBacktester engine (11 mixins) |
| `pipeline/backtester/run_mixin.py` | HPO loop + walk-forward optimization (4,094 lines) |
| `pipeline/backtester/real_trading_mixin.py` | Real trading simulation + equity curve |
| `pipeline/backtester/features_mixin.py` | Feature engineering (TA indicators) |
| `pipeline/execution/` | Position sizing, stops, trailing, risk management |
| `pipeline/committee/` | Multi-agent committee system (11 files) |
| `pipeline/regime/` | HMM regime detection |
| `pipeline/llm/` | LLM sentiment + advisor |
| `pipeline/models/` | Model persistence + registry + fast retrain |
| `pipeline/tuning/` | Optuna HPO lifecycle |
| `pipeline/metrics/` | 16-metric evaluation + overfitting detection |
| `pipeline/features/` | Feature engineering + BorutaSHAP + caching |
| `pipeline/data/` | SQLite store + candle syncer + downloader |
| `pipeline/forward_test.py` | Forward testing for saved models |
| `pipeline/main_cli.py` | CLI runner (headless backtesting) |
| `pipeline/runtime.py` | GPU detection, CUDA config |
| `pipeline/model_comparison.py` | Model comparison & leaderboard |
| `pipeline/hardware_profile.py` | CPU/GPU detection + VRAM measurement |
| `models/registry.py` | Model registry (18 types) |
| `trading/` | OANDA client + paper/live engines + risk controls |
| `news/` | RSS scraper + VADER/LLM sentiment + features |
| `schemas/` | Pydantic v2 config validators |
| `ROADMAP.md` | Full product roadmap (24 sprints) |
| `CLAUDE.md` | This file — AI assistant context |