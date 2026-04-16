# CLAUDE.md — Project Rules & Context for AI Assistants

> **Purpose**: Single source of truth for project identity, conventions, rules, and current state.
> **Used by**: OpenCode, Claude Code, or any AI coding assistant working on this project.

---

## Project Identity

- This is a **Forex ML Backtesting Pipeline** — a commercial-grade walk-forward FX backtesting platform.
- The repo name "thesisproj" is misleading — this has **NOTHING to do with any thesis or academic work**.
- Refer to it as "forex pipeline" or "the pipeline" in all conversations and documentation.
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
- Current branch: `feature/phase4-streamlit-ui`
- Remote: `https://github.com/rafa9-labs/thesisproj.git`

## Architecture Overview

### Entry Points
- `app.py` — Streamlit web UI entry point
- `pipeline/main_cli.py` — CLI runner (headless backtesting)
- `pipeline/model_comparison.py` — Model comparison & leaderboard (CLI)

### Pipeline Engine (`pipeline/`)
- `pipeline/backtester/composed.py` — **MLBacktester** class (core engine)
- `pipeline/backtester/features_mixin.py` — Feature engineering (TA indicators)
- `pipeline/backtester/execution_patches.py` — Execution simulation (slippage, costs)
- `pipeline/backtester/deep_mixin.py` — Deep learning model handling
- `pipeline/backtester/dqn_mixin.py` — DQN/RL model handling
- `pipeline/backtester/model_factory_mixin.py` — Model creation
- `pipeline/backtester/real_trading_mixin.py` — Real trading simulation
- `pipeline/backtester/strategy_mixin.py` — Strategy logic
- `pipeline/backtester/ensemble_mixin.py` — Ensemble model logic
- `pipeline/backtester/run_mixin.py` — Run orchestration

### Models (`models/`)
- `models/registry.py` — Model registry (maps names to classes)
- `models/base_model.py` — Abstract base model
- `models/logistic.py`, `models/xgboost_model.py`, `models/svm.py`, `models/random_forest.py`
- `models/cnn.py`, `models/lstm.py`, `models/transformer.py`
- `models/ensemble_cnn_lstm_xgboost.py`, `models/ensemble_adaptive_regime.py`
- **All 8 model types verified working end-to-end** (2026-04-15)

### UI (`ui/`)
- `ui/controls.py` — Sidebar nav + 6-tab layout + GPU warnings
- `ui/state.py` — AppState + backtest adapter
- `ui/results.py` — Results display + export
- `ui/dashboard.py` — Dashboard renderer
- `ui/charts.py` — Plotly chart builders
- `ui/validators.py` — Input validation

### Config & Schemas
- `config.py` — Global config (`PIPELINE_CONSTANTS`, `SEARCH_SPACE`)
- `schemas/` — Pydantic-like validators (backtest, features, hpo, settings)
- `pipeline/runtime.py` — GPU detection, thread budgets, CUDA config
- `pipeline/feature_cache.py` — Parquet disk cache for features

### Data (`csv_data/`)
- `EURUSD_10_years_H1_OANDA.csv`
- `EURUSD_10_years_H4_OANDA.csv`
- `EURUSD_10_years_M30_OANDA.csv`

### HPO Configs (`hpo/`)
- Best configs for: cnn, lstm, transformer, xgboost, logistic, ensemble_adaptive_regime, ensemble_cnn_lstm_xgboost

### RL (`rl/`)
- `rl/dqn_agent.py` — Dueling DQN agent
- `rl/environment.py` — Trading environment (gym-style)
- `rl/replay_buffer.py` — Experience replay
- `rl/wrappers.py` — Reward processing, cost-aware wrappers

### Tests (`tests/`)
- 16+ test files covering pipeline, metrics, models, schemas, walk-forward integrity
- `tests/smoke_all_models.py` — Smoke test for all 8 model types
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
npx -y @modelcontextprotocol/server-postgres postgresql://forex_admin:changeme_secure_password@localhost:5432/forex_ml
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
| **Sprint 2** | Advanced Execution Models | 🔄 IN PROGRESS | 6-8h |
| **Sprint 3** | Multi-Currency Expansion | ⬜ TODO | 4-5h |
| **Sprint 4** | Docker + CI/CD | ⬜ TODO | 3-4h |
| **Sprint 5** | Comprehensive Tests + Benchmarks | ⬜ TODO | 4-6h |
| **Sprint 6** | News & Sentiment Features | ⬜ TODO | 6-8h |
| **Sprint 7** | FastAPI Backend | ⬜ TODO | 8-10h |
| **Sprint 8** | React Frontend | ⬜ TODO | 20-25h |
| **Sprint 9** | Electron Desktop Shell | ⬜ TODO | 10-12h |
| **Sprint 10** | Security & Licensing (Paddle) | ⬜ TODO | 12-15h |
| **Sprint 11** | Installer & Auto-Update | ⬜ TODO | 6-8h |
| **Sprint 12** | Commercial Infrastructure | ⬜ TODO | 8-10h |
| **Sprint 13** | Beta & Launch | ⬜ TODO | 6-8h |

**Product target**: Commercial Electron desktop app (React + FastAPI + Python), sold via Paddle.
**Pricing**: Hybrid — one-time purchase + annual updates subscription.

**Next task**: Sprint 2 — Advanced Execution Models (S2.1 done, S2.2 next)
- ~~Position sizing (Kelly, fixed fractional, ATR-based)~~ ✅ S2.1 DONE
- Stop-loss / take-profit management ← CURRENT
- Trailing stops (standard, ATR, Chandelier exit)
- Risk management framework (DD circuit breaker, loss limits)
- Integration into pipeline + UI

See `ROADMAP.md` for full sprint details with sub-tasks and file references.

## How to Run

```powershell
# Launch Streamlit UI
.\launch_ui.bat

# Run smoke test (all 8 models, 1 trial)
.\run_smoke.bat

# Run model comparison
.\run_comparison.bat smoke

# Run all tests
.\run_all_tests.bat

# GPU smoke test (Windows → WSL)
.\run_smoke_gpu.bat
```

## Key Files Quick Reference

| File | Role |
|------|------|
| `app.py` | Streamlit entry point (interim dev UI) |
| `config.py` | Global configuration + constants + ExecutionConfig |
| `pipeline/backtester/composed.py` | MLBacktester engine |
| `pipeline/backtester/execution_patches.py` | Execution loop + PatchConfig + LoopResult |
| `pipeline/execution/position_sizing.py` | Position sizing models (S2.1) |
| `pipeline/main_cli.py` | CLI runner |
| `pipeline/runtime.py` | GPU detection, CUDA config |
| `pipeline/model_comparison.py` | Model comparison & leaderboard |
| `models/registry.py` | Model registry |
| `ROADMAP.md` | Full product roadmap (13 sprints) |
| `OPENCODE_CONTINUE.md` | Continuation context for AI sessions |
| `CLAUDE.md` | This file — AI assistant context |