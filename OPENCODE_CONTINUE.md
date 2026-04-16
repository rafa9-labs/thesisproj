# OpenCode Continuation Prompt

> **Context for**: Starting a new opencode session to continue implementation
> **Date**: 2026-04-16
> **Branch**: `feature/sprint2-execution-models`
> **Last commit**: `6f00605` — S2.1 position sizing models (5 methods)

---

## What This Project Is

A **Forex ML Backtesting Pipeline** — a commercial-grade walk-forward FX backtesting platform with 8 ML models, Streamlit UI (interim), and one-command model comparison. **End goal**: commercial Electron desktop app sold via Paddle. The repo name "thesisproj" is misleading — this is NOT academic work. Refer to it as "forex pipeline" or "the pipeline". **Never use the word "thesis".**

## What's Done ✅

- **Phase 3**: Pipeline hardening — data leakage audit, feature disk cache (Parquet), Optuna search space simplification, named constants, walk-forward integrity tests (16/16 pass), import chain fixes, schema validation
- **Phase 4.5**: All 8 models verified end-to-end (logistic, xgboost, CNN, LSTM, Transformer, ensemble_cnn_lstm_xgboost, ensemble_adaptive_regime, DQN)
- **GPU detection** + warnings + smoke test launchers
- **Sprint 1**: Model Comparison & Leaderboard — `pipeline/model_comparison.py` + `run_comparison.bat`
- **Sprint 2.1**: Position Sizing Models — `pipeline/execution/position_sizing.py` (5 methods: fixed, fractional, Kelly, ATR, vol-target)

## What's Next: Sprint 2 (remaining) — Advanced Execution Models

### S2.2: Stop-Loss / Take-Profit Management (2h)
- Fixed SL/TP in pips
- ATR-based dynamic SL/TP
- Breakeven stop management
- Partial close (scale-out) at TP levels
- **New file**: `pipeline/execution/stops.py`
- **Modify**: `pipeline/backtester/execution_patches.py`

### S2.3: Trailing Stop Implementation (1.5h)
- Standard trailing stop (fixed pips)
- ATR trailing stop
- Chandelier exit
- Configurable activation threshold
- **New file**: `pipeline/execution/trailing.py`

### S2.4: Risk Management Framework (1.5h)
- Max drawdown circuit breaker (pause trading when DD > threshold)
- Max consecutive losses limit
- Daily loss limit
- Correlation-aware position limits
- **New file**: `pipeline/execution/risk_manager.py`

### S2.5: Execution Model Integration (1h)
- Wire execution models into `execution_patches.py`
- Add execution config to UI sidebar
- Execution model selection dropdown in backtest config
- Metrics breakdown: gross vs net, impact of each cost component
- **Modify**: `pipeline/backtester/execution_patches.py`, `ui/controls.py`

## Full Sprint Sequence

| Sprint | Topic | Est | Status |
|--------|-------|-----|--------|
| **S1** | Model Comparison & Leaderboard | 3-4h | ✅ DONE |
| **S2** | Advanced Execution Models | 6-8h | 🔄 IN PROGRESS (S2.1 done) |
| **S3** | Multi-Currency Expansion | 4-5h | ⬜ TODO |
| **S4** | Docker + CI/CD | 3-4h | ⬜ TODO |
| **S5** | Comprehensive Tests + Benchmarks | 4-6h | ⬜ TODO |
| **S6** | News & Sentiment Features | 6-8h | ⬜ TODO |
| **S7** | FastAPI Backend | 8-10h | ⬜ TODO |
| **S8** | React Frontend | 20-25h | ⬜ TODO |
| **S9** | Electron Desktop Shell | 10-12h | ⬜ TODO |
| **S10** | Security & Licensing (Paddle) | 12-15h | ⬜ TODO |
| **S11** | Installer & Auto-Update | 6-8h | ⬜ TODO |
| **S12** | Commercial Infrastructure | 8-10h | ⬜ TODO |
| **S13** | Beta & Launch | 6-8h | ⬜ TODO |

**Product target**: Commercial Electron desktop app (React + FastAPI + Python), sold via Paddle.
**Pricing**: Hybrid — one-time purchase + annual updates subscription.

## Key Files to Start With

Read these first to understand the codebase before continuing Sprint 2:

1. **`CLAUDE.md`** — Full project context, architecture map, conventions
2. **`ROADMAP.md`** — Complete sprint & phase details (restructured for desktop app)
3. **`pipeline/backtester/execution_patches.py`** — Execution loop (where S2 plugs in)
4. **`pipeline/execution/position_sizing.py`** — S2.1 sizing models (pattern to follow for stops/trailing)
5. **`config.py`** — Global config with `ExecutionConfig` (S2.1 added)
6. **`pipeline/metrics_tuples.py`** — `CLASS_DEFAULTS` with sizing params

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

# Run existing tests
.\run_all_tests.bat

# Run smoke test
.\run_smoke.bat

# Run model comparison
.\run_comparison.bat smoke