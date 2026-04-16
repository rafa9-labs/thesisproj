# OpenCode Continuation Prompt

> **Context for**: Starting a new opencode session to continue implementation
> **Date**: 2026-04-16
> **Branch**: `feature/phase4-streamlit-ui`
> **Last commit**: `2854576` — Sprint 2-7 added to ROADMAP + CLAUDE.md created

---

## What This Project Is

A **Forex ML Backtesting Pipeline** — a commercial-grade walk-forward FX backtesting platform with 8 ML models, Streamlit UI, and one-command model comparison. The repo name "thesisproj" is misleading — this is NOT academic work. Refer to it as "forex pipeline" or "the pipeline". **Never use the word "thesis".**

## What's Done ✅

- **Phase 3**: Pipeline hardening — data leakage audit, feature disk cache (Parquet), Optuna search space simplification, named constants, walk-forward integrity tests (16/16 pass), import chain fixes, schema validation
- **Phase 4.5**: All 8 models verified end-to-end (logistic, xgboost, CNN, LSTM, Transformer, ensemble_cnn_lstm_xgboost, ensemble_adaptive_regime, DQN)
- **GPU detection** + warnings + smoke test launchers
- **Sprint 1**: Model Comparison & Leaderboard — `pipeline/model_comparison.py` + `run_comparison.bat` (one-command multi-model runner with ASCII leaderboard, paired t-tests, equity curve overlay)

## What's Next: Sprint 2 — Advanced Execution Models (6-8h)

This is the immediate task to implement. Full details in `ROADMAP.md`.

### S2.1: Position Sizing Models (2h)
- Fixed fractional sizing (% of equity per trade)
- Kelly criterion position sizing
- Fixed lot sizing (current behavior, as baseline)
- Volatility-adjusted sizing (ATR-based)
- Configurable via `config.py` and UI controls
- **New file**: `pipeline/execution/position_sizing.py`
- **Modify**: `config.py` (add execution config section)

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

## Sprint Sequence After Sprint 2

| Sprint | Topic | Est |
|--------|-------|-----|
| Sprint 3 | Multi-Currency Expansion (5 pairs × 3 timeframes) | 4-5h |
| Sprint 4 | Docker + CI/CD (GitHub Actions) | 3-4h |
| Sprint 5 | Comprehensive Tests + Benchmarks | 4-6h |
| Sprint 6 | News & Sentiment Features | 6-8h |
| Sprint 7 | Professional UI Polish | 10h |

## Key Files to Start With

Read these first to understand the codebase before implementing Sprint 2:

1. **`CLAUDE.md`** — Full project context, architecture map, conventions
2. **`ROADMAP.md`** — Complete sprint & phase details
3. **`pipeline/backtester/execution_patches.py`** — Current execution simulation (where Sprint 2 plugs in)
4. **`config.py`** — Global config with `PIPELINE_CONSTANTS` (where new execution config goes)
5. **`pipeline/backtester/composed.py`** — MLBacktester engine (understand how execution is called)
6. **`ui/controls.py`** — UI sidebar (where execution model dropdown goes)

## Important Rules

1. **Terminal**: Windows PowerShell 5.1 — NEVER use `&&` (use `;`), NEVER use `cd /d`
2. **Git**: Always push to GitHub before ending a session
3. **Naming**: Never use "thesis" — always "forex pipeline" or "the pipeline"
4. **Patterns**: Use `_PC = config.PIPELINE_CONSTANTS` for named constants
5. **Testing**: After implementation, verify with `python -c "from pipeline.execution.position_sizing import ..."` etc.

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