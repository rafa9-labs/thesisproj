---
name: backtest-runner
description: Run the FX ML pipeline CLI with smoke/quick/full modes, parse results, surface equity curves and metrics summary. Detects GPU availability, selects correct runner script, and reports pass/fail with key Sharpe, max DD, and win rate stats.
---

# Skill: /backtest-runner

**Trigger:** User types `/backtest-runner` or asks to run a backtest.

**Objective:** Execute the pipeline correctly for this project, parse output, and present actionable results.

**Protocol:**

1. **Determine run mode:**
   - `smoke` — All 8 models, 1 trial, minimal data. Use for verification (~2-5 min).
   - `quick` — Logistic + XGBoost only, few trials. Use for fast iteration.
   - `full` — All models, all trials, full date range. Use for production runs.
   - `gpu` — Same as smoke but routes through WSL for CUDA. Use only on Windows with WSL GPU.

2. **Detect environment:**
   - Check if Redis is running: `redis-cli ping`
   - Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
   - If Redis not running: `redis-server` (start it)

3. **Execute the run:**
   `powershell
   # Smoke (default)
   .\run_smoke.bat

   # Quick
   python -m pipeline.main_cli --models logistic xgboost --n-months 3 --seeds 42

   # Full
   python -m pipeline.main_cli --all-models --n-months 0
   `

4. **Parse results:**
   - Find latest `results/` subdirectory by timestamp.
   - Read `summary.csv` or `leaderboard.csv`.
   - Extract key metrics: Sharpe, Sortino, Max DD, Win Rate, Total PnL, number of trades.
   - Compare across models.

5. **Output format:**
`
## Backtest Results (smoke, 8 models, 1 trial)

| Model | Sharpe | Max DD | Win Rate | Total PnL | Trades |
|-------|--------|--------|----------|-----------|--------|
| XGBoost | 1.42 | -8.3% | 54.2% | +,341 | 187 |
| LSTM | 0.89 | -12.1% | 51.8% | + | 203 |
| ... | ... | ... | ... | ... | ... |

**Best model:** XGBoost (Sharpe 1.42)
**Worst model:** SVM (Sharpe 0.31)
**Results path:** results/2026-05-03_14-22-31/
`

6. **Failure handling:**
   - If pipeline crashes: capture the traceback, identify the failing model/stage.
   - If OOM: suggest reducing `--n-months` or setting `SMOKE_TEST=1`.
   - If Redis error: remind to start Redis first.
