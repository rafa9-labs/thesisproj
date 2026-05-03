---
name: model-comparator
description: Run model comparison leaderboard across all 8 model types. Parses ranking CSVs, equity curves, and paired t-test significance. Surfaces which models are statistically indistinguishable vs clearly superior.
---

# Skill: /model-comparator

**Trigger:** User types `/model-comparator` or asks to compare models.

**Objective:** Run the model comparison pipeline and present a statistically-rigorous leaderboard.

**Protocol:**

1. **Determine mode:**
   - `smoke` — All 8 models, 1 trial, quick comparison.
   - `full` — All trials, all months, complete significance testing.
   - `analyze` — Use existing results only (no new backtests).
   - `gpu` — Route through WSL for CUDA.

2. **Execute:**
   `powershell
   # Smoke comparison
   .\run_comparison.bat smoke

   # Analyze existing results only
   python -m pipeline.model_comparison --analyze

   # Full comparison
   .\run_comparison.bat full
   `

3. **Parse results:**
   - Read `results/<latest>/leaderboard.csv` for model rankings.
   - Read `results/<latest>/significance.json` or paired t-test output.
   - Group models into tiers:
     - **Tier 1** (significantly better, p < 0.05)
     - **Tier 2** (not significantly different from Tier 1)
     - **Tier 3** (significantly worse)

4. **Output format:**
`
## Model Comparison Leaderboard

| Rank | Model | Sharpe | Sortino | Max DD | Win% | Monthly Return | p-value vs #1 |
|------|-------|--------|---------|--------|------|---------------|---------------|
| 1 | XGBoost | 1.42 | 1.89 | -8.3% | 54.2% | +2.1% | — |
| 2 | Ensemble | 1.38 | 1.81 | -7.9% | 53.8% | +1.9% | 0.72 |
| 3 | LSTM | 0.89 | 1.12 | -12.1% | 51.8% | +0.8% | 0.03* |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Tiers
- **Tier 1:** XGBoost, Ensemble (statistically tied)
- **Tier 2:** LSTM (significantly below Tier 1)
- **Tier 3:** SVM, Logistic (significantly below Tier 2)

* p < 0.05 vs rank 1
`

5. **Recommendations:**
   - If models are statistically tied: recommend using the simpler/faster model.
   - If ensemble underperforms best single model: flag for investigation.
   - If all models have negative Sharpe: suggest data/feature review.
