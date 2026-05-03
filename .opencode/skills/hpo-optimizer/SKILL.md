---
name: hpo-optimizer
description: Guide Optuna hyperparameter optimization runs. Suggests search space adjustments based on pipeline config, detects overfitting from trial history, recommends pruning, and validates HPO configs against the canonical search space in config.py.
---

# Skill: /hpo-optimizer

**Trigger:** User types `/hpo-optimizer` or asks about HPO/tuning/hyperparameters.

**Objective:** Guide efficient hyperparameter optimization using Optuna, detect common issues, and validate configs.

**Protocol:**

1. **Validate search space:**
   - Read `config.py` SEARCH\_SPACE dict.
   - Read model-specific HPO configs in `hpo/` directory.
   - Check that custom configs don't exceed canonical bounds.
   - Flag any parameter ranges that are too wide (wasted trials) or too narrow (missed optima).

2. **Detect overfitting:**
   - Read last N trial results from the Optuna study.
   - Check: is train score much higher than validation score? (classic overfitting)
   - Check: have the last K trials shown no improvement? (converged, should stop)
   - Suggest early stopping if appropriate.

3. **Recommend pruning:**
   - If Optuna `MedianPruner` or `HyperbandPruner` is not configured, suggest it.
   - Default: `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`

4. **Suggest adjustments:**
   - If best trial is at the edge of a range: expand that range.
   - If trials cluster around a value: narrow the range around it.
   - If a parameter has no effect on score: consider fixing it to reduce dimensionality.

5. **Output format:**
`
## HPO Analysis (XGBoost, 50 trials)

### Search Space Validation
- learning_rate [0.01, 0.3] -- OK
- max_depth [3, 10] -- WARNING: best trial at boundary (10), expand to [3, 15]
- n_estimators [100, 1000] -- OK, cluster around 300

### Overfitting Detection
- Mean train AUC: 0.89
- Mean val AUC: 0.72
- **GAP: 0.17** -- Moderate overfitting. Consider: reducing max\_depth, adding regularization.

### Recommendations
1. Expand max\_depth upper bound to 15
2. Reduce n\_estimators range to [100, 500] (no benefit above 500)
3. Fix min\_child\_weight=1 (no effect on score across 50 trials)
4. Enable MedianPruner to cut bad trials early
`

6. **Run HPO:**
   `python -m pipeline.main\_cli --models xgboost --n-trials 50`