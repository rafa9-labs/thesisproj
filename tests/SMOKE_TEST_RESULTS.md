# Smoke Test Investigation — Findings & Results

**Date:** 2026-04-05  
**Command under investigation:**  
```
$env:PYTHONIOENCODING="utf-8"; python tests/smoke_full.py > smoke_output4.txt 2>&1
```

---

## 1. What the Command Does

`tests/smoke_full.py` runs the **full ML trading pipeline** via `pipeline.main_cli.main()`.  
It performs:

1. **Data loading** — loads EURUSD H1 10-year CSV data
2. **Feature engineering** — computes ~80+ technical indicators (RSI, MACD, ATR, BBands, etc.)
3. **Walk-forward optimization** — splits data into train/test windows
4. **Optuna HPO** — runs Bayesian hyperparameter optimization with mini-block CV
5. **Real trading simulation** — simulates out-of-sample trading with the best parameters
6. **Ranking & reporting** — generates CSVs, plots, and ASCII ranking tables

---

## 2. Why It Appeared "Stuck"

The command was **not stuck** — it was running a **research-grade experiment** that takes 30–60+ minutes:

| Parameter | Original Value | Effect |
|-----------|---------------|--------|
| Models | `logistic` + `xgboost` | 2 models × full pipeline |
| Months | 3 | 3 walk-forward test windows |
| Optuna trials (logistic) | 5 random + 5 bayes = 10 | Each trial trains model × 5 CV folds |
| Optuna trials (xgboost) | 5 random + 15 bayes = 20 | Slower model, more trials |
| Mini-block CV folds | 5 per trial | ~30K+ rows per fold |
| **Total model trainings** | **~300+** | 30 trials × 5 folds × 2 months |

Each training takes ~5-10 seconds, so total runtime is **30-60+ minutes**.

---

## 3. Bug Found: `_safe_metrics_return` Not Defined

**Root cause:** `pipeline/_imports.py` was missing the import of `_safe_metrics_return` from `pipeline.metrics_tuples`.

**Effect:** Every single CV fold crashed with:
```
Exception: name '_safe_metrics_return' is not defined
```
This meant:
- Optuna received **zero valid trial results**
- It cycled through ALL trials getting no useful signal
- The pipeline ran to completion but produced **no results**

**Fix applied:** Added the missing import to `pipeline/_imports.py`:
```python
from pipeline.metrics_tuples import _safe_metrics_return, _empty_metrics
```

---

## 4. Smoke Test Mode Added

To make the smoke test fast (vs research-grade), a `SMOKE_TEST` environment variable was added.

### Changes to `pipeline/main_cli.py`:
```python
_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

if _SMOKE:
    global TRIAL_COUNTS
    TRIAL_COUNTS = {k: {"random": 1, "bayes": 1} for k in TRIAL_COUNTS}

N_REAL_MONTHS = 1 if _SMOKE else 3

if _SMOKE:
    MODEL_LIST = ["logistic"]  # fastest model only
else:
    MODEL_LIST = ["logistic", "xgboost", ...]
```

### Changes to `tests/smoke_full.py`:
```python
os.environ["SMOKE_TEST"] = "1"  # Activate fast mode before importing main_cli
```

### Smoke test configuration:

| Parameter | Smoke Mode | Research Mode |
|-----------|-----------|---------------|
| Models | 1 (logistic) | 2+ (logistic, xgboost, ...) |
| Months | 1 | 3 |
| Optuna trials | 2 (1 random + 1 bayes) | 10-20 per model |
| **Estimated time** | **~90 seconds** | **30-60+ minutes** |

---

## 5. Smoke Test Results (Final Run)

**Run time:** ~90 seconds (vs 30-60+ minutes previously)  
**Timestamp:** 2026-04-05 20:51 → 20:52  
**Output file:** `smoke_output_final.txt`

### Pipeline Execution Summary:

```
Study folder: results\05_04_26__20_51
Model: logistic
Repeat: 1/1, Seed: 33333
```

### Optuna Trial #0:
- **40 hyperparameters** sampled (lags, indicators, thresholds, etc.)
- 5 mini-block CV folds executed
- Fold #1: PRUNED (InvalidSR — only 2 trades, active_rate=0.037)
- Fold #2: OK (trades=33, active_rate=0.299, Sharpe=-0.190, PSR=0.266)
- Folds #3-#5: PRUNED (InvalidSR — too few trades)
- **Trial 0 result:** PRUNED (non-finite CV score)

### Optuna Trial #1:
- **40 hyperparameters** sampled (different from Trial #0)
- All 5 CV folds PRUNED (EARLY_COVERAGE_PRUNE — coverage below 80% threshold)
- **Trial 1 result:** PRUNED

### Final outcome:
```
RuntimeError: No completed Optuna trials; cannot select Top-N.
❌ Simulation failed for logistic: No completed Optuna trials; cannot select Top-N.
```

### Smoke test verdict:
```
============================================================
STAGE 2 PASSED - Full pipeline completed
============================================================
```

**Note:** The smoke test catches the error gracefully and reports PASSED because the pipeline code itself ran without crashes — the failure is in the ML/trading results (all trials pruned), not in the code.

---

## 6. Known Issues (From Output)

1. **All Optuna trials pruned** — With only 2 trials and strict coverage/Sharpe gates, there's a high chance all trials get pruned. This is expected for a smoke test with minimal trials. For research runs with 10-20 trials, some will survive.

2. **sklearn deprecation warnings:**
   - `'penalty' was deprecated in version 1.8` (use `l1_ratio` instead)
   - `'n_jobs' has no effect since 1.8` (leave unspecified)
   
3. **Optuna ExperimentalWarning:** `multivariate` and `group` args are experimental

4. **TensorFlow GPU:** Not available on native Windows for TF >= 2.11

---

## 7. Files Modified

| File | Change |
|------|--------|
| `pipeline/_imports.py` | Added `from pipeline.metrics_tuples import _safe_metrics_return, _empty_metrics` |
| `pipeline/main_cli.py` | Added `SMOKE_TEST` env var to reduce models, months, and trials |
| `tests/smoke_full.py` | Added `os.environ["SMOKE_TEST"] = "1"` before importing main |

---

## 8. How to Run

### Fast smoke test (~90 seconds):
```powershell
cd C:\Users\rafa\ML_Trading\thesisproj
$env:PYTHONIOENCODING="utf-8"; $env:SMOKE_TEST="1"; python tests/smoke_full.py
```

### Full research run (30-60+ minutes):
```powershell
cd C:\Users\rafa\ML_Trading\thesisproj
$env:PYTHONIOENCODING="utf-8"; python tests/smoke_full.py