# Findings: RuntimeError — No completed Optuna trials

## Error Chain
```
app.py → ui/state.py:123 run_backtest()
  → real_trading_mixin.py:537 real_trading_simulation()
    → run_mixin.py:3697 run_strategy()
      → runner.py:484 run_optuna_tuning()
        RuntimeError: No completed Optuna trials; cannot select Top-N.
```

## Root Cause
The **global HPO** step (run once before the monthly loop) fires Optuna with
`TRIAL_COUNTS` = `{random: 3, bayes: 3}` (quick-check mode = 6 total trials).

**ALL 6 trials failed** (every trial threw an exception inside `optuna_objective`).
When Optuna records only FAIL/PRUNED states, `completed = []` and line 484 raises.

## Why Trials Fail (most likely causes)
1. **Data too small / misaligned**: The `train_data` slice passed to Optuna may be
   empty or have insufficient bars for the model's lag + warmup requirements.
2. **Config mismatch**: `models_to_test` may not match what `optuna_objective` expects
   (e.g., CNN model selected but objective can't build features with current config).
3. **Objective crash**: An unhandled exception in the objective (import error, shape
   mismatch, NaN data) causes every trial to fail silently (Optuna catches and marks
   FAIL but doesn't surface the inner error).

## Why It's Hard to Diagnose
- Optuna **swallows** trial exceptions and marks them `FAIL` — the inner traceback
  goes to the Optuna warning log, not to the user.
- The `raise RuntimeError` on line 484 doesn't say **why** trials failed.

## Fix Plan (3 parts)

### Fix 1: runner.py — Better error message (show last trial error)
Instead of a bare RuntimeError, log the last few trial errors:
```python
if not completed:
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    _msgs = []
    for t in failed[-3:]:
        _msgs.append(f"  Trial {t.number}: {t.user_attrs.get('error', 'unknown')}")
    raise RuntimeError(
        f"No completed Optuna trials ({len(failed)} failed). Last errors:\n"
        + "\n".join(_msgs)
    )
```

### Fix 2: real_trading_mixin.py — Graceful fallback when global HPO fails
The global HPO block (lines ~520-550) should catch the RuntimeError and fall back
to running the monthly loop WITHOUT global HPO (per-month WFO or flat-month):
```python
try:
    res_hpo = self.run_strategy(hpo_cfg, ...)
except RuntimeError as e:
    if "No completed Optuna trials" in str(e):
        log_print(f"⚠️ Global HPO failed: {e}. Falling back to per-month WFO.")
        global_hpo_best = None
    else:
        raise
```

### Fix 3: Quick-check TRIAL_COUNTS should be higher minimum
3+3=6 trials is very fragile. Minimum should be ~5+5=10 to survive a few failures.
In `real_trading_mixin.py`:
```python
# Absolute minimum trials (avoid 0-completed with tiny samples)
_actual_trials = int(hpo_cfg.get("n_trials", 6))
if _actual_trials < 10:
    hpo_cfg["n_trials"] = max(_actual_trials, 10)
    hpo_cfg["n_startup_trials"] = max(int(hpo_cfg.get("n_startup_trials", 3)), 5)