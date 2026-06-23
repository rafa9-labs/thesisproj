# Test Automation Roadmap

> Generated: 2026-06-22 | Gatekeeper & Dispatcher Architecture (Phase 1+2)

---

## Status Legend

- `[ ]` Pending
- `[~]` In progress
- `[x]` Passing
- `[!]` Failing / blocked

---

## Layer 1 — Unit Tests (Core Logic & Math)

Pure functions with no I/O, no DB, no network. Mocked where needed.

---

### `tests/test_gatekeeper_vram_math.py`

**Target**: `api/process_manager.py` — `allocate_vram()`, `release_vram()`, VRAM properties

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [x] | `test_allocate_within_limit` | Allocating 4096 with 8192 total returns True, `gpu_vram_used_mb` == 4096 |
| 2 | [x] | `test_allocate_exceeds_limit` | Allocating 9000 with 8192 total returns False, used stays 0 |
| 3 | [x] | `test_allocate_zero_budget` | `allocate_vram(0)` returns True, no-op |
| 4 | [x] | `test_allocate_when_gpu_total_zero` | When `gpu_total_vram_mb == 0`, gate is bypassed (always True) |
| 5 | [x] | `test_allocate_exact_limit` | Allocating exactly 8192 with 8192 total returns True |
| 6 | [x] | `test_release_normal` | Allocate 4096, release 4096, used == 0 |
| 7 | [x] | `test_release_partial` | Allocate 4096, release 2048, used == 2048 |
| 8 | [x] | `test_release_over_release` | Release more than allocated — clamped to 0, never negative |
| 9 | [x] | `test_release_zero_budget` | `release_vram(0)` no-op |
| 10 | [x] | `test_concurrent_allocations` | 6 threads each trying to allocate 2048 on 8192 total → exactly 4 succeed, 2 fail |
| 11 | [x] | `test_vram_available_property` | After allocate(4096), `gpu_vram_available_mb` == total - 4096 |
| 12 | [x] | `test_double_initialize_is_noop` | Calling `initialize()` twice does not recreate pools |
| 13 | [x] | `test_submit_before_initialize_raises` | `pm.submit(...)` before `pm.initialize()` raises RuntimeError |
| 14 | [x] | `test_allocate_vram_thread_safety` | 100 concurrent `allocate_vram(100)` calls on 5000 total → exactly 50 succeed |

---

### `tests/test_resource_budget_math.py`

**Target**: `pipeline/resource_budget.py` — `compute_budget()`, `get_resource_budget()`

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [x] | `test_budget_8core_non_hybrid` | 8 physical cores → effective=8, target=4, blas=2, cv=2 |
| 2 | [x] | `test_budget_16core_hybrid` | 8P+8E → effective=12, target=7, blas=3, cv=2 |
| 3 | [x] | `test_budget_2core_minimum` | 2 physical cores → effective=2, target=2, blas=2, cv=1 |
| 4 | [x] | `test_budget_rtx3090_24gb` | 24GB VRAM → vram_limit=14745, batch=256 |
| 5 | [x] | `test_budget_gtx1060_6gb` | 6GB VRAM → vram_limit=3686, batch=32 |
| 6 | [x] | `test_budget_no_gpu` | No GPU → vram_limit=0, batch=32, xla=False |
| 7 | [x] | `test_budget_disabled_fallback` | `ResourceBudget.disabled()` returns safe minimums (blas=2, cv=1, ram=4) |
| 8 | [x] | `test_effective_cores_formula` | Non-hybrid → physical_cores; hybrid → p_cores + 0.5*e_cores |

---

### `tests/test_pnl_math.py`

**Target**: `pipeline/metrics_eval.py` — pure metric functions

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_sharpe_known_returns` | 12 monthly returns of 1% → Sharpe ≈ 3.46 |
| 2 | [ ] | `test_sharpe_zero_volatility` | Flat returns → handles division by zero gracefully |
| 3 | [ ] | `test_sharpe_negative_returns` | All negative → negative Sharpe |
| 4 | [ ] | `test_sortino_ratio` | Downside-only volatility, known values |
| 5 | [ ] | `test_max_drawdown_simple` | [100, 90, 80, 95, 85] → max DD = 20% |
| 6 | [ ] | `test_max_drawdown_new_high` | [100, 110, 105, 120, 115] → max DD from peak 120 to 115 = 4.17% |
| 7 | [ ] | `test_win_rate_empty` | Empty returns → 0.0 |
| 8 | [ ] | `test_win_rate_perfect` | All positive → 1.0 |
| 9 | [ ] | `test_directional_accuracy` | Predictions exactly match → 1.0; random → ~0.5 |
| 10 | [ ] | `test_hac_std_known_lag` | AR(1) series with known ρ → HAC SE matches Bartlett formula |
| 11 | [ ] | `test_geo_mean_annualized` | 1% per month → ~12.68% annualized |
| 12 | [ ] | `test_estimate_frequency_per_year_weekday` | H1 data on weekdays → ~6048 bars/year |
| 13 | [ ] | `test_estimate_frequency_per_year_crypto` | 24/7 data → ~8760 bars/year |

---

### `tests/test_timeframe_math.py`

**Target**: `config.py` — timeframe pure functions, `pipeline/metrics_eval.py` — frequency estimation

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_period_offset_months` | `period_offset(3, "months")` = 3-month DateOffset |
| 2 | [ ] | `test_period_offset_weeks` | `period_offset(2, "weeks")` = 2-week DateOffset |
| 3 | [ ] | `test_periods_between_monthly` | Jan 2024 to Apr 2024, months → 3 |
| 4 | [ ] | `test_periods_between_edge_case` | Same start/end → 0 |
| 5 | [ ] | `test_convert_month_count_to_periods_weeks` | 3 months → ~13 weeks |
| 6 | [ ] | `test_to_period_freq_weeks` | `to_period_freq("weeks")` → "W" |
| 7 | [ ] | `test_tf_hierarchy_bars_per_day` | M30 → 48, H1 → 24, H4 → 6 |
| 8 | [ ] | `test_tf_hierarchy_mtf_pairs` | M30 fast=H1, M30 slow=H4 |

---

### `tests/test_committee_voting.py`

**Target**: `pipeline/backtester/real_trading_mixin.py` — `_evaluate_with_topn_consensus`, `pipeline/committee_backtester.py` — `_blend_predictions`

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_majority_vote_unanimous_long` | 3 models all predict +1 → consensus = +1 |
| 2 | [ ] | `test_majority_vote_2v1_long` | [+1, +1, -1] → majority +1 → consensus = +1 |
| 3 | [ ] | `test_majority_vote_tie_goes_to_zero` | [+1, -1, 0] → sum = 0 → sign(0) = 0 → flat |
| 4 | [ ] | `test_majority_vote_all_zero` | [0, 0, 0] → consensus = 0 |
| 5 | [ ] | `test_majority_vote_with_news_blending` | Mock news sentiment adjustment to raw predictions |
| 6 | [ ] | `test_weighted_blend_equal_weights` | 2 models, equal weight → average of probabilities |
| 7 | [ ] | `test_weighted_blend_unbalanced` | [0.7, 0.3] weights → weighted average of probs |
| 8 | [ ] | `test_weighted_blend_missing_model` | One model returns None → skipped, renormalized |
| 9 | [ ] | `test_proba_to_trade_below_threshold` | Max prob < confidence_threshold → trade = 0 |
| 10 | [ ] | `test_proba_to_trade_above_threshold` | Max prob >= confidence_threshold → argmax direction |

---

### `tests/test_hardware_discovery.py`

**Target**: `api/hardware.py` — `discover_gpu_vram()`, `_try_tf_vram()`, `_try_pipeline_profile()`

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_discover_via_pynvml` | Mock pynvml returns 24576 MB → `discover_gpu_vram()` returns 24576 |
| 2 | [ ] | `test_discover_via_tf_fallback` | Mock pynvml throws; mock TF returns 16384 → returns 16384 |
| 3 | [ ] | `test_discover_via_pipeline_profiles` | Both pynvml and TF fail; mock HW profile returns 8192 → returns 8192 |
| 4 | [ ] | `test_discover_no_gpu_at_all` | All three probes fail → returns 0 |
| 5 | [ ] | `test_discover_pynvml_bytes_name` | pynvml returns bytes name → decoded to str |
| 6 | [ ] | `test_discover_tf_no_gpus` | TF `list_physical_devices("GPU")` returns [] → falls through to pipeline profile |

---

## Layer 2 — Integration Tests (Component Handshakes)

---

### `tests/test_gatekeeper_integration.py`

**Target**: `api/routers/backtest.py` — VRAM gate + job dispatch, `api/process_manager.py` — submit/allocate/release flow, `api/services/__init__.py` — `create_job_atomic`

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_gpu_job_allocates_vram_and_passes_env` | Submit GPU job → VRAM allocated, `CUDA_VRAM_LIMIT_MB` in env_vars |
| 2 | [ ] | `test_cpu_job_skips_vram_gate` | Submit CPU-only job → no VRAM allocated, no `CUDA_VRAM_LIMIT_MB` in env |
| 3 | [ ] | `test_second_gpu_job_blocked_when_full` | 8192 MB total, 4096 budget → 2nd job gets HTTP 409 with VRAM detail |
| 4 | [ ] | `test_vram_released_on_job_completion` | Job finishes → done_callback fires → VRAM ledger drops |
| 5 | [ ] | `test_vram_released_on_job_exception` | Job crashes → done_callback still fires → VRAM released |
| 6 | [ ] | `test_create_job_atomic_releases_vram_on_failure` | VRAM allocated, then `create_job_atomic` fails → VRAM released in except block |
| 7 | [ ] | `test_stale_jobs_dont_block_gate` | Insert stale job (updated_at 2h ago), submit new → gate passes |
| 8 | [ ] | `test_empty_models_list_handled` | Submit with models=[] → does not crash |
| 9 | [ ] | `test_max_concurrent_cpu_respected` | `max_concurrent_backtests=2`, submit 3 CPU jobs → 3rd gets 409 |
| 10 | [ ] | `test_process_manager_submit_raises_on_uninit` | `pm.submit()` without `pm.initialize()` → RuntimeError |
| 11 | [ ] | `test_env_vars_passed_to_worker_process` | Submit with `env_vars={"CUDA_VRAM_LIMIT_MB": "4096"}` → worker reads from `os.environ` |
| 12 | [ ] | `test_parent_os_environ_not_mutated` | Submit GPU job → parent `os.environ` does NOT contain `CUDA_VRAM_LIMIT_MB` |

---

### `tests/test_api_backtest_endpoint.py`

**Target**: `POST /api/v1/backtest` endpoint validation

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_submit_valid_cpu_backtest` | Valid payload → 200, returns `job_id`, `status="pending"` |
| 2 | [ ] | `test_submit_missing_models` | No models selected → 422 validation error |
| 3 | [ ] | `test_submit_invalid_pair` | Bogus pair → 422 validation error |
| 4 | [ ] | `test_submit_backtest_when_gate_full` | 4 active jobs → 5th returns 409 with detail message |
| 5 | [ ] | `test_get_active_backtests` | After 2 submissions → returns 2 jobs in response |
| 6 | [ ] | `test_get_active_backtests_empty` | No active jobs → returns empty list, total=0 |
| 7 | [ ] | `test_force_stop_existing_job` | Force-stop valid job_id → 200, status="failed" |
| 8 | [ ] | `test_force_stop_nonexistent_job` | Force-stop bogus job_id → 404 |
| 9 | [ ] | `test_get_job_status_completed` | Completed job → status="completed", has result |
| 10 | [ ] | `test_cancel_triggers_process_manager_signal` | Force-stop → `pm.request_cancellation` called with correct job_id |

---

### `tests/test_db_wal_concurrency.py`

**Target**: `pipeline/data_sqlite.py` — concurrent WAL writes, `api/services/__init__.py` — concurrent `create_job_atomic`

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_concurrent_writers_wal_mode` | 5 threads each insert 10 rows → 50 rows total, no SQLITE_BUSY |
| 2 | [ ] | `test_concurrent_readers_during_write` | Writer inserting 1000 rows while 5 readers query → readers never blocked |
| 3 | [ ] | `test_concurrent_create_job_atomic_race` | 10 threads simultaneously `create_job_atomic` with max_active=2 → exactly 2 succeed |
| 4 | [ ] | `test_wal_mode_verified_on_connect` | New connection → `PRAGMA journal_mode` returns "wal" |
| 5 | [ ] | `test_synchronous_normal_set` | New connection → `PRAGMA synchronous` returns 1 (NORMAL) |
| 6 | [ ] | `test_cache_size_64mb` | New connection → `PRAGMA cache_size` returns -64000 |
| 7 | [ ] | `test_cursor_rollback_on_exception` | Insert then force error → row not persisted |
| 8 | [ ] | `test_cursor_commit_on_success` | Insert → connection closed → reopened → row present |
| 9 | [ ] | `test_connection_timeout_respected` | Hold write lock in another connection → new connect waits up to 30s |
| 10 | [ ] | `test_stale_cleanup_transactional` | `_cleanup_stale_jobs` runs, then `create_job_atomic` counts → cleaned jobs excluded |
| 11 | [ ] | `test_job_events_concurrent_write` | 4 workers appending events to same job_id → no lost events |

---

### `tests/test_process_manager_lifecycle.py`

**Target**: `api/process_manager.py` — init/shutdown/submit lifecycle

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_initialize_creates_pools` | After `initialize(2, 1, True)` → CPU pool has 2 workers, GPU pool has 1 |
| 2 | [ ] | `test_initialize_gpu_disabled` | `initialize(gpu_enabled=False)` → GPU pool is None |
| 3 | [ ] | `test_shutdown_cleans_up_pools` | After `shutdown()` → `_cpu_pool` and `_gpu_pool` are None |
| 4 | [ ] | `test_shutdown_signals_cancellation` | Active job → `shutdown()` sets cancel event for that job |
| 5 | [ ] | `test_shutdown_escalation_timeout` | Job doesn't stop → after 3s, pools are force-shutdown |
| 6 | [ ] | `test_atexit_handler_registered` | `initialize()` → `atexit._exithandlers` contains `shutdown` |
| 7 | [ ] | `test_gpu_job_routed_to_gpu_pool` | GPU models → submitted to `_gpu_pool` |
| 8 | [ ] | `test_cpu_job_routed_to_cpu_pool` | CPU models → submitted to `_cpu_pool` |
| 9 | [ ] | `test_gpu_job_falls_back_to_cpu` | GPU models but `_gpu_pool is None` → submitted to `_cpu_pool` |
| 10 | [ ] | `test_active_count_reflects_submissions` | 3 submits → `active_count` == 3 |
| 11 | [ ] | `test_active_job_ids_returns_keys` | 2 submits → `active_job_ids` returns both IDs |

---

### `tests/test_env_vars_isolation.py`

**Target**: `api/process_manager.py:257-288` — `_run_backtest_in_worker` env var application

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_mlb_threads_applied_in_child` | `env_vars={"MLB_THREADS": "2"}` → child `os.environ["MLB_THREADS"]` == "2" |
| 2 | [ ] | `test_cuda_vram_limit_applied_in_child` | `env_vars={"CUDA_VRAM_LIMIT_MB": "4096"}` → child reads it |
| 3 | [ ] | `test_threadpool_limits_called_in_child` | `env_vars` with `MLB_THREADS` → `threadpool_limits` called with correct limit |
| 4 | [ ] | `test_parent_env_not_mutated_by_submit` | Submit with env_vars → parent `os.environ` unchanged |
| 5 | [ ] | `test_empty_env_vars_noop` | `env_vars={}` → no crash, no env changes in child |
| 6 | [ ] | `test_env_vars_cast_to_string` | `env_vars={"MLB_THREADS": 4}` (int) → cast to "4" in child |
| 7 | [ ] | `test_multiple_env_vars_all_applied` | 5 env vars → all present in child environ |

---

## Layer 3 — ML & Data Pipeline Tests (Determinism & Leakage)

---

### `tests/test_determinism.py`

**Target**: Seed reproducibility across the full pipeline

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_logistic_deterministic_same_seed` | Two runs with seed=42 → identical model weights (coef_ allclose) |
| 2 | [ ] | `test_logistic_deterministic_different_seed` | Seed=42 vs seed=43 → different weights |
| 3 | [ ] | `test_xgboost_deterministic_same_seed` | Same seed → identical `predict_proba` output |
| 4 | [ ] | `test_random_forest_deterministic` | Same seed → identical tree structures |
| 5 | [ ] | `test_lstm_deterministic_same_seed` | Same seed, TF deterministic ops → identical predictions (allclose) |
| 6 | [ ] | `test_cnn_deterministic_same_seed` | Same seed → identical output probabilities |
| 7 | [ ] | `test_transformer_deterministic_same_seed` | Same seed → identical output |
| 8 | [ ] | `test_set_global_determinism_sets_all_seeds` | Call with seed=99 → verify `random.random()`, `np.random.rand()`, `tf.random.uniform()` are deterministic |
| 9 | [ ] | `test_feature_cache_key_deterministic` | Same config, same file → identical SHA256 cache key |
| 10 | [ ] | `test_optuna_sampler_deterministic_seed` | Same sampler_seed → same trial sequence |
| 11 | [ ] | `test_walk_forward_fold_determinism` | Same full pipeline, same seed → identical OOS Sharpe across all folds |
| 12 | [ ] | `test_hpo_same_seed_same_best_params` | Same seed, same data → Optuna best_params match |

---

### `tests/test_vram_lock_determinism.py`

**Target**: `pipeline/runtime.py:156-191` — `apply_vram_lock()` + logical device configuration

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_vram_lock_sets_logical_device_config` | With `CUDA_VRAM_LIMIT_MB=4096` → TF `set_logical_device_configuration` called with memory_limit=4096 |
| 2 | [ ] | `test_vram_lock_falls_back_to_memory_growth` | Logical config fails → `set_memory_growth(True)` called |
| 3 | [ ] | `test_vram_lock_noop_without_env_var` | No `CUDA_VRAM_LIMIT_MB` set → only `set_memory_growth(True)` called |
| 4 | [ ] | `test_vram_lock_noop_without_gpu` | No TF GPU devices → function returns early, no config calls |
| 5 | [ ] | `test_vram_lock_model_output_unchanged_with_lock` | Same seed, same data, with/without VRAM lock → identical predictions |
| 6 | [ ] | `test_vram_lock_module_level_skips_memory_growth` | `CUDA_VRAM_LIMIT_MB=4096` at import time → runtime.py does NOT call `set_memory_growth` at module level |
| 7 | [ ] | `test_vram_lock_applied_before_tf_import_in_child` | Child process sets `CUDA_VRAM_LIMIT_MB` before TF import → logical config takes effect |

---

### `tests/test_data_leakage_extended.py`

**Target**: Additional leakage checks beyond existing `test_walk_forward_integrity.py`

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_label_computed_on_train_only` | Labels built in walk-forward → no label value depends on test window data |
| 2 | [ ] | `test_scaler_fit_on_train_only` | StandardScaler mean/std from train only → test data not in fit() |
| 3 | [ ] | `test_feature_shift_invariant` | Same input shifted 1 bar → features shift exactly 1 bar |
| 4 | [ ] | `test_rolling_indicator_no_peek` | 20-bar SMA computed with `.shift(0)` → contains future bar (FAIL — validates guard) |
| 5 | [ ] | `test_mtf_indicators_pre_shifted` | H4 MA on H1 data → lagged to prevent look-ahead |
| 6 | [ ] | `test_embargo_creates_data_gap` | Embargo bars exist between train and test in walk-forward splits |
| 7 | [ ] | `test_news_future_timestamp_rejected` | News article with timestamp > prediction bar → excluded from features |
| 8 | [ ] | `test_ensemble_calibration_on_train_tail_only` | Calibration temperature fit on train[-ncal:] only, not on test data |
| 9 | [ ] | `test_bidirectional_lstm_default_false` | Default LSTM parameter `bidirectional=False` to avoid temporal leakage |
| 10 | [ ] | `test_execution_delay_first_bar_zero` | First bar position is ALWAYS 0.0 (1-bar delay guarantee) |
| 11 | [ ] | `test_execution_delay_recovery_from_nan` | NaN in raw_pred doesn't cascade into delayed pred forever |

---

## Layer 4 — Concurrency & Stress Tests (The Fire Drill)

---

### `tests/test_vram_gatekeeper_stress.py`

**Target**: Gatekeeper under load

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_queue_5_gpu_jobs_with_2_slots` | 5 GPU jobs submitted concurrently, 2 VRAM slots → 2 succeed, 3 get 409 |
| 2 | [ ] | `test_vram_released_between_submissions` | Submit job1 (fills VRAM), wait for completion, submit job2 → succeeds |
| 3 | [ ] | `test_simultaneous_allocate_no_oversubscription` | 10 threads simultaneously `allocate_vram(2048)` on 8192 total → sum_allocated ≤ 8192 |
| 4 | [ ] | `test_vram_ledger_consistent_after_100_rapid_cycles` | Allocate+release cycle × 100 → final used == 0 |
| 5 | [ ] | `test_mixed_cpu_gpu_submissions` | 3 CPU + 2 GPU jobs → CPU pass through, GPU gated by VRAM |
| 6 | [ ] | `test_rapid_force_stop_releases_vram` | Submit GPU job → force-stop before completion → VRAM released immediately |

---

### `tests/test_cancel_isolation.py`

**Target**: Scoped cancellation — canceling one job doesn't kill others

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_cancel_backtest_does_not_kill_other` | 2 backtests running → force-stop backtest A → backtest B continues |
| 2 | [ ] | `test_force_stop_updates_only_target_job` | Force-stop job_id="A" → only job A marked "failed", job B still "running" |
| 3 | [ ] | `test_clear_pending_queue_scoped` | `clear_pending_queue("backtest")` → only backtest-type jobs affected |
| 4 | [ ] | `test_joblib_cleanup_does_not_kill_global` | Cleanup job A → job B's joblib workers still functional |
| 5 | [ ] | `test_process_manager_cancel_scoped` | `pm.request_cancellation(job_a)` → only job_a cancel event set |
| 6 | [ ] | `test_cancel_event_isolated_per_job` | Cancel event dict: job A=True, job B=False (not affected) |
| 7 | [ ] | `test_cancel_during_hpo_trial` | Force-stop during Optuna trial → trial stops cleanly, no orphaned processes |
| 8 | [ ] | `test_simultaneous_force_stop_two_jobs` | Stop job A and B simultaneously → both marked failed, no race errors |

---

### `tests/test_process_crash_recovery.py`

**Target**: VRAM ledger integrity when child processes crash

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_child_process_crash_releases_vram` | Worker process raises uncaught exception → done_callback fires → VRAM released |
| 2 | [ ] | `test_child_process_killed_releases_vram` | `pool.shutdown(cancel_futures=True)` → VRAM released for cancelled jobs |
| 3 | [ ] | `test_worker_oserror_during_import` | Worker dies during import (`sys.exit(1)`) → VRAM released |
| 4 | [ ] | `test_worker_memory_error_releases_vram` | Worker raises MemoryError → VRAM released, ledger correct |
| 5 | [ ] | `test_vram_ledger_after_pool_shutdown` | `pm.shutdown()` with active jobs → all VRAM returned to ledger |
| 6 | [ ] | `test_vram_not_double_released_on_crash_then_complete` | Crash triggers done_callback, then manual release called → no negative VRAM |

---

### `tests/test_pipeline_stress.py`

**Target**: Full pipeline under concurrent load

| # | Status | Test Name | Verification |
|---|--------|-----------|-------------|
| 1 | [ ] | `test_four_cpu_backtests_concurrent` | 4 simultaneous logistic backtests → all complete, no SQLITE_BUSY |
| 2 | [ ] | `test_cpu_backtests_do_not_block_api` | Backtest running → `GET /active` returns immediately |
| 3 | [ ] | `test_websocket_events_not_cross_contaminated` | Backtest A events don't appear in Backtest B's WebSocket stream |
| 4 | [ ] | `test_redis_channels_per_job_isolated` | `job:A` pub/sub events only go to subscribers of channel `job:A` |
| 5 | [ ] | `test_db_connections_cleaned_up` | 10 rapid backtest submissions → no leaked SQLite connections |
| 6 | [ ] | `test_joblib_temp_dirs_not_colliding` | 3 concurrent backtests → each has unique `JOBLIB_TEMP_FOLDER` |
| 7 | [ ] | `test_thread_budget_per_process` | 4 concurrent CPU backtests on 8-core → each gets `MLB_THREADS=2` |

---

## File Creation Summary

| # | File | Layer | Target Modules | Cases |
|---|------|-------|---------------|-------|
| 1 | `tests/test_gatekeeper_vram_math.py` | Unit | `api/process_manager.py` allocate/release | 14 |
| 2 | `tests/test_resource_budget_math.py` | Unit | `pipeline/resource_budget.py` | 8 |
| 3 | `tests/test_pnl_math.py` | Unit | `pipeline/metrics_eval.py` pure functions | 13 |
| 4 | `tests/test_timeframe_math.py` | Unit | `config.py` timeframe functions | 8 |
| 5 | `tests/test_committee_voting.py` | Unit | committee voting/blending | 10 |
| 6 | `tests/test_hardware_discovery.py` | Unit | `api/hardware.py` | 6 |
| 7 | `tests/test_gatekeeper_integration.py` | Integration | router + ProcessManager + JobManager | 12 |
| 8 | `tests/test_api_backtest_endpoint.py` | Integration | `POST /api/v1/backtest` + related | 10 |
| 9 | `tests/test_db_wal_concurrency.py` | Integration | `pipeline/data_sqlite.py` + JobManager | 11 |
| 10 | `tests/test_process_manager_lifecycle.py` | Integration | `api/process_manager.py` lifecycle | 11 |
| 11 | `tests/test_env_vars_isolation.py` | Integration | ProcessManager env var passing | 7 |
| 12 | `tests/test_determinism.py` | ML Pipeline | Model seeds, pipeline reproducibility | 12 |
| 13 | `tests/test_vram_lock_determinism.py` | ML Pipeline | `pipeline/runtime.py` VRAM lock | 7 |
| 14 | `tests/test_data_leakage_extended.py` | ML Pipeline | Feature/label/execution leakage | 11 |
| 15 | `tests/test_vram_gatekeeper_stress.py` | Concurrency | Gatekeeper under load | 6 |
| 16 | `tests/test_cancel_isolation.py` | Concurrency | Scoped cancellation | 8 |
| 17 | `tests/test_process_crash_recovery.py` | Concurrency | Crash → VRAM release | 6 |
| 18 | `tests/test_pipeline_stress.py` | Concurrency | Full pipeline concurrent load | 7 |

**Total: 18 new test files, ~160 test cases across 4 testing layers.**
