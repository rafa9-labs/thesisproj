# FX MLBacktester — Refactoring & UI Master Plan

> **Last Updated**: 2024-04-04  
> **Status**: Phase 0 (Prerequisites) — In Progress

---

## 1. Program Overview

FX MLBacktester is a production-grade, bar-based foreign exchange ML trading pipeline that enables rigorous, leak-free comparison of diverse model families — from classical classifiers (Logistic Regression, SVM, Random Forest, XGBoost) through deep learning architectures (CNN, LSTM, Transformer) and ensemble strategies (stacked meta-learners, regime-adaptive Mixture-of-Experts) to reinforcement learning (Dueling DQN with Prioritized Experience Replay).

### Core Value Proposition
Walk-forward, cost-aware backtesting with causal feature/label construction, ensuring all metrics reflect realistic after-cost, after-slippage trading performance with a strict one-bar execution delay.

### Key Files

| File | Role | Lines |
|------|------|-------|
| `MLBacktesterNoWFO.py` | Main orchestrator (training, HPO, monthly walk-forward, evaluation) | ~20,000+ |
| `utilsNoWFO.py` | Feature engineering, metrics, calibration, gating, I/O, plotting | ~6,600 |
| `tuningNoWFO.py` | Optuna HPO search space definition | ~4,500 |
| `models/cnn.py` | 1D-CNN classifier (dual Conv1D + GlobalAveragePooling) | ~200 |
| `models/lstm.py` | Stacked LSTM (bidirectional, dropout, batch norm) | ~200 |
| `models/transformer.py` | Time-series Transformer (sinusoidal PE, multi-head attention) | ~250 |
| `models/ensemble_cnn_lstm_xgboost.py` | CNN+LSTM → XGBoost stacking ensemble | ~500 |
| `models/ensemble_adaptive_regime.py` | Regime-adaptive Mixture-of-Experts | ~600 |
| `rl/dqn_agent.py` | Dueling Double-DQN with PER | ~400 |
| `rl/environment.py` | Trading environment (gym-style) | ~300 |
| `rl/replay_buffer.py` | Prioritized Experience Replay buffer | ~150 |
| `deep_subprocess_worker.py` | Subprocess worker for deep model training | ~200 |
| `run_one_month_worker.py` | Subprocess worker for monthly evaluation | ~200 |
| `configs/feature_config.json` | Feature engineering parameters | — |
| `configs/dqn_grid_config.json` | DQN hyperparameter grid | — |

### Technology Stack
- **Language**: Python 3
- **ML/DL**: TensorFlow/Keras, scikit-learn, XGBoost
- **HPO**: Optuna (TPE sampler)
- **Data**: pandas, NumPy, pyarrow
- **Indicators**: `ta` library
- **Parallelism**: joblib, concurrent.futures, threadpoolctl
- **Hardware**: GPU-accelerated via TensorFlow CUDA; mixed precision (fp16) supported

---

## 2. Identified Bottlenecks & Bad Practices

### Critical (🔴)
| ID | Issue | File(s) |
|----|-------|---------|
| A1 | Monolithic orchestrator (~965 KB, 20K+ lines) | `MLBacktesterNoWFO.py` |
| A2 | Monolithic utils (~6,600 lines, cyclomatic >40) | `utilsNoWFO.py` |
| A4 | Subprocess spawning on Windows (5-10s per spawn) | `deep_subprocess_worker.py`, `run_one_month_worker.py` |
| B1 | `os.environ` as global config scattered across 5+ files | Multiple |
| B4 | No `if __name__ == "__main__"` guard → Windows crashes | `MLBacktesterNoWFO.py` |
| B6 | Zero unit tests | — |

### High (🟠)
| ID | Issue | File(s) |
|----|-------|---------|
| A6 | No lazy imports (15-30s cold start for TF) | `MLBacktesterNoWFO.py` |
| B2 | `print()` instead of `logging` module (hundreds of calls) | All files |
| B7 | Inconsistent model interfaces (massive if/elif chains) | `MLBacktesterNoWFO.py` |
| C1 | Redundant feature recomputation (hours of waste) | `MLBacktesterNoWFO.py` |
| D1 | No memory cleanup between model trains (GPU OOM) | `MLBacktesterNoWFO.py` |
| E1 | BLAS thread oversubscription race conditions | `MLBacktesterNoWFO.py`, `tuningNoWFO.py` |
| F1 | Hardcoded model lists in source code | `MLBacktesterNoWFO.py` |
| F2 | No structured progress reporting | All files |

### Medium (🟡)
| ID | Issue | File(s) |
|----|-------|---------|
| A3 | Repeated TF/GPU initialization (3x) | Multiple |
| A5 | Global CSV cache with `.copy()` (2.5GB transient) | `MLBacktesterNoWFO.py` |
| C2 | Full DataFrame copied multiple times in evaluation | `utilsNoWFO.py` |
| D2 | Optuna study lives in memory (hundreds of MB) | `tuningNoWFO.py` |
| F3 | Results scattered across 50+ files | All output dirs |

### Low (🟢)
| ID | Issue | File(s) |
|----|-------|---------|
| B3 | Duplicate import statements | `MLBacktesterNoWFO.py` |
| B5 | Magic numbers everywhere | All files |
| C3 | DQN environment creates arrays on every step | `rl/environment.py` |
| E2 | `LD_LIBRARY_PATH` manipulation (Linux-only) | `MLBacktesterNoWFO.py` |

---

## 3. Phased Implementation Plan

### Phase 0: Prerequisites & Foundation (Current)
**Branch**: `refactor/phase0-foundation`  
**Goal**: Fix critical Windows blockers and establish clean project structure.

| Step | Description | Addresses | Status |
|------|-------------|-----------|--------|
| 0.1 | Create `PROJECT_PLAN.md` (this file) | — | ✅ |
| 0.2 | Create `config.py` — centralized settings from `.env` + JSON | B1, B5 | ⬜ |
| 0.3 | Add `if __name__ == "__main__"` guard to orchestrator | B4 | ⬜ |
| 0.4 | Create `logging_config.py` — replace `print()`/`log_print()` with `logging` | B2 | ⬜ |
| 0.5 | Remove duplicate imports and redundant TF initialization | B3, A3 | ⬜ |
| 0.6 | Add lazy imports for TF/XGBoost (15-30s faster startup) | A6 | ⬜ |
| 0.7 | Fix `LD_LIBRARY_PATH` and Linux-only assumptions | E2 | ⬜ |
| 0.8 | Convert file paths to `pathlib.Path` throughout | Windows | ⬜ |

### Phase 1: Windows Native & Worker Consolidation
**Branch**: `refactor/phase1-windows-native` (branched from Phase 0)  
**Goal**: Make the program run natively on Windows without WSL.

| Step | Description | Addresses | Status |
|------|-------------|-----------|--------|
| 1.1 | Replace subprocess workers with `ProcessPoolExecutor` (spawn-safe) | A4 | ⬜ |
| 1.2 | Consolidate `deep_subprocess_worker.py` + `run_one_month_worker.py` → `workers.py` | A4 | ⬜ |
| 1.3 | Fix multiprocessing pickling issues (lambdas, closures, TF models) | B4, A4 | ⬜ |
| 1.4 | Test full pipeline on Windows (all model types) | — | ⬜ |

### Phase 2: Architecture — Extract Pipeline Modules
**Branch**: `refactor/phase2-pipeline-modules`  
**Goal**: Break monoliths into clean, testable modules.

| Step | Description | Addresses | Status |
|------|-------------|-----------|--------|
| 2.1 | Extract `pipeline/data_loader.py` from MLBacktesterNoWFO | A1 | ⬜ |
| 2.2 | Extract `pipeline/features.py` from utilsNoWFO | A1, A2 | ⬜ |
| 2.3 | Extract `pipeline/labels.py` (triple-barrier) | A2 | ⬜ |
| 2.4 | Extract `pipeline/trainer.py` (model training + calibration) | A1 | ⬜ |
| 2.5 | Extract `pipeline/evaluator.py` from utilsNoWFO | A2 | ⬜ |
| 2.6 | Extract `pipeline/backtest.py` (walk-forward loop) | A1 | ⬜ |
| 2.7 | Create `models/base_model.py` (BaseModel ABC) | B7 | ⬜ |
| 2.8 | Wrap all models to conform to BaseModel | B7 | ⬜ |
| 2.9 | Add feature caching (keyed by data_hash + config_hash) | C1 | ⬜ |
| 2.10 | Add explicit memory cleanup between model trains | D1 | ⬜ |

### Phase 3: Simplification & Cleanup
**Branch**: `refactor/phase3-simplification`  
**Goal**: Reduce complexity, remove dead code, simplify HPO.

| Step | Description | Addresses | Status |
|------|-------------|-----------|--------|
| 3.1 | Strip dead code from `compute_full_evaluation_metrics` | A2 | ⬜ |
| 3.2 | Move optional patches (TWAP, kill-switch, regime) to strategy pattern | A2 | ⬜ |
| 3.3 | Simplify Optuna search space (remove legacy profile, reduce dimensions) | A2 | ⬜ |
| 3.4 | Extract evaluation patches into plugin architecture | A2 | ⬜ |
| 3.5 | Replace magic numbers with named constants | B5 | ⬜ |
| 3.6 | Add `tests/test_causality.py` and `tests/test_metrics.py` | B6 | ⬜ |

### Phase 4: Desktop UI Application
**Branch**: `feature/phase4-desktop-ui`  
**Goal**: Professional PySide6 desktop application.

| Step | Description | Addresses | Status |
|------|-------------|-----------|--------|
| 4.1 | Set up PySide6 project structure (`ui/` package) | — | ⬜ |
| 4.2 | Build main window with tab layout | — | ⬜ |
| 4.3 | Build Data tab (CSV loading, preview, timeframe selection) | F1 | ⬜ |
| 4.4 | Build Models tab (checkbox grid, model-specific params) | F1 | ⬜ |
| 4.5 | Build Features tab (indicator toggles, windows, lag config) | F1 | ⬜ |
| 4.6 | Build HPO tab (Optuna trials, TA profile, train months) | F1 | ⬜ |
| 4.7 | Build Backtest tab (test months, seeds, costs, triple-barrier) | F1 | ⬜ |
| 4.8 | Build Results tab (equity plots, metric tables, rankings) | F3 | ⬜ |
| 4.9 | Implement `QThread` workers for long-running tasks | F2 | ⬜ |
| 4.10 | Wire log console to `logging` handler | B2 | ⬜ |
| 4.11 | Add progress bar with ETA estimation | F2 | ⬜ |
| 4.12 | Add export to CSV/PNG buttons | F3 | ⬜ |
| 4.13 | Create `main.py` entry point (UI or CLI mode) | — | ⬜ |

---

## 4. Git Branching Strategy

```
main
├── refactor/phase0-foundation          ← Current
│   └── refactor/phase1-windows-native  ← Branch from Phase 0
│       └── refactor/phase2-pipeline-modules
│           └── refactor/phase3-simplification
│               └── feature/phase4-desktop-ui
```

- Each phase branches from the **end of the previous phase**
- Each phase is a **PR** with review + testing before merge to `main`
- Commit messages follow conventional commits: `feat:`, `fix:`, `refactor:`, `chore:`

---

## 5. Validation Checkpoints

### Phase 0 Validation
- [ ] `python -c "import config"` works without errors
- [ ] `python -c "import logging_config"` works without errors
- [ ] `python MLBacktesterNoWFO.py` still runs (smoke test with 1 model, 1 month)
- [ ] No duplicate TF initialization warnings in console output
- [ ] All file paths work on Windows (no backslash/forward slash issues)

### Phase 1 Validation
- [ ] Full backtest runs on Windows without WSL
- [ ] No `subprocess` calls remain in the hot path
- [ ] Multiprocessing workers don't crash on spawn
- [ ] All model types complete at least 1 month of walk-forward

### Phase 2 Validation
- [ ] Extracted modules pass `pytest tests/`
- [ ] Walk-forward output is **bit-for-bit identical** to pre-refactor output
- [ ] Feature caching reduces repeated computation by >80%

### Phase 3 Validation
- [ ] `compute_full_evaluation_metrics` is <500 lines
- [ ] Optuna search space has <25 dimensions
- [ ] Causality tests pass
- [ ] Metrics tests pass

### Phase 4 Validation
- [ ] UI launches via `python main.py`
- [ ] All tabs functional and responsive
- [ ] Long-running backtest doesn't freeze UI
- [ ] Results tab shows correct equity curves and metrics
- [ ] Export buttons produce valid CSV/PNG files