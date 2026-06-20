# KodaQuant Upgrade Plan — de Prado AFML Alignment

> **Goal**: Transition from heuristic rules to probabilistic ML, following Marcos Lopez de Prado's
> *Advances in Financial Machine Learning* and *Machine Learning for Asset Managers*.

---

## Audit Summary

| de Prado AFML Topic | KodaQuant Status | Verdict |
|---------------------|-----------------|---------|
| Triple-Barrier Labeling | Vol-scaled barriers + vertical barrier + HPO over params | MATCH |
| Meta-Labeling (Ch. 3.6) | Not implemented — confidence-threshold gate only | **MISSING** |
| Purged K-Fold CV (Ch. 7) | Custom chronological blocked CV, embargo gap but no purging, ensemble CV uses `shuffle=True` | **VIOLATION** |
| Regime Detection (Ch. 4, 15) | Rule-based ADX/EMA/ATR thresholds | **BRITTLE** |
| Conviction Sizing (Ch. 3, 10) | Hardcoded 3-tier step function (0.5/1.0/1.5) | **HARDCODED** |
| Fractional Differentiation (Ch. 5) | Correct algorithm, but `d` tuned by Optuna for Sharpe (not ADF) | **WRONG TARGET** |
| Feature Importance (Ch. 8) | Not implemented | **MISSING** |

---

## Corrected Roadmap

**Foundation rule**: You cannot build meta-labeling on leaky primary models. CV must be fixed first.

| Phase | Task | Library | Impact | Effort |
|-------|------|---------|--------|--------|
| **P0** | Purged Blocked K-Fold CV + Shuffle Fix | Pure numpy/pandas, `TimeSeriesSplit` | Stops label leakage in HPO and ensemble | 2-3h |
| **P1** | Meta-Labeling (binary secondary model) | LightGBM (in stack) | Filters false positives — biggest alpha gain | 3-4h |
| **P2** | HMM Regime Detection | `hmmlearn` (new dep, MIT) | Probabilistic states replace lagging ADX rules | 3-4h |
| **P3** | Sigmoid Conviction Sizing | `scipy.optimize` (in stack) | Smooth sizing curve, no boundary cliffs | 1-2h |
| **P4** | ADF Floor for Fracdiff `d` | `statsmodels` (in stack) | Correct optimization target per de Prado Ch.5 | 0.5h |
| **P5** | MDA Feature Importance | scikit-learn (in stack) | Drop noise features, faster inference | 1-2h |

### New Library

| Library | License | Reason |
|---------|---------|--------|
| `hmmlearn` | MIT | GaussianHMM for regime detection. 0 deps beyond numpy/scipy. |

### Libraries NOT Added

- `tslearn` — overkill; don't need DTW clustering
- `sktime` — heavy dependency tree, conflicts with custom CV
- `MlFinLab` paid — algorithms we need are in AFML book code snippets

---

## Phase 0 — Purged Blocked K-Fold CV (Foundation)

### P0a: Fix `shuffle=True` in Ensemble CV

**File**: `models/ensemble_cnn_lstm_xgboost.py:552-565`

**Problem**: `StratifiedKFold(n_splits=..., shuffle=True, random_state=seed)` and fallback
`KFold(n_splits=..., shuffle=True, random_state=seed)` randomize time order, allowing
future bars into training and past bars into validation.

**Fix**: Replace with `TimeSeriesSplit(n_splits=...)`. Acknowledged tradeoff: expanding windows
mean early folds have smaller training sets. This is mathematically safe. A full
`PurgedBlockedKFold` with training on both sides of validation is the eventual upgrade.

### P0b: Add `purge_train_set()` to Mini-Block CV

**File**: `pipeline/backtester/run_mixin.py:1290-1306`

**Problem**: The embargo gap (`tr_end_idx = split - embargo_bars`) is already enforced
at runtime via `embargo_bars = max(cv_embargo_bars, tb_max_holding)`. However, training
observations whose forward triple-barrier label window touches the validation set still
leak information. Example: bar at iloc=`split - embargo_bars` has label window
`[split-embargo_bars, split-embargo_bars+horizon]`. With embargo=horizon this reaches
exactly to `split` (first validation bar).

**Fix**: Purge any training bar where `iloc + label_horizon_bars >= val_start_iloc`.

```python
def purge_train_set(train_data, val_start_iloc, label_horizon_bars):
    """Remove training observations whose label window overlaps validation."""
    if label_horizon_bars <= 0 or len(train_data) == 0:
        return train_data
    # Both sides are in train_data's iloc space
    keep_mask = np.arange(len(train_data)) + label_horizon_bars < val_start_iloc
    return train_data.iloc[keep_mask]
```

Apply to both mini-block and monthly-roll paths.

### P0c: Default `cv_embargo_bars` = 48

**File**: `pipeline/metrics_tuples.py:507`

**Note**: Runtime code at `run_mixin.py:721-724` already does
`embargo_bars = max(cv_embargo_bars, tb_max_holding)`, overriding the config default.
This change is belt-and-suspenders: ensures the config file itself reflects the
correct default. Also flags `cv_embargo_frac` at line 508 as dead config (never read).

### P0d: Purge Integrity Tests

**File**: `tests/test_walk_forward_integrity.py` (extends existing `TestEmbargoEnforcement`)

**Tests**:
1. `test_purge_removes_label_overlap` — synthetic data, verify no training bar's
   `iloc + horizon >= val_start_iloc` remains after purging.
2. `test_purge_preserves_bars_before_threshold` — all bars where
   `iloc + horizon < val_start_iloc` are preserved.
3. `test_purge_noop_when_horizon_zero` — when `label_horizon=0`, no bars removed.

---

## Phase 1 — Meta-Labeling (Binary Secondary Model)

**Built on clean OOS predictions from Phase 0.**

### Architecture

```
Primary Model → [P(short), P(flat), P(long)]  ← raw 3-class probabilities
                                                    ↓
Meta-Labeler  ←  features + primary_probas       ← binary target
                ↓
        P(trade_is_winner | signal)
```

### Input Features

1. All standard features the primary model received
2. Primary model's raw 3-class probability array `[P_short, P_flat, P_long]` — NOT just argmax
3. `P_long - P_short` (conviction spread)
4. `1 - P_flat` (directional commitment — how far from neutral)
5. Market regime ID (from HMM, after Phase 2)
6. Recent bar volatility

### Training Target

Given triple-barrier labeled trade at bar `t`:
- If primary predicted `+1` (long): meta-label = `1` if upper barrier hit first, `0` if lower/vertical hit
- If primary predicted `-1` (short): meta-label = `1` if lower barrier hit first, `0` if upper/vertical hit
- If primary predicted `0` (flat): skip (no trade to meta-label)

### Inference Gating

```python
if primary_signal != 0:
    p_win = meta_model.predict_proba(features + primary_probas)[0, 1]
    if p_win >= 0.50:
        take_trade(primary_signal)
    else:
        stay_flat()
```

### Integration Points
- **Training**: `pipeline/committee_backtester.py` — alongside `_blend_predictions` and `_proba_to_trade`
- **Inference**: `trading/live_committee_runner.py` — gate between steps 4 (signal conversion) and 5 (conviction multiplier)

---

## Phase 2 — HMM Regime Detection

**Replaces `_classify_regime()` at `live_committee_runner.py:333-385`.**

### Architecture

```python
from hmmlearn.hmm import GaussianHMM

class HMMRegimeDetector:
    def __init__(self, n_states: Optional[int] = None):
        # n_states determined by BIC scan over [3, 10]
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,       # FIXED SEED — anchors semantics
            init_params="stmc",
        )
```

### Input Features (backward-looking only)

- 20-bar rolling returns (mean and std)
- 20-bar realized volatility
- Spread (ask - bid) / mid
- 5-bar return autocorrelation

### Semantic Consistency Across Folds

**Problem**: HMM re-initialization on each fold can flip state meanings (State 2 = "trend_up" in fold 1, "sideways" in fold 2).

**Fix**: Train HMM once on the full first-fold training data. Freeze and serialize. For all subsequent folds and live trading, load the frozen HMM and call `.predict_proba()` — no refitting. Only re-optimize on explicit quarterly refresh.

### BIC for n_states

```python
def select_n_states(features, max_states=10):
    bics = [GaussianHMM(n, random_state=42).fit(features).bic(features)
            for n in range(3, max_states + 1)]
    return np.argmin(bics) + 3
```

---

## Phase 3 — Sigmoid Conviction Sizing

**Replaces hardcoded 3-tier at `live_committee_runner.py:241-246`.**

### Fit from OOS Data

```python
from scipy.optimize import curve_fit

def logistic(x, L, k, c):
    return L / (1 + np.exp(-k * (x - c)))

# Bin OOS trades by primary model confidence, compute avg PnL per bin
# Fit sigmoid: position_multiplier vs max_probability
popt, _ = curve_fit(logistic, prob_bins, avg_pnl_per_bin,
                     p0=[1.5, 10.0, 0.65],
                     bounds=([0.5, 1.0, 0.50], [2.0, 30.0, 0.85]))

L_opt, k_opt, c_opt = popt
```

### Application

```python
size = base_size * logistic(max_prob, L_opt, k_opt, c_opt)
```

Produces smooth sizing: 0.649-confidence → ~0.98x, not a cliff from 0.50x to 1.0x at 0.65.

---

## Phase 4 — ADF Floor for Fracdiff `d`

**File**: `pipeline/tuning/sampler.py:343-347`

### Problem

Optuna tunes `fracdiff_d` for Sharpe ratio. Lower `d` = more trending bias = artificially
inflated backtest Sharpe. de Prado's approach: find minimum `d` where `ADF p < 0.05`
(stationarity achieved), use that as the floor.

### Fix

```python
from statsmodels.tsa.stattools import adfuller

def find_min_stationary_d(price_series, d_range=(0.05, 0.95, 0.05)):
    for d in np.arange(*d_range):
        fd = fracdiff(price_series, d=d)
        if adfuller(fd.dropna())[1] < 0.05:
            return d
    return 0.95

# In sampler:
d_floor = find_min_stationary_d(raw_price)
params["fracdiff_d"] = trial.suggest_float("fracdiff_d", d_floor, 0.9, step=0.05)
```

---

## Phase 5 — MDA Feature Importance

**Post-training purged Mean Decrease Accuracy (MDA).**

For each feature:
1. Shuffle feature values across observations (destroying signal)
2. Measure accuracy drop on purged validation set
3. Features with negative MDA (accuracy increased when shuffled) are noise — drop them

**Library**: scikit-learn (already in stack). Implement per de Prado AFML Ch. 8.

---

## Implementation Status

| Phase | Task | Status |
|-------|------|--------|
| P0a | `shuffle=True` → `TimeSeriesSplit` in ensemble CV | ✅ |
| P0b | `purge_train_set()` in mini-block CV | ✅ |
| P0c | Default `cv_embargo_bars` = 48 | ✅ |
| P0d | Purge integrity tests | ✅ |
| P1 | Meta-Labeling binary classifier | ✅ |
| P2 | HMM regime detection | ✅ |
| P3 | Sigmoid conviction sizing | ✅ |
| P4 | ADF floor for fracdiff `d` | ✅ |
| P5 | MDA feature importance | ✅ |
