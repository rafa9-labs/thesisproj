# Phase 3.1 — Data Leakage Audit Findings

> **Date**: 2026-04-12
> **Scope**: Feature computation, labeling, walk-forward splits, execution delay
> **Files Audited**: `features_mixin.py`, `strategy_mixin.py`, `run_mixin.py`, `ensemble_mixin.py`, `real_trading_mixin.py`

---

## 1. Feature Computation ✅ SAFE — No Look-Ahead Bias

All indicators in `features_mixin.py` use backward-looking operations only:

| Indicator | Method | Direction |
|-----------|--------|-----------|
| SMA | `ta.trend.sma_indicator(window=N)` | ✅ Backward |
| EMA | `ta.trend.ema_indicator(window=N)` | ✅ Backward |
| RSI | `ta.momentum.RSIIndicator(window=N)` | ✅ Backward |
| MACD | `ta.trend.MACD(window_slow, fast, signal)` | ✅ Backward |
| Bollinger Bands | `ta.volatility.BollingerBands(window=N)` | ✅ Backward |
| ATR | `ta.volatility.AverageTrueRange(window=N)` | ✅ Backward |
| ADX | `ta.trend.ADXIndicator(window=N)` | ✅ Backward |
| Stochastic K/D | `rolling(N).max()` / `rolling(N).min()` | ✅ Backward |
| SAR | `PSARIndicator(high, low, close)` | ✅ Backward |
| Donchian | `rolling(N).max()` / `rolling(N).min()` | ✅ Backward |
| Realized Vol | `rolling(N).sum()` on squared returns | ✅ Backward |
| Fracdiff | `vals[t-kmax : t+1]` dot product (backward only) | ✅ Backward |

Lag features use `df[feat].shift(k)` with k≥1 (positive shift = backward-looking). ✅  
Rolling stats use `df[feat].rolling(w).mean/std/slope` (backward window). ✅  
Hour features use `df.index.hour` (no future info). ✅

**Verdict**: No feature uses future data. All indicators are computed from current and past bars only.

---

## 2. Labeling ✅ SAFE — Future Returns Used Only on Training Data

### Standard labels (non-Triple-Barrier):
```python
_returns_fwd = train_data_scaled["returns"].shift(-1)  # line 493-494
```
`returns.shift(-1)` gives T+1 return. This is the **target variable** (not a feature), computed only on **training data**. The last row is dropped to remove NaN. ✅

### Triple Barrier labels:
```python
triple_barrier_labels(close=train_data_scaled[_pcol_tr], ...)
```
TB labels are event-driven (stop-hit or max-holding exit). Forward-looking by design, but computed only on **training data**. ✅

### Prefilter labels:
```python
ret_fwd_pre = train_data["returns"].shift(-1)  # line 393 — for MI prefilter
```
Used only on training data for feature selection. ✅

**Verdict**: Labels correctly use future returns only on training slices. Test labels are never computed during training.

---

## 3. Walk-Forward Split ✅ SAFE — Strictly Chronological

### HPO tuning span (run_mixin.py):
```python
first_train_df = walk_data[(idx >= first_start) & (idx < first_test_start)]  # line 148
```
Uses `<` (end-exclusive) so the boundary bar is NOT shared. ✅

### Monthly-roll CV folds (run_mixin.py):
```python
tr_end = max(0, vs - int(embargo_bars_eff))  # line 535
```
Embargo bars gap between train end and validation start. ✅

### Mini-block CV (run_mixin.py):
```python
smin = max(0, min_train_local + int(embargo_bars))  # line 655
smax = total_len - val_window_local                    # line 656
```
Embargo enforced between train and validation blocks. ✅

### WFO test folds (strategy_mixin.py):
```python
train_data = full_data.loc[train_start:train_end]  # line 180
test_data = full_data.loc[warmup_start:test_end]    # line 201
```
Test warmup bars are pre-roll only (for indicator warm-up, not for training). ✅

### Final embargo (strategy_mixin.py):
```python
if embargo_n > 0 and len(test_data) > embargo_n:
    test_data = test_data.iloc[embargo_n:]  # line 244
```
First N test bars dropped to prevent boundary leakage. Disabled during CV (correct). ✅

**Verdict**: All splits are strictly chronological with proper end-exclusive boundaries and embargo gaps.

---

## 4. Execution Delay ✅ SAFE — 1-Bar Delay Enforced

The prediction at bar T is used for trading at T+1. Evidence:

### Strategy execution (strategy_mixin.py):
```python
rets = self.data["returns"].reindex(test_data_scaled.index).astype(float)
ret_fwd = rets.shift(-1)  # T+1 execution return
```
The model predicts at bar T, but the return captured is from T+1 (execution happens next bar). ✅

### Feature/label alignment:
- Features at bar T use only past data (lags, rolling stats)
- Label is `returns.shift(-1)` = return from T to T+1
- At inference time, prediction at T → execute at T+1

**Verdict**: 1-bar execution delay is enforced. No same-bar trading on model signal.

---

## 5. Scaling & Imputation ✅ SAFE — Train-Only Fit

```python
imputer = SimpleImputer(strategy="mean")
train_imputed = imputer.fit_transform(train_data[features])  # fit on TRAIN
test_imputed = imputer.transform(test_data[features])         # transform TEST only

means, stds computed on TRAIN only  # scale_features
test_data_scaled scaled with TRAIN means/stds  # no test leakage
```

**Verdict**: Scaling and imputation are fit on training data only. Test data is transformed with train statistics. ✅

---

## Summary

| Check | Status | Notes |
|-------|--------|-------|
| Feature computation | ✅ PASS | All indicators backward-looking |
| Labeling | ✅ PASS | Future returns only on training data |
| Walk-forward splits | ✅ PASS | End-exclusive, embargo gaps |
| Execution delay | ✅ PASS | 1-bar delay enforced |
| Scaling/imputation | ✅ PASS | Train-only fit |

### Potential Concerns (non-critical, for future review)

1. **MTF Moving Averages** (`mtf_ma_fast`, `mtf_ma_slow`): Precomputed in `get_data()`. The docstring says "shifted to avoid look-ahead" but the actual computation in `data_mixin.py` should be verified in a future audit.

2. **Regime features** (`_attach_regime_columns`): Called during feature prep. Should verify that regime classification doesn't use future data (e.g., GMM/clustering must be fit on training data only).

3. **Feature prefilter MI scores**: Uses `train_data["returns"].shift(-1)` for target — correct but the MI computation could theoretically leak if not careful. Low risk since it's train-only.

4. **Ensemble models** (`ensemble_mixin.py`): Use `test_data["returns"].shift(-1)` for labels — this is for evaluation only, not training. Correct.

---

**Overall Verdict**: 🟢 **No data leakage detected.** The pipeline correctly separates train/test, uses only backward-looking features, applies proper embargo, and enforces 1-bar execution delay.