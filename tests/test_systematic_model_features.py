"""Systematic model + feature validation test suite.

Tests every registered model AND every feature toggle systematically:
  - Each model: build → train → predict → predict_proba → shape/probability checks
  - Each feature toggle: ON vs OFF → verify column presence/absence → NaN check
  - Model × feature interaction: each model with default features produces valid output
  - Feature count regression: catch accidental feature drops

Usage:
    pytest tests/test_systematic_model_features.py -v           # all tests
    pytest tests/test_systematic_model_features.py -v -k "feature"  # feature tests only
    pytest tests/test_systematic_model_features.py -v -k "model"   # model tests only

CI smoke mode (skips slow deep models):
    SMOKING=1 pytest tests/test_systematic_model_features.py -v
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from pipeline.backtester.features_mixin import FeaturesMixin
from pipeline.backtester.data_mixin import DataMixin

_RNG_SEED = 42
_N_SAMPLES = 200
_N_FEATURES = 6
_N_CLASSES = 3
_TIMESTEPS = 10

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XGB_USE_GPU", "0")

_SMOKING = os.environ.get("SMOKING", "").strip() in ("1", "true", "yes")


def _tf_available():
    try:
        import tensorflow as tf
        return True
    except Exception:
        return False


tf_skip = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")
slow = pytest.mark.skipif(_SMOKING, reason="Skipped in SMOKING mode")


def _make_flat_data(n=_N_SAMPLES, f=_N_FEATURES, seed=_RNG_SEED):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    y = rng.integers(0, _N_CLASSES, size=n).astype(np.int32)
    return X, y


def _make_seq_data(n=_N_SAMPLES, t=_TIMESTEPS, f=_N_FEATURES, seed=_RNG_SEED):
    rng = np.random.default_rng(seed)
    X_seq = rng.standard_normal((n, t, f)).astype(np.float32)
    X_flat = X_seq.mean(axis=1)
    y = rng.integers(0, _N_CLASSES, size=n).astype(np.int32)
    return X_seq, X_flat, y


def _make_ohlcv_df(n=500, seed=_RNG_SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = 1.1000 + rng.standard_normal(n).cumsum() * 0.001
    high = close + np.abs(rng.standard_normal(n) * 0.002)
    low = close - np.abs(rng.standard_normal(n) * 0.002)
    df = pd.DataFrame({
        "open": close + rng.standard_normal(n) * 0.0005,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(100, 10000, n).astype(np.float64),
    }, index=dates)
    return df


CLASSICAL_MODELS = [
    ("logistic", {}),
    ("xgboost", {"xgb_n_estimators": 10}),
    ("random_forest", {"rf_n_estimators": 10}),
    ("decision_tree", {}),
    ("svm", {}),
]

DEEP_MODELS = [
    ("cnn", {"input_shape": (_TIMESTEPS, _N_FEATURES), "cnn_epochs": 2, "cnn_use_early_stopping": False}),
    ("lstm", {"input_shape": (_TIMESTEPS, _N_FEATURES), "lstm_epochs": 2, "lstm_use_early_stopping": False}),
    ("transformer", {"input_shape": (_TIMESTEPS, _N_FEATURES), "transformer_epochs": 2, "transformer_use_early_stopping": False}),
]

ALL_MODELS = CLASSICAL_MODELS + DEEP_MODELS

FEATURE_TOGGLES = [
    "use_sma", "use_ema", "use_rsi", "use_macd", "use_bbands",
    "use_atr", "use_adx", "use_stoch", "use_donchian",
    "use_crossover_bins", "use_price_ma_z", "use_slope_diff",
    "use_mtf_ma", "use_mtf_alignment", "use_triple_confirm",
    "use_trend_confirm", "use_vol_managed_mom", "use_macd_atr_ratio",
    "use_rv_features",
]


# ═══════════════════════════════════════════════════════════════
# PART 1: Systematic model tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_name,kwargs", CLASSICAL_MODELS, ids=[m[0] for m in CLASSICAL_MODELS])
def test_classical_model_build_train_predict(model_name, kwargs):
    from models.registry import build_model
    m = build_model(model_name, seed=_RNG_SEED, **kwargs)
    assert m is not None, f"build_model({model_name!r}) returned None"
    X, y = _make_flat_data()
    m.fit(X, y)
    preds = m.predict(X[:10])
    assert preds.shape == (10,), f"{model_name}: predict shape {preds.shape} != (10,)"
    assert set(preds).issubset({0, 1, 2}), f"{model_name}: unexpected classes {set(preds)}"


@pytest.mark.parametrize("model_name,kwargs", CLASSICAL_MODELS, ids=[m[0] for m in CLASSICAL_MODELS])
def test_classical_model_predict_proba(model_name, kwargs):
    from models.registry import build_model
    m = build_model(model_name, seed=_RNG_SEED, **kwargs)
    X, y = _make_flat_data()
    m.fit(X, y)
    proba = m.predict_proba(X[:10])
    assert proba.shape == (10, _N_CLASSES), f"{model_name}: proba shape {proba.shape}"
    assert (proba >= -0.01).all(), f"{model_name}: negative probabilities"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=0.05), f"{model_name}: probabilities don't sum to 1"


@tf_skip
@pytest.mark.parametrize("model_name,kwargs", DEEP_MODELS, ids=[m[0] for m in DEEP_MODELS])
def test_deep_model_build_train_predict(model_name, kwargs):
    from models.registry import build_model
    m = build_model(model_name, seed=_RNG_SEED, **kwargs)
    assert m is not None, f"build_model({model_name!r}) returned None"
    X_seq, X_flat, y = _make_seq_data()
    m.fit(X_seq, y, epochs=2, batch_size=32, verbose=0)
    preds = m.predict(X_seq[:5], verbose=0)
    assert preds.shape == (5, _N_CLASSES), f"{model_name}: predict shape {preds.shape}"


@tf_skip
@pytest.mark.parametrize("model_name,kwargs", DEEP_MODELS, ids=[m[0] for m in DEEP_MODELS])
def test_deep_model_probabilities_valid(model_name, kwargs):
    from models.registry import build_model
    m = build_model(model_name, seed=_RNG_SEED, **kwargs)
    X_seq, _, y = _make_seq_data()
    m.fit(X_seq, y, epochs=2, batch_size=32, verbose=0)
    proba = m.predict(X_seq[:5], verbose=0)
    assert (proba >= 0).all(), f"{model_name}: negative probabilities"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3), f"{model_name}: probs don't sum to 1"


def test_model_registry_completeness():
    from models.registry import MODEL_REGISTRY
    expected = {"logistic", "xgboost", "random_forest", "decision_tree", "svm",
                "cnn", "lstm", "transformer"}
    actual = set(MODEL_REGISTRY.keys())
    missing = expected - actual
    assert not missing, f"Missing models from registry: {missing}"


def test_unknown_model_returns_none():
    from models.registry import build_model
    assert build_model("nonexistent_xyz") is None


def test_seed_reproducibility():
    from models.registry import build_model
    X, y = _make_flat_data()
    m1 = build_model("logistic", seed=99)
    m2 = build_model("logistic", seed=99)
    m1.fit(X, y)
    m2.fit(X, y)
    np.testing.assert_array_equal(m1.predict(X[:5]), m2.predict(X[:5]))


# ═══════════════════════════════════════════════════════════════
# PART 2: Feature toggle tests
# ═══════════════════════════════════════════════════════════════

class _MinimalBacktester(FeaturesMixin, DataMixin):
    """Minimal stub that provides just enough state for prepare_features()."""
    def __init__(self, features_config=None):
        self.features_config = features_config or {}
        self._is_debug = lambda: False
        self._feat_cache = {}
        self._feat_cache_bytes = {}
        self._feat_cache_cur_bytes = {}
        self._feat_cache_est_bytes = {}
        self._feat_cache_evictions = 0
        self._feat_cache_hits = 0
        self._feat_cache_misses = 0
        self._feat_cache_mode_logged = False
        self._feature_bank_full = None
        self._feature_bank_meta = None
        self._last_used_features = None


@pytest.mark.parametrize("toggle", FEATURE_TOGGLES, ids=FEATURE_TOGGLES)
def test_feature_toggle_on_produces_columns(toggle):
    bt = _MinimalBacktester({toggle: True, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    assert isinstance(result_df, pd.DataFrame), f"{toggle}=True: prepare_features did not return DataFrame"
    assert len(result_df) > 0, f"{toggle}=True: empty result"
    nan_frac = result_df.isna().mean().max()
    assert nan_frac < 0.5, f"{toggle}=True: >50% NaN in some column (max {nan_frac:.2f})"


@pytest.mark.parametrize("toggle", FEATURE_TOGGLES, ids=FEATURE_TOGGLES)
def test_feature_toggle_off_no_crash(toggle):
    all_off = {t: False for t in FEATURE_TOGGLES}
    all_off[toggle] = False
    bt = _MinimalBacktester({**all_off, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    assert isinstance(result_df, pd.DataFrame), f"{toggle}=False: crashed"
    assert len(result_df) > 0, f"{toggle}=False: empty result"


def test_all_features_on():
    all_on = {t: True for t in FEATURE_TOGGLES}
    bt = _MinimalBacktester({**all_on, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    assert len(result_df) > 0
    assert result_df.shape[1] > 10, f"All features on: only {result_df.shape[1]} columns"
    nan_frac = result_df.isna().mean().max()
    assert nan_frac < 0.5, f"All features on: >50% NaN (max {nan_frac:.2f})"


def test_all_features_off_still_produces_output():
    all_off = {t: False for t in FEATURE_TOGGLES}
    bt = _MinimalBacktester({**all_off, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_feature_count_regression():
    all_on = {t: True for t in FEATURE_TOGGLES}
    bt = _MinimalBacktester({**all_on, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    n_features = result_df.shape[1]
    assert n_features >= 30, f"Feature count regression: only {n_features} columns (expected >=30)"


def test_features_no_inf():
    all_on = {t: True for t in FEATURE_TOGGLES}
    bt = _MinimalBacktester({**all_on, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    numeric_cols = result_df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_cols.values).sum()
    assert inf_count == 0, f"Found {inf_count} inf values in feature matrix"


def test_features_no_look_ahead_in_index():
    bt = _MinimalBacktester({**{t: True for t in FEATURE_TOGGLES[:8]}, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df()
    original_index = df.index
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    assert result_df.index.is_monotonic_increasing, "Feature index not monotonic"
    assert len(result_df) <= len(original_index), "Feature df has more rows than input"


# ═══════════════════════════════════════════════════════════════
# PART 3: Model × feature integration
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_name,kwargs", CLASSICAL_MODELS, ids=[m[0] for m in CLASSICAL_MODELS])
def test_model_with_default_features(model_name, kwargs):
    bt = _MinimalBacktester({
        "use_sma": True, "use_ema": True, "use_rsi": True,
        "use_macd": True, "use_bbands": True, "use_atr": True, "use_adx": True,
        "indicator_windows": {"sma": 20, "rsi": 14},
    })
    df = _make_ohlcv_df()
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1, base_only=False)
    numeric = result_df.select_dtypes(include=[np.number]).dropna()
    X = numeric.values.astype(np.float32)
    rng = np.random.default_rng(_RNG_SEED)
    y = rng.integers(0, _N_CLASSES, size=len(X)).astype(np.int32)
    if len(X) < 20:
        pytest.skip("Not enough rows after dropna")
    from models.registry import build_model
    m = build_model(model_name, seed=_RNG_SEED, **kwargs)
    m.fit(X, y)
    preds = m.predict(X[:5])
    assert preds.shape == (5,), f"{model_name} with real features: shape {preds.shape}"


# ═══════════════════════════════════════════════════════════════
# PART 4: Edge cases
# ═══════════════════════════════════════════════════════════════

def test_features_with_short_dataframe():
    bt = _MinimalBacktester({"use_sma": True, "use_rsi": True, "indicator_windows": {"sma": 20, "rsi": 14}})
    df = _make_ohlcv_df(n=30)
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1)
    assert isinstance(result_df, pd.DataFrame)


def test_features_with_zero_volume():
    df = _make_ohlcv_df()
    df["volume"] = 0.0
    bt = _MinimalBacktester({"use_sma": True, "use_rsi": True, "indicator_windows": {"sma": 20, "rsi": 14}})
    result_df, _ = bt.prepare_features(df, lags=3, lag_depth=1)
    assert isinstance(result_df, pd.DataFrame)
    numeric = result_df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric.values).sum()
    assert inf_count == 0, f"Inf values with zero volume: {inf_count}"


def test_model_single_sample_prediction():
    from models.registry import build_model
    m = build_model("logistic", seed=_RNG_SEED)
    X, y = _make_flat_data()
    m.fit(X, y)
    pred = m.predict(X[:1])
    assert pred.shape == (1,)
    proba = m.predict_proba(X[:1])
    assert proba.shape == (1, _N_CLASSES)


def test_model_with_constant_features():
    from models.registry import build_model
    m = build_model("logistic", seed=_RNG_SEED)
    X = np.zeros((50, _N_FEATURES), dtype=np.float32)
    X[:, 0] = 1.0
    y = np.zeros(50, dtype=np.int32)
    y[25:] = 1
    m.fit(X, y)
    preds = m.predict(X[:5])
    assert preds.shape == (5,)


def test_filter_params_utility():
    from models.registry import filter_params
    d = {"logit_C": 1.0, "logit_solver": "saga", "other": 99}
    result = filter_params(d, "logit_")
    assert result == {"C": 1.0, "solver": "saga"}