"""Comprehensive build → train → predict tests for all registered models.

Each model type is tested for:
  - constructibility via build_model()
  - fit() succeeds with small synthetic data
  - predict() returns correct shape
  - predict_proba() returns correct shape and valid probabilities
  - edge cases (single sample, minimal features)
"""
import os
import sys
import pytest
import numpy as np

_RNG_SEED = 42
_N_SAMPLES = 80
_N_FEATURES = 6
_N_CLASSES = 3
_TIMESTEPS = 10


def _tf_available():
    try:
        import tensorflow as tf
        return True
    except Exception:
        return False


def _rl_available():
    try:
        from rl.dqn_agent import DQNAgent
        return True
    except Exception:
        return False


tf_skip = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")
rl_skip = pytest.mark.skipif(not _rl_available(), reason="RL dependencies not available")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XGB_USE_GPU", "0")


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


# ═══════════════════════════════════════════════════════════════════
# Classical models
# ═══════════════════════════════════════════════════════════════════

class TestLogistic:
    def test_build(self):
        from models.registry import build_model
        m = build_model("logistic", seed=_RNG_SEED)
        assert m is not None

    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("logistic", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        preds = m.predict(X[:5])
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba_shape(self):
        from models.registry import build_model
        m = build_model("logistic", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, _N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0).all()

    def test_single_sample(self):
        from models.registry import build_model
        m = build_model("logistic", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        p = m.predict(X[:1])
        assert p.shape == (1,)

    def test_with_hyperparams(self):
        from models.registry import build_model
        m = build_model("logistic", seed=_RNG_SEED, logit_C=0.01, logit_solver="saga")
        X, y = _make_flat_data()
        m.fit(X, y)
        assert m.predict(X[:5]).shape == (5,)


class TestSVM:
    def test_build(self):
        from models.registry import build_model
        m = build_model("svm", seed=_RNG_SEED)
        assert m is not None

    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("svm", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        preds = m.predict(X[:5])
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba_shape(self):
        from models.registry import build_model
        m = build_model("svm", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, _N_CLASSES)
        assert (proba >= -0.01).all()

    def test_with_hyperparams(self):
        from models.registry import build_model
        m = build_model("svm", seed=_RNG_SEED, svm_C=10.0, svm_gamma="scale")
        X, y = _make_flat_data()
        m.fit(X, y)
        assert m.predict(X[:5]).shape == (5,)


class TestRandomForest:
    def test_build(self):
        from models.registry import build_model
        m = build_model("random_forest", seed=_RNG_SEED)
        assert m is not None

    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("random_forest", seed=_RNG_SEED, rf_n_estimators=10)
        X, y = _make_flat_data()
        m.fit(X, y)
        preds = m.predict(X[:5])
        assert preds.shape == (5,)

    def test_predict_proba_shape(self):
        from models.registry import build_model
        m = build_model("random_forest", seed=_RNG_SEED, rf_n_estimators=10)
        X, y = _make_flat_data()
        m.fit(X, y)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, _N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestDecisionTree:
    def test_build(self):
        from models.registry import build_model
        m = build_model("decision_tree", seed=_RNG_SEED)
        assert m is not None

    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("decision_tree", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        preds = m.predict(X[:5])
        assert preds.shape == (5,)

    def test_predict_proba_shape(self):
        from models.registry import build_model
        m = build_model("decision_tree", seed=_RNG_SEED)
        X, y = _make_flat_data()
        m.fit(X, y)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, _N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestXGBoost:
    def test_build(self):
        from models.registry import build_model
        m = build_model("xgboost", seed=_RNG_SEED)
        assert m is not None

    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("xgboost", seed=_RNG_SEED, xgb_n_estimators=10)
        X, y = _make_flat_data()
        m.fit(X, y)
        preds = m.predict(X[:5])
        assert preds.shape == (5,)

    def test_predict_proba_shape(self):
        from models.registry import build_model
        m = build_model("xgboost", seed=_RNG_SEED, xgb_n_estimators=10)
        X, y = _make_flat_data()
        m.fit(X, y)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, _N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════
# Deep models (Keras) — require TensorFlow
# ═══════════════════════════════════════════════════════════════════

class TestCNN:
    @tf_skip
    def test_build(self):
        from models.registry import build_model
        m = build_model("cnn", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES))
        assert m is not None

    @tf_skip
    def test_fit_predict(self):
        import tensorflow as tf
        from models.registry import build_model
        m = build_model("cnn", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES),
                         cnn_filters1=16, cnn_filters2=32, cnn_epochs=2,
                         cnn_use_early_stopping=False)
        X_seq, X_flat, y = _make_seq_data()
        m.fit(X_seq, y, epochs=2, batch_size=16, verbose=0)
        preds = m.predict(X_seq[:5], verbose=0)
        assert preds.shape == (5, _N_CLASSES)

    @tf_skip
    def test_probabilities_valid(self):
        from models.registry import build_model
        m = build_model("cnn", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES),
                         cnn_epochs=2, cnn_use_early_stopping=False)
        X_seq, _, y = _make_seq_data()
        m.fit(X_seq, y, epochs=2, batch_size=16, verbose=0)
        proba = m.predict(X_seq[:5], verbose=0)
        assert (proba >= 0).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


class TestLSTM:
    @tf_skip
    def test_build(self):
        from models.registry import build_model
        m = build_model("lstm", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES))
        assert m is not None

    @tf_skip
    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("lstm", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES),
                         lstm_units=32, lstm_num_layers=1, lstm_epochs=2,
                         lstm_use_early_stopping=False)
        X_seq, _, y = _make_seq_data()
        m.fit(X_seq, y, epochs=2, batch_size=16, verbose=0)
        preds = m.predict(X_seq[:5], verbose=0)
        assert preds.shape == (5, _N_CLASSES)

    @tf_skip
    def test_probabilities_valid(self):
        from models.registry import build_model
        m = build_model("lstm", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES),
                         lstm_epochs=2, lstm_use_early_stopping=False)
        X_seq, _, y = _make_seq_data()
        m.fit(X_seq, y, epochs=2, batch_size=16, verbose=0)
        proba = m.predict(X_seq[:5], verbose=0)
        assert (proba >= 0).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    @tf_skip
    def test_invalid_input_shape_raises(self):
        from models.registry import build_model
        with pytest.raises((ValueError, TypeError)):
            build_model("lstm", seed=_RNG_SEED, input_shape=(5,))


class TestTransformer:
    @tf_skip
    def test_build(self):
        from models.registry import build_model
        m = build_model("transformer", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES))
        assert m is not None

    @tf_skip
    def test_fit_predict(self):
        from models.registry import build_model
        m = build_model("transformer", seed=_RNG_SEED, input_shape=(_TIMESTEPS, _N_FEATURES),
                         transformer_d_model=32, transformer_num_heads=2,
                         transformer_num_blocks=1, transformer_epochs=2,
                         transformer_use_early_stopping=False)
        X_seq, _, y = _make_seq_data()
        m.fit(X_seq, y, epochs=2, batch_size=16, verbose=0)
        preds = m.predict(X_seq[:5], verbose=0)
        assert preds.shape == (5, _N_CLASSES)


# ═══════════════════════════════════════════════════════════════════
# DQN (RL) — requires rl module
# ═══════════════════════════════════════════════════════════════════

class TestDQN:
    @rl_skip
    def test_build(self):
        from models.registry import build_model
        m = build_model("dqn", seed=_RNG_SEED, input_shape=(_N_FEATURES,))
        assert m is not None

    @rl_skip
    def test_agent_attributes(self):
        from models.registry import build_model
        m = build_model("dqn", seed=_RNG_SEED, input_shape=(_N_FEATURES,))
        assert hasattr(m, "act")
        assert hasattr(m, "remember")
        assert hasattr(m, "replay")


# ═══════════════════════════════════════════════════════════════════
# Ensemble models
# ═══════════════════════════════════════════════════════════════════

class TestEnsembleAdaptiveRegime:
    @tf_skip
    def test_build(self):
        from models.registry import build_model
        m = build_model("ensemble_adaptive_regime", seed=_RNG_SEED,
                         input_shape=(_TIMESTEPS, _N_FEATURES))
        assert m is not None
        assert hasattr(m, "fit")
        assert hasattr(m, "predict")

    @tf_skip
    def test_fit_predict_with_regime(self):
        import pandas as pd
        from models.registry import build_model
        m = build_model("ensemble_adaptive_regime", seed=_RNG_SEED,
                         input_shape=(_TIMESTEPS, _N_FEATURES),
                         lstm_units=16, lstm_num_layers=1,
                         rf_n_estimators=10)
        X_seq, X_flat, y = _make_seq_data()
        rng = np.random.default_rng(_RNG_SEED)
        regime_df = pd.DataFrame({
            "adx_14": rng.uniform(10, 50, _N_SAMPLES),
            "rolling_std_20": rng.uniform(0.0005, 0.005, _N_SAMPLES),
        })
        m.fit(X_seq, X_flat, y, X_flat_with_regime=regime_df)
        regime_subset = regime_df.iloc[:5].reset_index(drop=True)
        preds = m.predict(X_seq[:5], X_flat[:5], regime_source=regime_subset)
        assert preds.shape == (5,)

    @tf_skip
    def test_predict_proba_with_regime(self):
        import pandas as pd
        from models.registry import build_model
        m = build_model("ensemble_adaptive_regime", seed=_RNG_SEED,
                         input_shape=(_TIMESTEPS, _N_FEATURES),
                         lstm_units=16, rf_n_estimators=10)
        X_seq, X_flat, y = _make_seq_data()
        rng = np.random.default_rng(_RNG_SEED)
        regime_df = pd.DataFrame({
            "adx_14": rng.uniform(10, 50, _N_SAMPLES),
            "rolling_std_20": rng.uniform(0.0005, 0.005, _N_SAMPLES),
        })
        m.fit(X_seq, X_flat, y, X_flat_with_regime=regime_df)
        regime_subset = regime_df.iloc[:5].reset_index(drop=True)
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_seq[:5], X_flat[:5], regime_source=regime_subset)
            assert proba.shape[0] == 5


class TestEnsembleCNNLSTMXGBoost:
    @tf_skip
    def test_build(self):
        from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
        m = EnsembleCNNLSTMXGBoost(
            input_shape=(_TIMESTEPS, _N_FEATURES),
            cnn_config={"filters1": 16, "filters2": 32},
            lstm_config={"units": 16},
            xgb_config={"n_estimators": 10},
        )
        assert m is not None
        assert hasattr(m, "fit")
        assert hasattr(m, "predict")

    @tf_skip
    def test_fit_predict(self):
        from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
        m = EnsembleCNNLSTMXGBoost(
            input_shape=(_TIMESTEPS, _N_FEATURES),
            cnn_config={"filters1": 16, "filters2": 32, "epochs": 2, "use_early_stopping": False},
            lstm_config={"units": 16, "epochs": 2, "use_early_stopping": False},
            xgb_config={"n_estimators": 10},
        )
        X_seq, X_flat, y = _make_seq_data()
        m.fit(X_seq, X_flat, y)
        preds = m.predict(X_seq[:5], X_flat[:5])
        assert preds.shape == (5,)

    @tf_skip
    def test_predict_proba_shape(self):
        from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
        m = EnsembleCNNLSTMXGBoost(
            input_shape=(_TIMESTEPS, _N_FEATURES),
            cnn_config={"filters1": 16, "filters2": 32, "epochs": 2, "use_early_stopping": False},
            lstm_config={"units": 16, "epochs": 2, "use_early_stopping": False},
            xgb_config={"n_estimators": 10},
        )
        X_seq, X_flat, y = _make_seq_data()
        m.fit(X_seq, X_flat, y)
        proba = m.predict_proba(X_seq[:5], X_flat[:5])
        assert proba.shape == (5, _N_CLASSES)


# ═══════════════════════════════════════════════════════════════════
# Cross-cutting tests
# ═══════════════════════════════════════════════════════════════════

class TestBuildModelEdgeCases:
    def test_unknown_model_returns_none(self):
        from models.registry import build_model
        assert build_model("nonexistent_xyz") is None

    def test_seed_reproducibility_classical(self):
        from models.registry import build_model
        m1 = build_model("logistic", seed=7)
        m2 = build_model("logistic", seed=7)
        X, y = _make_flat_data()
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict(X[:5])
        p2 = m2.predict(X[:5])
        np.testing.assert_array_equal(p1, p2)

    def test_use_proba_false_still_builds(self):
        from models.registry import build_model
        m = build_model("logistic", use_proba=False, seed=_RNG_SEED)
        assert m is not None
        X, y = _make_flat_data()
        m.fit(X, y)
        assert m.predict(X[:5]).shape == (5,)

    def test_filter_params_utility(self):
        from models.registry import filter_params
        d = {"logit_C": 1.0, "logit_solver": "saga", "other": 99}
        result = filter_params(d, "logit_")
        assert result == {"C": 1.0, "solver": "saga"}

    def test_ensure_dict_utility(self):
        from models.registry import ensure_dict
        assert ensure_dict(None) == {}
        assert ensure_dict({"a": 1}) == {"a": 1}
        assert ensure_dict("bad") == {}