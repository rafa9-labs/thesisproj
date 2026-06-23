"""ML pipeline determinism tests — seed reproducibility.

Tests that set_global_determinism produces identical model behavior
for the same seed, and different behavior for different seeds.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from utilsNoWFO import set_global_determinism

_RNG_SEED = 42
_N_SAMPLES = 40
_N_FEATURES = 5
_N_CLASSES = 3


def _make_synthetic_data(n_samples=_N_SAMPLES, n_features=_N_FEATURES, seed=_RNG_SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, _N_CLASSES, n_samples).astype(np.int32)
    return X, y


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


def _tf_available():
    try:
        import tensorflow as tf
        return True
    except Exception:
        return False


tf_mark = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")


class TestSetGlobalDeterminism:

    def test_random_deterministic_same_seed(self):
        set_global_determinism(42)
        a = random.random()
        set_global_determinism(42)
        b = random.random()
        assert a == b

    def test_random_different_seed_different(self):
        set_global_determinism(42)
        a = random.random()
        set_global_determinism(99)
        b = random.random()
        assert a != b

    def test_numpy_deterministic_same_seed(self):
        set_global_determinism(42)
        a = np.random.rand(10)
        set_global_determinism(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_numpy_different_seed_different(self):
        set_global_determinism(42)
        a = np.random.rand(10)
        set_global_determinism(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_env_vars_set(self):
        set_global_determinism(42)
        import os
        assert os.environ.get("PYTHONHASHSEED") == "42"
        assert os.environ.get("TF_DETERMINISTIC_OPS") == "1"


@tf_mark
class TestTensorFlowDeterminism:

    def test_tf_random_deterministic_same_seed(self):
        import tensorflow as tf
        set_global_determinism(42)
        a = tf.random.uniform((5,)).numpy()
        set_global_determinism(42)
        b = tf.random.uniform((5,)).numpy()
        np.testing.assert_array_almost_equal(a, b)

    def test_tf_random_different_seed_different(self):
        import tensorflow as tf
        set_global_determinism(42)
        a = tf.random.uniform((5,)).numpy()
        set_global_determinism(99)
        b = tf.random.uniform((5,)).numpy()
        assert not np.allclose(a, b)


class TestLogisticDeterminism:

    def test_same_seed_same_weights(self, synthetic_data):
        X, y = synthetic_data
        from sklearn.linear_model import LogisticRegression

        set_global_determinism(42)
        m1 = LogisticRegression(random_state=42, max_iter=200).fit(X, y)

        set_global_determinism(42)
        m2 = LogisticRegression(random_state=42, max_iter=200).fit(X, y)

        np.testing.assert_array_almost_equal(m1.coef_, m2.coef_)

    def test_different_seed_different_weights(self, synthetic_data):
        X, y = synthetic_data
        from sklearn.linear_model import LogisticRegression

        set_global_determinism(42)
        m1 = LogisticRegression(random_state=42, max_iter=200, solver="saga").fit(X, y)

        set_global_determinism(99)
        m2 = LogisticRegression(random_state=99, max_iter=200, solver="saga").fit(X, y)

        assert not np.allclose(m1.coef_, m2.coef_, atol=1e-12)


class TestXGBoostDeterminism:

    def test_same_seed_same_predictions(self, synthetic_data):
        X, y = synthetic_data
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pytest.skip("xgboost not installed")

        set_global_determinism(42)
        m1 = XGBClassifier(random_state=42, n_estimators=10, eval_metric="mlogloss")
        m1.fit(X, y)
        p1 = m1.predict_proba(X)

        set_global_determinism(42)
        m2 = XGBClassifier(random_state=42, n_estimators=10, eval_metric="mlogloss")
        m2.fit(X, y)
        p2 = m2.predict_proba(X)

        np.testing.assert_array_almost_equal(p1, p2)

    def test_different_seed_different_predictions(self, synthetic_data):
        X, y = synthetic_data
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pytest.skip("xgboost not installed")

        set_global_determinism(42)
        m1 = XGBClassifier(random_state=42, n_estimators=30, eval_metric="mlogloss")
        m1.fit(X, y)

        set_global_determinism(99)
        m2 = XGBClassifier(random_state=99, n_estimators=30, eval_metric="mlogloss")
        m2.fit(X, y)

        p1 = m1.predict_proba(X)
        p2 = m2.predict_proba(X)
        assert len(p1) == len(p2)
        assert p1.shape == p2.shape
        assert (p1 >= 0).all() and (p1 <= 1).all()


class TestRandomForestDeterminism:

    def test_same_seed_same_output(self, synthetic_data):
        X, y = synthetic_data
        from sklearn.ensemble import RandomForestClassifier

        set_global_determinism(42)
        m1 = RandomForestClassifier(random_state=42, n_estimators=10)
        m1.fit(X, y)
        p1 = m1.predict_proba(X)

        set_global_determinism(42)
        m2 = RandomForestClassifier(random_state=42, n_estimators=10)
        m2.fit(X, y)
        p2 = m2.predict_proba(X)

        np.testing.assert_array_almost_equal(p1, p2)
