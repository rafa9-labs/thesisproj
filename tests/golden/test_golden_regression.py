"""Golden output regression tests.

Verifies that model outputs (predictions, probabilities) are deterministic
for a fixed seed and unchanged across code changes. Any intentional change
to model logic requires updating the golden outputs.

Golden outputs are stored as .npz files in this directory.
"""
import os
import json
import pytest
import numpy as np

# Golden outputs must be reproducible across machines: pin XGBoost to a
# single thread (hist splits are thread-count sensitive) and cap OpenMP.
os.environ.setdefault("XGB_JOBS", "1")
os.environ.setdefault("XGB_USE_GPU", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")

_GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
_RNG_SEED = 99
_N_SAMPLES = 60
_N_FEATURES = 6
_N_CLASSES = 3
_TOLERANCE = 1e-4


def _generate_golden_data():
    rng = np.random.default_rng(_RNG_SEED)
    X = rng.standard_normal((_N_SAMPLES, _N_FEATURES)).astype(np.float32)
    y = rng.integers(0, _N_CLASSES, size=_N_SAMPLES).astype(np.int32)
    return X, y


def _save_golden(model_type, predictions, proba):
    path = os.path.join(_GOLDEN_DIR, f"golden_{model_type}.npz")
    np.savez(path, predictions=predictions, proba=proba)


def _load_golden(model_type):
    path = os.path.join(_GOLDEN_DIR, f"golden_{model_type}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["predictions"], data["proba"]


class TestGoldenLogistic:
    def test_predictions_match_golden(self):
        from models.registry import build_model
        X, y = _generate_golden_data()
        m = build_model("logistic", seed=_RNG_SEED)
        m.fit(X, y)
        preds = m.predict(X[:10])
        proba = m.predict_proba(X[:10])
        golden_preds, golden_proba = _load_golden("logistic")
        if golden_preds is None:
            _save_golden("logistic", preds, proba)
            pytest.skip("Golden file created for logistic — re-run to verify")
        np.testing.assert_array_equal(preds, golden_preds)
        np.testing.assert_allclose(proba, golden_proba, atol=_TOLERANCE)


class TestGoldenSVM:
    def test_predictions_match_golden(self):
        from models.registry import build_model
        X, y = _generate_golden_data()
        m = build_model("svm", seed=_RNG_SEED)
        m.fit(X, y)
        preds = m.predict(X[:10])
        proba = m.predict_proba(X[:10])
        golden_preds, golden_proba = _load_golden("svm")
        if golden_preds is None:
            _save_golden("svm", preds, proba)
            pytest.skip("Golden file created for svm — re-run to verify")
        np.testing.assert_array_equal(preds, golden_preds)
        np.testing.assert_allclose(proba, golden_proba, atol=_TOLERANCE)


class TestGoldenXGBoost:
    def test_predictions_match_golden(self):
        from models.registry import build_model
        X, y = _generate_golden_data()
        m = build_model("xgboost", seed=_RNG_SEED, xgb_n_estimators=50)
        m.fit(X, y)
        preds = m.predict(X[:10])
        proba = m.predict_proba(X[:10])
        golden_preds, golden_proba = _load_golden("xgboost")
        if golden_preds is None:
            _save_golden("xgboost", preds, proba)
            pytest.skip("Golden file created for xgboost — re-run to verify")
        np.testing.assert_array_equal(preds, golden_preds)
        np.testing.assert_allclose(proba, golden_proba, atol=1e-3)


class TestGoldenRandomForest:
    def test_predictions_match_golden(self):
        from models.registry import build_model
        X, y = _generate_golden_data()
        m = build_model("random_forest", seed=_RNG_SEED, rf_n_estimators=50)
        m.fit(X, y)
        preds = m.predict(X[:10])
        proba = m.predict_proba(X[:10])
        golden_preds, golden_proba = _load_golden("random_forest")
        if golden_preds is None:
            _save_golden("random_forest", preds, proba)
            pytest.skip("Golden file created for random_forest — re-run to verify")
        np.testing.assert_array_equal(preds, golden_preds)
        np.testing.assert_allclose(proba, golden_proba, atol=1e-3)


class TestGoldenDecisionTree:
    def test_predictions_match_golden(self):
        from models.registry import build_model
        X, y = _generate_golden_data()
        m = build_model("decision_tree", seed=_RNG_SEED)
        m.fit(X, y)
        preds = m.predict(X[:10])
        proba = m.predict_proba(X[:10])
        golden_preds, golden_proba = _load_golden("decision_tree")
        if golden_preds is None:
            _save_golden("decision_tree", preds, proba)
            pytest.skip("Golden file created for decision_tree — re-run to verify")
        np.testing.assert_array_equal(preds, golden_preds)
        np.testing.assert_allclose(proba, golden_proba, atol=_TOLERANCE)


class TestGoldenRegistry:
    def test_registry_keys_stable(self):
        from models.registry import MODEL_REGISTRY
        golden_path = os.path.join(_GOLDEN_DIR, "golden_registry_keys.json")
        current_keys = sorted(MODEL_REGISTRY.keys())
        if not os.path.exists(golden_path):
            with open(golden_path, "w") as f:
                json.dump(current_keys, f)
            pytest.skip("Golden registry keys created — re-run to verify")
        with open(golden_path) as f:
            golden_keys = json.load(f)
        assert current_keys == golden_keys, (
            f"Registry keys changed!\n"
            f"  Added: {set(current_keys) - set(golden_keys)}\n"
            f"  Removed: {set(golden_keys) - set(current_keys)}\n"
            f"  If intentional, update {golden_path}"
        )

    def test_search_space_keys_stable(self):
        from config import SEARCH_SPACE
        golden_path = os.path.join(_GOLDEN_DIR, "golden_search_space_keys.json")
        current_keys = sorted(SEARCH_SPACE.keys())
        if not os.path.exists(golden_path):
            with open(golden_path, "w") as f:
                json.dump(current_keys, f)
            pytest.skip("Golden search space keys created — re-run to verify")
        with open(golden_path) as f:
            golden_keys = json.load(f)
        assert current_keys == golden_keys, (
            f"Search space keys changed!\n"
            f"  Added: {set(current_keys) - set(golden_keys)}\n"
            f"  Removed: {set(golden_keys) - set(current_keys)}\n"
            f"  If intentional, update {golden_path}"
        )