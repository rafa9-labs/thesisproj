"""Phase C tests: MetaEnsemble signal committee."""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, r"C:\Users\rafa\ML_Trading\thesisproj")


def _make_data(n=100, n_feat=4, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestMajorityVote:
    def test_3_models_buy_wins(self):
        from models.meta_ensemble import MetaEnsemble, _majority_vote

        X, _ = _make_data(10)
        p1 = np.tile([0.1, 0.2, 0.7], (10, 1))  # sell
        p2 = np.tile([0.6, 0.3, 0.1], (10, 1))  # buy
        p3 = np.tile([0.7, 0.2, 0.1], (10, 1))  # buy
        result = _majority_vote([p1, p2, p3])
        labels = np.argmax(result, axis=1)
        assert np.all(labels == 0), "2 buy + 1 sell → buy (class 0)"

    def test_tie_returns_neutral(self):
        from models.meta_ensemble import _majority_vote

        X, _ = _make_data(10)
        p1 = np.tile([0.7, 0.2, 0.1], (10, 1))  # buy
        p2 = np.tile([0.1, 0.2, 0.7], (10, 1))  # sell
        result = _majority_vote([p1, p2])
        # Tie → all zeros
        assert np.allclose(result.sum(axis=1), 0.0)


class TestSoftVoting:
    def test_probability_average(self):
        from models.meta_ensemble import MetaEnsemble

        X, _ = _make_data(10)
        m = MetaEnsemble(sub_models=[], method="soft")

        p1 = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
        p2 = np.array([[0.3, 0.5, 0.2], [0.3, 0.2, 0.5]])

        class FakeModel:
            def __init__(self, p):
                self._p = p
            def predict_proba(self, X):
                return self._p

        m._sub_models = [FakeModel(p1), FakeModel(p2)]
        m._fitted = True
        result = m.predict_proba(X)
        expected = (p1 + p2) / 2.0
        assert np.allclose(result, expected)


class TestWeightedVoting:
    def test_high_sharpe_has_more_influence(self):
        from models.meta_ensemble import MetaEnsemble

        X, _ = _make_data(10)
        m = MetaEnsemble(sub_models=[], method="weighted", weights=[3.0, 1.0])

        p_best = np.tile([0.8, 0.1, 0.1], (10, 1))
        p_worst = np.tile([0.1, 0.1, 0.8], (10, 1))

        class FakeModel:
            def __init__(self, p):
                self._p = p
            def predict_proba(self, X):
                return self._p

        m._sub_models = [FakeModel(p_best), FakeModel(p_worst)]
        m._fitted = True
        result = m.predict_proba(X)

        # With 3:1 weighting, result should be closer to p_best
        expected = (3 * p_best + 1 * p_worst) / 4.0
        assert np.allclose(result, expected)


class TestRegistryBuild:
    def test_build_via_registry(self):
        X, y = _make_data(50, 6)
        from models.registry import build_model, MODEL_REGISTRY

        assert "meta_ensemble" in MODEL_REGISTRY

        model = build_model("meta_ensemble",
                            use_proba=True,
                            meta_sub_models=["logistic", "xgboost"],
                            meta_combination_method="majority",
                            seed=42)
        assert model is not None
        assert model.model_type == "meta_ensemble"

    def test_fit_and_predict(self):
        X, y = _make_data(50, 6)
        from models.registry import build_model

        model = build_model("meta_ensemble",
                            use_proba=True,
                            meta_sub_models=["logistic", "xgboost", "decision_tree"],
                            meta_combination_method="soft",
                            seed=42,
                            logit_max_iter=500,
                            xgb_n_estimators=50,
                            dt_max_depth=5)
        model.fit(X, y)
        assert model.is_fitted

        proba = model.predict_proba(X)
        assert proba.shape == (50, 3)
        assert np.all(proba >= 0) and np.all(proba <= 1)

        preds = model.predict(X)
        assert preds.shape == (50,)
        assert set(preds).issubset({-1, 0, 1})


class TestSingleModel:
    def test_committee_of_one_works(self):
        X, y = _make_data(30, 4)
        from models.registry import build_model

        model = build_model("meta_ensemble",
                            meta_sub_models=["logistic"],
                            meta_combination_method="majority",
                            seed=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (30, 3)


class TestEmptyCommittee:
    def test_empty_committee_returns_zeros(self):
        from models.meta_ensemble import MetaEnsemble

        X, _ = _make_data(10)
        m = MetaEnsemble(sub_models=[], method="majority")
        proba = m.predict_proba(X)
        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(), 0.0)
