"""Tests for the model registry (Step 2.8)."""
import pytest


class TestBaseModel:
    """BaseModel ABC tests."""

    def test_import(self):
        from models.base_model import BaseModel
        assert BaseModel is not None

    def test_cannot_instantiate(self):
        from models.base_model import BaseModel
        with pytest.raises(TypeError):
            BaseModel()

    def test_concrete_subclass(self):
        from models.base_model import BaseModel

        class DummyModel(BaseModel):
            model_type = "dummy"
            def fit(self, X, y, **kw): self._fitted = True; return self
            def predict(self, X): return [0]

        m = DummyModel()
        assert m.model_type == "dummy"
        assert not m.is_fitted
        m.fit([1], [0])
        assert m.is_fitted
        assert m.predict([1]) == [0]
        assert m.get_params() == {}

    def test_predict_proba_default_raises(self):
        from models.base_model import BaseModel

        class NoProba(BaseModel):
            model_type = "noprob"
            def fit(self, X, y, **kw): return self
            def predict(self, X): return []

        m = NoProba()
        with pytest.raises(NotImplementedError, match="predict_proba"):
            m.predict_proba([1])


class TestRegistry:
    """Registry + builder tests (classical models only — no TF/GPU needed)."""

    @pytest.fixture(autouse=True)
    def _import_registry(self):
        from models import registry
        self.reg = registry

    def test_registry_has_all_types(self):
        expected = {
            "logistic", "svm", "random_forest", "decision_tree", "xgboost",
            "cnn", "lstm", "transformer", "dqn", "ensemble_adaptive_regime",
        }
        assert expected.issubset(set(self.reg.MODEL_REGISTRY.keys()))

    def test_build_logistic(self):
        model = self.reg.build_model("logistic", seed=42)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_build_random_forest(self):
        model = self.reg.build_model("random_forest", seed=42)
        assert model is not None
        assert hasattr(model, "fit")

    def test_build_decision_tree(self):
        model = self.reg.build_model("decision_tree", seed=42)
        assert model is not None

    def test_build_unknown_returns_none(self):
        assert self.reg.build_model("nonexistent_model") is None

    def test_logistic_fit_predict(self):
        """End-to-end: build → fit → predict with tiny data."""
        import numpy as np
        model = self.reg.build_model("logistic", seed=42)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 5))
        y = rng.integers(0, 3, size=60)
        model.fit(X, y)
        preds = model.predict(X[:10])
        proba = model.predict_proba(X[:10])
        assert len(preds) == 10
        assert proba.shape == (10, 3)

    def test_lazy_load_via_package(self):
        """models.build_model lazy-loads from registry."""
        from models import build_model
        assert build_model is not None
        m = build_model("decision_tree", seed=0)
        assert m is not None