"""Test Step 1: Backend API canonical lists for all 17 models.

Verifies that MODEL_DESCRIPTIONS and licensing gates in
api/routers/models.py and api/licensing/gates.py are complete
and consistent with the model registry.

Note: ensemble_cnn_lstm_xgboost has a non-standard interface
(fit(X_seq, X_flat, y)) and is handled by special code paths,
not through the model registry. It is excluded from registry-based
lists but included in licensing gates (pre-existing).
"""
import pytest
from fastapi.testclient import TestClient


try:
    from api.main import app
    _API_AVAILABLE = True
except Exception:
    _API_AVAILABLE = False

client = TestClient(app) if _API_AVAILABLE else None
_API_PREFIX = "/api/v1"

# Expected model counts
_EXPECTED_TOTAL = 18  # registry-compatible models
_CLASSICAL_MODELS = {"logistic", "svm", "random_forest", "decision_tree", "xgboost", "lightgbm", "catboost"}
_DEEP_MODELS = {"cnn", "lstm", "transformer", "gru", "gru_lstm"}
_RL_MODELS = {"dqn"}
_ENSEMBLE_MODELS = {"ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost", "meta_ensemble", "stacking_ensemble", "regime_classifier"}
_ALL_DESC_EXPECTED = _CLASSICAL_MODELS | _DEEP_MODELS | _RL_MODELS | _ENSEMBLE_MODELS

# No special (non-registry) models: every model is registered and described.
_SPECIAL_MODELS = set()


class TestModelDescriptionsCompleteness:
    """MODEL_DESCRIPTIONS must cover all registered models."""

    def test_descriptions_has_all_registry_models(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        from models.registry import MODEL_REGISTRY

        registry_keys = set(MODEL_REGISTRY.keys())
        desc_keys = set(MODEL_DESCRIPTIONS.keys())

        missing_in_desc = registry_keys - desc_keys
        assert not missing_in_desc, (
            f"MODEL_DESCRIPTIONS missing {len(missing_in_desc)} model(s): {sorted(missing_in_desc)}"
        )

    def test_no_orphan_descriptions(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        from models.registry import MODEL_REGISTRY

        registry_keys = set(MODEL_REGISTRY.keys())
        desc_keys = set(MODEL_DESCRIPTIONS.keys())

        extra_in_desc = desc_keys - registry_keys
        assert not extra_in_desc, (
            f"MODEL_DESCRIPTIONS has {len(extra_in_desc)} orphan(s) not in registry: {sorted(extra_in_desc)}"
        )

    def test_total_model_count(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        assert len(MODEL_DESCRIPTIONS) == _EXPECTED_TOTAL, (
            f"Expected {_EXPECTED_TOTAL} models, got {len(MODEL_DESCRIPTIONS)}"
        )


class TestModelDescriptionsCategories:
    """Every model must have a valid category."""

    def test_category_values_are_valid(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        valid = {"classical", "deep", "rl", "ensemble"}
        for name, (display, category, desc) in MODEL_DESCRIPTIONS.items():
            assert category in valid, f"{name}: invalid category '{category}'"

    def test_classical_models(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        for name in _CLASSICAL_MODELS:
            _, category, _ = MODEL_DESCRIPTIONS[name]
            assert category == "classical", f"{name}: expected 'classical', got '{category}'"

    def test_deep_models(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        for name in _DEEP_MODELS:
            _, category, _ = MODEL_DESCRIPTIONS[name]
            assert category == "deep", f"{name}: expected 'deep', got '{category}'"

    def test_rl_models(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        for name in _RL_MODELS:
            _, category, _ = MODEL_DESCRIPTIONS[name]
            assert category == "rl", f"{name}: expected 'rl', got '{category}'"

    def test_ensemble_models(self):
        from api.routers.models import MODEL_DESCRIPTIONS
        for name in _ENSEMBLE_MODELS:
            _, category, _ = MODEL_DESCRIPTIONS[name]
            assert category == "ensemble", f"{name}: expected 'ensemble', got '{category}'"


class TestLicensingGatesCompleteness:
    """License gates must cover all registered models + special models."""

    def test_all_models_covers_all_registry_and_specials(self):
        from api.licensing.gates import ALL_MODELS
        from models.registry import MODEL_REGISTRY

        all_known = set(MODEL_REGISTRY.keys()) | _SPECIAL_MODELS
        missing = all_known - set(ALL_MODELS)
        assert not missing, (
            f"License ALL_MODELS missing {len(missing)} model(s): {sorted(missing)}"
        )

    def test_no_spurious_in_license_gates(self):
        from api.licensing.gates import ALL_MODELS
        from models.registry import MODEL_REGISTRY

        all_known = set(MODEL_REGISTRY.keys()) | _SPECIAL_MODELS
        extra = set(ALL_MODELS) - all_known
        assert not extra, (
            f"License ALL_MODELS has {len(extra)} spurious model(s) not in registry or specials: {sorted(extra)}"
        )

    def test_all_models_is_free_plus_paid(self):
        from api.licensing.gates import ALL_MODELS, FREE_MODELS, PAID_MODELS
        assert ALL_MODELS == FREE_MODELS | PAID_MODELS, (
            "ALL_MODELS must be FREE_MODELS + PAID_MODELS"
        )
        assert not (set(FREE_MODELS) & set(PAID_MODELS)), (
            "FREE_MODELS and PAID_MODELS must not overlap"
        )

    def test_free_models_are_correct(self):
        from api.licensing.gates import FREE_MODELS
        assert set(FREE_MODELS) == {"logistic", "xgboost", "random_forest"}

    def test_paid_models_cover_all_non_free(self):
        from api.licensing.gates import FREE_MODELS, PAID_MODELS
        from models.registry import MODEL_REGISTRY

        expected_paid = set(MODEL_REGISTRY.keys()) - set(FREE_MODELS) | _SPECIAL_MODELS
        assert set(PAID_MODELS) == expected_paid, (
            f"PAID_MODELS mismatch:\n  Missing: {expected_paid - set(PAID_MODELS)}\n  Extra: {set(PAID_MODELS) - expected_paid}"
        )


class TestLicensingCheckFunctions:
    """check_model and get_available_models must work correctly."""

    def test_check_model_free_user_free_model(self):
        from api.licensing.gates import check_model
        assert check_model("logistic", "free") is True
        assert check_model("xgboost", "free") is True
        assert check_model("random_forest", "free") is True

    def test_check_model_free_user_paid_model(self):
        from api.licensing.gates import check_model
        assert check_model("lightgbm", "free") is False
        assert check_model("catboost", "free") is False
        assert check_model("gru", "free") is False
        assert check_model("stacking_ensemble", "free") is False

    def test_check_model_pro_user_all_registry_models(self):
        from api.licensing.gates import check_model
        from models.registry import MODEL_REGISTRY
        for m in MODEL_REGISTRY:
            assert check_model(m, "pro") is True, f"Pro user rejected: {m}"

    def test_check_model_pro_user_special_models(self):
        from api.licensing.gates import check_model
        for m in _SPECIAL_MODELS:
            assert check_model(m, "pro") is True, f"Pro user rejected special model: {m}"

    def test_check_model_trial_user_all_known(self):
        from api.licensing.gates import check_model, ALL_MODELS
        for m in ALL_MODELS:
            assert check_model(m, "trial") is True, f"Trial user rejected: {m}"

    def test_check_model_unknown_model_free_user(self):
        from api.licensing.gates import check_model
        assert check_model("nonexistent_model", "free") is False

    def test_get_available_models_free(self):
        from api.licensing.gates import get_available_models, FREE_MODELS
        models = get_available_models("free")
        assert set(models) == set(FREE_MODELS)

    def test_get_available_models_pro(self):
        from api.licensing.gates import get_available_models, ALL_MODELS
        models = get_available_models("pro")
        assert set(models) == set(ALL_MODELS)

    def test_get_available_models_trial(self):
        from api.licensing.gates import get_available_models, ALL_MODELS
        models = get_available_models("trial")
        assert set(models) == set(ALL_MODELS)

    def test_get_available_models_unknown_plan(self):
        from api.licensing.gates import get_available_models, FREE_MODELS
        models = get_available_models("nonexistent")
        assert set(models) == set(FREE_MODELS)


@pytest.mark.skipif(not _API_AVAILABLE, reason="FastAPI not importable")
class TestModelApiEndpoint:
    """HTTP endpoint must return correct model metadata."""

    def test_get_models_returns_all_registry_models(self):
        resp = client.get(f"{_API_PREFIX}/models")
        assert resp.status_code == 200, f"Got {resp.status_code}"
        data = resp.json()
        names = {m["name"] for m in data["models"]}
        assert len(names) == _EXPECTED_TOTAL, f"Expected {_EXPECTED_TOTAL}, got {len(names)}"
        for name in _ALL_DESC_EXPECTED:
            assert name in names, f"API missing: {name}"
        for name in _SPECIAL_MODELS:
            assert name not in names, f"API should NOT include special model: {name}"

    def test_get_models_each_has_category(self):
        resp = client.get(f"{_API_PREFIX}/models")
        assert resp.status_code == 200
        data = resp.json()
        valid = {"classical", "deep", "rl", "ensemble"}
        for m in data["models"]:
            assert m["category"] in valid, f"{m['name']}: invalid category '{m['category']}'"

    def test_get_models_each_has_display_name(self):
        resp = client.get(f"{_API_PREFIX}/models")
        assert resp.status_code == 200
        data = resp.json()
        for m in data["models"]:
            assert m["display_name"], f"{m['name']}: empty display_name"

    def test_hyperparams_returns_multiple_models(self):
        resp = client.get(f"{_API_PREFIX}/models/hyperparams")
        assert resp.status_code == 200, f"Got {resp.status_code}"
        data = resp.json()
        models = data.get("models", [])
        assert len(models) > 1, (
            f"Hyperparams endpoint returned only {len(models)} model(s). "
            "This confirms the indentation bug is fixed."
        )

    def test_hyperparams_new_models_have_hpo(self):
        resp = client.get(f"{_API_PREFIX}/models/hyperparams")
        assert resp.status_code == 200
        data = resp.json()
        by_name = {m["model"]: m for m in data.get("models", [])}

        tunable_models = {"lightgbm", "catboost", "gru", "gru_lstm", "stacking_ensemble"}
        for name in tunable_models:
            assert name in by_name, f"{name} missing from hyperparams"
            assert by_name[name].get("tunable", False), f"{name} should be tunable"
