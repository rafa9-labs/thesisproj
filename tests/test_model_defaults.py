"""Test suite for KodaQuant v3.0 Three-Tier Hyperparameter Architecture.

Contract:
  - ModelDefaultsConfig is the single source of truth for all defaults.
  - SEARCH_SPACE auto-derived from Tier 1 params only.
  - FIXED_DEFAULTS auto-derived from Tier 3 params only.
  - HYPERPARAM_ALIASES covers all 16 tunable models.
  - Sampler produces correct registry prefixes (lgbm_ not lightgbm_, cb_ not catboost_).
  - All 8 prior discrepancies resolved to unified defaults.

Classes:
  TestStructuralIntegrity   — global pattern checks (5 tests)
  TestTierAssignments        — exact counts for critical models, patterns for others (6 tests)
  TestDerivedConfigs         — SEARCH_SPACE + FIXED_DEFAULTS derivation (3 tests)
  TestDiscrepancyResolution  — 8 discrepancies now unified (5 tests)
  TestAliasCoverage          — HYPERPARAM_ALIASES completeness (3 tests)
  TestSamplerIntegration     — prefix correctness in production sampler (4 tests)
  TestEdgeCases              — defensive behaviour (2 tests)
"""

import os
import sys
from copy import deepcopy

import pytest
import optuna


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def all_params():
    from pipeline.models.model_defaults import MODEL_PARAMS
    return MODEL_PARAMS


def _make_bare_study():
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Class 1 — Structural Integrity (global pattern scans)
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralIntegrity:

    def test_all_18_models_defined(self, all_params):
        assert len(all_params) == 18, f"Expected 18 models, got {len(all_params)}"
        expected_models = {
            "logistic", "svm", "random_forest", "decision_tree",
            "xgboost", "lightgbm", "catboost",
            "cnn", "lstm", "transformer", "gru", "gru_lstm",
            "dqn",
            "ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost",
            "meta_ensemble", "stacking_ensemble",
            "regime_classifier",
        }
        actual = set(all_params.keys())
        missing = expected_models - actual
        extra = actual - expected_models
        assert not missing, f"Missing models: {missing}"
        assert not extra, f"Unexpected models: {extra}"

    def test_no_duplicate_keys_within_model(self, all_params):
        """No two params in the SAME model share the same internal key.
        Cross-model key sharing (e.g. ensemble sub-models reusing lstm_ prefix)
        is intentional — the registry uses filter_params to namespace them."""
        for model_key, params in all_params.items():
            keys = [p.key for p in params]
            duplicates = {k for k in keys if keys.count(k) > 1}
            assert not duplicates, (
                f"Model '{model_key}' has duplicate internal keys: {duplicates}"
            )

    def test_all_tier1_have_non_none_range(self, all_params):
        for model_key, params in all_params.items():
            for p in params:
                if p.tier == 1:
                    assert p.range is not None, (
                        f"{model_key}::{p.key} is Tier 1 but has range=None"
                    )

    def test_all_tier2_have_non_none_range(self, all_params):
        """Tier 2 params must have a defined range (list for choices, tuple for numeric)."""
        for model_key, params in all_params.items():
            for p in params:
                if p.tier == 2:
                    assert p.range is not None, (
                        f"{model_key}::{p.key} is Tier 2 but has range=None"
                    )

    def test_all_tier3_have_non_none_default(self, all_params):
        for model_key, params in all_params.items():
            for p in params:
                if p.tier == 3:
                    assert p.default is not None, (
                        f"{model_key}::{p.key} is Tier 3 but has default=None"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# Class 2 — Tier Assignments (exact for critical, pattern for others)
# ═══════════════════════════════════════════════════════════════════════════

class TestTierAssignments:

    def test_logistic_tier_spread(self, all_params):
        """Logistic: exactly 1 T1 (C), 1 T2 (class_weight), 4 T3."""
        params = all_params["logistic"]
        t1 = [p for p in params if p.tier == 1]
        t2 = [p for p in params if p.tier == 2]
        t3 = [p for p in params if p.tier == 3]
        assert len(t1) == 1, f"Expected 1 T1, got {len(t1)}: {[p.key for p in t1]}"
        assert t1[0].key == "logit_C", f"Expected C as only T1, got {t1[0].key}"
        assert len(t2) == 1, f"Expected 1 T2, got {len(t2)}: {[p.key for p in t2]}"
        assert t2[0].key == "logit_class_weight", f"Expected class_weight as T2, got {t2[0].key}"
        assert len(t3) == 4, f"Expected 4 T3, got {len(t3)}"

    def test_ensemble_adaptive_pruned(self, all_params):
        """Adaptive regime: pruned from 11 to exactly 3 T1 params."""
        params = all_params["ensemble_adaptive_regime"]
        t1 = [p for p in params if p.tier == 1]
        assert len(t1) == 3, (
            f"AdaptiveRegime must have exactly 3 T1 params (pruned from 11). "
            f"Got {len(t1)}: {[p.key for p in t1]}"
        )

    def test_ensemble_cnn_lstm_xgb_pruned(self, all_params):
        """CNN-LSTM-XGB ensemble: pruned from 9 to exactly 4 T1 params."""
        params = all_params["ensemble_cnn_lstm_xgboost"]
        t1 = [p for p in params if p.tier == 1]
        assert len(t1) == 4, (
            f"CNN-LSTM-XGB must have exactly 4 T1 params (pruned from 9). "
            f"Got {len(t1)}: {[p.key for p in t1]}"
        )

    def test_meta_stacking_have_zero_t1(self, all_params):
        """Meta ensemble and stacking ensemble have 0 T1 params (T2 only)."""
        for model_key in ("meta_ensemble", "stacking_ensemble"):
            params = all_params[model_key]
            t1 = [p for p in params if p.tier == 1]
            assert len(t1) == 0, (
                f"{model_key} must have 0 T1 params. Got {len(t1)}: {[p.key for p in t1]}"
            )

    def test_dqn_regime_have_only_t3(self, all_params):
        """DQN and regime_classifier have ONLY Tier 3 params (no HPO, no user dropdowns)."""
        for model_key in ("dqn", "regime_classifier"):
            params = all_params[model_key]
            non_t3 = [p for p in params if p.tier != 3]
            assert len(non_t3) == 0, (
                f"{model_key} must have only T3 params. "
                f"Got non-T3: {[(p.key, p.tier) for p in non_t3]}"
            )

    def test_all_other_models_have_at_least_one_t1(self, all_params):
        """Every model not in the 'T2 only' or 'T3 only' set must have T1 params."""
        excluded = {"meta_ensemble", "stacking_ensemble", "dqn", "regime_classifier"}
        for model_key, params in all_params.items():
            if model_key in excluded:
                continue
            t1 = [p for p in params if p.tier == 1]
            assert len(t1) >= 1, (
                f"{model_key} is a tunable model but has 0 T1 params"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Class 3 — Derived Configs (SEARCH_SPACE + FIXED_DEFAULTS)
# ═══════════════════════════════════════════════════════════════════════════

class TestDerivedConfigs:

    def test_search_space_contains_only_tier1(self, all_params):
        from pipeline.models.model_defaults import build_search_space

        ss = build_search_space()
        for model_key, entries in ss.items():
            model_params = {p.hpo_key: p for p in all_params.get(model_key, [])}
            for hpo_key in entries:
                p = model_params.get(hpo_key)
                assert p is not None, (
                    f"SEARCH_SPACE key '{hpo_key}' not found in {model_key} params"
                )
                assert p.tier == 1, (
                    f"SEARCH_SPACE key '{hpo_key}' in {model_key} is tier {p.tier}, expected 1"
                )

    def test_fixed_defaults_contains_only_tier3(self, all_params):
        from pipeline.models.model_defaults import build_fixed_defaults

        fd = build_fixed_defaults()
        for model_key, entries in fd.items():
            model_params = {p.key: p for p in all_params.get(model_key, [])}
            for internal_key in entries:
                p = model_params.get(internal_key)
                assert p is not None, (
                    f"FIXED_DEFAULTS key '{internal_key}' not found in {model_key} params"
                )
                assert p.tier == 3, (
                    f"FIXED_DEFAULTS key '{internal_key}' in {model_key} is tier {p.tier}, expected 3"
                )

    def test_logistic_search_space_is_C_only(self):
        from config import SEARCH_SPACE

        ss_logistic = SEARCH_SPACE.get("logistic", {})
        assert set(ss_logistic.keys()) == {"C"}, (
            f"Logistic SEARCH_SPACE should be {'C'} only, got {set(ss_logistic.keys())}"
        )
        c_spec = ss_logistic["C"]
        assert isinstance(c_spec, tuple)
        assert c_spec[0] == 0.01
        assert c_spec[1] == 100.0
        assert c_spec[2] is True  # log_scale


# ═══════════════════════════════════════════════════════════════════════════
# Class 4 — Discrepancy Resolution (8 bugs now fixed)
# ═══════════════════════════════════════════════════════════════════════════

class TestDiscrepancyResolution:

    def test_logistic_solver_lbfgs_everywhere(self, all_params):
        p = next(x for x in all_params["logistic"] if x.key == "logit_solver")
        assert p.default == "lbfgs", f"logit_solver default is {p.default}, expected lbfgs"
        assert p.tier == 3
        from config import FIXED_DEFAULTS
        assert FIXED_DEFAULTS["logistic"]["logit_solver"] == "lbfgs"

    def test_logistic_max_iter_1000(self, all_params):
        p = next(x for x in all_params["logistic"] if x.key == "logit_max_iter")
        assert p.default == 1000, f"logit_max_iter default is {p.default}, expected 1000"
        from config import FIXED_DEFAULTS
        assert FIXED_DEFAULTS["logistic"]["logit_max_iter"] == 1000

    def test_logistic_tol_1e_4(self, all_params):
        p = next(x for x in all_params["logistic"] if x.key == "logit_tol")
        assert p.default == 0.0001, f"logit_tol default is {p.default}, expected 0.0001"

    def test_lstm_clipnorm_1_dot_0(self, all_params):
        p = next(x for x in all_params["lstm"] if x.key == "lstm_clipnorm")
        assert p.default == 1.0, f"lstm_clipnorm default is {p.default}, expected 1.0"
        from config import FIXED_DEFAULTS
        assert FIXED_DEFAULTS["lstm"]["lstm_clipnorm"] == 1.0

    def test_transformer_pooling_cls(self, all_params):
        p = next(x for x in all_params["transformer"] if x.key == "transformer_pooling")
        assert p.default == "cls", f"transformer_pooling default is {p.default}, expected cls"
        from config import FIXED_DEFAULTS
        assert FIXED_DEFAULTS["transformer"]["transformer_pooling"] == "cls"

    def test_rf_class_weight_balanced_subsample(self, all_params):
        p = next(x for x in all_params["random_forest"] if x.key == "rf_class_weight")
        assert p.default == "balanced_subsample", (
            f"rf_class_weight default is {p.default}, expected balanced_subsample"
        )
        # Verify sampler no longer injects "balanced" via FIXED_DEFAULTS
        from config import FIXED_DEFAULTS
        assert FIXED_DEFAULTS["random_forest"]["rf_class_weight"] == "balanced_subsample"


# ═══════════════════════════════════════════════════════════════════════════
# Class 5 — Alias Coverage (frontend-to-backend key mapping)
# ═══════════════════════════════════════════════════════════════════════════

class TestAliasCoverage:

    def test_all_16_tunable_models_have_aliases(self, all_params):
        from pipeline.hpo.hyperparam_aliases import HYPERPARAM_ALIASES

        tunable = {
            k for k, v in all_params.items()
            if any(p.tier == 1 or p.tier == 2 for p in v)
        }
        aliased = set(HYPERPARAM_ALIASES.keys())
        missing = tunable - aliased
        assert not missing, (
            f"Models with T1/T2 params but no HYPERPARAM_ALIASES entry: {missing}"
        )

    def test_cnn_filters1_filters2_both_mapped(self):
        from pipeline.hpo.hyperparam_aliases import HYPERPARAM_ALIASES

        cnn_aliases = HYPERPARAM_ALIASES.get("cnn", {})
        assert "filters1" in cnn_aliases, (
            f"cnn__filters1 not mapped. CNN aliases: {cnn_aliases}"
        )
        assert "filters2" in cnn_aliases, (
            f"cnn__filters2 not mapped. CNN aliases: {cnn_aliases}"
        )
        assert cnn_aliases["filters1"] == "cnn_filters1"
        assert cnn_aliases["filters2"] == "cnn_filters2"

    def test_lightgbm_catboost_prefixes_mapped(self):
        from pipeline.hpo.hyperparam_aliases import HYPERPARAM_ALIASES

        lgbm = HYPERPARAM_ALIASES.get("lightgbm", {})
        assert "max_depth" in lgbm, "lightgbm missing max_depth alias"
        assert lgbm["max_depth"] == "lgbm_max_depth", (
            f"Expected lgbm_max_depth, got {lgbm['max_depth']}"
        )
        assert "n_estimators" in lgbm
        assert lgbm["n_estimators"] == "lgbm_n_estimators"

        cb = HYPERPARAM_ALIASES.get("catboost", {})
        assert "iterations" in cb, "catboost missing iterations alias"
        assert cb["iterations"] == "cb_iterations", (
            f"Expected cb_iterations, got {cb['iterations']}"
        )
        assert "depth" in cb
        assert cb["depth"] == "cb_depth"


# ═══════════════════════════════════════════════════════════════════════════
# Class 6 — Sampler Integration (production prefix validation)
# ═══════════════════════════════════════════════════════════════════════════

class TestSamplerIntegration:

    def test_sampler_lightgbm_produces_lgbm_prefix(self):
        """LightGBM params must use lgbm_ prefix (not lightgbm_) in sampler output."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(trial, ["lightgbm"])

        # HPO-sampled T1 params: must use lgbm_ prefix
        lgbm_keys = [k for k in params if k.startswith("lgbm_")]
        assert len(lgbm_keys) > 0, (
            f"No lgbm_* keys in params. Sampler may still be using lightgbm_ prefix. "
            f"Keys found: {sorted(params.keys())[:20]}"
        )
        # Verify NO lightgbm_ prefix exists
        wrong = [k for k in params if k.startswith("lightgbm_")]
        assert len(wrong) == 0, (
            f"Sampler produced lightgbm_ prefixed keys (will be ignored by registry): {wrong}"
        )
        # Verify T3 defaults injected with correct prefix
        assert params.get("lgbm_boosting_type") == "gbdt", (
            f"Expected lgbm_boosting_type=gbdt, got {params.get('lgbm_boosting_type')}"
        )
        assert params.get("lgbm_min_child_samples") == 20

    def test_sampler_catboost_produces_cb_prefix(self):
        """CatBoost params must use cb_ prefix (not catboost_) in sampler output."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(trial, ["catboost"])

        cb_keys = [k for k in params if k.startswith("cb_")]
        assert len(cb_keys) > 0, (
            f"No cb_* keys in params. Sampler may still be using catboost_ prefix. "
            f"Keys found: {sorted(params.keys())[:20]}"
        )
        wrong = [k for k in params if k.startswith("catboost_")]
        assert len(wrong) == 0, "Sampler produced catboost_ prefixed keys"
        assert params.get("cb_border_count") == 128
        assert params.get("cb_loss_function") == "MultiClass"

    def test_sampler_handles_meta_ensemble_gracefully(self):
        """meta_ensemble has no SEARCH_SPACE (0 T1) — sampler must not crash."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(trial, ["meta_ensemble"])
        assert params.get("model_type") == "meta_ensemble", (
            f"Expected meta_ensemble as model_type, got {params.get('model_type')}"
        )
        # T3 defaults should still be injected
        assert params.get("meta_sub_models") is not None

    def test_sampler_ensemble_adaptive_no_double_prefix(self):
        """Ensemble params must not be double-prefixed (no __ prefix duplication)."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(trial, ["ensemble_adaptive_regime"])

        # These params come from SEARCH_SPACE for the ensemble (empty prefix)
        # They should be unprefixed or use sub-model prefixes (rf_, lstm_)
        param_keys = set(params.keys())
        # Must NOT contain double-prefixed keys like ensemble_adaptive_regime_rf_max_depth
        bad = [k for k in param_keys if "__" in str(k)]
        assert len(bad) == 0, (
            f"Double-prefixed keys found in ensemble params: {bad}"
        )

    def test_sampler_respects_user_hpo_ranges(self):
        """When UserFixedConfig.model_param_ranges is set, sampler constrains bounds."""
        from pipeline.tuning.fixed_config import UserFixedConfig
        from pipeline.tuning.sampler import sample_param_set

        # User defines narrowed range for xgboost max_depth: [4, 6] instead of [3, 8]
        uc = UserFixedConfig(model_param_ranges={
            "xgb_max_depth": (4.0, 6.0),
        })

        # Run multiple trials to ensure constraint holds across randomness
        for _ in range(10):
            study = _make_bare_study()
            trial = study.ask()
            params = sample_param_set(trial, ["xgboost"], user_config=uc)
            if params.get("model_type") == "xgboost":
                val = params.get("xgb_max_depth")
                assert val is not None, "xgb_max_depth not in params"
                assert 4 <= val <= 6, (
                    f"xgb_max_depth={val} outside user range [4, 6]"
                )

    def test_sampler_user_range_no_effect_on_unrelated_params(self):
        """User HPO range only affects the specified param, not others."""
        from pipeline.tuning.fixed_config import UserFixedConfig
        from pipeline.tuning.sampler import sample_param_set

        uc = UserFixedConfig(model_param_ranges={
            "xgb_max_depth": (4.0, 6.0),
        })

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(trial, ["xgboost"], user_config=uc)
        if params.get("model_type") == "xgboost":
            # Other params should still be sampled from full SEARCH_SPACE ranges
            lr = params.get("xgb_learning_rate")
            assert lr is not None
            # Full range is [0.005, 0.3] — should be within that
            assert 0.005 <= lr <= 0.3, f"learning_rate={lr} outside [0.005, 0.3]"


# ═══════════════════════════════════════════════════════════════════════════
# Class 7 — Edge Cases (defensive behaviour)
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_get_defaults_unknown_model_returns_empty(self):
        from pipeline.models.model_defaults import get_defaults

        result = get_defaults("nonexistent_model_v42")
        assert result == {}, f"Expected {{}}, got {result}"

    def test_build_funcs_handle_empty_model_params(self, monkeypatch):
        import pipeline.models.model_defaults as md

        original = md.MODEL_PARAMS
        try:
            monkeypatch.setattr(md, "MODEL_PARAMS", {})
            ss = md.build_search_space()
            assert ss == {}, f"build_search_space with empty MODEL_PARAMS should return {{}}, got {ss}"
            fd = md.build_fixed_defaults()
            assert fd == {}, f"build_fixed_defaults with empty MODEL_PARAMS should return {{}}, got {fd}"
        finally:
            monkeypatch.setattr(md, "MODEL_PARAMS", original)
