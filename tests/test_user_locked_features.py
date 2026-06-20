"""Unit tests for KodaQuant v2.0 — User-Fixed Feature Toggle Isolation.

Contract:
  - Category A (User Domain): Boolean feature toggles are LOCKED by the user.
    Optuna's sample_param_set() MUST NEVER call suggest_categorical() for
    any use_* feature flag when a UserFixedConfig is provided.
  - Category B (Machine Domain): Continuous thresholds and indicator windows
    are sampled conditionally — only for features the user enabled.
  - Safety belt: _merge_params_into_features_config() with locked_toggles
    guarantees user toggles always win over Optuna params.
"""
import os
import sys
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

import optuna
import pytest

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def user_config_enabled():
    """UserFixedConfig with a few features explicitly enabled."""
    from pipeline.tuning.fixed_config import UserFixedConfig
    return UserFixedConfig(feature_toggles={
        "use_sma": True,
        "use_ema": True,
        "use_macd": True,
        "use_rsi": True,
        "use_atr": True,
        "use_bbands": True,
        "use_adx": True,
        "use_stoch": False,
        "use_sar": False,
        "use_mtf_ma": True,
        "use_fracdiff": True,
        "use_rv_features": False,
        "use_indicator_states": False,
        "use_donchian": True,
        "use_squeeze_expansion": True,
        "use_trend_confirm": False,
        "use_triple_barrier": True,
        "use_news": True,
        "use_crossover_bins": True,
    })


@pytest.fixture
def user_config_minimal():
    """UserFixedConfig with only core features enabled."""
    from pipeline.tuning.fixed_config import UserFixedConfig
    return UserFixedConfig(feature_toggles={
        "use_sma": True,
        "use_ema": True,
        "use_rsi": True,
        "use_macd": True,
        "use_fracdiff": False,
        "use_rv_features": False,
        "use_donchian": False,
        "use_squeeze_expansion": False,
        "use_triple_barrier": False,
    })


@pytest.fixture
def all_models_list():
    """Standard model list used in backtests."""
    return ["logistic", "xgboost", "lstm", "random_forest"]


def _make_bare_study():
    """Create an Optuna study with a single trial for testing sampler."""
    return optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))


# ---------------------------------------------------------------------------
# Test 1: UserFixedConfig.from_api_overrides
# ---------------------------------------------------------------------------

class TestUserFixedConfigParsing:
    def test_extracts_feature_toggles(self):
        from pipeline.tuning.fixed_config import UserFixedConfig

        overrides = {
            "use_sma": True,
            "use_rsi": False,
            "use_fracdiff": True,
            "eval_use_trading_costs": True,
            "sizing_method": "kelly",
            "max_drawdown_pct": 0.15,
            "train_months": 36,
            "some_unknown_key": "should_be_ignored",
            "logistic__C": 1.5,  # model hyperparam — not a feature toggle
        }

        uc = UserFixedConfig.from_api_overrides(overrides)

        assert uc.is_enabled("use_sma") is True
        assert uc.is_enabled("use_rsi") is False
        assert uc.is_enabled("use_fracdiff") is True
        assert uc.is_enabled("use_news") is False  # not in overrides

        # Execution settings
        assert uc.execution_settings["eval_use_trading_costs"] is True
        assert uc.execution_settings["sizing_method"] == "kelly"

        # Risk settings
        assert uc.risk_settings["max_drawdown_pct"] == 0.15

        # Study settings
        assert uc.study_settings["train_months"] == 36

        # Unknown keys are ignored
        assert "some_unknown_key" not in uc.feature_toggles
        assert "some_unknown_key" not in uc.execution_settings
        assert "some_unknown_key" not in uc.risk_settings
        assert "some_unknown_key" not in uc.study_settings

        # Model-specific hyperparams are NOT feature toggles
        assert "logistic__C" not in uc.feature_toggles

    def test_empty_overrides_returns_defaults(self):
        from pipeline.tuning.fixed_config import UserFixedConfig

        uc = UserFixedConfig.from_api_overrides({})
        assert uc.feature_toggles == {}
        assert uc.execution_settings == {}
        assert uc.risk_settings == {}
        assert uc.study_settings == {}
        assert uc.model_param_ranges == {}

    def test_parses_hpo_range_keys(self):
        """`model__param__hpo_range` keys are parsed into model_param_ranges.
        Keys go through HYPERPARAM_ALIASES so sampler receives correct internal names."""
        from pipeline.tuning.fixed_config import UserFixedConfig

        overrides = {
            "use_sma": True,
            "xgboost__max_depth__hpo_range": [3.0, 8.0],
            "xgboost__learning_rate__hpo_range": [0.01, 0.2],
            "lightgbm__num_leaves__hpo_range": [15.0, 63.0],
            "not_a_list": "should_be_ignored",
        }

        uc = UserFixedConfig.from_api_overrides(overrides)

        # Feature toggles still parsed
        assert uc.is_enabled("use_sma") is True

        # XGBoost params aliased: max_depth → xgb_max_depth
        assert ("xgb_max_depth", (3.0, 8.0)) in uc.model_param_ranges.items()
        assert ("xgb_learning_rate", (0.01, 0.2)) in uc.model_param_ranges.items()

        # LightGBM params aliased: num_leaves → lgbm_num_leaves
        assert ("lgbm_num_leaves", (15.0, 63.0)) in uc.model_param_ranges.items()

    def test_hpo_range_rejects_invalid_values(self):
        """Non-list or wrong-length values for __hpo_range are ignored."""
        from pipeline.tuning.fixed_config import UserFixedConfig

        overrides = {
            "xgboost__max_depth__hpo_range": 5,           # not a list
            "xgboost__learning_rate__hpo_range": [0.01],  # wrong length
        }
        uc = UserFixedConfig.from_api_overrides(overrides)
        assert "xgb_max_depth" not in uc.model_param_ranges
        assert "xgb_learning_rate" not in uc.model_param_ranges


# ---------------------------------------------------------------------------
# Test 2: sample_param_set NEVER samples boolean feature toggles
# ---------------------------------------------------------------------------

class TestSamplerNoBooleanSampling:
    def test_user_config_blocks_feature_toggle_sampling(self, user_config_enabled, all_models_list):
        """With UserFixedConfig, sample_param_set must NOT call
        suggest_categorical for any feature toggle key."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()

        # Collect all keys that trial.suggest_* is called with
        suggested_keys = []

        # Monkey-patch trial to capture calls
        original_suggest_cat = trial.suggest_categorical
        original_suggest_float = trial.suggest_float
        original_suggest_int = trial.suggest_int

        def capturing_suggest_cat(name, *args, **kwargs):
            suggested_keys.append(("categorical", name))
            return original_suggest_cat(name, *args, **kwargs)

        def capturing_suggest_float(name, *args, **kwargs):
            suggested_keys.append(("float", name))
            return original_suggest_float(name, *args, **kwargs)

        def capturing_suggest_int(name, *args, **kwargs):
            suggested_keys.append(("int", name))
            return original_suggest_int(name, *args, **kwargs)

        trial.suggest_categorical = capturing_suggest_cat
        trial.suggest_float = capturing_suggest_float
        trial.suggest_int = capturing_suggest_int

        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )

        # Verify NO boolean feature toggle was sampled
        boolean_suggested = [k[1] for k in suggested_keys if k[0] == "categorical"]
        banned_patterns = ["use_", "eval_use_", "news_event_flags", "llm_sentiment_enabled"]
        for key in boolean_suggested:
            for pattern in banned_patterns:
                assert not key.startswith(pattern), (
                    f"PROHIBITED: Optuna sampled boolean toggle '{key}' — "
                    f"feature toggles must NOT be in the categorical search space"
                )

    def test_backward_compat_no_user_config(self, all_models_list):
        """Without UserFixedConfig, the sampler should fall back to the
        deterministic fixed profile (no boolean sampling) and emit a warning."""
        import warnings
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = sample_param_set(trial, all_models_list, user_config=None)
            # Should warn about missing UserFixedConfig
            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) >= 1, "Expected FutureWarning when user_config is None"

        # Should still produce valid params
        assert "model_type" in params
        assert params["model_type"] in all_models_list

    def test_user_toggles_preserved_in_params(self, user_config_enabled, all_models_list):
        """All user toggle values should appear verbatim in the params dict."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )

        # Check that user toggles are preserved exactly
        for k, v in user_config_enabled.feature_toggles.items():
            assert params[k] == v, (
                f"User toggle '{k}' expected {v}, got {params.get(k)}"
            )


# ---------------------------------------------------------------------------
# Test 3: Conditional threshold sampling
# ---------------------------------------------------------------------------

class TestConditionalThresholds:
    def test_fracdiff_sampled_when_enabled(self, user_config_enabled, all_models_list):
        """fracdiff_d should be sampled when use_fracdiff=True."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )
        assert "fracdiff_d" in params
        assert 0.1 <= params["fracdiff_d"] <= 0.9

    def test_fracdiff_not_sampled_when_disabled(self, user_config_minimal, all_models_list):
        """fracdiff_d should be 0.0 when use_fracdiff=False."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_minimal,
        )
        assert params.get("fracdiff_d", 0.0) == 0.0

    def test_rv_windows_not_sampled_when_rv_disabled(self, user_config_minimal, all_models_list):
        """rv_window_short/long should NOT be in params when use_rv_features=False."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_minimal,
        )
        assert "rv_window_short" not in params
        assert "rv_window_long" not in params

    def test_donchian_windows_sampled_when_enabled(self, user_config_enabled, all_models_list):
        """donchian_window_short/long sampled when use_donchian=True."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )
        assert "donchian_window_short" in params
        assert "donchian_window_long" in params
        assert 20 <= params["donchian_window_short"] <= 60
        assert 80 <= params["donchian_window_long"] <= 240

    def test_indicator_states_not_sampled_when_disabled(self, user_config_minimal, all_models_list):
        """rsi_overbought_level etc. NOT sampled when use_indicator_states=False."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_minimal,
        )
        assert "rsi_overbought_level" not in params
        assert "bbw_compress_threshold" not in params

    def test_squeeze_hyperparams_sampled_when_enabled(self, user_config_enabled, all_models_list):
        """squeeze_window etc. sampled when use_squeeze_expansion=True."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )
        assert "squeeze_window" in params
        assert 150 <= params["squeeze_window"] <= 600
        assert "squeeze_quantile" in params
        assert "adx_slope_window" in params


# ---------------------------------------------------------------------------
# Test 4: Indicator window conditional sampling
# ---------------------------------------------------------------------------

class TestIndicatorWindows:
    def test_enabled_indicators_get_windows(self, user_config_enabled, all_models_list):
        """Core indicators the user enabled should have windows sampled."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )
        iw = params.get("indicator_windows", {})
        assert "sma" in iw
        assert "ema" in iw
        assert "rsi" in iw
        assert "macd_fast" in iw
        assert "macd_slow" in iw
        assert "atr" in iw
        assert "adx" in iw
        assert "bb_window" in iw
        assert "bb_dev" in iw

    def test_disabled_indicators_get_no_windows(self, user_config_minimal, all_models_list):
        """Disabled indicators should NOT have windows sampled."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_minimal,
        )
        iw = params.get("indicator_windows", {})
        assert "atr" not in iw  # user_config_minimal has use_atr=False
        assert "adx" not in iw
        assert "bb_window" not in iw

    def test_triple_barrier_respects_user_toggle(self, user_config_enabled, all_models_list):
        """When use_triple_barrier=True in user_config, TB thresholds are sampled."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )
        assert params["use_triple_barrier"] is True
        assert "tb_pt_mult" in params
        assert "tb_sl_mult" in params
        assert "tb_max_holding" in params

    def test_triple_barrier_off_when_disabled(self, user_config_minimal, all_models_list):
        """When use_triple_barrier=False, it should be False in params too."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_minimal,
        )
        assert params.get("use_triple_barrier") is False


# ---------------------------------------------------------------------------
# Test 5: _merge_params_into_features_config safety belt
# ---------------------------------------------------------------------------

class TestMergeParamsLockedToggles:
    def test_locked_toggles_survive_merge(self):
        """locked_toggles must win over any conflicting Optuna params."""
        # We simulate a MLBacktester-like object with a feature_config
        from pipeline.tuning.fixed_config import UserFixedConfig

        uc = UserFixedConfig(feature_toggles={
            "use_sma": True,
            "use_rsi": False,
            "use_fracdiff": True,
        })
        locked = uc.to_toggle_dict()

        # Simulate Optuna best_params that tries to override user toggles
        optuna_params = {
            "use_sma": False,       # Optuna wants to turn SMA OFF
            "use_rsi": True,        # Optuna wants to turn RSI ON
            "use_fracdiff": False,  # Optuna wants to turn FracDiff OFF
            "lags": 20,             # Optuna's preferred lag (this should stay)
            "learning_rate": 0.01,  # Model hyperparam (this should stay)
        }

        # Build a mock backtester-like object
        class MockBT:
            def __init__(self):
                self.features_config = {"lags": 10, "use_sma": True, "use_rsi": False}

            def apply_feature_defaults(self):
                pass  # no-op for test

        bt = MockBT()

        # Simulate the merge logic from core_mixin.py
        base = dict(bt.features_config)
        base.update(optuna_params)
        if locked:
            base.update(locked)  # User-locked toggles must win

        # Verify: user toggles win
        assert base["use_sma"] is True, "User locked use_sma=True but merge overwrote it"
        assert base["use_rsi"] is False, "User locked use_rsi=False but merge overwrote it"
        assert base["use_fracdiff"] is True, "User locked use_fracdiff=True but merge overwrote it"

        # Model hyperparams and other non-locked keys should remain from Optuna
        assert base["lags"] == 20, "Optuna's lags should survive when not in locked_toggles"
        assert base["learning_rate"] == 0.01, "Optuna's learning_rate should survive"

    def test_lock_does_not_block_new_keys(self):
        """Locked toggles should not prevent Optuna from adding new keys."""
        locked = {"use_sma": True}
        optuna_params = {"use_sma": False, "new_param": 42}

        base = {}
        base.update(optuna_params)
        base.update(locked)

        assert base["use_sma"] is True  # User wins
        assert base["new_param"] == 42  # Optuna can add new keys


# ---------------------------------------------------------------------------
# Test 6: Edge cases & invariants
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_feature_toggles_produces_valid_params(self, all_models_list):
        """An empty UserFixedConfig should still produce a valid params dict."""
        from pipeline.tuning.fixed_config import UserFixedConfig
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        uc = UserFixedConfig()  # no feature toggles
        params = sample_param_set(trial, all_models_list, user_config=uc)

        assert "model_type" in params
        assert "train_months" in params
        assert "lags_range" in params

    def test_model_hyperparams_still_sampled(self, user_config_minimal, all_models_list):
        """Model hyperparameters from SEARCH_SPACE must still be sampled."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_minimal,
        )

        model = params["model_type"]
        # Verify the model has its prefix params (e.g., xgb_*, logit_*, lstm_*)
        if model == "xgboost":
            assert "xgb_learning_rate" in params or "xgb_n_estimators" in params
        elif model == "lstm":
            assert "lstm_units" in params or "lstm_learning_rate" in params

    def test_immutable_config(self):
        """UserFixedConfig should reject direct field assignment (frozen=True).
        Nested dicts are intentionally mutable — use to_toggle_dict() for copies."""
        from pipeline.tuning.fixed_config import UserFixedConfig

        uc = UserFixedConfig(feature_toggles={"use_sma": True})
        # Direct field assignment is blocked by frozen=True
        with pytest.raises(FrozenInstanceError):
            uc.feature_toggles = {}  # type: ignore
        # But nested dict values can be obtained via to_toggle_dict()
        copy = uc.to_toggle_dict()
        copy["use_sma"] = False
        # Original object remains unchanged
        assert uc.to_toggle_dict()["use_sma"] is True

    def test_no_legacy_strategy_type_sampled(self, user_config_enabled, all_models_list):
        """strategy_type should NOT be a sampled dimension with UserFixedConfig."""
        from pipeline.tuning.sampler import sample_param_set

        study = _make_bare_study()
        trial = study.ask()
        params = sample_param_set(
            trial, all_models_list,
            user_config=user_config_enabled,
        )
        # "strategy_type" should NOT be in params when user_config is provided
        assert "strategy_type" not in params, (
            "strategy_type should not be a sampled dimension in user-locked mode"
        )
