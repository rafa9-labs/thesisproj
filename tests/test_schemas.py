"""
tests.test_schemas — Validates all Pydantic models in the schemas/ package.

═══════════════════════════════════════════════════════════════════════════════
EDUCATIONAL: What these tests demonstrate
═══════════════════════════════════════════════════════════════════════════════

1. Construction from defaults     → models work with no input
2. Construction from dicts        → model_validate() parses raw dicts
3. Type coercion                  → "42" becomes 42, "3.14" becomes 3.14
4. Constraint enforcement         → ge/le constraints reject invalid values
5. Cross-field validation         → model_validator catches inconsistent combos
6. Serialization roundtrip        → model_dump() → model_validate() is lossless
7. JSON schema generation         → .model_json_schema() works
8. Extra key handling             → extra="ignore" drops unknown keys
9. Backward compatibility         → CLASS_DEFAULTS dict parses cleanly
10. Nested model validation       → inner models are validated recursively
"""

import json
import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError


# ═══════════════════════════════════════════════════════════════════════════
# Test BacktestParams
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestParams:
    """Tests for schemas.backtest.BacktestParams."""
    
    def test_default_construction(self):
        """Model can be constructed with all defaults."""
        from schemas.backtest import BacktestParams
        params = BacktestParams()
        assert params.model_type == "logistic"
        assert params.train_months == 36
        assert params.confidence_threshold == 0.80
        assert params.calibrate_method == "sigmoid"
    
    def test_from_dict(self):
        """model_validate() parses a raw dict correctly."""
        from schemas.backtest import BacktestParams
        raw = {
            "model_type": "xgboost",
            "train_months": 24,
            "confidence_threshold": 0.9,
            "unknown_key": "should be ignored",
        }
        params = BacktestParams.model_validate(raw)
        assert params.model_type == "xgboost"
        assert params.train_months == 24
        assert params.confidence_threshold == 0.9
        # Unknown key is silently ignored (extra="ignore")
        assert not hasattr(params, "unknown_key")
    
    def test_type_coercion(self):
        """Pydantic coerces compatible types: '42' → 42, '3.14' → 3.14."""
        from schemas.backtest import BacktestParams
        raw = {
            "train_months": "24",      # str → int
            "confidence_threshold": "0.5",  # str → float
            "use_triple_barrier": "true",   # str → bool
        }
        params = BacktestParams.model_validate(raw)
        assert params.train_months == 24
        assert isinstance(params.train_months, int)
        assert params.confidence_threshold == 0.5
        assert isinstance(params.confidence_threshold, float)
        assert params.use_triple_barrier is True
    
    def test_constraint_enforcement(self):
        """Field constraints reject out-of-range values."""
        from schemas.backtest import BacktestParams
        
        # confidence_threshold must be [0.0, 1.0]
        with pytest.raises(ValidationError) as exc_info:
            BacktestParams(confidence_threshold=5.0)
        assert "less than or equal to 1" in str(exc_info.value).lower()
        
        # train_months must be >= 1
        with pytest.raises(ValidationError):
            BacktestParams(train_months=0)
        
        # fracdiff_d must be [0.0, 1.0]
        with pytest.raises(ValidationError):
            BacktestParams(fracdiff_d=1.5)
    
    def test_literal_constraint(self):
        """Literal[...] rejects values not in the allowed set."""
        from schemas.backtest import BacktestParams
        
        with pytest.raises(ValidationError):
            BacktestParams(calibrate_method="invalid_method")
    
    def test_cross_field_validation(self):
        """model_validator catches inconsistent field combinations."""
        from schemas.backtest import BacktestParams
        
        # lbfgs + l1 is incompatible
        with pytest.raises(ValidationError):
            BacktestParams(logit_solver="lbfgs", logit_penalty="l1")
        
        # saga + l1 is fine
        params = BacktestParams(logit_solver="saga", logit_penalty="l1")
        assert params.logit_penalty == "l1"
    
    def test_feature_explosion_guard(self):
        """lags * lag_depth > 100 raises ValidationError."""
        from schemas.backtest import BacktestParams
        
        with pytest.raises(ValidationError) as exc_info:
            BacktestParams(lags=50, lag_depth=3)  # 150 features
        assert "features" in str(exc_info.value).lower()
    
    def test_serialization_roundtrip(self):
        """model_dump() → model_validate() is lossless."""
        from schemas.backtest import BacktestParams
        
        original = BacktestParams(model_type="cnn", train_months=12, seed=123)
        dumped = original.model_dump()
        restored = BacktestParams.model_validate(dumped)
        
        assert restored.model_type == original.model_type
        assert restored.train_months == original.train_months
        assert restored.seed == original.seed
    
    def test_json_schema_generation(self):
        """model_json_schema() produces a valid JSON Schema."""
        from schemas.backtest import BacktestParams
        
        schema = BacktestParams.model_json_schema()
        assert "properties" in schema
        assert "model_type" in schema["properties"]
        assert "confidence_threshold" in schema["properties"]


# ═══════════════════════════════════════════════════════════════════════════
# Test AggregateMetrics
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregateMetrics:
    """Tests for schemas.backtest.AggregateMetrics."""
    
    def test_default_construction(self):
        from schemas.backtest import AggregateMetrics
        metrics = AggregateMetrics()
        assert metrics.total_return_pct == 0.0
        assert metrics.trades == 0
        assert metrics.sharpe is None  # Optional
    
    def test_from_dict(self):
        from schemas.backtest import AggregateMetrics
        raw = {
            "total_return_pct": 15.3,
            "sharpe": 1.45,
            "drawdown": -8.2,
            "trades": 142,
            "win_rate": 0.58,
            "unknown_metric": 999,
        }
        metrics = AggregateMetrics.model_validate(raw)
        assert metrics.total_return_pct == 15.3
        assert metrics.sharpe == 1.45
        assert metrics.drawdown == -8.2
        assert metrics.trades == 142
    
    def test_drawdown_constraint(self):
        """Drawdown must be <= 0."""
        from schemas.backtest import AggregateMetrics
        
        # Negative is fine
        metrics = AggregateMetrics(drawdown=-5.0)
        assert metrics.drawdown == -5.0
        
        # Zero is fine
        metrics = AggregateMetrics(drawdown=0.0)
        assert metrics.drawdown == 0.0
        
        # Positive is rejected
        with pytest.raises(ValidationError):
            AggregateMetrics(drawdown=5.0)
    
    def test_rate_constraints(self):
        """win_rate, active_rate must be [0, 1]."""
        from schemas.backtest import AggregateMetrics
        
        with pytest.raises(ValidationError):
            AggregateMetrics(win_rate=1.5)
        
        with pytest.raises(ValidationError):
            AggregateMetrics(active_rate=-0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Test BacktestResult
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestResult:
    """Tests for schemas.backtest.BacktestResult."""
    
    def test_construction_with_pandas(self):
        """Can hold pandas Series and DataFrame."""
        from schemas.backtest import BacktestResult, AggregateMetrics
        
        result = BacktestResult(
            metrics=AggregateMetrics(sharpe=1.2, trades=50),
            equity_curve=pd.Series([1.0, 1.01, 1.03, 0.99]),
            monthly_df=pd.DataFrame({"month": [1, 2], "return": [0.01, -0.02]}),
            model_type="logistic",
        )
        assert result.model_type == "logistic"
        assert len(result.equity_curve) == 4
    
    def test_to_serializable(self):
        """to_serializable() converts pandas types to JSON-safe types."""
        from schemas.backtest import BacktestResult, AggregateMetrics
        
        result = BacktestResult(
            metrics=AggregateMetrics(),
            equity_curve=pd.Series([1.0, 1.05]),
            monthly_df=pd.DataFrame({"a": [1, 2]}),
            model_type="cnn",
        )
        serialized = result.to_serializable()
        
        assert isinstance(serialized["equity_curve"], list)
        assert isinstance(serialized["monthly_df"], list)
        assert serialized["model_type"] == "cnn"
        assert isinstance(serialized["metrics"], dict)
    
    def test_from_pipeline_output(self):
        """from_pipeline_output() adapts raw dict to typed model."""
        from schemas.backtest import BacktestResult
        
        raw_metrics = {
            "total_return_pct": 12.5,
            "sharpe": 0.9,
            "trades": 80,
            "win_rate": 0.55,
            "drawdown": -3.2,
        }
        result = BacktestResult.from_pipeline_output(
            metrics_dict=raw_metrics,
            equity_curve=np.array([1.0, 1.05, 1.12]),
            monthly_df=pd.DataFrame(),
            model_type="xgboost",
        )
        assert result.metrics.total_return_pct == 12.5
        assert result.metrics.sharpe == 0.9
        assert result.model_type == "xgboost"


# ═══════════════════════════════════════════════════════════════════════════
# Test IndicatorWindows
# ═══════════════════════════════════════════════════════════════════════════

class TestIndicatorWindows:
    """Tests for schemas.features.IndicatorWindows."""
    
    def test_defaults(self):
        from schemas.features import IndicatorWindows
        windows = IndicatorWindows()
        assert windows.sma == 20
        assert windows.rsi == 14
        assert windows.bb_dev == 2.0
    
    def test_constraint_enforcement(self):
        from schemas.features import IndicatorWindows
        
        with pytest.raises(ValidationError):
            IndicatorWindows(sma=0)  # ge=1
        
        with pytest.raises(ValidationError):
            IndicatorWindows(rsi=1)  # ge=2
    
    def test_nested_in_features_config(self):
        """IndicatorWindows is validated when nested in FeaturesConfig."""
        from schemas.features import FeaturesConfig
        
        # Valid nested config
        config = FeaturesConfig(
            indicator_windows={"sma": 50, "rsi": 21}
        )
        assert config.indicator_windows.sma == 50
        
        # Invalid nested config (sma=-1)
        with pytest.raises(ValidationError):
            FeaturesConfig(
                indicator_windows={"sma": -1}
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test FeaturesConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestFeaturesConfig:
    """Tests for schemas.features.FeaturesConfig."""
    
    def test_defaults(self):
        from schemas.features import FeaturesConfig
        config = FeaturesConfig()
        assert config.use_triple_barrier is True
        assert config.fracdiff_d == 0.5
        assert config.confidence_threshold == 0.80
    
    def test_from_dict_with_nested(self):
        """Nested indicator_windows dict is parsed into IndicatorWindows model."""
        from schemas.features import FeaturesConfig
        
        raw = {
            "use_rsi": True,
            "indicator_windows": {
                "sma": 30,
                "ema": 25,
                "rsi": 21,
            },
            "unknown_key": "ignored",
        }
        config = FeaturesConfig.model_validate(raw)
        assert config.use_rsi is True
        assert config.indicator_windows.sma == 30
        assert config.indicator_windows.rsi == 21
    
    def test_from_class_defaults(self):
        """from_class_defaults() successfully parses CLASS_DEFAULTS dict."""
        from schemas.features import FeaturesConfig
        config = FeaturesConfig.from_class_defaults()
        assert config.use_triple_barrier is True
        assert config.tb_pt_mult == 2.0
        from schemas.features import IndicatorWindows
        assert isinstance(config.indicator_windows, IndicatorWindows)


# ═══════════════════════════════════════════════════════════════════════════
# Test CVConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestCVConfig:
    """Tests for schemas.features.CVConfig."""
    
    def test_defaults(self):
        from schemas.features import CVConfig
        config = CVConfig()
        assert config.cv_mode == "mini_block"
        assert config.cv_blocks == 5
        assert config.n_trials == 0
    
    def test_from_class_defaults(self):
        """from_class_defaults() successfully parses CLASS_DEFAULTS["cv"]."""
        from schemas.features import CVConfig
        config = CVConfig.from_class_defaults()
        assert config.cv_fold_aggregator == "ivw_sharpe_capped"
        assert config.eval_spread_pips == 0.8
    
    def test_constraint_enforcement(self):
        from schemas.features import CVConfig
        
        with pytest.raises(ValidationError):
            CVConfig(cv_blocks=1)  # ge=2
        
        with pytest.raises(ValidationError):
            CVConfig(cv_val_frac=0.0)  # ge=0.01


# ═══════════════════════════════════════════════════════════════════════════
# Test HPOConfigPayload
# ═══════════════════════════════════════════════════════════════════════════

class TestHPOConfigPayload:
    """Tests for schemas.hpo.HPOConfigPayload."""
    
    def test_default_construction(self):
        from schemas.hpo import HPOConfigPayload
        payload = HPOConfigPayload()
        assert payload.model_type == "logistic"
        assert payload.best_params == {}
        assert payload.n_trials == 0
    
    def test_from_dict(self):
        from schemas.hpo import HPOConfigPayload
        
        raw = {
            "model_type": "cnn",
            "best_params": {"lr": 0.001, "layers": 3},
            "best_value": 1.45,
            "n_trials": 50,
            "timestamp": "2026-04-14T12:00:00",
        }
        payload = HPOConfigPayload.model_validate(raw)
        assert payload.model_type == "cnn"
        assert payload.best_params["lr"] == 0.001
        assert payload.best_value == 1.45
    
    def test_json_roundtrip(self, tmp_path):
        """to_json_file() → from_json_file() is lossless."""
        from schemas.hpo import HPOConfigPayload
        
        original = HPOConfigPayload(
            model_type="lstm",
            best_params={"hidden_size": 64},
            best_value=0.95,
            n_trials=25,
        )
        path = str(tmp_path / "test_config.json")
        original.to_json_file(path)
        
        loaded = HPOConfigPayload.from_json_file(path)
        assert loaded.model_type == original.model_type
        assert loaded.best_params == original.best_params
        assert loaded.best_value == original.best_value
    
    def test_existing_hpo_files_loadable(self):
        """All existing hpo/*.json files can be loaded without errors."""
        import os
        from schemas.hpo import HPOConfigPayload
        
        hpo_dir = os.path.join(os.path.dirname(__file__), "..", "hpo")
        if not os.path.isdir(hpo_dir):
            pytest.skip("hpo/ directory not found")
        
        json_files = [f for f in os.listdir(hpo_dir) if f.endswith(".json")]
        if not json_files:
            pytest.skip("No HPO JSON files found")
        
        for fname in json_files:
            path = os.path.join(hpo_dir, fname)
            payload = HPOConfigPayload.from_json_file(path)
            assert payload.model_type in fname or payload.best_params is not None, \
                f"Failed to validate {fname}"


# ═══════════════════════════════════════════════════════════════════════════
# Test ParamImportance
# ═══════════════════════════════════════════════════════════════════════════

class TestParamImportance:
    """Tests for schemas.hpo.ParamImportance."""
    
    def test_valid(self):
        from schemas.hpo import ParamImportance
        pi = ParamImportance(param_name="learning_rate", importance=0.85)
        assert pi.param_name == "learning_rate"
        assert pi.importance == 0.85
    
    def test_empty_name_rejected(self):
        from schemas.hpo import ParamImportance
        with pytest.raises(ValidationError):
            ParamImportance(param_name="", importance=0.5)
    
    def test_importance_range(self):
        from schemas.hpo import ParamImportance
        with pytest.raises(ValidationError):
            ParamImportance(param_name="x", importance=1.5)


# ═══════════════════════════════════════════════════════════════════════════
# Test PydanticSettingsConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestPydanticSettingsConfig:
    """Tests for schemas.settings.PydanticSettingsConfig."""
    
    def test_defaults(self):
        from schemas.settings import PydanticSettingsConfig
        config = PydanticSettingsConfig()
        assert config.data_key == "EURUSD_H1"
        assert config.log_mode == "COMPACT"
        assert config.cv_debug is False
    
    def test_literal_constraint(self):
        from schemas.settings import PydanticSettingsConfig
        
        with pytest.raises(ValidationError):
            PydanticSettingsConfig(log_mode="invalid")
    
    def test_to_dict(self):
        from schemas.settings import PydanticSettingsConfig
        config = PydanticSettingsConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["data_key"] == "EURUSD_H1"


# ═══════════════════════════════════════════════════════════════════════════
# Test package-level imports
# ═══════════════════════════════════════════════════════════════════════════

class TestPackageImports:
    """Tests that schemas/__init__.py re-exports work correctly."""
    
    def test_all_models_importable(self):
        from schemas import (
            BacktestParams,
            AggregateMetrics,
            BacktestResult,
            IndicatorWindows,
            FeaturesConfig,
            CVConfig,
            HPOConfigPayload,
            ParamImportance,
            PydanticSettingsConfig,
        )
        # Just verify they're all classes
        assert all(isinstance(cls, type) for cls in [
            BacktestParams, AggregateMetrics, BacktestResult,
            IndicatorWindows, FeaturesConfig, CVConfig,
            HPOConfigPayload, ParamImportance, PydanticSettingsConfig,
        ])