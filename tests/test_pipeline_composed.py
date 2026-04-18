"""Test 2: MLBacktester composed class has all required methods."""
import pytest


# Key methods the composed class MUST have (from all 11 mixins)
REQUIRED_METHODS = [
    # CoreMixin
    "__init__",
    "__repr__",
    # DataMixin
    "get_data",
    # FeaturesMixin
    "prepare_features",
    "scale_features",
    # StrategyMixin
    "test_strategy",
    # EnsembleMixin
    "test_ensemble_strategy",
    # DQNMixin
    "test_dqn_strategy",
    # ModelFactoryMixin
    "get_model",
    # EvaluationMixin
    "evaluate_strategy",
    # RunMixin
    "run_strategy",
    # RealTradingMixin
    "real_trading_simulation",
]

# All 11 mixins that compose MLBacktester
REQUIRED_MIXINS = [
    "CoreMixin",
    "DataMixin",
    "FeaturesMixin",
    "DeepMixin",
    "StrategyMixin",
    "EnsembleMixin",
    "DQNMixin",
    "ModelFactoryMixin",
    "EvaluationMixin",
    "RunMixin",
    "RealTradingMixin",
]


def test_composed_class_exists(ml_backtester_class):
    """MLBacktester must be importable from composed.py."""
    assert ml_backtester_class is not None
    assert ml_backtester_class.__name__ == "MLBacktester"


def test_mro_includes_all_mixins(ml_backtester_class):
    """MRO must include all 11 mixin classes."""
    mro_names = [cls.__name__ for cls in ml_backtester_class.__mro__]
    for mixin in REQUIRED_MIXINS:
        assert mixin in mro_names, f"Missing mixin in MRO: {mixin}"


@pytest.mark.parametrize("method", REQUIRED_METHODS)
def test_has_required_method(ml_backtester_class, method):
    """MLBacktester must have each required method."""
    assert hasattr(ml_backtester_class, method), f"Missing method: {method}"
    assert callable(getattr(ml_backtester_class, method))


def test_public_method_count(ml_backtester_class):
    """MLBacktester should have a reasonable number of public methods."""
    public = [m for m in dir(ml_backtester_class) if not m.startswith("_")]
    assert len(public) >= 15, f"Only {len(public)} public methods, expected >= 15"


def test_mixin_modules_all_importable():
    """Each mixin module file must be importable individually."""
    import importlib
    MIXIN_FILE_MAP = {
        "CoreMixin": "core_mixin",
        "DataMixin": "data_mixin",
        "FeaturesMixin": "features_mixin",
        "DeepMixin": "deep_mixin",
        "StrategyMixin": "strategy_mixin",
        "EnsembleMixin": "ensemble_mixin",
        "DQNMixin": "dqn_mixin",
        "ModelFactoryMixin": "model_factory_mixin",
        "EvaluationMixin": "evaluation_mixin",
        "RunMixin": "run_mixin",
        "RealTradingMixin": "real_trading_mixin",
    }
    for mixin_cls, module_name in MIXIN_FILE_MAP.items():
        mod = importlib.import_module(f"pipeline.backtester.{module_name}")
        assert mod is not None
