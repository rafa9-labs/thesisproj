"""Test 1: All pipeline modules import cleanly without errors."""
import pytest


STANDALONE_MODULES = [
    "pipeline._imports",
    "pipeline.runtime",
    "pipeline.standalone_utils",
    "pipeline.memory_utils",
    "pipeline.dqn_config",
    "pipeline.hpo_persistence",
    "pipeline.metrics",
    "pipeline.metrics_tuples",
    "pipeline.workers",
]

MIXIN_MODULES = [
    "pipeline.backtester.core_mixin",
    "pipeline.backtester.data_mixin",
    "pipeline.backtester.features_mixin",
    "pipeline.backtester.deep_mixin",
    "pipeline.backtester.strategy_mixin",
    "pipeline.backtester.ensemble_mixin",
    "pipeline.backtester.dqn_mixin",
    "pipeline.backtester.model_factory_mixin",
    "pipeline.backtester.evaluation_mixin",
    "pipeline.backtester.run_mixin",
    "pipeline.backtester.real_trading_mixin",
]

TUNING_MODULES = [
    "pipeline.tuning.helpers",
    "pipeline.tuning.sampler",
    "pipeline.tuning.objective",
    "pipeline.tuning.refit",
    "pipeline.tuning.runner",
]


@pytest.mark.parametrize("mod_name", STANDALONE_MODULES, ids=lambda x: x.split(".")[-1])
def test_standalone_module_imports(mod_name):
    """Each standalone pipeline module must import without error."""
    import importlib
    mod = importlib.import_module(mod_name)
    assert mod is not None


@pytest.mark.parametrize("mod_name", MIXIN_MODULES, ids=lambda x: x.split(".")[-1])
def test_mixin_module_imports(mod_name):
    """Each mixin module must import without error."""
    import importlib
    mod = importlib.import_module(mod_name)
    assert mod is not None


@pytest.mark.parametrize("mod_name", TUNING_MODULES, ids=lambda x: x.split(".")[-1])
def test_tuning_module_imports(mod_name):
    """Each tuning submodule must import without error."""
    import importlib
    mod = importlib.import_module(mod_name)
    assert mod is not None


def test_pipeline_tuning_exposes_expected_symbols():
    """pipeline.tuning must expose expected tuning symbols."""
    from pipeline.tuning import run_optuna_tuning, optuna_objective
    assert callable(run_optuna_tuning)
    assert callable(optuna_objective)


def test_utilsnowfo_imports():
    """utilsNoWFO must expose critical symbols."""
    import utilsNoWFO
    must_have = [
        "log_print", "compute_full_evaluation_metrics",
        "build_features_from_params", "fracdiff",
        "triple_barrier_labels", "sanitize_proba",
        "compute_metrics", "hac_std",
    ]
    for sym in must_have:
        assert hasattr(utilsNoWFO, sym), f"utilsNoWFO missing: {sym}"