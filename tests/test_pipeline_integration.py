"""Test 6: Cross-module integration — symbols flow correctly."""
import pytest
import numpy as np


def test_imports_expose_standalone_utils():
    """pipeline._imports re-exports standalone_utils symbols."""
    import pipeline._imports as pi
    assert hasattr(pi, "_norm_class_counts")
    assert hasattr(pi, "print_block_summary")


def test_imports_expose_memory_utils():
    """pipeline._imports re-exports memory_utils symbols."""
    import pipeline._imports as pi
    assert hasattr(pi, "_hard_free")
    assert hasattr(pi, "_apply_low_ram_overrides")


def test_imports_expose_dqn_config():
    """pipeline._imports re-exports DQN config symbols."""
    import pipeline._imports as pi
    assert hasattr(pi, "_load_default_dqn_cfg")
    assert hasattr(pi, "_coerce_dqn_cfg")


def test_imports_expose_metrics():
    """pipeline._imports re-exports metric functions."""
    import pipeline._imports as pi
    assert hasattr(pi, "_psr")
    assert hasattr(pi, "_dsr_sign")
    assert hasattr(pi, "_apply_temperature_to_proba")


def test_imports_expose_hpo_persistence():
    """pipeline._imports re-exports HPO persistence."""
    import pipeline._imports as pi
    assert hasattr(pi, "save_hpo_config_to_disk")
    assert hasattr(pi, "load_hpo_config_from_disk")


def test_imports_expose_metrics_tuples():
    """pipeline._imports re-exports metric tuple helpers."""
    import pipeline._imports as pi
    assert hasattr(pi, "_safe_metrics_return")
    assert hasattr(pi, "_empty_metrics")


def test_imports_expose_runtime():
    """pipeline._imports re-exports runtime knobs."""
    import pipeline._imports as pi
    assert hasattr(pi, "SAFE_CORES")
    assert hasattr(pi, "CPU_TOTAL")


def test_imports_expose_utilsnowfo():
    """pipeline._imports re-exports critical utilsNoWFO symbols."""
    import pipeline._imports as pi
    assert hasattr(pi, "log_print")
    assert hasattr(pi, "build_features_from_params")
    assert hasattr(pi, "compute_full_evaluation_metrics")
    assert hasattr(pi, "triple_barrier_labels")
    assert hasattr(pi, "sanitize_proba")


def test_lazy_tf_not_loaded_initially():
    """Lazy TF proxy should report 'not yet loaded' before use."""
    from pipeline._imports import tf
    # Just check it's a _LazyModule (not real TF)
    from pipeline._imports import _LazyModule
    assert isinstance(tf, _LazyModule)


def test_lazy_xgb_not_loaded_initially():
    """Lazy XGBoost proxy should not be loaded before use."""
    from pipeline._imports import xgb, _LazyModule
    assert isinstance(xgb, _LazyModule)


def test_lazy_optuna_not_loaded_initially():
    """Lazy Optuna proxy should not be loaded before use."""
    from pipeline._imports import optuna, _LazyModule
    assert isinstance(optuna, _LazyModule)


def test_tuning_package_exports():
    """pipeline.tuning exports the 4 key symbols."""
    from pipeline.tuning import (
        run_optuna_tuning,
        optuna_objective,
        sample_param_set,
        final_refit_if_deep,
    )
    assert callable(run_optuna_tuning)
    assert callable(optuna_objective)
    assert callable(sample_param_set)
    assert callable(final_refit_if_deep)


def test_backtester_package_imports():
    """pipeline.backtester exposes __init__ and composed."""
    import pipeline.backtester
    assert hasattr(pipeline.backtester, "composed")


def test_utilsnowfo_dead_code_removed():
    """Removed symbols should no longer exist in utilsNoWFO."""
    import utilsNoWFO
    dead = [
        "stationary_bootstrap", "spa_pvalue_single",
        "plot_full_cumulative_growth", "direction_safe_penalty",
        "save_underwater_curve_group",
    ]
    for sym in dead:
        assert not hasattr(utilsNoWFO, sym), f"Dead code not removed: {sym}"


def test_utilsnowfo_critical_symbols_present():
    """All critical utilsNoWFO symbols must still be available."""
    import utilsNoWFO
    critical = [
        "log_print", "compute_full_evaluation_metrics",
        "build_features_from_params", "fracdiff",
        "triple_barrier_labels", "month_dir_path",
        "ensure_model_dirs", "sanitize_proba",
        "compute_metrics", "hac_std",
        "probabilistic_sharpe_ratio", "set_paper_style",
        "_compute_drawdown", "_format_run_stamp",
        "realized_vol", "compute_drawdown_curve",
    ]
    for sym in critical:
        assert hasattr(utilsNoWFO, sym), f"Missing critical symbol: {sym}"


def test_mixin_cross_imports():
    """Each mixin can import pipeline._imports without circular errors."""
    import pipeline.backtester.core_mixin
    import pipeline.backtester.data_mixin
    import pipeline.backtester.features_mixin
    import pipeline.backtester.strategy_mixin
    # If we got here, no circular import errors
    assert True


def test_end_to_end_metric_pipeline():
    """Full metric pipeline: temp scaling -> PSR -> DSR works end-to-end."""
    from pipeline.metrics.metrics import (
        _apply_temperature_to_proba,
        _fit_temperature_from_proba,
        _psr, _dsr_sign,
    )
    np.random.seed(42)
    proba = np.random.dirichlet([1, 1, 1], size=100).astype(np.float32)
    y_true = np.random.randint(0, 3, size=100)

    # Calibrate temperature
    T = _fit_temperature_from_proba(proba, y_true)
    assert T > 0

    # Apply temperature
    calibrated = _apply_temperature_to_proba(proba, T)
    assert calibrated.shape == proba.shape
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-4)

    # Compute PSR and DSR
    sr = 1.5
    psr = _psr(sr=sr, n_eff=100)
    dsr = _dsr_sign(sr=sr, n_eff=100, sr_max=0.0)
    assert 0.0 <= psr <= 1.0
    assert dsr > 0