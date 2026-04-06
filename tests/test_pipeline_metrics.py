"""Test 4: Pipeline metrics functions produce correct results."""
import pytest
import numpy as np


def test_temperature_calibration_basic():
    """_fit_temperature_from_proba returns a positive float."""
    from pipeline.metrics import _fit_temperature_from_proba, _apply_temperature_to_proba
    np.random.seed(42)
    proba = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.7, 0.2, 0.1]], dtype=np.float32)
    y_true = np.array([2, 1, 0])
    T = _fit_temperature_from_proba(proba, y_true)
    assert isinstance(T, float)
    assert T > 0


def test_temperature_shape_preserved():
    """_apply_temperature_to_proba preserves input shape."""
    from pipeline.metrics import _apply_temperature_to_proba
    proba = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]], dtype=np.float32)
    result = _apply_temperature_to_proba(proba, T=1.5)
    assert result.shape == proba.shape


def test_temperature_sums_to_one():
    """After temperature scaling, probabilities must sum to 1."""
    from pipeline.metrics import _apply_temperature_to_proba
    proba = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]], dtype=np.float32)
    result = _apply_temperature_to_proba(proba, T=2.0)
    sums = result.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_temperature_clamps_low_T():
    """Temperature T must be clamped to >= 1e-3."""
    from pipeline.metrics import _apply_temperature_to_proba
    proba = np.array([[0.1, 0.9]], dtype=np.float32)
    result = _apply_temperature_to_proba(proba, T=-1.0)
    assert result.shape == (1, 2)
    assert np.isfinite(result).all()


def test_psr_basic():
    """_psr returns a probability in [0, 1]."""
    from pipeline.metrics import _psr
    result = _psr(sr=1.0, n_eff=100)
    assert 0.0 <= result <= 1.0


def test_psr_high_sharpe_high_probability():
    """High Sharpe ratio should give high PSR."""
    from pipeline.metrics import _psr
    high_sr = _psr(sr=3.0, n_eff=200)
    low_sr = _psr(sr=0.1, n_eff=200)
    assert high_sr > low_sr


def test_psr_low_n_eff_returns_low():
    """Very low n_eff should give low PSR."""
    from pipeline.metrics import _psr
    result = _psr(sr=1.0, n_eff=2)
    assert result < 0.99


def test_psr_nan_sharpe():
    """NaN Sharpe ratio should return 0.0."""
    from pipeline.metrics import _psr
    result = _psr(sr=float("nan"), n_eff=100)
    assert result == 0.0


def test_dsr_sign_positive():
    """_dsr_sign should be positive when sr > sr_max."""
    from pipeline.metrics import _dsr_sign
    result = _dsr_sign(sr=2.0, n_eff=100, sr_max=1.0)
    assert result > 0


def test_dsr_sign_negative():
    """_dsr_sign should be negative when sr < sr_max."""
    from pipeline.metrics import _dsr_sign
    result = _dsr_sign(sr=0.5, n_eff=100, sr_max=1.0)
    assert result < 0


def test_cv_status_is_ok():
    """_cv_status_is_ok recognizes valid status strings."""
    from pipeline.metrics import _cv_status_is_ok
    assert _cv_status_is_ok("🟢 OK") is True
    assert _cv_status_is_ok("OK") is True
    assert _cv_status_is_ok("FAIL") is False
    assert _cv_status_is_ok("") is False
    assert _cv_status_is_ok(None) is False


def test_empty_metrics_tuple():
    """_empty_metrics returns a tuple of NaN values."""
    from pipeline.metrics_tuples import _empty_metrics
    from utilsNoWFO import N_METRICS
    metrics = _empty_metrics(context="test")
    assert isinstance(metrics, tuple)
    assert len(metrics) == N_METRICS


def test_safe_metrics_return_valid():
    """_safe_metrics_return accepts a correctly-sized list."""
    from pipeline.metrics_tuples import _safe_metrics_return
    from utilsNoWFO import N_METRICS, ensure_metric_tuple
    raw = [float(i) for i in range(N_METRICS)]
    result = _safe_metrics_return(raw, context="test")
    assert isinstance(result, tuple)
    assert len(result) == N_METRICS


def test_cv_reliability_gate_low_trades():
    """CV reliability gate rejects when trades are too low."""
    from pipeline.metrics import _cv_reliability_gate
    fcfg = {
        "gating_mode": "bets_psr",
        "min_trades_per_block": 30,
        "min_independent_bets": 20,
        "psr_alpha": 0.05,
        "dsr_prune": True,
        "floor_cv_final": -6.0,
    }
    cfg = {"features": fcfg}
    ok, reason = _cv_reliability_gate(score=1.5, trades=5, avg_hold_bars=10, params={}, cfg=cfg)
    assert ok is False
    assert "trades" in reason.lower()