"""Automated tests for Sprint 16 new features.

Tests the key new capabilities without requiring a full backtest run:
  S16.1  — train_sharpe population, block bootstrap, gap computation
  S16.2  — signals/gate data, regime distribution
  S16.3  — prediction histogram, confusion matrix format
  S16.4  — summary generation
  S16.5  — SEARCH_SPACE changes (SVM gamma, RF max_depth)
  Tiers  — feature families, model-specific importance methods, VIF
  Fixes  — deep model gradient importance, DSR threshold
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import pytest


# =====================================================================
# S16.1 — Overfitting detection helpers
# =====================================================================

class TestS161Overfitting:
    """train_sharpe population, gap computation, block bootstrap."""

    def test_sharpe_gap_computation(self):
        """sharpe_gap_pct = (train - test) / max(|train|, 0.01) * 100"""
        train, test = 1.2, 0.9
        gap = (train - test) / max(abs(train), 0.01) * 100
        assert round(gap, 1) == 25.0, f"Expected 25.0, got {gap}"

        train, test = -0.5, 0.2
        gap = (train - test) / max(abs(train), 0.01) * 100
        assert round(gap, 1) == -140.0, f"Expected -140.0, got {gap}"

        train, test = 0.0, 0.0
        gap = (train - test) / max(abs(train), 0.01) * 100
        assert round(gap, 1) == 0.0, f"Expected 0.0, got {gap}"

    def test_cv_value_extraction(self):
        """_safe_get_cv_value should check __cv_value, __hpo_best_score, etc."""
        candidate = {"__cv_value": 1.5, "model_type": "xgboost"}
        for key in ("__cv_value", "__hpo_best_score", "__verify_best_score"):
            v = float(candidate.get(key, float("nan")))
            if math.isfinite(v):
                assert key == "__cv_value"
                assert v == 1.5
                break

    def test_block_bootstrap_imports(self):
        """Block bootstrap functions are importable."""
        from pipeline.metrics.overfitting import _block_bootstrap_ci, _optimal_block_length, _classify_risk
        vals = np.random.default_rng(42).normal(0.8, 0.3, 24)
        bl = _optimal_block_length(vals)
        assert bl >= 3
        lo, hi, mean = _block_bootstrap_ci(vals)
        assert lo < mean < hi

    def test_risk_classification(self):
        from pipeline.metrics.overfitting import _classify_risk
        assert _classify_risk(10) == ("low", "green")
        assert _classify_risk(30) == ("medium", "yellow")
        assert _classify_risk(60) == ("high", "red")

    def test_period_breakdown_includes_gaps(self):
        """compute_period_breakdown should include sharpe_gap_pct, return_gap_pct, cv_fold_sharpes."""
        from pipeline.metrics.overfitting import compute_period_breakdown
        records = [
            {"test_start": "2023-01-01", "test_end": "2023-02-01",
             "train_start": "2020-01-01", "train_end": "2022-12-31",
             "strategy_return": 0.02, "bh_return": 0.01,
             "sharpe": 1.2, "trades": 10, "train_sharpe": 1.5,
             "sharpe_gap_pct": 20.0, "return_gap_pct": 15.0, "cv_fold_sharpes": [1.1, 1.3, 1.2],
             "signals_raw": 20, "signals_passed_gate": 12,
             "pct_sideways": 0.3, "pct_trend": 0.6, "pct_volatile": 0.1},
        ]
        out = compute_period_breakdown(records)
        assert len(out) == 1
        assert out[0]["sharpe_gap_pct"] == 20.0
        assert out[0]["return_gap_pct"] == 15.0
        assert out[0]["cv_fold_sharpes"] == [1.1, 1.3, 1.2]


# =====================================================================
# S16.2 — Walk-forward transparency
# =====================================================================

class TestS162Transparency:
    """Signals/gate data, regime distribution."""

    def test_signal_gate_rate(self):
        signals_raw = 25
        signals_passed = 18
        gate_rate = signals_passed / signals_raw if signals_raw > 0 else 0
        assert round(gate_rate * 100) == 72
        assert gate_rate <= 1.0


# =====================================================================
# S16.3 — Training diagnostics
# =====================================================================

class TestS163Diagnostics:
    """Prediction histogram, confusion matrix format."""

    def test_prediction_histogram(self):
        from pipeline.metrics.diagnostics import compute_prediction_histogram
        data = [np.array([0.55, 0.72, 0.88, 0.91, 0.63, 0.77, 0.95, 0.51])]
        bins = compute_prediction_histogram(data, n_bins=10)
        assert len(bins) == 10
        total = sum(b.count for b in bins)
        assert total == 8

    def test_confusion_matrix_dict_format(self):
        """API expects dict with matrix + labels keys, not raw list."""
        cm = {"matrix": [[5, 1, 0], [2, 8, 1], [0, 1, 6]], "labels": ["Short", "Flat", "Long"]}
        assert "matrix" in cm
        assert "labels" in cm
        assert len(cm["matrix"]) == 3


# =====================================================================
# S16.4 — Plain-English summary generator
# =====================================================================

class TestS164Summary:
    """Backtest summary text generation."""

    def test_summary_generation(self):
        from pipeline.metrics.summary_generator import generate_summary
        metrics = {
            "model": "xgboost", "sharpe": 1.24, "win_rate": 0.58,
            "max_drawdown": -0.083, "total_return_pct": 0.142, "total_trades": 142,
            "overfitting": {"risk_level": "low", "overfit_score": 18,
                           "is_mean_sharpe": 1.35, "oos_mean_sharpe": 1.24, "train_oos_gap_pct": 8.9},
            "walkforward_periods": [
                {"test_sharpe": 0.5, "pct_sideways": 0.6, "pct_trend": 0.3, "pct_volatile": 0.1},
                {"test_sharpe": 2.1, "pct_sideways": 0.1, "pct_trend": 0.8, "pct_volatile": 0.1},
                {"test_sharpe": 1.5, "pct_sideways": 0.2, "pct_trend": 0.6, "pct_volatile": 0.2},
            ],
        }
        config = {"pair": "EURUSD", "timeframe": "H1", "start_date": "2023-01-01", "end_date": "2024-12-31"}
        text = generate_summary(metrics, config)
        assert isinstance(text, str)
        assert len(text) > 50
        assert "XGBoost" in text
        assert "EURUSD" in text
        assert "strong Sharpe" in text or "exceptional Sharpe" in text
        assert "low" in text or "moderate" in text or "high" in text

    def test_summary_empty_overfitting(self):
        from pipeline.metrics.summary_generator import generate_summary
        metrics = {"model": "logistic", "sharpe": 0.5, "win_rate": 0.5,
                   "total_trades": 5, "overfitting": None}
        text = generate_summary(metrics, {"pair": "EURUSD"})
        assert isinstance(text, str)

    def test_summary_low_trades_message(self):
        from pipeline.metrics.summary_generator import _sentence_overfitting
        msg = _sentence_overfitting("low", 18, 8.9, 1.35, 1.24, 3)
        assert "very few trades" in msg or "limited" in msg

    def test_summary_regime_sentence(self):
        from pipeline.metrics.summary_generator import _sentence_regimes
        periods = [
            {"test_sharpe": 2.1, "pct_sideways": 0.1, "pct_trend": 0.8, "pct_volatile": 0.1},
            {"test_sharpe": -0.3, "pct_sideways": 0.7, "pct_trend": 0.1, "pct_volatile": 0.2},
        ]
        s = _sentence_regimes(periods)
        assert s is not None
        assert "trend" in s or "sideways" in s


# =====================================================================
# S16.5 — Better defaults (search space changes)
# =====================================================================

class TestS165Defaults:
    """SEARCH_SPACE changes: SVM gamma cap, RF max_depth removal, etc."""

    def test_svm_gamma_is_categorical(self):
        from config import SEARCH_SPACE
        svm = SEARCH_SPACE.get("svm", {})
        gamma = svm.get("gamma", [])
        assert isinstance(gamma, list), "gamma should be a list (categorical)"
        assert 1e-4 in gamma
        assert 0.05 in gamma
        assert gamma == [1e-4, 1e-3, 1e-2, 0.05]

    def test_rf_max_depth_no_none(self):
        from config import SEARCH_SPACE
        rf = SEARCH_SPACE.get("random_forest", {})
        md = rf.get("max_depth", [])
        assert None not in md, "RF max_depth should not include None"

    def test_cv_prune_relax_default(self):
        from pipeline.metrics.metrics_tuples import CLASS_DEFAULTS
        cv = CLASS_DEFAULTS.get("cv", {})
        relax = cv.get("cv_prune_relax", None)
        assert relax is not None
        assert relax == 0.75, f"Expected 0.75, got {relax}"

    def test_calibrate_method_hpo_only_sigmoid(self):
        """In sampler.py, calibrate_method should only suggest ['sigmoid']."""
        import ast, inspect
        from pipeline.tuning import sampler
        src = inspect.getsource(sampler)
        # Check that "isotonic" appears only in comments/docstrings, not as HPO choice
        assert "sigmoid" in src
        # The HPO suggest_categorical should not include isotonic anymore
        lines = [l for l in src.split("\n") if "calibrate_method" in l and "suggest_categorical" in l]
        for line in lines:
            assert "isotonic" not in line, f"Found isotonic in: {line}"


# =====================================================================
# Feature Families taxonomy
# =====================================================================

class TestFeatureFamilies:
    """Feature family classification and budget enforcement."""

    def test_classify_feature(self):
        from pipeline.features.feature_utils import _classify_feature
        from pipeline.metrics.metrics_tuples import FEATURE_FAMILIES
        assert _classify_feature("sma_20", FEATURE_FAMILIES) == "trend"
        assert _classify_feature("macd_signal", FEATURE_FAMILIES) == "momentum"
        assert _classify_feature("atr_14", FEATURE_FAMILIES) == "volatility"
        assert _classify_feature("nonexistent_col", FEATURE_FAMILIES) == "other"

    def test_feature_families_defined(self):
        from pipeline.metrics.metrics_tuples import FEATURE_FAMILIES, MAX_FEATURES_PER_FAMILY
        assert "trend" in FEATURE_FAMILIES
        assert "momentum" in FEATURE_FAMILIES
        assert "rolling" in FEATURE_FAMILIES
        assert MAX_FEATURES_PER_FAMILY["trend"] == 8
        assert MAX_FEATURES_PER_FAMILY["momentum"] == 6

    def test_per_family_budget(self):
        from pipeline.features.feature_utils import _apply_per_family_budget, _classify_feature
        from pipeline.metrics.metrics_tuples import FEATURE_FAMILIES, MAX_FEATURES_PER_FAMILY
        feats = [f"sma_{i}" for i in range(1, 12)] + [f"macd_{i}" for i in range(1, 8)]
        X = pd.DataFrame(np.random.randn(50, len(feats)), columns=feats)
        y = pd.Series(np.random.choice([0, 1, 2], 50))
        kept = _apply_per_family_budget(feats, X, y)
        trend_count = sum(1 for f in kept if _classify_feature(f, FEATURE_FAMILIES) == "trend")
        mom_count = sum(1 for f in kept if _classify_feature(f, FEATURE_FAMILIES) == "momentum")
        assert trend_count <= MAX_FEATURES_PER_FAMILY["trend"]
        assert mom_count <= MAX_FEATURES_PER_FAMILY["momentum"]


# =====================================================================
# Model-Specific Importance Methods
# =====================================================================

class TestImportanceMethods:
    """get_importance_method returns correct method per model type."""

    def test_method_xgboost(self):
        from pipeline.metrics.diagnostics import get_importance_method
        m = get_importance_method("xgboost")
        assert m in ("shap", "gain")

    def test_method_logistic(self):
        from pipeline.metrics.diagnostics import get_importance_method
        assert get_importance_method("logistic") == "coefficients"

    def test_method_svm(self):
        from pipeline.metrics.diagnostics import get_importance_method
        m = get_importance_method("svm")
        assert m in ("permutation", "none")

    def test_method_cnn(self):
        from pipeline.metrics.diagnostics import get_importance_method
        assert get_importance_method("cnn") == "gradient"

    def test_method_lstm(self):
        from pipeline.metrics.diagnostics import get_importance_method
        assert get_importance_method("lstm") == "gradient"

    def test_method_transformer(self):
        from pipeline.metrics.diagnostics import get_importance_method
        assert get_importance_method("transformer") == "gradient"

    def test_classify_feature_families(self):
        from pipeline.metrics.diagnostics import classify_feature_families
        families = classify_feature_families(["sma_20", "macd_signal", "atr_14", "rsi_10", "bb_20"])
        assert families.get("trend", 0) >= 1
        assert families.get("momentum", 0) >= 1
        assert families.get("volatility", 0) >= 1


# =====================================================================
# SVM permutation importance configuration
# =====================================================================

class TestSVMPermutation:
    """SVM uses n_repeats=5 for permutation importance."""

    def test_svm_repeats_value(self):
        import ast, inspect
        import pipeline.metrics.diagnostics as diag
        src = inspect.getsource(diag)
        assert "n_repeats=5" in src


# =====================================================================
# VIF computation
# =====================================================================

class TestVIF:
    """Variance Inflation Factor."""

    def test_compute_vif(self):
        from pipeline.metrics.diagnostics import compute_vif
        X = np.random.randn(100, 5)
        vif = compute_vif(X)
        assert len(vif) == 5
        assert all(np.isfinite(v) for v in vif)

    def test_vif_low_for_random(self):
        from pipeline.metrics.diagnostics import compute_vif
        np.random.seed(42)
        X = np.random.randn(200, 4)
        vif = compute_vif(X)
        for v in vif:
            assert v < 3, f"VIF={v} should be low for independent features"


# =====================================================================
# LightGBM proxy
# =====================================================================

class TestLightGBMProxy:
    """LightGBM proxy pre-filter."""

    def test_proxy_import(self):
        from pipeline.models.lightgbm_proxy import LightGBMProxy
        proxy = LightGBMProxy(top_k=5)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.choice([0, 1, 2], 100))
        selected = proxy.select(X, y)
        assert isinstance(selected, list)
        assert len(selected) > 0

    def test_proxy_reduces_features(self):
        from pipeline.models.lightgbm_proxy import LightGBMProxy
        proxy = LightGBMProxy(top_k=3)
        X = pd.DataFrame(np.random.randn(200, 20), columns=[f"f{i}" for i in range(20)])
        y = pd.Series(np.random.choice([0, 1, 2], 200))
        selected = proxy.select(X, y)
        assert len(selected) <= 3


# =====================================================================
# DSR threshold
# =====================================================================

class TestDSRThreshold:
    """DSR minimum Sharpe threshold for significance."""

    def test_dsr_computation(self):
        from pipeline.metrics.overfitting import _compute_dsr_min_sharpe
        threshold = _compute_dsr_min_sharpe(50, 24, 60)
        assert threshold is not None
        # Paper-consistent MinSR: (E[max_N] + z_0.95)/sqrt(n) * sqrt(12).
        # With 50 trials and 24 OOS months this is ~2.7 (annualized), far
        # above the old lenient Sidak heuristic (~0.5).
        assert 1.0 < threshold < 5.0

    def test_dsr_more_trials_higher_threshold(self):
        from pipeline.metrics.overfitting import _compute_dsr_min_sharpe
        t1 = _compute_dsr_min_sharpe(10, 24, 60)
        t2 = _compute_dsr_min_sharpe(200, 24, 60)
        assert t2 > t1

    def test_dsr_more_periods_lower_threshold(self):
        from pipeline.metrics.overfitting import _compute_dsr_min_sharpe
        t1 = _compute_dsr_min_sharpe(50, 24, 60)
        t2 = _compute_dsr_min_sharpe(50, 96, 60)
        assert t2 < t1


# =====================================================================
# Study presets
# =====================================================================

class TestStudyPresets:
    """Placeholder — frontend preset constants tested via TypeScript/CI."""


# =====================================================================
# Walk-forward transparency panel
# =====================================================================

class TestWalkForwardPeriod:
    """Per-period breakdown test (full round-trip)."""

    def test_period_breakdown_structure(self):
        from pipeline.metrics.overfitting import compute_period_breakdown
        records = [
            {"test_start": "2023-01-01", "test_end": "2023-02-01",
             "train_start": "2022-06-01", "train_end": "2022-12-31",
             "strategy_return": 0.015, "bh_return": 0.008,
             "sharpe": 1.1, "trades": 12, "train_sharpe": 1.3,
             "sharpe_gap_pct": 15.4, "return_gap_pct": 46.7, "cv_fold_sharpes": [1.0, 1.2],
             "signals_raw": 30, "signals_passed_gate": 18,
             "pct_sideways": 0.4, "pct_trend": 0.4, "pct_volatile": 0.2},
        ]
        out = compute_period_breakdown(records)
        entry = out[0]
        for key in ("period_start", "period_end", "test_sharpe", "train_sharpe",
                    "strategy_return", "bh_return", "trades", "signals_raw",
                    "signals_passed_gate", "pct_sideways", "pct_trend", "pct_volatile",
                    "sharpe_gap_pct", "return_gap_pct", "cv_fold_sharpes"):
            assert key in entry, f"Missing key: {key}"
