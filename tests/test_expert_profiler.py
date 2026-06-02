"""Tests for ExpertProfiler — Phase B of the Multi-Agent Exploration Engine.

Tests the performance matrix construction, regime tagging, statistical
tests, and serialization. Uses synthetic fold data — no live WFO needed.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.expert_profiler import (
    FoldResult,
    RegimeModelMatrix,
    ExpertProfileResult,
    ExpertProfiler,
)

from pipeline.regime_utils import _REGIME_NAMES


_RNG = np.random.default_rng(42)


def _make_fold_results(
    models: list = None,
    n_folds: int = 6,
    seed: int = 42,
) -> list:
    """Generate synthetic fold results with known regime patterns.

    Creates folds where specific models are designed to outperform in
    specific regimes (for testing the matrix logic).
    """
    if models is None:
        models = ["logistic", "xgboost", "lstm", "random_forest"]

    rng = np.random.default_rng(seed)
    regimes = list(_REGIME_NAMES.values())
    base_time = pd.Timestamp("2023-01-01")

    folds = []
    for fold_i in range(n_folds):
        test_start = base_time + pd.DateOffset(months=fold_i)
        test_end = test_start + pd.DateOffset(months=1)

        # Random dominant regime for this fold
        dominant = regimes[rng.integers(0, len(regimes))]

        # Regime distribution (weighted toward dominant)
        regime_counts = {}
        for r in regimes:
            count = (
                max(1, int(rng.exponential(scale=10.0)))
                if r == dominant
                else max(0, int(rng.exponential(scale=3.0)))
            )
            regime_counts[r] = count

        for model in models:
            # Model performance varies by regime:
            #   logistic → best in sideways/mean_reverting
            #   xgboost → best in trend_up/trend_down
            #   lstm → best in trend_up/trend_down
            #   random_forest → best in high_volatile/breakout

            base_sharpe = 0.2 + rng.uniform(-0.1, 0.3)

            if model == "logistic" and dominant in ("sideways", "mean_reverting", "quiet_squeeze"):
                base_sharpe += 0.3
            elif model in ("xgboost", "lstm") and dominant in ("trend_up", "trend_down"):
                base_sharpe += 0.4
            elif model == "random_forest" and dominant in ("high_volatile", "breakout"):
                base_sharpe += 0.3

            base_sharpe += rng.uniform(-0.05, 0.05)

            folds.append(FoldResult(
                model=model,
                fold_idx=fold_i,
                train_start=test_start - pd.DateOffset(months=12),
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                sharpe=float(base_sharpe),
                trades=int(max(1, 20 + base_sharpe * 50 + rng.integers(-5, 5))),
                active_rate=float(max(0.01, 0.10 + base_sharpe * 0.05 + rng.uniform(-0.02, 0.02))),
                win_rate=float(max(0.20, min(0.80, 0.45 + base_sharpe * 0.15 + rng.uniform(-0.05, 0.05)))),
                performance=float(base_sharpe + rng.uniform(-0.05, 0.05)),
                return_val=float(base_sharpe * 0.1),
                drawdown=float(max(0.01, 0.05 - base_sharpe * 0.02)),
                geo_mean_ann=float(base_sharpe * 0.9),
                directional_accuracy=float(max(0.3, min(0.9, 0.5 + base_sharpe * 0.1))),
                f1_macro=float(max(0.1, 0.3 + base_sharpe * 0.15)),
                regime_counts=dict(regime_counts),
                dominant_regime=dominant,
            ))

    return folds


# ════════════════════════════════════════════════════════════════════
# FoldResult
# ════════════════════════════════════════════════════════════════════

class TestFoldResult:
    def test_construct(self):
        f = FoldResult(
            model="xgboost", fold_idx=0,
            train_start="2023-01-01", train_end="2023-12-31",
            test_start="2024-01-01", test_end="2024-02-01",
            sharpe=0.45, trades=30, active_rate=0.15, win_rate=0.52,
            performance=0.42, return_val=0.05, drawdown=0.03, geo_mean_ann=0.40,
            directional_accuracy=0.55, f1_macro=0.35,
        )
        assert f.model == "xgboost"
        assert f.sharpe == 0.45
        assert f.trades == 30
        assert f.regime_counts == {}
        assert f.dominant_regime == ""

    def test_with_regime(self):
        f = FoldResult(
            model="lstm", fold_idx=1,
            train_start="2023-02-01", train_end="2024-01-31",
            test_start="2024-02-01", test_end="2024-03-01",
            sharpe=0.62, trades=45, active_rate=0.2, win_rate=0.58,
            performance=0.60, return_val=0.08, drawdown=0.02, geo_mean_ann=0.55,
            directional_accuracy=0.62, f1_macro=0.40,
            regime_counts={"trend_up": 80, "sideways": 20, "volatile": 10},
            dominant_regime="trend_up",
        )
        assert f.dominant_regime == "trend_up"
        assert f.regime_counts["trend_up"] == 80


# ════════════════════════════════════════════════════════════════════
# RegimeModelMatrix
# ════════════════════════════════════════════════════════════════════

class TestRegimeModelMatrix:
    def test_construct_empty(self):
        m = RegimeModelMatrix()
        assert len(m.regimes) == 0
        assert m.sharpe_matrix.size == 0

    def test_construct_with_data(self):
        models = ["logistic", "xgboost"]
        regimes = list(_REGIME_NAMES.values())
        sharpe = np.array([[0.2, 0.4, 0.1, 0.3, 0.5, 0.05, 0.15],
                           [0.5, 0.6, 0.2, 0.1, 0.3, 0.1, 0.08]])
        m = RegimeModelMatrix(
            regimes=regimes,
            models=models,
            sharpe_matrix=sharpe,
            trade_matrix=np.ones((2, 7)) * 20,
            hitrate_matrix=np.ones((2, 7)) * 0.5,
            fold_counts=np.ones((2, 7), dtype=int) * 3,
        )
        assert m.sharpe_matrix.shape == (2, 7)
        assert m.sharpe_matrix[0, 1] == 0.4
        assert m.sharpe_matrix[1, 0] == 0.5

    def test_top_model_per_regime(self):
        models = ["a", "b", "c"]
        regimes = list(_REGIME_NAMES.values())
        # model 'a' is best in regime 0, 'b' in 1, 'c' in 2, tie in others
        sharpe = np.array([
            [0.5, 0.2, 0.1, 0.3, 0.2, 0.1, 0.0],
            [0.3, 0.6, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.1, 0.5, 0.1, 0.3, 0.1, 0.2],
        ])
        m = RegimeModelMatrix(
            regimes=regimes, models=models, sharpe_matrix=sharpe,
            trade_matrix=np.ones((3, 7)) * 10,
            hitrate_matrix=np.ones((3, 7)) * 0.5,
            fold_counts=np.ones((3, 7), dtype=int) * 2,
        )

        top = m.top_model_per_regime(top_k=2)
        assert len(top) == 7
        assert top[regimes[0]][0][0] == "a"  # a is best in regime 0
        assert top[regimes[1]][0][0] == "b"  # b is best in regime 1
        assert top[regimes[2]][0][0] == "c"  # c is best in regime 2

    def test_top_model_handles_nan(self):
        models = ["x", "y"]
        regimes = list(_REGIME_NAMES.values())
        sharpe = np.full((2, 7), np.nan)
        sharpe[0, 0] = 0.5
        sharpe[1, 0] = 0.3

        m = RegimeModelMatrix(
            regimes=regimes, models=models, sharpe_matrix=sharpe,
            trade_matrix=np.zeros((2, 7)),
            hitrate_matrix=np.zeros((2, 7)),
            fold_counts=np.zeros((2, 7), dtype=int),
        )

        top = m.top_model_per_regime()
        assert len(top[regimes[0]]) == 2  # both have valid values
        assert top[regimes[0]][0][0] == "x"
        assert top[regimes[1]] == []  # all NaN → empty

    def test_to_dict(self):
        m = RegimeModelMatrix(
            regimes=["sideways", "trend_up"],
            models=["logistic"],
            sharpe_matrix=np.array([[0.2, 0.4]]),
            trade_matrix=np.array([[10, 20]]),
            hitrate_matrix=np.array([[0.5, 0.6]]),
            fold_counts=np.array([[3, 3]]),
        )
        d = m.to_dict()
        assert d["regimes"] == ["sideways", "trend_up"]
        assert d["sharpe"] == [[0.2, 0.4]]


# ════════════════════════════════════════════════════════════════════
# ExpertProfiler._build_matrix
# ════════════════════════════════════════════════════════════════════

class TestBuildMatrix:
    def test_build_from_synthetic_folds(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic", "xgboost", "lstm", "random_forest"], n_folds=8)
        matrix = profiler._build_matrix(folds)

        assert len(matrix.models) == 4
        assert len(matrix.regimes) == 7
        assert matrix.sharpe_matrix.shape == (4, 7)
        assert matrix.trade_matrix.shape == (4, 7)
        assert not np.all(np.isnan(matrix.sharpe_matrix))

    def test_single_model(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic"], n_folds=3)
        matrix = profiler._build_matrix(folds)
        assert len(matrix.models) == 1
        assert matrix.sharpe_matrix.shape == (1, 7)

    def test_nan_sharpe_folds(self):
        """Folds with NaN Sharpe should be excluded from weights but not crash."""
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic"], n_folds=5)
        # Corrupt two folds
        folds[1].sharpe = np.nan
        folds[3].sharpe = np.nan
        matrix = profiler._build_matrix(folds)
        assert not np.all(np.isnan(matrix.sharpe_matrix))

    def test_all_nan_returns_nan_matrix(self):
        profiler = ExpertProfiler()
        folds = [
            FoldResult(model="logistic", fold_idx=i, sharpe=np.nan, trades=0, active_rate=0.0,
                       win_rate=np.nan, performance=np.nan, return_val=np.nan, drawdown=np.nan,
                       geo_mean_ann=np.nan, directional_accuracy=np.nan, f1_macro=np.nan,
                       train_start="2023-01-01", train_end="2023-06-01",
                       test_start="2023-06-01", test_end="2023-07-01",
                       regime_counts={"sideways": 100})
            for i in range(3)
        ]
        matrix = profiler._build_matrix(folds)
        assert np.all(np.isnan(matrix.sharpe_matrix))


# ════════════════════════════════════════════════════════════════════
# Signifiance tests
# ════════════════════════════════════════════════════════════════════

class TestSignificanceTests:
    def test_runs_without_crash(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic", "xgboost", "lstm"], n_folds=12, seed=123)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)
        assert len(sig) == 7
        for regime in _REGIME_NAMES.values():
            assert regime in sig

    def test_top_models_present(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic", "xgboost"], n_folds=10, seed=42)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)
        # At least one regime should have results
        any_has = any(len(info.get("top_models", [])) > 0 for info in sig.values())
        assert any_has

    def test_single_model_no_pairs(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic"], n_folds=3, seed=42)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)
        for info in sig.values():
            assert info["significant_pairs"] == []

    def test_insufficient_folds_no_error(self):
        profiler = ExpertProfiler()
        # Only 2 folds per model — not enough for t-test
        folds = _make_fold_results(models=["logistic", "xgboost"], n_folds=2, seed=42)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)
        for info in sig.values():
            assert len(info.get("top_models", [])) >= 0
            # should not crash


# ════════════════════════════════════════════════════════════════════
# Serialization
# ════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_save_load_roundtrip(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic", "xgboost"], n_folds=4, seed=42)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)

        result = ExpertProfileResult(
            models_run=["logistic", "xgboost"],
            total_folds=8,
            matrix=matrix,
            significance=sig,
            execution_time_seconds=1.5,
            warnings=["test warning"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_matrix.json")
            profiler.save_matrix(result, path)
            assert os.path.exists(path)

            loaded = ExpertProfiler.load_matrix(path)
            assert loaded.regimes == matrix.regimes
            assert loaded.models == matrix.models
            assert np.allclose(loaded.sharpe_matrix, matrix.sharpe_matrix, equal_nan=True)

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            ExpertProfiler.load_matrix("/tmp/nonexistent_matrix_xyz.json")


# ════════════════════════════════════════════════════════════════════
# Regime distribution fallback
# ════════════════════════════════════════════════════════════════════

class TestRegimeFallback:
    def test_assigns_regime_counts(self):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic"], n_folds=3)
        # Reset regime counts to force fallback
        for f in folds:
            f.regime_counts = {}
            f.dominant_regime = ""

        profiler._attach_regime_from_fallback(folds)

        for f in folds:
            assert len(f.regime_counts) == 7
            assert sum(f.regime_counts.values()) > 0
            assert f.dominant_regime != ""
            assert f.dominant_regime in _REGIME_NAMES.values()

    def test_handles_nan_sharpe(self):
        profiler = ExpertProfiler()
        fold = FoldResult(
            model="test", fold_idx=0, sharpe=np.nan, trades=5, active_rate=0.05,
            win_rate=np.nan, performance=np.nan, return_val=np.nan, drawdown=np.nan,
            geo_mean_ann=np.nan, directional_accuracy=np.nan, f1_macro=np.nan,
            train_start="2023-01-01", train_end="2023-12-31",
            test_start="2024-01-01", test_end="2024-02-01",
        )
        profiler._attach_regime_from_fallback([fold])
        assert fold.dominant_regime != ""
        assert sum(fold.regime_counts.values()) > 0


# ════════════════════════════════════════════════════════════════════
# Print summary (coverage)
# ════════════════════════════════════════════════════════════════════

class TestPrintSummary:
    def test_does_not_crash(self, capsys):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic", "xgboost"], n_folds=4)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)

        result = ExpertProfileResult(
            models_run=["logistic", "xgboost"],
            total_folds=8,
            matrix=matrix,
            significance=sig,
            execution_time_seconds=2.3,
        )
        profiler.print_summary(result)
        out = capsys.readouterr().out
        assert "EXPERT PROFILER" in out

    def test_with_warnings(self, capsys):
        profiler = ExpertProfiler()
        folds = _make_fold_results(models=["logistic"], n_folds=2)
        matrix = profiler._build_matrix(folds)
        sig = profiler._run_significance_tests(folds, matrix)

        result = ExpertProfileResult(
            models_run=["logistic"],
            total_folds=2,
            matrix=matrix,
            significance=sig,
            execution_time_seconds=0.5,
            warnings=["warning 1", "warning 2"],
        )
        profiler.print_summary(result)
        out = capsys.readouterr().out
        assert "Warnings: 2" in out
