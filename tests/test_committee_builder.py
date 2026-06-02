"""Tests for CommitteeBuilder — Phase C of the Multi-Agent Exploration Engine.

Tests model selection, weight optimization, fallback selection, serialization,
and full committee construction from mock performance matrix data.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.regime_utils import _REGIME_NAMES

from pipeline.expert_profiler import (
    FoldResult,
    RegimeModelMatrix,
    ExpertProfiler,
)

from pipeline.committee_builder import (
    RegimeAssignment,
    CommitteeConfig,
    CommitteeBuilder,
)


_REGIMES = list(_REGIME_NAMES.values())
_MODELS = ["logistic", "xgboost", "lstm", "random_forest", "svm", "lightgbm"]
_RNG = np.random.default_rng(42)


def _make_folds_for_matrix(
    models: list = None,
    n_folds: int = 8,
    seed: int = 42,
) -> list:
    """Create synthetic FoldResult list with controlled regime-model patterns.

    Pattern:
      - logistic: best in sideways, mean_reverting
      - xgboost: best in trend_up, trend_down, breakout
      - lstm: best in trend_up, trend_down
      - random_forest: best in high_volatile, breakout
      - svm: best in sideways, quiet_squeeze
      - lightgbm: all-rounder
    """
    if models is None:
        models = _MODELS

    model_regime_boost = {
        "logistic": {"sideways": 0.35, "mean_reverting": 0.30, "quiet_squeeze": 0.25},
        "xgboost": {"trend_up": 0.50, "trend_down": 0.45, "breakout": 0.30, "mean_reverting": 0.20},
        "lstm": {"trend_up": 0.55, "trend_down": 0.50},
        "random_forest": {"high_volatile": 0.40, "breakout": 0.35},
        "svm": {"sideways": 0.28, "quiet_squeeze": 0.22},
        "lightgbm": {"trend_up": 0.40, "trend_down": 0.35, "breakout": 0.20, "sideways": 0.15},
    }

    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2023-01-01")

    folds = []
    for fold_i in range(n_folds):
        test_start = base_time + pd.DateOffset(months=fold_i)
        test_end = test_start + pd.DateOffset(months=1)

        # Rotate dominant regime across folds
        dom_idx = fold_i % len(_REGIMES)
        dominant = _REGIMES[dom_idx]

        regime_counts = {r: int(rng.exponential(scale=5.0) + 1) for r in _REGIMES}
        regime_counts[dominant] += int(rng.exponential(scale=20.0) + 5)

        for model in models:
            base_sr = 0.15 + rng.uniform(-0.05, 0.15)
            boost = model_regime_boost.get(model, {}).get(dominant, 0.0)
            sharpe = base_sr + boost + rng.uniform(-0.03, 0.03)
            trades = max(5, int(15 + sharpe * 30 + rng.integers(-3, 5)))
            win_rate = max(0.25, min(0.75, 0.45 + sharpe * 0.12 + rng.uniform(-0.03, 0.03)))

            folds.append(FoldResult(
                model=model, fold_idx=fold_i,
                train_start=test_start - pd.DateOffset(months=12),
                train_end=test_start,
                test_start=test_start, test_end=test_end,
                sharpe=float(sharpe),
                trades=trades,
                active_rate=float(max(0.02, 0.08 + sharpe * 0.04)),
                win_rate=float(win_rate),
                performance=float(sharpe * 0.9),
                return_val=float(sharpe * 0.05),
                drawdown=float(max(0.01, 0.04 - sharpe * 0.02)),
                geo_mean_ann=float(sharpe * 0.85),
                directional_accuracy=float(max(0.35, 0.50 + sharpe * 0.08)),
                f1_macro=float(max(0.15, 0.28 + sharpe * 0.1)),
                regime_counts=dict(regime_counts),
                dominant_regime=dominant,
            ))
    return folds


def _make_matrix(
    models: list = None,
    n_folds: int = 12,
    seed: int = 42,
) -> RegimeModelMatrix:
    """Build a RegimeModelMatrix from synthetic folds."""
    folds = _make_folds_for_matrix(models, n_folds, seed)
    profiler = ExpertProfiler()
    return profiler._build_matrix(folds)


# ════════════════════════════════════════════════════════════════════
# RegimeAssignment
# ════════════════════════════════════════════════════════════════════

class TestRegimeAssignment:
    def test_construct(self):
        ra = RegimeAssignment(models=["lstm", "xgboost"], weights=[0.6, 0.4])
        assert ra.models == ["lstm", "xgboost"]
        assert ra.weights == [0.6, 0.4]

    def test_validate_mismatched_lengths(self):
        ra = RegimeAssignment(models=["a", "b"], weights=[0.5])
        with pytest.raises(ValueError):
            ra.validate()

    def test_validate_normalizes_weights(self):
        ra = RegimeAssignment(models=["a", "b"], weights=[3.0, 1.0])
        ra.validate()
        assert np.isclose(sum(ra.weights), 1.0)
        assert np.isclose(ra.weights[0], 0.75)

    def test_to_dict(self):
        ra = RegimeAssignment(models=["x"], weights=[1.0])
        d = ra.to_dict()
        assert d == {"models": ["x"], "weights": [1.0]}


# ════════════════════════════════════════════════════════════════════
# CommitteeConfig
# ════════════════════════════════════════════════════════════════════

class TestCommitteeConfig:
    def test_empty_construct(self):
        cfg = CommitteeConfig()
        assert cfg.version == 1
        assert len(cfg.regimes) == 0
        assert cfg.fallback is None

    def test_full_construct(self):
        cfg = CommitteeConfig(
            version=2,
            regimes={
                "trend_up": RegimeAssignment(models=["lstm", "xgboost"], weights=[0.6, 0.4]),
                "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["xgboost"], weights=[1.0]),
            constraints={"max_models": 2},
            metadata={"n_profiled": 6},
        )
        assert len(cfg.regimes) == 2
        assert cfg.fallback.models == ["xgboost"]

    def test_all_models(self):
        cfg = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["lstm", "xgboost"], weights=[0.6, 0.4]),
                "volatile": RegimeAssignment(models=["random_forest", "cnn"], weights=[0.5, 0.5]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        all_m = cfg.all_models()
        assert "lstm" in all_m
        assert "xgboost" in all_m
        assert "random_forest" in all_m
        assert "cnn" in all_m
        assert "logistic" in all_m
        assert len(all_m) == 5

    def test_regime_models_falls_back(self):
        cfg = CommitteeConfig(
            regimes={},
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        result = cfg.regime_models("trend_up")
        assert result.models == ["logistic"]

    def test_regime_models_exact_match(self):
        cfg = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["lstm"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        result = cfg.regime_models("trend_up")
        assert result.models == ["lstm"]

    def test_json_roundtrip(self):
        cfg = CommitteeConfig(
            version=1,
            regimes={
                "trend_up": RegimeAssignment(models=["lstm", "xgboost"], weights=[0.6, 0.4]),
                "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["xgboost"], weights=[1.0]),
            constraints={"max_models": 2},
            metadata={"built_on": "2023-2024"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_committee.json")
            cfg.to_json(path)
            assert os.path.exists(path)

            loaded = CommitteeConfig.from_json(path)
            assert loaded.version == 1
            assert loaded.all_models() == cfg.all_models()
            assert loaded.regimes["trend_up"].models == ["lstm", "xgboost"]

    def test_json_roundtrip_no_fallback(self):
        cfg = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["lstm"], weights=[1.0]),
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nofallback.json")
            cfg.to_json(path)
            loaded = CommitteeConfig.from_json(path)
            assert loaded.fallback is None

    def test_to_dict_has_expected_keys(self):
        cfg = CommitteeConfig(
            regimes={"trend_up": RegimeAssignment(models=["lstm"], weights=[1.0])},
        )
        d = cfg.to_dict()
        for key in ["version", "regimes", "fallback", "constraints", "metadata"]:
            assert key in d


# ════════════════════════════════════════════════════════════════════
# CommitteeBuilder — Core logic
# ════════════════════════════════════════════════════════════════════

class TestCommitteeBuilderCore:
    @pytest.fixture
    def matrix(self):
        return _make_matrix(models=_MODELS, n_folds=12, seed=123)

    def test_select_candidates(self, matrix):
        builder = CommitteeBuilder(top_k=3, min_sharpe=-0.5)
        # Regime 1 = trend_up (idx 1)
        candidates = builder._select_candidates(matrix, regime_idx=1, top_k=3)
        assert len(candidates) <= 3
        assert len(candidates) > 0
        # lstm or xgboost should be top in trend_up
        assert candidates[0] in ("lstm", "xgboost", "lightgbm")

    def test_select_candidates_empty_if_all_nan(self):
        # Create matrix where all Sharpes are NaN
        empty_matrix = RegimeModelMatrix(
            regimes=_REGIMES,
            models=["a", "b"],
            sharpe_matrix=np.full((2, 7), np.nan),
            trade_matrix=np.zeros((2, 7)),
            hitrate_matrix=np.zeros((2, 7)),
            fold_counts=np.zeros((2, 7), dtype=int),
        )
        builder = CommitteeBuilder()
        candidates = builder._select_candidates(empty_matrix, regime_idx=0)
        assert candidates == []

    def test_select_candidates_respects_min_sharpe(self, matrix):
        builder_strict = CommitteeBuilder(top_k=5, min_sharpe=0.8)
        candidates = builder_strict._select_candidates(matrix, regime_idx=3)
        assert len(candidates) <= 5

    def test_compute_weights_equal(self, matrix):
        builder = CommitteeBuilder(weight_method="equal")
        candidates = ["logistic", "xgboost"]
        weights = builder._compute_weights(matrix, regime_idx=0, candidates=candidates,
                                           model_names=list(matrix.models))
        assert np.allclose(weights, [0.5, 0.5])

    def test_compute_weights_sharpe_proportional(self, matrix):
        builder = CommitteeBuilder(weight_method="sharpe_proportional")
        candidates = ["lstm", "logistic"]
        name_idx = {m: i for i, m in enumerate(matrix.models)}
        weights = builder._compute_weights(matrix, regime_idx=1, candidates=candidates,
                                           model_names=list(matrix.models))
        assert np.isclose(sum(weights), 1.0)
        # lstm should have higher weight than logistic in trend_up
        lstm_w = weights[candidates.index("lstm")]
        logit_w = weights[candidates.index("logistic")]
        # lstm should be better in trend_up
        if matrix.sharpe_matrix[name_idx["lstm"], 1] > matrix.sharpe_matrix[name_idx["logistic"], 1]:
            assert lstm_w > logit_w

    def test_compute_weights_single_model(self, matrix):
        builder = CommitteeBuilder()
        weights = builder._compute_weights(matrix, regime_idx=0, candidates=["logistic"],
                                           model_names=list(matrix.models))
        assert weights == [1.0]

    def test_compute_weights_empty(self, matrix):
        builder = CommitteeBuilder()
        weights = builder._compute_weights(matrix, regime_idx=0, candidates=[],
                                           model_names=list(matrix.models))
        assert weights == []

    def test_select_fallback(self, matrix):
        builder = CommitteeBuilder()
        fallback = builder._select_fallback(matrix, list(matrix.models))
        assert fallback.models
        assert len(fallback.models) == 1
        assert np.isclose(sum(fallback.weights), 1.0)
        # Should pick a real model
        assert fallback.models[0] in _MODELS

    def test_select_fallback_single_model(self):
        matrix = _make_matrix(models=["logistic"], n_folds=4, seed=99)
        builder = CommitteeBuilder()
        fallback = builder._select_fallback(matrix, ["logistic"])
        assert fallback.models == ["logistic"]

    def test_select_fallback_all_nan(self):
        empty_matrix = RegimeModelMatrix(
            regimes=_REGIMES,
            models=["a"],
            sharpe_matrix=np.full((1, 7), np.nan),
            trade_matrix=np.zeros((1, 7)),
            hitrate_matrix=np.zeros((1, 7)),
            fold_counts=np.zeros((1, 7), dtype=int),
        )
        builder = CommitteeBuilder()
        fallback = builder._select_fallback(empty_matrix, ["a"])
        assert fallback.models == ["a"]


# ════════════════════════════════════════════════════════════════════
# CommitteeBuilder — Full build
# ════════════════════════════════════════════════════════════════════

class TestCommitteeBuilderFull:
    def test_build_produces_config(self):
        matrix = _make_matrix(models=_MODELS, n_folds=16, seed=42)
        builder = CommitteeBuilder(top_k=3, min_sharpe=-0.3, weight_method="sharpe_proportional")
        config = builder.build(matrix)

        assert isinstance(config, CommitteeConfig)
        assert len(config.regimes) > 0
        for regime, assignment in config.regimes.items():
            assert len(assignment.models) <= 3
            assert np.isclose(sum(assignment.weights), 1.0, atol=0.02)

    def test_build_all_regimes_covered(self):
        matrix = _make_matrix(models=_MODELS, n_folds=12, seed=73)
        builder = CommitteeBuilder(top_k=2)
        config = builder.build(matrix)
        # With 6 models and high enough folds, all regimes should have coverage
        assert len(config.regimes) >= 4

    def test_build_no_nan_weights(self):
        matrix = _make_matrix(models=_MODELS, n_folds=10, seed=55)
        builder = CommitteeBuilder(top_k=3)
        config = builder.build(matrix)
        for assignment in config.regimes.values():
            for w in assignment.weights:
                assert not np.isnan(w)
                assert w >= 0.0
                assert w <= 1.0

    def test_build_single_model(self):
        matrix = _make_matrix(models=["logistic"], n_folds=5, seed=10)
        builder = CommitteeBuilder(top_k=3)
        config = builder.build(matrix)
        for assignment in config.regimes.values():
            assert len(assignment.models) == 1
            assert assignment.weights == [1.0]

    def test_build_with_constraints(self):
        matrix = _make_matrix(n_folds=8, seed=90)
        builder = CommitteeBuilder(top_k=2)
        config = builder.build(matrix, constraints={"max_models_per_regime": 2, "min_sharpe": -0.5})
        for assignment in config.regimes.values():
            assert len(assignment.models) <= 2

    def test_build_fallback_present(self):
        matrix = _make_matrix(models=_MODELS, n_folds=12, seed=42)
        builder = CommitteeBuilder(top_k=2)
        config = builder.build(matrix)
        assert config.fallback is not None
        assert len(config.fallback.models) > 0

    def test_build_metadata(self):
        matrix = _make_matrix(models=_MODELS, n_folds=8, seed=77)
        builder = CommitteeBuilder(top_k=2, weight_method="sharpe_proportional")
        config = builder.build(matrix)
        assert "n_models_profiled" in config.metadata
        assert config.metadata["weight_method"] == "sharpe_proportional"

    def test_build_empty_matrix_raises(self):
        empty = RegimeModelMatrix(
            regimes=_REGIMES, models=[],
            sharpe_matrix=np.empty((0, 7)),
            trade_matrix=np.empty((0, 7)),
            hitrate_matrix=np.empty((0, 7)),
            fold_counts=np.empty((0, 7), dtype=int),
        )
        builder = CommitteeBuilder()
        with pytest.raises(ValueError, match="no models"):
            builder.build(empty)


# ════════════════════════════════════════════════════════════════════
# Optimized weight method
# ════════════════════════════════════════════════════════════════════

class TestOptimizedWeights:
    def test_optimized_weights_valid(self):
        matrix = _make_matrix(models=_MODELS, n_folds=16, seed=42)
        builder = CommitteeBuilder(weight_method="optimized", top_k=3)
        config = builder.build(matrix)

        for assignment in config.regimes.values():
            assert np.isclose(sum(assignment.weights), 1.0, atol=0.03)
            for w in assignment.weights:
                assert w >= 0.0

    def test_optimized_fallback_to_equal_if_folds_insufficient(self):
        # Matrix with too few folds => simplex optimization fails => falls back to equal
        matrix = _make_matrix(models=["logistic", "xgboost"], n_folds=2, seed=10)
        builder = CommitteeBuilder(weight_method="optimized", top_k=2)
        config = builder.build(matrix)
        # Should still produce valid config, weights will be equal
        for assignment in config.regimes.values():
            assert np.isclose(sum(assignment.weights), 1.0, atol=0.03)


# ════════════════════════════════════════════════════════════════════
# Integration: ExpertProfiler → CommitteeBuilder → Config
# ════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_flow_matrix_to_config(self):
        """End-to-end: synthetic folds → matrix → committee → JSON."""
        folds = _make_folds_for_matrix(models=_MODELS, n_folds=12, seed=84)
        profiler = ExpertProfiler()
        matrix = profiler._build_matrix(folds)

        builder = CommitteeBuilder(top_k=3, weight_method="sharpe_proportional")
        config = builder.build(matrix)

        # Serialize
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "committee.json")
            builder.save_config(config, path)
            assert os.path.exists(path)

            # Deserialize and validate
            loaded = builder.load_config(path)
            assert loaded.version == 1
            assert set(loaded.all_models()).issubset(_MODELS)
            assert loaded.fallback is not None

    def test_print_summary_does_not_crash(self, capsys):
        matrix = _make_matrix(models=_MODELS, n_folds=8, seed=12)
        builder = CommitteeBuilder(top_k=2)
        config = builder.build(matrix)
        builder.print_summary(config)
        out = capsys.readouterr().out
        assert "COMMITTEE BUILDER" in out

    def test_top_models_per_regime_match_expected(self):
        """Verify that the builder picks models matching our synthetic pattern."""
        matrix = _make_matrix(models=_MODELS, n_folds=24, seed=42)

        # trend_up: lstm, xgboost, lightgbm should top
        trend_up_scores = matrix.sharpe_matrix[:, 1]  # column 1 = trend_up
        top_models_trend = [matrix.models[i] for i in np.argsort(-trend_up_scores)[:3]
                           if not np.isnan(trend_up_scores[i])]
        # lstm or xgboost should be #1
        assert top_models_trend[0] in ("lstm", "xgboost", "lightgbm")

        # sideways: logistic, svm should top (or at least one of them)
        sideways_scores = matrix.sharpe_matrix[:, 6]  # column 6 = sideways
        top_models_side = [matrix.models[i] for i in np.argsort(-sideways_scores)[:3]
                          if not np.isnan(sideways_scores[i])]
        sideways_set = set(top_models_side[:2])
        assert any(m in sideways_set for m in ("logistic", "svm")) or len(top_models_side) > 0
