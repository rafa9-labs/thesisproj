"""
Tests for Sprint 16 — Overfitting Detection, Walk-Forward Transparency, and Training Diagnostics.

Covers:
- pipeline/overfitting.py: OverfittingReport, bootstrap CI, composite score
- pipeline/diagnostics.py: feature importance, confusion matrix, confidence bands, histograms
- api/schemas/backtest.py: Pydantic models for OverfittingReport, TrainingDiagnostics
- api/routers/backtest.py: _parse_diagnostics helper
"""
import numpy as np
import pytest


class TestOverfittingEngine:
    def test_overfitting_report_low_risk(self):
        from pipeline.metrics.overfitting import compute_overfitting_report
        records = [
            {"sharpe": 1.2, "train_sharpe": 1.5, "strategy_return": 0.08, "trades": 50, "test_start": "2024-01-01", "test_end": "2024-02-01"},
            {"sharpe": 1.1, "train_sharpe": 1.4, "strategy_return": 0.06, "trades": 45, "test_start": "2024-02-01", "test_end": "2024-03-01"},
        ]
        report = compute_overfitting_report(records, "xgboost")
        assert report.overfit_score >= 0
        assert report.risk_level in ("low", "medium", "high")
        assert report.n_periods == 2
        assert report.sharpe_ci is not None

    def test_overfitting_report_empty_records(self):
        from pipeline.metrics.overfitting import compute_overfitting_report
        report = compute_overfitting_report([], "logistic")
        assert report.overfit_score == 0.0
        assert report.n_periods == 0

    def test_composite_score_bounds(self):
        from pipeline.metrics.overfitting import compute_overfitting_report
        records = [
            {"sharpe": 0.5, "train_sharpe": 2.5, "strategy_return": 0.01, "trades": 10, "test_start": "2024-01-01", "test_end": "2024-02-01"},
        ]
        report = compute_overfitting_report(records, "xgboost", hpo_best_value=2.0)
        assert 0 <= report.overfit_score <= 100

    def test_bootstrap_ci_shape(self):
        from pipeline.metrics.overfitting import compute_overfitting_report
        records = [
            {"sharpe": 1.0, "strategy_return": 0.05, "trades": 30, "test_start": "2024-01-01", "test_end": "2024-02-01"},
        ]
        report = compute_overfitting_report(records, "xgboost")
        assert report.sharpe_ci is not None
        assert isinstance(report.sharpe_ci, dict)
        assert "low" in report.sharpe_ci


class TestDiagnosticsEngine:
    def test_compute_feature_importance_xgboost(self):
        from pipeline.metrics.diagnostics import compute_feature_importance
        class FakeBooster:
            def get_score(self, importance_type="gain"):
                return {"f0": 5.0, "f1": 3.0, "f2": 1.0}
        class FakeXGB:
            def get_booster(self):
                return FakeBooster()
        result = compute_feature_importance(FakeXGB(), "xgboost", feature_names=["feat_a", "feat_b", "feat_c"])
        assert len(result) == 3
        assert result[0].feature == "feat_a"

    def test_compute_feature_importance_rf(self):
        from pipeline.metrics.diagnostics import compute_feature_importance
        class FakeRF:
            feature_importances_ = np.array([0.5, 0.3, 0.2])
        result = compute_feature_importance(FakeRF(), "random_forest", feature_names=["a", "b", "c"])
        assert len(result) == 3
        assert result[0].feature == "a"

    def test_compute_feature_importance_svm_empty(self):
        from pipeline.metrics.diagnostics import compute_feature_importance
        result = compute_feature_importance(None, "svm")
        assert result == []

    def test_compute_feature_importance_deep_empty(self):
        from pipeline.metrics.diagnostics import compute_feature_importance
        result = compute_feature_importance(None, "cnn")
        assert result == []

    def test_compute_feature_importance_ensemble_adaptive(self):
        from pipeline.metrics.diagnostics import compute_feature_importance
        class FakeAdaptive:
            rf = type("RF", (), {"feature_importances_": np.array([0.4, 0.6])})()
            xgb = None
        result = compute_feature_importance(FakeAdaptive(), "ensemble_adaptive_regime", feature_names=["x", "y"])
        assert len(result) == 2
        assert result[0].feature == "y"  # 0.6 > 0.4

    def test_prediction_histogram(self):
        from pipeline.metrics.diagnostics import compute_prediction_histogram
        conf_arrays = [np.array([0.55, 0.65, 0.75, 0.85, 0.95])]
        result = compute_prediction_histogram(conf_arrays, n_bins=5)
        assert len(result) == 5
        assert all(b.count >= 0 for b in result)
        assert result[0].bin_start == 0.5

    def test_prediction_histogram_empty(self):
        from pipeline.metrics.diagnostics import compute_prediction_histogram
        result = compute_prediction_histogram([])
        assert result == []

    def test_aggregate_confusion_matrices(self):
        from pipeline.metrics.diagnostics import aggregate_confusion_matrices
        cm1 = np.array([[10, 2, 1], [3, 15, 2], [1, 1, 8]])
        cm2 = np.array([[8, 1, 0], [2, 12, 1], [0, 2, 10]])
        result = aggregate_confusion_matrices([cm1, cm2])
        assert result.matrix[0][0] == 18
        assert result.labels == ["Short", "Flat", "Long"]

    def test_aggregate_confusion_matrices_empty(self):
        from pipeline.metrics.diagnostics import aggregate_confusion_matrices
        result = aggregate_confusion_matrices([])
        assert result.matrix == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def test_confidence_bands(self):
        from pipeline.metrics.diagnostics import compute_confidence_bands
        conf = [np.array([0.55, 0.65, 0.75, 0.85, 0.95])]
        outcomes = [np.array([1, 0, 1, 1, 0])]
        returns = [np.array([0.001, -0.002, 0.003, 0.004, -0.001])]
        result = compute_confidence_bands(conf, outcomes, returns)
        assert len(result) > 0
        assert all(b.count >= 0 for b in result)
        assert all(0 <= b.accuracy <= 1 for b in result)

    def test_confidence_bands_empty(self):
        from pipeline.metrics.diagnostics import compute_confidence_bands
        result = compute_confidence_bands([], [], [])
        assert result == []


class TestPydanticSchemas:
    def test_overfitting_report_schema(self):
        from api.schemas.backtest import OverfittingReport, OverfittingCI
        ci = OverfittingCI(low=0.5, high=1.5, mean=1.0)
        report = OverfittingReport(overfit_score=25.0, risk_level="medium", risk_color="yellow", sharpe_ci=ci)
        assert report.overfit_score == 25.0
        assert report.sharpe_ci.low == 0.5

    def test_training_diagnostics_schema(self):
        from api.schemas.backtest import TrainingDiagnostics, FeatureImportanceEntry, ConfusionMatrixData, ConfidenceBand
        diag = TrainingDiagnostics(
            feature_importance=[FeatureImportanceEntry(feature="rsi_14", importance=0.35)],
            confusion_matrix=ConfusionMatrixData(matrix=[[10, 2, 1], [3, 15, 2], [1, 1, 8]]),
            confidence_bands=[ConfidenceBand(band_min=0.5, band_max=0.6, count=50, accuracy=0.55, mean_return=0.002)],
        )
        assert len(diag.feature_importance) == 1
        assert diag.confusion_matrix.matrix[0][0] == 10

    def test_backtest_result_metrics_diagnostics_field(self):
        from api.schemas.backtest import BacktestResultMetrics, TrainingDiagnostics
        m = BacktestResultMetrics(model="xgboost", diagnostics=TrainingDiagnostics())
        assert m.diagnostics is not None
        assert m.diagnostics.feature_importance is None


class TestParseDiagnostics:
    def test_parse_diagnostics_none(self):
        from api.routers.backtest import _parse_diagnostics
        assert _parse_diagnostics(None) is None

    def test_parse_diagnostics_full(self):
        from api.routers.backtest import _parse_diagnostics
        raw = {
            "feature_importance": [{"feature": "rsi", "importance": 0.5}],
            "prediction_histogram": [{"bin_start": 0.5, "bin_end": 0.6, "bin_center": 0.55, "count": 10}],
            "confusion_matrix": {"matrix": [[5, 1, 0], [2, 8, 1], [0, 1, 6]], "labels": ["Short", "Flat", "Long"]},
            "confidence_bands": [{"band_min": 0.5, "band_max": 0.6, "count": 10, "accuracy": 0.55, "mean_return": 0.002}],
        }
        result = _parse_diagnostics(raw)
        assert result is not None
        assert len(result.feature_importance) == 1
        assert result.feature_importance[0].feature == "rsi"
        assert result.confusion_matrix is not None
        assert result.confidence_bands is not None
        assert len(result.confidence_bands) == 1

    def test_parse_diagnostics_empty_dict(self):
        from api.routers.backtest import _parse_diagnostics
        result = _parse_diagnostics({})
        assert result is not None
        assert result.feature_importance is None


class TestMacroPrecF1FromConfusion:
    def test_returns_3_tuple(self):
        from pipeline.metrics.metrics_eval import _macro_prec_f1_from_confusion
        y_true = [1, 0, -1, 1, 0]
        y_pred = [1, 0, 0, 1, -1]
        prec, f1, cm = _macro_prec_f1_from_confusion(y_true, y_pred)
        assert isinstance(prec, float)
        assert isinstance(f1, float)
        assert cm is not None
        assert cm.shape == (3, 3)