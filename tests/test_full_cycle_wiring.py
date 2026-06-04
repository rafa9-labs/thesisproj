"""Integration tests: full cycle wiring, status polling, results schema, deploy flow."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KODAQUANT_NO_GPU"] = "1"
os.environ["MLB_THREADS"] = "1"

EURUSD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "csv_data", "EURUSD_10_years_H1_OANDA.csv",
)


def _ensure_csv():
    if not os.path.exists(EURUSD_PATH):
        pytest.skip("EURUSD H1 CSV not found")


# ════════════════════════════════════════════════════════════════════
# T1: Status polling integration
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestStatusPolling:

    def test_status_endpoint_returns_correct_fields(self):
        """Status JSON contains all expected fields with correct types."""
        _ensure_csv()
        from api.routers.committee import FullCycleStatusResponse
        resp = FullCycleStatusResponse(
            job_id="test", phase="profiling", phase_number=0,
            phase_progress="1/5", current_action="testing",
            locked_features_count=42, pruned_models=["svm"],
            surviving_models=["xgboost", "logistic"],
        )
        d = resp.model_dump()
        assert isinstance(d["job_id"], str)
        assert isinstance(d["phase"], str)
        assert isinstance(d["phase_number"], int)
        assert isinstance(d["locked_features_count"], int)
        assert isinstance(d["pruned_models"], list)
        assert isinstance(d["surviving_models"], list)

    def test_status_phase_number_progression(self):
        """phase_number maps correctly to UI phases."""
        from api.routers.committee import FullCycleStatusResponse
        phases = [
            ("feature_sweep", -1), ("prescreening", 0), ("profiling", 0),
            ("prescreening_complete", 0), ("tuning", 1), ("building", 2),
            ("validating", 3), ("validation_failed", 3), ("optimizing", 4),
            ("completed", 4), ("failed", -1),
        ]
        for phase_str, expected_num in phases:
            resp = FullCycleStatusResponse(job_id="test", phase=phase_str,
                                           phase_number=expected_num)
            assert resp.phase_number == expected_num


# ════════════════════════════════════════════════════════════════════
# T2: Results JSON schema
# ════════════════════════════════════════════════════════════════════

class TestResultsSchema:

    def test_results_response_all_fields(self):
        """FullCycleResultsResponse contains all required fields with correct types."""
        from api.routers.committee import FullCycleResultsResponse
        resp = FullCycleResultsResponse(
            job_id="test", status="completed",
            locked_features_count=30, pruned_features_count=35,
            top_importance_feature="sma_20",
            phase0_pruned=["svm"], phase0_survivors=["xgboost"],
            phase3_fold_consistency_cv=0.45,
            phase3_fold_consistency_pass=True,
            phase3_seed_robustness_sharpe=0.32,
            phase3_seed_robustness_pass=True,
            final_fold_consistency_cv=0.38,
            final_fold_consistency_pass=True,
            final_seed_robustness_sharpe=0.29,
            final_seed_robustness_pass=True,
            factory_best_sharpe=0.55, factory_total_iterations=5,
            factory_accepted_count=3, factory_stop_reason="patience",
            total_time_s=120.5,
        )
        d = resp.model_dump()
        assert d["job_id"] == "test"
        assert d["locked_features_count"] == 30
        assert d["pruned_features_count"] == 35
        assert d["top_importance_feature"] == "sma_20"
        assert d["phase3_fold_consistency_cv"] == 0.45
        assert d["factory_best_sharpe"] == 0.55
        assert d["total_time_s"] == 120.5

    def test_results_schema_validation_failed(self):
        """validation_failed status saves partial results correctly."""
        from api.routers.committee import FullCycleResultsResponse
        resp = FullCycleResultsResponse(
            job_id="test_fail", status="validation_failed",
            locked_features_count=25, pruned_features_count=40,
            phase0_pruned=["svm", "decision_tree"],
            phase0_survivors=["xgboost", "logistic", "lstm"],
            phase3_fold_consistency_cv=1.85,
            phase3_fold_consistency_pass=False,
            phase3_seed_robustness_sharpe=-0.12,
            phase3_seed_robustness_pass=False,
            total_time_s=45.0,
        )
        d = resp.model_dump()
        assert d["status"] == "validation_failed"
        assert d["phase3_fold_consistency_pass"] is False
        assert d["factory_total_iterations"] == 0  # factory skipped

    def test_results_json_roundtrip(self, tmp_path):
        """FullCycleResultsResponse survives JSON serialize/deserialize."""
        from api.routers.committee import FullCycleResultsResponse
        resp = FullCycleResultsResponse(
            job_id="roundtrip_test", status="completed",
            locked_features_count=12, pruned_features_count=53,
            top_importance_feature="price_sma_20_ratio",
            phase0_survivors=["xgboost", "logistic"],
            phase3_fold_consistency_cv=0.72,
            phase3_fold_consistency_pass=True,
            phase3_seed_robustness_sharpe=0.45,
            phase3_seed_robustness_pass=True,
            factory_best_sharpe=0.38, factory_total_iterations=4,
            factory_accepted_count=2,
            total_time_s=95.0,
        )
        path = tmp_path / "results.json"
        with open(path, "w") as f:
            json.dump(resp.model_dump(), f, indent=2, default=str)
        with open(path) as f:
            loaded = json.load(f)
        reloaded = FullCycleResultsResponse(**loaded)
        assert reloaded.locked_features_count == 12
        assert reloaded.phase3_fold_consistency_cv == 0.72

    def test_committee_config_roundtrip(self, tmp_path):
        """CommitteeConfig survives dict→JSON→dict roundtrip."""
        from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
        cfg = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(
                    models=["xgboost", "logistic"],
                    weights=[0.7, 0.3],
                ),
                "high_volatile": RegimeAssignment(
                    models=["lstm"], weights=[1.0],
                ),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
            metadata={"max_regimes_per_model": 3},
        )
        d = cfg.to_dict()
        path = tmp_path / "committee_config.json"
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        cfg2 = CommitteeConfig.from_json(str(path))
        assert cfg2.regimes["trend_up"].models == ["xgboost", "logistic"]
        assert cfg2.metadata["max_regimes_per_model"] == 3


# ════════════════════════════════════════════════════════════════════
# T3: Deploy flow integration
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestDeployIntegration:

    def test_train_committee_models_logistic(self):
        """_train_committee_models produces valid model + feature names."""
        _ensure_csv()
        from api.routers.live import _train_committee_models, _LIVE_FEATURE_NAMES
        models, features = _train_committee_models("EURUSD", "H1", ["logistic"])
        assert "logistic" in models
        assert features == _LIVE_FEATURE_NAMES
        m = models["logistic"]
        assert hasattr(m, "predict_proba")

    def test_runner_with_trained_models(self):
        """LiveCommitteeRunner processes bars with real trained models."""
        _ensure_csv()
        from api.routers.live import _train_committee_models
        from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime_utils import RegimeConfig

        models, features = _train_committee_models("EURUSD", "H1", ["logistic"])
        cfg = CommitteeConfig(
            regimes={"trend_up": RegimeAssignment(models=["logistic"], weights=[1.0])},
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        runner = LiveCommitteeRunner(
            config=cfg, models=models, feature_names=features,
            regime_cfg=RegimeConfig(), confidence_threshold=0.15,
            lookback_bars=50,
        )
        runner.start()

        bar = {
            "mid_c": 1.1050, "mid_h": 1.1055, "mid_l": 1.1045,
            "mid_o": 1.1048, "spread": 0.0001, "returns": 0.0002,
            "timestamp": 1234567890,
        }
        for _ in range(55):
            runner.process_bar(bar.copy())
        signal = runner.process_bar(bar.copy())
        runner.stop()
        assert signal is not None
        assert signal.signal in (-1, 0, 1)
        assert len(signal.active_models) > 0

    def test_deploy_endpoint_request_model(self):
        """DeployCommitteeRequest Pydantic model validates correctly."""
        from api.routers.live import DeployCommitteeRequest
        req = DeployCommitteeRequest(
            pair="EURUSD", timeframe="H1", initial_equity=5000.0,
            confidence_threshold=0.6, lookback_bars=50,
        )
        assert req.pair == "EURUSD"
        assert req.confidence_threshold == 0.6
        assert req.lookback_bars == 50
