"""Integration test: backtest → save → activate → predict → compare.

Runs a mini backtest, saves the model snapshot, validates it,
activates it, then tests the prediction and comparison endpoints.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings
from copy import deepcopy

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, r"C:\Users\rafa\ML_Trading\thesisproj")

for _m in ["config", "pipeline.metrics_tuples",
           "pipeline.backtester.composed"]:
    sys.modules.pop(_m, None)

START = "2017-06-01 00:00:00"
END = "2018-01-01 00:00:00"
MONTHS = 1


def _run_mini_backtest(model_type="logistic"):
    """Run a minimal backtest and return the MLBacktester instance + metrics."""
    from pipeline.metrics_tuples import CLASS_DEFAULTS
    from pipeline.backtester.composed import MLBacktester

    cfg = deepcopy(CLASS_DEFAULTS["features"])
    cfg.update(deepcopy(CLASS_DEFAULTS["cv"]))
    cfg.update({
        "model_type": model_type, "rep": 1, "trading_costs": False,
        "seed": 42, "period_unit": "months", "cv_blocks": 2,
        "slice_cache_enabled": True, "use_triple_barrier": True,
        "lag_depth": 1, "roll_windows": [5, 10, 30],
        "n_trials": 3, "n_startup_trials": 1,
        "use_cached_global_hpo": False,
        "use_bbands": False, "use_stoch": False, "use_sar": False,
        "use_squeeze_breakout": False, "use_squeeze_expansion": False,
        "use_atr_channel_breakout": False, "use_ext_atr_low_adx": False,
        "use_reentry_mom": False, "use_slope_diff": False,
        "use_news": False, "llm_sentiment_enabled": False,
    })

    bt = MLBacktester(
        symbol="EURUSD", start=START, end=END,
        trading_costs=False, features_config=cfg,
    )
    df_sim = bt.real_trading_simulation(deepcopy(cfg), models_to_test=[model_type], months=MONTHS)
    return bt, df_sim, cfg


class TestFullPipeline:
    """End-to-end: backtest -> snapshot -> activate -> predict -> compare."""

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_full_lifecycle(self, tmp_path):
        from pipeline.model_persistence import (
            save_snapshot, load_snapshot, read_metadata,
            validate_snapshot, DEPLOY_ROOT,
        )
        from pipeline.model_registry_disk import (
            register_snapshot, get_all_deployed,
            activate_model, deactivate_model, delete_model,
        )

        # Redirect deploy root to temp dir
        orig_root = DEPLOY_ROOT
        import pipeline.model_persistence as mp
        mp.DEPLOY_ROOT = str(tmp_path)
        import pipeline.model_registry_disk as mrd
        os.makedirs(str(tmp_path), exist_ok=True)

        try:
            # ── Phase 1: Run backtest ──
            bt, df_sim, cfg = _run_mini_backtest("logistic")

            assert df_sim is not None, "Backtest should produce results"
            assert not df_sim.empty, "Backtest should have data"

            # Extract metrics
            model_obj = getattr(bt, "model", None)
            assert model_obj is not None, "Should have trained model"
            cov_thr = getattr(bt, "_coverage_conf_thr", None)
            feat_names = getattr(bt, "_diagnostics_feature_names", [])
            feat_cfg = getattr(bt, "features_config", {})

            shrp = float(df_sim["sharpe"].mean()) if "sharpe" in df_sim.columns else 0.0
            trd = int(df_sim["trades"].sum()) if "trades" in df_sim.columns else 0
            metric_row = {"model": "logistic", "n_trials": 1, "sharpe": shrp,
                         "win_rate": 0.5, "total_return_pct": 1.0,
                         "max_drawdown": -5.0, "total_trades": trd}

            # ── Phase 2: Save snapshot ──
            snap_dir = save_snapshot(
                model=model_obj,
                model_type="logistic",
                best_params=feat_cfg,
                coverage_conf_thr=float(cov_thr) if cov_thr is not None else None,
                feature_names=list(feat_names) if feat_names else None,
                features_config=feat_cfg,
                calibrate_method="sigmoid",
                train_start=START[:10], train_end=END[:10],
                seed=42, parent_job_id="test-parent-001",
                metrics=metric_row,
            )
            assert os.path.isdir(snap_dir), f"Snapshot dir missing: {snap_dir}"
            assert os.path.isfile(os.path.join(snap_dir, "model.joblib"))
            assert os.path.isfile(os.path.join(snap_dir, "metadata.json"))
            assert os.path.isfile(os.path.join(snap_dir, "manifest.sha256"))

            ok, reason = validate_snapshot(snap_dir)
            assert ok, f"Snapshot invalid: {reason}"

            # ── Phase 3: Load and verify ──
            loaded = load_snapshot(snap_dir)
            assert "model" in loaded
            assert loaded["model"] is not None
            meta = read_metadata(snap_dir)
            assert meta["model_type"] == "logistic"
            assert meta["parent_job_id"] == "test-parent-001"

            # Verify model produces output on dummy data
            X = np.random.RandomState(7).randn(5, 3)
            try:
                proba = loaded["model"].predict_proba(X)
                assert proba.shape[0] == 5
                assert np.all(proba >= -0.01) and np.all(proba <= 1.01)
            except Exception:
                preds = loaded["model"].predict(X)
                assert preds.shape[0] == 5

            # ── Phase 4: Activate model ──
            db_path = os.path.join(str(tmp_path), "test.db")
            mid = register_snapshot(snap_dir, db_path)
            assert mid, "Register should return model ID"

            rows = get_all_deployed(db_path)
            assert len(rows) >= 1
            assert rows[0]["model_type"] == "logistic"
            assert rows[0]["status"] == "inactive"

            ok = activate_model(mid, db_path)
            assert ok, "Activation should succeed"

            rows = get_all_deployed(db_path)
            assert rows[0]["status"] == "active"

            # Verify active pointer
            active_id = mp.get_active_model_id("logistic")
            assert active_id == mid, f"Active pointer mismatch: {active_id} != {mid}"

            # ── Phase 5: Predict with deployed model ──
            # Test predict-with-data using registered model
            pred_path = mp.get_active_model_id("logistic")
            assert pred_path is not None

            from pipeline.model_persistence import load_model_only
            pred_model = load_model_only(snap_dir)
            X_pred = np.random.RandomState(42).randn(3, 3)
            proba_pred = pred_model.predict_proba(X_pred)
            assert proba_pred.shape == (3, 3) or proba_pred.shape[0] == 3

            # ── Phase 6: Cleanup ──
            ok, reason = delete_model(mid, db_path)
            assert ok, f"Delete failed: {reason}"
            rows = get_all_deployed(db_path)
            assert len(rows) == 0
            assert not os.path.isdir(snap_dir), "Snapshot dir should be deleted"

            bt.free(release_data=True)
            del bt

        finally:
            mp.DEPLOY_ROOT = orig_root
            mrd.DEPLOY_ROOT = str(tmp_path)

    def test_api_predict_endpoint(self):
        """Verify the predict API endpoint returns correct structure."""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        response = client.post("/api/v1/models/active/predict",
                              json={"pair": "EURUSD", "timeframe": "H1"})
        assert response.status_code in (200, 200), f"Status: {response.status_code}"
        data = response.json()
        assert "status" in data
        # Either "ready" if active model exists, or "no_active_model" if not
        assert data["status"] in ("ready", "no_active_model")

    def test_api_compare_endpoint(self):
        """Verify the compare API endpoint returns correct structure."""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        response = client.get("/api/v1/models/active/compare")
        assert response.status_code == 200
        data = response.json()
        assert "active_models" in data
        assert isinstance(data["active_models"], list)

    def test_api_deployed_list(self):
        """Verify deployed models list endpoint."""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        response = client.get("/api/v1/models/deployed")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_api_model_catalog(self):
        """Verify meta_ensemble appears in the models catalog."""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        model_names = [m["name"] for m in data.get("models", [])]
        assert "meta_ensemble" in model_names, f"meta_ensemble missing from: {model_names}"


class TestRoundtripPrediction:
    """Model saved after backtest can make predictions identical to original."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_prediction_fidelity_after_save_load(self):
        from pipeline.model_persistence import save_snapshot, load_snapshot
        import pipeline.model_persistence as mp

        with tempfile.TemporaryDirectory() as td:
            mp.DEPLOY_ROOT = td
            try:
                bt, df_sim, cfg = _run_mini_backtest("logistic")
                model_obj = getattr(bt, "model", None)
                if model_obj is None:
                    bt.free(release_data=True)
                    pytest.skip("No model produced by backtest")

                snap_dir = save_snapshot(
                    model=model_obj, model_type="logistic",
                    best_params={}, feature_names=["a", "b", "c"],
                    metrics={"sharpe": 0.5},
                )
                loaded = load_snapshot(snap_dir)

                X = np.random.RandomState(99).randn(10, 3)
                orig_preds = model_obj.predict(X)
                loaded_preds = loaded["model"].predict(X)
                assert np.array_equal(orig_preds, loaded_preds), \
                    "Predictions differ after save/load roundtrip"

                bt.free(release_data=True)
                del bt
            finally:
                mp.DEPLOY_ROOT = "deployed_models"
