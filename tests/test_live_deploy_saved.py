"""Tests for live trading bridge — deploy saved model instead of training fresh."""
import pytest
import os
import tempfile


def _make_dummy_snapshot(tmp_path):
    """Create a minimal snapshot that can be loaded for live trading tests."""
    from sklearn.linear_model import LogisticRegression
    import joblib
    import json
    import hashlib

    snap_dir = os.path.join(str(tmp_path), "logistic_live_test")
    os.makedirs(snap_dir, exist_ok=True)

    X = __import__("numpy").random.randn(100, 10)
    y = __import__("numpy").random.choice([-1, 0, 1], 100)
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    joblib.dump(model, os.path.join(snap_dir, "model.joblib"))

    metadata = {
        "model_type": "logistic",
        "features_config": {"model_type": "logistic", "lags": 10},
        "feature_names": [f"f{i}" for i in range(10)],
        "train_start": "2023-01-01",
        "train_end": "2025-12-31",
    }
    with open(os.path.join(snap_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    mf = {}
    for fname in ["model.joblib"]:
        p = os.path.join(snap_dir, fname)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            h.update(fh.read())
        mf[fname] = h.hexdigest()
    with open(os.path.join(snap_dir, "manifest.sha256"), "w") as f:
        for k, v in mf.items():
            f.write(f"{v}  {k}\n")

    return snap_dir


class TestLiveDeployWithSavedModel:
    """Tests for the live trading bridge."""

    def test_deploy_request_accepts_model_id(self):
        from api.routers.live import DeployRequest
        req = DeployRequest(
            pair="EURUSD",
            model="logistic",
            timeframe="M30",
            model_id="test-model-123",
        )
        assert req.model_id == "test-model-123"

    def test_deploy_request_optional_model_id(self):
        from api.routers.live import DeployRequest
        req = DeployRequest(pair="EURUSD", model="logistic", timeframe="M30")
        assert req.model_id is None

    def test_live_deploy_top_level_imports(self):
        """Verify the live module imports correctly."""
        from api.routers.live import (
            DeployRequest, SessionInfo, router,
            active_sessions, _run_backtest_for_model,
            _predict_signal, _signal_loop,
        )
        assert router is not None
        assert active_sessions is not None

    def test_snapshot_loadable_for_live(self, tmp_path):
        """Test that a snapshot can be loaded for live trading use."""
        snapshot_dir = _make_dummy_snapshot(tmp_path)

        from pipeline.model_persistence import (
            load_model_only, read_metadata, validate_snapshot,
        )

        ok, reason = validate_snapshot(snapshot_dir)
        assert ok, f"Snapshot failed validation: {reason}"

        model = load_model_only(snapshot_dir)
        assert model is not None
        assert hasattr(model, "predict_proba")

        meta = read_metadata(snapshot_dir)
        assert meta["model_type"] == "logistic"

    def test_model_predicts_for_live_path(self, tmp_path):
        """Test that a loaded model can predict on dummy data (live path)."""
        import numpy as np
        snapshot_dir = _make_dummy_snapshot(tmp_path)

        from pipeline.model_persistence import load_model_only
        model = load_model_only(snapshot_dir)

        X = np.random.randn(3, 10)
        proba = model.predict_proba(X)
        assert proba.shape == (3, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestLiveBackendCompatibility:
    """Verify backward compatibility of the existing endpoint."""

    def test_existing_backtest_for_model_exists(self):
        from api.routers.live import _run_backtest_for_model
        assert callable(_run_backtest_for_model)

    def test_signal_loop_top_level_accessible(self):
        from api.routers.live import _signal_loop
        assert callable(_signal_loop)
