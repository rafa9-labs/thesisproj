"""Phase A tests: Model Snapshot System."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, r"C:\Users\rafa\ML_Trading\thesisproj")


@pytest.fixture
def snapshot_dir():
    """Create a temporary directory for test snapshots."""
    d = tempfile.mkdtemp(prefix="test_snapshot_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def patch_deploy_root(snapshot_dir):
    """Redirect DEPLOY_ROOT to temp dir for all tests."""
    import pipeline.models.model_persistence as mp
    orig = mp.DEPLOY_ROOT
    mp.DEPLOY_ROOT = snapshot_dir
    yield
    mp.DEPLOY_ROOT = orig


class TestSnapshotSaveLoad:
    """Core snapshot save/load roundtrip."""

    def test_save_load_roundtrip(self, snapshot_dir):
        """Train logistic, save snapshot, load, predict — identical output."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.random.RandomState(42).randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="lbfgs"))])
        model.fit(X, y)

        from pipeline.models.model_persistence import save_snapshot, load_snapshot, read_metadata

        d = save_snapshot(
            model=model,
            model_type="logistic",
            best_params={"C": 1.0},
            feature_names=[f"feat_{i}" for i in range(10)],
            train_start="2020-01-01",
            train_end="2025-01-01",
            seed=42,
            metrics={"sharpe": 0.8, "win_rate": 0.55, "total_trades": 100},
        )

        assert os.path.isdir(d)
        assert os.path.isfile(os.path.join(d, "model.joblib"))
        assert os.path.isfile(os.path.join(d, "metadata.json"))
        assert os.path.isfile(os.path.join(d, "manifest.sha256"))

        loaded = load_snapshot(d)
        assert "model" in loaded
        assert "metadata" in loaded

        orig_preds = model.predict(X)
        loaded_preds = loaded["model"].predict(X)
        assert np.array_equal(orig_preds, loaded_preds)

    def test_metadata_completeness(self, snapshot_dir):
        """Verify all required metadata fields are present."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.RandomState(1).randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)

        from pipeline.models.model_persistence import save_snapshot, read_metadata

        d = save_snapshot(
            model=model,
            model_type="logistic",
            best_params={"C": 0.5},
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            coverage_conf_thr=0.55,
            calibrate_method="sigmoid",
            input_shape=(5,),
            train_start="2020-01-01",
            train_end="2025-01-01",
            seed=42,
            pip_freeze="numpy==1.26.0\nscikit-learn==1.5.0\n",
            parent_job_id="parent-123",
            metrics={"sharpe": 0.72, "win_rate": 0.54, "total_return_pct": 15.0, "max_drawdown": -10.0, "total_trades": 80},
        )

        meta = read_metadata(d)
        required = {
            "schema_version", "model_type", "best_params", "coverage_conf_thr",
            "calibrate_method", "input_shape", "feature_names", "features_config",
            "train_start", "train_end", "seed", "created_at_utc", "parent_job_id",
            "pip_freeze", "metrics",
        }
        missing = required - set(meta.keys())
        assert not missing, f"Missing metadata fields: {missing}"

        assert meta["schema_version"] == 1
        assert meta["model_type"] == "logistic"
        assert meta["parent_job_id"] == "parent-123"
        assert meta["pip_freeze"] == "numpy==1.26.0\nscikit-learn==1.5.0\n"
        assert meta["metrics"]["sharpe"] == 0.72

    def test_missing_scaler_graceful(self, snapshot_dir):
        """Tree-based model (no scaler) saves/loads without error."""
        from sklearn.tree import DecisionTreeClassifier

        X = np.random.RandomState(3).randn(100, 4)
        y = (X[:, -1] > 0).astype(int)
        model = DecisionTreeClassifier().fit(X, y)

        from pipeline.models.model_persistence import save_snapshot, load_snapshot

        d = save_snapshot(
            model=model,
            model_type="decision_tree",
            best_params={"max_depth": 5},
            feature_names=["a", "b", "c", "d"],
        )
        loaded = load_snapshot(d)
        assert loaded["scaler"] is None
        assert loaded["imputer"] is None
        assert loaded["calibration"] is None
        assert np.array_equal(model.predict(X), loaded["model"].predict(X))


class TestManifestValidation:
    """Manifest tamper detection."""

    def test_valid_manifest_passes(self, snapshot_dir):
        from sklearn.linear_model import LogisticRegression
        from pipeline.models.model_persistence import save_snapshot, validate_snapshot

        X = np.random.RandomState(1).randn(20, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        d = save_snapshot(model=model, model_type="logistic", best_params={}, feature_names=["a", "b", "c"])
        ok, reason = validate_snapshot(d)
        assert ok, f"Validation failed: {reason}"

    def test_corrupt_file_detected(self, snapshot_dir):
        from sklearn.linear_model import LogisticRegression
        from pipeline.models.model_persistence import save_snapshot, validate_snapshot

        X = np.random.RandomState(2).randn(20, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        d = save_snapshot(model=model, model_type="logistic", best_params={}, feature_names=["a", "b", "c"])

        with open(os.path.join(d, "model.joblib"), "ab") as f:
            f.write(b"\x00\x01\x02")

        ok, reason = validate_snapshot(d)
        assert not ok
        assert "Manifest checksum mismatch" in reason


class TestExportImport:
    """Export and import .koda file roundtrip."""

    def test_export_import_roundtrip(self, snapshot_dir):
        from sklearn.linear_model import LogisticRegression
        from pipeline.models.model_persistence import save_snapshot, export_snapshot, import_snapshot, load_snapshot, read_metadata

        X = np.random.RandomState(5).randn(50, 4)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)

        d = save_snapshot(
            model=model,
            model_type="logistic",
            best_params={"C": 0.8},
            feature_names=["f1", "f2", "f3", "f4"],
        )
        koda_path = export_snapshot(d)
        assert os.path.isfile(koda_path), f"Exported .koda missing: {koda_path}"

        imported = import_snapshot(koda_path)
        assert os.path.isdir(imported)

        meta = read_metadata(imported)
        assert meta.get("model_type") == "logistic"

        loaded = load_snapshot(imported)
        assert np.array_equal(model.predict(X), loaded["model"].predict(X))


class TestActivePointer:
    """File-based active model pointer — one global active model."""

    def test_active_pointer_crud(self, snapshot_dir):
        from pipeline.models.model_persistence import (
            get_active_model_id, set_active_model_id, clear_active_model_id,
        )

        assert get_active_model_id() is None

        set_active_model_id("logistic", "snap-001")
        assert get_active_model_id() == "snap-001"
        assert get_active_model_id("logistic") == "snap-001"

        set_active_model_id("logistic", "snap-002")
        assert get_active_model_id() == "snap-002"
        assert get_active_model_id("logistic") == "snap-002"

        clear_active_model_id()
        assert get_active_model_id() is None
        assert get_active_model_id("logistic") is None

    def test_active_pointer_global_singleton(self, snapshot_dir):
        """Setting a new active model replaces the previous one globally."""
        from pipeline.models.model_persistence import (
            get_active_model_id, set_active_model_id,
        )

        set_active_model_id("logistic", "snap-L")
        assert get_active_model_id() == "snap-L"
        assert get_active_model_id("logistic") == "snap-L"

        # Setting a different type replaces the previous global active
        set_active_model_id("xgboost", "snap-X")
        assert get_active_model_id() == "snap-X"
        assert get_active_model_id("xgboost") == "snap-X"
        assert get_active_model_id("logistic") is None

        set_active_model_id("logistic", "snap-L2")
        assert get_active_model_id() == "snap-L2"
        assert get_active_model_id("logistic") == "snap-L2"
        assert get_active_model_id("xgboost") is None


class TestPipFreeze:
    """Environment capture."""

    def test_pip_freeze_in_metadata(self, snapshot_dir):
        from sklearn.linear_model import LogisticRegression
        from pipeline.models.model_persistence import save_snapshot, read_metadata

        X = np.random.RandomState(1).randn(20, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        d = save_snapshot(
            model=model, model_type="logistic", best_params={},
            feature_names=["a", "b", "c"], pip_freeze=None,
        )
        meta = read_metadata(d)
        assert "pip_freeze" in meta
        assert len(meta["pip_freeze"]) > 0
        assert "pip" in meta["pip_freeze"].lower() or "==" in meta["pip_freeze"]


class TestParentLineage:
    """Experiment lineage via parent_job_id."""

    def test_lineage_persisted_in_metadata(self, snapshot_dir):
        from sklearn.linear_model import LogisticRegression
        from pipeline.models.model_persistence import save_snapshot, read_metadata

        X = np.random.RandomState(1).randn(20, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression().fit(X, y)

        d1 = save_snapshot(model=model, model_type="logistic", best_params={},
                           feature_names=["a", "b", "c"], parent_job_id=None)
        m1 = read_metadata(d1)
        assert m1["parent_job_id"] is None

        d2 = save_snapshot(model=model, model_type="logistic", best_params={},
                           feature_names=["a", "b", "c"], parent_job_id="job-abc123")
        m2 = read_metadata(d2)
        assert m2["parent_job_id"] == "job-abc123"


class TestDataStoreMigration:
    """Verify parent_job_id column exists in DB."""

    def test_parent_job_id_column_exists(self):
        from pipeline.data.data_sqlite import DataStore
        import tempfile, os

        db = os.path.join(tempfile.mkdtemp(), "test.db")
        try:
            store = DataStore(db)
            with store._cursor() as (conn, cur):
                cur.execute("PRAGMA table_info(jobs)")
                cols = {row[1] for row in cur.fetchall()}
            assert "parent_job_id" in cols, "parent_job_id column missing from jobs table"
        finally:
            try:
                os.unlink(db)
                os.rmdir(os.path.dirname(db))
            except OSError:
                pass
