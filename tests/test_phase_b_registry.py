"""Phase B tests: deployed models registry + experiment tracking."""
from __future__ import annotations

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
def tmp_db():
    """Fresh SQLite database for each test."""
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    yield db
    try:
        os.unlink(db)
        os.rmdir(os.path.dirname(db))
    except OSError:
        pass


@pytest.fixture
def tmp_snapshot_dir():
    d = tempfile.mkdtemp(prefix="test_deploy_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _create_snapshot(snapshot_dir: str, model_type="logistic", sharpe=0.8, return_pct=15.0,
                     parent_job_id=None):
    """Helper to create a valid snapshot on disk."""
    from sklearn.linear_model import LogisticRegression
    from pipeline.model_persistence import save_snapshot

    X = np.random.RandomState(42).randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return save_snapshot(
        model=model, model_type=model_type,
        best_params={"C": 1.0},
        feature_names=["a", "b", "c"],
        seed=42, parent_job_id=parent_job_id,
        metrics={"sharpe": sharpe, "win_rate": 0.55, "total_return_pct": return_pct,
                 "max_drawdown": -10.0, "total_trades": 80},
    )


class TestDeployedModelsCRUD:
    """Register, list, activate, delete deployed models."""

    def test_register_and_list(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import register_snapshot, get_all_deployed

        import pipeline.model_persistence as mp
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            path = _create_snapshot(tmp_snapshot_dir)
            mid = register_snapshot(path, tmp_db)
            assert mid == os.path.basename(path)

            rows = get_all_deployed(tmp_db)
            assert len(rows) == 1
            assert rows[0]["model_type"] == "logistic"
            assert rows[0]["status"] == "inactive"
        finally:
            mp.DEPLOY_ROOT = orig

    def test_activate_and_delete(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import (
            register_snapshot, get_all_deployed, activate_model, delete_model,
        )
        import pipeline.model_persistence as mp
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            path = _create_snapshot(tmp_snapshot_dir, sharpe=0.9)
            mid = register_snapshot(path, tmp_db)

            ok = activate_model(mid, tmp_db)
            assert ok
            rows = get_all_deployed(tmp_db)
            assert rows[0]["status"] == "active"

            ok, reason = delete_model(mid, tmp_db)
            assert ok
            rows = get_all_deployed(tmp_db)
            assert len(rows) == 0
            assert not os.path.isdir(path)
        finally:
            mp.DEPLOY_ROOT = orig

    def test_only_one_active_per_model_type(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import (
            register_snapshot, get_all_deployed, activate_model,
        )
        import pipeline.model_persistence as mp
        import time
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            p1 = _create_snapshot(tmp_snapshot_dir, sharpe=0.8)
            time.sleep(1.1)  # avoid timestamp collision
            p2 = _create_snapshot(tmp_snapshot_dir, sharpe=0.9)
            id1 = register_snapshot(p1, tmp_db)
            id2 = register_snapshot(p2, tmp_db)

            activate_model(id1, tmp_db)
            rows = get_all_deployed(tmp_db)
            statuses = {r["id"]: r["status"] for r in rows}
            assert statuses[id1] == "active"
            assert statuses[id2] == "inactive"

            activate_model(id2, tmp_db)
            rows = get_all_deployed(tmp_db)
            statuses = {r["id"]: r["status"] for r in rows}
            assert statuses[id1] == "inactive"
            assert statuses[id2] == "active"
        finally:
            mp.DEPLOY_ROOT = orig

    def test_tags_update(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import register_snapshot, update_tags

        import pipeline.model_persistence as mp
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            path = _create_snapshot(tmp_snapshot_dir)
            mid = register_snapshot(path, tmp_db)

            tags = update_tags(mid, tmp_db, "add", "good")
            assert "good" in tags

            tags = update_tags(mid, tmp_db, "add", "verified")
            assert "good" in tags and "verified" in tags

            tags = update_tags(mid, tmp_db, "remove", "good")
            assert "good" not in tags
            assert "verified" in tags
        finally:
            mp.DEPLOY_ROOT = orig

    def test_scan_and_repair(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import scan_and_repair, get_all_deployed

        import pipeline.model_persistence as mp
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            _create_snapshot(tmp_snapshot_dir)
            _create_snapshot(tmp_snapshot_dir, model_type="xgboost")

            result = scan_and_repair(tmp_db)
            assert result["registered"] >= 2

            rows = get_all_deployed(tmp_db)
            assert len(rows) >= 2
            types = {r["model_type"] for r in rows}
            assert "logistic" in types
            assert "xgboost" in types
        finally:
            mp.DEPLOY_ROOT = orig


class TestActivePointer:
    """File-based active model pointer."""

    def test_activation_writes_pointer(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import register_snapshot, activate_model
        from pipeline.model_persistence import get_active_model_id

        import pipeline.model_persistence as mp
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            path = _create_snapshot(tmp_snapshot_dir)
            mid = register_snapshot(path, tmp_db)
            activate_model(mid, tmp_db)
            assert get_active_model_id("logistic") == mid
        finally:
            mp.DEPLOY_ROOT = orig

    def test_deactivation_clears_pointer(self, tmp_snapshot_dir, tmp_db):
        from pipeline.model_registry_disk import register_snapshot, activate_model, deactivate_model
        from pipeline.model_persistence import get_active_model_id

        import pipeline.model_persistence as mp
        orig = mp.DEPLOY_ROOT
        mp.DEPLOY_ROOT = tmp_snapshot_dir
        try:
            path = _create_snapshot(tmp_snapshot_dir)
            mid = register_snapshot(path, tmp_db)
            activate_model(mid, tmp_db)
            deactivate_model(mid, tmp_db)
            assert get_active_model_id("logistic") is None
        finally:
            mp.DEPLOY_ROOT = orig
