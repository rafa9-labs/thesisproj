"""Tests for the licensing module — storage, fingerprint, gates, manager."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from api.licensing.gates import (
    FREE_MODELS, PAID_MODELS, ALL_MODELS, LOCKED_FEATURES,
    check_feature, check_model, get_available_models, get_locked_models,
)
from api.licensing.fingerprint import machine_fingerprint, fingerprint_components, matches_stored_fingerprint


class TestFeatureGates:
    def test_free_models_set(self):
        assert "logistic" in FREE_MODELS
        assert "xgboost" in FREE_MODELS
        assert "random_forest" in FREE_MODELS

    def test_paid_models_set(self):
        assert "cnn" in PAID_MODELS
        assert "lstm" in PAID_MODELS
        assert "transformer" in PAID_MODELS
        assert "dqn" in PAID_MODELS
        assert "ensemble_cnn_lstm_xgboost" in PAID_MODELS

    def test_check_model_free_plan(self):
        assert check_model("logistic", "free") is True
        assert check_model("cnn", "free") is False
        assert check_model("xgboost", "free") is True
        assert check_model("dqn", "free") is False

    def test_check_model_pro_plan(self):
        assert check_model("cnn", "pro") is True
        assert check_model("dqn", "pro") is True
        assert check_model("logistic", "pro") is True

    def test_check_model_trial_plan(self):
        assert check_model("cnn", "trial") is True
        assert check_model("ensemble_cnn_lstm_xgboost", "trial") is True

    def test_check_feature_free_plan(self):
        assert check_feature("hpo", "free") is False
        assert check_feature("news_sentiment", "free") is False
        assert check_feature("basic_backtest", "free") is True

    def test_check_feature_pro_plan(self):
        assert check_feature("hpo", "pro") is True
        assert check_feature("news_sentiment", "pro") is True

    def test_get_available_models(self):
        free = get_available_models("free")
        assert "logistic" in free
        assert "cnn" not in free
        pro = get_available_models("pro")
        assert "cnn" in pro
        assert len(pro) == len(ALL_MODELS)

    def test_get_locked_models(self):
        locked = get_locked_models("free")
        assert len(locked) > 0
        assert "cnn" in locked
        pro_locked = get_locked_models("pro")
        assert len(pro_locked) == 0


class TestMachineFingerprint:
    def test_returns_hex_string(self):
        fp = machine_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)

    def test_consistent_results(self):
        fp1 = machine_fingerprint()
        fp2 = machine_fingerprint()
        assert fp1 == fp2

    def test_fingerprint_components_returns_list(self):
        comps = fingerprint_components()
        assert isinstance(comps, list)


class TestSecureStorage:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def storage(self, tmp_dir):
        from api.licensing.storage import SecureStorage
        s = SecureStorage(
            db_path=str(tmp_dir / "test_secure.db"),
            machine_id="test_machine_123",
        )
        os.environ["APP_SECRET"] = "test-secret-key-for-testing"
        yield s
        s.close()
        os.environ.pop("APP_SECRET", None)

    def test_store_and_retrieve_api_key(self, storage):
        storage.store_api_key("oanda", "secret_token_123")
        val = storage.get_api_key("oanda")
        assert val == "secret_token_123"

    def test_missing_api_key_returns_none(self, storage):
        assert storage.get_api_key("nonexistent") is None

    def test_delete_api_key(self, storage):
        storage.store_api_key("test", "val")
        assert storage.delete_api_key("test") is True
        assert storage.get_api_key("test") is None

    def test_store_and_retrieve_license(self, storage):
        storage.store_license(
            license_key="ABC-123-DEF",
            activation_id="act_001",
            machine_fingerprint="fp123",
            plan="pro",
            activated_at=datetime.utcnow().isoformat(),
        )
        lic = storage.get_license()
        assert lic is not None
        assert lic["license_key"] == "ABC-123-DEF"
        assert lic["plan"] == "pro"
        assert lic["activation_id"] == "act_001"

    def test_delete_license(self, storage):
        storage.store_license("KEY", "act", "fp", "pro", datetime.utcnow().isoformat())
        assert storage.delete_license() is True
        assert storage.get_license() is None

    def test_trial_lifecycle(self, storage):
        result = storage.start_trial(duration_days=14)
        assert "started_at" in result
        trial = storage.get_trial()
        assert trial is not None
        assert trial["active"] is True
        storage.end_trial()
        trial = storage.get_trial()
        assert trial["active"] is False

    def test_kv_store(self, storage):
        storage.set_kv("test_key", "test_value")
        assert storage.get_kv("test_key") == "test_value"
        assert storage.get_kv("missing") is None


class TestPaddleClient:
    def test_not_configured_by_default(self):
        from api.licensing.paddle_client import PaddleClient
        client = PaddleClient()
        assert not client.is_configured()
        client.close()

    def test_sandbox_mode(self):
        from api.licensing.paddle_client import PaddleClient
        client = PaddleClient(sandbox=True)
        assert client.sandbox is True
        assert "sandbox" in client._base
        client.close()

    def test_grace_period_within_range(self):
        from api.licensing.paddle_client import PaddleClient
        client = PaddleClient()
        recent = (datetime.utcnow() - timedelta(days=3)).isoformat()
        assert client.check_grace_period(recent) is True
        client.close()

    def test_grace_period_expired(self):
        from api.licensing.paddle_client import PaddleClient
        client = PaddleClient()
        old = (datetime.utcnow() - timedelta(days=10)).isoformat()
        assert client.check_grace_period(old) is False
        client.close()


class TestLicenseManager:
    @pytest.fixture
    def manager(self, tmp_path):
        import api.licensing.manager as mgr_module
        mgr_module._manager = None
        os.environ["APP_SECRET"] = "test-secret"
        with patch("api.licensing.manager.machine_fingerprint", return_value="test_fp_123"):
            from api.licensing.manager import LicenseManager
            m = LicenseManager()
            m._storage = SecureStorage(
                db_path=str(tmp_path / "test.db"),
                machine_id="test_fp_123",
            )
            yield m
            m.close()
            mgr_module._manager = None
            os.environ.pop("APP_SECRET", None)

    def test_initial_status_is_free(self, manager):
        status = manager.get_status()
        assert status.plan == "free"
        assert status.licensed is False
        assert status.needs_activation is True

    def test_start_trial(self, manager):
        result = manager.start_trial()
        assert result["success"] is True
        status = manager.get_status()
        assert status.plan == "trial"
        assert status.trial_active is True
        assert status.trial_days_left > 0

    def test_start_trial_twice_fails(self, manager):
        manager.start_trial()
        result = manager.start_trial()
        assert result["success"] is False

    def test_status_dict(self, manager):
        d = manager.get_status().to_dict()
        assert "plan" in d
        assert "licensed" in d
        assert "available_models" in d
        assert "locked_models" in d

    def test_check_feature_gates(self, manager):
        assert manager.check_feature("hpo") is False
        assert manager.check_model("cnn") is False
        assert manager.check_model("logistic") is True

    def test_trial_grants_features(self, manager):
        manager.start_trial()
        assert manager.check_feature("hpo") is True
        assert manager.check_model("cnn") is True


from api.licensing.storage import SecureStorage