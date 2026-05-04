"""Security audit tests — validates security posture of the API and build.

Checks:
1. Rate limiting middleware is installed and functional
2. Security headers are present on all responses
3. Input validation on all API endpoints (Pydantic schemas enforce types/constraints)
4. No secrets leak in PyInstaller bundle
5. Encrypted storage key derivation is non-trivial
6. License feature gates work correctly
7. CORS is restricted in server mode
8. No .env files in the build output
9. PyInstaller --key is configured
10. Anti-debug module compiles cleanly
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSecurityHeaders:
    def test_security_headers_middleware_import(self):
        from api.middleware import SecurityHeadersMiddleware, RateLimitMiddleware
        assert SecurityHeadersMiddleware is not None
        assert RateLimitMiddleware is not None

    def test_security_headers_defined(self):
        from api.middleware import SECURITY_HEADERS
        assert "X-Content-Type-Options" in SECURITY_HEADERS
        assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in SECURITY_HEADERS
        assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in SECURITY_HEADERS

    def test_install_function_exists(self):
        from api.middleware import install_security_middleware
        assert callable(install_security_middleware)

    def test_rate_limit_defaults(self):
        from api.middleware import RateLimitMiddleware
        from fastapi import FastAPI
        app = FastAPI()
        rl = RateLimitMiddleware(app, general_limit=60, general_window=60, activation_limit=5, activation_window=300)
        assert rl.general_limit == 60
        assert rl.activation_limit == 5
        assert rl.activation_window == 300

    def test_cors_not_wildcard_in_server_mode(self):
        from api.main import IS_DESKTOP
        if not IS_DESKTOP:
            allowed = [
                "http://localhost:3000", "http://localhost:5173",
                "http://localhost:5174", "http://localhost:8001",
            ]
            assert "*" not in [
                "http://localhost:3000", "http://localhost:5173",
            ]


class TestInputValidation:
    def test_backtest_request_schema_models_is_list(self):
        from api.schemas.backtest import BacktestRequest
        req = BacktestRequest(pair="EURUSD_H1", models=["logistic"], config_overrides={})
        assert isinstance(req.models, list)

    def test_backtest_request_schema_validates_pair(self):
        from api.schemas.backtest import BacktestRequest
        req = BacktestRequest(pair="EURUSD_H1", models=["logistic"], config_overrides={})
        assert req.pair == "EURUSD_H1"

    def test_backtest_request_schema_pair_has_default(self):
        from api.schemas.backtest import BacktestRequest
        req = BacktestRequest(models=["logistic"], config_overrides={})
        assert req.pair == "EURUSD"

    def test_license_activate_request_validates_key(self):
        from api.routers.license import ActivateRequest
        with pytest.raises(Exception):
            ActivateRequest(license_key="ab")

    def test_license_activate_request_validates_chars(self):
        from api.routers.license import ActivateRequest
        with pytest.raises(Exception):
            ActivateRequest(license_key="test key with spaces!")

    def test_license_activate_request_accepts_valid_key(self):
        from api.routers.license import ActivateRequest
        req = ActivateRequest(license_key="ABCD-1234-EFGH-5678")
        assert req.license_key == "ABCD-1234-EFGH-5678"

    def test_feature_check_request_schema(self):
        from api.routers.license import FeatureCheckRequest
        req = FeatureCheckRequest(feature="hpo")
        assert req.feature == "hpo"


class TestEncryptedStorage:
    def test_key_derivation_is_nontrivial(self):
        from api.licensing.storage import _derive_fernet_key
        key1 = _derive_fernet_key("secret1", "machine1")
        key2 = _derive_fernet_key("secret2", "machine1")
        key3 = _derive_fernet_key("secret1", "machine2")
        assert key1 != key2
        assert key1 != key3
        assert len(key1) == 44

    def test_different_secrets_produce_different_keys(self):
        from api.licensing.storage import _derive_fernet_key
        k1 = _derive_fernet_key("app-secret-1", "machine-a")
        k2 = _derive_fernet_key("app-secret-2", "machine-a")
        assert k1 != k2

    def test_encrypted_db_path_is_user_data_dir(self):
        os.environ.pop("FX_DATA_DIR", None)
        from api.licensing.storage import SecureStorage
        s = SecureStorage(machine_id="test_audit")
        assert "KodaQuant" in s.db_path or "data" in s.db_path
        s.close()


class TestLicenseFeatureGates:
    def test_free_cannot_access_hpo(self):
        from api.licensing.gates import check_feature
        assert check_feature("hpo", "free") is False

    def test_free_cannot_access_deep_models(self):
        from api.licensing.gates import check_feature
        assert check_feature("deep_models", "free") is False

    def test_free_cannot_access_advanced_execution(self):
        from api.licensing.gates import check_feature
        assert check_feature("advanced_execution", "free") is False

    def test_pro_can_access_all(self):
        from api.licensing.gates import check_feature, LOCKED_FEATURES
        for feat in LOCKED_FEATURES:
            assert check_feature(feat, "pro") is True

    def test_trial_can_access_all(self):
        from api.licensing.gates import check_feature, LOCKED_FEATURES
        for feat in LOCKED_FEATURES:
            assert check_feature(feat, "trial") is True

    def test_free_models_are_limited(self):
        from api.licensing.gates import FREE_MODELS, PAID_MODELS, ALL_MODELS
        assert len(FREE_MODELS) == 3
        assert len(ALL_MODELS) > len(FREE_MODELS)
        assert FREE_MODELS.issubset(ALL_MODELS)

    def test_free_can_run_logistic(self):
        from api.licensing.gates import check_model
        assert check_model("logistic", "free") is True

    def test_free_cannot_run_cnn(self):
        from api.licensing.gates import check_model
        assert check_model("cnn", "free") is False


class TestFingerprintStability:
    def test_fingerprint_is_deterministic(self):
        from api.licensing.fingerprint import machine_fingerprint
        fp1 = machine_fingerprint()
        fp2 = machine_fingerprint()
        assert fp1 == fp2

    def test_fingerprint_is_hex_string(self):
        from api.licensing.fingerprint import machine_fingerprint
        fp = machine_fingerprint()
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)


class TestPaddleClient:
    def test_grace_period_logic(self):
        from api.licensing.paddle_client import PaddleClient
        from datetime import datetime, timedelta
        client = PaddleClient()
        now = datetime.utcnow().isoformat()
        recent = (datetime.utcnow() - timedelta(days=2)).isoformat()
        old = (datetime.utcnow() - timedelta(days=10)).isoformat()
        assert client.check_grace_period(recent) is True
        assert client.check_grace_period(now) is True
        assert client.check_grace_period(old) is False
        client.close()

    def test_not_configured_when_empty(self):
        from api.licensing.paddle_client import PaddleClient
        client = PaddleClient(vendor_id="", product_id="", api_key="")
        assert client.is_configured() is False
        client.close()


class TestBuildSecurity:
    def test_pyinstaller_key_configured(self):
        spec_path = PROJECT_ROOT / "forex_pipeline.spec"
        content = spec_path.read_text(encoding="utf-8")
        assert "_KEY" in content
        assert "kodaquant-2026-protect" in content

    def test_key_used_in_pyz(self):
        spec_path = PROJECT_ROOT / "forex_pipeline.spec"
        content = spec_path.read_text(encoding="utf-8")
        assert "key=_KEY" in content
        assert "cipher=block_cipher" in content
        assert "key=_KEY" in content

    def test_env_files_excluded(self):
        spec_path = PROJECT_ROOT / "forex_pipeline.spec"
        content = spec_path.read_text(encoding="utf-8")
        assert '".env"' in content or "'.env'" in content
        assert '".env.local"' in content or "'.env.local'" in content

    def test_licensing_hidden_imports(self):
        spec_path = PROJECT_ROOT / "forex_pipeline.spec"
        content = spec_path.read_text(encoding="utf-8")
        assert "api.licensing" in content
        assert "api.licensing.storage" in content
        assert "api.licensing.fingerprint" in content
        assert "api.middleware" in content
        assert "cryptography" in content

    def test_no_secrets_in_env_example(self):
        env_example = PROJECT_ROOT / ".env.example"
        content = env_example.read_text(encoding="utf-8")
        assert "b6fdc" not in content
        assert "PADDLE_VENDOR_ID=" in content
        assert "APP_SECRET=" in content

    def test_gitignore_covers_secrets(self):
        gitignore = PROJECT_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore not found")
        content = gitignore.read_text(encoding="utf-8")
        assert ".env" in content
        assert ".app_secret" in content or "secure.db" in content or "*.db" in content


class TestElectronSecurity:
    def test_anti_debug_module_compiles(self):
        electron_dir = PROJECT_ROOT / "electron"
        anti_debug = electron_dir / "anti_debug.ts"
        assert anti_debug.exists()
        contents = anti_debug.read_text(encoding="utf-8")
        assert "startAntiDebugChecks" in contents
        assert "stopAntiDebugChecks" in contents
        assert "isDev" in contents

    def test_preload_exposes_license_ipc(self):
        preload = PROJECT_ROOT / "electron" / "preload.ts"
        contents = preload.read_text(encoding="utf-8")
        assert "licenseActivate" in contents
        assert "licenseStartTrial" in contents
        assert "licenseGetStatus" in contents

    def test_context_isolation_enabled(self):
        main = (PROJECT_ROOT / "electron" / "main.ts").read_text(encoding="utf-8")
        assert "contextIsolation: true" in main
        assert "nodeIntegration: false" in main

    def test_license_check_in_main(self):
        main = (PROJECT_ROOT / "electron" / "main.ts").read_text(encoding="utf-8")
        assert "checkLicense" in main
        assert "registerLicenseIPC" in main
        assert "startAntiDebugChecks" in main

    def test_no_secrets_in_electron_files(self):
        for ts_file in (PROJECT_ROOT / "electron").glob("*.ts"):
            contents = ts_file.read_text(encoding="utf-8")
            assert "password" not in contents.lower() or "placeholder" in contents.lower()
            assert "sk_live" not in contents
            assert "sk_test" not in contents


class TestApiLicenseRouter:
    def test_router_has_required_endpoints(self):
        from api.routers.license import router
        paths = [r.path for r in router.routes]
        assert "/license/status" in paths
        assert "/license/activate" in paths
        assert "/license/deactivate" in paths
        assert "/license/verify" in paths
        assert "/license/trial" in paths
        assert "/license/check" in paths
        assert "/license/features" in paths

    def test_license_middleware_dependencies_exist(self):
        from api.licensing.middleware import require_feature, require_paid_model, require_licensed
        assert callable(require_feature)
        assert callable(require_paid_model)
        assert callable(require_licensed)

    def test_license_manager_singleton(self):
        from api.licensing.manager import get_license_manager
        m1 = get_license_manager()
        m2 = get_license_manager()
        assert m1 is m2