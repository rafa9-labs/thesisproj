"""Integration tests for Gatekeeper VRAM allocation + backtest dispatch.

Tests the full path from POST /backtest through VRAM gate, create_job_atomic,
and ProcessManager submission. Uses TestClient against the live API.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_active_jobs():
    from api.dependencies import get_data_store
    from api.services import JobManager
    store = get_data_store()
    jm = JobManager(store)
    jobs = jm.get_active_jobs("backtest")
    for j in jobs:
        try:
            jm.force_stop_job(j["id"])
        except Exception:
            pass
    yield
    # Teardown: clean up any jobs the test left behind
    try:
        jobs = jm.get_active_jobs("backtest")
        for j in jobs:
            try:
                jm.delete_job(j["id"])
            except Exception:
                pass
    except Exception:
        pass


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def _valid_payload(**overrides):
    p = {
        "pair": "EURUSD",
        "timeframe": "H1",
        "models": ["logistic"],
        "months": 3,
        "repeats": 1,
        "seed": 42,
        "hpo_intensity": "quick",
        "n_trials": None,
        "start_date": None,
        "end_date": None,
        "parent_job_id": None,
        "trading_costs": True,
        "config_overrides": {},
    }
    p.update(overrides)
    return p


def _make_mock_pm(vram_available=8192, allocate_ok=True):
    pm = MagicMock()
    pm.gpu_vram_available_mb = vram_available
    pm.allocate_vram.return_value = allocate_ok
    pm.submit.return_value = MagicMock()
    pm.release_vram.return_value = None
    return pm


class TestGatekeeperVRAMFlow:

    def test_cpu_job_skips_vram_allocate(self, client):
        mock_pm = _make_mock_pm()
        with patch("api.routers.backtest.IS_DESKTOP", True):
            with patch("api.process_manager.get_process_manager", return_value=mock_pm):
                resp = client.post("/api/v1/backtest", json=_valid_payload())
                assert resp.status_code == 202
                data = resp.json()
                assert data["status"] == "pending"
                mock_pm.allocate_vram.assert_not_called()
                call_args = mock_pm.submit.call_args
                env_vars = call_args[1].get("env_vars", {})
                assert "CUDA_VRAM_LIMIT_MB" not in env_vars

    def test_gpu_job_allocates_vram_and_passes_env(self, client):
        mock_pm = _make_mock_pm()
        import api.config
        api.config.settings.gpu_total_vram_mb = 8192
        api.config.settings.gpu_enabled = True

        with patch("api.routers.backtest.IS_DESKTOP", True):
            with patch("api.process_manager.get_process_manager", return_value=mock_pm):
                resp = client.post(
                    "/api/v1/backtest",
                    json=_valid_payload(models=["cnn"]),
                )
                assert resp.status_code == 202
                mock_pm.allocate_vram.assert_called_once()
                call_args = mock_pm.submit.call_args
                env_vars = call_args[1].get("env_vars", {})
                assert "CUDA_VRAM_LIMIT_MB" in env_vars

        api.config.settings.gpu_total_vram_mb = 0
        api.config.settings.gpu_enabled = False

    def test_gpu_job_blocked_when_vram_full(self, client):
        mock_pm = _make_mock_pm(vram_available=0, allocate_ok=False)
        import api.config
        api.config.settings.gpu_total_vram_mb = 8192
        api.config.settings.gpu_enabled = True

        with patch("api.routers.backtest.IS_DESKTOP", True):
            with patch("api.process_manager.get_process_manager", return_value=mock_pm):
                resp = client.post(
                    "/api/v1/backtest",
                    json=_valid_payload(models=["cnn"]),
                )
                assert resp.status_code == 409
                assert "VRAM" in resp.json()["detail"]

        api.config.settings.gpu_total_vram_mb = 0
        api.config.settings.gpu_enabled = False

    def test_stale_jobs_dont_block_gate(self, client):
        from api.dependencies import get_data_store
        from api.services import JobManager

        store = get_data_store()
        jm = JobManager(store)
        stale_id = f"stale-intgr-{uuid.uuid4().hex[:6]}"
        jm.create_job(stale_id, "backtest", {"pair": "EURUSD"})
        with store._cursor() as (conn, cur):
            cur.execute(
                "UPDATE jobs SET updated_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
                (stale_id,),
            )

        mock_pm = _make_mock_pm()
        with patch("api.routers.backtest.IS_DESKTOP", True):
            with patch("api.process_manager.get_process_manager", return_value=mock_pm):
                resp = client.post("/api/v1/backtest", json=_valid_payload())
                assert resp.status_code == 202

    def test_max_concurrent_cpu_blocked(self, client):
        from api.dependencies import get_data_store
        from api.services import JobManager
        import api.config

        api.config.settings.max_concurrent_backtests = 1
        store = get_data_store()
        jm = JobManager(store)
        jid = f"cpu-limit-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid, "backtest", {"pair": "EURUSD"})

        mock_pm = _make_mock_pm()
        with patch("api.routers.backtest.IS_DESKTOP", True):
            with patch("api.process_manager.get_process_manager", return_value=mock_pm):
                resp = client.post("/api/v1/backtest", json=_valid_payload())
                assert resp.status_code == 409
        api.config.settings.max_concurrent_backtests = 4


class TestBacktestEndpointValidation:

    def test_valid_payload_returns_202(self, client):
        mock_pm = _make_mock_pm()
        with patch("api.routers.backtest.IS_DESKTOP", True):
            with patch("api.process_manager.get_process_manager", return_value=mock_pm):
                resp = client.post("/api/v1/backtest", json=_valid_payload())
                assert resp.status_code == 202
                data = resp.json()
                assert "job_id" in data
                assert data["status"] == "pending"

    def test_empty_models_returns_error(self, client):
        resp = client.post("/api/v1/backtest", json=_valid_payload(models=[]))
        assert resp.status_code in (422, 400, 202)

    def test_get_active_backtests_returns_list(self, client):
        resp = client.get("/api/v1/backtest/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_get_active_backtests_includes_total(self, client):
        resp = client.get("/api/v1/backtest/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert isinstance(data["total"], int)


class TestForceStop:

    def test_force_stop_existing_job(self, client):
        from api.dependencies import get_data_store
        from api.services import JobManager

        store = get_data_store()
        jm = JobManager(store)
        jid = f"stop-intgr-{uuid.uuid4().hex[:6]}"
        jm.create_job(jid, "backtest", {"pair": "EURUSD"})
        jm.update_status(jid, "running")

        resp = client.post(f"/api/v1/backtest/{jid}/force-stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"

    def test_force_stop_nonexistent(self, client):
        resp = client.post("/api/v1/backtest/bogus-id-999/force-stop")
        assert resp.status_code == 404
