"""Vast.ai router endpoint tests — fully mocked (no live API, no storage)."""
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import vast as vast_router
from api.services.vast_client import VastClient
from api.services.vast_executor import VastExecError

app = FastAPI()
app.include_router(vast_router.router, prefix="/api/v1")
client = TestClient(app)


def _fake_client(handler) -> VastClient:
    return VastClient("test-key", transport=httpx.MockTransport(handler))


def _vast_handler(instances=None, status_code=200):
    instances = instances if instances is not None else [_running_instance()]

    def handler(request: httpx.Request):
        return httpx.Response(status_code, json={"instances": instances})

    return handler


def _running_instance(**overrides):
    inst = {
        "id": 555,
        "actual_status": "running",
        "gpu_name": "RTX 4090",
        "dph_total": 0.30,
        "public_ipaddr": "1.2.3.4",
        "ports": {"8000/tcp": [{"HostPort": 41234}]},
    }
    inst.update(overrides)
    return inst


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    monkeypatch.setattr(vast_router, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        vast_router,
        "_EXEC_PATH",
        pytest.importorskip("pathlib").Path("/tmp/vast_test_exec_config.json"),
    )
    monkeypatch.setattr(vast_router, "_mirror_local_job", lambda *a, **kw: None)
    monkeypatch.setattr(vast_router, "_update_local_mirror", lambda *a, **kw: None)
    yield
    try:
        vast_router._EXEC_PATH.unlink(missing_ok=True)
    except Exception:
        pass


class TestSettingsEndpoints:
    def test_get_settings_lists_defaults(self):
        resp = client.get("/api/v1/vast/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vast_remote_port"] == 8000
        assert data["has_api_key"] is True

    def test_put_settings_persists(self, monkeypatch):
        original = vast_router.settings.vast_remote_port
        monkeypatch.setattr(vast_router.settings, "vast_remote_port", original)
        resp = client.put("/api/v1/vast/settings", json={"vast_remote_port": 8100})
        assert resp.status_code == 200
        assert vast_router.settings.vast_remote_port == 8100
        saved = json.loads(vast_router._EXEC_PATH.read_text())
        assert saved["vast_remote_port"] == 8100

    def test_settings_without_key(self, monkeypatch):
        monkeypatch.setattr(vast_router, "_get_api_key", lambda: None)
        resp = client.get("/api/v1/vast/settings")
        assert resp.json()["has_api_key"] is False


class TestApiKeyEndpoints:
    def test_store_api_key(self, monkeypatch):
        stored = {}

        class FakeStorage:
            def __init__(self):
                pass

            def store_api_key(self, name, value):
                stored[name] = value

        monkeypatch.setattr("api.licensing.storage.SecureStorage", FakeStorage)
        resp = client.post("/api/v1/vast/api-key", json={"value": "new-key"})
        assert resp.status_code == 200
        assert stored.get("vast") == "new-key"

    def test_store_empty_key_rejected(self):
        resp = client.post("/api/v1/vast/api-key", json={"value": "  "})
        assert resp.status_code == 400


class TestInstancesEndpoint:
    def test_filters_to_running_only_with_api_url(self, monkeypatch):
        handler = _vast_handler(
            instances=[
                _running_instance(id=1, gpu_name="RTX 4090"),
                _running_instance(id=2, gpu_name="RTX 3090", actual_status="stopped"),
                _running_instance(
                    id=3,
                    gpu_name="A6000",
                    public_ipaddr="5.6.7.8",
                    ports={},
                    direct_port_start=30000,
                ),
            ]
        )
        monkeypatch.setattr(vast_router, "_get_client", lambda: _fake_client(handler))
        resp = client.get("/api/v1/vast/instances")
        assert resp.status_code == 200
        instances = resp.json()["instances"]
        assert [i["id"] for i in instances] == [1, 3]
        assert instances[0]["api_url"] == "http://1.2.3.4:41234"
        assert instances[1]["api_url"] == "http://5.6.7.8:34000"

    def test_instances_without_api_key(self, monkeypatch):
        monkeypatch.setattr(vast_router, "_get_api_key", lambda: None)
        resp = client.get("/api/v1/vast/instances")
        assert resp.status_code == 400

    def test_vast_error_mapped_to_502(self, monkeypatch):
        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(_vast_handler(status_code=500))
        )
        resp = client.get("/api/v1/vast/instances")
        assert resp.status_code == 502


class TestRunBacktest:
    def test_success_submits_and_mirrors(self, monkeypatch):
        submitted = {}

        def fake_submit(api_url, config):
            submitted["api_url"] = api_url
            submitted["config"] = config
            return {"job_id": "remote-123", "status": "pending"}

        monkeypatch.setattr(vast_router, "submit_remote_backtest", fake_submit)
        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(_vast_handler())
        )
        mirror_calls = []
        monkeypatch.setattr(
            vast_router, "_mirror_local_job", lambda *a, **kw: mirror_calls.append(a)
        )
        payload = {"instance_id": 555, "config": {"pair": "EURUSD", "models": ["logistic"]}}
        resp = client.post("/api/v1/vast/run-backtest", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "remote-123"
        assert body["api_url"] == "http://1.2.3.4:41234"
        assert submitted["config"] == {"pair": "EURUSD", "models": ["logistic"]}
        assert mirror_calls[0][0] == "remote-123"

    def test_unknown_instance_404(self, monkeypatch):
        monkeypatch.setattr(vast_router, "_get_client", lambda: _fake_client(_vast_handler()))
        resp = client.post(
            "/api/v1/vast/run-backtest", json={"instance_id": 999, "config": {}}
        )
        assert resp.status_code == 404

    def test_stopped_instance_400(self, monkeypatch):
        handler = _vast_handler(instances=[_running_instance(actual_status="stopped")])
        monkeypatch.setattr(vast_router, "_get_client", lambda: _fake_client(handler))
        resp = client.post(
            "/api/v1/vast/run-backtest", json={"instance_id": 555, "config": {}}
        )
        assert resp.status_code == 400

    def test_unreachable_remote_502(self, monkeypatch):
        def fake_submit(api_url, config):
            raise VastExecError("Remote instance unreachable")

        monkeypatch.setattr(vast_router, "submit_remote_backtest", fake_submit)
        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(_vast_handler())
        )
        resp = client.post(
            "/api/v1/vast/run-backtest", json={"instance_id": 555, "config": {}}
        )
        assert resp.status_code == 502

    def test_no_reachable_port_400(self, monkeypatch):
        handler = _vast_handler(
            instances=[_running_instance(ports={}, direct_port_start=None)]
        )
        monkeypatch.setattr(vast_router, "_get_client", lambda: _fake_client(handler))
        resp = client.post(
            "/api/v1/vast/run-backtest", json={"instance_id": 555, "config": {}}
        )
        assert resp.status_code == 400


class TestRunStatusProxy:
    def test_running_status_proxied(self, monkeypatch):
        monkeypatch.setattr(
            vast_router,
            "poll_remote_status",
            lambda api_url, job_id: {
                "status": "running",
                "progress": {"fold": 2, "total_folds": 12},
            },
        )
        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(_vast_handler())
        )
        resp = client.get("/api/v1/vast/runs/remote-123?instance_id=555")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["results"] is None
        assert body["progress"]["fold"] == 2

    def test_completed_attaches_results_and_mirrors(self, monkeypatch):
        results = {
            "job_id": "remote-123",
            "pair": "EURUSD",
            "models": ["logistic"],
            "metrics": [{"sharpe": 1.4, "total_return_pct": 12.5}],
        }
        monkeypatch.setattr(
            vast_router,
            "poll_remote_status",
            lambda api_url, job_id: {"status": "completed"},
        )
        monkeypatch.setattr(
            vast_router, "fetch_remote_results", lambda api_url, job_id: results
        )
        mirror_calls = []
        monkeypatch.setattr(
            vast_router, "_update_local_mirror", lambda *a, **kw: mirror_calls.append(a)
        )
        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(_vast_handler())
        )
        resp = client.get("/api/v1/vast/runs/remote-123?instance_id=555")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["results"]["metrics"][0]["sharpe"] == 1.4
        assert mirror_calls[0][:3] == ("remote-123", "completed", results)

    def test_failed_mirrors_error(self, monkeypatch):
        monkeypatch.setattr(
            vast_router,
            "poll_remote_status",
            lambda api_url, job_id: {"status": "failed", "error": "boom"},
        )
        mirror_calls = []
        monkeypatch.setattr(
            vast_router, "_update_local_mirror", lambda *a, **kw: mirror_calls.append(a)
        )
        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(_vast_handler())
        )
        resp = client.get("/api/v1/vast/runs/remote-123?instance_id=555")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        assert mirror_calls[0][:3] == ("remote-123", "failed", None)
