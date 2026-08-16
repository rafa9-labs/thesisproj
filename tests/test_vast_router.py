"""Vast.ai router endpoint tests — fully mocked (no live API, no storage)."""
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import vast as vast_router
from api.services.vast_client import VastClient, VastError

app = FastAPI()
app.include_router(vast_router.router, prefix="/api/v1")
client = TestClient(app)


def _fake_client(handler) -> VastClient:
    return VastClient("test-key", transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    monkeypatch.setattr(vast_router, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        vast_router,
        "_EXEC_PATH",
        pytest.importorskip("pathlib").Path("/tmp/vast_test_exec_config.json"),
    )
    yield
    try:
        vast_router._EXEC_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _offer(**overrides):
    offer = {
        "id": 1001,
        "machine_id": 42,
        "gpu_name": "RTX 3090",
        "gpu_ram": 24250,
        "dph_total": 0.30,
        "dlperf": 45.0,
        "num_gpus": 1,
        "cpu_cores": 8,
        "verification": "verified",
        "reliability": 0.99,
    }
    offer.update(overrides)
    return offer


class TestSettingsEndpoints:
    def test_get_settings_lists_defaults(self):
        resp = client.get("/api/v1/vast/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vast_min_gpu_class"] == "RTX 3090"
        assert data["has_api_key"] is True

    def test_put_settings_persists(self, monkeypatch):
        original = vast_router.settings.vast_max_dph
        monkeypatch.setattr(vast_router.settings, "vast_max_dph", original)
        resp = client.put("/api/v1/vast/settings", json={"vast_max_dph": 0.25})
        assert resp.status_code == 200
        assert vast_router.settings.vast_max_dph == 0.25
        saved = json.loads(vast_router._EXEC_PATH.read_text())
        assert saved["vast_max_dph"] == 0.25

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

            def get_api_key(self, name):
                return stored.get(name)

        monkeypatch.setattr("api.licensing.storage.SecureStorage", FakeStorage)
        resp = client.post("/api/v1/vast/api-key", json={"value": "new-key"})
        assert resp.status_code == 200
        assert stored.get("vast") == "new-key"

    def test_store_empty_key_rejected(self):
        resp = client.post("/api/v1/vast/api-key", json={"value": "  "})
        assert resp.status_code == 400

    def test_delete_api_key(self, monkeypatch):
        stored = {"vast": "k"}

        class FakeStorage:
            def __init__(self):
                pass

            def store_api_key(self, name, value):
                stored[name] = value

            def get_api_key(self, name):
                return stored.get(name)

        monkeypatch.setattr("api.licensing.storage.SecureStorage", FakeStorage)
        resp = client.delete("/api/v1/vast/api-key")
        assert resp.status_code == 200
        assert stored.get("vast") == ""


class TestOffersEndpoint:
    def test_offers_filtered_by_gpu_class(self, monkeypatch):
        def handler(request: httpx.Request):
            return httpx.Response(
                200,
                json={
                    "offers": [
                        _offer(id=1, gpu_name="RTX 4090", dph_total=0.4),
                        _offer(id=2, gpu_name="RTX 3070", dph_total=0.2),
                    ]
                },
            )

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        resp = client.get("/api/v1/vast/offers?gpu_class=RTX 3090")
        assert resp.status_code == 200
        ids = [o["ask_id"] for o in resp.json()["offers"]]
        assert ids == [1]

    def test_offers_without_api_key(self, monkeypatch):
        monkeypatch.setattr(vast_router, "_get_api_key", lambda: None)
        resp = client.get("/api/v1/vast/offers")
        assert resp.status_code == 400

    def test_offers_vast_error_mapped_to_502(self, monkeypatch):
        def handler(request: httpx.Request):
            return httpx.Response(500, json={"error": "boom"})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        resp = client.get("/api/v1/vast/offers")
        assert resp.status_code == 502


class TestInstancesEndpoints:
    def test_launch_auto_picks_cheapest_offer(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request):
            if request.url.path == "/api/v0/bundles/":
                return httpx.Response(
                    200,
                    json={
                        "offers": [
                            _offer(id=1, gpu_name="RTX 4090", dph_total=0.45),
                            _offer(id=2, gpu_name="RTX 4090", dph_total=0.35),
                        ]
                    },
                )
            captured["ask_path"] = request.url.path
            return httpx.Response(200, json={"success": True, "new_contract": 777})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        monkeypatch.setattr(vast_router.settings, "vast_repo_url", "https://x/y.git")
        resp = client.post(
            "/api/v1/vast/instances",
            json={"gpu_class": "RTX 4090", "disk_gb": 80},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["instance_id"] == 777
        assert captured["ask_path"] == "/api/v0/asks/2/"

    def test_launch_explicit_ask_id(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request):
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "new_contract": 42})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        monkeypatch.setattr(vast_router.settings, "vast_repo_url", "https://x/y.git")
        resp = client.post(
            "/api/v1/vast/instances",
            json={"ask_id": 1001, "image": "custom:img", "onstart": "echo custom"},
        )
        assert resp.status_code == 200
        assert captured["path"] == "/api/v0/asks/1001/"
        assert captured["body"]["image"] == "custom:img"
        assert captured["body"]["onstart"] == "echo custom"

    def test_launch_no_matching_offers_404(self, monkeypatch):
        def handler(request: httpx.Request):
            return httpx.Response(200, json={"offers": [_offer(gpu_name="RTX 3060")]})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        resp = client.post(
            "/api/v1/vast/instances", json={"gpu_class": "RTX 3090"}
        )
        assert resp.status_code == 404

    def test_launch_requires_repo_url_for_default_onstart(self, monkeypatch):
        def handler(request: httpx.Request):
            return httpx.Response(200, json={"offers": [_offer()]})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        monkeypatch.setattr(vast_router.settings, "vast_repo_url", "")
        resp = client.post("/api/v1/vast/instances", json={"ask_id": 1001})
        assert resp.status_code == 400

    def test_list_instances(self, monkeypatch):
        def handler(request: httpx.Request):
            return httpx.Response(
                200,
                json={
                    "instances": [
                        {
                            "id": 555,
                            "actual_status": "running",
                            "public_ipaddr": "1.2.3.4",
                            "ports": {"8001/tcp": [{"HostPort": 41234}]},
                        }
                    ]
                },
            )

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        resp = client.get("/api/v1/vast/instances")
        assert resp.status_code == 200
        inst = resp.json()["instances"][0]
        assert inst["id"] == 555
        assert inst["remote_api_url"] == "https://1.2.3.4:41234"

    def test_get_instance_404(self, monkeypatch):
        def handler(request: httpx.Request):
            return httpx.Response(200, json={"instances": []})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        resp = client.get("/api/v1/vast/instances/999")
        assert resp.status_code == 404

    def test_destroy_instance(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request):
            captured["method"] = request.method
            return httpx.Response(200, json={"success": True})

        monkeypatch.setattr(
            vast_router, "_get_client", lambda: _fake_client(handler)
        )
        resp = client.delete("/api/v1/vast/instances/555")
        assert resp.status_code == 200
        assert captured["method"] == "DELETE"
