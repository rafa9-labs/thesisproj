"""Vast.ai client unit tests (mocked HTTP transport — no live API needed)."""
import httpx
import pytest

from api.services.vast_client import (
    VastClient,
    VastError,
    resolve_api_url,
)


def _instance(**overrides):
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


# ── resolve_api_url ──────────────────────────────────────────────────


class TestResolveApiUrl:
    def test_prefers_explicit_port_mapping(self):
        assert resolve_api_url(_instance()) == "http://1.2.3.4:41234"

    def test_direct_port_fallback(self):
        inst = _instance(ports={}, direct_port_start=30000)
        assert resolve_api_url(inst, port=8000) == "http://1.2.3.4:34000"

    def test_direct_port_only_for_ports_above_4000(self):
        inst = _instance(ports={}, direct_port_start=30000)
        assert resolve_api_url(inst, port=3000) is None

    def test_no_ip_returns_none(self):
        assert resolve_api_url(_instance(public_ipaddr=None)) is None

    def test_custom_port_and_scheme(self):
        inst = _instance(ports={"8501/tcp": [{"HostPort": 99}]})
        assert resolve_api_url(inst, port=8501, scheme="https") == "https://1.2.3.4:99"


# ── Client with mocked transport ─────────────────────────────────────


def _mock_transport(handler):
    return httpx.MockTransport(handler)


class TestVastClient:
    def test_missing_api_key_raises(self):
        with pytest.raises(VastError):
            VastClient("")

    def test_list_instances_sends_bearer_auth(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["auth"] = request.headers.get("Authorization")
            captured["path"] = request.url.path
            return httpx.Response(200, json={"instances": [_instance()]})

        client = VastClient("test-key", transport=_mock_transport(handler))
        instances = client.list_instances()
        assert len(instances) == 1
        assert captured["auth"] == "Bearer test-key"
        assert captured["path"] == "/api/v1/instances/"
        client.close()

    def test_http_error_raises_vast_error(self):
        def handler(request: httpx.Request):
            return httpx.Response(401, json={"msg": "Unauthorized"})

        client = VastClient("bad-key", transport=_mock_transport(handler))
        with pytest.raises(VastError, match="401"):
            client.list_instances()
        client.close()

    def test_non_json_response_raises(self):
        def handler(request: httpx.Request):
            return httpx.Response(502, text="<html>bad gateway</html>")

        client = VastClient("k", transport=_mock_transport(handler))
        with pytest.raises(VastError):
            client.list_instances()
        client.close()

    def test_get_instance_found_and_missing(self):
        def handler(request: httpx.Request):
            return httpx.Response(200, json={"instances": [_instance()]})

        client = VastClient("k", transport=_mock_transport(handler))
        assert client.get_instance(555) == _instance()
        assert client.get_instance(1) is None
        client.close()
