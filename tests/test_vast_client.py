"""Vast.ai client unit tests (mocked HTTP transport — no live API needed)."""
import json

import httpx
import pytest

from api.services.vast_client import (
    GPU_CLASS_RANKING,
    VastClient,
    VastError,
    build_kodaquant_onstart,
    filter_offers_by_gpu_class,
    gpu_class_rank,
)


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


# ── GPU class ranking ────────────────────────────────────────────────


class TestGpuClassRank:
    def test_known_gpu_ranked(self):
        assert gpu_class_rank("RTX 3090") is not None
        assert gpu_class_rank("RTX 4090") < gpu_class_rank("RTX 3090")
        assert gpu_class_rank("A6000") < gpu_class_rank("RTX 3080")

    def test_variant_prefix_matches_base_class(self):
        assert gpu_class_rank("RTX 3090 SUPER") == gpu_class_rank("RTX 3090")
        assert gpu_class_rank("RTX 3090 Ti") < gpu_class_rank("RTX 3090")

    def test_unknown_gpu_returns_none(self):
        assert gpu_class_rank("NoSuchGPU 9000") is None

    def test_ranking_descending(self):
        ranks = [gpu_class_rank(g) for g in GPU_CLASS_RANKING]
        assert ranks == list(range(len(GPU_CLASS_RANKING)))


class TestFilterOffersByGpuClass:
    def test_filters_below_min_class(self):
        offers = [_offer(gpu_name="RTX 4090", id=1), _offer(gpu_name="RTX 3070", id=2)]
        out = filter_offers_by_gpu_class(offers, min_class="RTX 3090")
        assert [o["id"] for o in out] == [1]

    def test_keeps_min_class_and_better(self):
        offers = [
            _offer(gpu_name="RTX 3090", id=1),
            _offer(gpu_name="RTX 3090 Ti", id=2),
            _offer(gpu_name="A6000", id=3),
        ]
        out = filter_offers_by_gpu_class(offers, min_class="RTX 3090")
        assert {o["id"] for o in out} == {1, 2, 3}

    def test_min_vram_and_price_caps(self):
        offers = [
            _offer(id=1, gpu_ram=16384, dph_total=0.2),
            _offer(id=2, gpu_ram=8192, dph_total=0.1),
            _offer(id=3, gpu_ram=24576, dph_total=1.5),
        ]
        out = filter_offers_by_gpu_class(
            offers, min_class="RTX 3090", min_vram_gb=16.0, max_dph=0.5
        )
        assert [o["id"] for o in out] == [1]

    def test_unknown_gpu_excluded_by_default(self):
        offers = [_offer(gpu_name="Mystery GPU", id=1)]
        assert filter_offers_by_gpu_class(offers, min_class="RTX 3090") == []
        assert (
            len(
                filter_offers_by_gpu_class(
                    offers, min_class="RTX 3090", include_unknown=True
                )
            )
            == 1
        )

    def test_invalid_min_class_raises(self):
        with pytest.raises(ValueError):
            filter_offers_by_gpu_class([], min_class="NotAGPU")


# ── Client with mocked transport ─────────────────────────────────────


def _mock_transport(handler):
    return httpx.MockTransport(handler)


class TestVastClient:
    def test_missing_api_key_raises(self):
        with pytest.raises(VastError):
            VastClient("")

    def test_search_offers_builds_query(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            assert request.url.host == "console.vast.ai"
            return httpx.Response(200, json={"offers": [_offer()]})

        client = VastClient("test-key", transport=_mock_transport(handler))
        offers = client.search_offers(min_vram_gb=16.0, per_page=10)
        assert len(offers) == 1
        assert captured["auth"] == "Bearer test-key"
        params = captured["url"]
        assert "gpu_ram" in params and "verified" in params
        assert "type=on-demand" in params
        client.close()

    def test_search_offers_http_error_raises_vast_error(self):
        def handler(request: httpx.Request):
            return httpx.Response(401, json={"msg": "Unauthorized"})

        client = VastClient("bad-key", transport=_mock_transport(handler))
        with pytest.raises(VastError, match="401"):
            client.search_offers()
        client.close()

    def test_launch_instance_posts_launch_body(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "new_contract": 777})

        client = VastClient("k", transport=_mock_transport(handler))
        instance_id = client.launch_instance(
            ask_id=1001,
            image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            env={"VAST_API_PORT": "8001"},
            disk_gb=60,
            onstart="echo hi",
        )
        assert instance_id == 777
        assert captured["method"] == "POST"
        assert captured["path"] == "/api/v0/asks/1001/"
        assert captured["body"]["client_id"] == "me"
        assert captured["body"]["disk"] == 60
        assert captured["body"]["env"]["VAST_API_PORT"] == "8001"
        assert captured["body"]["onstart"] == "echo hi"
        assert captured["body"]["extra"] == {"ssh": True}
        client.close()

    def test_launch_failure_raises(self):
        def handler(request: httpx.Request):
            return httpx.Response(200, json={"success": False, "error": "no funds"})

        client = VastClient("k", transport=_mock_transport(handler))
        with pytest.raises(VastError, match="no funds"):
            client.launch_instance(ask_id=1, image="img")
        client.close()

    def test_list_and_get_instance(self):
        inst = {
            "id": 555,
            "actual_status": "running",
            "gpu_name": "RTX 4090",
            "public_ipaddr": "1.2.3.4",
            "ports": {"8001/tcp": [{"HostPort": 41234}]},
        }

        def handler(request: httpx.Request):
            return httpx.Response(200, json={"instances": [inst]})

        client = VastClient("k", transport=_mock_transport(handler))
        assert client.list_instances() == [inst]
        assert client.get_instance(555) == inst
        assert client.get_instance(1) is None
        assert client.remote_api_url(inst) == "https://1.2.3.4:41234"
        client.close()

    def test_destroy_instance(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["method"] = request.method
            captured["path"] = request.url.path
            return httpx.Response(200, json={"success": True})

        client = VastClient("k", transport=_mock_transport(handler))
        assert client.destroy_instance(555) is True
        assert captured["method"] == "DELETE"
        assert captured["path"] == "/api/v0/instances/555/"
        client.close()

    def test_wait_until_running_polls(self):
        calls = {"n": 0}

        def handler(request: httpx.Request):
            calls["n"] += 1
            status = "running" if calls["n"] >= 2 else "loading"
            return httpx.Response(
                200, json={"instances": [{"id": 9, "actual_status": status}]}
            )

        client = VastClient("k", transport=_mock_transport(handler))
        inst = client.wait_until_running(9, timeout=60, interval=0.05)
        assert inst["actual_status"] == "running"
        assert calls["n"] == 2
        client.close()

    def test_wait_until_running_timeout(self):
        def handler(request: httpx.Request):
            return httpx.Response(
                200, json={"instances": [{"id": 9, "actual_status": "loading"}]}
            )

        client = VastClient("k", transport=_mock_transport(handler))
        with pytest.raises(VastError, match="did not start"):
            client.wait_until_running(9, timeout=0.3, interval=0.1)
        client.close()

    def test_non_json_response_raises(self):
        def handler(request: httpx.Request):
            return httpx.Response(502, text="<html>bad gateway</html>")

        client = VastClient("k", transport=_mock_transport(handler))
        with pytest.raises(VastError):
            client.search_offers()
        client.close()


class TestBuildOnstart:
    def test_requires_repo_url(self):
        with pytest.raises(VastError):
            build_kodaquant_onstart("")

    def test_script_contains_stack_bootstrap(self):
        script = build_kodaquant_onstart("https://github.com/example/kodaquant.git")
        assert "get.docker.com" in script
        assert "docker compose up -d api worker redis" in script
        assert "https://github.com/example/kodaquant.git" in script
