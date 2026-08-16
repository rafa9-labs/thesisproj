"""Vast.ai API client for renting GPU instances.

Wraps the Vast.ai public API (https://console.vast.ai/api/v0) with the
operations KodaQuant needs: offer search, instance launch/status/destroy.

The transport is injectable (httpx.MockTransport) so the whole layer can be
tested without a live Vast.ai account.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import httpx

VAST_API_BASE = "https://console.vast.ai/api/v0"

# GPU classes in descending performance order. Offer filtering keeps machines
# whose GPU name starts with an entry ranked at-or-better than the minimum.
GPU_CLASS_RANKING: List[str] = [
    "H200", "H100", "A100", "RTX 6000 ADA", "L40S", "L40", "RTX 4090",
    "A6000", "RTX 4080", "RTX 3090 Ti", "RTX 3090", "A5000", "RTX 4070 Ti",
    "RTX 4070", "RTX 3080 Ti", "RTX 3080", "A4000", "RTX 3070", "RTX 3060 Ti",
    "RTX 3060", "RTX 2080 Ti", "P6000", "T4", "P5000", "RTX 2070", "P4000",
]


def gpu_class_rank(gpu_name: str) -> Optional[int]:
    """Return the ranking index for a GPU name, or None if unknown."""
    norm = re.sub(r"\s+", " ", (gpu_name or "").strip().upper())
    for idx, entry in enumerate(GPU_CLASS_RANKING):
        entry_u = entry.upper()
        if norm == entry_u or norm.startswith(entry_u + " "):
            return idx
    return None


def filter_offers_by_gpu_class(
    offers: List[Dict[str, Any]],
    min_class: str = "RTX 3090",
    min_vram_gb: float = 0.0,
    max_dph: float = 0.0,
    include_unknown: bool = False,
) -> List[Dict[str, Any]]:
    """Filter offer dicts by GPU class (ranked), VRAM and hourly price."""
    min_rank = gpu_class_rank(min_class)
    if min_rank is None:
        raise ValueError(f"Unknown GPU class: {min_class}")
    out: List[Dict[str, Any]] = []
    for offer in offers:
        rank = gpu_class_rank(str(offer.get("gpu_name", "")))
        if rank is None and not include_unknown:
            continue
        if rank is not None and rank > min_rank:
            continue
        gpu_ram_gb = float(offer.get("gpu_ram", 0) or 0) / 1024.0
        if gpu_ram_gb < min_vram_gb:
            continue
        dph = float(offer.get("dph_total", 0) or 0)
        if max_dph > 0 and dph > max_dph:
            continue
        out.append(offer)
    return out


class VastError(Exception):
    """Raised for Vast.ai API or transport failures."""


class VastClient:
    """Thin HTTP client for the Vast.ai API v0."""

    def __init__(
        self,
        api_key: str,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = 30.0,
        base_url: str = VAST_API_BASE,
    ):
        if not api_key:
            raise VastError("Vast.ai API key is empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            transport=transport,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VastClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _request(self, method: str, path: str, **kw: Any) -> Dict[str, Any]:
        try:
            resp = self._client.request(method, path, **kw)
        except httpx.TimeoutException as e:
            raise VastError(f"Vast.ai request timed out: {path}") from e
        except httpx.HTTPError as e:
            raise VastError(f"Vast.ai request failed: {e}") from e
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.text[:200]
            except Exception:
                pass
            raise VastError(
                f"Vast.ai API error {resp.status_code} on {method} {path}: {detail}"
            )
        try:
            data = resp.json()
        except ValueError as e:
            raise VastError(f"Vast.ai returned non-JSON on {method} {path}") from e
        return data

    # ── Offers ────────────────────────────────────────────────────────

    def search_offers(
        self,
        min_vram_gb: float = 0.0,
        verified_only: bool = True,
        per_page: int = 30,
        order: str = "dph_total",
    ) -> List[Dict[str, Any]]:
        """Search on-demand offers ordered by cheapest hourly price."""
        q: Dict[str, Any] = {"rentable": {"eq": True}}
        if verified_only:
            q["verified"] = {"eq": True}
        if min_vram_gb > 0:
            q["gpu_ram"] = {"gte": int(min_vram_gb * 1024)}
        order_map = {"dph_total": [["dph_total", "asc"]], "score": [["score", "desc"]]}
        params: Dict[str, Any] = {
            "q": _json_dump(q),
            "type": "on-demand",
            "order": order_map.get(order, order_map["dph_total"]),
            "per_page": per_page,
            "disable_bundling": "true",
        }
        data = self._request("GET", "/bundles/", params=params)
        return list(data.get("offers", []))

    # ── Instances ─────────────────────────────────────────────────────

    def launch_instance(
        self,
        ask_id: int,
        image: str,
        env: Optional[Dict[str, str]] = None,
        disk_gb: int = 60,
        label: str = "kodaquant-worker",
        onstart: Optional[str] = None,
        runtype: str = "ssh",
    ) -> int:
        """Launch an instance from an offer; returns the instance (contract) id."""
        body: Dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": int(disk_gb),
            "label": label,
            "runtype": runtype,
            "extra": {"ssh": True},
        }
        if env:
            body["env"] = env
        if onstart:
            body["onstart"] = onstart
        data = self._request("POST", f"/asks/{int(ask_id)}/", json=body)
        if not data.get("success"):
            raise VastError(f"Vast.ai launch failed: {data}")
        contract_id = data.get("new_contract")
        if not contract_id:
            raise VastError(f"Vast.ai launch returned no contract id: {data}")
        return int(contract_id)

    def list_instances(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/instances/")
        return list(data.get("instances", []))

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        for inst in self.list_instances():
            if int(inst.get("id", -1)) == int(instance_id):
                return inst
        return None

    def destroy_instance(self, instance_id: int) -> bool:
        data = self._request("DELETE", f"/instances/{int(instance_id)}/")
        return bool(data.get("success"))

    def wait_until_running(
        self, instance_id: int, timeout: float = 900.0, interval: float = 10.0
    ) -> Dict[str, Any]:
        """Poll until an instance reaches a stable status; returns the instance."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            inst = self.get_instance(instance_id)
            if inst is not None:
                status = str(inst.get("actual_status", ""))
                if status in ("running", "error", "exited"):
                    return inst
            time.sleep(interval)
        raise VastError(f"Instance {instance_id} did not start within {timeout}s")

    def remote_api_url(self, instance: Dict[str, Any], port: int = 8001) -> Optional[str]:
        """Best-effort public URL for the mapped API port of an instance."""
        ports = instance.get("ports") or {}
        for key, mappings in ports.items():
            if key.startswith(f"{port}/"):
                if mappings and isinstance(mappings, list):
                    host_port = mappings[0].get("HostPort")
                    if host_port:
                        return f"https://{instance.get('public_ipaddr')}:{host_port}"
        return None


def _json_dump(obj: Any) -> str:
    import json
    return json.dumps(obj)


def build_kodaquant_onstart(repo_url: str, api_port: int = 8001) -> str:
    """Generate the onstart script that deploys the KodaQuant stack on the
    rented instance (Docker install, clone, compose up)."""
    if not repo_url:
        raise VastError("Repository URL is required for the deploy onstart script")
    return f"""#!/bin/bash
set -e
echo "[KodaQuant] bootstrapping rented instance"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl git
curl -fsSL https://get.docker.com | sh
git clone --depth 1 {repo_url} /opt/kodaquant
cd /opt/kodaquant
docker compose up -d api worker redis
echo "[KodaQuant] stack up — API mapped to public port {api_port}"
"""
