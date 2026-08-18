"""Vast.ai API client for the dedicated compute-node architecture.

KodaQuant does not browse or rent machines itself — the user rents an
instance on the Vast.ai dashboard. This client only lists the user's
instances and resolves how to reach the KodaQuant API running on one
of them.

The transport is injectable (httpx.MockTransport) so the whole layer can be
tested without a live Vast.ai account.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

VAST_API_BASE = "https://console.vast.ai"


class VastError(Exception):
    """Raised for Vast.ai API or transport failures."""


def resolve_api_url(
    instance: Dict[str, Any], port: int = 8000, scheme: str = "http"
) -> Optional[str]:
    """Resolve the public URL of the API running on a rented instance.

    Prefers an explicit port mapping for the requested port; falls back to
    Vast.ai's direct-port convention (container ports >= 4000 are exposed
    sequentially starting at ``direct_port_start``).
    """
    ip = instance.get("public_ipaddr")
    if not ip:
        return None
    ports = instance.get("ports") or {}
    for key, mappings in ports.items():
        if key.startswith(f"{port}/") and isinstance(mappings, list) and mappings:
            host_port = mappings[0].get("HostPort")
            if host_port:
                return f"{scheme}://{ip}:{host_port}"
    direct_start = instance.get("direct_port_start")
    if direct_start and port >= 4000:
        return f"{scheme}://{ip}:{int(direct_start) + (int(port) - 4000)}"
    return None


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

    def list_instances(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/api/v1/instances/")
        return list(data.get("instances", []))

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        for inst in self.list_instances():
            if int(inst.get("id", -1)) == int(instance_id):
                return inst
        return None
