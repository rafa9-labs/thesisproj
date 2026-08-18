"""Remote execution bridge for the dedicated compute-node architecture.

Sends a backtest payload to the KodaQuant API running on a rented Vast.ai
instance and proxies its job lifecycle (submit -> status -> results).

The remote instance runs the same application stack, so the endpoints
mirror the local ones under /api/v1.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "error"}


class VastExecError(Exception):
    """Raised when remote execution fails (transport or application level)."""


def _client(
    api_url: str,
    transport: Optional[httpx.BaseTransport] = None,
    timeout: float = 30.0,
) -> httpx.Client:
    return httpx.Client(
        base_url=api_url.rstrip("/"),
        transport=transport,
        timeout=timeout,
    )


def submit_remote_backtest(
    api_url: str,
    config: Dict[str, Any],
    transport: Optional[httpx.BaseTransport] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """POST the backtest config to the remote API. Returns its submit response."""
    try:
        with _client(api_url, transport, timeout) as client:
            resp = client.post("/api/v1/backtest", json=config)
    except httpx.HTTPError as e:
        raise VastExecError(f"Remote instance unreachable ({api_url}): {e}") from e
    if resp.status_code >= 400:
        raise VastExecError(
            f"Remote rejected backtest ({resp.status_code}): {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as e:
        raise VastExecError("Remote returned non-JSON submit response") from e
    if not data.get("job_id"):
        raise VastExecError(f"Remote submit response missing job_id: {data}")
    return data


def poll_remote_status(
    api_url: str,
    job_id: str,
    transport: Optional[httpx.BaseTransport] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Fetch the remote job status (status, error, progress...)."""
    try:
        with _client(api_url, transport, timeout) as client:
            resp = client.get(f"/api/v1/backtest/{job_id}")
    except httpx.HTTPError as e:
        raise VastExecError(f"Remote instance unreachable ({api_url}): {e}") from e
    if resp.status_code >= 400:
        raise VastExecError(
            f"Remote status lookup failed ({resp.status_code}): {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as e:
        raise VastExecError("Remote returned non-JSON status response") from e


def fetch_remote_results(
    api_url: str,
    job_id: str,
    transport: Optional[httpx.BaseTransport] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Fetch the completed remote job's results (metrics/trades/charts)."""
    try:
        with _client(api_url, transport, timeout) as client:
            resp = client.get(f"/api/v1/backtest/{job_id}/results")
    except httpx.HTTPError as e:
        raise VastExecError(f"Remote instance unreachable ({api_url}): {e}") from e
    if resp.status_code >= 400:
        raise VastExecError(
            f"Remote results lookup failed ({resp.status_code}): {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as e:
        raise VastExecError("Remote returned non-JSON results response") from e
