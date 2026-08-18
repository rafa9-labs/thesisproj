"""Vast.ai dedicated compute-node endpoints.

The user rents machines on the Vast.ai dashboard; KodaQuant only lists
running instances and dispatches backtest payloads to them. The API key
is stored encrypted via SecureStorage (never returned by any endpoint).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.config import settings
from api.services.vast_client import (
    VastClient,
    VastError,
    resolve_api_url,
)
from api.services.vast_executor import (
    TERMINAL_STATUSES,
    VastExecError,
    fetch_remote_results,
    poll_remote_status,
    submit_remote_backtest,
)

router = APIRouter(prefix="/vast", tags=["vast"])

_EXEC_PATH = Path(os.environ.get("FX_EXEC_CONFIG_PATH", "fx_exec_config.json"))
_VAST_KEY_NAME = "vast"


# ── Schemas ──────────────────────────────────────────────────────────


class VastSettingsPayload(BaseModel):
    vast_remote_port: Optional[int] = None


class ApiKeyPayload(BaseModel):
    value: str


class RunBacktestPayload(BaseModel):
    instance_id: int
    config: Dict[str, Any] = {}


# ── Key + settings helpers ───────────────────────────────────────────


def _get_api_key() -> Optional[str]:
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key
    try:
        from api.licensing.storage import SecureStorage
        return SecureStorage().get_api_key(_VAST_KEY_NAME)
    except Exception:
        return None


def _get_client() -> VastClient:
    key = _get_api_key()
    if not key:
        raise HTTPException(400, "Vast.ai API key not configured")
    try:
        return VastClient(key)
    except VastError as e:
        raise HTTPException(400, str(e))


def _persist_settings(payload: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if _EXEC_PATH.exists():
        try:
            data = json.loads(_EXEC_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data.update(payload)
    _EXEC_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_instance_or_404(client: VastClient, instance_id: int) -> Dict[str, Any]:
    try:
        inst = client.get_instance(instance_id)
    except VastError as e:
        raise HTTPException(502, f"Vast.ai instance lookup failed: {e}")
    if inst is None:
        raise HTTPException(404, f"Instance {instance_id} not found")
    return inst


def _mirror_local_job(remote_job_id: str, config: Dict[str, Any]) -> None:
    """Record the remote run in the local jobs table so the Results page lists it."""
    try:
        from api.services import JobManager
        from pipeline.data.data_sqlite import DataStore
        jm = JobManager(DataStore(settings.db_full_path))
        jm.create_job(remote_job_id, "backtest", config)
        jm.update_status(remote_job_id, "running")
    except Exception:
        pass


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/settings")
def get_vast_settings():
    return {
        "vast_remote_port": settings.vast_remote_port,
        "has_api_key": bool(_get_api_key()),
    }


@router.put("/settings")
def update_vast_settings(payload: VastSettingsPayload):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    for k, v in updates.items():
        setattr(settings, k, v)
    try:
        _persist_settings(updates)
    except OSError as e:
        raise HTTPException(500, f"Failed to save vast settings: {e}")
    return {"status": "ok", **updates}


@router.post("/api-key")
def store_api_key(payload: ApiKeyPayload):
    value = payload.value.strip()
    if not value:
        raise HTTPException(400, "API key must not be empty")
    try:
        from api.licensing.storage import SecureStorage
        SecureStorage().store_api_key(_VAST_KEY_NAME, value)
    except Exception as e:
        raise HTTPException(500, f"Failed to store API key: {e}")
    return {"status": "ok", "has_api_key": True}


@router.delete("/api-key")
def delete_api_key():
    try:
        from api.licensing.storage import SecureStorage
        SecureStorage().store_api_key(_VAST_KEY_NAME, "")
    except Exception as e:
        raise HTTPException(500, f"Failed to clear API key: {e}")
    return {"status": "ok", "has_api_key": False}


@router.get("/instances")
def list_running_instances():
    """Active (running) instances rented on the user's Vast.ai account."""
    client = _get_client()
    try:
        instances = client.list_instances()
    except VastError as e:
        raise HTTPException(502, f"Vast.ai instances lookup failed: {e}")
    finally:
        client.close()
    running = []
    for inst in instances:
        if str(inst.get("actual_status", "")).lower() != "running":
            continue
        running.append(
            {
                "id": inst.get("id"),
                "gpu_name": inst.get("gpu_name"),
                "dph_total": inst.get("dph_total"),
                "public_ipaddr": inst.get("public_ipaddr"),
                "api_url": resolve_api_url(inst, settings.vast_remote_port),
            }
        )
    return {"instances": running}


@router.post("/run-backtest")
def run_backtest(payload: RunBacktestPayload):
    """Submit the backtest config to the selected instance's KodaQuant API."""
    client = _get_client()
    try:
        inst = _get_instance_or_404(client, payload.instance_id)
        if str(inst.get("actual_status", "")).lower() != "running":
            raise HTTPException(400, f"Instance {payload.instance_id} is not running")
        api_url = resolve_api_url(inst, settings.vast_remote_port)
        if not api_url:
            raise HTTPException(
                400,
                f"Instance {payload.instance_id} has no reachable API port "
                f"({settings.vast_remote_port})",
            )
        remote = submit_remote_backtest(api_url, payload.config)
    except HTTPException:
        raise
    except VastExecError as e:
        raise HTTPException(502, str(e))
    except VastError as e:
        raise HTTPException(502, str(e))
    finally:
        client.close()

    remote_job_id = remote["job_id"]
    _mirror_local_job(remote_job_id, payload.config)
    return {
        "success": True,
        "job_id": remote_job_id,
        "instance_id": payload.instance_id,
        "api_url": api_url,
        "status": remote.get("status", "pending"),
    }


@router.get("/runs/{job_id}")
def get_run_status(job_id: str, instance_id: int):
    """Proxy the remote run's status; attach results once terminal."""
    client = _get_client()
    try:
        inst = _get_instance_or_404(client, instance_id)
        api_url = resolve_api_url(inst, settings.vast_remote_port)
        if not api_url:
            raise HTTPException(
                400,
                f"Instance {instance_id} has no reachable API port "
                f"({settings.vast_remote_port})",
            )
        status_data = poll_remote_status(api_url, job_id)
    except HTTPException:
        raise
    except VastExecError as e:
        raise HTTPException(502, str(e))
    except VastError as e:
        raise HTTPException(502, str(e))
    finally:
        client.close()

    status = str(status_data.get("status", "")).lower()
    results: Optional[Dict[str, Any]] = None
    if status in TERMINAL_STATUSES and status == "completed":
        try:
            results = fetch_remote_results(api_url, job_id)
        except VastExecError as e:
            raise HTTPException(502, str(e))
        _update_local_mirror(job_id, "completed", results)
    elif status in TERMINAL_STATUSES:
        _update_local_mirror(job_id, "failed", None, status_data.get("error"))

    return {
        "job_id": job_id,
        "instance_id": instance_id,
        "status": status_data.get("status"),
        "error": status_data.get("error"),
        "progress": status_data.get("progress"),
        "results": results,
    }


def _update_local_mirror(
    job_id: str, status: str, results: Optional[Dict[str, Any]], error: Optional[str] = None
) -> None:
    try:
        from api.services import JobManager
        from pipeline.data.data_sqlite import DataStore
        jm = JobManager(DataStore(settings.db_full_path))
        jm.update_status(job_id, status, result=results, error=error)
    except Exception:
        pass
