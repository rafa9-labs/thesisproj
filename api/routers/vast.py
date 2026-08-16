"""Vast.ai GPU rental endpoints.

Manages rented GPU instances: offer search, one-click launch of the
KodaQuant stack, status, and teardown. The API key is stored encrypted
via SecureStorage (never returned by any endpoint).
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
    build_kodaquant_onstart,
    filter_offers_by_gpu_class,
)

router = APIRouter(prefix="/vast", tags=["vast"])

_EXEC_PATH = Path(os.environ.get("FX_EXEC_CONFIG_PATH", "fx_exec_config.json"))
_VAST_KEY_NAME = "vast"


# ── Schemas ──────────────────────────────────────────────────────────


class VastSettingsPayload(BaseModel):
    vast_enabled: Optional[bool] = None
    vast_min_gpu_class: Optional[str] = None
    vast_min_vram_gb: Optional[float] = None
    vast_max_dph: Optional[float] = None
    vast_disk_gb: Optional[int] = None
    vast_image: Optional[str] = None
    vast_repo_url: Optional[str] = None
    vast_remote_api_url: Optional[str] = None


class ApiKeyPayload(BaseModel):
    value: str


class LaunchPayload(BaseModel):
    ask_id: Optional[int] = None
    image: Optional[str] = None
    disk_gb: Optional[int] = None
    label: str = "kodaquant-worker"
    gpu_class: Optional[str] = None
    min_vram_gb: Optional[float] = None
    max_dph: Optional[float] = None
    onstart: Optional[str] = None


class OfferSummary(BaseModel):
    ask_id: int
    machine_id: Optional[int] = None
    gpu_name: str
    gpu_ram_gb: float
    dph_total: float
    dlperf: Optional[float] = None
    num_gpus: int = 1
    cpu_cores: Optional[float] = None
    reliability: Optional[float] = None


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


def _persist_vast_settings(payload: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if _EXEC_PATH.exists():
        try:
            data = json.loads(_EXEC_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data.update(payload)
    _EXEC_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _current_vast_settings() -> Dict[str, Any]:
    return {
        "vast_enabled": settings.vast_enabled,
        "vast_min_gpu_class": settings.vast_min_gpu_class,
        "vast_min_vram_gb": settings.vast_min_vram_gb,
        "vast_max_dph": settings.vast_max_dph,
        "vast_disk_gb": settings.vast_disk_gb,
        "vast_image": settings.vast_image,
        "vast_repo_url": settings.vast_repo_url,
        "vast_remote_api_url": settings.vast_remote_api_url,
        "has_api_key": bool(_get_api_key()),
    }


def _summarize_offer(offer: Dict[str, Any]) -> OfferSummary:
    gpu_ram = float(offer.get("gpu_ram", 0) or 0)
    return OfferSummary(
        ask_id=int(offer["id"]),
        machine_id=offer.get("machine_id"),
        gpu_name=str(offer.get("gpu_name", "")),
        gpu_ram_gb=round(gpu_ram / 1024.0, 1),
        dph_total=float(offer.get("dph_total", 0) or 0),
        dlperf=offer.get("dlperf"),
        num_gpus=int(offer.get("num_gpus", 1) or 1),
        cpu_cores=offer.get("cpu_cores"),
        reliability=offer.get("reliability"),
    )


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/settings")
def get_vast_settings():
    return _current_vast_settings()


@router.put("/settings")
def update_vast_settings(payload: VastSettingsPayload):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    for k, v in updates.items():
        setattr(settings, k, v)
    try:
        _persist_vast_settings(updates)
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


@router.get("/offers")
def get_offers(
    gpu_class: Optional[str] = None,
    min_vram_gb: Optional[float] = None,
    max_dph: Optional[float] = None,
    per_page: int = 30,
    include_unknown: bool = False,
):
    gpu_class = gpu_class or settings.vast_min_gpu_class
    min_vram_gb = min_vram_gb if min_vram_gb is not None else settings.vast_min_vram_gb
    max_dph = max_dph if max_dph is not None else settings.vast_max_dph
    client = _get_client()
    try:
        offers = client.search_offers(
            min_vram_gb=min_vram_gb,
            per_page=max(1, min(int(per_page), 100)),
        )
        filtered = filter_offers_by_gpu_class(
            offers,
            min_class=gpu_class,
            min_vram_gb=min_vram_gb,
            max_dph=max_dph,
            include_unknown=include_unknown,
        )
    except VastError as e:
        raise HTTPException(502, f"Vast.ai offers lookup failed: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        client.close()
    return {"offers": [_summarize_offer(o) for o in filtered]}


@router.post("/instances")
def launch_instance(payload: LaunchPayload):
    client = _get_client()
    try:
        ask_id = payload.ask_id
        if ask_id is None:
            gpu_class = payload.gpu_class or settings.vast_min_gpu_class
            min_vram_gb = (
                payload.min_vram_gb
                if payload.min_vram_gb is not None
                else settings.vast_min_vram_gb
            )
            max_dph = payload.max_dph if payload.max_dph is not None else settings.vast_max_dph
            offers = client.search_offers(min_vram_gb=min_vram_gb, per_page=50)
            candidates = filter_offers_by_gpu_class(
                offers,
                min_class=gpu_class,
                min_vram_gb=min_vram_gb,
                max_dph=max_dph,
            )
            if not candidates:
                raise HTTPException(
                    404,
                    f"No Vast.ai offers match gpu_class={gpu_class}, "
                    f"min_vram_gb={min_vram_gb}, max_dph={max_dph}",
                )
            candidates.sort(key=lambda o: float(o.get("dph_total", 0) or 0))
            ask_id = int(candidates[0]["id"])

        onstart = payload.onstart
        if onstart is None:
            if not settings.vast_repo_url:
                raise HTTPException(
                    400,
                    "vast_repo_url must be configured in rental settings "
                    "(or pass an explicit onstart script)",
                )
            onstart = build_kodaquant_onstart(settings.vast_repo_url)

        instance_id = client.launch_instance(
            ask_id=ask_id,
            image=payload.image or settings.vast_image,
            disk_gb=payload.disk_gb or settings.vast_disk_gb,
            label=payload.label,
            onstart=onstart,
        )
    except VastError as e:
        raise HTTPException(502, f"Vast.ai launch failed: {e}")
    finally:
        client.close()
    return {"success": True, "instance_id": instance_id, "ask_id": ask_id}


@router.get("/instances")
def list_instances():
    client = _get_client()
    try:
        instances = client.list_instances()
    except VastError as e:
        raise HTTPException(502, f"Vast.ai instances lookup failed: {e}")
    finally:
        client.close()
    return {
        "instances": [
            {
                "id": inst.get("id"),
                "actual_status": inst.get("actual_status"),
                "status_msg": inst.get("status_msg"),
                "gpu_name": inst.get("gpu_name"),
                "dph_total": inst.get("dph_total"),
                "ssh_host": inst.get("ssh_host"),
                "ssh_port": inst.get("ssh_port"),
                "public_ipaddr": inst.get("public_ipaddr"),
                "remote_api_url": client.remote_api_url(inst),
            }
            for inst in instances
        ]
    }


@router.get("/instances/{instance_id}")
def get_instance(instance_id: int):
    client = _get_client()
    try:
        inst = client.get_instance(instance_id)
    except VastError as e:
        raise HTTPException(502, f"Vast.ai instance lookup failed: {e}")
    finally:
        client.close()
    if inst is None:
        raise HTTPException(404, f"Instance {instance_id} not found")
    return {
        "id": inst.get("id"),
        "actual_status": inst.get("actual_status"),
        "status_msg": inst.get("status_msg"),
        "gpu_name": inst.get("gpu_name"),
        "dph_total": inst.get("dph_total"),
        "ssh_host": inst.get("ssh_host"),
        "ssh_port": inst.get("ssh_port"),
        "public_ipaddr": inst.get("public_ipaddr"),
        "remote_api_url": client.remote_api_url(inst),
    }


@router.delete("/instances/{instance_id}")
def destroy_instance(instance_id: int):
    client = _get_client()
    try:
        ok = client.destroy_instance(instance_id)
    except VastError as e:
        raise HTTPException(502, f"Vast.ai destroy failed: {e}")
    finally:
        client.close()
    if not ok:
        raise HTTPException(500, "Vast.ai reported failure destroying the instance")
    if settings.vast_remote_api_url:
        settings.vast_remote_api_url = ""
        try:
            _persist_vast_settings({"vast_remote_api_url": ""})
        except OSError:
            pass
    return {"success": True}
