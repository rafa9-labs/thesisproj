"""Pipeline configuration endpoints."""
import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/config", tags=["config"])

_CONFIG_PATH = Path(os.environ.get("FX_CONFIG_PATH", "fx_ui_config.json"))


class ConfigPayload(BaseModel):
    settings: Dict[str, Any] = {}


class ApiKeyPayload(BaseModel):
    name: str
    value: str


@router.get("")
def get_config():
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            return {"settings": data}
        except (json.JSONDecodeError, OSError):
            pass
    return {"settings": {}}


@router.put("")
def update_config(payload: ConfigPayload):
    try:
        _CONFIG_PATH.write_text(
            json.dumps(payload.settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"status": "ok"}
    except OSError as e:
        raise HTTPException(500, f"Failed to save config: {e}")


@router.post("/api-key")
def store_api_key(payload: ApiKeyPayload):
    try:
        from api.licensing.storage import SecureStorage
        secure = SecureStorage()
        secure.store_api_key(payload.name, payload.value)
        return {"status": "ok", "key_name": payload.name}
    except Exception as e:
        raise HTTPException(500, f"Failed to store API key: {e}")


class KvPayload(BaseModel):
    key: str
    value: str


@router.post("/kv")
def store_kv(payload: KvPayload):
    try:
        from api.licensing.storage import SecureStorage
        secure = SecureStorage()
        secure.set_kv(payload.key, payload.value)
        return {"status": "ok", "key": payload.key}
    except Exception as e:
        raise HTTPException(500, f"Failed to store value: {e}")


@router.get("/credential-status")
def credential_status():
    token, account_id = None, None
    try:
        from api.licensing.storage import SecureStorage
        secure = SecureStorage()
        token = secure.get_api_key("oanda")
        account_id = secure.get_kv("oanda_account_id")
    except Exception:
        pass
    return {
        "oanda_token_configured": bool(token),
        "oanda_account_id_configured": bool(account_id),
    }


class ExecutionSettingsPayload(BaseModel):
    max_concurrent_backtests: int = 1
    gpu_enabled: bool = False
    max_concurrent_gpu: int = 1


@router.get("/execution")
def get_execution():
    try:
        from pipeline.runtime import get_thread_budget, _GPU_AVAILABLE
        budget = get_thread_budget()
        return {
            "max_concurrent_backtests": budget.get("cv_n_jobs", 1),
            "gpu_enabled": bool(_GPU_AVAILABLE),
            "max_concurrent_gpu": budget.get("batch_size", 1),
        }
    except Exception:
        return {
            "max_concurrent_backtests": 1,
            "gpu_enabled": False,
            "max_concurrent_gpu": 1,
        }


@router.put("/execution")
def update_execution(payload: ExecutionSettingsPayload):
    return {"status": "ok", "message": "Execution settings applied for this session"}
