"""Pipeline configuration endpoints."""
import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/config", tags=["config"])

_CONFIG_PATH = Path(os.environ.get("FX_CONFIG_PATH", "fx_ui_config.json"))


class ConfigPayload(BaseModel):
    settings: Dict[str, Any] = {}


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
        from fastapi import HTTPException
        raise HTTPException(500, f"Failed to save config: {e}")
