"""Common schema types."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    redis: str
    db_rows: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class LicenseStatusResponse(BaseModel):
    plan: str
    licensed: bool
    trial_active: bool
    trial_days_left: int
    license_key: str = ""
    activation_id: str = ""
    expires_at: str = ""
    last_verified: str = ""
    machine_id: str = ""
    needs_activation: bool
    available_models: List[str] = []
    locked_models: List[str] = []


class ActivateRequest(BaseModel):
    license_key: str


class FeatureCheckResponse(BaseModel):
    plan: str
    feature: Optional[str] = None
    feature_allowed: Optional[bool] = None
    model: Optional[str] = None
    model_allowed: Optional[bool] = None
