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
