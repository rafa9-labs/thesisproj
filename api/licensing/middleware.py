"""FastAPI dependency for license-gated endpoints."""

from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException

from api.licensing.manager import get_license_manager, LicenseManager
from api.licensing.gates import check_feature, check_model


def _get_manager() -> LicenseManager:
    return get_license_manager()


def require_feature(feature: str) -> Callable:
    async def _check(manager: LicenseManager = Depends(_get_manager)):
        if not manager.check_feature(feature):
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature}' requires a KodaQuant Pro license",
            )
    return _check


def require_paid_model(model_name: str) -> Callable:
    async def _check(manager: LicenseManager = Depends(_get_manager)):
        if not manager.check_model(model_name):
            raise HTTPException(
                status_code=403,
                detail=f"Model '{model_name}' requires a KodaQuant Pro license",
            )
    return _check


def require_licensed() -> Callable:
    async def _check(manager: LicenseManager = Depends(_get_manager)):
        status = manager.get_status()
        if status.plan == "free" and not status.trial_active:
            raise HTTPException(
                status_code=403,
                detail="This action requires an active KodaQuant license or trial",
            )
    return _check