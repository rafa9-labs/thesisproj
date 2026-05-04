"""Paddle API client for license verification and activation.

Paddle v3 License API:
  - POST /api/license/verify    → check if a license key is valid
  - POST /api/license/activate  → activate a license on this machine
  - POST /api/license/deactivate → deactivate (for machine transfer)

Offline fallback:
  - Last valid verification is stored locally (encrypted in SecureStorage)
  - If Paddle API is unreachable AND within 7-day grace period → allow
  - If grace period exceeded → block until online verification succeeds
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_PADDLE_BASE = "https://v3.paddlespecial.com"
_PADDLE_SANDBOX_BASE = "https://v3-sandbox.paddlespecial.com"
_GRACE_PERIOD_DAYS = 7
_VERIFY_TIMEOUT = 10


class PaddleLicenseError(Exception):
    pass


class VerificationResult:
    __slots__ = ("valid", "activation_id", "expires_at", "plan", "features", "raw")

    def __init__(
        self,
        valid: bool,
        activation_id: str = "",
        expires_at: str = "",
        plan: str = "free",
        features: Optional[Dict[str, bool]] = None,
        raw: Optional[Dict[str, Any]] = None,
    ):
        self.valid = valid
        self.activation_id = activation_id
        self.expires_at = expires_at
        self.plan = plan
        self.features = features or {}
        self.raw = raw or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "activation_id": self.activation_id,
            "expires_at": self.expires_at,
            "plan": self.plan,
            "features": self.features,
        }


class ActivationResult:
    __slots__ = ("success", "activation_id", "expires_at", "error")

    def __init__(
        self,
        success: bool,
        activation_id: str = "",
        expires_at: str = "",
        error: str = "",
    ):
        self.success = success
        self.activation_id = activation_id
        self.expires_at = expires_at
        self.error = error


class PaddleClient:
    def __init__(
        self,
        vendor_id: Optional[str] = None,
        product_id: Optional[str] = None,
        api_key: Optional[str] = None,
        sandbox: bool = False,
    ):
        self.vendor_id = vendor_id or os.environ.get("PADDLE_VENDOR_ID", "")
        self.product_id = product_id or os.environ.get("PADDLE_PRODUCT_ID", "")
        self.api_key = api_key or os.environ.get("PADDLE_API_KEY", "")
        self.sandbox = sandbox or os.environ.get("PADDLE_SANDBOX", "").lower() in ("1", "true")
        self._base = _PADDLE_SANDBOX_BASE if self.sandbox else _PADDLE_BASE
        self._http = httpx.Client(timeout=_VERIFY_TIMEOUT)

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def is_configured(self) -> bool:
        return bool(self.vendor_id and self.product_id and self.api_key)

    async def verify_license(
        self,
        license_key: str,
        machine_id: str,
    ) -> VerificationResult:
        if not self.is_configured():
            logger.warning("Paddle not configured — skipping online verification")
            return VerificationResult(valid=False, plan="free")

        payload = {
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "license_key": license_key,
            "machine_id": machine_id,
        }
        try:
            resp = await self._post("/api/license/verify", payload)
        except Exception as e:
            logger.warning("Paddle verify request failed: %s", e)
            return VerificationResult(valid=False, plan="free")

        if resp.get("response", {}).get("type") == "error":
            logger.info("Paddle verify returned error: %s", resp)
            return VerificationResult(valid=False, plan="free", raw=resp)

        data = resp.get("response", {})
        valid = data.get("state", "") == "active"
        activation_id = data.get("activation_id", "")
        expires_at = data.get("expires_at", "")
        plan = self._infer_plan(data)
        features = data.get("features", {})

        return VerificationResult(
            valid=valid,
            activation_id=activation_id,
            expires_at=expires_at,
            plan=plan,
            features=features,
            raw=data,
        )

    async def activate_license(
        self,
        license_key: str,
        machine_id: str,
        instance_name: str = "",
    ) -> ActivationResult:
        if not self.is_configured():
            return ActivationResult(success=False, error="Paddle not configured")

        payload = {
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "license_key": license_key,
            "machine_id": machine_id,
            "instance_name": instance_name or f"kodaquant-{machine_id[:8]}",
        }
        try:
            resp = await self._post("/api/license/activate", payload)
        except Exception as e:
            return ActivationResult(success=False, error=str(e))

        if resp.get("response", {}).get("type") == "error":
            err_msg = resp.get("response", {}).get("error", {}).get("message", "Unknown error")
            return ActivationResult(success=False, error=err_msg)

        data = resp.get("response", {})
        return ActivationResult(
            success=True,
            activation_id=data.get("activation_id", ""),
            expires_at=data.get("expires_at", ""),
        )

    async def deactivate_license(
        self,
        license_key: str,
        activation_id: str,
    ) -> bool:
        if not self.is_configured():
            return False

        payload = {
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "license_key": license_key,
            "activation_id": activation_id,
        }
        try:
            resp = await self._post("/api/license/deactivate", payload)
        except Exception as e:
            logger.warning("Paddle deactivate failed: %s", e)
            return False

        return resp.get("response", {}).get("type") != "error"

    def check_grace_period(self, last_verified: str) -> bool:
        try:
            last = datetime.fromisoformat(last_verified)
            return datetime.utcnow() - last < timedelta(days=_GRACE_PERIOD_DAYS)
        except (ValueError, TypeError):
            return False

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        url = f"{self._base}{path}"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._http.post(url, json=payload, headers=self._headers()),
        )
        resp.raise_for_status()
        return resp.json()

    def _infer_plan(self, data: Dict[str, Any]) -> str:
        info = data.get("license_key_info", {})
        product_name = info.get("product_name", "").lower()
        if "team" in product_name:
            return "team"
        if "pro" in product_name:
            return "pro"
        quantity = info.get("quantity", 1)
        if quantity and int(quantity) > 1:
            return "team"
        return "pro"

    def close(self) -> None:
        self._http.close()