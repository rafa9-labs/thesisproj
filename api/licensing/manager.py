"""License manager — orchestrates verification, activation, trial, and feature gating.

Single entry point for all license operations.
Cached in-process for performance (refreshed periodically).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional

from api.licensing.fingerprint import machine_fingerprint
from api.licensing.gates import check_feature, check_model, get_available_models, get_locked_models
from api.licensing.paddle_client import PaddleClient
from api.licensing.storage import SecureStorage

logger = logging.getLogger(__name__)


class Plan(str, Enum):
    FREE = "free"
    TRIAL = "trial"
    PRO = "pro"
    TEAM = "team"


class LicenseStatus:
    def __init__(
        self,
        plan: str = "free",
        licensed: bool = False,
        trial_active: bool = False,
        trial_days_left: int = 0,
        license_key: str = "",
        activation_id: str = "",
        expires_at: str = "",
        last_verified: str = "",
        machine_id: str = "",
        needs_activation: bool = True,
    ):
        self.plan = plan
        self.licensed = licensed
        self.trial_active = trial_active
        self.trial_days_left = trial_days_left
        self.license_key = license_key
        self.activation_id = activation_id
        self.expires_at = expires_at
        self.last_verified = last_verified
        self.machine_id = machine_id
        self.needs_activation = needs_activation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan,
            "licensed": self.licensed,
            "trial_active": self.trial_active,
            "trial_days_left": self.trial_days_left,
            "license_key": self._mask_key(self.license_key),
            "activation_id": self.activation_id,
            "expires_at": self.expires_at,
            "last_verified": self.last_verified,
            "machine_id": self.machine_id,
            "needs_activation": self.needs_activation,
            "available_models": get_available_models(self.plan),
            "locked_models": get_locked_models(self.plan),
        }

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key or len(key) < 8:
            return ""
        return key[:4] + "****" + key[-4:]


class LicenseManager:
    _TRIAL_DAYS = 14

    def __init__(self):
        self._machine_id = machine_fingerprint()
        self._storage = SecureStorage(machine_id=self._machine_id)
        self._paddle = PaddleClient()
        self._status: Optional[LicenseStatus] = None

    def get_status(self) -> LicenseStatus:
        if self._status is not None:
            return self._status
        self._status = self._compute_status()
        return self._status

    def refresh(self) -> LicenseStatus:
        self._status = self._compute_status()
        return self._status

    def _compute_status(self) -> LicenseStatus:
        license_data = self._storage.get_license()
        if license_data and license_data.get("plan") in ("pro", "team"):
            last_verified = license_data.get("last_verified", "")
            in_grace = self._paddle.check_grace_period(last_verified) if last_verified else True
            return LicenseStatus(
                plan=license_data["plan"],
                licensed=True,
                trial_active=False,
                trial_days_left=0,
                license_key=license_data.get("license_key", ""),
                activation_id=license_data.get("activation_id", ""),
                expires_at=license_data.get("expires_at", ""),
                last_verified=last_verified,
                machine_id=self._machine_id,
                needs_activation=False,
            )

        trial = self._storage.get_trial()
        if trial and trial.get("active"):
            expires = trial.get("expires_at", "")
            days_left = self._days_until(expires)
            if days_left > 0:
                return LicenseStatus(
                    plan="trial",
                    licensed=False,
                    trial_active=True,
                    trial_days_left=days_left,
                    machine_id=self._machine_id,
                    needs_activation=False,
                )
            else:
                self._storage.end_trial()

        if license_data and license_data.get("plan") == "free":
            return LicenseStatus(
                plan="free",
                licensed=False,
                machine_id=self._machine_id,
                needs_activation=True,
            )

        return LicenseStatus(
            plan="free",
            licensed=False,
            machine_id=self._machine_id,
            needs_activation=True,
        )

    async def activate(self, license_key: str) -> Dict[str, Any]:
        result = await self._paddle.activate_license(license_key, self._machine_id)
        if not result.success:
            return {"success": False, "error": result.error}

        plan = "pro"
        self._storage.store_license(
            license_key=license_key,
            activation_id=result.activation_id,
            machine_fingerprint=self._machine_id,
            plan=plan,
            activated_at=datetime.utcnow().isoformat(),
            expires_at=result.expires_at,
        )
        self._status = self._compute_status()
        return {"success": True, "plan": plan, "activation_id": result.activation_id}

    async def verify(self) -> Dict[str, Any]:
        license_data = self._storage.get_license()
        if not license_data:
            return {"valid": False, "reason": "no_license"}

        key = license_data.get("license_key", "")
        result = await self._paddle.verify_license(key, self._machine_id)

        if result.valid:
            self._storage.update_verification()
            self._status = self._compute_status()
            return {"valid": True, "plan": result.plan}

        last_verified = license_data.get("last_verified", "")
        if last_verified and self._paddle.check_grace_period(last_verified):
            return {"valid": True, "plan": license_data.get("plan", "pro"), "grace": True}

        self._status = self._compute_status()
        return {"valid": False, "reason": "verification_failed"}

    async def deactivate(self) -> Dict[str, Any]:
        license_data = self._storage.get_license()
        if not license_data:
            return {"success": False, "error": "no_license"}

        key = license_data.get("license_key", "")
        act_id = license_data.get("activation_id", "")
        ok = await self._paddle.deactivate_license(key, act_id)
        if ok:
            self._storage.delete_license()
            self._status = self._compute_status()
            return {"success": True}
        return {"success": False, "error": "deactivation_failed"}

    def start_trial(self) -> Dict[str, Any]:
        existing = self._storage.get_trial()
        if existing and existing.get("active"):
            days_left = self._days_until(existing.get("expires_at", ""))
            if days_left > 0:
                return {"success": False, "error": "trial_already_active", "days_left": days_left}

        result = self._storage.start_trial(duration_days=self._TRIAL_DAYS)
        self._status = self._compute_status()
        return {"success": True, **result}

    def check_feature(self, feature: str) -> bool:
        status = self.get_status()
        return check_feature(feature, status.plan)

    def check_model(self, model_name: str) -> bool:
        status = self.get_status()
        return check_model(model_name, status.plan)

    @staticmethod
    def _days_until(iso_date: str) -> int:
        try:
            target = datetime.fromisoformat(iso_date)
            delta = target - datetime.utcnow()
            return max(0, int(delta.total_seconds() / 86400))
        except (ValueError, TypeError):
            return 0

    def close(self) -> None:
        self._storage.close()
        self._paddle.close()


_manager: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    global _manager
    if _manager is None:
        _manager = LicenseManager()
    return _manager