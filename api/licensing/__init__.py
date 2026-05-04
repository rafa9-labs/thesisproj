"""Licensing module — Paddle-based license verification and feature gating.

Public API:
    get_license_manager()  → singleton LicenseManager
    get_license_status()   → current LicenseStatus
    check_feature(feature) → bool
    check_model(model)    → bool
"""

from api.licensing.storage import SecureStorage
from api.licensing.fingerprint import machine_fingerprint
from api.licensing.paddle_client import PaddleClient
from api.licensing.gates import FREE_MODELS, FREE_EXECUTION_TYPES, LOCKED_FEATURES, check_feature, check_model
from api.licensing.manager import LicenseManager, get_license_manager, LicenseStatus, Plan

__all__ = [
    "SecureStorage",
    "machine_fingerprint",
    "PaddleClient",
    "LicenseManager",
    "LicenseStatus",
    "Plan",
    "get_license_manager",
    "check_feature",
    "check_model",
    "FREE_MODELS",
    "FREE_EXECUTION_TYPES",
    "LOCKED_FEATURES",
]