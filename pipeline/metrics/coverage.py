"""Coverage policy, confidence threshold, and target-rate helpers.

Thin re-export shim -- canonical source is utilsNoWFO.py.
Gradual migration: move function bodies here when convenient.
"""
from __future__ import annotations

from utilsNoWFO import (  # noqa: F401
    is_coverage_intent,
    freeze_confidence_threshold,
    target_coverage_policy,
    enforce_target_coverage_policy,
)