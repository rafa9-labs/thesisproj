"""
Regime Throttle — per-regime position sizing adjustment.

Reads Phase 5 regime_coverage_report from full cycle results
and produces per-regime sizing multipliers for live trading.

Three tiers derived from Phase 5 WFO data:
  covered + Sharpe > 0   → 1.0  ("full")    — proven profitable
  covered + Sharpe <= 0  → 0.5  ("half")    — enough data, negative edge
  not covered             → 0.0  ("observe") — insufficient trades / unknown
"""
from __future__ import annotations

from typing import Dict


class RegimeThrottle:
    """Per-regime multiplier based on Phase 5 coverage report.

    Parameters
    ----------
    coverage : dict
        Phase 5 regime_coverage_report from results.json.
        Format: {"regime_name": {"sharpe": float, "trades": int, "covered": bool}}
    committee_regimes : set[str]
        All regime names in the committee config (ensures coverage for
        every regime, defaulting unknown ones to 1.0).
    """

    def __init__(self, coverage: dict, committee_regimes: set[str]):
        self._throttles: Dict[str, float] = {}

        for regime, data in coverage.items():
            if not isinstance(data, dict):
                continue
            if data.get("covered", False) and data.get("sharpe", 0) > 0:
                self._throttles[regime] = 1.0
            elif data.get("covered", False):
                self._throttles[regime] = 0.5
            else:
                self._throttles[regime] = 0.0

        for regime in committee_regimes:
            if regime not in self._throttles:
                self._throttles[regime] = 1.0

    def get_multiplier(self, regime: str) -> float:
        """Return 0.0–1.0 sizing multiplier for this regime."""
        return self._throttles.get(regime, 1.0)

    def get_throttle_level(self, regime: str) -> str:
        """Return 'full', 'half', or 'observe' for this regime."""
        m = self.get_multiplier(regime)
        if m >= 1.0:
            return "full"
        if m >= 0.5:
            return "half"
        return "observe"

    @property
    def any_throttled(self) -> bool:
        return any(m < 1.0 for m in self._throttles.values())

    @property
    def all_observe(self) -> bool:
        return bool(self._throttles) and all(
            m == 0.0 for m in self._throttles.values()
        )

    def to_dict(self) -> dict:
        return {
            r: {"multiplier": m, "level": self.get_throttle_level(r)}
            for r, m in self._throttles.items()
        }
