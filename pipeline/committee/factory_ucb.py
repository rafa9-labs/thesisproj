"""
UCB1 Proposer for Phase 6 Factory optimization.

Implements the Upper Confidence Bound 1 (UCB1) Multi-Armed Bandit algorithm
to balance exploration and exploitation when proposing model swaps in the
committee factory optimization loop.

Each arm represents a candidate action (swap/add/remove model in a regime).
UCB1 tracks the running mean Sharpe improvement of each arm and selects the
arm with the highest upper confidence bound:

    UCB_j = ^mu_j + c * sqrt(2 * ln(t) / n_j)

where:
  ^mu_j = running mean Sharpe delta for arm j
  c     = exploration constant (default 2.0)
  t     = total iterations
  n_j   = number of times arm j was pulled

Reference:
  Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
  Finite-time Analysis of the Multiarmed Bandit Problem.
  Machine Learning, 47(2), 235-256.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.committee.factory_proposer import ActionProposal


def _arm_hash(regime: str, action: str, model_remove: str, model_add: str) -> str:
    raw = f"{regime}|{action}|{model_remove}|{model_add}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class UCBArm:
    arm_hash: str
    action_type: str
    regime: str
    model_add: str
    model_remove: str
    running_mean: float = 0.0
    pull_count: int = 0
    last_tested: int = -1

    @classmethod
    def from_proposal(cls, proposal: ActionProposal) -> "UCBArm":
        return cls(
            arm_hash=_arm_hash(
                proposal.regime, proposal.type,
                proposal.model_remove, proposal.model_add,
            ),
            action_type=proposal.type,
            regime=proposal.regime,
            model_add=proposal.model_add,
            model_remove=proposal.model_remove,
        )

    def to_proposal(self) -> ActionProposal:
        if self.action_type == "halt":
            return ActionProposal.halt()
        return ActionProposal(
            type=self.action_type,
            regime=self.regime,
            model_add=self.model_add,
            model_remove=self.model_remove,
            rationale=f"UCB1: mean={self.running_mean:.4f} pulls={self.pull_count}",
        )


class UCB1Proposer:
    def __init__(self, c: float = 2.0):
        self.c = float(c)
        self._arms: Dict[str, UCBArm] = {}

    def load_shortlist(self, candidates: List[ActionProposal], t: int = 0):
        new_arms: Dict[str, UCBArm] = {}

        for proposal in candidates:
            if proposal.type == "halt":
                continue

            arm = UCBArm.from_proposal(proposal)

            existing = self._arms.get(arm.arm_hash)
            if existing is not None:
                new_arms[arm.arm_hash] = existing
            else:
                new_arms[arm.arm_hash] = arm

        self._arms = new_arms

    def _compute_ucb(self, arm: UCBArm, t: int) -> float:
        if arm.pull_count == 0:
            return float("inf")
        exploration = self.c * math.sqrt(2.0 * math.log(max(t, 2)) / arm.pull_count)
        return arm.running_mean + exploration

    def propose(self, state: Any = None) -> ActionProposal:
        if not self._arms:
            return ActionProposal.halt()

        t = self._total_pulls() + 1

        untested = [a for a in self._arms.values() if a.pull_count == 0]
        if untested:
            chosen = untested[0]
            chosen.pull_count += 1
            chosen.last_tested = t
            return chosen.to_proposal()

        best_arm = max(self._arms.values(), key=lambda a: self._compute_ucb(a, t))
        best_arm.pull_count += 1
        best_arm.last_tested = t
        return best_arm.to_proposal()

    def record_result(self, arm_hash: str, delta_sharpe: float):
        arm = self._arms.get(arm_hash)
        if arm is None:
            return

        n = max(arm.pull_count, 1)
        arm.running_mean = arm.running_mean + (delta_sharpe - arm.running_mean) / n

    def has_converged(self, baseline_sharpe: float = 0.0, min_delta: float = 0.005) -> bool:
        if not self._arms:
            return True

        t = max(self._total_pulls(), 2)
        best_ucb = max(self._compute_ucb(a, t) for a in self._arms.values())
        return best_ucb <= (baseline_sharpe + min_delta)

    def _total_pulls(self) -> int:
        return sum(a.pull_count for a in self._arms.values())

    @property
    def arm_stats(self) -> List[Dict[str, Any]]:
        return [
            {
                "arm_hash": a.arm_hash,
                "action": a.action_type,
                "regime": a.regime,
                "model_add": a.model_add,
                "model_remove": a.model_remove,
                "running_mean": round(a.running_mean, 6),
                "pull_count": a.pull_count,
                "last_tested": a.last_tested,
            }
            for a in self._arms.values()
        ]
