"""Deterministic greedy proposer for the Factory loop.

Suggests actions without an LLM:
  - Find the weakest regime
  - Replace the worst model in that regime with the best available alternative
  - Escalate to add/remove when swaps are exhausted
  - Halt when no improvement is possible
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.committee.factory_state import FactoryState


@dataclass
class ActionProposal:
    type: str                     # swap_model, add_model, remove_model, halt
    regime: str = ""
    model_add: str = ""
    model_remove: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "regime": self.regime,
            "model_add": self.model_add,
            "model_remove": self.model_remove,
            "rationale": self.rationale,
        }

    @classmethod
    def halt(cls) -> "ActionProposal":
        return cls(type="halt", rationale="No more untested improvements available")

    @classmethod
    def swap(cls, regime: str, remove: str, add: str, worst_s: float,
             best_s: float) -> "ActionProposal":
        return cls(
            type="swap_model",
            regime=regime,
            model_remove=remove,
            model_add=add,
            rationale=(f"{regime}: swap {remove} (Sharpe={worst_s:.3f}) "
                       f"for {add} (Sharpe={best_s:.3f})"),
        )

    @classmethod
    def add(cls, regime: str, add: str, best_s: float) -> "ActionProposal":
        return cls(
            type="add_model",
            regime=regime,
            model_add=add,
            rationale=f"{regime}: add {add} (Sharpe={best_s:.3f}) for diversification",
        )

    @classmethod
    def remove(cls, regime: str, remove: str, worst_s: float) -> "ActionProposal":
        return cls(
            type="remove_model",
            regime=regime,
            model_remove=remove,
            rationale=f"{regime}: remove {remove} (Sharpe={worst_s:.3f}) — redundant",
        )


class DeterministicProposer:
    def __init__(self, min_sharpe_delta: float = 0.02):
        self.min_sharpe_delta = min_sharpe_delta

    def propose(self, state: FactoryState) -> ActionProposal:
        if state.matrix is None or not state.matrix.models:
            return ActionProposal.halt()

        # ── Strategy 1: Swap worst model in weakest regime ──
        weak_regime = state.weakest_regime()
        if weak_regime is None:
            return ActionProposal.halt()

        worst_model = state.worst_model_in_regime(weak_regime)
        best_candidate = state.best_candidate_for_regime(weak_regime)

        if worst_model is not None and best_candidate is not None:
            worst_s = state._model_sharpe_in_regime(weak_regime, worst_model)
            best_s = state._model_sharpe_in_regime(weak_regime, best_candidate)
            if np.isfinite(worst_s) and np.isfinite(best_s):
                if best_s > worst_s + self.min_sharpe_delta:
                    return ActionProposal.swap(
                        weak_regime, worst_model, best_candidate, worst_s, best_s)

        # ── Strategy 2: Add model to diversify a regime with only 1 model ──
        for regime_name in sorted(state.config.regimes.keys()):
            current = state._regime_models_for(regime_name)
            if current and len(current) == 1:
                existing_s = state._model_sharpe_in_regime(regime_name, current[0])
                best = state.best_candidate_for_regime(regime_name)
                if best is not None:
                    best_s = state._model_sharpe_in_regime(regime_name, best)
                    if (np.isfinite(best_s) and np.isfinite(existing_s)
                            and best_s > existing_s + self.min_sharpe_delta):
                        return ActionProposal.add(regime_name, best, best_s)

        # ── Strategy 3: Remove redundant model from regime with 3+ models ──
        for regime_name in sorted(state.config.regimes.keys()):
            current = state._regime_models_for(regime_name)
            if current and len(current) >= 3:
                worst = state.worst_model_in_regime(regime_name)
                if worst is not None:
                    worst_s = state._model_sharpe_in_regime(regime_name, worst)
                    # Only remove if very subpar
                    if np.isfinite(worst_s) and worst_s < -0.1:
                        return ActionProposal.remove(regime_name, worst, worst_s)

        return ActionProposal.halt()
