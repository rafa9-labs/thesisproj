"""
Hybrid LLM + UCB1 Proposer for Phase 6 Factory optimization.

The LLM acts as the Strategic Director — it prunes fundamentally illogical
model/regime pairings (e.g., stopping the system from trying to fit a
mean-reversion model to a breakout regime). UCB1 takes this pruned shortlist
of logical candidates and acts as the Tactical Manager, mathematically
balancing exploration and exploitation.

LLM refresh schedule: every K iterations or when UCB1 converges (all arms
below baseline Sharpe + min_delta). This prevents excessive API calls while
ensuring the shortlist stays fresh as the committee evolves.
"""
from __future__ import annotations

from typing import List

from pipeline.factory_llm import LLMProposer
from pipeline.factory_proposer import ActionProposal
from pipeline.factory_ucb import UCB1Proposer, _arm_hash


class HybridLLMUCB1Proposer:
    def __init__(
        self,
        llm_proposer: LLMProposer,
        ucb_proposer: UCB1Proposer = None,
        llm_refresh_interval: int = 5,
        c: float = 2.0,
    ):
        self.llm = llm_proposer
        self.ucb = ucb_proposer or UCB1Proposer(c=c)
        self.refresh_interval = llm_refresh_interval
        self._last_llm_refresh: int = -1
        self._shortlist: List[ActionProposal] = []

    def propose(self, state) -> ActionProposal:
        t = state.iteration if hasattr(state, "iteration") else 0
        try:
            baseline = float(getattr(state, "global_best_sharpe", 0.0))
        except (TypeError, ValueError):
            baseline = 0.0

        if (
            t - self._last_llm_refresh >= self.refresh_interval
            or self.ucb.has_converged(baseline)
            or not self._shortlist
        ):
            self._shortlist = self.llm.shortlist(state)
            self.ucb.load_shortlist(self._shortlist, t)
            self._last_llm_refresh = t

        return self.ucb.propose(state)

    def record_result(self, proposal: ActionProposal, delta_sharpe: float):
        if proposal.type == "halt":
            return
        arm_hash = _arm_hash(
            proposal.regime, proposal.type,
            proposal.model_remove, proposal.model_add,
        )
        self.ucb.record_result(arm_hash, delta_sharpe)
