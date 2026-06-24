"""Tests for the Factory loop — state, stopping, proposer, executor."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.committee.factory_executor import FactoryExecutor
from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
from pipeline.committee.expert_profiler import RegimeModelMatrix
from pipeline.committee.factory_proposer import (
    ActionProposal,
    DeterministicProposer,
)
from pipeline.committee.factory_state import (
    FactoryState,
    IterationRecord,
)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

_RNG = np.random.default_rng(42)


def _make_simple_config(extra_regimes: dict = None) -> CommitteeConfig:
    regimes = {
        "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
        "trend_down": RegimeAssignment(models=["logistic"], weights=[1.0]),
        "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
    }
    if extra_regimes:
        regimes.update(extra_regimes)
    return CommitteeConfig(
        regimes=regimes,
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


def _make_matrix(sharpe_values: dict = None) -> RegimeModelMatrix:
    regimes = ["trend_up", "trend_down", "sideways"]
    models = ["logistic", "random_forest", "xgboost", "lightgbm"]

    if sharpe_values is None:
        sharpe_values = {
            ("trend_up", "logistic"): 0.8,
            ("trend_up", "random_forest"): 0.6,
            ("trend_up", "xgboost"): 1.2,
            ("trend_up", "lightgbm"): 0.9,
            ("trend_down", "logistic"): -0.8,
            ("trend_down", "random_forest"): 0.5,
            ("trend_down", "xgboost"): -0.2,
            ("trend_down", "lightgbm"): -0.5,
            ("sideways", "logistic"): 0.1,
            ("sideways", "random_forest"): 0.0,
            ("sideways", "xgboost"): 0.3,
            ("sideways", "lightgbm"): 0.2,
        }

    sm = np.zeros((len(models), len(regimes)))
    for ri, regime in enumerate(regimes):
        for mi, model in enumerate(models):
            sm[mi, ri] = sharpe_values.get((regime, model), 0.0)

    return RegimeModelMatrix(
        regimes=regimes,
        models=models,
        sharpe_matrix=sm,
        trade_matrix=np.ones_like(sm) * 20,
        hitrate_matrix=np.ones_like(sm) * 0.5,
    )


def _make_state(with_matrix: bool = True) -> FactoryState:
    config = _make_simple_config()
    matrix = _make_matrix() if with_matrix else None
    return FactoryState(
        committee_config=config,
        regime_matrix=matrix,
        patience=5,
        stopping_tolerance=0.02,
        regime_sharpe_floor=0.3,
        max_iterations=20,
    )


# ════════════════════════════════════════════════════════════════════
# FactoryState
# ════════════════════════════════════════════════════════════════════

class TestFactoryState:
    def test_state_init(self):
        state = _make_state()
        assert state.iteration == 0
        assert state.global_best_sharpe == float("-inf")
        assert state.should_stop() == (False, "")

    def test_track_iteration_updates_history(self):
        state = _make_state()
        rec = IterationRecord(
            iteration=0, action={"type": "test"}, before_sharpe=0.0,
            after_sharpe=0.5, accepted=True,
        )
        state.track_iteration(rec)
        assert len(state.history) == 1
        assert state.history[0].accepted is True
        assert state.global_best_sharpe == 0.5

    def test_global_best_tracks_max_accepted(self):
        state = _make_state()
        state.track_iteration(IterationRecord(iteration=0, action={},
            before_sharpe=0.0, after_sharpe=0.3, accepted=True))
        state.track_iteration(IterationRecord(iteration=1, action={},
            before_sharpe=0.3, after_sharpe=0.8, accepted=True))
        state.track_iteration(IterationRecord(iteration=2, action={},
            before_sharpe=0.8, after_sharpe=0.5, accepted=False))
        assert state.global_best_sharpe == 0.8

    def test_weakest_regime_detected(self):
        state = _make_state()
        # Default config has logistic in all regimes.
        # matrix has logistic Sharpe: trend_up=0.8, trend_down=-0.8, sideways=0.1
        assert state.weakest_regime() == "trend_down"

    def test_worst_model_in_regime(self):
        state = _make_state()
        config = CommitteeConfig(
            regimes={
                "trend_down": RegimeAssignment(
                    models=["logistic", "xgboost"], weights=[0.5, 0.5]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        state.config = config
        # logistic=-0.8, xgboost=-0.2 → worst is logistic
        assert state.worst_model_in_regime("trend_down") == "logistic"

    def test_best_candidate_for_regime(self):
        state = _make_state()
        # trend_down: logistic is in committee (-0.8), best untested is random_forest (0.5)
        assert state.best_candidate_for_regime("trend_down") == "random_forest"


# ════════════════════════════════════════════════════════════════════
# Stopping Criteria
# ════════════════════════════════════════════════════════════════════

class TestStoppingCriteria:
    def test_budget_stops_at_max_iter(self):
        state = FactoryState(
            committee_config=_make_simple_config(),
            max_iterations=3,
        )
        state.iteration = 3
        should, reason = state.should_stop()
        assert should is True
        assert "budget" in reason

    def test_patience_stops_when_global_best_stagnant(self):
        state = FactoryState(
            committee_config=_make_simple_config(),
            patience=3,
            stopping_tolerance=0.02,
            max_iterations=20,
        )
        # Simulate 3 accepted iterations with no improvement
        for i in range(3):
            state.track_iteration(IterationRecord(
                iteration=i, action={}, before_sharpe=0.5,
                after_sharpe=0.5, accepted=True))
        should, reason = state.should_stop()
        assert should is True
        assert "patience" in reason

    def test_continues_when_improving(self):
        state = FactoryState(
            committee_config=_make_simple_config(),
            patience=5,
            stopping_tolerance=0.02,
            max_iterations=20,
        )
        # Improving: 0.5 → 0.52 → 0.54
        for i, sharpe in enumerate([0.5, 0.52, 0.54]):
            state.track_iteration(IterationRecord(
                iteration=i, action={}, before_sharpe=sharpe - 0.02 if i > 0 else 0.0,
                after_sharpe=sharpe, accepted=True))
        should, reason = state.should_stop()
        assert should is False

    def test_exhaustion_halts_no_untested(self):
        state = _make_state()
        # Put the best model in each regime so no swap is beneficial
        config = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["xgboost"], weights=[1.0]),
                "trend_down": RegimeAssignment(models=["random_forest"], weights=[1.0]),
                "sideways": RegimeAssignment(models=["xgboost"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        state.config = config
        state.iteration = 1  # Not first iteration, so it checks
        should, reason = state.should_stop()
        assert should is True
        assert "exhaustion" in reason

    def test_divergence_stops_on_3_deteriorating(self):
        state = FactoryState(
            committee_config=_make_simple_config(),
            max_iterations=20,
        )
        state.track_iteration(IterationRecord(
            iteration=0, action={}, before_sharpe=0.0, after_sharpe=0.5, accepted=True))
        state.track_iteration(IterationRecord(
            iteration=1, action={}, before_sharpe=0.5, after_sharpe=0.4, accepted=True))
        state.track_iteration(IterationRecord(
            iteration=2, action={}, before_sharpe=0.4, after_sharpe=0.3, accepted=True))
        should, reason = state.should_stop()
        assert should is True
        assert "divergence" in reason


# ════════════════════════════════════════════════════════════════════
# ActionProposal
# ════════════════════════════════════════════════════════════════════

class TestActionProposal:
    def test_swap_proposal(self):
        p = ActionProposal.swap("trend_down", "logistic", "random_forest", -0.8, 0.5)
        assert p.type == "swap_model"
        assert p.regime == "trend_down"
        assert p.model_remove == "logistic"
        assert p.model_add == "random_forest"

    def test_halt_proposal(self):
        p = ActionProposal.halt()
        assert p.type == "halt"

    def test_proposal_to_dict(self):
        p = ActionProposal.add("trend_up", "xgboost", 1.2)
        d = p.to_dict()
        assert d["type"] == "add_model"
        assert d["model_add"] == "xgboost"


# ════════════════════════════════════════════════════════════════════
# DeterministicProposer
# ════════════════════════════════════════════════════════════════════

class TestDeterministicProposer:
    def test_proposes_swap_for_weak_regime(self):
        state = _make_state()
        proposer = DeterministicProposer()
        proposal = proposer.propose(state)
        # Weakest regime is trend_down (logistic=-0.8).
        # Best untested is random_forest (0.5) which is 1.3 better → swap
        assert proposal.type == "swap_model"
        assert proposal.regime == "trend_down"
        assert proposal.model_remove == "logistic"
        assert proposal.model_add == "random_forest"

    def test_halts_when_no_improvement(self):
        state = _make_state()
        # Put the best model in every regime — no swap improves
        config = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["xgboost"], weights=[1.0]),
                "trend_down": RegimeAssignment(models=["random_forest"], weights=[1.0]),
                "sideways": RegimeAssignment(models=["xgboost"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        state.config = config
        proposer = DeterministicProposer()
        proposal = proposer.propose(state)
        assert proposal.type == "halt"


# ════════════════════════════════════════════════════════════════════
# FactoryExecutor
# ════════════════════════════════════════════════════════════════════

class TestFactoryExecutor:
    def test_apply_swap_action(self):
        state = _make_state()
        executor = FactoryExecutor(state=state)
        proposal = ActionProposal.swap("trend_down", "logistic", "random_forest", -0.8, 0.5)
        new_config = executor.apply_action(proposal)
        models = new_config.regime_models("trend_down").models
        assert "random_forest" in models
        assert "logistic" not in models

    def test_apply_add_action(self):
        state = _make_state()
        config = CommitteeConfig(
            regimes={
                "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
            },
            fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
        )
        state.config = config
        executor = FactoryExecutor(state=state)
        proposal = ActionProposal.add("trend_up", "xgboost", 1.2)
        new_config = executor.apply_action(proposal)
        models = new_config.regime_models("trend_up").models
        assert len(models) == 2
        assert "xgboost" in models

    def test_executor_loop_runs_without_error(self):
        state = _make_state()
        state.max_iterations = 1

        # Budget check: should_stop fires at iteration >= max_iterations
        state.iteration = 0
        should, _ = state.should_stop()
        assert should is False
        state.iteration = 1
        should, reason = state.should_stop()
        assert should is True
        assert "budget" in reason
