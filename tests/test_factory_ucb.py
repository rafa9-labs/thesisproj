"""Tests for pipeline/factory_ucb.py + pipeline/factory_hybrid.py."""
import math
import pytest
from unittest.mock import MagicMock, AsyncMock

from pipeline.committee.factory_ucb import UCB1Proposer, UCBArm, _arm_hash
from pipeline.committee.factory_hybrid import HybridLLMUCB1Proposer
from pipeline.committee.factory_proposer import ActionProposal


# ============================================================
# Arm hash
# ============================================================

class TestArmHash:
    def test_hash_deterministic(self):
        h1 = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        h2 = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        assert h1 == h2
        assert len(h1) == 12

    def test_hash_different_action_produces_different_hash(self):
        h1 = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        h2 = _arm_hash("trend_up", "add_model", "xgboost", "cnn")
        assert h1 != h2

    def test_hash_different_regime_produces_different_hash(self):
        h1 = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        h2 = _arm_hash("trend_down", "swap_model", "xgboost", "cnn")
        assert h1 != h2


# ============================================================
# UCBArm
# ============================================================

class TestUCBArm:
    def test_from_proposal_creates_correct_arm(self):
        prop = ActionProposal(
            type="swap_model", regime="sideways",
            model_add="svm", model_remove="logistic",
        )
        arm = UCBArm.from_proposal(prop)
        assert arm.action_type == "swap_model"
        assert arm.regime == "sideways"
        assert arm.model_add == "svm"
        assert arm.model_remove == "logistic"
        assert arm.pull_count == 0

    def test_to_proposal_roundtrips(self):
        arm = UCBArm(
            arm_hash="abc123", action_type="swap_model",
            regime="trend_up", model_add="lstm", model_remove="rf",
            running_mean=0.05, pull_count=3,
        )
        prop = arm.to_proposal()
        assert prop.type == "swap_model"
        assert prop.regime == "trend_up"
        assert prop.model_add == "lstm"
        assert "mean=0.0500" in prop.rationale
        assert "pulls=3" in prop.rationale


# ============================================================
# UCB1Proposer
# ============================================================

class TestUCB1Proposer:
    def test_halt_on_empty_arms(self):
        ucb = UCB1Proposer(c=2.0)
        proposal = ucb.propose()
        assert proposal.type == "halt"

    def test_cold_start_force_explores_untested(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
            ActionProposal("swap_model", "sideways", "svm", "logistic"),
        ]
        ucb.load_shortlist(candidates)
        proposal = ucb.propose()
        assert proposal.type != "halt"

    def test_explores_all_untested_before_repeating(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
            ActionProposal("swap_model", "trend_down", "lstm", "rf"),
            ActionProposal("add_model", "sideways", "svm", ""),
        ]
        ucb.load_shortlist(candidates)
        seen = set()
        for _ in range(3):
            prop = ucb.propose()
            seen.add(prop.model_add)
        assert len(seen) == 3

    def test_record_result_updates_running_mean(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        ucb.load_shortlist(candidates)
        arm_hash = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        ucb.propose()  # pull_count=1
        ucb.record_result(arm_hash, 0.05)
        assert abs(ucb._arms[arm_hash].running_mean - 0.05) < 0.0001

    def test_record_result_increments_pull_count(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        ucb.load_shortlist(candidates)
        arm_hash = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        for _ in range(3):
            ucb.propose()
            ucb.record_result(arm_hash, 0.01)
        assert ucb._arms[arm_hash].pull_count == 3

    def test_exploration_bonus_decreases_with_pulls(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        ucb.load_shortlist(candidates)
        arm_hash = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")

        for _ in range(3):
            ucb.propose()
            ucb.record_result(arm_hash, 0.0)

        ucb_at_1 = ucb._compute_ucb(ucb._arms[arm_hash], t=10)

        for _ in range(20):
            ucb.propose()
            ucb.record_result(arm_hash, 0.0)

        ucb_at_20 = ucb._compute_ucb(ucb._arms[arm_hash], t=10)
        assert ucb_at_20 < ucb_at_1

    def test_high_c_explores_more_arms(self):
        ucbl = UCB1Proposer(c=0.5)
        ucbe = UCB1Proposer(c=5.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", f"m{i}", "r{j}")
            for i in range(4) for j in range(4)
        ]
        ucbl.load_shortlist(candidates)
        ucbe.load_shortlist(candidates)

        for _ in range(20):
            prop_l = ucbl.propose()
            prop_e = ucbe.propose()
            if prop_l.type != "halt":
                ucbl.record_result(
                    _arm_hash(prop_l.regime, prop_l.type, prop_l.model_remove, prop_l.model_add),
                    0.01,
                )
            if prop_e.type != "halt":
                ucbe.record_result(
                    _arm_hash(prop_e.regime, prop_e.type, prop_e.model_remove, prop_e.model_add),
                    0.01,
                )

        ucbl_unique = sum(1 for a in ucbl._arms.values() if a.pull_count > 0)
        ucbe_unique = sum(1 for a in ucbe._arms.values() if a.pull_count > 0)
        assert ucbe_unique >= ucbl_unique

    def test_has_converged_true_when_all_below_baseline(self):
        ucb = UCB1Proposer(c=0.1)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        ucb.load_shortlist(candidates)
        arm_hash = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        ucb.propose()
        ucb.record_result(arm_hash, -0.05)
        ucb.propose()
        ucb.record_result(arm_hash, -0.04)
        ucb.propose()
        ucb.record_result(arm_hash, -0.03)
        assert ucb.has_converged(baseline_sharpe=1.0, min_delta=0.01)

    def test_has_converged_false_when_arm_above_baseline(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        ucb.load_shortlist(candidates)
        arm_hash = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        for _ in range(5):
            ucb.propose()
            ucb.record_result(arm_hash, 0.1)
        assert not ucb.has_converged(baseline_sharpe=0.0, min_delta=0.005)

    def test_arm_stats_returns_valid_data(self):
        ucb = UCB1Proposer(c=2.0)
        candidates = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        ucb.load_shortlist(candidates)
        ucb.propose()
        stats = ucb.arm_stats
        assert isinstance(stats, list)
        assert len(stats) == 1
        assert "arm_hash" in stats[0]
        assert stats[0]["pull_count"] == 1


# ============================================================
# HybridLLMUCB1Proposer
# ============================================================

class TestHybridLLMUCB1Proposer:
    def test_propose_calls_llm_on_first_call(self):
        llm = MagicMock()
        llm.shortlist.return_value = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        hybrid = HybridLLMUCB1Proposer(llm_proposer=llm, llm_refresh_interval=5)
        hybrid.propose(MagicMock(iteration=0))
        llm.shortlist.assert_called_once()

    def test_propose_does_not_call_llm_within_refresh_interval(self):
        llm = MagicMock()
        llm.shortlist.return_value = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        hybrid = HybridLLMUCB1Proposer(llm_proposer=llm, llm_refresh_interval=5)
        for i in range(4):
            state = MagicMock(iteration=i)
            hybrid.propose(state)
        assert llm.shortlist.call_count == 1

    def test_propose_calls_llm_after_refresh_interval(self):
        llm = MagicMock()
        llm.shortlist.return_value = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
            ActionProposal("add_model", "sideways", "svm", ""),
        ]
        hybrid = HybridLLMUCB1Proposer(llm_proposer=llm, llm_refresh_interval=3)
        for i in range(6):
            hybrid.propose(MagicMock(iteration=i))
        assert llm.shortlist.call_count >= 2

    def test_record_result_updates_ucb(self):
        llm = MagicMock()
        llm.shortlist.return_value = [
            ActionProposal("swap_model", "trend_up", "cnn", "xgboost"),
        ]
        hybrid = HybridLLMUCB1Proposer(llm_proposer=llm, llm_refresh_interval=5, c=2.0)
        hybrid.propose(MagicMock(iteration=0))
        prop = ActionProposal("swap_model", "trend_up", "cnn", "xgboost")
        hybrid.record_result(prop, 0.05)
        ah = _arm_hash("trend_up", "swap_model", "xgboost", "cnn")
        assert abs(hybrid.ucb._arms[ah].running_mean - 0.05) < 0.0001

    def test_record_result_ignores_halt(self):
        llm = MagicMock()
        llm.shortlist.return_value = []
        hybrid = HybridLLMUCB1Proposer(llm_proposer=llm, llm_refresh_interval=5)
        halt = ActionProposal.halt()
        hybrid.record_result(halt, 0.0)
        assert len(hybrid.ucb._arms) == 0

    def test_ucb_formula_matches_reference(self):
        c = 2.0
        mean = 0.1
        n_j = 4
        t = 20
        expected = mean + c * math.sqrt(2 * math.log(t) / n_j)
        ucb = UCB1Proposer(c=c)
        arm = UCBArm("test", "swap", "r", "add", "rem", running_mean=mean, pull_count=n_j)
        computed = ucb._compute_ucb(arm, t)
        assert abs(computed - expected) < 0.0001
