"""Tests for LLM-driven Factory proposer (F4)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.committee.factory_llm import (
    PromptBuilder,
    ResponseParser,
    LLMProposer,
    create_llm_proposer,
    SYSTEM_PROMPT,
)
from pipeline.committee.factory_state import FactoryState
from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
from pipeline.committee.expert_profiler import RegimeModelMatrix

_RNG = np.random.default_rng(42)


# ════════════════════════════════════════════════════════════════════
# Helpers (same as test_factory.py)
# ════════════════════════════════════════════════════════════════════

def _make_config() -> CommitteeConfig:
    regimes = {
        "trend_up": RegimeAssignment(models=["logistic"], weights=[1.0]),
        "trend_down": RegimeAssignment(models=["logistic"], weights=[1.0]),
        "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
    }
    return CommitteeConfig(
        regimes=regimes,
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


def _make_matrix() -> RegimeModelMatrix:
    regimes = ["trend_up", "trend_down", "sideways"]
    models = ["logistic", "random_forest", "xgboost", "lightgbm"]
    sm = np.array([
        [0.8, -0.8, 0.1],
        [0.6, 0.5, 0.0],
        [1.2, -0.2, 0.3],
        [0.9, -0.5, 0.2],
    ])
    return RegimeModelMatrix(
        regimes=regimes, models=models,
        sharpe_matrix=sm,
        trade_matrix=np.ones_like(sm) * 20,
        hitrate_matrix=np.ones_like(sm) * 0.5,
    )


def _make_state() -> FactoryState:
    return FactoryState(
        committee_config=_make_config(),
        regime_matrix=_make_matrix(),
        patience=5,
        stopping_tolerance=0.02,
        regime_sharpe_floor=0.3,
        max_iterations=20,
    )


# ════════════════════════════════════════════════════════════════════
# PromptBuilder
# ════════════════════════════════════════════════════════════════════

class TestPromptBuilder:
    def test_system_prompt_present(self):
        assert "Forex trading" in SYSTEM_PROMPT
        assert "committee optimization" in SYSTEM_PROMPT.lower()
        assert "swap" in SYSTEM_PROMPT.lower()

    def test_build_returns_two_strings(self):
        state = _make_state()
        builder = PromptBuilder()
        sys_p, user_p = builder.build(state)
        assert isinstance(sys_p, str) and len(sys_p) > 100
        assert isinstance(user_p, str) and len(user_p) > 200

    def test_user_prompt_contains_matrix_table(self):
        state = _make_state()
        builder = PromptBuilder()
        _, user_p = builder.build(state)
        assert "## Regime x Model Performance Matrix" in user_p
        assert "logistic" in user_p
        assert "xgboost" in user_p
        assert "+0.80" in user_p or "0.80" in user_p

    def test_user_prompt_marks_weakest_regime(self):
        state = _make_state()
        builder = PromptBuilder()
        _, user_p = builder.build(state)
        assert "WEAKEST REGIME" in user_p

    def test_user_prompt_without_matrix(self):
        state = FactoryState(
            committee_config=_make_config(),
            regime_matrix=None,
        )
        builder = PromptBuilder()
        _, user_p = builder.build(state)
        assert "NOT AVAILABLE" in user_p


# ════════════════════════════════════════════════════════════════════
# ResponseParser
# ════════════════════════════════════════════════════════════════════

class TestResponseParser:
    def test_parses_valid_swap(self):
        state = _make_state()
        parser = ResponseParser()
        raw = json.dumps({
            "analysis": "trend_down has worst Sharpe",
            "action": {
                "type": "swap_model", "regime": "trend_down",
                "model_add": "random_forest", "model_remove": "logistic",
            },
            "confidence": "high",
            "rationale": "RF 0.5 > logistic -0.8 in trend_down",
        })
        proposal = parser.parse(raw, state)
        assert proposal.type == "swap_model"
        assert proposal.regime == "trend_down"
        assert proposal.model_add == "random_forest"
        assert proposal.model_remove == "logistic"

    def test_parses_halt(self):
        state = _make_state()
        parser = ResponseParser()
        raw = json.dumps({
            "analysis": "All regimes optimized",
            "action": {"type": "halt"},
            "confidence": "low",
            "rationale": "No improvement possible",
        })
        proposal = parser.parse(raw, state)
        assert proposal.type == "halt"

    def test_invalid_regime_halted(self):
        state = _make_state()
        parser = ResponseParser()
        raw = json.dumps({
            "action": {"type": "swap_model", "regime": "invalid_regime",
                       "model_add": "xgboost", "model_remove": "logistic"},
            "confidence": "high",
        })
        proposal = parser.parse(raw, state)
        assert proposal.type == "halt"

    def test_invalid_json_returns_halt(self):
        state = _make_state()
        parser = ResponseParser()
        proposal = parser.parse("this is not json at all", state)
        assert proposal.type == "halt"

    def test_extracts_json_from_markdown_fence(self):
        state = _make_state()
        parser = ResponseParser()
        raw = "Some text\n```json\n" + json.dumps({
            "action": {"type": "halt"}, "confidence": "low",
        }) + "\n```\nMore text"
        proposal = parser.parse(raw, state)
        assert proposal.type == "halt"

    def test_add_model_parsed(self):
        state = _make_state()
        parser = ResponseParser()
        raw = json.dumps({
            "action": {"type": "add_model", "regime": "trend_up",
                       "model_add": "xgboost"},
            "confidence": "medium",
        })
        proposal = parser.parse(raw, state)
        assert proposal.type == "add_model"
        assert proposal.model_add == "xgboost"


# ════════════════════════════════════════════════════════════════════
# LLMProposer with Mock Backend
# ════════════════════════════════════════════════════════════════════

class MockBackend:
    def __init__(self, response: str = None):
        self.response = response or json.dumps({
            "analysis": "mock",
            "action": {"type": "halt"},
            "confidence": "low",
        })
        self.calls = []

    def complete(self, sys_prompt, user_prompt):
        self.calls.append((sys_prompt, user_prompt))
        return self.response


class TestLLMProposer:
    def test_proposes_swap_from_valid_response(self):
        state = _make_state()
        backend = MockBackend(response=json.dumps({
            "analysis": "trend_down is weakest",
            "action": {
                "type": "swap_model", "regime": "trend_down",
                "model_add": "random_forest", "model_remove": "logistic",
            },
            "confidence": "high",
            "rationale": "RF has higher Sharpe",
        }))
        proposer = LLMProposer()
        proposer._llm = backend
        proposer._fallback = None

        proposal = proposer.propose(state)
        assert proposal.type == "swap_model"
        assert proposal.regime == "trend_down"
        assert proposal.model_add == "random_forest"

    def test_halts_on_halt_response(self):
        state = _make_state()
        backend = MockBackend(response=json.dumps({
            "action": {"type": "halt"},
            "confidence": "low",
        }))
        proposer = LLMProposer()
        proposer._llm = backend
        proposer._fallback = None

        proposal = proposer.propose(state)
        assert proposal.type == "halt"

    def test_retries_on_first_halt(self):
        state = _make_state()
        backend = MockBackend(response=json.dumps({
            "action": {"type": "halt"},
            "confidence": "low",
        }))
        proposer = LLMProposer(max_retries=1)
        proposer._llm = backend
        proposer._fallback = None

        proposal = proposer.propose(state)
        # Should retry once and then halt
        assert proposal.type == "halt"
        assert len(backend.calls) == 2

    def test_falls_back_to_deterministic(self):
        state = _make_state()
        proposer = LLMProposer()
        proposer._llm = None  # No backend = fallback
        proposer._fallback = None  # No deterministic fallback either
        proposal = proposer.propose(state)
        # With no fallback, halts
        assert proposal.type == "halt"

    def test_create_llm_proposer_defaults_to_deepseek(self):
        proposer = create_llm_proposer(backend="deepseek", api_key="sk-test")
        assert proposer.backend_name == "deepseek"
        assert proposer._llm is not None

    def test_create_llm_proposer_none_falls_back(self):
        proposer = create_llm_proposer(backend="none")
        assert proposer._llm is None
        state = _make_state()
        proposal = proposer.propose(state)
        assert proposal.type == "swap_model"  # deterministic fallback
