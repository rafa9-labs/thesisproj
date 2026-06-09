"""Factory LLM Proposer — LLM-driven committee optimization agent.

Replaces DeterministicProposer with an LLM that analyzes the regime×model
matrix, current committee config, and iteration history to propose the most
improvement-likely action.

Backends:
  - DeepSeek (default, deepseek-chat / V4 Pro model)
  - Ollama (local, free)
  - OpenAI-compatible (generic chat completions endpoint)
  - Anthropic (Claude API)

Falls back to DeterministicProposer when no LLM backend is configured.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import requests

from pipeline.factory_proposer import ActionProposal, DeterministicProposer
from pipeline.factory_state import FactoryState
from pipeline.regime_utils import _REGIME_NAMES

_DEBUG = os.environ.get("FACTORY_LLM_DEBUG", "").strip().lower() in ("1", "true")


# ════════════════════════════════════════════════════════════════════
# Backends
# ════════════════════════════════════════════════════════════════════

class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompts to LLM, return raw text response."""


class DeepSeekBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1",
                 timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("DeepSeek API key not configured")
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.1",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]


class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307",
                 timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Anthropic API key not configured")
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0.3,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body["content"][0]["text"]


def _resolve_backend(backend_name: str, api_key: str, model: str = "",
                     base_url: str = "") -> Optional[LLMBackend]:
    name = backend_name.strip().lower()
    if name in ("none", "", "deterministic"):
        return None
    if name == "deepseek":
        return DeepSeekBackend(
            api_key=api_key,
            model=model or "deepseek-chat",
            base_url=base_url or "https://api.deepseek.com/v1",
        )
    if name == "ollama":
        return OllamaBackend(
            model=model or "llama3.1",
            base_url=base_url or "http://localhost:11434",
        )
    if name in ("openai", "openai_compat"):
        return OpenAICompatBackend(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            base_url=base_url or "https://api.openai.com/v1",
        )
    if name == "anthropic":
        return AnthropicBackend(
            api_key=api_key,
            model=model or "claude-3-haiku-20240307",
        )
    # Generic OpenAI-compatible (custom base URL)
    return OpenAICompatBackend(
        api_key=api_key,
        model=model or "deepseek-chat",
        base_url=base_url,
    )


# ════════════════════════════════════════════════════════════════════
# Prompt Builder
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a committee optimization agent for a Forex trading system.
Your job is to analyze the performance matrix below and propose ONE action
to improve the committee's overall Sharpe ratio.

Rules:
- Propose exactly ONE action: swap, add, remove, or halt.
- Only swap if the candidate model has HIGHER regime Sharpe than the model
  being replaced, by a meaningful margin (>0.02).
- Only add a model if it improves diversification — its regime Sharpe is
  higher than the current WORST model in that regime.
- Only remove a model if the regime has 3+ models and the target has
  negative Sharpe (< -0.1).
- Return "halt" only if no action would meaningfully improve the committee.

Output format (JSON only, no markdown fences, no explanations outside JSON):
{
  "analysis": "1-2 sentences about the weakest regime and why",
  "action": {
    "type": "swap_model",
    "regime": "regime_name",
    "model_add": "model_name",
    "model_remove": "model_name"
  },
  "confidence": "high",
  "rationale": "One sentence with expected Sharpe change"
}

For "add_model": omit model_remove, include model_add.
For "remove_model": omit model_add, include model_remove.
For "halt": omit model_add and model_remove, set type to "halt".
Confidence must be "high", "medium", or "low"."""

SHORTLIST_SYSTEM_PROMPT = """You are a strategic committee optimization agent for a Forex trading system.
Your job is to analyze the performance matrix and current committee configuration
and propose up to 5 LOGICALLY SOUND candidate actions. UCB1 will then optimize
over your shortlist.

Strategy Rules (prune illogical pairings):
- Do NOT propose mean-reversion models (svm, logistic) for trending regimes (trend_up, trend_down).
- Do NOT propose momentum models (cnn, lstm) for mean-reverting regimes (sideways).
- Only propose swaps where the candidate model has HIGHER regime Sharpe than the target.
- Only propose adding a model if it diversifies the regime (different model family).
- Propose removing a model only if it has negative Sharpe in its regime and the regime has 3+ models.
- Propose at most 5 candidates. Fewer is better if fewer are logical.

Output format (JSON only, no markdown fences, no explanations outside JSON):
{
  "analysis": "2-3 sentence strategic assessment of the committee",
  "shortlist": [
    {
      "type": "swap_model",
      "regime": "trend_down",
      "model_add": "cnn",
      "model_remove": "xgboost",
      "rationale": "CNN processes non-linear trend signals better than XGBoost for this regime"
    }
  ],
  "pruned_actions": [
    {"type": "swap_model", "regime": "sideways", "model_add": "lstm", "model_remove": "xgboost", "prune_reason": "LSTM not suitable for mean-reverting regime"}
  ]
}

For "add_model": omit model_remove, include model_add.
For "remove_model": omit model_add, include model_remove.
For "halt": return empty shortlist array, explain why in analysis."""


class PromptBuilder:
    def build(self, state: FactoryState) -> Tuple[str, str]:
        user_prompt_parts = []

        user_prompt_parts.append(self._build_matrix_section(state))
        user_prompt_parts.append(self._build_config_section(state))
        user_prompt_parts.append(self._build_history_section(state))
        user_prompt_parts.append(self._build_available_section(state))
        user_prompt_parts.append(
            f"Current Best Sharpe: {state.global_best_sharpe:.4f}\n"
            f"Total Iterations: {state.iteration}"
        )

        return SYSTEM_PROMPT, "\n\n".join(user_prompt_parts)

    def build_shortlist(self, state: FactoryState, max_candidates: int = 5) -> str:
        parts = []

        parts.append(self._build_matrix_section(state))
        parts.append(self._build_config_section(state))
        parts.append(self._build_available_section(state))

        model_families = {}
        for m in (state.matrix.models if state.matrix else []):
            if m in ("cnn", "lstm", "transformer", "gru", "gru_lstm"):
                model_families[m] = "deep"
            elif m in ("ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost", "stacking_ensemble", "meta_ensemble"):
                model_families[m] = "ensemble"
            else:
                model_families[m] = "classical"

        parts.append("## Model Families\n")
        for m, fam in sorted(model_families.items()):
            parts.append(f"- {m}: {fam}")

        parts.append(
            f"\n## Instructions\n"
            f"Propose at most {max_candidates} logically sound candidate actions.\n"
            f"Current Best Sharpe: {state.global_best_sharpe:.4f}\n"
            f"Iteration: {state.iteration}\n"
            f"Return ONLY a JSON object with 'shortlist' array and 'analysis' string."
        )

        return "\n\n".join(parts)

    def _build_matrix_section(self, state: FactoryState) -> str:
        if state.matrix is None or not state.matrix.models:
            return "## Regime x Model Matrix: NOT AVAILABLE"

        regimes = list(state.matrix.regimes)
        models = list(state.matrix.models)
        header = "| Model | " + " | ".join(regimes) + " |"
        sep = "|-------|" + "|".join("--------" for _ in regimes) + "|"

        rows = []
        for mi, model in enumerate(models):
            vals = []
            for ri, regime in enumerate(regimes):
                try:
                    s = float(state.matrix.sharpe_matrix[mi, ri])
                    vals.append(f"{s:+.2f}")
                except (IndexError, ValueError):
                    vals.append("N/A")
            rows.append(f"| {model} | " + " | ".join(vals) + " |")

        return "## Regime x Model Performance Matrix (Sharpe)\n" + \
               header + "\n" + sep + "\n" + "\n".join(rows)

    def _build_config_section(self, state: FactoryState) -> str:
        lines = ["## Current Committee Configuration"]
        weakest = state.weakest_regime()

        for rname in sorted(state.config.regimes.keys()):
            assignment = state.config.regime_models(rname)
            if assignment is None:
                continue
            model_weights = []
            for m, w in zip(assignment.models, assignment.weights):
                model_weights.append(f"{m} ({w * 100:.0f}%)")
            line = f"{rname}: {', '.join(model_weights)}"
            if rname == weakest:
                line += "  ← WEAKEST REGIME"
            lines.append(line)

        fallback = state.config.regime_models(
            next(iter(state.config.regimes))) if state.config.regimes else None
        if fallback:
            lines.append(f"\nFallback: {', '.join(fallback.models)}")

        return "\n".join(lines)

    def _build_history_section(self, state: FactoryState) -> str:
        if not state.history:
            return "## Iteration History\n(no iterations yet)"

        lines = ["## Iteration History (last 5)"]
        for rec in state.history[-5:]:
            action = rec.action
            a_type = action.get("type", "?")
            regime = action.get("regime", "")
            add = action.get("model_add", "")
            remove = action.get("model_remove", "")
            status = "ACCEPTED" if rec.accepted else "REJECTED"
            delta = rec.after_sharpe - rec.before_sharpe

            desc = f"#{rec.iteration} {a_type.upper()}"
            if add:
                desc += f" +{add}"
            if remove:
                desc += f" -{remove}"
            if regime:
                desc += f" in {regime}"
            desc += f"  Δ={delta:+.4f}  {status}"
            lines.append(desc)

        return "\n".join(lines)

    def _build_available_section(self, state: FactoryState) -> str:
        if state.matrix is None:
            return "## Available Models\n(no matrix loaded)"

        all_models = set(state.matrix.models)
        used_models: set[str] = set()
        for rname in state.config.regimes:
            assignment = state.config.regime_models(rname)
            if assignment:
                used_models.update(assignment.models)
        available = sorted(all_models - used_models)

        lines = ["## Available Models (not in any regime committee)"]
        lines.append(", ".join(available) if available else "(all models in use)")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# Response Parser
# ════════════════════════════════════════════════════════════════════

class ResponseParser:
    def parse(self, raw: str, state: FactoryState) -> ActionProposal:
        """Parse LLM response into a validated ActionProposal."""
        data = self._extract_json(raw)
        if data is None:
            return ActionProposal.halt()

        action_type = self._safe_str(data.get("action", {}).get("type", ""))
        regime = self._safe_str(data.get("action", {}).get("regime", ""))
        model_add = self._safe_str(data.get("action", {}).get("model_add", ""))
        model_remove = self._safe_str(data.get("action", {}).get("model_remove", ""))
        rationale = self._safe_str(data.get("rationale", ""))

        # Validate
        if action_type == "halt":
            return ActionProposal(
                type="halt", rationale=rationale or "LLM concluded no action needed",
            )

        if not regime or regime not in _REGIME_NAMES.values():
            if _DEBUG:
                print(f"[FACTORY-LLM] Invalid regime '{regime}' — halting")
            return ActionProposal.halt()

        available_models = set(state.matrix.models) if state.matrix else set()

        if action_type == "swap_model":
            if (not model_add or not model_remove
                    or model_add not in available_models
                    or model_remove not in available_models):
                return ActionProposal.halt()
            return ActionProposal(
                type="swap_model",
                regime=regime,
                model_add=model_add,
                model_remove=model_remove,
                rationale=rationale or f"LLM swap {model_remove}→{model_add} in {regime}",
            )

        if action_type == "add_model":
            if not model_add or model_add not in available_models:
                return ActionProposal.halt()
            return ActionProposal(
                type="add_model",
                regime=regime,
                model_add=model_add,
                rationale=rationale or f"LLM add {model_add} to {regime}",
            )

        if action_type == "remove_model":
            if not model_remove or model_remove not in available_models:
                return ActionProposal.halt()
            return ActionProposal(
                type="remove_model",
                regime=regime,
                model_remove=model_remove,
                rationale=rationale or f"LLM remove {model_remove} from {regime}",
            )

        return ActionProposal.halt()

    def _extract_json(self, text: str) -> Optional[dict]:
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try markdown code block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try extracting first JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _safe_str(val: Any) -> str:
        return str(val).strip() if val else ""


# ════════════════════════════════════════════════════════════════════
# LLM Proposer
# ════════════════════════════════════════════════════════════════════

class LLMProposer:
    def __init__(
        self,
        backend: str = "deepseek",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        max_retries: int = 2,
        fallback_to_deterministic: bool = True,
    ):
        self.backend_name = backend
        self._llm = _resolve_backend(backend, api_key, model, base_url)
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()
        self.max_retries = max_retries
        self._fallback = DeterministicProposer() if fallback_to_deterministic else None

    def propose(self, state: FactoryState) -> ActionProposal:
        if self._llm is None:
            if self._fallback:
                return self._fallback.propose(state)
            return ActionProposal.halt()

        sys_prompt, user_prompt = self.prompt_builder.build(state)

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._llm.complete(sys_prompt, user_prompt)
                if _DEBUG:
                    print(f"[FACTORY-LLM] Raw response:\n{raw[:500]}")
                proposal = self.parser.parse(raw, state)
                if proposal.type != "halt":
                    return proposal
                if attempt == self.max_retries:
                    return ActionProposal.halt()
                # Halt from LLM — retry with more urgency
                last_error = "LLM returned halt"
            except requests.Timeout:
                last_error = "timeout"
                if attempt < self.max_retries:
                    time.sleep(2)
            except requests.RequestException as e:
                last_error = str(e)
                if _DEBUG:
                    print(f"[FACTORY-LLM] API error: {e}")
                if attempt < self.max_retries:
                    time.sleep(2)
            except Exception as e:
                last_error = str(e)
                if _DEBUG:
                    print(f"[FACTORY-LLM] Unexpected error: {e}")
                break

        if self._fallback:
            if _DEBUG:
                print(f"[FACTORY-LLM] Fallback to deterministic ({last_error})")
            return self._fallback.propose(state)
        return ActionProposal.halt()

    def shortlist(self, state: FactoryState, max_candidates: int = 5) -> List[ActionProposal]:
        if self._llm is None:
            return self._enumerate_candidates(state)[:max_candidates]

        user_prompt = self.prompt_builder.build_shortlist(state, max_candidates)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._llm.complete(SHORTLIST_SYSTEM_PROMPT, user_prompt)
                if _DEBUG:
                    print(f"[FACTORY-LLM] Shortlist raw:\n{raw[:500]}")
                candidates = self._parse_shortlist(raw, state)
                if candidates:
                    return candidates[:max_candidates]
                if attempt == self.max_retries:
                    break
            except requests.Timeout:
                if attempt < self.max_retries:
                    time.sleep(2)
            except requests.RequestException:
                if attempt < self.max_retries:
                    time.sleep(2)
            except Exception:
                break

        return self._enumerate_candidates(state)[:max_candidates]

    def _parse_shortlist(self, raw: str, state: FactoryState) -> List[ActionProposal]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.split("```")[1]
                if raw_clean.startswith("json"):
                    raw_clean = raw_clean[4:]
                try:
                    data = json.loads(raw_clean)
                except json.JSONDecodeError:
                    return []
            else:
                return []

        items = data.get("shortlist", [])
        if not isinstance(items, list):
            return []

        candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue
            action_type = item.get("type", "")
            if action_type not in ("swap_model", "add_model", "remove_model"):
                continue
            regime = item.get("regime", "")
            model_add = item.get("model_add", "")
            model_remove = item.get("model_remove", "")

            if action_type == "swap_model" and model_add and model_remove:
                candidates.append(ActionProposal(
                    type=action_type, regime=regime,
                    model_add=model_add, model_remove=model_remove,
                    rationale=item.get("rationale", ""),
                ))
            elif action_type == "add_model" and model_add:
                candidates.append(ActionProposal(
                    type=action_type, regime=regime,
                    model_add=model_add, model_remove="",
                    rationale=item.get("rationale", ""),
                ))
            elif action_type == "remove_model" and model_remove:
                candidates.append(ActionProposal(
                    type=action_type, regime=regime,
                    model_add="", model_remove=model_remove,
                    rationale=item.get("rationale", ""),
                ))

        return candidates

    def _enumerate_candidates(self, state: FactoryState) -> List[ActionProposal]:
        if self._fallback is None:
            return []
        from pipeline.factory_proposer import DeterministicProposer
        fallback = self._fallback if isinstance(self._fallback, DeterministicProposer) else DeterministicProposer()
        candidates = []
        for _ in range(10):
            proposal = fallback.propose(state)
            if proposal.type == "halt":
                break
            already = any(
                p.regime == proposal.regime and p.model_add == proposal.model_add
                and p.model_remove == proposal.model_remove
                for p in candidates
            )
            if not already:
                candidates.append(proposal)
        return candidates


def create_llm_proposer(
    backend: str = "",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> LLMProposer:
    """Create an LLMProposer from env vars or explicit parameters.

    Priority: explicit params > FACTORY_LLM_* env vars > llm_* config keys.
    """
    _backend = (
        backend
        or os.environ.get("FACTORY_LLM_BACKEND", "")
        or os.environ.get("LLM_BACKEND", "")
        or "deepseek"
    )
    _api_key = (
        api_key
        or os.environ.get("FACTORY_LLM_API_KEY", "")
        or os.environ.get("DEEPSEEK_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    _model = (
        model
        or os.environ.get("FACTORY_LLM_MODEL", "")
    )
    _base_url = (
        base_url
        or os.environ.get("FACTORY_LLM_BASE_URL", "")
    )

    if _DEBUG:
        print(f"[FACTORY-LLM] Backend: {_backend}  Model: {_model or 'default'}"
              f"  API key: {'present' if _api_key else 'missing'}")

    return LLMProposer(
        backend=_backend,
        api_key=_api_key,
        model=_model,
        base_url=_base_url,
        fallback_to_deterministic=True,
    )
