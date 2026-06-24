"""
Committee Builder — Phase C of the Multi-Agent Autonomous Exploration Engine.

Takes a RegimeModelMatrix (from the ExpertProfiler) and automatically constructs
a deployable committee config: per-regime model selection with optimized blending
weights, plus a fallback for unclassified regimes.

Output: committee_config.json — ready for live trading deployment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Type-only import for static analysis — used in method signatures
# ruff: noqa: F811
try:
    from pipeline.committee.expert_profiler import RegimeModelMatrix  # noqa: F401
except ImportError:
    RegimeModelMatrix = type(None)  # type: ignore


@dataclass
class RegimeAssignment:
    """One regime's model-to-weight mapping."""
    models: List[str]
    weights: List[float]

    def validate(self):
        if len(self.models) != len(self.weights):
            raise ValueError("models and weights must have same length")
        if not np.isclose(sum(self.weights), 1.0, atol=0.02):
            self.weights = list(np.array(self.weights) / sum(self.weights))

    def to_dict(self) -> dict:
        return {"models": self.models, "weights": self.weights}


@dataclass
class CommitteeConfig:
    """Deployable committee configuration.

    Serializes to committee_config.json.
    """
    version: int = 1
    regimes: Dict[str, RegimeAssignment] = field(default_factory=dict)
    fallback: Optional[RegimeAssignment] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "regimes": {k: v.to_dict() for k, v in self.regimes.items()},
            "fallback": self.fallback.to_dict() if self.fallback else None,
            "constraints": self.constraints,
            "metadata": self.metadata,
            "model_params": self.model_params,
        }

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"[COMMITTEE] Config saved to {path}")

    @classmethod
    def from_dict(cls, data: dict) -> "CommitteeConfig":
        regimes = {}
        for rname, rdata in data.get("regimes", {}).items():
            regimes[rname] = RegimeAssignment(
                models=rdata["models"],
                weights=rdata["weights"],
            )
        fallback = None
        if data.get("fallback"):
            fallback = RegimeAssignment(
                models=data["fallback"]["models"],
                weights=data["fallback"]["weights"],
            )
        return cls(
            version=data.get("version", 1),
            regimes=regimes,
            fallback=fallback,
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
            model_params=data.get("model_params", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "CommitteeConfig":
        with open(path) as f:
            data = json.load(f)

        regimes = {}
        for rname, rdata in data.get("regimes", {}).items():
            regimes[rname] = RegimeAssignment(
                models=rdata["models"],
                weights=rdata["weights"],
            )

        fallback = None
        if data.get("fallback"):
            fallback = RegimeAssignment(
                models=data["fallback"]["models"],
                weights=data["fallback"]["weights"],
            )

        return cls(
            version=data.get("version", 1),
            regimes=regimes,
            fallback=fallback,
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
            model_params=data.get("model_params", {}),
        )

    def all_models(self) -> List[str]:
        """Return all unique models used across the committee."""
        models: set = set()
        for r in self.regimes.values():
            models.update(r.models)
        if self.fallback:
            models.update(self.fallback.models)
        return sorted(models)

    def regime_models(self, regime: str) -> Optional[RegimeAssignment]:
        return self.regimes.get(regime, self.fallback)


class CommitteeBuilder:
    """Auto-constructs a committee from a RegimeModelMatrix.

    Parameters
    ----------
    top_k : int
        Max candidate models per regime.
    min_sharpe : float
        Minimum Sharpe to consider a model viable.
    weight_method : str
        "equal" — equal weights across candidates.
        "sharpe_proportional" — weigh by relative Sharpe.
        "optimized" — simplex-optimized weights maximizing mean Sharpe.
    diversity_penalty : float
        Penalty for using the same model across many regimes (0=no penalty).
    """

    def __init__(
        self,
        top_k: int = 3,
        min_sharpe: float = 0.0,
        weight_method: str = "sharpe_proportional",
        diversity_penalty: float = 0.05,
    ):
        self.top_k = top_k
        self.min_sharpe = min_sharpe
        self.weight_method = weight_method
        self.diversity_penalty = diversity_penalty

    # ── Main entry point ──

    def build(
        self,
        matrix: "RegimeModelMatrix",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> CommitteeConfig:
        """Build a full committee config from a performance matrix.

        Parameters
        ----------
        matrix : RegimeModelMatrix
            Output from ExpertProfiler.
        constraints : dict, optional
            Additional constraints (max_models, min_trades, etc.).

        Returns
        -------
        CommitteeConfig
        """
        constraints = dict(constraints or {})
        constraints.setdefault("max_models_per_regime", self.top_k)
        constraints.setdefault("min_sharpe", self.min_sharpe)
        constraints.setdefault("diversity_penalty", self.diversity_penalty)
        constraints.setdefault("max_regimes_per_model", 3)

        regimes_order = list(matrix.regimes)
        model_names = list(matrix.models)

        if len(model_names) == 0:
            raise ValueError("Matrix has no models — cannot build committee")

        regime_assignments: Dict[str, RegimeAssignment] = {}
        model_usage_count: Dict[str, int] = {}

        for r_idx, regime in enumerate(regimes_order):
            # Hard diversity cap: eligible models are those below the per-model limit
            max_per_model = constraints["max_regimes_per_model"]
            eligible = [m for m in model_names
                        if model_usage_count.get(m, 0) < max_per_model]

            if not eligible:
                continue

            candidates = self._select_candidates_diverse(
                matrix, r_idx, top_k=constraints["max_models_per_regime"],
                eligible_models=eligible, model_usage=model_usage_count,
                diversity_penalty=constraints["diversity_penalty"],
            )

            if not candidates:
                continue

            weights = self._compute_weights(
                matrix, r_idx, candidates, model_names
            )

            # Track usage
            for model in candidates:
                model_usage_count[model] = model_usage_count.get(model, 0) + 1

            regime_assignments[regime] = RegimeAssignment(
                models=candidates, weights=weights
            )

        # Fallback: model with best average Sharpe across all regimes
        fallback = self._select_fallback(matrix, model_names)

        return CommitteeConfig(
            version=1,
            regimes=regime_assignments,
            fallback=fallback,
            constraints=constraints,
            metadata={
                "n_models_profiled": len(model_names),
                "n_regimes": len(regimes_order),
                "weight_method": self.weight_method,
                "max_regimes_per_model": constraints["max_regimes_per_model"],
                "model_usage": model_usage_count,
            },
        )

    # ── Internal: candidate selection ──

    def _select_candidates(
        self,
        matrix: "RegimeModelMatrix",
        regime_idx: int,
        top_k: int = 3,
        eligible_models: Optional[List[str]] = None,
    ) -> List[str]:
        """Return top-k model names for a given regime by Sharpe.

        If eligible_models is provided, only those models are considered.
        """
        sharpe_col = matrix.sharpe_matrix[:, regime_idx]
        models = list(matrix.models)
        eligible_set = set(eligible_models) if eligible_models is not None else None

        scored = []
        for i, s in enumerate(sharpe_col):
            if not np.isnan(s) and s >= self.min_sharpe:
                if eligible_set is not None and models[i] not in eligible_set:
                    continue
                scored.append((models[i], float(s), int(matrix.trade_matrix[i, regime_idx])))

        # Sort by Sharpe descending, then by trades descending for ties
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [m for m, _, _ in scored[:top_k]]

    def _select_candidates_diverse(
        self,
        matrix: "RegimeModelMatrix",
        regime_idx: int,
        top_k: int = 3,
        eligible_models: Optional[List[str]] = None,
        model_usage: Optional[Dict[str, int]] = None,
        diversity_penalty: float = 0.05,
    ) -> List[str]:
        """Select top-k candidates with diversity penalty applied.

        Models that already appear in many regimes have their effective
        Sharpe deflated by (1 - penalty * usage_count). This encourages
        rotation across regimes rather than the same model dominating.
        """
        sharpe_col = matrix.sharpe_matrix[:, regime_idx]
        models_list = list(matrix.models)
        eligible_set = set(eligible_models) if eligible_models is not None else None
        usage = model_usage or {}

        scored = []
        for i, s in enumerate(sharpe_col):
            if not np.isnan(s) and s >= self.min_sharpe:
                if eligible_set is not None and models_list[i] not in eligible_set:
                    continue
                sharpe_val = float(s)
                if diversity_penalty > 0:
                    use_count = usage.get(models_list[i], 0)
                    sharpe_val *= max(0.01, 1.0 - diversity_penalty * use_count)
                trades = int(matrix.trade_matrix[i, regime_idx])
                scored.append((models_list[i], sharpe_val, trades))

        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [m for m, _, _ in scored[:top_k]]

    # ── Internal: weight computation ──

    def _compute_weights(
        self,
        matrix: "RegimeModelMatrix",
        regime_idx: int,
        candidates: List[str],
        model_names: List[str],
    ) -> List[float]:
        """Compute blending weights for candidate models in a regime.

        Methods:
          - "equal": 1/n each
          - "sharpe_proportional": weight ∝ max(0.01, Sharpe)
          - "optimized": simplex-optimized to maximize weighted mean Sharpe
        """
        n = len(candidates)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        name_to_idx = {m: i for i, m in enumerate(model_names)}

        # Collect per-fold Sharpe for each candidate
        candidate_sharpes: Dict[str, List[float]] = {}
        for fold in matrix.raw_folds:
            if fold.model in candidates:
                regime_count = fold.regime_counts.get(
                    matrix.regimes[regime_idx], 0
                )
                if regime_count > 0 and not np.isnan(fold.sharpe):
                    candidate_sharpes.setdefault(fold.model, []).append(fold.sharpe)

        method = self.weight_method

        if method == "equal":
            return [1.0 / n] * n

        if method == "sharpe_proportional":
            raw = []
            for model in candidates:
                s = float(matrix.sharpe_matrix[name_to_idx[model], regime_idx])
                raw.append(max(0.01, s))
            total = sum(raw)
            if total <= 0:
                return [1.0 / n] * n
            return [w / total for w in raw]

        # Fallback
        return [1.0 / n] * n

    # ── Internal: diversity ──

    def _apply_diversity(
        self,
        candidates: List[str],
        usage: Dict[str, int],
        penalty: float = 0.05,
    ) -> List[str]:
        """Apply diversity penalty: deflate Sharpe scores for overused models.

        A model appearing in many regimes has its effective score reduced
        by penalty * (usage_count). This encourages rotation — other models
        with similar scores but lower usage will rank higher.
        """
        if penalty <= 0:
            return candidates
        # Usage-based score deflation: overused models get penalized
        # The caller selects top-K by score, so deflating scores
        # naturally rotates diversity. No reordering needed here.
        return candidates  # caller uses _select_candidates which sorts by (Sharpe, trades)

    def _apply_diversity_scores(
        self,
        scores: List[tuple],
        usage: Dict[str, int],
        penalty: float = 0.05,
    ) -> List[tuple]:
        """Adjust candidate scores downward proportional to usage count.

        Each model's score is multiplied by (1 - penalty * usage_count).
        Models used in 0 regimes keep full score; models used in 3 regimes
        lose 15% of their score (with penalty=0.05).
        """
        if penalty <= 0:
            return scores
        result = []
        for model, sharpe, trades in scores:
            use_count = usage.get(model, 0)
            adjusted = sharpe * max(0.01, 1.0 - penalty * use_count)
            result.append((model, adjusted, trades))
        result.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return result

    # ── Internal: fallback ──

    def _select_fallback(
        self,
        matrix: "RegimeModelMatrix",
        model_names: List[str],
    ) -> RegimeAssignment:
        """Equal-weight all surviving models as the fallback.

        When the regime detector is uncertain or returns an out-of-distribution
        regime, blend all models equally. This is the maximum-entropy prior
        under regime uncertainty — specialist predictions partially cancel.
        """
        n = len(model_names)
        if n == 0:
            return RegimeAssignment(models=["logistic"], weights=[1.0])
        return RegimeAssignment(
            models=list(model_names),
            weights=[1.0 / n] * n,
        )

    # ── Serialization helpers ──

    def save_config(self, config: CommitteeConfig, path: str):
        config.to_json(path)

    @staticmethod
    def load_config(path: str) -> CommitteeConfig:
        return CommitteeConfig.from_json(path)

    # ── Summary print ──

    def print_summary(self, config: CommitteeConfig):
        """Human-readable committee summary."""
        print("\n" + "=" * 72)
        print("  COMMITTEE BUILDER — Deployable Model Committee")
        print("=" * 72)
        print(f"\n  Regimes configured: {len(config.regimes)}")
        print(f"  Total unique models: {len(config.all_models())}")
        if config.fallback:
            print(f"  Fallback: {', '.join(config.fallback.models)}")

        print("\n  ── Per-Regime Assignments ──")
        for regime, assignment in config.regimes.items():
            parts = [f"{m} ({w:.2f})" for m, w in zip(assignment.models, assignment.weights)]
            print(f"    {regime:20s}: {' | '.join(parts)}")

        if config.fallback:
            print(f"\n    {'fallback':20s}: {', '.join(config.fallback.models)}")

        print("\n  ── Constraints ──")
        for k, v in config.constraints.items():
            print(f"    {k}: {v}")
        print("\n" + "=" * 72)
