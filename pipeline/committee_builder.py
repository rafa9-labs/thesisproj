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
    from pipeline.expert_profiler import RegimeModelMatrix  # noqa: F401
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

            candidates = self._select_candidates(
                matrix, r_idx, top_k=constraints["max_models_per_regime"],
                eligible_models=eligible,
            )

            if not candidates:
                continue

            # Apply diversity penalty
            candidates = self._apply_diversity(
                candidates, model_usage_count, penalty=constraints["diversity_penalty"]
            )

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

        if method == "optimized":
            return self._simplex_optimize(candidates, candidate_sharpes)

        # Fallback
        return [1.0 / n] * n

    def _simplex_optimize(
        self,
        candidates: List[str],
        candidate_sharpes: Dict[str, List[float]],
    ) -> List[float]:
        """Optimize weights on the simplex to maximize mean Sharpe.

        Uses scipy.optimize.minimize with constraints sum(w)=1, w_i >= 0.
        Objective: -mean_weighted_sharpe + 0.1 * std_weighted_sharpe (risk-averse).
        """
        n = len(candidates)

        # Build Sharpe matrix: n_candidates × n_folds
        min_len = min(
            (len(candidate_sharpes.get(m, [])) for m in candidates),
            default=0,
        )
        if min_len < 2:
            return [1.0 / n] * n

        S = np.zeros((n, min_len))
        for i, model in enumerate(candidates):
            S[i, :] = candidate_sharpes[model][:min_len]

        def objective(w):
            portfolio = w @ S
            mean_sr = np.mean(portfolio)
            std_sr = np.std(portfolio, ddof=1)
            return -mean_sr + 0.1 * std_sr

        try:
            from scipy.optimize import minimize

            x0 = np.ones(n) / n
            bounds = [(0.0, 1.0)] * n
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            result = minimize(
                objective, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-8},
            )

            if result.success:
                w = np.maximum(result.x, 0.0)
                s = w.sum()
                if s > 0:
                    return list(w / s)
        except ImportError:
            pass

        return [1.0 / n] * n

    # ── Internal: diversity ──

    def _apply_diversity(
        self,
        candidates: List[str],
        usage: Dict[str, int],
        penalty: float = 0.05,
    ) -> List[str]:
        """Apply diversity penalty: if a model is overused, try to rotate.

        Note: this is a soft penalty. If fewer alternatives exist, retain anyway.
        """
        if penalty <= 0:
            return candidates
        # For now: no reordering, just track usage for metadata.
        # Full diversity rotation can be added when we have a larger pool.
        return candidates

    # ── Internal: fallback ──

    def _select_fallback(
        self,
        matrix: "RegimeModelMatrix",
        model_names: List[str],
    ) -> RegimeAssignment:
        """Select the best model across all regimes as the fallback."""
        n_regimes = matrix.sharpe_matrix.shape[1]

        best_model = None
        best_avg = -np.inf

        for i, model in enumerate(model_names):
            row = matrix.sharpe_matrix[i, :]
            valid = row[~np.isnan(row)]
            if len(valid) == 0:
                continue
            avg_sr = float(np.mean(valid))
            # Prefer models that work in more regimes
            coverage = len(valid) / n_regimes
            score = avg_sr + 0.05 * coverage

            if score > best_avg:
                best_avg = score
                best_model = model

        if best_model is None:
            best_model = model_names[0] if model_names else "logistic"

        return RegimeAssignment(models=[best_model], weights=[1.0])

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
