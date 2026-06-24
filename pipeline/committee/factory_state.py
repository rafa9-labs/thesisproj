"""Factory State — experiment tracking, iteration history, and stopping criteria.

The Factory loop optimizes a committee config by iteratively proposing changes,
executing them via Phase D backtest, evaluating the delta, and deciding whether
to accept/revert/stop.

Stopping rules (evaluated in order):
  1. Budget:        iteration >= max_iterations
  2. Hard gate:      ALL 7 regimes have committee Sharpe >= regime_sharpe_floor
  3. Patience:       global_best_Sharpe hasn't improved by >= tolerance for N iterations
  4. Exhaustion:     no untested model×regime swap has higher Sharpe than current
  5. Divergence:     3 consecutive accepted moves produced worse Sharpe
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
from pipeline.regime.regime_utils import _REGIME_NAMES


@dataclass
class IterationRecord:
    iteration: int
    action: Dict[str, Any] = field(default_factory=dict)
    before_sharpe: float = 0.0
    after_sharpe: float = 0.0
    accepted: bool = False
    per_regime_delta: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    rationale: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "action": self.action,
            "before_sharpe": self.before_sharpe,
            "after_sharpe": self.after_sharpe,
            "accepted": self.accepted,
            "per_regime_delta": self.per_regime_delta,
            "timestamp": self.timestamp,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IterationRecord":
        return cls(**data)


class FactoryState:
    def __init__(
        self,
        committee_config: CommitteeConfig,
        regime_matrix: Optional[Any] = None,
        patience: int = 5,
        stopping_tolerance: float = 0.02,
        regime_sharpe_floor: float = 0.3,
        max_iterations: int = 20,
    ):
        self.config = committee_config
        self.matrix = regime_matrix  # RegimeModelMatrix from ExpertProfiler
        self.patience = patience
        self.stopping_tolerance = stopping_tolerance
        self.regime_sharpe_floor = regime_sharpe_floor
        self.max_iterations = max_iterations

        self.iteration = 0
        self.global_best_sharpe: float = float("-inf")
        self.global_best_config: Optional[Dict[str, Any]] = None
        self.history: List[IterationRecord] = []
        self._tried_combinations: List[Tuple[str, str, str]] = []

        # Track backtest result per iteration
        self._last_result: Optional[Any] = None

    def track_iteration(self, record: IterationRecord):
        record.iteration = self.iteration
        self.history.append(record)

        if record.accepted and record.after_sharpe > self.global_best_sharpe:
            self.global_best_sharpe = record.after_sharpe
            self.global_best_config = self.config.to_dict()

        self.iteration += 1

    def track_unmodel_combinations(self, regime: str, model: str):
        if model not in (self.config.regime_models(regime) or RegimeAssignment([], [])).models:
            self._tried_combinations.append((regime, model, "tried"))

    def should_stop(self) -> Tuple[bool, str]:
        # Rule 1: Budget
        if self.iteration >= self.max_iterations:
            return True, f"budget: reached max {self.max_iterations} iterations"

        # Rule 5: Divergence (3 consecutive accepted moves worse)
        if len(self.history) >= 3:
            last3 = self.history[-3:]
            if all(r.accepted for r in last3):
                if (last3[0].after_sharpe > last3[1].after_sharpe >
                        last3[2].after_sharpe):
                    return True, "divergence: 3 consecutive deteriorating iterations"

        # Rule 4: Exhaustion — no untested model beats worst in any weak regime
        if self.matrix is not None:
            for regime_name in _REGIME_NAMES.values():
                current_models = self._regime_models_for(regime_name)
                if not current_models:
                    continue
                worst_in_regime = self._worst_model_in_regime(regime_name)
                best_untested = self._best_untested_model(regime_name)
                if worst_in_regime is not None and best_untested is not None:
                    worst_sharpe = self._model_sharpe_in_regime(regime_name, worst_in_regime)
                    best_sharpe = self._model_sharpe_in_regime(regime_name, best_untested)
                    if best_sharpe > worst_sharpe + 0.01:
                        return False, ""  # still have moves
            # No regime has promising swaps
            if self.iteration > 0:
                return True, "exhaustion: no untested model beats worst in any regime"

        # Rule 2: Hard gate — all regimes above floor
        if self._last_result is not None:
            try:
                folds = getattr(self._last_result, "folds", [])
                regime_sharpes: Dict[str, List[float]] = {}
                for fold in folds:
                    rd = getattr(fold, "regime_distribution", {}) or {}
                    for rname, frac in rd.items():
                        if frac > 0:
                            regime_sharpes.setdefault(rname, []).append(
                                getattr(fold, "sharpe", 0.0))
                all_good = True
                for rname in regime_sharpes:
                    avg = np.mean(regime_sharpes[rname]) if regime_sharpes[rname] else 0.0
                    if avg < self.regime_sharpe_floor:
                        all_good = False
                        break
                if all_good and len(regime_sharpes) >= 2:
                    return True, f"hard_gate: all {len(regime_sharpes)} regimes >= {self.regime_sharpe_floor}"
            except Exception:
                pass

        # Rule 3: Patience — global best hasn't improved
        if len(self.history) >= self.patience:
            recent_bests = []
            best_so_far = float("-inf")
            for rec in self.history:
                if rec.accepted:
                    best_so_far = max(best_so_far, rec.after_sharpe)
                recent_bests.append(best_so_far)
            recent = recent_bests[-self.patience:]
            if (recent[-1] - recent[0]) < self.stopping_tolerance:
                return True, (f"patience: global best improved by "
                              f"{recent[-1] - recent[0]:.4f} < {self.stopping_tolerance} "
                              f"over last {self.patience} iterations")

        return False, ""

    def weakest_regime(self) -> Optional[str]:
        if self.matrix is None or not self.matrix.models:
            return None
        candidates = {}
        for regime_name in _REGIME_NAMES.values():
            assignment = self.config.regime_models(regime_name)
            if assignment is None or len(assignment.models) == 0:
                continue
            total_sharpe = 0.0
            count = 0
            for model in assignment.models:
                s = self._model_sharpe_in_regime(regime_name, model)
                if np.isfinite(s):
                    total_sharpe += s
                    count += 1
            if count > 0:
                candidates[regime_name] = total_sharpe / count
        if not candidates:
            return None
        return min(candidates, key=candidates.get)

    def worst_model_in_regime(self, regime: str) -> Optional[str]:
        assignment = self._regime_models_for(regime)
        if not assignment:
            return None
        scores = [(m, self._model_sharpe_in_regime(regime, m)) for m in assignment]
        scores = [(m, s) for m, s in scores if np.isfinite(s)]
        if not scores:
            return None
        return min(scores, key=lambda x: x[1])[0]

    def best_candidate_for_regime(self, regime: str) -> Optional[str]:
        if self.matrix is None:
            return None
        current = set(self._regime_models_for(regime) or [])
        best_model = None
        best_sharpe = float("-inf")
        for model in self.matrix.models:
            if model in current:
                continue
            s = self._model_sharpe_in_regime(regime, model)
            if np.isfinite(s) and s > best_sharpe:
                best_sharpe = s
                best_model = model
        return best_model

    def _regime_models_for(self, regime: str) -> Optional[List[str]]:
        assignment = self.config.regime_models(regime)
        if assignment is None:
            return None
        return assignment.models

    def _model_sharpe_in_regime(self, regime: str, model: str) -> float:
        if self.matrix is None:
            return float("nan")
        try:
            ri = self.matrix.regimes.index(regime) if regime in self.matrix.regimes else -1
            mi = self.matrix.models.index(model) if model in self.matrix.models else -1
            if ri >= 0 and mi >= 0:
                return float(self.matrix.sharpe_matrix[mi, ri])
        except (ValueError, IndexError):
            pass
        return float("nan")

    def _worst_model_in_regime(self, regime: str) -> Optional[str]:
        return self.worst_model_in_regime(regime)

    def _best_untested_model(self, regime: str) -> Optional[str]:
        return self.best_candidate_for_regime(regime)

    def summary(self) -> dict:
        last_accepted = [r for r in self.history if r.accepted]
        return {
            "iteration": self.iteration,
            "global_best_sharpe": self.global_best_sharpe,
            "total_moves": len(last_accepted),
            "stopped": self.should_stop()[0],
            "config_regimes": len(self.config.regimes),
            "matrix_models": len(self.matrix.models) if self.matrix else 0,
        }


def load_state_from_disk(
    config_path: str,
    matrix_path: str,
    patience: int = 5,
    tolerance: float = 0.02,
    floor: float = 0.3,
    max_iter: int = 20,
) -> Optional[FactoryState]:
    config = CommitteeConfig.from_json(config_path)
    try:
        with open(matrix_path) as f:
            matrix_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    from pipeline.committee.expert_profiler import RegimeModelMatrix
    matrix = RegimeModelMatrix(
        regimes=matrix_data.get("regimes", []),
        models=matrix_data.get("models", []),
        sharpe_matrix=np.array(matrix_data.get("sharpe", [])),
        trade_matrix=np.array(matrix_data.get("trades", [])),
        hitrate_matrix=np.array(matrix_data.get("hit_rate", [])),
    )

    return FactoryState(
        committee_config=config,
        regime_matrix=matrix,
        patience=patience,
        stopping_tolerance=tolerance,
        regime_sharpe_floor=floor,
        max_iterations=max_iter,
    )
