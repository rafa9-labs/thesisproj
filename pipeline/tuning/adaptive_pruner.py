"""
AdaptivePruner — ASHA-inspired relative median pruner per model family.

Replaces hardcoded pruning thresholds with dynamic cutoffs computed
from the top 20% of completed trials for the same model type.
This ensures high-variance models (Random Forest) are not penalised
for their naturally wider fold-score distribution.

Usage
-----
At start of a model's HPO run:
    AdaptivePruner.reset(model_name="random_forest", grace_trials=10)

After each trial completes (not pruned):
    AdaptivePruner.record_trial(trial_idx, fold_scores, final_score)

Inside fold-level CV loop, after each fold:
    should_prune, cutoff, reason = AdaptivePruner.should_prune_fold(
        trial_idx, fold_idx, cumulative_score,
    )
    if should_prune:
        raise RuntimeError(f"FoldPrunedByGate: {reason} cutoff={cutoff:.4f}")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


_TrialRecord = Dict[str, object]  # {"trial_idx", "fold_scores", "final_score"}


class AdaptivePruner:
    """Per-model-family relative median pruner.

    Module-level state — reset per model via ``reset()``.
    Thread-safe only if called from a single-threaded loop (the default
    Optuna path: n_jobs=1, sequential trials).
    """

    _model_name: str = ""
    _grace_trials: int = 10
    _trials: List[_TrialRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls, model_name: str, grace_trials: int = 10) -> None:
        """Initialise tracking for a new model type."""
        cls._model_name = str(model_name).lower()
        cls._grace_trials = max(1, int(grace_trials))
        cls._trials = []

    @classmethod
    def record_trial(
        cls,
        trial_idx: int,
        fold_scores: List[float],
        final_score: float,
    ) -> None:
        """Record a completed trial's fold-level scores and final objective score."""
        cls._trials.append({
            "trial_idx": int(trial_idx),
            "fold_scores": [float(s) if np.isfinite(s) else np.nan for s in fold_scores],
            "final_score": float(final_score),
        })

    @classmethod
    def completed_trials(cls) -> int:
        """Number of completed (non-pruned) trials recorded so far."""
        return len(cls._trials)

    @classmethod
    def should_prune_fold(
        cls,
        trial_idx: int,
        fold_idx: int,
        cumulative_score: float,
        fold_scores_so_far: Optional[List[float]] = None,
    ) -> Tuple[bool, Optional[float], str]:
        """Decide whether to prune a trial based on its cumulative fold score.

        Parameters
        ----------
        trial_idx : int
            Optuna trial number.
        fold_idx : int
            Zero-based fold index being evaluated (0, 1, 2, ...).
        cumulative_score : float
            The trial's aggregate (recency-weighted or raw mean) score at this fold.
        fold_scores_so_far : list of float, optional
            All valid fold scores up to this point (not currently used, reserved).

        Returns
        -------
        (prune, cutoff, reason)
            prune  — True if the trial should be pruned.
            cutoff — The dynamic threshold that was exceeded.
            reason — Short string describing the decision.
        """
        # Rule 1: folds 0 and 1 are never pruned (need ≥ 3 folds for signal).
        if fold_idx < 2:
            return False, None, "fold_grace"

        # Rule 2: grace period — first N trials of any model are never pruned.
        if cls.completed_trials() < cls._grace_trials:
            return False, None, "trial_grace"

        # Rule 3: fallback wide cutoff when champion pool is too small.
        n_champions = max(1, cls.completed_trials() // 5)
        if n_champions < 1:
            cutoff = -5.0
            if not np.isfinite(cumulative_score) or cumulative_score < cutoff:
                return True, cutoff, "fallback_cutoff"
            return False, cutoff, "ok"

        # Rule 4: champion-based adaptive cutoff.
        return cls._eval_champion_fold_cutoff(fold_idx, cumulative_score)

    @classmethod
    def should_prune_trial(
        cls,
        trial_idx: int,
        fold_scores: List[float],
    ) -> Tuple[bool, Optional[float], str]:
        """Post-hoc trial-level evaluation (replaces the EARLY_HOPELESS gate).

        Checks whether the trial's mean fold score is significantly below
        the champion baseline.  This is a coarser, trial-wide check used
        *after* a batch of folds have been evaluated.
        """
        if not fold_scores:
            return True, None, "no_folds"

        valid = [s for s in fold_scores if np.isfinite(s)]
        if len(valid) < 3:
            return False, None, "insufficient_folds"

        mean_score = float(np.mean(valid))

        # Grace period — never prune.
        if cls.completed_trials() < cls._grace_trials:
            return False, None, "trial_grace"

        # Fallback.
        n_champions = max(1, cls.completed_trials() // 5)
        if n_champions < 1:
            cutoff = -5.0
            if mean_score < cutoff:
                return True, cutoff, "fallback_cutoff"
            return False, cutoff, "ok"

        # Champion baseline.
        champions = cls._top_champions(n_champions)
        champion_means = [
            float(np.mean([s for s in t["fold_scores"] if np.isfinite(s)]))
            for t in champions
        ]
        champion_means = [v for v in champion_means if np.isfinite(v)]
        if len(champion_means) < 2:
            return False, -5.0, "insufficient_champion_folds"

        median = float(np.median(champion_means))
        mad = float(np.median(np.abs(np.array(champion_means, dtype=float) - median)))
        mad = max(mad, 1e-10)
        cutoff = median - 1.5 * mad

        if mean_score < cutoff:
            return (
                True,
                cutoff,
                f"trial_hopeless mean={mean_score:.4f} cutoff={cutoff:.4f}",
            )
        return False, cutoff, "ok"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _top_champions(cls, n: int) -> List[_TrialRecord]:
        """Return the top-*n* trials by ``final_score`` (descending)."""
        sorted_trials = sorted(
            cls._trials,
            key=lambda t: float(t.get("final_score", -9999)),
            reverse=True,
        )
        return sorted_trials[: max(1, n)]

    @classmethod
    def _eval_champion_fold_cutoff(
        cls,
        fold_idx: int,
        cumulative_score: float,
    ) -> Tuple[bool, Optional[float], str]:
        """Compute champion cutoff at a specific fold index and compare."""
        n_champions = max(1, cls.completed_trials() // 5)
        champions = cls._top_champions(n_champions)

        champion_values: List[float] = []
        for t in champions:
            scores = t.get("fold_scores", [])
            if fold_idx < len(scores):
                cum = float(np.nanmean(scores[: fold_idx + 1]))
                if np.isfinite(cum):
                    champion_values.append(cum)

        if len(champion_values) < 2:
            cutoff = -5.0
            if not np.isfinite(cumulative_score) or cumulative_score < cutoff:
                return True, cutoff, "insufficient_champions"
            return False, cutoff, "ok"

        median = float(np.median(champion_values))
        mad = float(np.median(
            np.abs(np.array(champion_values, dtype=float) - median)
        ))
        mad = max(mad, 1e-10)
        cutoff = median - 1.5 * mad

        if not np.isfinite(cumulative_score) or cumulative_score < cutoff:
            return (
                True,
                cutoff,
                f"champion_cutoff median={median:.4f} mad={mad:.4f}",
            )
        return False, cutoff, f"ok"
