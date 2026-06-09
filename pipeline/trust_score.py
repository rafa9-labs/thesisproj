"""
Trust Score — composite metric from Phase 5 validation outputs.

Integrates PBO, DSR, regime coverage, and min-fold Sharpe to produce
a single 0-1 score with action thresholds.
"""
from __future__ import annotations

from typing import Dict


def compute_trust_score(
    pbo: float,
    dsr: float,
    regime_coverage_ratio: float,
    min_fold_sharpe: float = 0.0,
) -> Dict[str, float]:
    """Compute composite trust score from validation metrics.

    Parameters
    ----------
    pbo : float
        Probability of Backtest Overfitting (PBO) in [0, 1].
        Lower = less likely overfit.
    dsr : float
        Deflated Sharpe Ratio in [0, 1].
        Higher = Sharpe more likely genuinely positive.
    regime_coverage_ratio : float
        Fraction of configured regimes with >= 30 OOS trades.
    min_fold_sharpe : float
        Worst fold Sharpe across the WFO folds.

    Returns
    -------
    dict
        {
            "trust_score": float,     # 0-1 composite
            "action": str,            # "deploy" | "proceed" | "flag" | "reject"
            "sub_scores": {           # individual component contributions
                "pbo_contribution": float,
                "dsr_contribution": float,
                "coverage_contribution": float,
                "floor_contribution": float,
            }
        }
    """
    pbo = max(0.0, min(1.0, pbo)) if not (hasattr(pbo, '__isnan__') and pbo != pbo) else 1.0
    dsr = max(0.0, min(1.0, dsr)) if not (hasattr(dsr, '__isnan__') and dsr != dsr) else 0.0
    regime_coverage_ratio = max(0.0, min(1.0, regime_coverage_ratio))
    min_fold_sharpe = min_fold_sharpe if not (hasattr(min_fold_sharpe, '__isnan__') and min_fold_sharpe != min_fold_sharpe) else -float("inf")

    # Component contributions
    pbo_contrib = 0.35 * (1.0 - pbo)          # lower PBO = higher trust
    dsr_contrib = 0.30 * dsr                  # higher DSR = higher trust
    cov_contrib = 0.20 * regime_coverage_ratio # more regimes covered = higher trust
    floor_contrib = 0.15 * (1.0 if min_fold_sharpe > -0.5 else 0.0)  # no catastrophic folds

    trust_score = pbo_contrib + dsr_contrib + cov_contrib + floor_contrib

    # Action thresholds
    if trust_score >= 0.80:
        action = "deploy"
    elif trust_score >= 0.60:
        action = "proceed"
    elif trust_score >= 0.40:
        action = "flag"
    else:
        action = "reject"

    return {
        "trust_score": round(trust_score, 4),
        "action": action,
        "sub_scores": {
            "pbo_contribution": round(pbo_contrib, 4),
            "dsr_contribution": round(dsr_contrib, 4),
            "coverage_contribution": round(cov_contrib, 4),
            "floor_contribution": round(floor_contrib, 4),
        },
    }
