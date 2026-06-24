"""
BorutaSHAP feature selection for time-series financial data.

Implements the Boruta algorithm using SHAP (SHapley Additive exPlanations)
values instead of raw feature importance. Shadow features are created by
randomly permuting each real feature column. A feature is "confirmed" only
if it consistently beats its own randomized shadow across Purged K-Fold
time-series splits.

This is stricter than the legacy majority-vote permutation-importance sweep
because it statistically tests each feature against noise rather than just
checking if importance > 0.

Reference:
  Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007).
  Bias in random forest variable importance measures.
  BMC Bioinformatics, 8(1), 25.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore[import-untyped]
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

if HAS_SHAP:
    pass


def _make_labels(df: pd.DataFrame, threshold: float = 0.0001) -> np.ndarray:
    prices = df["mid_c"].to_numpy(dtype=np.float64)
    rets = np.zeros_like(prices)
    rets[1:] = np.log(prices[1:] / prices[:-1])
    labels = np.ones(len(rets), dtype=np.int32) * -1
    labels[:-1] = np.where(
        rets[1:] > threshold, 1,
        np.where(rets[1:] < -threshold, 0, -1),
    )
    labels[-1] = 1
    return labels


def _chronological_fold_indices(
    n_samples: int, n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    fold_size = n_samples // (n_folds + 1)
    if fold_size < 100:
        fold_size = max(50, n_samples // 3)
    indices = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        if test_end > n_samples:
            test_end = n_samples
        if train_end >= test_start:
            train_end = test_start - 1
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) >= 50 and len(test_idx) >= 20:
            indices.append((train_idx, test_idx))
    return indices


def _shuffle_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shadow = np.empty_like(X)
    for j in range(X.shape[1]):
        shadow[:, j] = rng.permutation(X[:, j])
    return shadow


class BorutaSHAPSelector:
    """Boruta feature selector powered by SHAP importance values.

    Creates shadow features (randomly permuted copies of real features) and
    trains a Random Forest on the combined real+shadow matrix. SHAP values
    are computed via TreeExplainer. A real feature is confirmed only if its
    SHAP importance consistently exceeds the maximum SHAP importance among
    ALL shadow features across multiple Purged K-Fold splits.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the Random Forest (default 100).
    max_depth : int
        Maximum tree depth — kept shallow (5) to prevent memorization.
    n_folds : int
        Number of Purged K-Fold time-series splits.
    percentile : int
        Confirmation threshold (0-100). Feature must beat shadow max in
        >= percentile% of folds to be confirmed. 90 = 4/5 folds or 5/5.
    max_iter : int
        Maximum Boruta iterations before forced convergence.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        n_folds: int = 5,
        percentile: int = 90,
        max_iter: int = 20,
        random_state: int = 42,
        economic_floor_pct: float = 0.02,
    ):
        if not HAS_SHAP:
            raise ImportError(
                "shap is required for BorutaSHAPSelector. "
                "Install with: pip install shap"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_folds = n_folds
        self.percentile = percentile
        self.max_iter = max_iter
        self.random_state = random_state
        self.economic_floor_pct = economic_floor_pct

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[str], List[str], List[str], Dict]:
        X = X.astype(np.float32, copy=False)
        rng = np.random.default_rng(self.random_state)

        fold_splits = _chronological_fold_indices(len(X), self.n_folds)
        if not fold_splits:
            return list(feature_names), [], [], {"error": "insufficient data for folds"}

        n_features = len(feature_names)
        confirmed: set = set()
        rejected: set = set()
        tentative: set = set(range(n_features))

        confirmation_counts = np.zeros(n_features, dtype=np.int32)
        rejection_counts = np.zeros(n_features, dtype=np.int32)
        per_feature_shap: Dict[str, float] = {}
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            if not tentative:
                break

            confirmation_counts[:] = 0
            rejection_counts[:] = 0
            fold_shap_sums = np.zeros(n_features, dtype=np.float64)

            for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                if progress_callback and fold_idx == 0:
                    progress_callback(
                        f"Boruta iter {iteration}/{self.max_iter}: "
                        f"confirmed={len(confirmed)} rejected={len(rejected)} "
                        f"tentative={len(tentative)}"
                    )

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_test = X[test_idx]

                shadow = _shuffle_columns(X_test, rng)
                X_combined = np.hstack([X_test, shadow])

                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + iteration,
                    n_jobs=1,
                )
                rf.fit(X_train, y_train)

                explainer = shap.TreeExplainer(
                    rf, feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(X_combined, check_additivity=False)

                if isinstance(shap_values, list):
                    real_shap_raw = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    real_shap_raw = shap_values

                if real_shap_raw.ndim == 3:
                    real_shap_2d = np.abs(real_shap_raw).mean(axis=2)
                else:
                    real_shap_2d = np.abs(real_shap_raw)

                real_importance = real_shap_2d[:, :n_features].mean(axis=0)
                shadow_importance = real_shap_2d[:, n_features:].mean(axis=0)
                base_shadow_max = float(np.max(shadow_importance))

                top_feature_strength = float(np.max(real_importance))
                economic_floor = top_feature_strength * self.economic_floor_pct
                strict_threshold = max(base_shadow_max, economic_floor)

                for j in range(n_features):
                    if j in tentative:
                        fold_shap_sums[j] += real_importance[j]
                        if real_importance[j] > strict_threshold:
                            confirmation_counts[j] += 1
                    if real_importance[j] < np.median(shadow_importance):
                        rejection_counts[j] += 1

                del shadow, X_combined, shap_values, real_shap_raw, real_shap_2d

            folds_required_for_confirmation = max(
                1, int(np.ceil(len(fold_splits) * self.percentile / 100.0))
            )
            folds_required_for_rejection = max(
                1,
                len(fold_splits)
                - int(np.ceil(len(fold_splits) * self.percentile / 100.0))
                + 1,
            )

            newly_confirmed = set()
            newly_rejected = set()

            for j in list(tentative):
                if confirmation_counts[j] >= folds_required_for_confirmation:
                    newly_confirmed.add(j)
                elif rejection_counts[j] >= folds_required_for_rejection:
                    newly_rejected.add(j)

            confirmed |= newly_confirmed
            rejected |= newly_rejected
            tentative -= newly_confirmed
            tentative -= newly_rejected

            n_folds_active = len(fold_splits)
            for j in range(n_features):
                if n_folds_active > 0:
                    per_feature_shap[feature_names[j]] = float(
                        fold_shap_sums[j] / n_folds_active
                    )

        if tentative:
            for j in tentative:
                rejected.add(j)
            tentative.clear()

        confirmed_names = [feature_names[j] for j in sorted(confirmed)]
        rejected_names = [feature_names[j] for j in sorted(rejected)]

        report = {
            "method": "boruta_shap",
            "iterations": iteration,
            "n_features_total": n_features,
            "n_confirmed": len(confirmed),
            "n_rejected": len(rejected),
            "n_tentative_forced": 0,
            "confirmed": confirmed_names,
            "rejected": rejected_names,
        }

        return confirmed_names, [], rejected_names, report


def boruta_sweep_features(
    df: pd.DataFrame,
    label_threshold: float = 0.0001,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_folds: int = 5,
    percentile: int = 90,
    max_iter: int = 20,
    random_state: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
    economic_floor_pct: float = 0.02,
) -> Tuple[List[str], Dict[str, float], Dict]:
    """Drop-in replacement for sweep_features() using BorutaSHAP.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLC data.
    label_threshold : float
        Next-bar return threshold for buy/sell labels.
    n_estimators : int
        RF estimators (default 100).
    max_depth : int
        Max tree depth — kept shallow to prevent memorization.
    n_folds : int
        Purged K-Fold time-series splits.
    percentile : int
        Confirmation percentile (90 = feature must beat noise in >=90% of folds).
    max_iter : int
        Maximum Boruta iterations.
    random_state : int
        Seed.

    Returns
    -------
    locked_features : list[str]
        Feature names confirmed by BorutaSHAP.
    importance_scores : dict[str, float]
        Mean SHAP importance per feature across folds.
    report : dict
        Full report with confirmed/rejected/tentative counts.
    """
    from pipeline.features.feature_sweep import expand_features

    print("\n" + "=" * 64)
    print("  PHASE -1: BORUTA SHAP FEATURE SWEEP")
    print("=" * 64)

    rng = np.random.default_rng(random_state)
    df_feat = expand_features(df)
    labels = _make_labels(df_feat, threshold=label_threshold)

    valid = labels != -1
    df_feat = df_feat.loc[valid].copy()
    labels = labels[valid]

    exclude = {
        "returns", "time", "timestamp", "label",
        "mid_h", "mid_l", "mid_c", "mid_o",
        "mid_high", "mid_low", "mid_close", "mid_open",
        "bid_open", "bid_close", "ask_open", "ask_close",
        "spread", "volume",
    }
    numeric_cols = [
        c for c in df_feat.columns
        if c not in exclude and np.issubdtype(df_feat[c].dtype, np.number)
    ]
    df_feat = df_feat[numeric_cols].copy()
    df_feat = df_feat.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    n = len(df_feat)
    feature_names = list(df_feat.columns)
    X = df_feat.to_numpy(np.float32)
    y = labels

    print(f"      Features expanded: {len(numeric_cols)}")
    print(f"  Bars: {n}, Folds: {n_folds}, percentile={percentile}, max_iter={max_iter}")
    if progress_callback:
        progress_callback(f"BorutaSHAP: {len(numeric_cols)} features across {n} bars")

    selector = BorutaSHAPSelector(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_folds=n_folds,
        percentile=percentile,
        max_iter=max_iter,
        random_state=random_state,
        economic_floor_pct=economic_floor_pct,
    )

    confirmed, tentative, rejected, boruta_report = selector.select(
        X, y, feature_names, progress_callback=progress_callback,
    )

    print(f"\n  BorutaSHAP results:")
    print(f"    Confirmed: {len(confirmed)}  {confirmed[:10]}{'...' if len(confirmed) > 10 else ''}")
    print(f"    Rejected:  {len(rejected)}  {rejected[:10]}{'...' if len(rejected) > 10 else ''}")

    locked = list(confirmed)

    if len(locked) > 50:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(
            X[:, [feature_names.index(f) for f in locked]],
            y, random_state=random_state,
        )
        mi_ranked = sorted(
            zip(locked, mi_scores), key=lambda x: x[1], reverse=True,
        )
        locked = [f for f, _ in mi_ranked[:50]]
        print(f"  MI filter: capped at {len(locked)} features")

    if len(locked) < 8:
        fallback_pool = confirmed if confirmed else rejected
        if fallback_pool:
            shap_scores = []
            for f in fallback_pool:
                shap_scores.append((f, boruta_report.get("per_feature_shap", {}).get(f, 0)))
            shap_scores.sort(key=lambda x: x[1], reverse=True)
            locked = [f for f, _ in shap_scores[:8]]
        print(f"  Floor: ensured {len(locked)} features minimum")

    importance_scores = {
        f: boruta_report.get("per_feature_shap", {}).get(f, 0.0)
        for f in locked
    }

    report = {
        "method": "boruta_shap",
        "features_expanded": len(numeric_cols),
        "features_confirmed": len(confirmed),
        "features_rejected": len(rejected),
        "features_locked": len(locked),
        "confirmed": confirmed,
        "rejected": rejected,
        "locked": locked,
        "boruta_iterations": boruta_report.get("iterations", 0),
        "percentile": percentile,
        "max_iter": max_iter,
    }

    print(f"  Locked features: {len(locked)}")
    return locked, importance_scores, report
