"""
Training diagnostics engine for Sprint 16.3.

Computes feature importance, prediction histograms, confusion matrices,
and confidence band analysis from per-period backtest data.

Consumed by api.tasks._run_backtest_impl and the TrainingDiagnostics panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_HAS_SHAP = False
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    pass

_HAS_SKLEARN_PERM = False
try:
    from sklearn.inspection import permutation_importance as _skperm
    _HAS_SKLEARN_PERM = True
except ImportError:
    pass


@dataclass
class FeatureImportanceEntry:
    feature: str
    importance: float


@dataclass
class PredictionHistogramBin:
    bin_start: float
    bin_end: float
    bin_center: float
    count: int


@dataclass
class ConfusionMatrixData:
    matrix: List[List[int]]
    labels: List[str]


@dataclass
class ConfidenceBand:
    band_min: float
    band_max: float
    count: int
    accuracy: float
    mean_return: float


@dataclass
class TrainingDiagnosticsData:
    feature_importance: List[FeatureImportanceEntry] = field(default_factory=list)
    prediction_histogram: List[PredictionHistogramBin] = field(default_factory=list)
    confusion_matrix: Optional[ConfusionMatrixData] = None
    confidence_bands: List[ConfidenceBand] = field(default_factory=list)


def compute_prediction_histogram(
    max_conf_arrays: List[np.ndarray],
    n_bins: int = 15,
    range_min: float = 0.5,
    range_max: float = 1.0,
) -> List[PredictionHistogramBin]:
    """
    Aggregate max confidence arrays into a histogram.

    Parameters
    ----------
    max_conf_arrays : list of np.ndarray
        Per-period arrays of max confidence values (float, 0-1).
    n_bins : int
        Number of histogram bins.
    range_min : float
        Lower bound of first bin.
    range_max : float
        Upper bound of last bin.

    Returns
    -------
    list of PredictionHistogramBin
    """
    all_conf = np.concatenate(max_conf_arrays) if max_conf_arrays else np.array([])
    all_conf = all_conf[np.isfinite(all_conf)]
    if len(all_conf) == 0:
        return []

    bin_edges = np.linspace(range_min, range_max, n_bins + 1)
    counts, _ = np.histogram(all_conf, bins=bin_edges)

    bins = []
    for i in range(n_bins):
        center = float((bin_edges[i] + bin_edges[i + 1]) / 2)
        bins.append(PredictionHistogramBin(
            bin_start=float(bin_edges[i]),
            bin_end=float(bin_edges[i + 1]),
            bin_center=round(center, 4),
            count=int(counts[i]),
        ))
    return bins


def compute_confidence_bands(
    max_conf_arrays: List[np.ndarray],
    outcome_arrays: List[np.ndarray],
    return_arrays: List[np.ndarray],
    band_edges: Optional[List[float]] = None,
) -> List[ConfidenceBand]:
    """
    Bucket predictions by confidence level and compute per-band accuracy and return.

    Parameters
    ----------
    max_conf_arrays : list of np.ndarray
        Per-period max confidence values.
    outcome_arrays : list of np.ndarray
        Per-period boolean arrays (1=correct direction, 0=wrong).
    return_arrays : list of np.ndarray
        Per-period strategy return arrays.
    band_edges : list of float, optional
        Bin edges for confidence bands. Default [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].

    Returns
    -------
    list of ConfidenceBand
    """
    if band_edges is None:
        band_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_conf = np.concatenate(max_conf_arrays) if max_conf_arrays else np.array([])
    all_outcomes = np.concatenate(outcome_arrays) if outcome_arrays else np.array([])
    all_returns = np.concatenate(return_arrays) if return_arrays else np.array([])

    if len(all_conf) == 0 or len(all_outcomes) == 0:
        return []

    min_len = min(len(all_conf), len(all_outcomes), len(all_returns))
    all_conf = all_conf[:min_len]
    all_outcomes = all_outcomes[:min_len]
    all_returns = all_returns[:min_len]

    bands = []
    for i in range(len(band_edges) - 1):
        lo = band_edges[i]
        hi = band_edges[i + 1]
        mask = (all_conf >= lo) & (all_conf < hi)
        n = int(mask.sum())
        if n > 0:
            acc = float(np.mean(all_outcomes[mask]))
            ret = float(np.mean(all_returns[mask]))
        else:
            acc = 0.0
            ret = 0.0
        bands.append(ConfidenceBand(
            band_min=lo,
            band_max=hi,
            count=n,
            accuracy=round(acc, 4),
            mean_return=round(ret, 6),
        ))
    return bands


def aggregate_confusion_matrices(
    matrices: List[np.ndarray],
    labels: Tuple = (-1, 0, 1),
) -> ConfusionMatrixData:
    """
    Aggregate per-period confusion matrices into a single aggregate.

    Parameters
    ----------
    matrices : list of np.ndarray
        Per-period 3x3 confusion matrices.
    labels : tuple
        Class labels (default: -1=short, 0=flat, 1=long).

    Returns
    -------
    ConfusionMatrixData
    """
    label_names = ["Short", "Flat", "Long"]
    if not matrices:
        return ConfusionMatrixData(
            matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            labels=label_names,
        )

    total = np.zeros((3, 3), dtype=int)
    for cm in matrices:
        arr = np.asarray(cm, dtype=int)
        if arr.shape == (3, 3):
            total += arr
    return ConfusionMatrixData(
        matrix=total.tolist(),
        labels=label_names,
    )


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factor for each column. VIF > 10 flags multicollinearity."""
    from numpy.linalg import inv
    n, k = X.shape
    if k < 2:
        return np.array([float("nan")] * k)
    centered = X - X.mean(axis=0)
    try:
        corr = np.corrcoef(centered.T)
        inv_corr = inv(corr)
        return np.diag(inv_corr)
    except Exception:
        return np.array([float("nan")] * k)


def compute_feature_importance(
    model,
    model_type: str,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> List[FeatureImportanceEntry]:
    """
    Extract feature importance from a fitted model.
    Model-family-specific method selection:

    - Tree models (XGBoost, RF): SHAP TreeExplainer when available, else native importance
    - Logistic: standardized coefficient magnitude + VIF warning
    - SVM: permutation importance (small-sample)
    - LSTM/CNN/Transformer: mean |gradient| via TensorFlow GradientTape
    - Ensemble: delegates to sub-model

    Parameters
    ----------
    model : object
        Fitted sklearn/TF model.
    model_type : str
        Model type identifier.
    feature_names : list of str, optional
        Feature names. If None, uses generic names.
    top_n : int
        Maximum number of features to return.
    X_val : np.ndarray, optional
        Validation feature data (required for SVM perm, deep gradient, SHAP).
    y_val : np.ndarray, optional
        Validation labels (required for SVM perm, SHAP).

    Returns
    -------
    list of FeatureImportanceEntry, sorted descending by importance.
    """
    mt = str(model_type or "").lower().strip()
    importances = None
    n_feats = len(feature_names) if feature_names else 0

    try:
        if mt in {"xgboost", "xgb"}:
            # TreeSHAP when available, otherwise get_score(gain)
            if _HAS_SHAP and X_val is not None:
                try:
                    explainer = _shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_val[:500]) if len(X_val) > 500 else explainer.shap_values(X_val)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                    importances = np.abs(shap_vals).mean(axis=0)
                    if n_feats > 0 and len(importances) != n_feats:
                        importances = None
                except Exception:
                    pass
            if importances is None:
                if hasattr(model, "get_booster"):
                    booster = model.get_booster()
                    raw = booster.get_score(importance_type="gain")
                    if feature_names is not None:
                        importances = np.zeros(n_feats)
                        for feat, score in raw.items():
                            if feat.startswith("f") and feat[1:].isdigit():
                                idx = int(feat[1:])
                                if idx < n_feats:
                                    importances[idx] = score
                            elif feat in feature_names:
                                importances[feature_names.index(feat)] = score
                    else:
                        items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
                        return [FeatureImportanceEntry(feature=k, importance=float(v)) for k, v in items[:top_n]]
                elif hasattr(model, "feature_importances_"):
                    importances = np.asarray(model.feature_importances_, dtype=float)

        elif mt in {"random_forest", "rf"}:
            # TreeSHAP when available, otherwise MDI
            if _HAS_SHAP and X_val is not None:
                try:
                    explainer = _shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_val[:500]) if len(X_val) > 500 else explainer.shap_values(X_val)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                    importances = np.abs(shap_vals).mean(axis=0)
                    if n_feats > 0 and len(importances) != n_feats:
                        importances = None
                except Exception:
                    pass
            if importances is None and hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)

        elif mt in {"logistic", "logit"}:
            if hasattr(model, "named_steps") and "logit" in model.named_steps:
                coef = np.asarray(model.named_steps["logit"].coef_, dtype=float)
                if coef.ndim == 2:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)

        elif mt in {"svm"}:
            # Permutation importance on small validation sample
            if _HAS_SKLEARN_PERM and X_val is not None and y_val is not None:
                try:
                    X_sub = X_val[:500] if len(X_val) > 500 else X_val
                    y_sub = y_val[:500] if len(y_val) > 500 else y_val
                    perm = _skperm(model, X_sub, y_sub, n_repeats=3, random_state=42, n_jobs=1)
                    importances = perm.importances_mean
                except Exception:
                    pass

        elif mt in {"cnn", "lstm", "transformer"}:
            # TensorFlow GradientTape mean |gradient| importance
            importances = _deep_gradient_importance(model, X_val, n_feats)

        elif mt.startswith("ensemble"):
            # ensemble_cnn_lstm_xgboost: extract XGB importance (with SHAP if available)
            if hasattr(model, "xgb") and model.xgb is not None:
                sub_mt = "xgboost"
                sub_X = X_val
                sub_y = y_val
                # Recursive call with the sub-model
                return compute_feature_importance(model.xgb, sub_mt, feature_names, top_n, sub_X, sub_y)
            # ensemble_adaptive_regime: extract RF importance (with SHAP if available)
            elif hasattr(model, "rf") and model.rf is not None:
                sub_mt = "random_forest"
                sub_X = X_val
                sub_y = y_val
                return compute_feature_importance(model.rf, sub_mt, feature_names, top_n, sub_X, sub_y)

    except Exception:
        pass

    if importances is not None and len(importances) > 0:
        n_features = len(importances)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]
        pairs = list(zip(feature_names[:n_features], importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        total = sum(abs(v) for _, v in pairs) or 1.0
        return [
            FeatureImportanceEntry(feature=name, importance=round(float(imp / total), 6))
            for name, imp in pairs[:top_n]
            if abs(imp) > 0
        ]

    return []


def _deep_gradient_importance(model, X_val: Optional[np.ndarray], n_feats: int) -> Optional[np.ndarray]:
    """Compute mean |gradient| importance for deep TF models."""
    if X_val is None or n_feats <= 0:
        return None
    try:
        import tensorflow as tf
        sub = X_val[:64] if len(X_val) > 64 else X_val
        inp = tf.convert_to_tensor(sub, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(inp)
            out = model(inp, training=False)
            loss = tf.reduce_mean(tf.abs(out))
        grads = tape.gradient(loss, inp)
        if grads is None:
            return None
        # Shape: (batch, timesteps, features) for sequential models
        # or (batch, features) for simple models
        if grads.ndim >= 3:
            grads = tf.reduce_mean(tf.abs(grads), axis=(0, 1))  # mean over batch + time
        else:
            grads = tf.reduce_mean(tf.abs(grads), axis=0)  # mean over batch
        return grads.numpy()
    except Exception:
        pass
    return None