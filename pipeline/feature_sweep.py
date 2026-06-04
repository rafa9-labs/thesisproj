"""
Phase -1: Feature Sweep — Grid Expansion + Weak RF + Permutation Importance.

Expands a raw OHLC DataFrame into a rich indicator set with multiple window
variants, then trains a shallow Random Forest on next-bar return labels and
uses permutation importance to prune useless features before the heavy ML
pipeline runs.

Design:
  - max_depth=5, n_estimators=100 — RF cannot memorize, only uses obviously
    predictive features.
  - 3-fold time-series split for cross-validation stability.
  - Only features with positive permutation importance in ALL folds survive.
  - Output is a JSON-locked feature list consumed by downstream Phases 0-4.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta


# ── Indicator expansion grid ────────────────────────────────────────

INDICATOR_GRID: Dict[str, List[int]] = {
    "sma": [10, 20, 50, 100, 200],
    "ema": [10, 20, 50, 100, 200],
    "rsi": [7, 14, 21],
    "adx": [7, 14, 28],
    "atr": [7, 14, 28],
    "bbands": [10, 20, 50, 100],
    "donchian": [20, 60],
}

RETURNS_LAGS = [1, 2, 3, 5, 10]


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, dst in [("mid_high", "mid_h"), ("mid_low", "mid_l"),
                     ("mid_close", "mid_c")]:
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def expand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicator variants from INDICATOR_GRID on raw OHLC data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLC with columns: mid_h, mid_l, mid_c (or mid_high/mid_low/mid_close).

    Returns
    -------
    pd.DataFrame
        Original columns plus all indicator columns. NaN in rows where the
        indicator window is not yet satisfied.
    """
    out = _normalize_ohlc(df)
    high = out["mid_h"].astype(np.float64)
    low = out["mid_l"].astype(np.float64)
    close = out["mid_c"].astype(np.float64)

    out["returns"] = np.log(close / close.shift(1)).astype(np.float32)

    out["spread"] = out.get("spread", pd.Series(0.0, index=df.index))

    for w in RETURNS_LAGS:
        out[f"returns_lag{w}"] = out["returns"].shift(w).astype(np.float32)

    for w in INDICATOR_GRID["sma"]:
        out[f"sma_{w}"] = close.rolling(w, min_periods=1).mean().astype(np.float32)
        out[f"price_sma_{w}_ratio"] = (close / out[f"sma_{w}"].replace(0, np.nan)).astype(np.float32)

    for w in INDICATOR_GRID["ema"]:
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean().astype(np.float32)
        out[f"price_ema_{w}_ratio"] = (close / out[f"ema_{w}"].replace(0, np.nan)).astype(np.float32)

    for w in INDICATOR_GRID["rsi"]:
        out[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=close, window=w).rsi().astype(np.float32)

    for w in INDICATOR_GRID["adx"]:
        out[f"adx_{w}"] = ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=w,
        ).adx().astype(np.float32)

    for w in INDICATOR_GRID["atr"]:
        atr_arr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=w,
        ).average_true_range()
        out[f"atr_{w}"] = atr_arr.astype(np.float32)
        out[f"atr_{w}_ratio"] = (atr_arr / close.replace(0, np.nan)).astype(np.float32)

    for w in INDICATOR_GRID["bbands"]:
        bb = ta.volatility.BollingerBands(close=close, window=w, window_dev=2)
        out[f"bb_upper_{w}"] = bb.bollinger_hband().astype(np.float32)
        out[f"bb_lower_{w}"] = bb.bollinger_lband().astype(np.float32)
        out[f"bbw_{w}"] = bb.bollinger_wband().astype(np.float32)
        out[f"bb_pct_{w}"] = bb.bollinger_pband().astype(np.float32)

    for w in INDICATOR_GRID["donchian"]:
        out[f"donchian_up_{w}"] = high.rolling(w).max().astype(np.float32)
        out[f"donchian_dn_{w}"] = low.rolling(w).min().astype(np.float32)
        out[f"donchian_break_up_{w}"] = (
            (close > out[f"donchian_up_{w}"].shift(1)).astype(np.int8)
        )
        out[f"donchian_break_dn_{w}"] = (
            (close < out[f"donchian_dn_{w}"].shift(1)).astype(np.int8)
        )

    # Standard MACD (single variant)
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["macd_diff"] = macd.macd_diff().astype(np.float32)
    out["macd_signal"] = macd.macd_signal().astype(np.float32)

    # Realized volatility at short/long windows
    ret_sq = out["returns"] ** 2
    out["rv_48"] = (ret_sq.rolling(48).sum() ** 0.5).astype(np.float32)
    out["rv_240"] = (ret_sq.rolling(240).sum() ** 0.5).astype(np.float32)

    return out


def _make_labels(df: pd.DataFrame, threshold: float = 0.0001) -> np.ndarray:
    """3-class labels from next-bar returns.

    0 = sell (return < -threshold)
    1 = neutral (|return| <= threshold)
    2 = buy (return > threshold)
    """
    fwd_returns = df["returns"].shift(-1).values
    labels = np.ones(len(df), dtype=np.int32)
    labels[fwd_returns > threshold] = 2
    labels[fwd_returns < -threshold] = 0
    labels[-1] = -1
    return labels


def sweep_features(
    df: pd.DataFrame,
    label_threshold: float = 0.0001,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_folds: int = 3,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[List[str], Dict[str, float], Dict]:
    """Run feature sweep: expand → label → 3-fold time-series RF → prune.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLC data.
    label_threshold : float
        Next-bar return threshold for buy/sell labels (default 1 pip).
    n_estimators : int
        RF estimators (default 100).
    max_depth : int
        Max tree depth — kept shallow (default 5) to prevent memorization.
    n_folds : int
        Time-series folds for cross-validation consistency.
    n_repeats : int
        Permutation shuffles per feature (higher = more stable).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    locked_features : list[str]
        Feature names that survived all folds with positive importance.
    importance_scores : dict[str, float]
        Mean permutation importance per feature across all folds.
    report : dict
        Full report with fold-level details, pruned features, etc.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    print("\n" + "=" * 64)
    print("  PHASE -1: FEATURE SWEEP")
    print("=" * 64)

    rng = np.random.default_rng(random_state)

    df_feat = expand_features(df)
    labels = _make_labels(df_feat, threshold=label_threshold)

    valid = labels != -1
    df_feat = df_feat.loc[valid].copy()
    labels = labels[valid]

    exclude = {"returns", "time", "timestamp", "label",
               "mid_h", "mid_l", "mid_c", "mid_o",
               "mid_high", "mid_low", "mid_close", "mid_open",
               "bid_open", "bid_close", "ask_open", "ask_close",
               "spread", "volume"}
    numeric_cols = [c for c in df_feat.columns
                    if c not in exclude and np.issubdtype(df_feat[c].dtype, np.number)]
    df_feat = df_feat[numeric_cols].copy()

    df_feat = df_feat.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    n = len(df_feat)
    fold_size = n // (n_folds + 1)
    if fold_size < 100:
        fold_size = max(50, n // 3)

    print(f"  Features expanded: {len(numeric_cols)}")
    print(f"  Bars: {n}, Folds: {n_folds}, RF: depth={max_depth}, trees={n_estimators}")

    feature_names = list(df_feat.columns)
    X = df_feat.to_numpy(np.float32)
    y = labels

    fold_importances: List[Dict[str, float]] = []

    for fi in range(n_folds):
        train_end = min((fi + 1) * fold_size, n - fold_size)
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 50:
            continue

        X_train, X_test = X[:train_end], X[test_start:test_end]
        y_train, y_test = y[:train_end], y[test_start:test_end]

        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state + fi, n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)

        perm = permutation_importance(
            rf, X_test, y_test, n_repeats=n_repeats,
            random_state=random_state + fi, scoring="accuracy",
        )

        fold_imp = dict(zip(feature_names, perm.importances_mean))
        fold_importances.append(fold_imp)

        n_pos = int(np.sum(perm.importances_mean > 0))
        acc = rf.score(X_test, y_test)
        print(f"  Fold {fi + 1}/{n_folds}: accuracy={acc:.4f}, "
              f"features with importance>0: {n_pos}/{len(feature_names)}")

    # ── Consensus: positive importance in ALL folds ──
    mean_imp: Dict[str, float] = {}
    consensus: List[str] = []
    all_pos = set(feature_names)
    for fold_imp in fold_importances:
        all_pos &= {k for k, v in fold_imp.items() if v > 0}

    for name in feature_names:
        vals = [fi.get(name, 0.0) for fi in fold_importances]
        mean_imp[name] = float(np.mean(vals))

    consensus = sorted(all_pos, key=lambda x: mean_imp[x], reverse=True)

    pruned = sorted(set(feature_names) - all_pos)

    print(f"\n  Pruned: {len(pruned)} features with zero or negative importance in >=1 fold")
    print(f"  Locked: {len(consensus)} features")

    report = {
        "total_features": len(feature_names),
        "pruned_count": len(pruned),
        "locked_count": len(consensus),
        "pruned_features": pruned,
        "locked_features": consensus,
        "importance_scores": mean_imp,
        "fold_reports": [{
            f"fold_{i}": dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20])
        } for i, fi in enumerate(fold_importances)],
        "config": {
            "label_threshold": label_threshold,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_folds": n_folds,
            "n_repeats": n_repeats,
        },
    }

    return consensus, mean_imp, report


def save_locked_features(features: List[str], path: str) -> str:
    """Save locked feature names to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"locked_features": features, "version": 1}, f, indent=2)
    print(f"  Locked features saved to {path}")
    return path


def load_locked_features(path: str) -> Optional[List[str]]:
    """Load locked feature names from JSON. Returns None if file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("locked_features", None)


def run_phase_minus1(
    df: pd.DataFrame,
    output_path: str = "results/locked_features.json",
    label_threshold: float = 0.0001,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_folds: int = 3,
    random_state: int = 42,
) -> Tuple[List[str], Dict]:
    """Run the full Phase -1 pipeline and save results.

    Returns
    -------
    locked_features : list[str]
        Feature names to use in downstream phases.
    report : dict
        Full sweep report with importance scores and pruned features.
    """
    locked, scores, report = sweep_features(
        df, label_threshold=label_threshold,
        n_estimators=n_estimators, max_depth=max_depth,
        n_folds=n_folds, random_state=random_state,
    )
    save_locked_features(locked, output_path)

    report_path = output_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return locked, report
