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
from typing import Callable, Dict, List, Optional, Tuple

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
    "stochastic": [7, 14, 21],
    "kaufman_er": [10, 20, 50],
    "hv": [5, 10, 20, 50],
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
        try:
            out[f"rsi_{w}"] = ta.momentum.RSIIndicator(close=close, window=w).rsi().astype(np.float32)
        except Exception:
            out[f"rsi_{w}"] = np.full(len(out), 50.0, dtype=np.float32)

    for w in INDICATOR_GRID["adx"]:
        try:
            out[f"adx_{w}"] = ta.trend.ADXIndicator(
                high=high, low=low, close=close, window=w,
            ).adx().astype(np.float32)
        except Exception:
            out[f"adx_{w}"] = np.full(len(out), 20.0, dtype=np.float32)

    for w in INDICATOR_GRID["atr"]:
        try:
            atr_arr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=w,
            ).average_true_range()
            out[f"atr_{w}"] = atr_arr.astype(np.float32)
            out[f"atr_{w}_ratio"] = (atr_arr / close.replace(0, np.nan)).astype(np.float32)
        except Exception:
            out[f"atr_{w}"] = np.full(len(out), 0.001, dtype=np.float32)
            out[f"atr_{w}_ratio"] = np.full(len(out), 0.001, dtype=np.float32)

    for w in INDICATOR_GRID["bbands"]:
        try:
            bb = ta.volatility.BollingerBands(close=close, window=w, window_dev=2)
            out[f"bb_upper_{w}"] = bb.bollinger_hband().astype(np.float32)
            out[f"bb_lower_{w}"] = bb.bollinger_lband().astype(np.float32)
            out[f"bbw_{w}"] = bb.bollinger_wband().astype(np.float32)
            out[f"bb_pct_{w}"] = bb.bollinger_pband().astype(np.float32)
        except Exception:
            out[f"bb_upper_{w}"] = close.astype(np.float32) * 1.02
            out[f"bb_lower_{w}"] = close.astype(np.float32) * 0.98
            out[f"bbw_{w}"] = np.full(len(out), 0.04, dtype=np.float32)
            out[f"bb_pct_{w}"] = np.full(len(out), 0.5, dtype=np.float32)

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
    try:
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        out["macd_diff"] = macd.macd_diff().astype(np.float32)
        out["macd_signal"] = macd.macd_signal().astype(np.float32)
    except Exception:
        out["macd_diff"] = np.zeros(len(out), dtype=np.float32)
        out["macd_signal"] = np.zeros(len(out), dtype=np.float32)

    # Stochastic oscillator %K and %D (Lane, 1950s)
    for w in INDICATOR_GRID["stochastic"]:
        roll_high = high.rolling(w).max()
        roll_low = low.rolling(w).min()
        raw_k = 100.0 * (close - roll_low) / (roll_high - roll_low + 1e-12)
        out[f"stoch_k_{w}"] = raw_k.astype(np.float32)
        out[f"stoch_d_{w}"] = raw_k.rolling(3, min_periods=1).mean().astype(np.float32)

    # Kaufman Efficiency Ratio — directional efficiency (Kaufman, 2013)
    for w in INDICATOR_GRID["kaufman_er"]:
        net_change = np.abs(close - close.shift(w))
        path_length = np.abs(close.diff()).rolling(w, min_periods=1).sum()
        out[f"er_{w}"] = (net_change / (path_length + 1e-12)).astype(np.float32)

    # Historical volatility — rolling std of returns at short horizons
    for w in INDICATOR_GRID["hv"]:
        out[f"hv_{w}"] = out["returns"].rolling(w, min_periods=2).std().astype(np.float32)

    # Realized volatility at short/long windows
    ret_sq = out["returns"] ** 2
    out["rv_48"] = (ret_sq.rolling(48).sum() ** 0.5).astype(np.float32)
    out["rv_240"] = (ret_sq.rolling(240).sum() ** 0.5).astype(np.float32)

    return out


def _make_labels(df: pd.DataFrame, threshold: float = 0.0001) -> np.ndarray:
    """3-class labels from next-bar returns.

    Convention (unified with boruta_sweep and the pipeline):
    -1 = sell (return < -threshold)
     0 = neutral (|return| <= threshold)
    +1 = buy (return > threshold)
    The last bar has no forward return and is labelled neutral (0); callers
    should drop it before fitting.
    """
    fwd_returns = df["returns"].shift(-1).values
    labels = np.zeros(len(df), dtype=np.int32)
    labels[fwd_returns > threshold] = 1
    labels[fwd_returns < -threshold] = -1
    return labels


def sweep_features(
    df: pd.DataFrame,
    label_threshold: float = 0.0001,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_folds: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_boruta: bool = True,
    boruta_percentile: int = 90,
    boruta_max_iter: int = 20,
    boruta_economic_floor_pct: float = 0.02,
) -> Tuple[List[str], Dict[str, float], Dict]:
    """Run feature sweep: expand -> label -> BorutaSHAP or majority-vote RF -> MI filter.

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
    use_boruta : bool
        If True (default), use BorutaSHAP (game-theoretic, shadow features).
        If False, use legacy majority-vote permutation importance.
    boruta_percentile : int
        Confirmation percentile for BorutaSHAP (90 = must beat noise in >=90% of folds).
    boruta_max_iter : int
        Maximum Boruta iterations before forced convergence.

    Returns
    -------
    locked_features : list[str]
        Feature names that passed selection.
    importance_scores : dict[str, float]
        Mean importance per locked feature.
    report : dict
        Full report with fold-level details, pruned features, etc.
    """
    if use_boruta:
        from pipeline.features.boruta_sweep import boruta_sweep_features
        return boruta_sweep_features(
            df, label_threshold=label_threshold,
            n_estimators=n_estimators, max_depth=max_depth,
            n_folds=n_folds, percentile=boruta_percentile,
            max_iter=boruta_max_iter, random_state=random_state,
            progress_callback=progress_callback,
            economic_floor_pct=boruta_economic_floor_pct,
        )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.feature_selection import mutual_info_classif

    print("\n" + "=" * 64)
    print("  PHASE -1: FEATURE SWEEP")
    print("=" * 64)

    rng = np.random.default_rng(random_state)

    df_feat = expand_features(df)
    labels = _make_labels(df_feat, threshold=label_threshold)

    # Drop only the last bar (no forward label exists for it). Neutral bars
    # are kept so selection reflects the full 3-class problem.
    df_feat = df_feat.iloc[:-1].copy()
    labels = labels[:-1]

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

    print(f"      Features expanded: {len(numeric_cols)}")
    print(f"  Bars: {n}, Folds: {n_folds}, RF: depth={max_depth}, trees={n_estimators}")
    if progress_callback:
        progress_callback(f"Expanded {len(numeric_cols)} features across {n} bars")

    feature_names = list(df_feat.columns)
    X = df_feat.to_numpy(np.float32)
    y = labels

    fold_importances: List[Dict[str, float]] = []

    for fi in range(n_folds):
        split_at = min((fi + 1) * fold_size, n - fold_size)
        # Purge 1 bar: labels are next-bar returns, so the last train bar's
        # label window touches the test block.
        train_end = max(0, split_at - 1)
        test_start = split_at
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 50:
            continue

        if train_end < 50:
            # Degenerate fold geometry (tiny datasets): not enough training
            # history for a reliable fold — skip it.
            continue

        X_train, X_test = X[:train_end], X[test_start:test_end]
        y_train, y_test = y[:train_end], y[test_start:test_end]

        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state + fi, n_jobs=-1,
            class_weight="balanced",
        )
        if progress_callback:
            progress_callback(f"Fold {fi + 1}/{n_folds}: training RF...")
        rf.fit(X_train, y_train)

        if progress_callback:
            progress_callback(f"Fold {fi + 1}/{n_folds}: permutation importance ({n_repeats} repeats)...")
        perm = permutation_importance(
            rf, X_test, y_test, n_repeats=n_repeats,
            random_state=random_state + fi, scoring="accuracy",
        )

        fold_imp = dict(zip(feature_names, perm.importances_mean))
        fold_importances.append(fold_imp)

        n_pos = int(np.sum(perm.importances_mean > 0))
        acc = rf.score(X_test, y_test)
        msg = f"Fold {fi + 1}/{n_folds}: accuracy={acc:.4f}, features with importance>0: {n_pos}/{len(feature_names)}"
        print(f"  {msg}")
        if progress_callback:
            progress_callback(msg)

    # -- Majority vote: feature survives if importance > 0 in >= majority of folds --
    majority_threshold = max(1, (len(fold_importances) + 1) // 2)  # ceil(n/2)
    mean_imp: Dict[str, float] = {}
    pos_count: Dict[str, int] = {}

    for name in feature_names:
        vals = [fi.get(name, 0.0) for fi in fold_importances]
        mean_imp[name] = float(np.mean(vals))
        pos_count[name] = sum(1 for v in vals if v > 0)

    consensus = sorted(
        [n for n in feature_names if pos_count[n] >= majority_threshold],
        key=lambda x: mean_imp[x], reverse=True,
    )

    pruned = sorted(set(feature_names) - set(consensus))

    print(f"\n  Pruned (majority vote < {majority_threshold}/{len(fold_importances)} folds): {len(pruned)} features")
    print(f"  Survived: {len(consensus)} features")

    # -- Stage 2: Mutual information filter (catches nonlinear patterns RF misses) --
    if len(consensus) > 50:
        X_consensus = df_feat[consensus].to_numpy(np.float32)
        mi_scores = mutual_info_classif(
            X_consensus, y, discrete_features=False,
            random_state=random_state,
        )
        mi_ranked = sorted(zip(consensus, mi_scores), key=lambda x: x[1], reverse=True)
        consensus = [n for n, _ in mi_ranked[:50]]
        print(f"  MI filter: capped at 50 features")

    # -- Stage 3: Minimum floor --
    MIN_FEATURES = 8
    if len(consensus) < MIN_FEATURES and len(feature_names) >= MIN_FEATURES:
        # Fall back to top-N by mean importance for the difference
        fallback = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)
        fallback = [n for n, _ in fallback if n not in consensus][:MIN_FEATURES - len(consensus)]
        consensus = sorted(set(consensus + fallback), key=lambda x: mean_imp[x], reverse=True)
        print(f"  Feature floor: {len(consensus)} features (added {len(fallback)} by mean importance)")

    if progress_callback:
        progress_callback(f"Feature sweep complete: {len(consensus)} locked, {len(pruned)} pruned")
        top5_str = ", ".join(f"{n}={mean_imp.get(n, 0):.4f}" for n in consensus[:5])
        progress_callback(f"Top 5 by importance: {top5_str}")

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
            "signature": f"n{n_estimators}_d{max_depth}_f{n_folds}",
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
    n_folds: int = 5,
    random_state: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_boruta: bool = True,
    boruta_percentile: int = 90,
    boruta_max_iter: int = 20,
) -> Tuple[List[str], Dict]:
    """Run the full Phase -1 pipeline and save results.

    Returns
    -------
    locked_features : list[str]
        Feature names to use in downstream phases.
    report : dict
        Full sweep report with importance scores and pruned features.
    """
    if progress_callback:
        progress_callback("Expanding features...")
    locked, scores, report = sweep_features(
        df, label_threshold=label_threshold,
        n_estimators=n_estimators, max_depth=max_depth,
        n_folds=n_folds, random_state=random_state,
        progress_callback=progress_callback,
        use_boruta=use_boruta,
        boruta_percentile=boruta_percentile,
        boruta_max_iter=boruta_max_iter,
    )
    save_locked_features(locked, output_path)

    report_path = output_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return locked, report


def _generate_feature_names() -> list[str]:
    """Derive the canonical list of indicator columns from INDICATOR_GRID."""
    names: list[str] = []
    for w in RETURNS_LAGS:
        names.append(f"returns_lag{w}")
    for w in INDICATOR_GRID["sma"]:
        names.append(f"sma_{w}")
        names.append(f"price_sma_{w}_ratio")
    for w in INDICATOR_GRID["ema"]:
        names.append(f"ema_{w}")
        names.append(f"price_ema_{w}_ratio")
    for w in INDICATOR_GRID["rsi"]:
        names.append(f"rsi_{w}")
    for w in INDICATOR_GRID["adx"]:
        names.append(f"adx_{w}")
    for w in INDICATOR_GRID["atr"]:
        names.append(f"atr_{w}")
        names.append(f"atr_{w}_ratio")
    for w in INDICATOR_GRID["bbands"]:
        names.append(f"bb_upper_{w}")
        names.append(f"bb_lower_{w}")
        names.append(f"bbw_{w}")
        names.append(f"bb_pct_{w}")
    for w in INDICATOR_GRID["donchian"]:
        names.append(f"donchian_up_{w}")
        names.append(f"donchian_dn_{w}")
        names.append(f"donchian_break_up_{w}")
        names.append(f"donchian_break_dn_{w}")
    names.append("macd_diff")
    names.append("macd_signal")
    names.append("rv_48")
    names.append("rv_240")
    for w in INDICATOR_GRID["stochastic"]:
        names.append(f"stoch_k_{w}")
        names.append(f"stoch_d_{w}")
    for w in INDICATOR_GRID["kaufman_er"]:
        names.append(f"er_{w}")
    for w in INDICATOR_GRID["hv"]:
        names.append(f"hv_{w}")
    return sorted(names)


FEATURE_NAMES: list[str] = _generate_feature_names()
"""All indicator column names produced by expand_features() (sorted)."""


def compute_feature_matrix(
    df: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
    include_ohlc: bool = True,
) -> pd.DataFrame:
    """Expand OHLC data into indicator features and filter to a requested set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLC with columns: mid_h, mid_l, mid_c
        (or mid_high/mid_low/mid_close — auto-renamed).
    feature_names : list[str] or None
        Requested feature columns. If None, returns all FEATURE_NAMES.
    include_ohlc : bool
        If True, also include mid_o, mid_h, mid_l, mid_c columns in output.

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix with NaN/inf filled to 0.0, dtype float32.
    """
    expanded = expand_features(df)
    names = list(feature_names) if feature_names else list(FEATURE_NAMES)
    ohlc = ["mid_o", "mid_h", "mid_l", "mid_c"] if include_ohlc else []
    available = [c for c in names + ohlc if c in expanded.columns]
    result = expanded[available].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return result.astype(np.float32)
