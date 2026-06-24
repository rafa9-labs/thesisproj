"""Feature engineering utilities backported from utilsNoWFO.py.

Phase 4.2a -- feature builders, selectors, and transformers.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence

try:
    from sklearn.feature_selection import mutual_info_classif
except ImportError:
    mutual_info_classif = None

from logging_config import log_print


def _fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
    """Compute fracdiff weight vector; trim trailing weights below thresh."""
    w = [1.0]
    for k in range(1, size):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
    return np.array(w, dtype="float64")


def add_cyclic_hour_features(df: pd.DataFrame, hour_col: str = "hour") -> pd.DataFrame:
    """Add sin/cos encodings for hour-of-day (0..23). Assumes hour_col exists and is integer."""
    if hour_col in df:
        df["hour_sin"] = np.sin(2 * np.pi * df[hour_col] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df[hour_col] / 24.0)
    return df


def fracdiff(series: pd.Series, d: float = 0.4, max_size: int = 2000, thresh: float = 1e-4) -> pd.Series:
    """
    Fractionally difference a price/return series (no future info).
    Output aligned to original index with NaNs for the warmup.
    """
    series = series.astype("float64")
    w = _fracdiff_weights(d, min(max_size, len(series)), thresh=thresh)
    out = np.full_like(series.values, fill_value=np.nan, dtype="float64")
    k_max = len(w) - 1
    for t in range(k_max, len(series)):
        window = series.values[t - k_max : t + 1]
        out[t] = np.dot(w[::-1], window)
    return pd.Series(out, index=series.index, name=f"fd_{series.name}_d{d:.2f}")


def find_min_stationary_d(
    price: pd.Series,
    d_range: tuple = (0.05, 0.95, 0.05),
) -> float:
    """Find minimum fractional differentiation order that achieves stationarity.

    Per de Prado AFML Ch.5: ADF test determines the smallest d where
    p-value < 0.05, preserving maximum memory in the differentiated series.
    This floor prevents Optuna from selecting sub-stationarity d values
    that produce artificially inflated backtest Sharpe from trending bias.

    Parameters
    ----------
    price : pd.Series
        Raw price series (e.g. mid_c).
    d_range : tuple
        (start, stop, step) for d grid search.

    Returns
    -------
    float
        Minimum stationary d, or 0.4 if ADF never passes.
    """
    from statsmodels.tsa.stattools import adfuller

    price_clean = price.dropna().astype(np.float64)
    if len(price_clean) < 100:
        return 0.4

    for d in np.arange(*d_range):
        try:
            fd = fracdiff(price_clean, d=d)
            fd_clean = fd.dropna()
            if len(fd_clean) < 50:
                continue
            _, p_value, *_ = adfuller(fd_clean.values, maxlag=int(min(30, len(fd_clean) / 4)), autolag="AIC")
            if p_value < 0.05:
                return round(float(d), 2)
        except Exception:
            continue

    return 0.4


def triple_barrier_labels(
    close: pd.Series,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_holding: int = 48,
    vol: Optional[pd.Series] = None,
    neutral_zone: float = 0.0,
    neutral_zone_is_sigma: bool = False,
) -> pd.Series:
    """
    3-class labels via triple barrier.
    Returns {0: down/short, 2: up/long, 1: neutral/hold} to match the existing 0/1/2 mapping used elsewhere (2=long).
    If neutral_zone > 0, small outcomes inside [-neutral_zone, +neutral_zone] become '2'.

    If neutral_zone_is_sigma is True, `neutral_zone` is interpreted as a
    multiplier on the local volatility series, so the per-bar neutral band is
    neutral_zone * vol[t] (vol must be on the same grid as `close`).
    """
    close = close.astype("float64")
    if vol is None:
        vol = realized_vol(close.pct_change().fillna(0), window=30).bfill().fillna(0)
    pt = pt_mult * vol
    sl = sl_mult * vol
    
    # Optional volatility-scaled neutral band
    if neutral_zone_is_sigma:
        # Per-bar neutral band: k * vol[t]
        nz_series = (neutral_zone * vol).astype("float64")
    else:
        nz_series = None

    n = len(close)
    y = np.full(n, 2, dtype="int8")  # default neutral

    for t0 in range(n):
        c0 = close.iloc[t0]
        ub = c0 * (1.0 + pt.iloc[t0])
        lb = c0 * (1.0 - sl.iloc[t0])
        t1 = min(n - 1, t0 + max_holding)
        path = close.iloc[t0 : t1 + 1]

        hit_up = (path >= ub)
        hit_dn = (path <= lb)
        first_up = path.index[hit_up.argmax()] if hit_up.any() else None
        first_dn = path.index[hit_dn.argmax()] if hit_dn.any() else None

        if first_up is not None and (first_dn is None or first_up <= first_dn):
            y[t0] = 2       # LONG
        elif first_dn is not None and (first_up is None or first_dn < first_up):
            y[t0] = 0       # SHORT
        else:
            ret = (path.iloc[-1] / c0) - 1.0
            # Decide neutral band: either absolute or volatility-scaled
            if neutral_zone_is_sigma and nz_series is not None:
                nz = float(nz_series.iloc[t0])
            else:
                nz = float(neutral_zone)

            if ret > nz:
                y[t0] = 2   # LONG
            elif ret < -nz:
                y[t0] = 0   # SHORT
            else:
                y[t0] = 1   # NEUTRAL
                
    return pd.Series(y, index=close.index, name="label_tb")


def attach_macro_features(
    df_bars: pd.DataFrame,
    macro_specs,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Attach daily or lower-frequency macro features to an intraday bar DataFrame.

    Typical use-case:
    - df_bars: 30-minute EUR/USD bars with a tz-aware DatetimeIndex.
    - macro_specs: either
        * dict {label: csv_path}, or
        * list/tuple of csv_paths (label inferred from filename).
    - Each macro CSV should contain a date/time column (e.g. 'date' or 'time')
      and one or more numeric columns with macro values.

    Behaviour:
    - Parse macro dates, normalise to daily (midnight, no tz).
    - Optionally apply a lag of `lag_days` (shift values) so that bar-day t
      sees macro information from day t - lag_days (to avoid look-ahead).
    - Align to bar dates (index normalised to daily, tz-stripped), then
      forward-fill within and across days.
    - Columns are joined onto df_bars; when multiple sources are provided,
      their numeric columns are concatenated. If a dict is used, labels are
      taken from the dict keys; otherwise, filenames are used as prefixes.
    """
    import os
    import pandas as pd

    if df_bars is None or len(df_bars) == 0:
        return df_bars

    if macro_specs is None or macro_specs == {}:
        return df_bars

    # Normalise macro_specs into an iterable of (label, path)
    items: list[tuple[str, str]] = []
    if isinstance(macro_specs, dict):
        for key, path in macro_specs.items():
            if path is None:
                continue
            items.append((str(key), str(path)))
    else:
        # treat as list/tuple/single path
        if not isinstance(macro_specs, (list, tuple)):
            macro_specs = [macro_specs]
        for path in macro_specs:
            if not path:
                continue
            label = os.path.splitext(os.path.basename(str(path)))[0]
            items.append((label, str(path)))

    if not items:
        return df_bars

    # Prepare bar-level date index (naive daily)
    try:
        bar_dates = pd.to_datetime(df_bars.index).tz_convert(None).normalize()
    except Exception:
        bar_dates = pd.to_datetime(df_bars.index).normalize()

    macro_frames: list[pd.DataFrame] = []

    for label, path in items:
        try:
            mdf = pd.read_csv(path)
        except Exception:
            continue
        if mdf is None or len(mdf) == 0:
            continue

        # Heuristic: find a date-like column
        date_col = None
        for cand in ("date", "Date", "DATE", "time", "Time", "TIME"):
            if cand in mdf.columns:
                date_col = cand
                break
        if date_col is None:
            # fall back to first column
            date_col = mdf.columns[0]

        try:
            mdf[date_col] = pd.to_datetime(mdf[date_col], errors="coerce").dt.normalize()
        except Exception:
            continue
        mdf = mdf.set_index(date_col).sort_index()

        # Keep only numeric macro columns
        mdf = mdf.select_dtypes(include=["number"])
        if mdf.shape[1] == 0:
            continue

        # Apply daily lag by shifting values (not index)
        if lag_days and lag_days > 0:
            mdf = mdf.shift(int(lag_days))

        # Align to bar dates and forward-fill
        mdf_aligned = mdf.reindex(bar_dates).ffill()
        if mdf_aligned is None or len(mdf_aligned) == 0:
            continue

        # Use a prefix when multiple macro sources are present
        if len(items) > 1:
            mdf_aligned = mdf_aligned.add_prefix(f"{label}_")

        # Put back on the original bar index
        mdf_aligned.index = df_bars.index
        macro_frames.append(mdf_aligned)

    if not macro_frames:
        return df_bars

    macro_all = pd.concat(macro_frames, axis=1)
    # Avoid clobbering existing columns: only add new ones
    new_cols = [c for c in macro_all.columns if c not in df_bars.columns]
    if not new_cols:
        return df_bars

    return df_bars.join(macro_all[new_cols], how="left")


def build_features_from_params(df, params: dict, base_features):
    """
    Select concrete feature column names from (df, params, base_features).

    Strategy
    --------
    1) Start from any provided `base_features` (if any).
    2) Add indicator columns when toggled in `params` and present in `df`.
    3) Expand each selected indicator with lags and rolling statistics:
       - lags:    _lag{k} for k in [1..lag_depth]
       - rolling: _rollmean{w}, _rollstd{w}, _rollslope{w} for w in `roll_windows`
    4) Optionally include raw returns lags up to `lags` or `lags_range`.

    Notes
    -----
    - This function only *selects* names that already exist in `df`.
      If a requested column doesn't exist, we skip it (with a warning).
    - Keep this list purely declarative; do not mutate `df` here.

    Parameters
    ----------
    df : pandas.DataFrame
        Engineered feature frame (already contains candidate columns).
    params : dict
        Trial/sample parameters (expects keys like 'indicator_windows',
        'use_sma', 'use_ema', 'use_macd', etc., plus 'lag_depth', 'roll_windows',
        'lags' or 'lags_range', and 'include_raw_lags').
    base_features : iterable or None
        Baseline features to always include (if present in `df`).

    Returns
    -------
    list[str]
        Final ordered list of feature column names.
    """
    features = list(base_features or [])
    ind_win = params.get("indicator_windows", {}) or {}

    def add_feat(name: str) -> None:
        """Append `name` if the column exists and isn't already selected."""
        if name in df.columns and name not in features:
            features.append(name)
        # elif name not in df.columns:
        #     print(f"[WARN] build_features_from_params: '{name}' not in df.columns, skipping.")

    # 1) One-off indicators (controlled by toggles in params)
    if params.get("use_sma", False) and "sma" in ind_win:
        add_feat(f"sma_{ind_win['sma']}")
    if params.get("use_ema", False) and "ema" in ind_win:
        add_feat(f"ema_{ind_win['ema']}")
    if params.get("use_macd", False):
        for macd_col in ("macd_line", "macd_signal", "macd_diff"):
            add_feat(macd_col)
    if params.get("use_rsi", False) and "rsi" in ind_win:
        add_feat(f"rsi_{ind_win['rsi']}")
    if params.get("use_bbands", False):
        for bb_col in ("bb_upper", "bb_lower", "bb_pct", "bbw"):
            add_feat(bb_col)
    if params.get("use_atr", False) and "atr" in ind_win:
        add_feat(f"atr_{ind_win['atr']}")
    if params.get("use_adx", False) and "adx" in ind_win:
        add_feat(f"adx_{ind_win['adx']}")
    if params.get("use_stoch", False):
        for st_col in ("stoch_k", "stoch_d"):
            add_feat(st_col)
    if params.get("use_mtf_ma", False):
        for mtf_col in ("mtf_ma_fast", "mtf_ma_slow"):
            add_feat(mtf_col)

    # Optional SAR/Hour
    if "sar" in df.columns:
        add_feat("sar")
    if params.get("include_hour", False) and "hour" in df.columns:
        add_feat("hour")

    # 2) Window expansions
    lag_depth = int(params.get("lag_depth", 1))

    # accept list OR key strings
    roll_windows = params.get("roll_windows")
    if not roll_windows:
        rk = params.get("roll_windows_key_v2") or params.get("roll_windows_key")
        if rk:
            roll_windows = [int(x) for x in str(rk).split(",") if str(x).strip().isdigit()]
    if not roll_windows:
        roll_windows = [5]
    roll_windows = list(roll_windows)

    num_lags = int(params.get("lags", params.get("lags_range", 1)))

    # --- Feature budget clamp to avoid OOM from expansion ---
    MAX_EXPANDED = int(os.environ.get("MAX_EXPANDED_FEATURES", "5500"))
    import math as _math
    def _estimate_expanded(n_base, lag_depth, roll_windows, num_lags):
        per_feat = 1 + int(lag_depth) + 3 * len(roll_windows)
        return n_base * per_feat + int(num_lags)
    _est = _estimate_expanded(len(features), int(params.get("lag_depth", 1)), list(params.get("roll_windows", [])), int(params.get("lags", params.get("lags_range", 1))))
    _lag_depth = int(params.get("lag_depth", 1))
    _roll_windows = list(params.get("roll_windows", []))
    _num_lags = int(params.get("lags", params.get("lags_range", 1)))
    while _est > MAX_EXPANDED and _lag_depth > 1:
        _lag_depth -= 1
        _est = _estimate_expanded(len(features), _lag_depth, _roll_windows, _num_lags)
    while _est > MAX_EXPANDED and len(_roll_windows) > 1:
        _roll_windows = sorted(_roll_windows)[:max(1, len(_roll_windows)-1)]
        _est = _estimate_expanded(len(features), _lag_depth, _roll_windows, _num_lags)
    while _est > MAX_EXPANDED and _num_lags > 10:
        _num_lags = max(10, int(_math.floor(_num_lags * 0.8)))
        _est = _estimate_expanded(len(features), _lag_depth, _roll_windows, _num_lags)
    # overwrite local params for expansion below
    lag_depth = _lag_depth
    roll_windows = _roll_windows
    num_lags = _num_lags

    # Expand lags and rolling stats for each current feature
    for feat in list(features):
        for k in range(1, lag_depth + 1):
            add_feat(f"{feat}_lag{k}")
        for w in roll_windows:
            add_feat(f"{feat}_rollmean{w}")
            add_feat(f"{feat}_rollstd{w}")
            add_feat(f"{feat}_rollslope{w}")

    # 3) Raw returns lags
    if params.get("include_raw_lags", True):
        for lag in range(1, num_lags + 1):
            add_feat(f"returns_lag{lag}")

    if os.environ.get("LOG_MODE", "COMPACT").upper() == "DEBUG":
        # Only show in LOG_MODE=DEBUG
        log_print(
            "build_features_from_params selected features: " + str(features),
            level="DEBUG",
        )
        
    return features


def select_topk_by_mutual_info(X: pd.DataFrame, y: pd.Series, top_k: int = 64, random_state: int = 42) -> List[int]:
    """
    Returns column indices of top-k features by mutual information on X (2D) vs y.
    Works with DataFrame or ndarray; returns integer indices suitable for np.take.
    """
    if mutual_info_classif is None or top_k >= X.shape[1]:
        return list(range(X.shape[1]))
    Xv = X.values if isinstance(X, pd.DataFrame) else X
    yv = y.values if isinstance(y, pd.Series) else y
    mi = mutual_info_classif(Xv, yv, random_state=random_state)
    order = np.argsort(mi)[::-1][:top_k]
    return order.tolist()


def drop_near_constant_features(
    X: pd.DataFrame,
    min_unique_frac: float = 0.005,
    min_std: float = 1e-6,
) -> list[str]:
    """
    Keep columns with enough variation (unique fraction & std).
    - unique fraction: guards discrete/indicator-like features
    - std: guards continuous-scale dead features
    References: simple filter advocated in feature-selection primers [R1].
    """
    n = float(len(X)) if len(X) else 1.0
    uniq_ok = (X.nunique(dropna=True) / max(n, 1.0)) >= float(min_unique_frac)
    std_ok  = X.std(skipna=True) >= float(min_std)
    keep = list(X.columns[uniq_ok & std_ok])
    return keep


def drop_high_corr_features(
    X: pd.DataFrame,
    threshold: float = 0.96,
    prefer_prefixes: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Greedy correlation-threshold pruning: among |rho|>threshold pairs, drop one.
    - We *prefer* keeping columns starting with any of prefer_prefixes when ties occur
      (e.g., realized-volatility or price-normalized spreads you trust more).
    - Uses pairwise-complete Pearson corr (pandas handles NaNs pairwise).
    Reference: correlation filtering to mitigate collinearity effects [R2].
    """
    if X.shape[1] <= 1:
        return list(X.columns)

    # Order columns so preferred prefixes are seen earlier (kept more often).
    cols = list(X.columns)
    if prefer_prefixes:
        def pref_score(c):
            for rank, p in enumerate(prefer_prefixes):
                if c.startswith(str(p)):  # keep earlier
                    return -(len(prefer_prefixes) - rank)
            return 0
        cols = sorted(cols, key=pref_score)

    # Absolute upper-triangular correlation matrix
    corr = X[cols].corr().abs()
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

    to_drop = set()
    for i, ci in enumerate(cols):
        if ci in to_drop:
            continue
        # find highly correlated partners for ci in upper triangle
        high = [cols[j] for j in range(i + 1, len(cols)) if mask_upper[i, j] and corr.iloc[i, j] > threshold]
        for cj in high:
            if cj not in to_drop:
                to_drop.add(cj)

    keep = [c for c in cols if c not in to_drop]
    return keep


def prefilter_features_train(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> list[str]:
    """
    Orchestrates the *pre-CV* filtering stages:
      1) near-constant drop
      2) high-corr pruning
      3) MI top-K gate (optional)
      4) per-family budget cap (optional, enabled via config)

    Notes:
    - y must be aligned with X (same index); we silently align to the intersection.
    - MI step uses your existing select_topk_by_mutual_info() if present.
    - Per-family budget uses FEATURE_FAMILIES + MAX_FEATURES_PER_FAMILY from metrics_tuples.
    """
    try:
        from utilsNoWFO import select_topk_by_mutual_info
    except Exception:
        select_topk_by_mutual_info = None

    # Defaults (safe)
    min_uniq = float(cfg.get("prefilter_min_unique_frac", 0.005))
    min_std  = float(cfg.get("prefilter_min_std", 1e-6))
    rho_thr  = float(cfg.get("prefilter_max_corr", 0.96))
    prefer   = cfg.get("prefilter_prefer_prefixes", ["rv", "ema", "sma", "macd", "adx"])
    mi_topk  = cfg.get("mutual_info_top_k", "sqrt")
    rng      = int(cfg.get("prefilter_random_state", 42))
    use_family_budget = bool(cfg.get("use_feature_family_budget", True))

    # 0) ensure alignment (in case caller forgot)
    if not X.empty and not y.empty:
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

    if X.shape[1] == 0:
        return []

    # 1) drop near-constants (works on original X; no extra copy needed)
    keep = drop_near_constant_features(
        X, min_unique_frac=min_uniq, min_std=min_std
    )
    if not keep:
        return []

    # 2) prune highly correlated on the reduced view only
    X_corr = X[keep]
    keep = drop_high_corr_features(
        X_corr, threshold=rho_thr, prefer_prefixes=prefer
    )
    if not keep:
        return []

    # 3) MI top-K (optional)
    if select_topk_by_mutual_info is None or len(keep) <= 1:
        return list(keep)

    # Minimal fill (means) for MI computation only; does NOT leak into model pipeline
    Xmi = X[keep].fillna(X[keep].mean())
    ymi = y.loc[Xmi.index]

    # derive K
    if isinstance(mi_topk, str) and mi_topk.lower() == "sqrt":
        K = max(1, int(np.sqrt(max(1, Xmi.shape[1]))))
    else:
        K = int(mi_topk)
        K = min(max(1, K), Xmi.shape[1])

    if K >= Xmi.shape[1]:
        keep_final = list(Xmi.columns)
    else:
        idxs = select_topk_by_mutual_info(
            Xmi, ymi.astype(int), top_k=K, random_state=rng
        )
        keep_final = [Xmi.columns[i] for i in idxs]

    # 4) Per-family budget cap
    if use_family_budget and len(keep_final) > 1:
        try:
            from pipeline.metrics.metrics_tuples import FEATURE_FAMILIES, MAX_FEATURES_PER_FAMILY
            keep_final = _apply_per_family_budget(keep_final, Xmi, ymi)
        except Exception:
            pass

    return keep_final


def _classify_feature(col: str, families: dict) -> str:
    """Return the family name for a feature column, or 'other'."""
    col_lower = col.lower()
    for family, prefixes in sorted(families.items(), key=lambda x: -max(len(p) for p in x[1])):
        for prefix in prefixes:
            if col_lower.startswith(prefix):
                return family
    return "other"


def _apply_per_family_budget(features: list[str], X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Reduce features so each family stays within MAX_FEATURES_PER_FAMILY."""
    from pipeline.metrics.metrics_tuples import FEATURE_FAMILIES, MAX_FEATURES_PER_FAMILY

    groups: dict[str, list[str]] = {}
    for col in features:
        family = _classify_feature(col, FEATURE_FAMILIES)
        groups.setdefault(family, []).append(col)

    kept: list[str] = []
    for family, cols in sorted(groups.items()):
        cap = MAX_FEATURES_PER_FAMILY.get(family, 10)
        if len(cols) <= cap:
            kept.extend(cols)
            continue

        # Need to select within family: compute MI for these columns
        sub_X = X[cols].fillna(X[cols].mean())
        sub_y = y.loc[sub_X.index]
        try:
            if mutual_info_classif is None:
                kept.extend(cols[:cap])
            else:
                mi = mutual_info_classif(sub_X, sub_y.astype(int), random_state=42)
                ranked = [cols[i] for i in np.argsort(mi)[::-1]]
                kept.extend(ranked[:cap])
        except Exception:
            kept.extend(cols[:cap])

    return kept


def realized_vol(ser, window=96):
    ser = ser.astype(float).fillna(0.0)
    return ser.rolling(int(window), min_periods=max(2, int(window//4))).std(ddof=0)


def compute_mda(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_jobs: int = 1,
) -> dict:
    """Compute Mean Decrease Accuracy (MDA) per feature.

    Per de Prado AFML Ch.8: for each feature, shuffle its values across
    observations (destroying signal) and measure the drop in validation
    accuracy. Features with negative MDA (accuracy improves when destroyed)
    are pure noise and should be pruned.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must have a .score(X, y) method.
    X_train, y_train : np.ndarray
        Training data (used only for refit if needed; typically not refit).
    X_val, y_val : np.ndarray
        Validation data on which MDA is measured.
        Must be purged (no label overlap with training).
    feature_names : list[str]
        Names for each column in X.
    n_jobs : int
        Parallel workers (default 1 = sequential).

    Returns
    -------
    dict {feature_name: mda_score}
        Negative scores = noise (accuracy improved when shuffled).
        Positive scores = informative features.
        Scores are clipped to [-1.0, 1.0].
    """
    import numpy as np

    if len(feature_names) != X_val.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != X_val columns ({X_val.shape[1]})"
        )

    # Baseline accuracy on unshuffled validation data
    baseline_score = float(model.score(X_val, y_val))

    mda_scores = {}
    X_val_shuffled = X_val.copy()

    for i, name in enumerate(feature_names):
        # Shuffle column i across all validation observations
        col_copy = X_val_shuffled[:, i].copy()
        np.random.shuffle(col_copy)
        X_val_shuffled[:, i] = col_copy

        # Compute accuracy with this feature destroyed
        shuffled_score = float(model.score(X_val_shuffled, y_val))

        # MDA = baseline - shuffled (positive = feature was useful)
        mda = baseline_score - shuffled_score
        mda_scores[name] = float(np.clip(mda, -1.0, 1.0))

        # Restore original column for next iteration
        X_val_shuffled[:, i] = X_val[:, i]

    return mda_scores


def prune_noise_features(
    mda_scores: dict,
    threshold: float = 0.0,
) -> list[str]:
    """Return list of feature names with positive MDA (above threshold).

    Parameters
    ----------
    mda_scores : dict
        Output from compute_mda().
    threshold : float
        Minimum MDA score to keep (default 0 = drop negative MDA only).

    Returns
    -------
    list[str] of feature names to keep.
    """
    return sorted(
        [name for name, score in mda_scores.items() if score > threshold],
        key=lambda n: mda_scores[n],
        reverse=True,
    )

