"""
Regime Taxonomy — 7-class market state classifier.

Expands the legacy 3-class system (SIDEWAYS/TREND/VOLATILE) to 7 granular regimes
for model→regime performance profiling and autonomous committee routing.

Classes:
  0 = QUIET_SQUEEZE    — BB compression, low ATR, low RV. Expect mean-reversion.
  1 = TREND_UP          — ADX > thr, price > EMA, positive momentum. Trend following.
  2 = TREND_DOWN        — ADX > thr, price < EMA, negative momentum. Trend following.
  3 = MEAN_REVERTING    — RSI extreme (>70/<30) or BB% extreme (<0.1/>0.9). Counter-trend.
  4 = BREAKOUT          — Donchian breach + expanding BB width. Momentum continuation.
  5 = HIGH_VOLATILE     — ATR spike, wide BB, low ADX. Choppy/unpredictable.
  6 = SIDEWAYS          — Everything else. Default/fallback.

Design principles:
  - Every function is pure (in: DataFrame, out: np.ndarray or DataFrame).
  - Missing columns degrade gracefully (regime defaults to SIDEWAYS=6).
  - Thresholds are configurable; defaults work on EURUSD H1/H4.
  - Zero external dependencies beyond numpy + pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

_LEGACY_MAP: Dict[int, int] = {
    0: 0,  1: 1,  2: 1,  3: 0,  4: 1,  5: 2,  6: 0,
}
"""Map 7-class regime → legacy 3-class (0=SIDEWAYS, 1=TREND, 2=VOLATILE)."""

_REGIME_NAMES: Dict[int, str] = {
    0: "quiet_squeeze",
    1: "trend_up",
    2: "trend_down",
    3: "mean_reverting",
    4: "breakout",
    5: "high_volatile",
    6: "sideways",
}

LEGACY_REVERSE: Dict[int, str] = {0: "sideways", 1: "trend", 2: "volatile"}

_COLUMN_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "adx": ("adx_14", "adx_28", "adx"),
    "ema": ("ema_20", "ema_50", "ema"),
    "rsi": ("rsi_14", "rsi_7", "rsi"),
    "bbw": ("bbw", "bb_width"),
    "bb_pct": ("bb_pct", "bb_percent"),
    "atr": ("atr_14", "atr_7", "atr"),
    "rv": ("rv_48", "rv_240", "rv"),
    "macd": ("macd_diff", "macd_line", "macd"),
    "sma": ("sma_20", "sma_50", "sma"),
    "donch_up": ("donchian_break_up_20", "donchian_up_20"),
    "donch_dn": ("donchian_break_dn_20", "donchian_dn_20"),
    "stoch_k": ("stoch_k", "stoch_14_k"),
    "stoch_d": ("stoch_d", "stoch_14_d"),
}


def _resolve_col(df: pd.DataFrame, key: str) -> Optional[str]:
    """Find the first existing column matching a fallback list."""
    for candidate in _COLUMN_FALLBACKS.get(key, (key,)):
        if candidate in df.columns:
            return candidate
    return None


def _safe_col(df: pd.DataFrame, key: str, default: float = np.nan) -> np.ndarray:
    """Return numpy array for column, or constant default if missing."""
    col = _resolve_col(df, key)
    if col is None:
        return np.full(len(df), default, dtype=np.float64)
    return df[col].astype(np.float64).to_numpy(copy=False)


class RegimeConfig:
    """Thresholds for regime classification. All are configurable."""

    def __init__(self, **overrides):
        self.adx_thresh: float = float(overrides.get("adx_thresh", 20.0))
        self.atr_high_quantile: float = float(overrides.get("atr_high_quantile", 0.80))
        self.bbw_high_quantile: float = float(overrides.get("bbw_high_quantile", 0.80))
        self.bbw_low_quantile: float = float(overrides.get("bbw_low_quantile", 0.20))
        self.rv_low_quantile: float = float(overrides.get("rv_low_quantile", 0.25))
        self.rsi_high: float = float(overrides.get("rsi_high", 70.0))
        self.rsi_low: float = float(overrides.get("rsi_low", 30.0))
        self.bbpct_high: float = float(overrides.get("bbpct_high", 0.90))
        self.bbpct_low: float = float(overrides.get("bbpct_low", 0.10))
        self.macd_strength_min: float = float(overrides.get("macd_strength_min", 0.0))
        # How many bars back to check for BB width expansion
        self.bbw_expand_lookback: int = int(overrides.get("bbw_expand_lookback", 4))


def detect_regimes(df: pd.DataFrame, config: Optional[RegimeConfig] = None) -> np.ndarray:
    """Classify every row into one of 7 regime classes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain (or have fallback names for) adx, ema, rsi, bb_pct,
        bbw, atr, rv, macd_diff, and optionally donchian break columns.
        Columns are resolved via _COLUMN_FALLBACKS.
    config : RegimeConfig, optional
        Threshold configuration. Uses defaults if None.

    Returns
    -------
    np.ndarray of int8, shape (n,)
        Regime IDs: 0=quiet_squeeze, 1=trend_up, 2=trend_down,
        3=mean_reverting, 4=breakout, 5=high_volatile, 6=sideways.
    """
    if config is None:
        config = RegimeConfig()

    n = len(df)

    adx = _safe_col(df, "adx")
    ema = _safe_col(df, "ema")
    rsi = _safe_col(df, "rsi")
    bb_pct = _safe_col(df, "bb_pct")
    bbw = _safe_col(df, "bbw")
    atr = _safe_col(df, "atr")
    rv = _safe_col(df, "rv")
    macd_diff = _safe_col(df, "macd")
    # Price column — prefer mid_c, fallback to close
    price = _safe_col(df, "mid_c")
    if np.isnan(price).all():
        price = _safe_col(df, "close")

    rc = config

    # ── Adaptive thresholds from data quantiles ──
    atr_vals = atr[np.isfinite(atr)]
    atr_hi = float(np.nanquantile(atr_vals, rc.atr_high_quantile)) if len(atr_vals) > 0 else 0.001

    bbw_vals = bbw[np.isfinite(bbw)]
    bbw_hi = float(np.nanquantile(bbw_vals, rc.bbw_high_quantile)) if len(bbw_vals) > 0 else 0.05
    bbw_lo = float(np.nanquantile(bbw_vals, rc.bbw_low_quantile)) if len(bbw_vals) > 0 else 0.005

    rv_vals = rv[np.isfinite(rv)]
    rv_lo = float(np.nanquantile(rv_vals, rc.rv_low_quantile)) if len(rv_vals) > 0 else 1e-5

    # ── Flags ──
    has_adx = not np.isnan(adx).all()
    has_ema = not np.isnan(ema).all()
    has_rsi = not np.isnan(rsi).all()
    has_bb_pct = not np.isnan(bb_pct).all()
    has_bbw = not np.isnan(bbw).all()
    has_atr = not np.isnan(atr).all()
    has_rv = not np.isnan(rv).all()
    has_macd = not np.isnan(macd_diff).all()
    has_price = not np.isnan(price).all()

    # ── Default everything to SIDEWAYS (6) ──
    regime = np.full(n, 6, dtype=np.int8)

    # ── START: QUIET_SQUEEZE (0) — lowest priority, easiest to detect ──
    if has_bbw and has_atr and has_rv:
        sq = (bbw <= bbw_lo) & (atr <= atr_hi) & (rv <= rv_lo) & np.isfinite(bbw) & np.isfinite(atr) & np.isfinite(rv)
        regime[sq] = 0

    # ── HIGH_VOLATILE (5) — overrides quiet squeeze ──
    if has_atr and has_bbw and has_adx:
        hv = (atr > atr_hi) & (bbw > bbw_hi) & (adx < rc.adx_thresh / 2.0)
        hv = hv & np.isfinite(atr) & np.isfinite(bbw) & np.isfinite(adx)
        regime[hv] = 5

    # ── MEAN_REVERTING (3) — overrides previous ──
    mr_mask = None
    if has_rsi:
        rsi_extreme = (rsi > rc.rsi_high) | (rsi < rc.rsi_low)
        rsi_extreme = rsi_extreme & np.isfinite(rsi)
        if mr_mask is None:
            mr_mask = rsi_extreme
        else:
            mr_mask = mr_mask | rsi_extreme

    if has_bb_pct:
        bb_extreme = (bb_pct > rc.bbpct_high) | (bb_pct < rc.bbpct_low)
        bb_extreme = bb_extreme & np.isfinite(bb_pct)
        if mr_mask is None:
            mr_mask = bb_extreme
        else:
            mr_mask = mr_mask | bb_extreme

    if mr_mask is not None:
        regime[mr_mask] = 3

    # ── BREAKOUT (4) — Donchian breach + expanding BB, overrides MR ──
    if has_bbw:
        # Check if bbw is expanding (higher than N bars ago)
        bbw_finite = np.isfinite(bbw)
        bbw_expanding = np.zeros(n, dtype=bool)
        lookback = rc.bbw_expand_lookback
        for i in range(lookback, n):
            if bbw_finite[i] and bbw_finite[i - lookback]:
                bbw_expanding[i] = bbw[i] > bbw[i - lookback]

        donchian_break = np.zeros(n, dtype=bool)
        donch_up = _safe_col(df, "donch_up")
        donch_dn = _safe_col(df, "donch_dn")
        has_donch = not np.isnan(donch_up).all() or not np.isnan(donch_dn).all()

        if has_donch:
            donchian_break = (donch_up > 0) | (donch_dn > 0)
        elif has_macd and has_adx:
            # Fallback: strong MACD cross + elevated ADX as proxy for breakout
            donchian_break = (np.abs(macd_diff) > 0.0001) & (adx >= rc.adx_thresh)

        breakout = bbw_expanding & donchian_break & np.isfinite(bbw)
        regime[breakout] = 4

    # ── TREND_UP (1) / TREND_DOWN (2) ──
    if has_adx and has_ema and has_price:
        trend_mask = (adx >= rc.adx_thresh) & np.isfinite(adx)
        above_ema = (price > ema) & np.isfinite(price) & np.isfinite(ema)
        below_ema = (price <= ema) & np.isfinite(price) & np.isfinite(ema)

        # Only assign trend if not already classified as more specific
        unclassified = (regime == 6)

        regime[trend_mask & above_ema & unclassified] = 1   # TREND_UP
        regime[trend_mask & below_ema & unclassified] = 2    # TREND_DOWN

    return regime


def attach_regime_columns(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    include_legacy: bool = True,
) -> pd.DataFrame:
    """Augment a DataFrame with full 7-class regime columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain required indicator columns (see detect_regimes).
    config : RegimeConfig, optional
        Thresholds.
    include_legacy : bool
        If True, also add legacy 3-class columns (regime_id, regime_trend, etc.).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added columns:
          - regime_7class (int8, 0-6)
          - regime_name (object, string label)
          - If include_legacy: regime_id, regime_trend, regime_sideways, regime_volatile
    """
    out = df.copy()

    regime_7 = detect_regimes(out, config)
    out["regime_7class"] = regime_7

    names = np.array([_REGIME_NAMES.get(r, "sideways") for r in regime_7], dtype=object)
    out["regime_name"] = names

    if include_legacy:
        legacy = np.array([_LEGACY_MAP.get(int(r), 0) for r in regime_7], dtype=np.int8)
        out["regime_id"] = legacy
        out["regime_trend"] = (legacy == 1).astype(np.int8)
        out["regime_sideways"] = (legacy == 0).astype(np.int8)
        out["regime_volatile"] = (legacy == 2).astype(np.int8)

    return out


def regime_counts(regime_arr: np.ndarray) -> Dict[str, int]:
    """Return count of bars per regime class."""
    counts: Dict[str, int] = {}
    for cls_id in range(7):
        name = _REGIME_NAMES.get(cls_id, f"unknown_{cls_id}")
        counts[name] = int((regime_arr == cls_id).sum())
    return counts


def regime_transition_matrix(regime_arr: np.ndarray) -> np.ndarray:
    """Compute a 7x7 transition count matrix from consecutive regime labels."""
    n = len(regime_arr)
    if n < 2:
        return np.zeros((7, 7), dtype=np.int64)

    mat = np.zeros((7, 7), dtype=np.int64)
    for i in range(n - 1):
        fr = int(regime_arr[i])
        to = int(regime_arr[i + 1])
        if 0 <= fr < 7 and 0 <= to < 7:
            mat[fr, to] += 1
    return mat


def to_legacy(regime_7: np.ndarray) -> np.ndarray:
    """Convert 7-class regime array to legacy 3-class."""
    out = np.zeros(len(regime_7), dtype=np.int8)
    for k7, k3 in _LEGACY_MAP.items():
        out[regime_7 == k7] = k3
    return out


_STABLE_REGIMES_FOR_MODEL = {
    "lstm":         [1, 2],
    "transformer":  [1, 2],
    "cnn":          [4, 5],
    "random_forest":[3, 4, 5],
    "xgboost":      [1, 2, 3],
    "lightgbm":     [1, 2, 3],
    "catboost":     [1, 2, 3],
    "logistic":     [6],
    "svm":          [6],
    "decision_tree":[3, 4],
    "gru":          [1, 2],
    "gru_lstm":     [1, 2],
    "meta_ensemble":[1, 2, 3, 6],
    "stacking_ensemble":[1, 3, 6],
    "ensemble_adaptive_regime":[1, 2, 3, 4, 5, 6],
    "ensemble_cnn_lstm_xgboost":[1, 2, 4, 5],
    "dqn":          [1, 2],
}


def recommended_models_for_regime(regime_id: int) -> list:
    """Return model types recommended for a given regime class.

    Based on architectural priors (what each model architecture is known
    to handle well). These are initial defaults — superseded by empirically
    learned mappings once the ExpertProfiler runs.
    """
    matches = []
    for model, regimes in _STABLE_REGIMES_FOR_MODEL.items():
        if regime_id in regimes:
            matches.append(model)
    return matches
