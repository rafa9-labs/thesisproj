"""Tuning helpers: objective helpers, boundary tracking, TA profiles."""
import datetime
import gc
import json
import math
import os
import time
import traceback
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy

import numpy as np
import optuna
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from utilsNoWFO import (
    TRAIN_TEST_MONTHS,
    TRAIN_TEST_MONTHS_DEBUG,
    _bad_objective_for_direction,
    _norm_optuna_direction,
    compute_required_test_warmup_bars,
    log_print,
    save_optuna_progress_from_study,
    target_coverage_policy,
)

TRAIN_TEST_DEBUG_MODE = False


def _bad_obj(direction: str | None, magnitude: float = 9999.0) -> float:
    d = _norm_optuna_direction(direction)
    return _bad_objective_for_direction(d, magnitude=magnitude)


# ---- Tuning-time environment knobs ----------------------------------------------------
#
# SAVE_TRIAL_FEATURE_FREQ=1
#   Compute per-trial feature usage and enable trial-level feature-frequency
#   heatmaps after tuning. This forces an extra indicator/feature build per
#   trial and is OFF by default for speed.
#
# CV_DEBUG=1
#   Enable verbose Mini-Block CV debugging inside MLBacktesterNoWFO
#   (per-fold penalty reasons, Sharpe summaries, etc.). Default is 0 during
#   tuning to reduce logging and overhead.
#
# CV_TABLE_MODE=off|compact|verbose|full
#   Override the CV table rendering mode used by MLBacktesterNoWFO. When set
#   to "off" tuning will skip per-fold ASCII tables and only log high-level
#   summaries. If unset, the value from cv_config / global defaults is used.
#
# All of these switches are TELEMETRY-ONLY: they do not change models, data
# splits, or score aggregation; they only affect logging and diagnostics.

SAVE_TRIAL_FEATURE_FREQ = os.environ.get("SAVE_TRIAL_FEATURE_FREQ", "0") == "1"

# --- Hyperparameter boundary diagnostics (for range tuning) ---
# This dictionary is reset at the start of each run_optuna_tuning() call and
# incremented whenever a sampled value sits very close to its lower or upper
# bound. It is only used for *logging* and does not affect optimisation.
HP_BOUNDARY_HITS: dict[str, int] = {}
HP_BOUNDARY_HITS_MIN: dict[str, int] = {}
HP_BOUNDARY_HITS_MAX: dict[str, int] = {}
HP_BOUNDARY_RANGES: dict[str, tuple[float, float]] = {}


def _record_hp_boundary_hit(name: str, value: float, low: float, high: float, eps_frac: float = 0.01) -> None:
    """
    Track how often Optuna samples a value near the edge of its search range.

    Args:
        name:  Hyperparameter name (used as dict key).
        value: Sampled value.
        low:   Lower bound used in trial.suggest_*.
        high:  Upper bound used in trial.suggest_*.
        eps_frac: Fraction of range treated as 'near the edge'
                  (0.01 -> within 1% of [low, high]).
    """
    try:
        low = float(low)
        high = float(high)
        value = float(value)
    except Exception:
        return

    import numpy as _np

    if not _np.isfinite(low) or not _np.isfinite(high) or high <= low:
        return

    span = high - low
    if span <= 0:
        return

    # Normalised distance to each edge
    rel_low = (value - low) / span
    rel_high = (high - value) / span

    # Store last seen range for later diagnostics
    try:
        HP_BOUNDARY_RANGES[name] = (low, high)
    except Exception:
        pass

    _hit_any = False
    if rel_low <= eps_frac:
        HP_BOUNDARY_HITS_MIN[name] = HP_BOUNDARY_HITS_MIN.get(name, 0) + 1
        _hit_any = True
    if rel_high <= eps_frac:
        HP_BOUNDARY_HITS_MAX[name] = HP_BOUNDARY_HITS_MAX.get(name, 0) + 1
        _hit_any = True
    if _hit_any:
        HP_BOUNDARY_HITS[name] = HP_BOUNDARY_HITS.get(name, 0) + 1


# --- Quiet TensorFlow logs (no determinism; keep random seeding) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys as _sys
if _sys.platform != "win32":
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _ld:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(_ld.split(":")))

DISABLE_OPTUNA_PRUNING = os.getenv("MLB_DISABLE_OPTUNA_PRUNING", "0") == "1"
SKIP_PLOTS = os.environ.get("MLB_SKIP_PLOTS", "0") == "1"

# TA profile selector:
#   MLB_TA_MODE = "legacy" | "fixed" | "tuned"
MLB_TA_MODE = os.environ.get("MLB_TA_MODE", "tuned").strip().lower()


def _ta_profile_sanity(params: dict, context: str = "ta") -> None:
    """
    Debug-only guardrail: if an indicator is enabled but its window(s) are missing
    from params["indicator_windows"], fill deterministic defaults and (in DEBUG)
    log what was repaired.

    This does NOT add Optuna dimensions and should have no effect when configs
    are already well-formed.
    """
    ind = dict(params.get("indicator_windows") or {})
    changed = []

    def _ensure(k: str, v):
        if k not in ind or ind[k] is None:
            ind[k] = v
            changed.append(k)

    if params.get("use_sma"):
        _ensure("sma", 20)
    if params.get("use_ema"):
        _ensure("ema", 20)
    if params.get("use_rsi"):
        _ensure("rsi", 14)
    if params.get("use_atr"):
        _ensure("atr", 14)
    if params.get("use_adx"):
        _ensure("adx", 14)
    if params.get("use_bbands"):
        _ensure("bb_window", 20)
        _ensure("bb_dev", 2.0)
    if params.get("use_macd"):
        _ensure("macd_fast", 12)
        _ensure("macd_slow", 26)
        _ensure("macd_signal", 9)
        # Safety: ensure slow > fast
        try:
            if int(ind["macd_slow"]) <= int(ind["macd_fast"]):
                ind["macd_slow"] = int(ind["macd_fast"]) + 1
                if "macd_slow" not in changed:
                    changed.append("macd_slow")
        except Exception:
            pass
    if params.get("use_stoch"):
        _ensure("stoch_k", 14)
        _ensure("stoch_d", 3)
    if params.get("use_mtf_ma"):
        _ensure("mtf_ma_fast_window", 10)
        _ensure("mtf_ma_slow_window", 50)

    if changed:
        params["indicator_windows"] = ind
        _ta_profile_sanity(params, context="ungated")
        # DEBUG-only print (respects LOG_MODE via log_print)
        log_print(f"[TA][{context}] filled missing indicator_windows keys: {changed}", level="DEBUG")



def _apply_ta_profile_legacy(trial, params):
    """
    Legacy TA profile with explicit strategy_type gating.

    This is kept for backwards-compatibility and ablation runs when
    MLB_TA_MODE="legacy". For the main tuned mode we will use the
    unified profile (_apply_ta_profile_ungated) instead.
    """
    # === Strategy family (gates everything below) ===
    params["strategy_type"] = trial.suggest_categorical(
        "strategy_type", ["momentum", "contrarian", "volatility", "confirmation", "all"]
    )

    # Base indicator families per strategy
    strategy_map = {
        "momentum":     ["rsi", "macd", "ema", "sma"],
        "contrarian":   ["rsi", "bbands", "stoch", "ema", "sma"],
        "volatility":   ["atr", "bbands", "adx", "ema"],      # bbw from bbands
        "confirmation": ["adx", "mtf_ma", "ema", "macd", "sma", "sar"],
        "all":          ["rsi", "macd", "atr", "bbands", "stoch", "ema", "adx", "sma", "mtf_ma", "sar"],
    }
    sel = strategy_map[params["strategy_type"]]

    # Classic base toggles (strategy-gated)
    for feat in ["rsi", "macd", "atr", "bbands", "stoch", "ema", "adx", "sma", "mtf_ma", "sar"]:
        if params["strategy_type"] == "all":
            params[f"use_{feat}"] = trial.suggest_categorical(f"use_{feat}", [True, False])
        else:
            params[f"use_{feat}"] = (feat in sel)

    # === Advanced families (momentum extensions & composites), strategy-gated ===
    # Initialize all as False; we selectively enable/sample per family profile.
    for k in [
        "use_ma_spread","use_price_ma_z","use_crossover_bins","use_slope_diff",
        "use_reentry_mom","use_ext_atr_low_adx","use_squeeze_expansion","use_atr_channel_breakout",
        "use_trend_confirm","use_mtf_alignment","use_vol_managed_mom","use_macd_atr_ratio",
        # keep aliases ON together for runtime/tuner parity:
        "use_squeeze_breakout","use_triple_confirm","use_mtf_align","use_vm_mom"
    ]:
        params[k] = False

    st = params["strategy_type"]

    if st in ("all",):
        # Let Optuna freely explore everything
        params["use_ma_spread"]       = trial.suggest_categorical("use_ma_spread", [False, True])
        params["use_price_ma_z"]      = trial.suggest_categorical("use_price_ma_z", [False, True])
        params["use_crossover_bins"]  = trial.suggest_categorical("use_crossover_bins", [False, True])
        params["use_slope_diff"]      = trial.suggest_categorical("use_slope_diff", [False, True])

        params["use_reentry_mom"]     = trial.suggest_categorical("use_reentry_mom", [False, True])
        params["use_ext_atr_low_adx"] = trial.suggest_categorical("use_ext_atr_low_adx", [False, True])

        # squeeze family (alias both toggles)
        _sq = trial.suggest_categorical("use_squeeze_family", [False, True])
        params["use_squeeze_expansion"] = _sq
        params["use_squeeze_breakout"]  = _sq

        params["use_atr_channel_breakout"] = trial.suggest_categorical("use_atr_channel_breakout", [False, True])

        # trend confirm family (alias both toggles)
        _tc = trial.suggest_categorical("use_trend_confirm_family", [False, True])
        params["use_trend_confirm"] = _tc
        params["use_triple_confirm"] = _tc
        # MTF alignment (alias)
        _mtfa = trial.suggest_categorical("use_mtf_alignment_family", [False, True])
        params["use_mtf_alignment"] = _mtfa
        params["use_mtf_align"]     = _mtfa

        # volatility-managed momentum (alias)
        _vmm = trial.suggest_categorical("use_vol_managed_mom_family", [False, True])
        params["use_vol_managed_mom"] = _vmm
        params["use_vm_mom"]          = _vmm

        params["use_macd_atr_ratio"] = trial.suggest_categorical("use_macd_atr_ratio", [False, True])

    if st in ("momentum",):
        params["use_ma_spread"]  = True
        params["use_slope_diff"] = True
        params["use_crossover_bins"] = trial.suggest_categorical("use_crossover_bins", [False, True])
        # Risk-normalized momentum:
        _vmm = trial.suggest_categorical("use_vol_managed_mom_family", [False, True])
        params["use_vol_managed_mom"] = _vmm; params["use_vm_mom"] = _vmm
        params["use_macd_atr_ratio"] = trial.suggest_categorical("use_macd_atr_ratio", [False, True])

    if st in ("contrarian",):
        params["use_price_ma_z"]      = True
        params["use_reentry_mom"]     = trial.suggest_categorical("use_reentry_mom", [False, True])
        params["use_ext_atr_low_adx"] = trial.suggest_categorical("use_ext_atr_low_adx", [False, True])
        params["use_crossover_bins"]  = trial.suggest_categorical("use_crossover_bins", [False, True])

    if st in ("volatility",):
        _sq = trial.suggest_categorical("use_squeeze_family", [False, True])
        params["use_squeeze_expansion"] = _sq; params["use_squeeze_breakout"] = _sq
        params["use_atr_channel_breakout"] = trial.suggest_categorical("use_atr_channel_breakout", [False, True])

    if st in ("confirmation",):
        _tc = trial.suggest_categorical("use_trend_confirm_family", [False, True])
        params["use_trend_confirm"] = _tc; params["use_triple_confirm"] = _tc
        _mtfa = trial.suggest_categorical("use_mtf_alignment_family", [False, True])
        params["use_mtf_alignment"] = _mtfa; params["use_mtf_align"] = _mtfa

    # === Hyperparameters for families (only when active) ===
    if params.get("use_squeeze_expansion") or params.get("use_squeeze_breakout"):
        params["squeeze_window"]   = trial.suggest_int("squeeze_window", 150, 600)
        params["squeeze_quantile"] = trial.suggest_float("squeeze_quantile", 0.05, 0.20)
        params["adx_slope_window"] = trial.suggest_int("adx_slope_window", 5, 20)
    if params.get("use_atr_channel_breakout"):
        params["atr_channel_mult"] = trial.suggest_float("atr_channel_mult", 1.0, 3.0)
    if params.get("use_trend_confirm") or params.get("use_triple_confirm"):
        params["triple_confirm_lookback"] = trial.suggest_int("triple_confirm_lookback", 10, 40)
        params["adx_rise_thresh"]         = trial.suggest_float("adx_rise_thresh", 0.0, 5.0)
    if params.get("use_vol_managed_mom") or params.get("use_vm_mom"):
        params["vm_mom_scale"] = trial.suggest_float("vm_mom_scale", 0.5, 2.0)

    # === Indicator windows (only for what's needed) ===
    ind = {}

    # Base families first (as toggled above)
    if params.get("use_sma"):   ind["sma"]   = trial.suggest_int("sma_window", 10, 40)
    if params.get("use_ema"):   ind["ema"]   = trial.suggest_int("ema_window", 10, 40)
    if params.get("use_rsi"):   ind["rsi"]   = trial.suggest_int("rsi_window", 10, 20)
    if params.get("use_macd"):
        mf = trial.suggest_int("macd_fast", 8, 16)
        ms = trial.suggest_int("macd_slow", 22, 30)
        if ms <= mf: ms = mf + 1
        ind["macd_fast"], ind["macd_slow"] = mf, ms
        ind["macd_signal"] = trial.suggest_int("macd_signal", 6, 12)
    if params.get("use_bbands"):
        ind["bb_window"] = trial.suggest_int("bb_window", 16, 24)
        ind["bb_dev"]    = trial.suggest_float("bb_dev", 1.5, 2.5)
    if params.get("use_atr"):   ind["atr"]      = trial.suggest_int("atr_window", 10, 20)
    if params.get("use_stoch"):
        ind["stoch_k"] = trial.suggest_int("stoch_k_window", 10, 20)
        ind["stoch_d"] = trial.suggest_int("stoch_d_window", 3, 7)
    if params.get("use_mtf_ma"):
        ind["mtf_ma_fast_window"] = trial.suggest_int("mtf_ma_fast_window", 8, 14)     # ~H1 MA10
        ind["mtf_ma_slow_window"] = trial.suggest_int("mtf_ma_slow_window", 40, 60)    # ~H4 MA50

    # Prerequisites implied by advanced families (unchanged)
    need_bb   = params.get("use_squeeze_expansion") or params.get("use_squeeze_breakout") or params.get("use_reentry_mom") or params.get("use_price_ma_z")
    need_atr  = params.get("use_atr_channel_breakout") or params.get("use_macd_atr_ratio") or params.get("use_ext_atr_low_adx")
    need_adx  = params.get("use_trend_confirm") or params.get("use_triple_confirm") or params.get("use_squeeze_breakout") or params.get("use_ext_atr_low_adx")
    need_macd = params.get("use_trend_confirm") or params.get("use_triple_confirm") or params.get("use_macd_atr_ratio") or params.get("use_slope_diff")
    need_ma   = params.get("use_ma_spread") or params.get("use_reentry_mom") or params.get("use_crossover_bins") or params.get("use_slope_diff")
    need_mtf  = params.get("use_mtf_alignment") or params.get("use_mtf_align")

    # Ensure base toggles ON if a composite requires them (so values exist upstream)
    if need_bb and not params.get("use_bbands"): params["use_bbands"] = True
    if need_atr and not params.get("use_atr"):   params["use_atr"] = True
    if need_adx and not params.get("use_adx"):   params["use_adx"] = True
    if need_macd and not params.get("use_macd"): params["use_macd"] = True
    if need_ma:
        if not params.get("use_ema"): params["use_ema"] = True
        if not params.get("use_sma"): params["use_sma"] = True
    if need_mtf and not params.get("use_mtf_ma"): params["use_mtf_ma"] = True


def _apply_ta_profile_ungated(
    trial,
    params,
    *,
    skip_base_families=(),
    sample_base_windows: bool = True,
):
    """
    Unified TA profile without strategy_type gating.

    Behaviour:
      - Does NOT sample or use a 'strategy_type' dimension.
      - Treats all TA families as potentially available and lets Optuna
        toggle them on/off individually.
      - For advanced/composite families, mirrors the behaviour of the
        'strategy_type="all"' branch in _apply_ta_profile_legacy.
      - Indicator windows and prerequisite logic are identical to legacy.
    """
    # Base indicator families: sample on/off independently (unless skipped)
    skip = set(skip_base_families or ())
    for feat in ["rsi", "macd", "atr", "bbands", "stoch", "ema", "adx", "sma", "mtf_ma", "sar"]:
        if feat in skip:
            # Deterministic placeholder so downstream expects the key to exist.
            # (Skipped families are typically forced on/off by the caller.)
            params[f"use_{feat}"] = bool(params.get(f"use_{feat}", False))
        else:
            params[f"use_{feat}"] = trial.suggest_categorical(f"use_{feat}", [True, False])

    # Advanced families: initialize as False; we selectively enable/sample below.
    for k in [
        "use_ma_spread","use_price_ma_z","use_crossover_bins","use_slope_diff",
        "use_reentry_mom","use_ext_atr_low_adx","use_squeeze_expansion","use_atr_channel_breakout",
        "use_trend_confirm","use_mtf_alignment","use_vol_managed_mom","use_macd_atr_ratio",
        # keep aliases ON together for runtime/tuner parity:
        "use_squeeze_breakout","use_triple_confirm","use_mtf_align","use_vm_mom"
    ]:
        params[k] = False

    # Advanced families: same as 'all' branch in legacy profile
    params["use_ma_spread"]       = trial.suggest_categorical("use_ma_spread", [False, True])
    params["use_price_ma_z"]      = trial.suggest_categorical("use_price_ma_z", [False, True])
    params["use_crossover_bins"]  = trial.suggest_categorical("use_crossover_bins", [False, True])
    params["use_slope_diff"]      = trial.suggest_categorical("use_slope_diff", [False, True])

    params["use_reentry_mom"]     = trial.suggest_categorical("use_reentry_mom", [False, True])
    params["use_ext_atr_low_adx"] = trial.suggest_categorical("use_ext_atr_low_adx", [False, True])

    # squeeze family (alias both toggles)
    _sq = trial.suggest_categorical("use_squeeze_family", [False, True])
    params["use_squeeze_expansion"] = _sq
    params["use_squeeze_breakout"]  = _sq

    params["use_atr_channel_breakout"] = trial.suggest_categorical("use_atr_channel_breakout", [False, True])

    # trend confirm family (alias both toggles)
    _tc = trial.suggest_categorical("use_trend_confirm_family", [False, True])
    params["use_trend_confirm"] = _tc
    params["use_triple_confirm"] = _tc

    # MTF alignment (alias)
    _mtfa = trial.suggest_categorical("use_mtf_alignment_family", [False, True])
    params["use_mtf_alignment"] = _mtfa
    params["use_mtf_align"]     = _mtfa
    
    # volatility-managed momentum (alias)
    _vmm = trial.suggest_categorical("use_vol_managed_mom_family", [False, True])
    params["use_vol_managed_mom"] = _vmm
    params["use_vm_mom"]          = _vmm

    params["use_macd_atr_ratio"] = trial.suggest_categorical("use_macd_atr_ratio", [False, True])

    # Hyperparameters and indicator windows identical to legacy 'all' branch
    if params.get("use_squeeze_expansion") or params.get("use_squeeze_breakout"):
        params["squeeze_window"]   = trial.suggest_int("squeeze_window", 150, 600)
        params["squeeze_quantile"] = trial.suggest_float("squeeze_quantile", 0.05, 0.20)
        params["adx_slope_window"] = trial.suggest_int("adx_slope_window", 5, 20)
    if params.get("use_atr_channel_breakout"):
        params["atr_channel_mult"] = trial.suggest_float("atr_channel_mult", 1.0, 3.0)
    if params.get("use_trend_confirm") or params.get("use_triple_confirm"):
        params["triple_confirm_lookback"] = trial.suggest_int("triple_confirm_lookback", 10, 40)
        params["adx_rise_thresh"]         = trial.suggest_float("adx_rise_thresh", 0.0, 5.0)
    if params.get("use_vol_managed_mom") or params.get("use_vm_mom"):
        params["vm_mom_scale"] = trial.suggest_float("vm_mom_scale", 0.5, 2.0)

    # Start from any existing window dict (allows composition with other profiles)
    ind = dict(params.get("indicator_windows") or {})

    # Sample base windows only when requested. This lets callers force-enable
    # a backbone and override its windows separately, without double-sampling.
    if sample_base_windows:
        if params.get("use_sma") and ("sma" not in skip):
            ind["sma"] = trial.suggest_int("sma_window", 10, 40)
        if params.get("use_ema") and ("ema" not in skip):
            ind["ema"] = trial.suggest_int("ema_window", 10, 40)
        if params.get("use_rsi") and ("rsi" not in skip):
            ind["rsi"] = trial.suggest_int("rsi_window", 10, 20)
        if params.get("use_macd") and ("macd" not in skip):
            mf = trial.suggest_int("macd_fast", 8, 16)
            ms = trial.suggest_int("macd_slow", 22, 30)
            if ms <= mf:
                ms = mf + 1
            ind["macd_fast"], ind["macd_slow"] = mf, ms
            ind["macd_signal"] = trial.suggest_int("macd_signal", 6, 12)
        if params.get("use_bbands") and ("bbands" not in skip):
            ind["bb_window"] = trial.suggest_int("bb_window", 16, 24)
            ind["bb_dev"] = trial.suggest_float("bb_dev", 1.5, 2.5)
        if params.get("use_atr") and ("atr" not in skip):
            ind["atr"] = trial.suggest_int("atr_window", 10, 20)
        if params.get("use_stoch") and ("stoch" not in skip):
            ind["stoch_k"] = trial.suggest_int("stoch_k_window", 10, 20)
            ind["stoch_d"] = trial.suggest_int("stoch_d_window", 3, 7)
        if params.get("use_mtf_ma") and ("mtf_ma" not in skip):
            ind["mtf_ma_fast_window"] = trial.suggest_int("mtf_ma_fast_window", 8, 14)
            ind["mtf_ma_slow_window"] = trial.suggest_int("mtf_ma_slow_window", 40, 60)

    need_bb   = params.get("use_squeeze_expansion") or params.get("use_squeeze_breakout") or params.get("use_reentry_mom") or params.get("use_price_ma_z")
    need_atr  = params.get("use_atr_channel_breakout") or params.get("use_macd_atr_ratio") or params.get("use_ext_atr_low_adx")
    need_adx  = params.get("use_trend_confirm") or params.get("use_triple_confirm") or params.get("use_squeeze_breakout") or params.get("use_ext_atr_low_adx")
    need_macd = params.get("use_trend_confirm") or params.get("use_triple_confirm") or params.get("use_macd_atr_ratio") or params.get("use_slope_diff")
    need_ma   = params.get("use_ma_spread") or params.get("use_reentry_mom") or params.get("use_crossover_bins") or params.get("use_slope_diff")
    need_mtf  = params.get("use_mtf_alignment") or params.get("use_mtf_align")

    if need_bb and not params.get("use_bbands"): params["use_bbands"] = True
    if need_atr and not params.get("use_atr"):   params["use_atr"] = True
    if need_adx and not params.get("use_adx"):   params["use_adx"] = True
    if need_macd and not params.get("use_macd"): params["use_macd"] = True
    if need_ma:
        if not params.get("use_ema"): params["use_ema"] = True
        if not params.get("use_sma"): params["use_sma"] = True
    if need_mtf and not params.get("use_mtf_ma"): params["use_mtf_ma"] = True
    
    # ---- Window completion for any indicators that ended up enabled late ----
    # Prerequisite forcing can switch a base indicator ON *after* its windows
    # were sampled. To keep configs consistent (and avoid runtime defaults
    # varying silently across trials), ensure a deterministic window exists.
    def _set_default(k: str, v):
        if k not in ind or ind[k] is None:
            ind[k] = v

    if params.get("use_sma"):
        _set_default("sma", 20)
    if params.get("use_ema"):
        _set_default("ema", 20)
    if params.get("use_rsi"):
        _set_default("rsi", 14)
    if params.get("use_atr"):
        _set_default("atr", 14)
    if params.get("use_adx"):
        _set_default("adx", 14)
    if params.get("use_bbands"):
        _set_default("bb_window", 20)
        _set_default("bb_dev", 2.0)
    if params.get("use_macd"):
        _set_default("macd_fast", 12)
        _set_default("macd_slow", 26)
        _set_default("macd_signal", 9)
        # Safety: ensure slow > fast.
        try:
            if int(ind["macd_slow"]) <= int(ind["macd_fast"]):
                ind["macd_slow"] = int(ind["macd_fast"]) + 1
        except Exception:
            pass
    if params.get("use_stoch"):
        _set_default("stoch_k", 14)
        _set_default("stoch_d", 3)    
        
    if params.get("use_mtf_ma"):
        _set_default("mtf_ma_fast_window", 10)
        _set_default("mtf_ma_slow_window", 50)

    # Write back the final window dict
    params["indicator_windows"] = ind
    _ta_profile_sanity(params, context="tuned")


def _apply_ta_profile_fixed(trial, params):
    """Fixed TA / research-clean profile.

    Behaviour:
      - Does NOT sample any TA-related hyperparameters from Optuna.
      - Forces a canonical, static TA backbone:
          * use_* families: SMA, EMA, MACD, RSI, BBands, ATR, ADX, Stoch, MTF_MA -> all True
          * indicator_windows set to fixed defaults.
      - Leaves all non-TA parameters in `params` untouched.
    """
    # Base indicator families: always enabled
    for feat in ("sma", "ema", "macd", "rsi", "bbands", "atr", "adx", "stoch", "mtf_ma"):
        params[f"use_{feat}"] = True

    # Composite / interaction features: keep them ON (still fixed, not tuned).
    # These are deterministic functions of the fixed TA backbone and are used by
    # MLBacktesterNoWFO.prepare_features().
    for k in [
        # momentum extensions
        "use_ma_spread",
        "use_price_ma_z",
        "use_crossover_bins",
        "use_slope_diff",
        # composite features
        "use_reentry_mom",
        "use_ext_atr_low_adx",
        "use_squeeze_expansion",
        "use_atr_channel_breakout",
        "use_trend_confirm",
        "use_mtf_alignment",
        "use_vol_managed_mom",
        "use_macd_atr_ratio",
    ]:
        params[k] = True

    # Keep legacy/unused aliases OFF (not used in current feature builder path)
    for k in [
        "use_squeeze_breakout",
        "use_triple_confirm",
        "use_mtf_align",
        "use_vm_mom",
    ]:
        params[k] = False

    # Strategy label is still useful for logging; treat as 'all'
    params["strategy_type"] = "all"

    # Fixed canonical TA windows
    params["indicator_windows"] = {
        "sma": 20,
        "ema": 20,
        "rsi": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_window": 20,
        "bb_dev": 2.0,
        "atr": 14,
        "adx": 14,
        "stoch_k": 14,
        "stoch_d": 3,
        "mtf_ma_fast_window": 10,
        "mtf_ma_slow_window": 50,
    }

    # ATR-normalized spread only makes sense if ATR is present; here we keep it deterministic.
    params["use_spread_over_atr"] = False


def _apply_ta_profile_tuned(trial, params):
    """
    Tuned TA / core backbone profile.

    Behaviour:
      - Starts from a unified TA space (full TA families and composites)
        without any strategy_type gating.
      - Forces a stable core TA backbone:
          core_families = ["sma", "ema", "rsi", "macd", "atr", "bbands", "adx"]
        * For each core family: force use_* = True.
        * For core windows:
            - Override / set them via Optuna categorical choices over
              three literature-motivated values.
      - Leaves non-core / composite TA toggles as sampled by the unified        
      TA profile helper.
    """
    # Core TA families must always be enabled in tuned mode.
    # Define this early so we can also prevent Optuna from sampling redundant
    # (and later-overwritten) core toggles/windows in the ungated pass.
    core_families = ("sma", "ema", "rsi", "macd", "atr", "bbands", "adx")
    
    # 1) Run unified TA sampling to preserve current behaviour for
    #    non-core indicators and composite families, but without any
    #    strategy_type dimension.
    #    NOTE: skip core families to avoid dead trial dimensions; the backbone
    #    is forced ON and windows are overridden below.
    _apply_ta_profile_ungated(trial, params, skip_base_families=core_families)


    # 2) Core TA families must always be enabled in tuned mode
    for feat in core_families:
        params[f"use_{feat}"] = True

    # Work on a local copy of indicator_windows
    ind = dict(params.get("indicator_windows") or {})

    # --- SMA / EMA: short, medium, longer intraday trend horizons ---
    # (e.g. 10 ~= fast, 20 ~= standard, 40 ~= slower trend)
    ind["sma"] = trial.suggest_categorical(
        "sma_window_core",
        [10, 20, 40],
    )
    ind["ema"] = trial.suggest_categorical(
        "ema_window_core",
        [10, 20, 40],
    )

    # --- RSI: faster, canonical, slower oscillations ---
    ind["rsi"] = trial.suggest_categorical(
        "rsi_window_core",
        [10, 14, 21],
    )

    # --- ATR / ADX: short vs medium range and trend strength ---
    ind["atr"] = trial.suggest_categorical(
        "atr_window_core",
        [14, 20, 28],
    )
    ind["adx"] = trial.suggest_categorical(
        "adx_window_core",
        [14, 20, 28],
    )

    # --- Bollinger Bands: window and deviation ---
    ind["bb_window"] = trial.suggest_categorical(
        "bb_window_core",
        [16, 20, 24],
    )
    ind["bb_dev"] = trial.suggest_categorical(
        "bb_dev_core",
        [1.5, 2.0, 2.5],
    )

    # --- MACD: choose among 3 canonical (fast, slow, signal) triplets ---
    macd_fast, macd_slow, macd_signal = trial.suggest_categorical(
        "macd_core_variant",
        [
            (8, 17, 9),   # faster MACD
            (12, 26, 9),  # standard MACD
            (19, 39, 9),  # slower MACD
        ],
    )
    ind["macd_fast"] = macd_fast
    ind["macd_slow"] = macd_slow
    ind["macd_signal"] = macd_signal

    # 3) Write back
    params["indicator_windows"] = ind
    _ta_profile_sanity(params, context="tuned")
    
     # --- Hygiene: remove legacy window keys that are not authoritative in tuned mode ---
    # In tuned mode, indicator_windows is the single source of truth.
    # These keys may linger from other profiles / older runs and can confuse
    # saved JSON configs or Top-N consensus comparisons.
    for k in (
        # base windows
        "sma_window", "ema_window", "rsi_window", "atr_window", "adx_window",
        "bb_window", "bb_dev",
        "stoch_k_window", "stoch_d_window",
        "mtf_ma_fast_window", "mtf_ma_slow_window",
        # MACD legacy flat keys
        "macd_fast", "macd_slow", "macd_signal",
    ):
        if k in params:
            params.pop(k, None)



