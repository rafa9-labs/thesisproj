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
# optuna.logging.set_verbosity(optuna.logging.WARNING)
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
                  (0.01 → within 1% of [low, high]).
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

    # === Indicator windows (only for what’s needed) ===
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
          * use_* families: SMA, EMA, MACD, RSI, BBands, ATR, ADX, Stoch, MTF_MA → all True
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
    # (e.g. 10 ≈ fast, 20 ≈ standard, 40 ≈ slower trend)
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



def sample_param_set(trial, models_to_test, train_data=None, vol_stats=None, stage_config=None):
    """
    Drop-in replacement tuned for your model zoo, now strategy-gated.

    All composites only activate when their family is selected, and their
    prerequisite base indicators are also toggled + window-sampled.
    """
    import math
    import numpy as np
    try:
        import optuna  # for TrialPruned
    except Exception:
        optuna = None

    # Never tune DQN inside Optuna
    models_to_test = [m for m in models_to_test if m != "dqn"]
    if not models_to_test:
        raise ValueError("models_to_test is empty for Optuna. DQN must be tuned separately.")

    model_type = trial.suggest_categorical("model_type", models_to_test)
    params = {"model_type": model_type}
    

    # ------------------------------------------------------------
    # Two-stage HPO support (minimal, no redesign):
    #   - A_signal: tune predictive/signal params under FIXED exec/calibration
    #   - BC_exec_calib: freeze best A params, tune ONLY calibration/execution
    # ------------------------------------------------------------
    _stage_cfg = dict(stage_config or {})
    _hpo_stage = str(_stage_cfg.get("hpo_stage", "single") or "single").strip()
    _frozen = _stage_cfg.get("frozen_signal_params", None)
 
    # Stage BC: start from frozen A params (includes model_type, lags, model hparams, etc.)
    if _hpo_stage == "BC_exec_calib":
        if not isinstance(_frozen, dict) or not _frozen:
            raise ValueError("BC_exec_calib requires stage_config['frozen_signal_params']")
        from copy import deepcopy as _dc
        params = {k: _dc(v) for k, v in _frozen.items()}
        model_type = params.get("model_type", None)
        if not model_type:
            raise ValueError("frozen_signal_params missing required key: model_type")

        # --- (B) Calibration knobs ---
        params["calibrate_method"] = trial.suggest_categorical(
            "calibrate_method", ["", "isotonic", "sigmoid"]
        )

        deep_models = {"lstm", "cnn", "transformer"}
        if str(model_type).lower() in deep_models:
            params["deep_calibrate"] = True
            params["deep_calibration_method"] = "temperature"
            # narrow around baseline if present
            _base_frac = float(params.get("deep_calibration_frac", 0.12) or 0.12)
            _frac_m = float(_stage_cfg.get("bc_deep_calibration_frac_margin", 0.04) or 0.04)
            _lo = max(0.05, _base_frac - _frac_m)
            _hi = min(0.30, _base_frac + _frac_m)
            params["deep_calibration_frac"] = trial.suggest_float("deep_calibration_frac", _lo, _hi)

            _base_min = int(params.get("deep_calibration_min_samples", 1200) or 1200)
            _min_m = int(_stage_cfg.get("bc_deep_calibration_min_samples_margin", 400) or 400)
            _lo_i = max(400, _base_min - _min_m)
            _hi_i = min(5000, _base_min + _min_m)
            params["deep_calibration_min_samples"] = trial.suggest_int(
                "deep_calibration_min_samples", _lo_i, _hi_i, step=200
            )
        else:
            params["deep_calibrate"] = False
 
        #  --- (C) FINAL EXPERIMENT POLICY LOCK (do NOT optimize in Optuna) ---
        params["target_active_rate"] = float(target_coverage_policy(model_type))
        params["target_coverage"] = params["target_active_rate"]
        params["alpha_vol_z"] = 0.004
        params["beta_spread_norm"] = 0.008
        params["gamma_slip_norm"] = 0.004
        params["slip_norm_bps"] = float(params.get("slip_norm_bps", 0.25) or 0.25)
        params["vol_window_bars"] = 96
        params["high_vol_q"] = 0.85
        params["high_vol_conf_bump"] = 0.0
        params["runtime_active_band_margin"] = 0.08
        params["runtime_conf_nudge"] = 0.005
        params["runtime_coverage_window"] = 192
        
        params.setdefault("allow_conf_backoff_cv", False)
        params.setdefault("allow_conf_backoff_eval", False)
        params["eval_use_trading_costs"] = True

        return params
 
    # Stage single or A: model_type is a sampled dimension
    model_type = trial.suggest_categorical("model_type", models_to_test)
    params = {"model_type": model_type}

    # === Train/Test span ===
    CONFIG_TTM = TRAIN_TEST_MONTHS_DEBUG if TRAIN_TEST_DEBUG_MODE else TRAIN_TEST_MONTHS
    train_min, train_max = CONFIG_TTM[model_type]["train"]
    test_min,  test_max  = CONFIG_TTM[model_type]["test"]
    params["train_months"] = trial.suggest_int("train_months", train_min, train_max)
    params["test_months"]  = 1

    # === Core temporal knobs ===
    if str(model_type).startswith("ensemble_"):
        l_lo, l_hi = 8, 24
    else:
        l_lo, l_hi = 12, 40
    params["lags_range"] = trial.suggest_int("lags_range", l_lo, l_hi)
    
    # Allow deeper lag depth for ensembles (otherwise ensemble_* lag knobs are pointless)
    if str(model_type).startswith("ensemble_"):
        params["lag_depth"] = trial.suggest_int("lag_depth", 1, 4)
    else:
        params["lag_depth"] = trial.suggest_int("lag_depth", 1, 3)
    roll_key = trial.suggest_categorical(
        "roll_windows_key", ["5", "5,10", "5,10,20", "10,30,60", "20,60"]
    )
    # Store alias for backwards compatibility (NOT an Optuna dimension)
    params["roll_windows_key_v2"] = roll_key
    params["roll_windows_key"] = roll_key
    params["roll_windows"] = [int(x) for x in roll_key.split(",")]

    # === Volatility-scaled label threshold (k · σ) ===
    sigma = None

    # 1) Prefer precomputed σ from run_optuna_tuning (cheap, reused every trial)
    if isinstance(vol_stats, dict) and "sigma48" in vol_stats:
        try:
            sigma = float(vol_stats["sigma48"])
        except (TypeError, ValueError):
            sigma = None

    # 2) Fallback: compute σ from train_data only if not provided
    if (
        sigma is None
        and train_data is not None
        and hasattr(train_data, "columns")
        and "returns" in train_data.columns
    ):
        r = train_data["returns"].astype("float64").dropna()
        if r.size > 0:
            sigma = float(r.rolling(48).std().median())
            sigma = float(np.clip(sigma, 1e-5, 5e-3))

    # 3) Use σ if we have it, otherwise fall back to generic prior
    if sigma is not None:
        lo = max(7.5e-5, 0.45 * sigma)
        hi = min(5e-3, 1.10 * sigma)
        label_thr = trial.suggest_float("label_threshold", lo, hi, log=True)
        _record_hp_boundary_hit("label_threshold", label_thr, lo, hi)
    else:
        lo, hi = 2e-4, 5e-3
        label_thr = trial.suggest_float("label_threshold", lo, hi, log=True)
        _record_hp_boundary_hit("label_threshold", label_thr, lo, hi)

    params["label_threshold"] = label_thr

    
    # (Coverage-anchored gating and cost-aware knobs are configured later in a single unified block.)

    # --- Deep calibration ---
    # For deep models (LSTM/CNN/Transformer), always enable temperature
    # calibration so their probability scale is more stable across time.
    # For non-deep models this flag is ignored by the backtester.
    deep_models = {"lstm", "cnn", "transformer"}
    
    if model_type in deep_models:
        params["deep_calibrate"] = True
        params["deep_calibration_method"] = "temperature"
        if _hpo_stage == "A_signal":
            params["deep_calibration_frac"] = float(_stage_cfg.get("stageA_deep_calibration_frac", 0.12) or 0.12)
            params["deep_calibration_min_samples"] = int(_stage_cfg.get("stageA_deep_calibration_min_samples", 1200) or 1200)
        else:
            params["deep_calibration_frac"] = trial.suggest_float("deep_calibration_frac", 0.08, 0.20)
            params["deep_calibration_min_samples"] = trial.suggest_int("deep_calibration_min_samples", 800, 2000, step=200)
    else:
        params["deep_calibrate"] = False


    # === Strategy family & TA backbone (profiled via MLB_TA_MODE) ===
    ta_mode = MLB_TA_MODE or "legacy"
    if ta_mode == "fixed":
        _apply_ta_profile_fixed(trial, params)
    elif ta_mode == "tuned":
        _apply_ta_profile_tuned(trial, params)
    else:
        _apply_ta_profile_legacy(trial, params)

    # --- Guard ---
    if not models_to_test:
        raise ValueError("models_to_test must contain at least one model type!")

    # === Global opt-in knobs (feature engineering & evaluation) ===
    # === Labeling (triple-barrier) — locked ON, practitioner-friendly ranges ===
    params["use_triple_barrier"] = True

    # Tight stops + long holding horizons collapse the neutral class (timeouts),
    # causing 3-class folds to become effectively binary. Constrain TB so class=1
    # exists consistently across mini-block folds.
    tb_pt_low, tb_pt_high = 1.50, 3.00
    tb_sl_low, tb_sl_high = 1.50, 3.00
    tb_hold_low, tb_hold_high = 24, 48
    # Neutral band: multiplier on local σ; applies only on timeout.
    tb_nz_low, tb_nz_high = 1.00, 2.00
 

    params["tb_pt_mult"] = trial.suggest_float("tb_pt_mult", tb_pt_low, tb_pt_high, step=0.25)
    _record_hp_boundary_hit("tb_pt_mult", params["tb_pt_mult"], tb_pt_low, tb_pt_high)

    params["tb_sl_mult"] = trial.suggest_float("tb_sl_mult", tb_sl_low, tb_sl_high, step=0.25)
    _record_hp_boundary_hit("tb_sl_mult", params["tb_sl_mult"], tb_sl_low, tb_sl_high)

    params["tb_max_holding"] = trial.suggest_int("tb_max_holding", tb_hold_low, tb_hold_high, step=12)
    _record_hp_boundary_hit("tb_max_holding", params["tb_max_holding"], tb_hold_low, tb_hold_high)

    params["tb_neutral_zone"] = trial.suggest_float("tb_neutral_zone", tb_nz_low, tb_nz_high, step=0.25)
    _record_hp_boundary_hit("tb_neutral_zone", params["tb_neutral_zone"], tb_nz_low, tb_nz_high)

    # === Calibration method — probabilistic heads (STABLE CHOICES) ===
    # Use "" for "no calibration" to keep Optuna's distribution stable across the study.
    if _hpo_stage == "A_signal":
         params["calibrate_method"] = str(_stage_cfg.get("stageA_calibrate_method", "") or "")
    else:
        params["calibrate_method"] = trial.suggest_categorical("calibrate_method", ["", "isotonic", "sigmoid"])

    # === Feature engineering toggles ===
    params["use_fracdiff"]     = trial.suggest_categorical("use_fracdiff", [False, True])
    params["fracdiff_d"]       = trial.suggest_float("fracdiff_d", 0.4, 0.7, step=0.05)
    params["use_rv_features"]  = trial.suggest_categorical("use_rv_features", [False, True])
    params["rv_window_short"]  = trial.suggest_int("rv_window_short", 20, 60, step=10)
    params["rv_window_long"]   = trial.suggest_int("rv_window_long", 80, 240, step=20)
    
    # Indicator state features (oscillator & volatility regimes)
    #     # These capture overbought/oversold (RSI, Stoch) and BB-width-based
    # compression/expansion without hard-wiring trading rules.
    params["use_indicator_states"]   = trial.suggest_categorical("use_indicator_states", [False, True])

    # Oscillator thresholds: we search around common practitioner ranges.
    params["rsi_overbought_level"]   = trial.suggest_int("rsi_overbought_level", 65, 80, step=5)
    params["rsi_oversold_level"]     = trial.suggest_int("rsi_oversold_level", 20, 35, step=5)
    params["stoch_overbought_level"] = trial.suggest_int("stoch_overbought_level", 75, 90, step=5)
    params["stoch_oversold_level"]   = trial.suggest_int("stoch_oversold_level", 10, 30, step=5)

    # Bollinger-band width thresholds (dimensionless):
    # compress ~ low width (squeeze); expand ~ high width (volatility expansion).
    params["bbw_compress_threshold"] = trial.suggest_float("bbw_compress_threshold", 0.02, 0.15)
    params["bbw_expand_threshold"]   = trial.suggest_float("bbw_expand_threshold", 0.10, 0.40)
    
    # Donchian channel / breakout features:
    #  - use_donchian toggles the family on/off;
    #  - windows are chosen from ranges that roughly correspond to short/medium
    #    trend horizons on 30-minute bars (Brock et al. 1992 style tests).
    params["use_donchian"]            = trial.suggest_categorical("use_donchian", [False, True])
    params["donchian_window_short"]   = trial.suggest_int("donchian_window_short", 20, 60, step=10)
    params["donchian_window_long"]    = trial.suggest_int("donchian_window_long", 80, 240, step=20)

    # -------- Coverage-anchored gating + adaptive nudges (Optuna-optimized) --------

    mt = str(params.get("model_type", model_type)).lower()

    if mt == "lstm":
        # LSTM: most relaxed → highest target coverage, softest α/β/γ,
        # widest band, strongest nudge.
        # +0.12 shift on the coverage range compared to the base (0.20–0.40 → 0.32–0.52)
        targ_low, targ_high = 0.32, 0.52

        alpha_max = 0.040
        beta_max  = 0.080
        gamma_max = 0.080

        band_low, band_high   = 0.10, 0.22
        nudge_low, nudge_high = 0.020, 0.050

        vol_window_choices  = [32, 48, 64]
        runtime_window_min  = 36
        runtime_window_max  = 84
        runtime_window_step = 12
        high_vol_bump_max   = 0.03  # do not stack too aggressively with α·vol_z

    elif mt  == "logistic":
        # Logistic: mid softness.
        # Still looser than default, but a bit tighter and slightly lower coverage
        # than LSTM so realised activity sits between.
        targ_low, targ_high = 0.28, 0.48

        alpha_max = 0.035
        beta_max  = 0.070
        gamma_max = 0.070

        band_low, band_high   = 0.08, 0.18
        nudge_low, nudge_high = 0.015, 0.040

        vol_window_choices  = [32, 48, 64]
        runtime_window_min  = 48
        runtime_window_max  = 108
        runtime_window_step = 12
        high_vol_bump_max   = 0.03

    else:
        # Default behaviour for all other models (CNN, transformer, ensembles, classical, etc.).
        # Keep a substantially lower target_active_rate so models do not over-trade once costs are applied.
        targ_low, targ_high = 0.05, 0.22

        # Make cost-awareness matter more in bad regimes (high vol / wide spreads / high slippage).
        alpha_max = 0.040
        beta_max  = 0.080
        gamma_max = 0.080

        # Runtime coverage band + nudge:
        # - narrower band so coverage cannot drift too far from the target
        # - smaller nudge so we do NOT get pushed into "always in the market"
        band_low, band_high   = 0.02, 0.05
        nudge_low, nudge_high = 0.003, 0.010

        # Slower, smoother coverage control for non-LSTM/non-log models.
        vol_window_choices  = [32, 48, 64, 96]
        runtime_window_min  = 96
        runtime_window_max  = 192
        runtime_window_step = 24
        high_vol_bump_max   = 0.03


    # FINAL EXPERIMENT POLICY LOCK (do NOT optimize in Optuna)
    params["target_active_rate"] = float(target_coverage_policy(mt))
    params["target_coverage"] = params["target_active_rate"]
    params["alpha_vol_z"] = 0.004
    params["beta_spread_norm"] = 0.0008
    params["gamma_slip_norm"] = 0.004
    params["slip_norm_bps"] = 0.25
    params["vol_window_bars"] = 96
    params["high_vol_q"] = 0.85
    params["high_vol_conf_bump"] = 0.0
    params["runtime_active_band_margin"] = 0.02
    params["runtime_conf_nudge"] = 0.015
    params["runtime_coverage_window"] = 48

    # Keep backoff flags off under coverage-anchored regime.
    params.setdefault("allow_conf_backoff_cv",   False)
    params.setdefault("allow_conf_backoff_eval", False)

    # Ensure after-cost evaluation during CV (you can override this if you want raw PnL).
    params["eval_use_trading_costs"] = True


    # === Model-specific spaces ===
    if model_type == "svm":
        # C, gamma ranges expanded to align with common SVM grids (Hsu–Chang–Lin guide):
        #   C    ~ [1e-3, 1e+3]
        #   gamma~ [1e-5, 1e+1]
        # Still log-scaled to focus on orders of magnitude rather than linear steps.
        params["svm_C"]            = trial.suggest_float("svm_C", 1e-3, 1e3, log=True)
        params["svm_gamma"]        = trial.suggest_float("svm_gamma", 1e-5, 10.0, log=True)
        params["svm_kernel"]       = "rbf"    # fix kernel; keep search compact
        params["svm_class_weight"] = trial.suggest_categorical("svm_class_weight", [None, "balanced"])


    elif model_type == "decision_tree":
        # Pre-pruning + cost-complexity pruning; ranges aligned with tree practice.
        # Depth: allow shallow (3) to moderately deep (30) trees.
        params["dt_max_depth"]         = trial.suggest_int("dt_max_depth", 3, 30)

        # Split / leaf: cover conservative to more flexible configurations.
        params["dt_min_samples_split"] = trial.suggest_int("dt_min_samples_split", 2, 80)
        params["dt_min_samples_leaf"]  = trial.suggest_int("dt_min_samples_leaf", 1, 25)

        # Features per split: classic set used in literature and sklearn examples.
        params["dt_max_features"]      = trial.suggest_categorical(
            "dt_max_features", ["sqrt", "log2", None]
        )

        # Cost-complexity pruning (ccp_alpha): small band near zero; CV picks aggressiveness.
        params["dt_ccp_alpha"]         = trial.suggest_float("dt_ccp_alpha", 0.0, 0.01)


    elif model_type == "logistic":
        # Multinomial-friendly solvers; if you prefer stability, fix to 'lbfgs'
        solver  = trial.suggest_categorical("logit_solver", ["lbfgs", "newton-cg", "saga"])
        penalty = "l2" if solver in ("lbfgs", "newton-cg") else trial.suggest_categorical("logit_penalty", ["l1", "l2"])
        params["logit_solver"]       = solver
        params["logit_penalty"]      = penalty
        params["logit_C"] = trial.suggest_float("logit_C", 1e-4, 1e4, log=True)
        params["logit_tol"]          = trial.suggest_float("logit_tol", 1e-5, 1e-2, log=True)
        max_iter                     = trial.suggest_int("logit_max_iter", 200, 1000)
        if solver == "saga":
            max_iter = max(max_iter, 2000)
        params["logit_max_iter"]     = max_iter
        params["logit_class_weight"] = trial.suggest_categorical("logit_class_weight", [None, "balanced"])

    elif model_type == "random_forest":
        params["rf_n_jobs"]            = -1
        params["rf_n_estimators"]      = trial.suggest_int("rf_n_estimators", 200, 1200, step=50)
        params["rf_max_depth"]         = trial.suggest_categorical("rf_max_depth", [None, 6, 8, 10, 12, 16])
        params["rf_min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 20)
        params["rf_min_samples_leaf"]  = trial.suggest_int("rf_min_samples_leaf", 1, 20)
        params["rf_max_features"]      = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])
        params["rf_bootstrap"]         = trial.suggest_categorical("rf_bootstrap", [True, False])
        params["rf_class_weight"]      = trial.suggest_categorical(
            "rf_class_weight",
            [None, "balanced", "balanced_subsample"]
        )

    elif model_type == "xgboost":
        params["xgb_n_jobs"]           = -1
        params["xgb_n_estimators"]     = trial.suggest_int("xgb_n_estimators", 200, 1500, step=50)
        params["xgb_max_depth"]        = trial.suggest_int("xgb_max_depth", 3, 10)
        params["xgb_learning_rate"]    = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
        params["xgb_subsample"]        = trial.suggest_float("xgb_subsample", 0.6, 1.0)
        params["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0)
        params["xgb_gamma"]            = trial.suggest_float("xgb_gamma", 0.0, 5.0)
        params["xgb_min_child_weight"] = trial.suggest_int("xgb_min_child_weight", 1, 10)

        # Optuna log distributions require low > 0 (can't be 0.0)
        params["xgb_lambda"]           = trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True)  # reg_lambda
        params["xgb_alpha"]            = trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True)   # reg_alpha


    elif model_type == "lstm":
        # Capacity: modest width + shallow depth are standard in FX LSTMs
        # (e.g. 1–2 layers, 32–128 units) to avoid huge recurrent cores.
        params["lstm_units"]         = trial.suggest_int("lstm_units", 32, 128, step=32)
        params["lstm_num_layers"]    = trial.suggest_int("lstm_num_layers", 1, 2)
        params["lstm_dense_units"]   = trial.suggest_int("lstm_dense_units", 32, 192, step=32)

        # Dropout: enforce a healthier lower bound (≈0.2–0.5) instead of 0.05,
        # consistent with common practice for sequence models.
        params["lstm_dropout_rate"]  = trial.suggest_float("lstm_dropout_rate", 0.20, 0.50)

        # Learning rate: keep your existing, literature-consistent range
        # (~1e-3 baseline with log grid around it).
        params["lstm_learning_rate"] = trial.suggest_float("lstm_learning_rate", 1e-5, 5e-3, log=True)

        params["lstm_batch_size"]    = 256
        params["lstm_bidirectional"] = trial.suggest_categorical("lstm_bidirectional", [False, True])

        params["lstm_clipnorm"] = trial.suggest_categorical(
            "lstm_clipnorm",
            [0.0, 1.0, 5.0],
        )

        # === NEW: windowing/runtime knobs, mirroring CNN/Transformer ===
        # Whether to use the sequence-window branch (A) or simple 3D reshape (B).
        params["lstm_use_seq_windows"] = trial.suggest_categorical(
            "lstm_use_seq_windows",
            [True, False],
        )
        
        # ------------------------------------------------------------
        # Training-budget / hardware knobs are treated as *protocol*
        # constants (multi-fidelity CV), not Optuna dimensions.
        #
        # CV already enforces stride/window/epoch caps via cv_config,
        # while final refit runs at full fidelity. Keeping these fixed
        # avoids wasting trials on knobs that are later overridden.
        # ------------------------------------------------------------
        params["lstm_train_stride"] = 1
        params["lstm_mixed_precision"] = False

    elif model_type == "cnn":
        # CNN architecture & training hyperparameters.
        # We tune the *actual* Conv1D filters used by build_cnn() (filters1, filters2),
        # rather than unused proxy knobs (num_blocks, filters_base, stride_conv).

        # Conv filters: moderate-capacity CNN, aligned with typical 1D-CNN ranges
        # in time-series and financial applications (32–96 filters per layer).
        params["cnn_filters1"]      = trial.suggest_int("cnn_filters1", 32, 96)
        params["cnn_filters2"]      = trial.suggest_int("cnn_filters2", 32, 96)

        # Kernel size: small receptive fields (3–7) for local temporal patterns.
        params["cnn_kernel_size"]   = trial.suggest_categorical("cnn_kernel_size", [3, 5, 7])

        # Dense head width and dropout for regularisation.
        params["cnn_dense_units"]   = trial.suggest_int("cnn_dense_units", 32, 192, step=32)
        params["cnn_dropout_rate"]  = trial.suggest_float("cnn_dropout_rate", 0.05, 0.4)

        # Optimiser and batch size.
        params["cnn_learning_rate"] = trial.suggest_float("cnn_learning_rate", 1e-5, 5e-3, log=True)
        params["cnn_batch_size"]      = 256  # fixed: keep CV/refit consistent; avoid hardware-dependent HPO
        
        # Windowing / training strategy knobs (already used by test_strategy).
        params["cnn_use_seq_windows"] = trial.suggest_categorical("cnn_use_seq_windows", [True, False])
        # Protocol constants (see note in LSTM block).
        params["cnn_train_stride"]    = 1
        params["cnn_mixed_precision"] = False

    elif model_type == "transformer":
        # Heads & d_model (d_model = heads * mult)
        heads = trial.suggest_categorical("transformer_num_heads", [4, 8])
        min_mult = max(4, math.ceil(32 / heads))
        # Cap d_model to ≲128 to avoid extremely heavy Transformer configs.
        max_mult = min(16, 256 // heads)
        if min_mult > max_mult:
            raise optuna.TrialPruned(f"no valid d_model for heads={heads}")
        mult = trial.suggest_int("transformer_d_multiple_v2", min_mult, max_mult)
        d_model = heads * mult

        # ✅ ES REMOVED from Optuna search (CV guardrails will enforce ES)
        # Depth: keep very shallow (1–2 blocks) for FX TS to control overfitting.
        params["transformer_num_blocks"]    = trial.suggest_int("transformer_num_blocks", 1, 2)

        params["transformer_num_heads"]     = heads
        params["transformer_d_model"]       = d_model
        params["transformer_d_multiple_v2"] = mult

        # FFN width as 2–3× d_model (avoid 4× monsters).
        ff_mult = trial.suggest_categorical("transformer_ff_multiple", [2, 3])
        params["transformer_ff_multiple"]   = ff_mult
        params["transformer_ff_dim"]        = int(ff_mult * d_model)

        params["transformer_dropout_rate"]  = trial.suggest_float("transformer_dropout_rate", 0.05, 0.4)
        params["transformer_dense_units"]   = trial.suggest_int("transformer_dense_units", 64, 384, step=32)
        params["transformer_batch_size"]    = 256  # fixed: keep CV/refit consistent; avoid hardware-dependent HPO

        # Protocol constant (multi-fidelity CV will cap anyway; final refit uses stride=1).
        params["transformer_train_stride"]  = 2
        params["transformer_use_time2vec"]  = trial.suggest_categorical("transformer_use_time2vec", [False, True])
        params["transformer_pooling"]       = trial.suggest_categorical("transformer_pooling", ["cls", "mean"])


    # =====================================
    # ENSEMBLE: CNN + LSTM + XGBoost (CLX)
    # =====================================
    elif model_type == "ensemble_cnn_lstm_xgboost":
        
        params["fusion_alpha"]        = trial.suggest_float("fusion_alpha", 0.50, 0.80)

        # CNN head
        params["cnn_filters1"]      = trial.suggest_int("cnn_filters1", 32, 128)
        params["cnn_filters2"]      = trial.suggest_int("cnn_filters2", 32, 128)
        params["cnn_kernel_size"]   = trial.suggest_categorical("cnn_kernel_size", [3, 5])
        params["cnn_dense_units"]   = trial.suggest_int("cnn_dense_units", 32, 128)
        params["cnn_batch_size"]    = trial.suggest_categorical("cnn_batch_size", [128, 256, 384, 512])
        params["cnn_dropout_rate"]  = trial.suggest_float("cnn_dropout_rate", 0.20, 0.40)
        params["cnn_learning_rate"] = trial.suggest_float("cnn_learning_rate", 1e-4, 1e-3, log=True)

        # LSTM head
        params["lstm_units"]         = trial.suggest_int("lstm_units", 32, 128)
        params["lstm_dense_units"]   = trial.suggest_int("lstm_dense_units", 16, 64)
        params["lstm_batch_size"]    = 256
        params["lstm_dropout_rate"]  = trial.suggest_float("lstm_dropout_rate", 0.20, 0.40)
        params["lstm_learning_rate"] = trial.suggest_float("lstm_learning_rate", 1e-4, 1e-3, log=True)

        # XGBoost head
        params["xgb_n_estimators"]     = trial.suggest_int("xgb_n_estimators", 200, 600, step=50)
        params["xgb_learning_rate"]    = trial.suggest_float("xgb_learning_rate", 0.03, 0.20, log=True)
        params["xgb_max_depth"]        = trial.suggest_int("xgb_max_depth", 3, 7)
        params["xgb_min_child_weight"] = trial.suggest_int("xgb_min_child_weight", 1, 10)
        params["xgb_gamma"]            = trial.suggest_float("xgb_gamma", 0.0, 4.0)
        params["xgb_subsample"]        = trial.suggest_float("xgb_subsample", 0.6, 0.9)
        params["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.6, 0.9)

        # Ensemble plumbing
        # Protocol constant (CV enforces a minimum stride for cost control).
        params["ensemble_train_stride"] = 1


    # =====================================
    # ENSEMBLE: Adaptive Regime Router
    # =====================================
    elif model_type == "ensemble_adaptive_regime":
        adx_candidates = ["adx_14"]
        vol_candidates = ["rolling_std_20"]
        if train_data is not None and hasattr(train_data, "columns"):
            cols = list(train_data.columns)
            fa = sorted([c for c in cols if c.startswith("adx_")])
            fv = sorted([c for c in cols if c.startswith("rolling_std_") or c.startswith("atr_")])
            if fa: adx_candidates = fa
            if fv: vol_candidates = fv

        params["adx_col"]    = trial.suggest_categorical("adx_col", adx_candidates)
        params["vol_col"]    = trial.suggest_categorical("vol_col", vol_candidates)
        params["adx_thresh"] = trial.suggest_int("adx_thresh", 20, 25)

        # Data-aware prior for vol_thresh (kept from your logic)
        if train_data is not None and params["vol_col"] in getattr(train_data, "columns", []):
            _v = train_data[params["vol_col"]].dropna().astype(float)
            if _v.size > 100:
                _q_lo = float(_v.quantile(0.60))
                _q_hi = float(_v.quantile(0.80))
                params["vol_thresh"] = trial.suggest_float(
                    "vol_thresh",
                    max(_q_lo, 1e-12),
                    max(_q_hi, _q_lo + 1e-12)
                )
            else:
                params["vol_thresh"] = trial.suggest_float("vol_thresh", 1e-4, 5e-3, log=True)
        else:
            if train_data is not None and "returns" in getattr(train_data, "columns", []):
                _rv = train_data["returns"].rolling(20).std().dropna()
                if _rv.size:
                    _q_lo = float(_rv.quantile(0.60))
                    _q_hi = float(_rv.quantile(0.80))
                    params["vol_thresh"] = trial.suggest_float(
                        "vol_thresh",
                        max(_q_lo, 1e-12),
                        max(_q_hi, _q_lo + 1e-12)
                    )
                else:
                    params["vol_thresh"] = trial.suggest_float("vol_thresh", 1e-4, 5e-3, log=True)
            else:
                params["vol_thresh"] = trial.suggest_float("vol_thresh", 1e-4, 5e-3, log=True)

        # 🔹 Always train LSTM expert on trend-only windows (no toggle)
        params["train_lstm_on_trend_only"] = True

        # 🔹 LSTM expert (compact, single-layer)
        params["lstm_units"]         = trial.suggest_int("lstm_units", 32, 64)
        params["lstm_num_layers"]    = 1  # fixed to 1 layer for adaptive expert
        params["lstm_bidirectional"] = trial.suggest_categorical("lstm_bidirectional", [False])  # parity
        params["lstm_dense_units"]   = trial.suggest_int("lstm_dense_units", 16, 32)
        params["lstm_dropout_rate"]  = trial.suggest_float("lstm_dropout_rate", 0.2, 0.5)
        params["lstm_learning_rate"] = trial.suggest_float("lstm_learning_rate", 1e-4, 5e-3, log=True)
        params["lstm_batch_size"]    = 256  # fixed: keep CV/refit consistent; avoid hardware-dependent HPO

        # ES/patience disabled in your adaptive path by design

        # 🔹 RF expert (cheaper: fewer trees, shallower depth)
        params["rf_n_estimators"]     = trial.suggest_int("rf_n_estimators", 100, 400, step=50)
        params["rf_max_depth"]        = trial.suggest_int("rf_max_depth", 5, 10)
        params["rf_min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 6)
        params["rf_max_features"]     = trial.suggest_categorical("rf_max_features", ["sqrt"])
        params["rf_bootstrap"]        = trial.suggest_categorical("rf_bootstrap", [True])

        # Logit expert
        params["logit_C"]             = trial.suggest_float("logit_C", 1e-4, 1e4, log=True)
        params["logit_solver"]        = trial.suggest_categorical("logit_solver", ["lbfgs"])
        params["logit_class_weight"]  = trial.suggest_categorical("logit_class_weight", [None, "balanced"])

        # Ensemble plumbing
        # Protocol constant (CV enforces a minimum stride for cost control).
        params["ensemble_train_stride"] = 1

    # === Runtime guards for deep models (no wall-clock limits, just shapes) ===
    # LSTM/CNN: if we use sequence windows, do not allow stride=1 (window explosion).
    if model_type in ("lstm", "cnn"):
        use_seq_key = f"{model_type}_use_seq_windows"
        stride_key  = f"{model_type}_train_stride"
        use_seq     = params.get(use_seq_key, False)
        stride      = params.get(stride_key, 1)
        if use_seq and stride == 1:
            # Force a minimum stride of 2 when using seq windows to halve window count.
            params[stride_key] = 2

    # Transformer: prune the single most pathological combo (very wide + deep + stride=1).
    if model_type == "transformer":
        d_model = params.get("transformer_d_model", 64)
        blocks  = params.get("transformer_num_blocks", 1)
        ff_mult = params.get("transformer_ff_multiple", 2)
        # Default to 2 if missing (should be present; treated as protocol constant).
        stride  = params.get("transformer_train_stride", 2)
        if (
            d_model >= 128
            and blocks >= 2
            and ff_mult >= 3
            and stride == 1
            and optuna is not None
        ):
            raise optuna.TrialPruned("Pruned excessively heavy Transformer config (d_model/blocks/ff/stride).")

    return params



def _coerce_ensemble_lags(params: dict) -> dict:
    """
    Normalize namespaced ensemble lag keys to the generic ones used by the
    backtester/test_* entrypoints. Never overwrites explicit generic keys.
    """
    p = dict(params or {})
    if "ensemble_lags_range" in p and "lags_range" not in p:
        try:
            p["lags_range"] = int(p["ensemble_lags_range"])
        except Exception:
            pass
    if "ensemble_lag_depth" in p and "lag_depth" not in p:
        try:
            p["lag_depth"] = int(p["ensemble_lag_depth"])
        except Exception:
            pass
    # Convenience: if an explicit 'lags' is missing, derive it from lags_range.
    if "lags" not in p and "lags_range" in p:
        try:
            p["lags"] = int(p["lags_range"])
        except Exception:
            pass
    return p


def refit_cnn_with_overrides(backtester, best_params,
                             train_start, train_end, test_start, test_end,
                             overrides=None, **_):
    """
    Refit the CNN once with 'final' settings after Optuna tuning.
    Keeps the best architecture/opt params but relaxes speed caps.
    """
    overrides = overrides or {}

    # coerce any namespaced keys that might have come through
    best_params = _coerce_ensemble_lags(best_params)

    # figure out lags param (fallbacks cover your param names)
    lags = int(
        overrides.get("lags",
        best_params.get("lags",
        best_params.get("lags_range", 8)))
    )

    # merge configs → backtester reads self.features_config inside test_strategy
    final_cfg = dict(getattr(backtester, "features_config", {}) or {})
    final_cfg.update(best_params)
    final_cfg.update({
        # runtime overrides (make it “full” refit)
        "cnn_train_stride": 1,
        "cnn_epochs": max(30, int(best_params.get("cnn_epochs", 20))),
        "cnn_use_early_stopping": True,
        
        # Uncap deep training windows for final refit (CV is the cheap proxy)
        "deep_max_train_windows": 10**9,
        # keep windows mode consistent (or force True if you want)
        "cnn_use_seq_windows": final_cfg.get("cnn_use_seq_windows",
                             best_params.get("cnn_use_seq_windows", True)),
        # don’t force mixed precision unless you know you’re on GPU
        "cnn_mixed_precision": False,
    })
    # user-specified overrides win last
    final_cfg.update(overrides)


    # --- NEW: per-model warm-up bars ---
    # (Bugfix) CNN was incorrectly using the LSTM warm-up branch, which can
    # over-cut the eval window in real_trading_simulation and lead to 0 trades.
    model_label = "cnn"
    warm_bars = compute_required_test_warmup_bars({**final_cfg, "model_type": model_label})
    final_cfg["test_warmup_bars"] = int(warm_bars)

    backtester.features_config = final_cfg
    return backtester.test_strategy(
        train_start, train_end, test_start, test_end,
        lags=lags,

        confidence_threshold=float(final_cfg.get("confidence_threshold", 0.8)),
        label_threshold=float(final_cfg.get("label_threshold", 0.0001)),
    )


def refit_lstm_with_overrides(backtester, best_params,
                              train_start, train_end, test_start, test_end,
                              overrides=None, **_):
    overrides = overrides or {}
    best_params = _coerce_ensemble_lags(best_params)

    lags = int(overrides.get("lags",
             best_params.get("lags", best_params.get("lags_range", 8))))

    final_cfg = dict(getattr(backtester, "features_config", {}) or {})
    final_cfg.update(best_params)
    final_cfg.update({
        # full refit (no speed caps) — but KEEP the same windowing regime as CV
        "lstm_use_seq_windows": bool(
            best_params.get(
                "lstm_use_seq_windows",
                final_cfg.get("lstm_use_seq_windows", False)  # default: 3D-feed, like CV
            )
        ),
        "lstm_train_stride": 1,
        "lstm_epochs": max(30, int(best_params.get("lstm_epochs", 20))),
        "lstm_use_early_stopping": True,
        # Uncap deep training windows for final refit (CV is the cheap proxy)
        "deep_max_train_windows": 10**9,
        "lstm_mixed_precision": False,
    })

    final_cfg.update(overrides)

    # --- NEW: per-model warm-up bars ---
    model_label = "lstm"
    warm_bars = compute_required_test_warmup_bars({**final_cfg, "model_type": model_label})
    final_cfg["test_warmup_bars"] = int(warm_bars)

    backtester.features_config = final_cfg
    return backtester.test_strategy(
        train_start, train_end, test_start, test_end,
        lags=lags,
        confidence_threshold=float(final_cfg.get("confidence_threshold", 0.8)),
        label_threshold=float(final_cfg.get("label_threshold", 0.0001)),
    )


def refit_transformer_with_overrides(backtester, best_params,
                                     train_start, train_end, test_start, test_end,
                                     overrides=None, **_):
    overrides = overrides or {}
    best_params = _coerce_ensemble_lags(best_params)

    lags = int(overrides.get("lags",
             best_params.get("lags", best_params.get("lags_range", 8))))

    final_cfg = dict(getattr(backtester, "features_config", {}) or {})
    final_cfg.update(best_params)
    final_cfg.update({
        "transformer_train_stride": 1,
        "transformer_epochs": max(30, int(best_params.get("transformer_epochs", 20))),
        "transformer_use_early_stopping": True,
        # Uncap deep training windows for final refit (CV is the cheap proxy)
        "deep_max_train_windows": 10**9,
        "transformer_mixed_precision": False,  # safer unless GPU
    })
    final_cfg.update(overrides)

    # --- NEW: per-model warm-up bars ---
    model_label = "transformer"
    warm_bars = compute_required_test_warmup_bars({**final_cfg, "model_type": model_label})
    final_cfg["test_warmup_bars"] = int(warm_bars)

    backtester.features_config = final_cfg
    return backtester.test_strategy(
        train_start, train_end, test_start, test_end,
        lags=lags,
        confidence_threshold=float(final_cfg.get("confidence_threshold", 0.8)),
        label_threshold=float(final_cfg.get("label_threshold", 0.0001)),
    )


def refit_ensemble_cnn_lstm_xgb_with_overrides(backtester, best_params,
                                               train_start, train_end, test_start, test_end,
                                               overrides=None, **_):
    """
    Final, uncapped refit for the CNN+LSTM+XGB ensemble (one-shot; no Optuna).
    Uses the best params, disables time caps/stride, and bumps epochs—then evaluates the fold.
    """
    overrides = overrides or {}
    params = _coerce_ensemble_lags(dict(best_params))  # do not mutate caller

    # Lags / labels
    lags = int(params.get("lags", params.get("lags_range", 8)))
    label_threshold = float(params.get("label_threshold", 0.0001))

    # Start from any prepacked ensemble_config if present
    ensemble_config = dict(params.get("ensemble_config", {}))

    # Ensure all flat, namespaced keys from best_params (cnn_/lstm_/xgb_/ensemble_) are present
    for k, v in params.items():
        if isinstance(k, str) and (k.startswith("cnn_") or k.startswith("lstm_") or
                                   k.startswith("xgb_") or k.startswith("ensemble_")):
            ensemble_config.setdefault(k, v)

    # Apply caller overrides (namespaced keys expected)
    ensemble_config.update({k: v for k, v in overrides.items()})

    # 🚩 Final refit must be uncapped → hard overrides (no setdefault)
    ensemble_config["ensemble_train_stride"] = 1
    ensemble_config["ensemble_deep_max_train_windows"] = 10**9

    # Be a bit more patient on final fit
    if "cnn_epochs" in params:
        ensemble_config["cnn_epochs"] = max(int(params["cnn_epochs"]), 30)
    if "lstm_epochs" in params:
        ensemble_config["lstm_epochs"] = max(int(params["lstm_epochs"]), 30)
    ensemble_config["cnn_patience"]  = max(int(params.get("cnn_patience", 10)), 10)
    ensemble_config["lstm_patience"] = max(int(params.get("lstm_patience", 10)), 10)
    ensemble_config["early_stopping_rounds"] = max(int(ensemble_config.get("early_stopping_rounds", 100)), 100)

    # --- per-model warm-up bars for the ensemble ---
    model_label = "ensemble_cnn_lstm_xgboost"
    warm_bars = compute_required_test_warmup_bars({**ensemble_config, "model_type": model_label})
    ensemble_config["test_warmup_bars"] = int(warm_bars)

    # One-shot evaluation with full settings
    return backtester.test_ensemble_strategy(
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
        lags=lags, label_threshold=label_threshold,
        ensemble_config=ensemble_config, model_type="ensemble_cnn_lstm_xgboost"
    )


def refit_ensemble_adaptive_regime_with_overrides(backtester, best_params,
                                                  train_start, train_end, test_start, test_end,
                                                  overrides=None, **_):
    """
    Final, uncapped refit for the Adaptive Regime ensemble (one-shot; no Optuna).
    Uses the best params, disables time caps/stride, and bumps epochs—then evaluates the fold.
    """
    overrides = overrides or {}
    params = _coerce_ensemble_lags(dict(best_params))

    # Lags / labels
    lags = int(params.get("lags", params.get("lags_range", 8)))
    label_threshold = float(params.get("label_threshold", 0.0001))

    # Start from any prepacked ensemble_config if present
    ensemble_config = dict(params.get("ensemble_config", {}))

    # Propagate namespaced/bare keys from best_params
    for k, v in params.items():
        if isinstance(k, str) and (k.startswith("lstm_") or k.startswith("rf_") or
                                   k.startswith("logit_") or k.startswith("ensemble_") or
                                   k in {"adx_col","vol_col","adx_thresh","vol_thresh",
                                         "adx_thresh_q","train_lstm_on_trend_only"}):
            ensemble_config.setdefault(k, v)

    # Apply overrides
    ensemble_config.update({k: v for k, v in (overrides or {}).items()})

    # 🚩 Final refit must be uncapped → hard overrides (no setdefault)
    ensemble_config["ensemble_train_stride"] = 1
    ensemble_config["ensemble_deep_max_train_windows"] = 10**9

    if "lstm_epochs" in params:
        ensemble_config["lstm_epochs"] = max(int(params["lstm_epochs"]), 30)
    ensemble_config["lstm_patience"] = max(int(params.get("lstm_patience", 10)), 10)

    # --- Sanitize any stray NaNs in logit class_weight coming from Top-N payloads ---
    cw = None
    if "class_weight" in ensemble_config:
        cw = ensemble_config["class_weight"]
    elif "logit_class_weight" in ensemble_config:
        cw = ensemble_config["logit_class_weight"]

    try:
        import numpy as np
        if isinstance(cw, float) and (np.isnan(cw) or np.isinf(cw)):
            cw = None
    except Exception:
        pass
    if isinstance(cw, str) and cw.strip().lower() in ("nan", "null", "none", ""):
        cw = None
    if cw not in (None, "balanced") and not isinstance(cw, dict):
        cw = None

    # Normalize to the namespaced key your filter_params expects
    ensemble_config["logit_class_weight"] = cw
    ensemble_config["class_weight"] = cw  # mirror (optional)

    # --- per-model warm-up bars for the ensemble ---
    model_label = "ensemble_adaptive_regime"
    warm_bars = compute_required_test_warmup_bars({**ensemble_config, "model_type": model_label})
    ensemble_config["test_warmup_bars"] = int(warm_bars)

    return backtester.test_ensemble_strategy(
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
        lags=lags, label_threshold=label_threshold,
        ensemble_config=ensemble_config, model_type="ensemble_adaptive_regime"
    )


def _evaluate_original_no_refit(backtester, best_params,
                                train_start, train_end, test_start, test_end, allow_dqn: bool = False):
    """
    Re-run a single-fold evaluation with the original (Optuna-picked) params
    WITHOUT uncapping or overriding anything. Returns (metrics, model_type).
    """
    params = _coerce_ensemble_lags(dict(best_params))
    mtype  = params.get("model_type", getattr(backtester, "model_type", None))
    lags   = int(params.get("lags", params.get("lags_range", 8)))
    label_threshold = float(params.get("label_threshold", 0.0001))
    conf_thr = float(params.get("confidence_threshold", 0.8))

    # Prepare ensemble_config (if present)
    ensemble_config = dict(params.get("ensemble_config", {}))
    # Also propagate any namespaced keys (cnn_/lstm_/xgb_/rf_/logit_/ensemble_)
    for k, v in params.items():
        if isinstance(k, str) and (k.startswith(("cnn_","lstm_","xgb_","rf_","logit_","ensemble_")) or
                                   k in {"adx_col","vol_col","adx_thresh","vol_thresh","train_lstm_on_trend_only"}):
            ensemble_config.setdefault(k, v)

    # --- compute and attach warm-up (NO-REFIT path) ---
    model_label = str(mtype).lower() if isinstance(mtype, str) else ""
    cfg_for_warmup = {**params, **ensemble_config, "model_type": model_label}
    if model_label == "ensemble_transformer_xgb_dqn":
        cfg_for_warmup["uses_dqn"] = bool(allow_dqn)  # respect gating exactly
    warm_bars = compute_required_test_warmup_bars(cfg_for_warmup)

    # Compute the lags we will ACTUALLY use for evaluation (single source of truth)
    tuned_lags = int(params.get("lags", params.get("lags_range", lags)))

    # Build the exact eval features_config: defaults → (prior run cfg) → best_params
    # Force tuned lags into the merged config so the downstream pipeline (warmup, shapes)
    # stays consistent with what Optuna tuned/printed.
    cfg_merged = backtester._merge_params_into_features_config(params, force_lags=tuned_lags)
    # lock keys for this evaluation
    setattr(backtester, "_optuna_locked_keys", set(params.keys()))

    # tripwire (optional but helpful during bring-up)
    if getattr(backtester, "_optuna_locked_keys", None):
        clobbered = {k for k in backtester._optuna_locked_keys
                     if k in cfg_merged and cfg_merged[k] != params.get(k)}
        if clobbered:
            print(f"⚠️ Optuna keys changed in eval merge: {sorted(clobbered)}")

    cfg_merged["test_warmup_bars"] = int(warm_bars)

    # Lock the merged config for this evaluation only
    backtester.features_config = cfg_merged

    print(f"[EVAL-SNAPSHOT] model={mtype} | lags={tuned_lags} | "
        f"lag_depth={cfg_merged.get('lag_depth')} | "
        f"roll_windows={cfg_merged.get('roll_windows') or cfg_merged.get('roll_windows_key') or cfg_merged.get('roll_windows_key_v2')} | "
        f"use_fracdiff={cfg_merged.get('use_fracdiff')} | "
        f"confidence_threshold={conf_thr} | calibrate_method={cfg_merged.get('calibrate_method')}")
    
    # ---- Patch C: tuning clarity ----
    # If coverage/target_active_rate is set, the effective threshold is coverage-calibrated in MLBacktester,
    # so the tuned confidence_threshold is not the operative control knob.
    try:
        tar = cfg_merged.get("target_active_rate", cfg_merged.get("target_coverage", None))
        if tar is not None:
            print(f"[GateInfo][TUNING] target_active_rate={float(tar):.6f} is set → confidence_threshold is overridden by coverage thresholding.")
    except Exception:
        pass

    # Refuse to silently change the tuned lags
    allow_shrink = bool(cfg_merged.get("allow_eval_lag_shrink", False))
    if "lags_range" in params:
        tuned = int(params["lags_range"])
        if tuned_lags != tuned and not allow_shrink:
            raise RuntimeError(
                f"[EVAL] Refusing to change lags: tuned={tuned} != eval={tuned_lags}. "
                "Increase warm-up / adjust session/embargo, or set allow_eval_lag_shrink=True explicitly."
            )

    # Choose the right evaluation path
    is_ensemble = isinstance(mtype, str) and mtype.startswith("ensemble_")
    if is_ensemble:
        metrics = backtester.test_ensemble_strategy(
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            lags=tuned_lags,
            label_threshold=label_threshold,
            ensemble_config=ensemble_config,
            model_type=mtype,
        )
    else:
        metrics = backtester.test_strategy(
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            lags=tuned_lags,
            confidence_threshold=conf_thr,
            label_threshold=label_threshold,
        )

    return metrics, mtype


def _select_better_result(
    backtester,
    best_params,
    train_start,
    train_end,
    test_start,
    test_end,
    refit_func,
    select_metric="cstrategy",
    tolerance=0.01,
    overrides=None,
):
    """
    Runs both: (A) original best (no refit), (B) refit (uncapped),
    then returns the metrics of the better one.

    select_metric: which metric to maximize ("cstrategy" or "sharpe" are good choices).
    tolerance: relative tolerance (e.g., 0.01 = 1%) where ties prefer ORIGINAL.
    """
    metric_names = [
        "cstrategy", "outperformance", "creturns", "sharpe", "drawdown", "trades",
        "geo_mean_ann", "directional_accuracy", "precision_macro", "f1_macro",
        "active_rate", "profit_per_hit", "return_per_trade", "win_rate",
        "strategy_volatility", "kurtosis",
    ]
    if select_metric not in metric_names:
        raise ValueError(f"Unknown select_metric '{select_metric}'. Allowed: {metric_names}")

    # A) Original (no uncapping)
    orig_metrics, _ = _evaluate_original_no_refit(
        backtester,
        best_params,
        train_start,
        train_end,
        test_start,
        test_end,
    )

    def _safe_val(m):
        try:
            v = float(m[metric_names.index(select_metric)])
            return v if (v == v) else float("-inf")  # NaN check
        except Exception:
            return float("-inf")

    orig_val = _safe_val(orig_metrics)

    # Build a safe overrides dict that cannot clobber Optuna-picked keys
    locked = getattr(backtester, "_optuna_locked_keys", set())
    safe_overrides = {k: v for k, v in (overrides or {}).items() if k not in locked}

    # B) Refit (uncapped via provided refit_func)
    refit_metrics = refit_func(
        backtester,
        best_params,
        train_start,
        train_end,
        test_start,
        test_end,
        overrides=safe_overrides,
    )

    refit_val = _safe_val(refit_metrics)

    better = "original"
    chosen = orig_metrics
    if refit_val > orig_val * (1.0 + tolerance):
        better = "refit"
        chosen = refit_metrics
        print(
            f"[FINAL-SELECT] Metric={select_metric} | "
            f"original={orig_val:.6f} | refit={refit_val:.6f} | chosen={better}"
        )

    # Final sanity check on metrics shape (defensive)
    if not isinstance(chosen, (list, tuple)) or len(chosen) != len(metric_names):
        print("⚠️ Warning: invalid metrics shape from refit/selection; falling back to original.")
        return orig_metrics

    return chosen

def final_refit_if_deep(backtester, best_params,
                        train_start, train_end, test_start, test_end,
                        overrides=None,
                        select_metric="cstrategy", tolerance=0.01):
    """
    For deep/ensemble models: run a SINGLE uncapped refit with tuned params (+ overrides)
    and return its metrics. We DO NOT compare against the original snapshot here.
    
    For classical models: just re-run the original (no-refit) evaluation and
    return its metrics.

    This keeps the pipeline simple:
        Optuna best params → (optional) one deployment refit → trade.
    """
    # Lock tuned keys so overrides cannot silently clobber them
    setattr(backtester, "_optuna_locked_keys", set(best_params.keys()))
    m = best_params.get("model_type", getattr(backtester, "model_type", None))

    # --- Deep single models: one-shot uncapped refit ---
    if m == "cnn":
        return refit_cnn_with_overrides(
            backtester=backtester,
            best_params=best_params,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            overrides=overrides,
        )

    if m == "lstm":
        return refit_lstm_with_overrides(
            backtester=backtester,
            best_params=best_params,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            overrides=overrides,
        )

    if m == "transformer":
        return refit_transformer_with_overrides(
            backtester=backtester,
            best_params=best_params,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            overrides=overrides,
        )

    # --- Deep ensembles: one-shot uncapped refit ---
    if m == "ensemble_cnn_lstm_xgboost":
        return refit_ensemble_cnn_lstm_xgb_with_overrides(
            backtester=backtester,
            best_params=best_params,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            overrides=overrides,
        )

    if m == "ensemble_adaptive_regime":
        return refit_ensemble_adaptive_regime_with_overrides(
            backtester=backtester,
            best_params=best_params,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            overrides=overrides,
        )

    # --- Classical / unknown: just re-run original snapshot once ---
    metrics, _ = _evaluate_original_no_refit(
        backtester,
        best_params,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    return metrics



def _aggressive_free():
    try:
        import tensorflow as _tf
        _tf.keras.backend.clear_session()
    except Exception:
        pass

    # Optional: release joblib/loky workers (can reduce RAM high-water during long studies)
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass

    import gc as _gc, time as _time
    _gc.collect()
    _time.sleep(0.05)
    
    
def _assert_free_ram(need_gb: float | str, trial=None) -> bool:
    import os, psutil
    # --- Coerce need_gb to float (accept legacy label strings) ---
    try:
        need_gb = float(need_gb)
    except Exception:
        # fall back to env default if a label like "trial start" slips in
        need_gb = float(os.getenv("OPTUNA_MIN_FREE_GB", "0.40"))

    # --- NEW: percent-of-total + GPU VRAM thresholds ---
    total_gb = psutil.virtual_memory().total / (1024 ** 3)

    pct = float(os.getenv("OPTUNA_MIN_FREE_GB_PERCENT", "0"))
    if pct > 0:
        # enforce the larger of (absolute need_gb) vs (percent-of-total)
        need_gb = max(need_gb, total_gb * pct)

    # inside _assert_free_ram(...)
    vram_need = float(os.getenv("OPTUNA_MIN_FREE_VRAM_GB", "0"))

    def _has_vram():
        if vram_need <= 0:
            return True
        try:
            from pipeline.workers import get_gpu_free_memory_mb
            free_list = get_gpu_free_memory_mb()
            if not free_list:
                return True  # No GPU or nvidia-smi unavailable — don't block
            free_gb = free_list[0] / 1024.0  # Pick GPU 0
            return free_gb >= vram_need
        except Exception:
            # If nvidia-smi not present or parsing fails, don't block
            return True


    avail = psutil.virtual_memory().available / (1024 ** 3)
    if avail >= need_gb and _has_vram():
        return True

    _aggressive_free()
    avail = psutil.virtual_memory().available / (1024 ** 3)
    if avail >= need_gb and _has_vram():
        return True

    relax = float(os.getenv("OPTUNA_MIN_FREE_GB_RELAX", "0.5"))
    floor = float(os.getenv("OPTUNA_MIN_FREE_GB_FLOOR", "0.30"))
    relaxed_need = max(floor, need_gb * relax)
    if avail >= relaxed_need and trial is not None and int(getattr(trial, "number", 0)) <= 1 and _has_vram():
        print(f"⚠️ Low RAM: need>={need_gb:.2f}GB, avail={avail:.2f}GB — proceeding (bootstrap).")
        return True

    return False

def optuna_objective(trial, train_data, base_features, evaluate_cv_func, cv_config, models_to_test, vol_stats=None):
    """
    Objective used by Optuna. Adds:
      - Ensures indicator columns exist in the trial DF before feature selection
      - Regime knob merge into ensemble_config for the adaptive ensemble
      - Robust retry: if a worker process dies (TF + multiprocessing), re-run CV single-threaded
      - Store exact features used for this trial into trial.user_attrs["features_used"]
      - If CV yields *no valid blocks* (e.g., abstention / gating), **PRUNE** the trial (no sentinel scores).
    NOTE: Returns TRUE Sharpe (can be negative); no sign inversion anywhere.
    """
    import math, traceback, optuna
    import numpy as _np, pandas as _pd
    from joblib import parallel_backend
    from concurrent.futures.process import BrokenProcessPool

    import os, gc, shutil, psutil
    from datetime import datetime
    from threadpoolctl import threadpool_limits
    
    # Ensure 'direction' exists before any early returns that call _bad_obj(direction)
    direction = str((cv_config or {}).get("optuna_direction", os.getenv("OPTUNA_DIRECTION", "maximize"))).lower().strip()
    
    # --- Soft RAM guard: warn + GC, but do NOT prune on low RAM  ---
    DEFAULT_RAM_FLOOR = os.environ.get("OPTUNA_MIN_FREE_GB", "0.35")
    need = float(DEFAULT_RAM_FLOOR)

    if not _assert_free_ram(need, trial=trial):
        # _assert_free_ram already tried to free memory (TF clear_session + gc)
        avail = psutil.virtual_memory().available / (1024 ** 3)

        # Log, but keep going. We only abort if RAM is *critically* low.
        print(
            f"⚠️ [RAM-SoftGuard] Low RAM before trial {getattr(trial, 'number', '?')} start: "
            f"requested>={need:.2f}GB, avail={avail:.2f}GB — continuing anyway."
        )

        # Optional hard emergency floor to protect your WSL/PC.
        # Very low (0.15 GB) by default; you can adjust via OPTUNA_HARD_FLOOR_GB.
        hard_floor = float(os.environ.get("OPTUNA_HARD_FLOOR_GB", "0.15"))
        if avail < hard_floor:
            if DISABLE_OPTUNA_PRUNING:
                # If pruning is globally disabled, return a sentinel instead of crashing.
                print(
                    f"🔥 [RAM-SoftGuard] avail={avail:.2f}GB < hard floor={hard_floor:.2f}GB — "
                    "returning sentinel -9999.0 to stop this trial safely."
                )
                return _bad_obj(direction)
            else:
                # Abort the whole study to avoid exploding the machine.
                raise RuntimeError(
                    f"Aborting Optuna run due to critically low RAM "
                    f"(avail={avail:.2f}GB < hard_floor={hard_floor:.2f}GB)."
                )



    # 1) Per-trial joblib temp folder (prevents shared temp buildup across trials)
    _run_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    _trial_tmp = os.path.abspath(f"./joblib_tmp/trial_{os.getpid()}_{trial.number}_{_run_ts}")
    os.makedirs(_trial_tmp, exist_ok=True)
    
    # IMPORTANT: set per-trial (setdefault only applies once and can cause cross-trial tmp buildup)
    _prev_joblib_tmp = os.environ.get("JOBLIB_TEMP_FOLDER")
    os.environ["JOBLIB_TEMP_FOLDER"] = _trial_tmp  # used by joblib/loky for memmaps & tmp folders
    
    
    # 2) Hard cap BLAS/OpenMP for this trial (prevents thread oversubscription)
    # Respect BLAS_THREADS_PER_TRIAL; fall back to (cores-2)
    _threads_env = os.getenv("BLAS_THREADS_PER_TRIAL", "")
    _threads = int(_threads_env) if _threads_env.strip() else max(1, (os.cpu_count() or 8) - 2)
    _thread_cm = threadpool_limits(limits=_threads)
    _thread_cm.__enter__()

    def _avail_gb() -> float:
        return psutil.virtual_memory().available / (1024**3)

    # Use the same softer default here as well.
    _MIN_FREE_GB = float(os.environ.get("OPTUNA_MIN_FREE_GB", "0.35"))  # tweak if needed


    # 4) Wrap evaluate_cv_func so it always runs with RAM check + GC
    _orig_evaluate_cv = evaluate_cv_func
    def _wrapped_evaluate_cv(*args, **kwargs):
        _assert_free_ram(_MIN_FREE_GB, trial=trial)
        out = _orig_evaluate_cv(*args, **kwargs)
        gc.collect()
        return out

    evaluate_cv_func = _wrapped_evaluate_cv

    # 5) Ensure we have enough headroom before doing anything expensive
    ok_ram = _assert_free_ram(_MIN_FREE_GB, trial=trial)
    enforce = int(os.getenv("OPTUNA_ENFORCE_RAM_GUARD", "0"))
    if (not ok_ram) and enforce:
        raise optuna.TrialPruned(
            f"[RAM-Guard] insufficient free RAM for trial start "
            f"(need>={float(os.getenv('OPTUNA_MIN_FREE_GB','0.40')):.2f}GB)."
        )


    # ----- Ensure we always clean up no matter how the body exits -----
    def _trial_cleanup():
        # ------------------------------------------------------------
        # Patch 3 (fixed): backtester cleanup (prevents trial-to-trial RAM drift)
        # optuna_objective does not have a `backtester` local; instead infer it
        # from the bound method evaluate_cv_func (backtester.evaluate_cv_func).
        # ------------------------------------------------------------
        bt = None
        try:
            bt = getattr(evaluate_cv_func, "__self__", None)  # bound method -> instance
        except Exception:
            bt = None

        # Fallback: scan closure cells (handles wrapped/lambda cases)
        if bt is None:
            try:
                clos = getattr(evaluate_cv_func, "__closure__", None) or ()
                for cell in clos:
                    obj = getattr(cell, "cell_contents", None)
                    # cheap heuristic: looks like your backtester instance
                    if obj is not None and hasattr(obj, "test_strategy") and hasattr(obj, "features_config"):
                        bt = obj
                        break
            except Exception:
                bt = None

        if bt is not None:
            # Clear engineered feature cache (safe no-op if absent)
            try:
                if hasattr(bt, "_clear_feature_cache"):
                    bt._clear_feature_cache()
            except Exception:
                pass

            # Force TF/Keras cleanup at TRIAL boundary (extra safety)
            try:
                setattr(bt, "_tf_cleanup_do", True)
                setattr(bt, "_tf_cleanup_del_model", True)
                if hasattr(bt, "_maybe_tf_cleanup"):
                    bt._maybe_tf_cleanup()
            except Exception:
                pass

            # Drop CV diagnostics frames (defensive)
            try:
                if hasattr(bt, "_cv_last_eval_df"):
                    bt._cv_last_eval_df = None
                if hasattr(bt, "_cv_fold_eval_frames") and isinstance(bt._cv_fold_eval_frames, list):
                    bt._cv_fold_eval_frames.clear()
            except Exception:
                pass

        # Release BLAS caps context
        try:
            _thread_cm.__exit__(None, None, None)
        except Exception:
            pass
        # Remove the per-trial joblib tmp dir
        try:
            if os.path.isdir(_trial_tmp):
                shutil.rmtree(_trial_tmp, ignore_errors=True)
        except Exception:
            pass
        # Restore previous JOBLIB temp folder (prevents cross-trial tmp buildup)
        try:
            if _prev_joblib_tmp is None:
                os.environ.pop("JOBLIB_TEMP_FOLDER", None)
            else:
                os.environ["JOBLIB_TEMP_FOLDER"] = _prev_joblib_tmp
        except Exception:
            pass
        # One last GC sweep
        gc.collect()

    # --- helper: compute any indicator columns referenced by params toggles
    def _ensure_indicator_columns(df: _pd.DataFrame, p: dict) -> _pd.DataFrame:
        # minimalist, vectorized implementations; assumes price columns exist
        def _ema(x, n): return x.ewm(span=int(n), adjust=False).mean()
        def _sma(x, n): return x.rolling(int(n), min_periods=int(n)).mean()

        # choose a price series
        if "close" in df: 
            price = df["close"]
        else:
            price = df.get("price", df.get("Close", df.select_dtypes("number").iloc[:, 0]))

        # indicator windows container (canonical first, legacy fallback)
        iw = (p.get("indicator_windows") or {})
        def _win(key: str, default: int) -> int:
            # prefer canonical 'key', fallback to legacy 'key_window'
            return int(iw.get(key, iw.get(f"{key}_window", default)))

        # RSI
        if bool(p.get("use_rsi", False)) and "rsi" not in df:
            n = _win("rsi", 14)
            delta = price.diff()
            up = (delta.clip(lower=0)).rolling(n).mean()
            down = (-delta.clip(upper=0)).rolling(n).mean()
            rs = up / (down.replace(0, _np.nan))
            df["rsi"] = (100 - (100 / (1 + rs))).fillna(0.0)

        # MACD (+ signal + hist)
        if bool(p.get("use_macd", False)) and not {"macd","macd_signal","macd_hist"}.issubset(df.columns):
            f = int(iw.get("macd_fast", 12))
            s = int(iw.get("macd_slow", 26))
            m = int(iw.get("macd_signal", 9))
            ema_f = _ema(price, f); ema_s = _ema(price, s)
            macd = ema_f - ema_s
            signal = _ema(macd, m)
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = macd - signal

        # ATR
        if bool(p.get("use_atr", False)) and "atr" not in df:
            n = _win("atr", 14)
            H = df.get("high", price); L = df.get("low", price); C = df.get("close", price)
            tr = _np.maximum(H - L, _np.maximum((H - C.shift()).abs(), (L - C.shift()).abs()))
            df["atr"] = _pd.Series(tr).rolling(n, min_periods=n).mean()

        # Bollinger Bands (already canonical)
        if bool(p.get("use_bbands", False)) and not {"bb_upper","bb_lower"}.issubset(df.columns):
            n = int(iw.get("bb_window", 20)); dev = float(iw.get("bb_dev", 2.0))
            ma = _sma(price, n); sd = price.rolling(n, min_periods=n).std(ddof=0)
            df["bb_upper"] = ma + dev * sd
            df["bb_lower"] = ma - dev * sd

        # Stochastic Oscillator
        if bool(p.get("use_stoch", False)) and not {"stoch_k","stoch_d"}.issubset(df.columns):
            k = _win("stoch_k", 14); d = _win("stoch_d", 3)
            Hh = df.get("high", price).rolling(k).max()
            Ll = df.get("low", price).rolling(k).min()
            stoch_k = 100.0 * (price - Ll) / (Hh - Ll)
            df["stoch_k"] = stoch_k.replace([_np.inf, -_np.inf], _np.nan)
            df["stoch_d"] = df["stoch_k"].rolling(d).mean()

        # EMA/SMA columns (optional)
        if bool(p.get("use_ema", False)) and "ema" not in df:
            df["ema"] = _ema(price, _win("ema", 20))
        if bool(p.get("use_sma", False)) and "sma" not in df:
            df["sma"] = _sma(price, _win("sma", 20))

        # ADX (optional; needs H/L/C)
        if bool(p.get("use_adx", False)) and "adx" not in df and {"high","low","close"}.issubset(df.columns):
            n = _win("adx", 14)
            upMove = df["high"].diff()
            downMove = -df["low"].diff()
            plusDM  = _np.where((upMove > downMove) & (upMove > 0), upMove, 0.0)
            minusDM = _np.where((downMove > upMove) & (downMove > 0), downMove, 0.0)
            tr = _np.maximum(df["high"] - df["low"],
                _np.maximum((df["high"] - df["close"].shift()).abs(),
                            (df["low"]  - df["close"].shift()).abs()))
            atr = _pd.Series(tr).rolling(n, min_periods=n).mean()
            plusDI  = 100 * _pd.Series(plusDM).rolling(n, min_periods=n).mean() / atr
            minusDI = 100 * _pd.Series(minusDM).rolling(n, min_periods=n).mean() / atr
            dx = ((plusDI - minusDI).abs() / (plusDI + minusDI).replace(0, _np.nan)) * 100
            df["adx"] = dx.rolling(n, min_periods=n).mean().fillna(0.0)

        # ---- Momentum extensions (use canonical windows) ----
        sma_n = _win("sma", 20)
        ema_n = _win("ema", 20)
        _ema_local = price.ewm(span=ema_n, adjust=False).mean()
        _sma_local = price.rolling(sma_n, min_periods=sma_n).mean()

        # 1) EMA–SMA spread
        if bool(p.get("use_ma_spread", False)) and "ema_sma_spread" not in df:
            df["ema_sma_spread"] = (_ema_local - _sma_local)

        # 2) Price–MA z-scores
        if bool(p.get("use_price_ma_z", False)):
            if f"price_sma_z_{sma_n}" not in df:
                sd_sma = price.rolling(sma_n, min_periods=max(5, sma_n//3)).std(ddof=0)
                df[f"price_sma_z_{sma_n}"] = (price - _sma_local) / (sd_sma.replace(0.0, _np.nan))
            if f"price_ema_z_{ema_n}" not in df:
                sd_ema = price.rolling(ema_n, min_periods=max(5, ema_n//3)).std(ddof=0)
                df[f"price_ema_z_{ema_n}"] = (price - _ema_local) / (sd_ema.replace(0.0, _np.nan))

        # 3) Crossover binaries
        if bool(p.get("use_crossover_bins", False)):
            if "price_gt_sma" not in df:
                df["price_gt_sma"] = (price > _sma_local).astype(int)
            if "price_gt_ema" not in df:
                df["price_gt_ema"] = (price > _ema_local).astype(int)
            _trend_proxy = df["macd"] if "macd" in df else (_ema_local - _sma_local)
            if "ma_cross_up" not in df:
                df["ma_cross_up"] = (_trend_proxy > 0).astype(int)
            if "ma_cross_dn" not in df:
                df["ma_cross_dn"] = (_trend_proxy < 0).astype(int)

        # 4) Short–long slope differential (slope of spread)
        if bool(p.get("use_slope_diff", False)):
            w = max(5, min(ema_n, sma_n) // 2)
            _x = df["macd"] if "macd" in df else (_ema_local - _sma_local)
            def _roll_slope(x):
                t = _np.arange(len(x), dtype=float)
                tm = t.mean(); xm = _np.mean(x)
                denom = ((t - tm)**2).sum()
                if denom == 0: return 0.0
                return float(((t - tm)*(x - xm)).sum() / denom)
            coln = f"ma_spread_slope{w}"
            if coln not in df:
                df[coln] = _pd.Series(_x).rolling(w, min_periods=w).apply(_roll_slope, raw=True)


                # ---- Regime features (trend_score, vol_score, regime_id/one-hot) ----
        # Guarded by 'use_regime_features' so we can disable this block in configs if needed.
        if bool(p.get("use_regime_features", True)):
            try:
                trend_components = []

                # 1) Trend components: price–MA z-scores, ADX, EMA–SMA spread (if available)
                sma_z_col = f"price_sma_z_{sma_n}"
                if sma_z_col in df:
                    trend_components.append(df[sma_z_col].abs())
                elif "sma" in df:
                    # Fallback: on-the-fly normalized distance of price to SMA
                    sd_sma = price.rolling(sma_n, min_periods=max(5, sma_n//3)).std(ddof=0)
                    trend_components.append(((price - df["sma"]) / sd_sma.replace(0.0, _np.nan)).abs())

                ema_z_col = f"price_ema_z_{ema_n}"
                if ema_z_col in df:
                    trend_components.append(df[ema_z_col].abs())

                if "adx" in df:
                    # Scale ADX to roughly [0, 1.5] by a robust upper quantile
                    adx = df["adx"].astype(float)
                    try:
                        adx_q = float(_np.nanquantile(adx.values, 0.9))
                    except Exception:
                        adx_q = 25.0
                    adx_q = adx_q if adx_q > 0 else 25.0
                    trend_components.append(adx / adx_q)

                if "ema_sma_spread" in df:
                    spread = df["ema_sma_spread"].astype(float)
                    spread_sd = spread.rolling(sma_n, min_periods=max(5, sma_n//3)).std(ddof=0)
                    trend_components.append((spread / spread_sd.replace(0.0, _np.nan)).abs())

                if trend_components:
                    # Sum components; fill NaNs/inf with zero to keep scale sane
                    trend_score = trend_components[0].copy()
                    for comp in trend_components[1:]:
                        trend_score = trend_score.add(comp, fill_value=0.0)
                    trend_score = trend_score.replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
                    df["trend_score"] = trend_score

                # 2) Volatility components: ATR and Bollinger band width (if available)
                vol_components = []
                if "atr" in df:
                    vol_components.append(df["atr"].astype(float))
                if {"bb_upper", "bb_lower"}.issubset(df.columns):
                    bb_width = (df["bb_upper"] - df["bb_lower"]).abs().astype(float)
                    vol_components.append(bb_width)

                if vol_components:
                    vol_raw = vol_components[0].copy()
                    for comp in vol_components[1:]:
                        vol_raw = vol_raw.add(comp, fill_value=0.0)

                    # Normalize by a rolling median to keep the score dimensionless
                    vol_win = int(p.get("regime_vol_window", sma_n))
                    vol_roll = vol_raw.rolling(vol_win, min_periods=max(5, vol_win//3))
                    vol_med = vol_roll.median().replace(0.0, _np.nan)
                    vol_score = (vol_raw / vol_med).replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
                    df["vol_score"] = vol_score

                # 3) Convert scores to a discrete 3-state regime label
                if "trend_score" in df and "vol_score" in df:
                    ts = df["trend_score"].astype(float).values
                    vs = df["vol_score"].astype(float).values

                    # Robust quantiles with safe defaults
                    try:
                        q_trend = float(_np.nanquantile(ts, float(p.get("regime_trend_quantile", 0.7))))
                    except Exception:
                        q_trend = _np.nanmax(ts) if _np.isfinite(_np.nanmax(ts)) else 0.0
                    try:
                        q_vol_hi = float(_np.nanquantile(vs, float(p.get("regime_vol_high_quantile", 0.7))))
                    except Exception:
                        q_vol_hi = _np.nanmax(vs) if _np.isfinite(_np.nanmax(vs)) else 1.0
                    try:
                        q_vol_lo = float(_np.nanquantile(vs, float(p.get("regime_vol_low_quantile", 0.4))))
                    except Exception:
                        q_vol_lo = _np.nanmedian(vs) if _np.isfinite(_np.nanmedian(vs)) else 1.0

                    # Default: SIDEWAYS (0)
                    regime_id = _np.zeros(len(df), dtype="int8")

                    # TREND (1): strong trend score
                    mask_trend = ts >= q_trend
                    regime_id[mask_trend] = 1

                    # VOLATILE/CHOPPY (2): high vol, not in trend bucket
                    mask_volatile = (~mask_trend) & (vs >= q_vol_hi)
                    regime_id[mask_volatile] = 2

                    # SIDEWAYS (0): everything else
                    df["regime_id"] = regime_id
                    df["regime_sideways"] = (regime_id == 0).astype("int8")
                    df["regime_trend"]    = (regime_id == 1).astype("int8")
                    df["regime_volatile"] = (regime_id == 2).astype("int8")
            except Exception as _e:
                # Fail-safe: never break tuning because of regime feature construction
                print(f"⚠️ Regime feature construction failed: {_e}")

        return df


    try:
        # Sample params (two-stage aware)
        _stage_cfg = {
            "hpo_stage": (cv_config or {}).get("hpo_stage", "single"),
            "frozen_signal_params": (cv_config or {}).get("frozen_signal_params", None),
        }

        params = sample_param_set(
            trial,
            models_to_test,
            train_data=train_data,
            vol_stats=vol_stats,
            stage_config=_stage_cfg,
        )

        try:
            trial.set_user_attr("hpo_stage", str(_stage_cfg.get("hpo_stage", "single")))
        except Exception:
            pass

        # Ensure ensemble_config carries regime knobs for the adaptive ensemble
        if params.get("model_type") == "ensemble_adaptive_regime":
            ec = dict(params.get("ensemble_config", {}))
            for k in (
                "adx_col", "vol_col", "adx_thresh", "vol_thresh",
                "train_lstm_on_trend_only", "ensemble_train_stride", "ensemble_deep_max_train_windows"
            ):
                if k in params:
                    ec[k] = params[k]
            params["ensemble_config"] = ec
            
            
        # ------------------------------------------------------------
        # Stronger churn control for classical ML (use existing CV metrics)
        # ------------------------------------------------------------
        _mt = str(params.get("model_type", "")).lower()
        if _mt in {"logistic", "svm", "decision_tree", "random_forest", "xgboost"}:
            # Copy to avoid mutating a shared dict across trials
            cv_config = dict(cv_config or {})

            # Tighter turnover band for classical models
            cv_config.setdefault("cv_turnover_low", 0.01)     # classical default band was 0.03
            cv_config.setdefault("cv_turnover_high", 0.10)    # classical default band was 0.18

            # Stronger turnover penalty slope
            cv_config.setdefault("turnover_penalty_lambda", 0.25)  # default was 0.10

            # Make high-turnover violations expensive; don't punish low turnover here
            cv_config.setdefault("cv_turnover_low_lambda", 0.0)
            cv_config.setdefault("cv_turnover_high_lambda", 6.0)


        # --- Ensure the DF actually contains the indicators toggled in params ---
        # Use a shallow copy so we:
        #   - share base OHLC/returns blocks (no huge deep copy),
        #   - but keep per-trial indicator columns isolated.
        trial_data = _ensure_indicator_columns(train_data.copy(deep=False), params)

        # --- Expand base_features safely so it's not overly restrictive ---
        non_target_cols = [
            c for c in trial_data.columns
            if c.lower() not in ("target", "label", "y", "side", "_leak", "_future_return")
        ]
        if base_features is None or len(base_features) == 0:
            base_features = non_target_cols
        else:
            # union to avoid accidentally excluding freshly built indicators
            bf_set = set(base_features)
            base_features = list(bf_set.union(non_target_cols))

        # Record features used for this trial (optional/best-effort).
        # Guarded by SAVE_TRIAL_FEATURE_FREQ so that we skip this completely
        # in normal runs (it is only needed when plotting trial-level feature
        # frequency heatmaps after tuning).
        if SAVE_TRIAL_FEATURE_FREQ:
            try:
                from utilsNoWFO import build_features_from_params as _build_feats
                try:
                    features = _build_feats(trial_data, params, base_features)
                except Exception:
                    features = []
                try:
                    trial.set_user_attr("features_used", list(features))
                except Exception:
                    pass
            except Exception:
                pass
            
        # Persist effective params for two-stage freezing (after any param edits)
        try:
            import json as _json
            import numpy as _np

            def _json_sanitize(o):
                if isinstance(o, (_np.integer, _np.floating)):
                    return o.item()
                if isinstance(o, (str, int, float, bool)) or o is None:
                    return o
                if isinstance(o, (list, tuple)):
                    return [_json_sanitize(x) for x in o]
                if isinstance(o, dict):
                    return {str(k): _json_sanitize(v) for k, v in o.items()}
                return str(o)

            trial.set_user_attr("_full_params_json", _json.dumps(_json_sanitize(params)))
            trial.set_user_attr("full_params", _json_sanitize(params))
        except Exception:
            pass

        # --- Define the evaluation runner (must be defined before dispatch) ---
        # Guard #2: avoid KeyError on missing windows by using safe defaults
        # (defaults match your MiniBlockCV logs: ~28,032 train rows, 1,475 val rows)
        def _run_eval():
            min_tw = int(cv_config.get("min_train_window", 28032))
            val_w  = int(cv_config.get("val_window", 1475))
            return evaluate_cv_func(
                trial_data,
                params,
                min_train_window=min_tw,
                val_window=val_w,
                trial=trial,
                cv_config_override=cv_config,
            )


        # --- Dispatch: classical models → thread backend for speed & stability ---
        def _dispatch_eval():
            _mt = str(params.get("model_type", "")).lower()
            _cv_jobs = int((cv_config or {}).get("cv_n_jobs", os.cpu_count() or 1))
            
            # ------------------------------------------------------------------
            # Crash guard (segfault/core-dump): avoid nested/native parallelism.
            # RandomForest can spawn heavy native parallel work (joblib/OpenMP).
            # Running it inside a threaded joblib backend + BLAS/OpenMP threads
            # is a common cause of hard crashes. Force RF to single-threaded
            # execution during CV/Optuna evaluation.
            # ------------------------------------------------------------------
            if _mt == "random_forest":
                try:
                    # Optuna may propose rf_n_jobs=-1; override to keep RF stable.
                    params["rf_n_jobs"] = 1
                except Exception:
                    pass
                _cv_jobs = 1
                return _run_eval()

            # Other classical models: keep lightweight threading backend.
            if _mt in {"logistic", "svm", "xgboost"}:
                with parallel_backend("threading", n_jobs=max(1, _cv_jobs)):
                    return _run_eval()

            return _run_eval()
             
            

        # Allow controlling CV debug and table verbosity via env:
        #   CV_DEBUG=1        → enable detailed CV debug logs
        #   CV_TABLE_MODE=off → disable per-block CV tables
        _cv_debug = os.getenv("CV_DEBUG", "0") == "1"
        _cv_table_mode_env = os.getenv("CV_TABLE_MODE", "").strip().lower()
        if _cv_table_mode_env:
            cv_config["cv_table_mode"] = _cv_table_mode_env

        # Patch 6A: Respect caller/class defaults.
        # Env vars may FORCE overrides; otherwise only fill missing keys.
        if _cv_debug:
            cv_config["print_cv_debug"] = True
        else:
            cv_config.setdefault("print_cv_debug", False)

        cv_config.setdefault("cv_agg_mode", "tanh_mean")
        cv_config.setdefault("cv_trim_frac", 0.20)
        cv_config.setdefault("cv_use_recency_weight", False)
        cv_config.setdefault("cv_cscv_penalty_weight", 0.75)

        # Ensure the MLBacktester instance sees these CV knobs
        # (apply once before any CV attempt and its fallbacks)
        # Guard #1: warn if not a bound method; log what we applied when debug is on
        try:
            bound = getattr(evaluate_cv_func, "__self__", None)
            if bound is None:
                log_print(
                    "[CV CONFIG] INFO: evaluate_cv_func not bound; using explicit cv_config_override.",
                    level="DEBUG",
                )
            else:
                bound.config = dict(cv_config)
                if cv_config.get("print_cv_debug"):
                    log_print(
                        "[CV CONFIG] Applied to MLBacktester: "
                        f"agg={cv_config.get('cv_agg_mode')} | "
                        f"trim={cv_config.get('cv_trim_frac')} | "
                        f"recency={cv_config.get('cv_use_recency_weight')} | "
                        f"cscv_w={cv_config.get('cv_cscv_penalty_weight')}",
                        level="DEBUG",
                    )
        except Exception as e:
            log_print(f"[CV CONFIG] Injection failed: {e}", level="DEBUG")


        # CV_JOBS pulled EXACTLY like in your logging / threading code.
        cv_jobs = int(os.getenv("CV_JOBS", "1"))

        try:
            # Decide serial vs parallel using .env CV_JOBS
            if cv_jobs == 1:
                mean_score = _run_eval()
            else:
                mean_score = _dispatch_eval()
                
        except BrokenProcessPool as e:
            # Do NOT retry this trial: mark as failed and move on.
            cause = f"BrokenProcessPool: {e}"
            log_print(
                f"⚠️ {cause} — marking trial {trial.number} as failed (no retry).",
                level="COMPACT",
            )
            try:
                trial.set_user_attr("cv_failed", cause)
            except Exception:
                pass
            mean_score = _bad_obj(direction)


        except RuntimeError as e:
            msg = str(e)
            if "unexpectedly terminated" in msg or "Executor" in msg:
                # Again: do NOT retry, just flag and give a terrible score.
                cause = f"RuntimeError: {msg}"
                log_print(
                    f"⚠️ {cause} — marking trial {trial.number} as failed (no retry).",
                    level="COMPACT",
                )
                try:
                    trial.set_user_attr("cv_failed", msg[:200])
                except Exception:
                    pass
                mean_score = _bad_obj(direction)

            else:
                # Other RuntimeErrors are still unexpected → bubble up.
                raise

        # Normalize CV result → one float; prune if not finite (no-valid-blocks / gating)
        try:
            score = float(mean_score)
        except Exception:
            try:
                trial.set_user_attr("no_trades", True)
            except Exception:
                pass
            if DISABLE_OPTUNA_PRUNING:
                return _bad_obj(direction)
            else:
                raise optuna.TrialPruned("MiniBlockCV: CV score not numeric")

        if not _np.isfinite(score):
            try:
                trial.set_user_attr("no_trades", True)
            except Exception:
                pass
            if DISABLE_OPTUNA_PRUNING:
                return _bad_obj(direction)
            else:
                raise optuna.TrialPruned("MiniBlockCV: non-finite CV score (no-trades/gating)")


        # ------------------------------------------------------------------
        # Rationale:
        #   - Do NOT prune on macro precision (3-class over all bars).
        #   - Prune only on intent precision (quality when the model actually acts),
        #     and only when we have enough intent samples (intent_bars_cv >= Nmin).
        #
        # Requires CV to attach:
        #   trial.user_attrs["precision_intent_cv"]
        #   trial.user_attrs["intent_bars_cv"]
        #
        # Config (all optional; OFF by default):
        #   cv_prune_precision_intent: bool = False
        #   cv_prune_min_intent_bars: int = 100
        #   cv_prune_min_precision_intent: float = 0.38
        # ------------------------------------------------------------------
        try:
            _do_prune = bool((cv_config or {}).get("cv_prune_precision_intent", False))
        except Exception:
            _do_prune = False

        if _do_prune and (trial is not None):
            try:
                _nmin = int((cv_config or {}).get("cv_prune_min_intent_bars", 100))
            except Exception:
                _nmin = 100
            try:
                _pthr = float((cv_config or {}).get("cv_prune_min_precision_intent", 0.38))
            except Exception:
                _pthr = 0.38

            # Read CV-attached attrs (best-effort; missing attrs -> no prune).
            try:
                _p = trial.user_attrs.get("precision_intent_cv", None)
                _n = trial.user_attrs.get("intent_bars_cv", None)
            except Exception:
                _p, _n = None, None

            try:
                _p = float(_p) if _p is not None else float("nan")
            except Exception:
                _p = float("nan")
            try:
                _n = int(_n) if _n is not None else 0
            except Exception:
                _n = 0

            # Only enforce when sample size is meaningful.
            if (_n >= int(_nmin)) and (_np.isfinite(_p)) and (float(_p) < float(_pthr)):
                msg = f"MiniBlockCV: intent-precision gate failed (p={_p:.3f} < {_pthr:.3f}, n={_n} >= {_nmin})"
                try:
                    trial.set_user_attr("precision_intent_pruned", True)
                    trial.set_user_attr("precision_intent_cv", float(_p))
                    trial.set_user_attr("intent_bars_cv", int(_n))
                    trial.set_user_attr("precision_intent_prune_msg", msg)
                except Exception:
                    pass

                if DISABLE_OPTUNA_PRUNING:
                    return _bad_obj(direction)
                raise optuna.TrialPruned(msg)


        # ------------------------------------------------------------------
        # Penalize the base CV score by NLL with a small configurable lambda:
        #   final_score = base_score - lambda * nll
        # The NLL is attached by CV into trial.user_attrs["nll"].
        # ------------------------------------------------------------------
        try:
            _lam = (cv_config or {}).get("calib_nll_lambda", None)
            if _lam is None or (isinstance(_lam, str) and not _lam.strip()):
                _lam = os.environ.get("CALIB_NLL_LAMBDA", "0.05")
            _lam = float(_lam)

            _nll = None
            _brier = None
            if trial is not None:
                try:
                    _nll = trial.user_attrs.get("nll", None)
                    _brier = trial.user_attrs.get("brier", None)
                except Exception:
                    _nll = None
                    _brier = None

            _base = float(score)
            _final = _base
            _apply = False
            try:
                if _lam > 0.0 and _nll is not None and _np.isfinite(float(_nll)):
                    _final = _base - (_lam * float(_nll))
                    _apply = True
            except Exception:
                _apply = False

            if trial is not None:
                try:
                    trial.set_user_attr("base_score", float(_base))
                    trial.set_user_attr("final_score", float(_final))
                    trial.set_user_attr("calib_nll_lambda", float(_lam))
                except Exception:
                    pass

            if _apply:
                log_print(
                    f"[Select] base_score={_base:.6f} nll={float(_nll):.6f} "
                    f"lambda={_lam:.6f} final_score={_final:.6f}",
                    level="COMPACT",
                )
            else:
                # Keep noise low unless debug is enabled, but still auditable via user_attrs.
                if bool((cv_config or {}).get("print_cv_debug", False)) or os.getenv("HPO_SELECT_DEBUG", "0") == "1":
                    _nll_s = "NA" if _nll is None else f"{float(_nll):.6f}"
                    _brier_s = "NA" if _brier is None else f"{float(_brier):.6f}"
                    log_print(
                        f"[Select] base_score={_base:.6f} nll={_nll_s} brier={_brier_s} "
                        f"lambda={_lam:.6f} final_score={_final:.6f} (no_penalty)",
                        level="DEBUG",
                    )

            # Optuna ranking uses the returned value.
            score = float(_final)
        except Exception as _sel_e:
            if bool((cv_config or {}).get("print_cv_debug", False)) or os.getenv("HPO_SELECT_DEBUG", "0") == "1":
                log_print(f"⚠️ [Select] calibration penalty skipped: {_sel_e}", level="DEBUG")
                
        # ------------------------------------------------------------------
        # Store a decomposition in trial.user_attrs for audit:
        #   base_score, penalty_total, final_score + key CV stats if present.
        # Enforce direction invariants:
        #   - maximize: penalties must NOT increase score
        #   - minimize: penalties must NOT decrease score
        # If violated -> prune (or return -9999 if pruning is disabled).
        # ------------------------------------------------------------------
        try:
            direction = str((cv_config or {}).get("optuna_direction", os.getenv("OPTUNA_DIRECTION", "maximize"))).lower().strip()
            is_max = (direction != "minimize")

            # Prefer the base_score written above (pre-penalty). Fall back to current score.
            try:
                base_score = float(trial.user_attrs.get("base_score", score)) if trial is not None else float(score)
            except Exception:
                base_score = float(score)

            final_score = float(score)

            # Penalty magnitude in the intended direction (always >= 0 if penalties reduce score)
            penalty_total = (base_score - final_score) if is_max else (final_score - base_score)

            # Individual penalty components (best-effort)
            nll = None
            brier = None
            lam = None
            try:
                if trial is not None:
                    nll = trial.user_attrs.get("nll", None)
                    brier = trial.user_attrs.get("brier", None)
                    lam = trial.user_attrs.get("calib_nll_lambda", None)
            except Exception:
                nll = None
                brier = None
                lam = None

            penalty_nll = 0.0
            try:
                if lam is not None and nll is not None and _np.isfinite(float(lam)) and _np.isfinite(float(nll)):
                    penalty_nll = float(lam) * float(nll)
            except Exception:
                penalty_nll = 0.0

            # Key CV stats if the CV layer attached them already (non-fatal if missing)
            cv_k_valid = None
            cv_k_total = None
            cv_coverage = None
            cv_trades_valid = None
            try:
                if trial is not None:
                    cv_k_valid = trial.user_attrs.get("cv_k_valid", trial.user_attrs.get("k_valid", None))
                    cv_k_total = trial.user_attrs.get("cv_k_total", trial.user_attrs.get("k_total", None))
                    cv_coverage = trial.user_attrs.get("cv_coverage", trial.user_attrs.get("coverage", None))
                    cv_trades_valid = trial.user_attrs.get("cv_trades_valid", trial.user_attrs.get("trades_valid", None))
            except Exception:
                pass

            if trial is not None:
                try:
                    trial.set_user_attr("optuna_direction", direction)
                    trial.set_user_attr("penalty_total", float(penalty_total))
                    trial.set_user_attr("penalty_nll", float(penalty_nll))
                    # Keep these explicitly, even if already written, so the audit snapshot is consistent.
                    trial.set_user_attr("base_score", float(base_score))
                    trial.set_user_attr("final_score", float(final_score))
                except Exception:
                    pass

            # Always log a compact decomposition line (audit-grade, low noise).
            _k_s = "NA"
            try:
                if cv_k_valid is not None and cv_k_total is not None:
                    _k_s = f"{int(cv_k_valid)}/{int(cv_k_total)}"
            except Exception:
                _k_s = "NA"

            _cov_s = "NA"
            try:
                if cv_coverage is not None and _np.isfinite(float(cv_coverage)):
                    _cov_s = f"{float(cv_coverage):.3f}"
            except Exception:
                _cov_s = "NA"

            _tr_s = "NA"
            try:
                if cv_trades_valid is not None:
                    _tr_s = str(cv_trades_valid)
            except Exception:
                _tr_s = "NA"

            log_print(
                f"[ObjectiveGuard] dir={direction} base={float(base_score):.6f} "
                f"pen_total={float(penalty_total):.6f} pen_nll={float(penalty_nll):.6f} "
                f"final={float(final_score):.6f} k={_k_s} cov={_cov_s} trades={_tr_s}",
                level="COMPACT",
            )

            # Assert the direction invariant.
            if not _np.isfinite(float(penalty_total)):
                msg = f"ObjectiveGuard: non-finite penalty_total (dir={direction})"
                if trial is not None:
                    try:
                        trial.set_user_attr("objective_guard_violation", msg)
                    except Exception:
                        pass
                if DISABLE_OPTUNA_PRUNING:
                    return _bad_obj(direction)
                raise optuna.TrialPruned(msg)

            if float(penalty_total) < -1e-9:
                msg = (
                    f"ObjectiveGuard: penalty increased score under dir={direction} "
                    f"(base={base_score:.6f} final={final_score:.6f} pen_total={penalty_total:.6f})"
                )
                log_print(f"🚨 [{msg}]", level="COMPACT")
                if trial is not None:
                    try:
                        trial.set_user_attr("objective_guard_violation", msg)
                    except Exception:
                        pass
                if DISABLE_OPTUNA_PRUNING:
                    return _bad_obj(direction)
                raise optuna.TrialPruned(msg)

        except optuna.TrialPruned:
            raise
        except Exception as _og_e:
            # Guard must never crash the whole run; if something weird happens, just record it.
            if trial is not None:
                try:
                    trial.set_user_attr("objective_guard_error", str(_og_e)[:200])
                except Exception:
                    pass


    except optuna.TrialPruned as e:
        # Let Optuna-level prunes bubble up cleanly; this is expected control flow.
        log_print(f"Trial {trial.number} pruned: {e}", level="DEBUG")
        raise

    except Exception as e:
        # Real unexpected error: log and, unless disabled, prune so the study can continue.
        cause = f"{type(e).__name__}: {str(e)}"

        log_print("\n" + "#" * 80, level="COMPACT")
        log_print(f"⚠️ Error in optuna_objective(): {cause}", level="COMPACT")
        if 'params' in locals():
            log_print(f"Trial params were: {params}", level="COMPACT")

        # Keep the stack trace always visible; it's rare but important.
        traceback.print_exc()
        log_print("#" * 80 + "\n", level="COMPACT")

        if DISABLE_OPTUNA_PRUNING:
            return _bad_obj(direction)
        else:
            raise optuna.TrialPruned(f"Trial pruned due to error → {cause}")

    # Always run cleanup (success, prune, or error)
    try:
        pass
    finally:

        # 1) Per-trial cleanup (thread caps, JOBLIB temp, tmp dir)        
        _trial_cleanup()
        # 2) Aggressive free (TF clear_session + GC + tiny sleep)
        #    Helps reduce RAM high-water across long Optuna runs.
        try:
            _aggressive_free()
        except Exception:
            gc.collect()

    return score  # TRUE Sharpe (maximize)




def run_optuna_tuning(
        train_data, base_features, evaluate_cv_func, cv_config, models_to_test,
        n_trials=1, n_startup_trials=10, return_top_n=3, study=None, sampler_seed=None,
        month_out_dir: str | None = None, month_ix: int | None = None):

    """
    Runs Optuna tuning for a given model configuration, evaluates via CV,
    and returns the best parameter set along with embedded Top-N trial info.

    Returns:
        best_params (dict): Best trial parameters with extra Top-N metadata.
        best_score (float): study.best_value (TRUE Sharpe).
        topN_params (list[dict]): List of Top-N param dicts (ranked best→worse).
        study (optuna.study.Study): the Optuna study.
        consensus_pool (list[dict]): Small pool of candidate configs (all valid trials) for consensus selection.
     """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import os, json, datetime

    # Reset per-run hyperparameter boundary diagnostics
    global HP_BOUNDARY_HITS, HP_BOUNDARY_HITS_MIN, HP_BOUNDARY_HITS_MAX, HP_BOUNDARY_RANGES
    HP_BOUNDARY_HITS = {}
    HP_BOUNDARY_HITS_MIN = {}
    HP_BOUNDARY_HITS_MAX = {}
    HP_BOUNDARY_RANGES = {}

    # Lazy imports (avoid circulars)
    try:
        from utilsNoWFO import save_optuna_progress_from_study
    except Exception:
        save_optuna_progress_from_study = None

    try:
        from utilsNoWFO import save_feature_frequency_from_trials
    except Exception:
        save_feature_frequency_from_trials = None
        
    try:
        from utilsNoWFO import save_optuna_learning_summary
    except Exception:
        save_optuna_learning_summary = None
        
    try:
        from utilsNoWFO import save_hpo_config_to_disk, get_hpo_config_dir
    except Exception:
        save_hpo_config_to_disk = None
        get_hpo_config_dir = None
        
    # Work on a single consolidated copy of the DF across all trials
    train_data = train_data.copy()
    try:
        # Defragment once so Pandas storage is compact
        train_data._consolidate_inplace()
    except Exception:
        pass

    # Defensive
    models_to_test = sorted(list(models_to_test))
    
    
    # ------------------------------------------------------------
    # Model-family detection (shared by sampler + pruner)
    # ------------------------------------------------------------
    try:
        _model_name = models_to_test[0] if isinstance(models_to_test, (list, tuple)) else models_to_test
    except Exception:
        _model_name = models_to_test
    _model_name = str(_model_name).lower()
    _is_deep_family = (
        _model_name in {"cnn", "lstm", "transformer", "dqn"} or _model_name.startswith("ensemble_")
    )
    
    # Ensure CV parallelism is wired even if caller omitted it
    cv_config = dict(cv_config or {})
    if "cv_n_jobs" not in cv_config:
        import os
        import multiprocessing
        cv_config["cv_n_jobs"] = int(os.getenv("CV_JOBS", str(multiprocessing.cpu_count() or 16)))
        
    # ------------------------------------------------------------
    # Fast path: n_trials <= 0 => load cached HPO config (no Optuna)
    # ------------------------------------------------------------
    try:
        _ntr = int(n_trials or 0)
    except Exception:
        _ntr = 0

    if _ntr <= 0:
        import os, json, math

        base = os.environ.get(
            "MLB_HPO_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpo"),
        )

        safe = str(_model_name).replace("/", "_")
        candidates = [
            os.path.join(base, f"model_{_model_name}_hpo.json"),   # MLBacktester schema
            os.path.join(base, f"{safe}_best_config.json"),        # utilsNoWFO schema
            os.path.join(base, f"{_model_name}_best_config.json"), # legacy
        ]

        data = None
        used_path = None
        for p in candidates:
            if p and os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                    used_path = p
                    break
                except Exception:
                    data = None
                    used_path = None

        best = None
        topN = []
        if isinstance(data, dict):
            best = data.get("best_params") or data.get("best") or data
            topN = data.get("topN_params") or data.get("topN") or []

        if not isinstance(best, dict) or not best:
            raise RuntimeError(
                f"n_trials=0 but no cached HPO config found for '{_model_name}'. "
                f"Looked in: {candidates}. Set MLB_HPO_DIR to override."
            )

        # Preserve a score if present (used only for logging)
        try:
            _score = float(best.get('__cv_value', best.get('cv_value', best.get('value', float('nan')))))
        except Exception:
            _score = float('nan')

        # Strip internal metadata keys that can leak into model constructors
        best = {k: v for k, v in best.items() if not str(k).startswith("__")}
        best.setdefault("model_type", _model_name)

        # Minimal dummy study so downstream logging doesn't crash
        class _DummyTrial:
            number = -1

        class _DummyStudy:
            best_trial = _DummyTrial()
            best_value = _score
            trials = []

        study = _DummyStudy()
        print(f"[HPO] n_trials=0 → loaded cached config for {_model_name} from {used_path}")
        return best, study.best_value, list(topN or []), study, list(topN or [])

        
    
    # Consensus pool knobs (used later for Top-N consensus selection)
    consensus_pool_max_trials = int(cv_config.get("consensus_pool_max_trials", 0))
    consensus_pool_min_perf_frac = float(
        cv_config.get("consensus_pool_min_perf_frac", cv_config.get("topN_min_perf_frac", 0.60))
    )
    
    # --- Precompute volatility stats once for label_threshold scaling ---
    vol_stats: dict = {}
    if (
        train_data is not None
        and hasattr(train_data, "columns")
        and "returns" in train_data.columns
    ):
        import numpy as _np
        r = train_data["returns"].astype("float64").dropna()
        if r.size > 0:
            sigma = float(r.rolling(48).std().median())
            sigma = float(_np.clip(sigma, 1e-5, 5e-3))
            vol_stats["sigma48"] = sigma

    # ------------------------------------------------------------
    # Patch: model-aware startup trials
    # If caller passes default 10 → treat as "auto".
    # If caller passes anything else → respect it.
    # ------------------------------------------------------------
    try:
        _n_startup_arg = int(n_startup_trials)
    except Exception:
        _n_startup_arg = 10

    if _n_startup_arg == 10:
        if _is_deep_family:
            _n_startup = int(cv_config.get("n_startup_trials_deep", 25))
        else:
            _n_startup = int(cv_config.get("n_startup_trials_classical", 15))
    else:
        _n_startup = _n_startup_arg

    tpe_ei = int(os.environ.get("TPE_EI_CANDIDATES", "64"))
    sampler = TPESampler(
        n_startup_trials=_n_startup,
        multivariate=True,
        group=True,
        seed=sampler_seed,
        n_ei_candidates=tpe_ei,
    )
    print(f"[Optuna] TPESampler(multivariate=True, group=True, n_startup_trials={_n_startup}, "
          f"n_ei_candidates={tpe_ei}, seed={sampler_seed}) model={_model_name}")
    
    from optuna.pruners import SuccessiveHalvingPruner, NopPruner

    if _is_deep_family:
        _pruner_min_resource = int(cv_config.get("pruner_min_resource_deep", 4))
        _pruner_reduction_factor = int(cv_config.get("pruner_reduction_factor_deep", 3))
    else:
        _pruner_min_resource = int(cv_config.get("pruner_min_resource_classical", 2))
        _pruner_reduction_factor = int(cv_config.get("pruner_reduction_factor_classical", 2))

    # ASHA-style pruning (default), or no pruning if disabled
    if DISABLE_OPTUNA_PRUNING:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=_pruner_min_resource,       # “min_folds before prune”
            reduction_factor=_pruner_reduction_factor,
            bootstrap_count=0,
            min_early_stopping_rate=0                # allow pruning once min_resource is reached
        )
        print(f"[Optuna] SuccessiveHalvingPruner(min_resource={_pruner_min_resource}, "
              f"reduction_factor={_pruner_reduction_factor}) model={_model_name}")



    if study is None:

        sampler = TPESampler(
            seed=sampler_seed,
            n_startup_trials=_n_startup,
            multivariate=True,
            group=True,
            n_ei_candidates=tpe_ei,
        )

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    func = lambda trial: optuna_objective(
        trial,
        train_data,
        base_features,
        evaluate_cv_func,
        cv_config,
        models_to_test,
        vol_stats=vol_stats,
    )


    # --- CPU/BLAS parallelism controls (single wide trial) ---
    import os, multiprocessing
    from threadpoolctl import threadpool_limits

    # One Optuna worker only (sequential trials)
    n_jobs = int(os.getenv("OPTUNA_N_JOBS", "1"))  # keep = 1

    # Wide intra-trial parallelism via BLAS threads:
    _bl_env = os.getenv("BLAS_THREADS_PER_TRIAL", "").strip()
    if _bl_env:
        blas_threads = max(1, int(_bl_env))
    else:
        # CPU-centric fallback: ~75% of logical cores, leave 2 for OS
        _cpu = os.cpu_count() or multiprocessing.cpu_count() or 8
        blas_threads = max(2, min(_cpu - 2, int(round(_cpu * 0.75))))

    print(f"[Optuna] sequential n_jobs={n_jobs} | BLAS_THREADS_PER_TRIAL={blas_threads} | CV_JOBS={os.getenv('CV_JOBS', '?')}")

    # Cap NumPy/SciPy/Sklearn/XGB BLAS threads inside the trial
    with threadpool_limits(limits=blas_threads):
        # ------------------------------------------------------------
        # Patch: plateau early-stop (optional)
        # Stop if best_value hasn't improved by >= plateau_delta for
        # plateau_patience consecutive trials.
        # ------------------------------------------------------------
        plateau_patience = int(cv_config.get("plateau_patience", 0) or 0)
        plateau_delta = float(cv_config.get("plateau_delta", 0.0) or 0.0)
        plateau_min_trials = int(cv_config.get("plateau_min_trials", 0) or 0)

        if plateau_patience <= 0:
            # Backwards-compatible default behavior
            study.optimize(func, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)
        else:
            print(f"[Optuna] Plateau stop enabled: patience={plateau_patience} "
                  f"delta={plateau_delta} min_trials={plateau_min_trials}")

            _target_trials = int(n_trials) if n_trials is not None else 0
            _target_trials = max(0, _target_trials)

            # Support resumed studies (if study already has trials)
            try:
                _best = float(getattr(study, "best_value", None))
            except Exception:
                _best = None
            _no_improve = 0

            for _i in range(_target_trials):
                study.optimize(func, n_trials=1, n_jobs=n_jobs, gc_after_trial=True)

                _after = getattr(study, "best_value", None)
                if _after is None:
                    continue

                try:
                    _after_f = float(_after)
                except Exception:
                    # If comparison fails, don't early-stop
                    _no_improve = 0
                    continue

                if _best is None:
                    _best = _after_f
                    _no_improve = 0
                elif _after_f >= (_best + plateau_delta):
                    _best = _after_f
                    _no_improve = 0
                else:
                    _no_improve += 1

                _done = _i + 1
                _min_ok = (_done >= plateau_min_trials) if plateau_min_trials > 0 else True
                if _min_ok and (_no_improve >= plateau_patience):
                    print(f"[Optuna] Plateau stop: no improvement >= {plateau_delta} "
                          f"for {plateau_patience} trials (best={_best}). "
                          f"Stopped at {_done}/{_target_trials} trials.")
                    break


                
    # --- post-study cleanup (runs once per study) ---
    try:
        import tensorflow as _tf
        _tf.keras.backend.clear_session()
    except Exception:
        pass

    # Optional: only helps if loky reusable executor was created somewhere
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass

    import gc as _gc
    _gc.collect()

    # Log hyperparameter boundary hits (how often we sampled near the bounds)
    try:
        if HP_BOUNDARY_HITS:
            log_print("[Optuna] Hyperparameter boundary hits across all trials:", level="COMPACT")
            for _name, _count in sorted(HP_BOUNDARY_HITS.items(), key=lambda kv: (-kv[1], kv[0])):
                log_print(f"  - {_name}: {_count}", level="COMPACT")
                
            # ------------------------------------------------------------
            # Split min/max edge pressure and recommend small range expansion.
            # Does NOT change search ranges automatically.
            # ------------------------------------------------------------
            try:
                _n_trials_total = int(len(getattr(study, "trials", []) or []))
                _n_trials_total = max(1, _n_trials_total)
                _ratio_thr = float((cv_config or {}).get("range_suggest_ratio_thr", 0.25))
                _expand = float((cv_config or {}).get("range_suggest_expand_frac", 0.25))

                _items = []
                for _p in set(list(HP_BOUNDARY_HITS_MIN.keys()) + list(HP_BOUNDARY_HITS_MAX.keys()) + list(HP_BOUNDARY_RANGES.keys())):
                    _hm = int(HP_BOUNDARY_HITS_MIN.get(_p, 0) or 0)
                    _hM = int(HP_BOUNDARY_HITS_MAX.get(_p, 0) or 0)
                    if _hm <= 0 and _hM <= 0:
                        continue
                    _items.append((_p, _hm, _hM))

                def _is_loglike(_name: str, _low: float, _high: float) -> bool:
                    _n = str(_name).lower()
                    if any(k in _n for k in ["lr", "learning_rate", "label_threshold", "alpha", "beta", "gamma"]):
                        return True
                    try:
                        if _low > 0 and (_high / _low) >= 10.0:
                            return True
                    except Exception:
                        pass
                    return False

                _printed = 0
                for _p, _hm, _hM in sorted(_items, key=lambda x: (-(max(x[1], x[2])), x[0])):
                    _rng = HP_BOUNDARY_RANGES.get(_p, None)
                    if not _rng:
                        continue
                    _low, _high = float(_rng[0]), float(_rng[1])
                    if not (_high > _low):
                        continue
                    _span = _high - _low

                    _rmin = _hm / float(_n_trials_total)
                    _rmax = _hM / float(_n_trials_total)
                    if _rmin < _ratio_thr and _rmax < _ratio_thr:
                        continue

                    _loglike = _is_loglike(_p, _low, _high)
                    _sug_low = None
                    _sug_high = None

                    if _rmin >= _ratio_thr:
                        if _loglike and _low > 0:
                            _sug_low = _low / (1.0 + _expand)
                        else:
                            _sug_low = _low - (_expand * _span)
                        if _low >= 0.0:
                            _sug_low = max(0.0, float(_sug_low))

                    if _rmax >= _ratio_thr:
                        if _loglike and _high > 0:
                            _sug_high = _high * (1.0 + _expand)
                        else:
                            _sug_high = _high + (_expand * _span)

                    if _printed == 0:
                        log_print("[HPO][RANGE-SUGGEST] boundary pressure detected; consider adjusting ranges:", level="COMPACT")

                    _min_txt = f"{_hm}/{_n_trials_total}"
                    _max_txt = f"{_hM}/{_n_trials_total}"
                    _sl = "NA" if _sug_low is None else f"{float(_sug_low):.6g}"
                    _sh = "NA" if _sug_high is None else f"{float(_sug_high):.6g}"
                    log_print(
                        f"[HPO][RANGE-SUGGEST] param={_p} hits_min={_min_txt} hits_max={_max_txt} "
                        f"low={_low:.6g} high={_high:.6g} suggest_low={_sl} suggest_high={_sh}",
                        level="COMPACT",
                    )
                    _printed += 1
                    if _printed >= int((cv_config or {}).get("range_suggest_max_params", 12)):
                        break
            except Exception:
                pass
    except Exception:
        # Diagnostics-only; never break tuning if logging fails.
        pass
    
    # Completed trials only
    from optuna.trial import TrialState
    
    TOPN_FOR_WFO = 5
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]

    if not completed:
        raise RuntimeError("No completed Optuna trials; cannot select Top-N.")

    # 👉 Re-rank by Deflated Sharpe proxy (DSR)
    try:
        from utilsNoWFO import compute_dsr_scores
        _scores = [float(t.value) for t in completed]
        _dsr    = compute_dsr_scores(_scores)
        for t, d in zip(completed, _dsr):
            try:
                t.set_user_attr("dsr", float(d))
            except Exception:
                pass

        # Sort by DSR descending (more conservative than raw Sharpe)
        # Optional tie-breaker: if a calibration metric (brier or nll) is present
        # in trial.user_attrs, prefer better-calibrated configs among similar DSR.
        def _rank_key(t):
            dsr = float(t.user_attrs.get("dsr", 0.0))
            # Lower is better for Brier/NLL; default 0.0 means "no info"
            brier = t.user_attrs.get("brier", None)
            nll   = t.user_attrs.get("nll", None)
            if brier is not None:
                return (dsr, -float(brier))
            if nll is not None:
                return (dsr, -float(nll))
            return (dsr, 0.0)

        completed.sort(key=_rank_key, reverse=True)

        
    except Exception:
        # Fallback: raw value
        completed.sort(key=lambda t: t.value, reverse=True)
        
    # --- Diversity-aware Top-N helper ---
    def _build_diverse_top_trials(trials, top_n: int):
        """
        Select up to top_n trials, preserving performance order, but avoiding
        near-duplicate configs in a normalized hyperparameter space.

        Two trials are treated as near-duplicates if they:
          - share the same model_type and strategy_type,
          - use the same active feature subset (use_* flags),
          - and lie within a small normalized radius (GEOM_RADIUS) for key knobs:
            lags_range, lag_depth, target_active_rate, label_threshold,
            alpha_vol_z, beta_spread_norm, gamma_slip_norm.

        This approximates a "robust region" around the best trial instead of
        picking many almost-identical spike solutions.
        """
        import math

        selected   = []
        signatures = []

        # Hyperparameter ranges used only for distance normalization.
        # Keep these aligned with the suggest_* ranges above.
        HP_RANGES = {
            "lags_range":         (8.0, 40.0),   # 8–24 (ensembles) or 12–40 (others) → global span
            "lag_depth":          (1.0, 4.0),    # 1–3 or 2–4
            "target_active_rate": (0.24, 0.26),  # as in trial.suggest_float(...)
            "label_threshold":    (5e-5, 5e-3),  # covers dynamic σ-based bounds
            "alpha_vol_z":        (0.0, 0.03),
            "beta_spread_norm":   (0.0, 0.08),
            "gamma_slip_norm":    (0.0, 0.08),
        }

        # Radius in normalized space (≈10–15% of search span).
        GEOM_RADIUS = 0.10

        def _norm_dist(v1, v2, key):
            lo, hi = HP_RANGES.get(key, (None, None))
            if lo is None or hi is None or hi <= lo:
                return math.inf
            if v1 is None or v2 is None:
                return math.inf
            try:
                return abs(float(v1) - float(v2)) / (hi - lo)
            except Exception:
                return math.inf

        def _trial_signature(t):
            """
            Compact signature capturing:
              - model_type, strategy_type
              - active feature subset (use_* flags)
              - the raw values of the key hyperparameters we distance-check.
            """
            p = getattr(t, "params", {}) or {}
            model = p.get("model_type", None)
            strat = p.get("strategy_type", None)

            # Feature subset: names of toggles that are True.
            active_feats = tuple(
                sorted(k for k, v in p.items() if k.startswith("use_") and bool(v))
            )

            sig_vals = {k: p.get(k, None) for k in HP_RANGES.keys()}

            return (model, strat, active_feats, sig_vals)

        def _too_similar(sig, others):
            model, strat, feats, vals = sig
            if strat is None:
                # If no strategy_type, don't try to merge aggressively.
                return False

            for o_model, o_strat, o_feats, o_vals in others:
                # Only compare inside the same family + strategy + feature subset.
                if model != o_model:
                    continue
                if o_strat is None or strat != o_strat:
                    continue
                if feats != o_feats:
                    continue

                # Compute max normalized distance across the tracked knobs.
                max_d      = 0.0
                any_finite = False
                for k in HP_RANGES.keys():
                    d = _norm_dist(vals.get(k), o_vals.get(k), k)
                    if not math.isfinite(d):
                        continue
                    any_finite = True
                    if d > max_d:
                        max_d = d

                # If we had at least one comparable dimension and everything is within
                # GEOM_RADIUS, treat as "same robust region" → near-duplicate.
                if any_finite and max_d <= GEOM_RADIUS:
                    return True

            return False

        for t in trials:
            sig = _trial_signature(t)
            if _too_similar(sig, signatures):
                # Skip very similar config; we already have a representative.
                continue
            selected.append(t)
            signatures.append(sig)
            if len(selected) >= top_n:
                break

        if not selected:
            # Fallback: original behavior
            return trials[:top_n]
        return selected

    # Top-N (diversity-aware)
    top_n = max(1, int(return_top_n))
    top_trials = _build_diverse_top_trials(completed, top_n)
    
    def _merged_trial_params(_t):
        """Merge trial.params with user_attrs["full_params"] without losing keys.

        Some runs store derived/expanded keys (e.g., resolved roll windows) in full_params,
        but in a few cases full_params can be partial. We therefore merge on top of
        trial.params and refuse to clobber a non-empty value with an empty placeholder.
        """
        base = dict(getattr(_t, "params", {}) or {})
        fp = None
        try:
            fp = _t.user_attrs.get("full_params", None)
        except Exception:
            fp = None
        if isinstance(fp, dict) and fp:
            for k, v in fp.items():
                # Do not overwrite a non-empty base value with an empty/None placeholder.
                if v is None:
                    base.setdefault(k, v)
                    continue
                if isinstance(v, str) and v.strip() == "":
                    base.setdefault(k, v)
                    continue
                if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
                    base.setdefault(k, v)
                    continue
                base[k] = v
        return base

    top_params = [_merged_trial_params(t) for t in top_trials]

    
    # Decide output location:
    # - If month_out_dir is provided → save ONLY the per-month plot there with the project's filename.
    # - Else (legacy) → create an optuna_runs/<id> audit folder with the usual artifacts.
    legacy_optuna_dir = None
    if month_out_dir:
        out_dir = month_out_dir
    else:
        run_id  = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        legacy_optuna_dir = os.path.join("optuna_runs", run_id)
        out_dir = legacy_optuna_dir
    os.makedirs(out_dir, exist_ok=True)

    top_path = os.path.join(out_dir, f"top{top_n}.json")
    
    if legacy_optuna_dir and (save_optuna_learning_summary is not None):
        try:
            save_optuna_learning_summary(
                study,
                os.path.join(out_dir, "learning_summary.json"),
                n_startup=int(n_startup_trials),
            )

        except Exception as _e:
            print(f"⚠️ Could not save learning summary: {_e}")

    if legacy_optuna_dir:
        try:
            from optuna.importance import get_param_importances
            imps = get_param_importances(study)  # uses study.best_value as target by default
            import json as _json, collections as _c
            imps_path = os.path.join(out_dir, "param_importances.json")

            # Convert OrderedDict/Keys to plain dict for JSON
            imps = {str(k): float(v) for k, v in imps.items()}
            with open(imps_path, "w") as f:
                _json.dump(imps, f, indent=2, sort_keys=True)
            print(f"✅ Saved Optuna param importances → {imps_path}")
        except Exception as _e:
            print(f"⚠️ Could not compute/save param importances: {_e}")

    # --- Build Top-N, normalize, and save audit JSON ---

    from optuna.trial import TrialState
    from optuna.study import StudyDirection

    # Local helpers
    def _normalize_roll_windows_inplace(p: dict) -> None:
        """Ensure 'roll_windows' exists and drop legacy/versioned selector keys."""
        rk = p.get("roll_windows_key_v2") or p.get("roll_windows_key")
        if "roll_windows" not in p and rk is not None:
            p["roll_windows"] = [int(x) for x in str(rk).split(",") if str(x).strip() != ""]
        p.pop("roll_windows_key_v2", None)
        p.pop("roll_windows_key", None)

    def _json_sanitize_inplace(p: dict) -> None:
        """Cast numpy scalars to native types + sanitize lists for JSON."""
        import numpy as np
        for k, v in list(p.items()):
            if isinstance(v, np.generic):
                p[k] = v.item()
            elif isinstance(v, (list, tuple)):
                p[k] = [x.item() if isinstance(x, np.generic) else x for x in v]
                
    def _ensure_lags_inplace(p: dict) -> None:
        """Ensure 'lags' exists to avoid warnings later."""
        if "lags" not in p and "lags_range" in p:
            try:
                p["lags"] = int(p.get("lags_range"))
            except Exception:
                pass

    def _pick_preselected_winner(trials, min_trades_cv: float = 5.0, min_active_cv: float = 0.02) -> int:
        """Choose first Top-N trial whose CV metrics meet basic gates; else 0."""
        import math
        for j, t in enumerate(trials):
            try:
                tr = float(t.user_attrs.get("trades_cv", float("nan")))
                ar = float(t.user_attrs.get("active_rate_cv", float("nan")))
                if (not math.isnan(tr)) and (not math.isnan(ar)) and (tr >= float(min_trades_cv)) and (ar >= float(min_active_cv)):                    
                    return j
            except Exception:
                continue
        return 0

    # Build consensus pool from all completed trials (for Top-N consensus selection)
    # New behaviour:
    #   - Ignore any performance fraction filter.
    #   - Take up to 'consensus_pool_max_trials' best VALID trials
    #     in the DSR-ranked 'completed' list.
    #   - "Valid" = basic trades / active-rate gates.
    #   - Similarity vs. the CV winner (not equal / not too different) is
    #     enforced later in MLBacktester._evaluate_with_topn_consensus via
    #     style + geometry filters.
    consensus_pool: list[dict] = []
    import math as _math

    def _trial_is_valid_for_consensus(_t, min_trades: float = 5.0, min_active: float = 0.02) -> bool:
        """
        Basic validity gate for consensus use:
          - enough trades in CV,
          - non-tiny active rate (avoid degenerate near-always-neutral configs).
        """
        try:
            tr = float(_t.user_attrs.get("trades_cv", float("nan")))
            ar = float(_t.user_attrs.get("active_rate_cv", float("nan")))
        except Exception:
            return False
        if _math.isnan(tr) or _math.isnan(ar):
            return False
        return (tr >= float(min_trades)) and (ar >= float(min_active))

    if completed:
        # 'completed' is already DSR-ranked above (or raw-value sorted as fallback),
        # so we just walk it in that order and collect valid trials.
        valid_trials = [t for t in completed if _trial_is_valid_for_consensus(t)]

        # If nothing passes the validity gate, fall back to all completed trials
        # so that consensus still has something to work with.
        if not valid_trials:
            valid_trials = completed

        for _t in valid_trials[: int(consensus_pool_max_trials)]:
            try:
                v = float(_t.value)
            except Exception:
                continue
            if not _math.isfinite(v):
                continue

            # Build param dict in same style as the Top-N payload
            p = dict(_t.params)
            _normalize_roll_windows_inplace(p)
            _ensure_lags_inplace(p)
            _json_sanitize_inplace(p)

            # Attach CV metrics for diagnostics / runtime selection
            try:
                p["__cv_value"] = float(v)
            except Exception:
                pass
            try:
                p["__cv_psr"] = float(_t.user_attrs.get("psr", float("nan")))
            except Exception:
                pass
            try:
                p["__cv_dsr"] = float(_t.user_attrs.get("dsr", float("nan")))
            except Exception:
                pass
            try:
                p["__trades_cv"] = float(_t.user_attrs.get("trades_cv", float("nan")))
                p["__active_cv"] = float(_t.user_attrs.get("active_rate_cv", float("nan")))
            except Exception:
                pass
            try:
                p["__trial_number"] = int(getattr(_t, "number", -1))
            except Exception:
                pass

            consensus_pool.append(p)
    # Logging-only: summarize the consensus pool built from completed trials
    try:
        _trials = [int(p.get("__trial_number", -1)) for p in (consensus_pool or []) if isinstance(p, dict)]
        _vals   = [float(p.get("__cv_value", float("nan"))) for p in (consensus_pool or []) if isinstance(p, dict)]
        print(f"[TopN][BuiltPool] size={len(consensus_pool)} trials={_trials} cv_values={_vals}")
    except Exception:
        pass

    # 1) Compute Top-N trials if not already available
    try:
        top_params  # noqa: F401
        top_trials  # noqa: F401
    except NameError:
        
        # MAXIMIZE-ONLY safety: never allow silent direction inversions.
        if study.direction != StudyDirection.MAXIMIZE:
            raise RuntimeError(f"Only MAXIMIZE supported, got {study.direction}")
        completed = [
            t for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if completed:
            # MAXIMIZE: higher objective value = better trial
            completed.sort(key=lambda t: float(t.value), reverse=True)
            top_trials = completed[:max(int(top_n), 0)]
        else:
            top_trials = []
        top_params = [_merged_trial_params(t) for t in top_trials]

    # 2) Build/normalize best params

    # Prefer the *materialized* params recorded during the objective (includes
    # derived keys like roll_windows / indicator_windows). Falling back to
    # trial.params can cause replay drift in real_trading_simulation.
    _bt = study.best_trial
    _bp = None
    try:
        _bp = _bt.user_attrs.get("full_params", None)
    except Exception:
        _bp = None
    best_params = _merged_trial_params(_bt)
    _normalize_roll_windows_inplace(best_params)
    _json_sanitize_inplace(best_params)
    
    # Normalize 'lags' everywhere to remove eval warnings
    _ensure_lags_inplace(best_params)
    for p in top_params:
        _ensure_lags_inplace(p)

    # Pre-commit a CV winner index (0 = top by DSR/value order)
    winner_index = _pick_preselected_winner(top_trials, min_trades_cv=5.0, min_active_cv=0.02)
    best_params["__winner_index"] = int(winner_index)

    # (Optional) annotate CV rank on the Top-N payload (for audit)
    for i, t in enumerate(top_trials):
        try:
            t.set_user_attr("__cv_rank", int(i))
        except Exception:
            pass

    # 3) Normalize Top-N param dicts in-place
    for p in top_params:
        _normalize_roll_windows_inplace(p)
        _json_sanitize_inplace(p)

    # 4) Rebuild the "params-only" view AFTER normalization
    top_params_only = [{k: v for k, v in p.items() if not str(k).startswith("__")} for p in top_params]

    # 5) Prepare/normalize top_payload (create if absent)
    try:
        top_payload  # noqa: F401
    except NameError:
        top_payload = {
            "study_name": study.study_name,
            "direction": study.direction.name.lower(),
            "top_n": int(top_n),
            "trials": [
                {"number": t.number, "value": float(t.value), "params": dict(t.params)}
                for t in top_trials
            ],
        }
    # If it has embedded params, normalize them too
    if isinstance(top_payload, dict):
        if "top_params" in top_payload and isinstance(top_payload["top_params"], list):
            top_payload["top_params"] = top_params_only
        elif "trials" in top_payload and isinstance(top_payload["trials"], list):
            for t in top_payload["trials"]:
                if isinstance(t, dict) and "params" in t and isinstance(t["params"], dict):
                    _normalize_roll_windows_inplace(t["params"])
                    _json_sanitize_inplace(t["params"])

    # 6) Save Top-N JSON (audit)
    try:
        with open(top_path, "w") as f:
            json.dump(top_payload, f, indent=2, sort_keys=True)
    except Exception as _e:
        print(f"⚠️ Failed to write Top-{top_n} JSON: {_e}")

    # 7) Embed Top-N pointers into best_params (for downstream refit)
    best_params["__top5_params"] = top_params_only
    best_params["__top5_path"]   = top_path
    best_params["__top5_info"]   = top_payload
    
    # ------------------------------------------------------------------
    # Optional single-shot consensus finalization (freeze committee ONCE)
    #
    # If enabled via cv_config["use_consensus"], we build a fixed committee
    # *here* (end of global HPO) and store it into best_params so downstream
    # month-by-month simulation can reuse the same committee without
    # re-selecting neighbours each month.
    # ------------------------------------------------------------------
    try:
        _use_consensus = bool(cv_config.get("use_consensus", False))
    except Exception:
        _use_consensus = False

    if _use_consensus:
        try:
            from utilsNoWFO import _infer_family
            _family = _infer_family(str(best_params.get("model_type", "")))
        except Exception:
            _family = "Unknown"

        # committee size mirrors MLBacktester._evaluate_with_topn_consensus
        try:
            if _family == "Classical":
                _N_target = int(cv_config.get("topN_classical", 3))
            elif _family in {"RL"}:
                _N_target = int(cv_config.get("topN_deep", 2))
            elif _family in {"Ensembles"}:
                _N_target = int(cv_config.get("topN_ensemble", 2))
            elif _family in {"DQN"}:
                _N_target = int(cv_config.get("topN_dqn", 2))
            else:
                _N_target = int(cv_config.get("topN_default", 2))
        except Exception:
            _N_target = 2
        _N_target = max(2, int(_N_target))

        # Base params (strip helper keys)
        _base_core = {k: v for k, v in (best_params or {}).items() if not str(k).startswith("__")}

        def _params_key(d: dict) -> str:
            try:
                return json.dumps({k: v for k, v in (d or {}).items() if not str(k).startswith("__")}, sort_keys=True, default=str)
            except Exception:
                return str(d)

        _seen = set()
        _committee = []
        _committee.append(dict(_base_core))
        _seen.add(_params_key(_base_core))

        # Sort pool by stored CV objective value (respect study direction)
        _is_min = False
        try:
            _is_min = str(getattr(study, "direction", "maximize")).lower().startswith("min")
        except Exception:
            _is_min = False

        def _pool_value(p: dict):
            try:
                v = p.get("__cv_value", p.get("cv_value", p.get("value", None)))
                return float(v) if v is not None else float("nan")
            except Exception:
                return float("nan")

        _pool_sorted = list(consensus_pool or [])
        try:
            _pool_sorted.sort(key=_pool_value, reverse=(not _is_min))
        except Exception:
            pass

        _selected_trials = []
        for _p in _pool_sorted:
            if not isinstance(_p, dict):
                continue
            # Keep params, plus lightweight meta keys that Top-N can read.
            _cand = {k: v for k, v in _p.items() if not str(k).startswith("__")}
            try:
                if "trial_number" not in _cand and _p.get("__trial_number", None) is not None:
                    _cand["trial_number"] = int(_p.get("__trial_number"))
            except Exception:
                pass
            try:
                if "value" not in _cand and _p.get("__cv_value", None) is not None:
                    _cand["value"] = float(_p.get("__cv_value"))
            except Exception:
                pass

            _k = _params_key(_cand)
            if _k in _seen:
                continue
            _seen.add(_k)
            _committee.append(_cand)

            try:
                if _cand.get("trial_number", None) is not None:
                    _selected_trials.append(int(_cand.get("trial_number")))
            except Exception:
                pass

            if len(_committee) >= _N_target:
                break

        # Persist on the returned best_params (downstream will reuse)
        best_params["__committee_fixed"] = _committee
        best_params["__committee_fixed_info"] = {
            "enabled": True,
            "N_target": int(_N_target),
            "family": str(_family),
            "selected_trial_numbers": _selected_trials,
            "pool_size": int(len(consensus_pool or [])),
        }

        # Write an audit artifact alongside Top-N JSON
        try:
            _cons_path = os.path.join(out_dir, "consensus_frozen.json")
            with open(_cons_path, "w") as f:
                json.dump(
                    {
                        "info": best_params.get("__committee_fixed_info", {}),
                        "committee": best_params.get("__committee_fixed", []),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            best_params["__committee_fixed_path"] = _cons_path
        except Exception:
            pass
        
    try:
        _do_robust = bool((cv_config or {}).get("robustness_eval", False))
        _robust_src = "cv_config" if (isinstance(cv_config, dict) and ("robustness_eval" in cv_config)) else "default_off"
    except Exception:
        _do_robust = False
        _robust_src = "error"
        
    if _do_robust:
        # Determinism helper (already exists in your codebase)
        try:
            from utilsNoWFO import set_global_determinism
        except Exception:
            set_global_determinism = None

        class _DummyTrial:
            """Minimal Optuna-trial-like object to capture CV user_attrs without pruning."""
            def __init__(self):
                self.user_attrs = {}
            def set_user_attr(self, k, v):
                self.user_attrs[str(k)] = v
            def report(self, value, step):
                return
            def should_prune(self):
                return False

        def _iqr(x):
            xs = [float(v) for v in x if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
            if len(xs) == 0:
                return float("nan")
            xs.sort()
            def _q(p):
                if len(xs) == 1:
                    return xs[0]
                i = p * (len(xs) - 1)
                lo = int(math.floor(i))
                hi = int(math.ceil(i))
                if lo == hi:
                    return xs[lo]
                w = i - lo
                return xs[lo] * (1 - w) + xs[hi] * w
            return _q(0.75) - _q(0.25)

        def _median(x):
            xs = [float(v) for v in x if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
            if len(xs) == 0:
                return float("nan")
            xs.sort()
            n = len(xs)
            if n % 2 == 1:
                return xs[n // 2]
            return 0.5 * (xs[n // 2 - 1] + xs[n // 2])

        # Candidate set: best + optional top-K
        _topk = int(cv_config.get("robustness_top_k", 1))
        _topk = max(1, min(5, _topk))
        _cands = []
        try:
            _cands.append(dict(best_params))
        except Exception:
            _cands.append(best_params)
        try:
            for p in (top_params_only or []):
                if len(_cands) >= _topk:
                    break
                _cands.append(dict(p))
        except Exception:
            pass

        # Seeds: default 3, configurable to 5
        _seeds = cv_config.get("robustness_seeds", None)
        if not isinstance(_seeds, (list, tuple)) or len(_seeds) == 0:
            _seeds = [101, 202, 303]
        _seeds = list(_seeds)[: int(cv_config.get("robustness_max_seeds", 5))]

        # Conservative rejection rule (configurable)
        # - PASS if: median_SR >= min_median_sr
        #           and median_trades >= min_median_trades
        #           and IQR_SR <= max_iqr_sr
        #           and worst_SR >= min_worst_sr
        _min_med_sr = float(cv_config.get("robust_min_median_sr", 0.10))
        _min_med_tr = int(cv_config.get("robust_min_median_trades", 8))
        _max_iqr_sr = float(cv_config.get("robust_max_iqr_sr", 0.60))
        _min_worst_sr = float(cv_config.get("robust_min_worst_sr", -0.50))

        def _rule_text():
            return (f"median_SR>={_min_med_sr:.2f} "
                    f"median_trades>={_min_med_tr} "
                    f"IQR_SR<={_max_iqr_sr:.2f} "
                    f"worst_SR>={_min_worst_sr:.2f}")

        # Evaluate each candidate across seeds
        for i, cand in enumerate(_cands):
            sr_list, tr_list, dd_list, eq_list = [], [], [], []
            worst_sr = float("inf")
            for sd in _seeds:
                try:
                    if set_global_determinism is not None:
                        set_global_determinism(int(sd))
                except Exception:
                    pass

                _trial = _DummyTrial()
                try:
                    min_tw = int(cv_config.get("min_train_window", 28032))
                    val_w  = int(cv_config.get("val_window", 1475))
                    
                    # Evaluate via the existing CV function.
                    # Score is treated as Sharpe proxy (your objective is Sharpe-like)
                    _score = evaluate_cv_func(
                        train_data,
                        cand,
                        min_train_window=min_tw,
                        val_window=val_w,
                        trial=_trial,
                        cv_config_override=cv_config,
                    )
                    _sr = float(_score)
                except Exception as _e:
                    _sr = float("nan")

                sr_list.append(_sr)
                if math.isfinite(_sr):
                    worst_sr = min(worst_sr, _sr)

                # Trades from existing CV attrs (already set in MLBacktesterNoWFO)
                try:
                    tr_list.append(float(_trial.user_attrs.get("trades_cv", float("nan"))))
                except Exception:
                    tr_list.append(float("nan"))

                # Optional (will become real once Patch 6 adds these attrs)
                try:
                    dd_list.append(float(_trial.user_attrs.get("dd_cv", float("nan"))))
                except Exception:
                    dd_list.append(float("nan"))
                try:
                    eq_list.append(float(_trial.user_attrs.get("eq_end_cv", float("nan"))))
                except Exception:
                    eq_list.append(float("nan"))

            med_sr = _median(sr_list)
            iqr_sr = _iqr(sr_list)
            med_tr = _median(tr_list)
            med_dd = _median(dd_list)
            med_eq = _median(eq_list)

            passed = True
            if not (math.isfinite(med_sr) and med_sr >= _min_med_sr):
                passed = False
            if not (math.isfinite(med_tr) and med_tr >= float(_min_med_tr)):
                passed = False
            if math.isfinite(iqr_sr) and iqr_sr > _max_iqr_sr:
                passed = False
            if worst_sr is float("inf") or (math.isfinite(worst_sr) and worst_sr < _min_worst_sr):
                passed = False

            dd_txt = "NA" if (not math.isfinite(med_dd)) else f"{med_dd:.4f}"
            eq_txt = "NA" if (not math.isfinite(med_eq)) else f"{med_eq:.4f}"

            log_print(
                f"[ROBUST] cand={i} seeds={_seeds} median_SR={med_sr:.4f} IQR_SR={iqr_sr:.4f} "
                f"median_trades={med_tr:.1f} median_DD={dd_txt} median_eq={eq_txt} "
                f"rule='{_rule_text()}' {'PASS' if passed else 'FAIL'}",
                level="COMPACT",
            )

            # Store for downstream reporting
            try:
                if i == 0:
                    best_params["__robust_seeds"] = list(_seeds)
                    best_params["__robust_median_sr"] = float(med_sr) if math.isfinite(med_sr) else None
                    best_params["__robust_iqr_sr"] = float(iqr_sr) if math.isfinite(iqr_sr) else None
                    best_params["__robust_median_trades"] = float(med_tr) if math.isfinite(med_tr) else None
                    best_params["__robust_pass"] = bool(passed)
                    best_params["__robust_rule"] = _rule_text()
            except Exception:
                pass

            # Optional hard reject: if best cand fails, downgrade to next passing cand
            if i == 0 and (not passed) and bool(cv_config.get("robust_fail_downgrade_to_next", True)):
                # try to find a passing candidate among the remaining ones
                continue
            if i == 0:
                # keep going but do not auto-swap unless explicitly requested later
                pass

        # If requested, enforce that the top config must pass robustness
        if bool(cv_config.get("robust_require_pass", False)):
            if not bool(best_params.get("__robust_pass", False)):
                raise RuntimeError("[ROBUST] Best candidate FAILED robustness and robust_require_pass=True")
            
    # ------------------------------------------------------------------
    # Purpose:
    #   - Use fast mini-block CV for screening during HPO
    #   - Then confirm ONLY Top-N with a more realistic monthly-roll CV
    #
    # Controls:
    #   cv_config["verify_topn_monthly_roll"] = True/False (explicit)
    #   cv_config["verify_topn_count"]        = N (default 5)
    #   cv_config["verify_cv_blocks"]         = folds for monthly-roll (default 5)
    #   cv_config["verify_cv_val_months"]     = val months (default 1.0)
    # ------------------------------------------------------------------
    best_score_override = None
    try:
        _do_verify = bool((cv_config or {}).get("verify_topn_monthly_roll", False))
        _verify_src = "cv_config" if (isinstance(cv_config, dict) and ("verify_topn_monthly_roll" in cv_config)) else "default_off"
    except Exception:
        _do_verify = False
        _verify_src = "error"

    if _do_verify and train_data is not None and callable(evaluate_cv_func):
        try:
            _verify_n = int(cv_config.get("verify_topn_count", 5))
        except Exception:
            _verify_n = 5
        _verify_n = max(1, _verify_n)

        # Build a monthly-roll CV override for verification (final exam)
        verify_cv_config = dict(cv_config) if isinstance(cv_config, dict) else {}
        verify_cv_config["cv_mode"] = "monthly_roll"
        try:
            verify_blocks = int(verify_cv_config.get("verify_cv_blocks", cv_config.get("verify_cv_blocks", 5)))
        except Exception:
            verify_blocks = 5
        verify_blocks = max(2, verify_blocks)
        verify_cv_config["cv_blocks"] = verify_blocks
        verify_cv_config["cv_target_folds"] = int(verify_cv_config.get("cv_target_folds", verify_blocks))
        verify_cv_config["cv_tail_anchor"] = True
        try:
            verify_cv_config["cv_val_months"] = float(verify_cv_config.get("verify_cv_val_months", cv_config.get("verify_cv_val_months", 1.0)))
        except Exception:
            verify_cv_config["cv_val_months"] = 1.0

        # Evaluate only Top-N candidates
        candidates = []
        try:
            # top_params_only is the normalized params-only list
            candidates = list(top_params_only[:_verify_n]) if top_params_only else []
        except Exception:
            candidates = []

        # Ensure current best is included (front-load it for logging clarity)
        try:
            if isinstance(best_params, dict):
                _bp_clean = {k: v for k, v in best_params.items() if not str(k).startswith("__")}
                if _bp_clean and all((_bp_clean != c) for c in candidates if isinstance(c, dict)):
                    candidates = [_bp_clean] + candidates
        except Exception:
            pass

        # Minimal trial-like shim so evaluate_cv_func can write attrs safely
        class _DummyTrialVerify:
            def __init__(self):
                self.user_attrs = {}
            def set_user_attr(self, k, v):
                self.user_attrs[str(k)] = v
            def report(self, value, step):
                return
            def should_prune(self):
                return False

        try:
            min_tw = int(cv_config.get("min_train_window", 28032))
        except Exception:
            min_tw = 28032
        try:
            val_w = int(cv_config.get("val_window", 1475))
        except Exception:
            val_w = 1475

        verify_scores = []
        best_v_score = float("-inf")
        best_v_params = None
        best_v_idx = -1

        log_print(f"[VERIFY] Top-N monthly-roll verification enabled ({_verify_src}) "
                  f"model={_model_name} N={len(candidates)} folds={verify_blocks} val_months={verify_cv_config.get('cv_val_months')}",
                  level="COMPACT")

        for i, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            _trial = _DummyTrialVerify()
            try:
                _score = evaluate_cv_func(
                    train_data,
                    cand,
                    min_train_window=min_tw,
                    val_window=val_w,
                    trial=_trial,
                    cv_config_override=verify_cv_config,
                )
                _sr = float(_score)
            except Exception as e:
                _sr = float("-inf")
                try:
                    log_print(f"[VERIFY] Candidate {i} failed: {type(e).__name__}: {e}", level="DEBUG")
                except Exception:
                    pass

            verify_scores.append(_sr)
            if math.isfinite(_sr) and _sr > best_v_score:
                best_v_score = _sr
                best_v_params = dict(cand)
                best_v_idx = i

        # If verification produced a winner, promote it to final best_params
        if best_v_params is not None and math.isfinite(best_v_score):
            try:
                _hpo_best = float(getattr(study, "best_value", float("nan")))
            except Exception:
                _hpo_best = float("nan")

            # Preserve meta keys from current best_params (e.g., __top5_params, __winner_index)
            _meta = {}
            try:
                _meta = {k: v for k, v in best_params.items() if str(k).startswith("__")}
            except Exception:
                _meta = {}

            best_params = dict(best_v_params)
            best_params.update(_meta)
            best_params["__winner_index_preverify"] = int(best_params.get("__winner_index", -1))
            best_params["__verified_winner_index"] = int(best_v_idx)
            best_params["__verify_mode"] = "monthly_roll"
            best_params["__verify_topn_count"] = int(len(candidates))
            best_params["__hpo_best_score"] = _hpo_best
            best_params["__verify_best_score"] = float(best_v_score)

            # Override the returned best_score to match the verified winner
            best_score_override = float(best_v_score)

            log_print(f"[VERIFY] Promoted verified winner idx={best_v_idx} "
                      f"verify_score={best_v_score:.4f} (hpo_best={_hpo_best:.4f})",
                      level="COMPACT")


    
    

    # --- Patch B: persist tuned execution defaults (subset) ---
    TUNED_KEYS = [
        # labeling
        "use_triple_barrier","tb_pt_mult","tb_sl_mult","tb_max_holding","tb_neutral_zone",
        # calibration & features
        "calibrate_method","use_fracdiff","fracdiff_d","use_rv_features","rv_window_short","rv_window_long"
    ]

    try:
        import json, os
        tuned_subset = {k: best_params[k] for k in TUNED_KEYS if k in best_params}
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/best_exec_defaults.json", "w") as f:
            json.dump(tuned_subset, f, indent=2, sort_keys=True)
        print("[Tuning] Wrote tuned execution defaults → artifacts/best_exec_defaults.json")
    except Exception as e:
        print(f"[Tuning] Could not write tuned defaults: {e}")

    # If monthly-roll verification ran, return the verified score; otherwise keep study.best_value
    try:
        if "best_score_override" in locals() and (best_score_override is not None):
            best_score = float(best_score_override)
        else:
            best_score = float(study.best_value)
    except Exception:
        best_score = study.best_value

    # ---- Plots (only what's needed) ----
    # Only save Optuna progress for the *first* trading month.
    # Later months reuse tuned params, so we don't spam extra plots.
    if (save_optuna_progress_from_study is not None) and not SKIP_PLOTS:
        try:
            try:
                month_idx_int = int(month_ix) if month_ix is not None else 1
            except Exception:
                month_idx_int = 1

            if month_idx_int == 1:
                if month_out_dir:
                    # month_out_dir should now map to this model's graphs directory
                    base = os.path.join(month_out_dir, "optuna_scores_1")
                    save_optuna_progress_from_study(
                        study,
                        out_prefix=base,
                        metric_name="Sharpe",
                        style="nature",
                        palette="okabe_ito_no_black",
                    )
                else:
                    # Fallback: generic progress file in out_dir
                    save_optuna_progress_from_study(
                        study,
                        out_prefix=os.path.join(out_dir, "optuna_progress"),
                        metric_name="Sharpe",
                        style="nature",
                        palette="okabe_ito_no_black",
                    )
            else:
                print(
                    f"ℹ️ Skipping Optuna progress plot for month {month_idx_int} "
                    f"(only month 1 is plotted)."
                )
        except Exception as _e:
            print(f"⚠️ Failed to save Optuna progress: {_e}")

    # 2) Trial-level feature-frequency heatmap (top 20% trials, Sharpe-weighted)
    if legacy_optuna_dir and SAVE_TRIAL_FEATURE_FREQ and (save_feature_frequency_from_trials is not None) and not SKIP_PLOTS:

        try:
            save_feature_frequency_from_trials(
                study_or_trials=study,
                base_features=[],  # only engineered features
                out_png=os.path.join(out_dir, "feature_frequency_trials.png"),
                top_k=30,
                top_percent=0.20,
                weight_by_score=True,
                minimize_objective=False,              # TRUE Sharpe (maximize)
                style="nature",
                palette="okabe_ito_no_black",
                exclude_prefixes=("returns_lag","hour"),
                collapse_raw_lags=True,
            )
        except Exception as _e:
            print(f"⚠️ Failed to save trial-level feature frequency: {_e}")

    # 3) Trial-duration stats (per study/run)
    try:
        import csv
        import numpy as _np

        durations = []
        for t in study.trials:
            dur = getattr(t, "duration", None)
            if dur is not None:
                try:
                    durations.append(float(dur.total_seconds()))
                except Exception:
                    continue

        if durations:
            avg_sec = float(_np.mean(durations))
            med_sec = float(_np.median(durations))
            min_sec = float(_np.min(durations))
            max_sec = float(_np.max(durations))

            # month_ix is optional – best-effort cast
            try:
                m_ix = int(month_ix) if month_ix is not None else ""
            except Exception:
                m_ix = ""

            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "month_ix": m_ix,
                "n_trials": int(len(durations)),
                "avg_sec": avg_sec,
                "median_sec": med_sec,
                "min_sec": min_sec,
                "max_sec": max_sec,
                "models": ",".join(sorted(set(models_to_test))) if models_to_test else "",
            }

            stats_csv = os.path.join(out_dir, "optuna_trial_time_stats.csv")
            os.makedirs(out_dir, exist_ok=True)
            file_exists = os.path.exists(stats_csv)

            with open(stats_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            print(
                f"⏱️ Trial-time stats: avg={avg_sec:.2f}s "
                f"(n={len(durations)}) → {stats_csv}"
            )
        else:
            print("⏱️ No trial durations available to log.")
    except Exception as _e:
        print(f"⚠️ Could not save trial-time stats: {_e}")

    import gc as _gc
    _gc.collect()

    # ------------------------------------------------------------------
    # Persist best config / Top-N (or consensus pool) for later reuse
    # ------------------------------------------------------------------
    if save_hpo_config_to_disk is not None:
        try:
            # Best model_type should always be present in best_params
            model_type = str(best_params.get("model_type", "unknown"))

            # Minimal metadata about the study
            try:
                direction = study.direction.name if study is not None else None
                n_trials = len(study.trials) if study is not None else None
            except Exception:
                direction = None
                n_trials = None

            study_meta = {
                "best_score": float(best_score),
                "direction": direction,
                "n_trials": n_trials,
                "saved_at_utc": datetime.datetime.utcnow().isoformat(),
            }

            # Decide which configs to actually persist:
            # - If we built a consensus_pool, that is our "used configs" set.
            # - Else fall back to the Top-N list.
            # - As a last resort, persist just the single best config.
            try:
                if consensus_pool:
                    configs_to_persist = list(consensus_pool)
                elif top_params_only:
                    configs_to_persist = list(top_params_only)
                else:
                    configs_to_persist = [dict(best_params)]
            except Exception:
                configs_to_persist = [dict(best_params)]

            save_hpo_config_to_disk(
                model_type=model_type,
                best_params=best_params,
                topN_params=configs_to_persist,
                study_meta=study_meta,
            )
        except Exception as e:
            # Do not crash the tuning run just because persistence failed
            print(f"[HPO] Warning: failed to persist best config: {e}")

    return best_params, best_score, top_params_only, study, consensus_pool

