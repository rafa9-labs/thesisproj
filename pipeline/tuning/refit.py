"""Model refitting: final_refit_if_deep and per-model refit functions."""
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


from pipeline.tuning.sampler import _coerce_ensemble_lags

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

    # merge configs -> backtester reads self.features_config inside test_strategy
    final_cfg = dict(getattr(backtester, "features_config", {}) or {})
    final_cfg.update(best_params)
    final_cfg.update({
        # runtime overrides (make it "full" refit)
        "cnn_train_stride": 1,
        "cnn_epochs": max(30, int(best_params.get("cnn_epochs", 20))),
        "cnn_use_early_stopping": True,
        
        # Uncap deep training windows for final refit (CV is the cheap proxy)
        "deep_max_train_windows": 10**9,
        # keep windows mode consistent (or force True if you want)
        "cnn_use_seq_windows": final_cfg.get("cnn_use_seq_windows",
                             best_params.get("cnn_use_seq_windows", True)),
        # don't force mixed precision unless you know you're on GPU
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
        # full refit (no speed caps) -- but KEEP the same windowing regime as CV
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
    Uses the best params, disables time caps/stride, and bumps epochs--then evaluates the fold.
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

    # [FLAG] Final refit must be uncapped -> hard overrides (no setdefault)
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
    Uses the best params, disables time caps/stride, and bumps epochs--then evaluates the fold.
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

    # [FLAG] Final refit must be uncapped -> hard overrides (no setdefault)
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

    # Build the exact eval features_config: defaults -> (prior run cfg) -> best_params
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
            print(f"[WARN] Optuna keys changed in eval merge: {sorted(clobbered)}")

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
            print(f"[GateInfo][TUNING] target_active_rate={float(tar):.6f} is set -> confidence_threshold is overridden by coverage thresholding.")
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
        print("[WARN] Warning: invalid metrics shape from refit/selection; falling back to original.")
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
        Optuna best params -> (optional) one deployment refit -> trade.
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
                return True  # No GPU or nvidia-smi unavailable -- don't block
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
        print(f"[WARN] Low RAM: need>={need_gb:.2f}GB, avail={avail:.2f}GB -- proceeding (bootstrap).")
        return True

    return False

