"""Optuna objective function."""
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


from pipeline.tuning.helpers import (
    _bad_obj, _record_hp_boundary_hit,
    SAVE_TRIAL_FEATURE_FREQ, DISABLE_OPTUNA_PRUNING,
)
from pipeline.tuning.sampler import sample_param_set
from pipeline.tuning.refit import final_refit_if_deep, _aggressive_free, _assert_free_ram

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
            f"[WARN] [RAM-SoftGuard] Low RAM before trial {getattr(trial, 'number', '?')} start: "
            f"requested>={need:.2f}GB, avail={avail:.2f}GB -- continuing anyway."
        )

        # Optional hard emergency floor to protect your WSL/PC.
        # Very low (0.15 GB) by default; you can adjust via OPTUNA_HARD_FLOOR_GB.
        hard_floor = float(os.environ.get("OPTUNA_HARD_FLOOR_GB", "0.15"))
        if avail < hard_floor:
            if DISABLE_OPTUNA_PRUNING:
                # If pruning is globally disabled, return a sentinel instead of crashing.
                print(
                    f"[RAM] [RAM-SoftGuard] avail={avail:.2f}GB < hard floor={hard_floor:.2f}GB -- "
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

        # 1) EMA-SMA spread
        if bool(p.get("use_ma_spread", False)) and "ema_sma_spread" not in df:
            df["ema_sma_spread"] = (_ema_local - _sma_local)

        # 2) Price-MA z-scores
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

        # 4) Short-long slope differential (slope of spread)
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

                # 1) Trend components: price-MA z-scores, ADX, EMA-SMA spread (if available)
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
                print(f"[WARN] Regime feature construction failed: {_e}")

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


        # --- Dispatch: classical models -> thread backend for speed & stability ---
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
        #   CV_DEBUG=1        -> enable detailed CV debug logs
        #   CV_TABLE_MODE=off -> disable per-block CV tables
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
        _cv_jobs_raw = (os.getenv("CV_JOBS", "") or "").strip()
        try:
            cv_jobs = int(_cv_jobs_raw) if _cv_jobs_raw else 1
        except (ValueError, TypeError):
            cv_jobs = 1

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
                f"[WARN] {cause} -- marking trial {trial.number} as failed (no retry).",
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
                    f"[WARN] {cause} -- marking trial {trial.number} as failed (no retry).",
                    level="COMPACT",
                )
                try:
                    trial.set_user_attr("cv_failed", msg[:200])
                except Exception:
                    pass
                mean_score = _bad_obj(direction)

            else:
                # Other RuntimeErrors are still unexpected -> bubble up.
                raise

        # Normalize CV result -> one float; prune if not finite (no-valid-blocks / gating)
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
        # Quality Guard #1: Cap per-trial Sharpe at +/-6.0
        # Annualized Sharpe > 6 is unrealistic and usually from too few trades.
        # This prevents lucky low-trade folds from dominating the objective.
        # ------------------------------------------------------------------
        _SHARPE_CAP = 6.0
        if abs(score) > _SHARPE_CAP:
            try:
                trial.set_user_attr("sharpe_capped", f"{score:.3f}->{_SHARPE_CAP if score > 0 else -_SHARPE_CAP:.3f}")
            except Exception:
                pass
            score = _np.clip(score, -_SHARPE_CAP, _SHARPE_CAP)

        # ------------------------------------------------------------------
        # Quality Guard #2: Minimum total trades floor across valid folds
        # If the config produces < 40 trades total, it's not trading enough
        # to be statistically meaningful -> heavy penalty.
        # ------------------------------------------------------------------
        _MIN_TOTAL_TRADES = 40
        _trades_cv = None
        try:
            _trades_cv = trial.user_attrs.get("trades_cv", None)
            if _trades_cv is None:
                _trades_cv = trial.user_attrs.get("trades_valid", None)
        except Exception:
            pass
        if _trades_cv is not None:
            try:
                _total_trades = float(_trades_cv)
            except Exception:
                _total_trades = 0
            if _total_trades < _MIN_TOTAL_TRADES:
                _trade_penalty = 2.0 * (_MIN_TOTAL_TRADES - _total_trades) / _MIN_TOTAL_TRADES
                score = score - _trade_penalty
                try:
                    trial.set_user_attr("min_trades_penalty", f"trades={_total_trades:.0f}<{_MIN_TOTAL_TRADES} pen={_trade_penalty:.3f}")
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Quality Guard #3: Penalize excessive active rate (> 0.30)
        # An active rate >> target (0.15) means the confidence threshold
        # is miscalibrated -> penalty proportional to the overshoot.
        # ------------------------------------------------------------------
        _ACTIVE_RATE_CAP = 0.30
        _ar_cv = None
        try:
            _ar_cv = trial.user_attrs.get("active_rate_cv", None)
        except Exception:
            pass
        if _ar_cv is not None:
            try:
                _ar = float(_ar_cv)
            except Exception:
                _ar = 0.0
            if _ar > _ACTIVE_RATE_CAP:
                _ar_penalty = 3.0 * (_ar - _ACTIVE_RATE_CAP) / _ACTIVE_RATE_CAP
                score = score - _ar_penalty
                try:
                    trial.set_user_attr("active_rate_penalty", f"ar={_ar:.3f}>{_ACTIVE_RATE_CAP} pen={_ar_penalty:.3f}")
                except Exception:
                    pass

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
                log_print(f"[WARN] [Select] calibration penalty skipped: {_sel_e}", level="DEBUG")
                
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
                level="DEBUG"
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
                log_print(f"[ALERT] [{msg}]", level="COMPACT")
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

        log_print("\n" + "#" * 80, level="DEBUG")
        log_print(f"[WARN] Error in optuna_objective(): {cause}", level="DEBUG")
        if 'params' in locals():
            log_print(f"Trial params were: {params}", level="DEBUG")

        # Keep the stack trace always visible; it's rare but important.
        traceback.print_exc()
        log_print("#" * 80 + "\n", level="DEBUG")

        if DISABLE_OPTUNA_PRUNING:
            return _bad_obj(direction)
        else:
            raise optuna.TrialPruned(f"Trial pruned due to error -> {cause}")

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




