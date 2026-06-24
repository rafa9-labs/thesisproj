"""Hyperparameter sampling: sample_param_set and TA profile application."""
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

from config import PIPELINE_CONSTANTS as _PC, SEARCH_SPACE, CV_SEARCH_SPACE
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
    _bad_obj,
    _record_hp_boundary_hit,
    _ta_profile_sanity,
    _apply_ta_profile_legacy,
    _apply_ta_profile_ungated,
    _apply_ta_profile_fixed,
    _apply_ta_profile_tuned,
    DISABLE_OPTUNA_PRUNING,
    MLB_TA_MODE,
)

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
            "calibrate_method", ["sigmoid"]
        )

        deep_models = {"lstm", "cnn", "transformer", "gru", "gru_lstm"}
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
        params["alpha_vol_z"]          = _PC["alpha_vol_z"]
        params["beta_spread_norm"]     = _PC["beta_spread_norm"]
        params["gamma_slip_norm"]      = _PC["gamma_slip_norm"]
        params["slip_norm_bps"]        = float(params.get("slip_norm_bps", _PC["slip_norm_bps"]) or _PC["slip_norm_bps"])
        params["vol_window_bars"]      = _PC["vol_window_bars"]
        params["high_vol_q"]           = _PC["high_vol_q"]
        params["high_vol_conf_bump"]   = _PC["high_vol_conf_bump"]
        params["runtime_active_band_margin"] = _PC["runtime_active_band_margin"]
        params["runtime_conf_nudge"]         = _PC["runtime_conf_nudge"]
        params["runtime_coverage_window"]    = _PC["runtime_coverage_window"]
        
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
    if (str(model_type).startswith("ensemble_") or
        str(model_type).lower() in {"stacking_ensemble", "meta_ensemble"}):
        l_lo, l_hi = 8, 24
    else:
        l_lo, l_hi = 12, 40
    params["lags_range"] = trial.suggest_int("lags_range", l_lo, l_hi)
    
    # Allow deeper lag depth for ensembles (otherwise ensemble lag knobs are pointless)
    if (str(model_type).startswith("ensemble_") or
        str(model_type).lower() in {"stacking_ensemble", "meta_ensemble"}):
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

    # === Volatility-scaled label threshold (k * sigma) ===
    sigma = None

    # 1) Prefer precomputed sigma from run_optuna_tuning (cheap, reused every trial)
    if isinstance(vol_stats, dict) and "sigma48" in vol_stats:
        try:
            sigma = float(vol_stats["sigma48"])
        except (TypeError, ValueError):
            sigma = None

    # 2) Fallback: compute sigma from train_data only if not provided
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

    # 3) Use sigma if we have it, otherwise fall back to generic prior
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
    ta_mode = os.environ.get("MLB_TA_MODE", "").strip().lower() or MLB_TA_MODE or "legacy"
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
    # === Labeling (triple-barrier) -- locked ON, practitioner-friendly ranges ===
    params["use_triple_barrier"] = True

    # Tight stops + long holding horizons collapse the neutral class (timeouts),
    # causing 3-class folds to become effectively binary. Constrain TB so class=1
    # exists consistently across mini-block folds.
    tb_pt_low, tb_pt_high = 1.00, 2.00
    tb_sl_low, tb_sl_high = 1.00, 2.00
    tb_hold_low, tb_hold_high = 24, 48
    # Neutral band: multiplier on local sigma; applies only on timeout.
    tb_nz_low, tb_nz_high = 0.25, 0.75
 

    params["tb_pt_mult"] = trial.suggest_float("tb_pt_mult", tb_pt_low, tb_pt_high, step=0.25)
    _record_hp_boundary_hit("tb_pt_mult", params["tb_pt_mult"], tb_pt_low, tb_pt_high)

    params["tb_sl_mult"] = trial.suggest_float("tb_sl_mult", tb_sl_low, tb_sl_high, step=0.25)
    _record_hp_boundary_hit("tb_sl_mult", params["tb_sl_mult"], tb_sl_low, tb_sl_high)

    params["tb_max_holding"] = trial.suggest_int("tb_max_holding", tb_hold_low, tb_hold_high, step=12)
    _record_hp_boundary_hit("tb_max_holding", params["tb_max_holding"], tb_hold_low, tb_hold_high)

    params["tb_neutral_zone"] = trial.suggest_float("tb_neutral_zone", tb_nz_low, tb_nz_high, step=0.25)
    _record_hp_boundary_hit("tb_neutral_zone", params["tb_neutral_zone"], tb_nz_low, tb_nz_high)

    # === Calibration method -- probabilistic heads (STABLE CHOICES) ===
    # Use "" for "no calibration" to keep Optuna's distribution stable across the study.
    if _hpo_stage == "A_signal":
         params["calibrate_method"] = str(_stage_cfg.get("stageA_calibrate_method", "sigmoid") or "sigmoid")
    else:
        params["calibrate_method"] = trial.suggest_categorical("calibrate_method", ["sigmoid"])

    # === Feature engineering toggles ===
    params["use_fracdiff"]     = trial.suggest_categorical("use_fracdiff", [False, True])
    # P4: ADF floor prevents sub-stationarity d values (de Prado AFML Ch.5)
    if params["use_fracdiff"]:
        d_floor = 0.4
        if train_data is not None and hasattr(train_data, "columns"):
            price_col = None
            for candidate in ("mid_c", "price", "close"):
                if candidate in train_data.columns:
                    price_col = candidate
                    break
            if price_col is not None:
                try:
                    from pipeline.features.feature_utils import find_min_stationary_d
                    d_floor = find_min_stationary_d(train_data[price_col])
                except Exception:
                    pass
        params["fracdiff_d"] = trial.suggest_float("fracdiff_d", max(0.1, d_floor), 0.9, step=0.05)
    else:
        params["fracdiff_d"] = 0.0
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
        # LSTM: most relaxed -> highest target coverage, softest alpha/beta/gamma,
        # widest band, strongest nudge.
        # +0.12 shift on the coverage range compared to the base (0.20-0.40 -> 0.32-0.52)
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
        high_vol_bump_max   = 0.03  # do not stack too aggressively with alpha*vol_z

    elif mt in {"logistic", "lightgbm", "catboost"}:
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
        # Default for all other models (gru, gru_lstm, cnn, transformer, svm, rf, dt, xgboost, ensembles).
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
    params["target_active_rate"]       = float(target_coverage_policy(mt))
    params["target_coverage"]          = params["target_active_rate"]
    params["alpha_vol_z"]              = _PC["alpha_vol_z"]
    params["beta_spread_norm"]         = _PC["beta_spread_norm"]
    params["gamma_slip_norm"]          = _PC["gamma_slip_norm"]
    params["slip_norm_bps"]            = _PC["slip_norm_bps"]
    params["vol_window_bars"]          = _PC["vol_window_bars"]
    params["high_vol_q"]               = _PC["high_vol_q"]
    params["high_vol_conf_bump"]       = _PC["high_vol_conf_bump"]
    params["runtime_active_band_margin"] = _PC["runtime_active_band_margin"]
    params["runtime_conf_nudge"]       = _PC["runtime_conf_nudge"]
    params["runtime_coverage_window"]  = _PC["runtime_coverage_window"]

    # Keep backoff flags off under coverage-anchored regime.
    params.setdefault("allow_conf_backoff_cv",   False)
    params.setdefault("allow_conf_backoff_eval", False)

    # Ensure after-cost evaluation during CV (you can override this if you want raw PnL).
    params["eval_use_trading_costs"] = True


    # === Model-specific hyperparameter space (driven by SEARCH_SPACE) ===
    _MODEL_PREFIX = {
        "svm": "svm", "logistic": "logit", "xgboost": "xgb",
        "random_forest": "rf", "decision_tree": "dt",
        "lstm": "lstm", "cnn": "cnn", "transformer": "transformer",
        "gru": "gru", "gru_lstm": "gru_lstm",
        "lightgbm": "lightgbm", "catboost": "catboost",
        "stacking_ensemble": "stack", "meta_ensemble": "meta",
        "ensemble_adaptive_regime": "", "ensemble_cnn_lstm_xgboost": "",
    }

    def _suggest_from_search_space(trial, model_name):
        specs = SEARCH_SPACE.get(model_name, {})
        prefix = _MODEL_PREFIX.get(model_name, model_name)
        for key, spec in specs.items():
            param_key = f"{prefix}_{key}" if prefix else key
            if isinstance(spec, list):
                params[param_key] = trial.suggest_categorical(param_key, spec)
            elif isinstance(spec, tuple):
                if len(spec) == 3 and isinstance(spec[2], bool):
                    params[param_key] = trial.suggest_float(param_key, spec[0], spec[1], log=spec[2])
                elif len(spec) == 3 and isinstance(spec[2], int):
                    params[param_key] = trial.suggest_int(param_key, spec[0], spec[1], step=spec[2])
                else:
                    params[param_key] = trial.suggest_float(param_key, spec[0], spec[1])
            else:
                params[param_key] = spec  # fixed value

        # Sample global CV geometry params (no model prefix — shared across models)
        for key, spec in CV_SEARCH_SPACE.items():
            if isinstance(spec, list):
                params[key] = trial.suggest_categorical(key, spec)
            elif isinstance(spec, tuple):
                if len(spec) == 3 and isinstance(spec[2], bool):
                    params[key] = trial.suggest_float(key, spec[0], spec[1], log=spec[2])
                elif len(spec) == 3 and isinstance(spec[2], int):
                    params[key] = trial.suggest_int(key, spec[0], spec[1], step=spec[2])
                else:
                    params[key] = trial.suggest_float(key, spec[0], spec[1])
            else:
                params[key] = spec

    # ensemble_adaptive_regime: TA profile must run BEFORE suggest (sets indicator toggles)
    if model_type == "ensemble_adaptive_regime":
        _apply_ta_profile_fixed(trial, params)
        _suggest_from_search_space(trial, model_type)
    else:
        _suggest_from_search_space(trial, model_type)

    # === Model-specific fixed defaults (not in SEARCH_SPACE) ===
    if model_type == "svm":
        params["svm_kernel"] = "rbf"
        params["svm_class_weight"] = "balanced"
    elif model_type == "logistic":
        params["logit_max_iter"] = 500
        params["logit_tol"] = 0.0001
    elif model_type == "xgboost":
        params["xgb_gamma"] = 0.0
        params["xgb_min_child_weight"] = 1
        params["xgb_reg_lambda"] = 1.0
        params["xgb_reg_alpha"] = 0.0
        params["xgb_device"] = "cuda"
    elif model_type == "lightgbm":
        params["lightgbm_boosting_type"] = "gbdt"
        params["lightgbm_min_child_samples"] = 20
    elif model_type == "catboost":
        params["catboost_border_count"] = 128
        params["catboost_loss_function"] = "MultiClass"
    elif model_type == "random_forest":
        params["rf_bootstrap"] = True
        params["rf_class_weight"] = "balanced"
        params["rf_n_jobs"] = -1
    elif model_type == "decision_tree":
        params["dt_class_weight"] = "balanced"
    elif model_type in ("lstm", "cnn", "gru"):
        params[f"{model_type}_dense_units"] = 64
        params[f"{model_type}_batch_size"] = 256
        params[f"{model_type}_use_seq_windows"] = False
        if model_type in ("lstm", "gru"):
            params[f"{model_type}_bidirectional"] = False
            params[f"{model_type}_clipnorm"] = 1.0
    elif model_type == "gru_lstm":
        params["gru_lstm_dense_units"] = 64
        params["gru_lstm_batch_size"] = 256
    elif model_type == "transformer":
        params["transformer_num_blocks"] = 1
        params["transformer_ff_multiple"] = 2
        params["transformer_dense_units"] = 128
        params["transformer_pooling"] = "cls"
        params["transformer_use_time2vec"] = False
        params["transformer_batch_size"] = 256
    elif model_type == "cnn":
        params["cnn_dropout"] = 0.3

    # === Runtime guards for deep models (no wall-clock limits, just shapes) ===
    # LSTM/CNN/GRU: if we use sequence windows, do not allow stride=1 (window explosion).
    if model_type in ("lstm", "cnn", "gru", "gru_lstm"):
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


