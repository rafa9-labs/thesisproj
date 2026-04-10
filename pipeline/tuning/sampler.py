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
            "calibrate_method", ["sigmoid", "isotonic"]
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
    tb_pt_low, tb_pt_high = 1.00, 2.00
    tb_sl_low, tb_sl_high = 1.00, 2.00
    tb_hold_low, tb_hold_high = 24, 48
    # Neutral band: multiplier on local σ; applies only on timeout.
    tb_nz_low, tb_nz_high = 0.25, 0.75
 

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
         params["calibrate_method"] = str(_stage_cfg.get("stageA_calibrate_method", "sigmoid") or "sigmoid")
    else:
        params["calibrate_method"] = trial.suggest_categorical("calibrate_method", ["sigmoid", "isotonic"])

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


