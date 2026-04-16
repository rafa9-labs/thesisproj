"""
Metrics tuple construction and validation.

Extracted from MLBacktesterNoWFO.py lines 920-1459.
"""

import numpy as np
from copy import deepcopy
from utilsNoWFO import (
    N_METRICS, ensure_metric_tuple, validate_metrics_shape,
)

def _empty_metrics(context: str = "") -> tuple:
    """
    Return a shape-correct metric tuple filled with NaNs.

    Used for:
    - invalid folds,
    - failed evaluations,
    - situations where we want to signal "no usable metrics" but
      keep the schema stable to avoid shape errors downstream.
    """
    raw = [np.nan] * N_METRICS
    metrics = ensure_metric_tuple(raw)
    # context helps debug which path produced empty metrics
    try:
        validate_metrics_shape(metrics, context=context or "empty_metrics")
    except Exception:
        # If even validation fails, fall back to a plain tuple (still correct length)
        metrics = tuple(raw)
    return metrics


def _safe_metrics_return(raw_metrics, context: str = "") -> tuple:
    """
    Strict contract enforcement: fail fast on metric arity drift (prevents silent corruption).
    """
    # Validate WITHOUT coercion (raises on mismatch)
    validate_metrics_shape(raw_metrics, context=context or "evaluation")
    # Safe to cast now (length already proven correct)
    return ensure_metric_tuple(raw_metrics)


# -----------------------------------------------------------------------------
# Output toggles: control which per-model artifacts are written.
# -----------------------------------------------------------------------------
SAVE_TRADES = {
    # Monthly summary of trades per rep
    # (1 row per month, per rep) → <run>/repetition_k/<Model>/csv/monthly_trade_summary_repK.csv
    "monthly_summary_per_rep_csv": True,

    # Per-trade BH vs model comparison (entry/exit)
    # → <run>/repetition_k/<Model>/csv/trade_entry_exit_compare_repK.csv
    "trade_entry_exit_compare_csv": True,

    # Reserved for future use:
    # "per_trade_month_csv": False,
    # "rep_summary_csv": False,
}

SAVE_EQUITY = {
    # Per-month equity PNG for each valid month of a given rep
    # → <Model>/graphs/monthly_equity_k.png
    # Disabled by default; enable explicitly when needed.
    "per_month_equity_png": True,

    # Mean equity over reps (full horizon) per model
    # (run-level mean curves; currently unused)
    "mean_equity_over_reps": True,
}

SAVE_METRICS = {
    # Per-month metrics CSV written during wrap-up
    # → <Model>/Months/k/csv/csv_month_k.csv
    # Disabled by default; enable explicitly when needed.
    "per_month_metrics_csv": True,

    # Aggregated monthly results:
    # → <RUN_DIR>/model_stats/monthly_results_all_<model>.csv
    "monthly_results_all_csv": True,

    # Split by rep:
    # → <RUN_DIR>/repetition_k/<Model>/csv/monthly_results_rep<k>_<model>.csv
    "monthly_results_per_rep_csv": True,
}

SAVE_FEATURES = {
    # Per-month feature heatmap:
    # → <Model>/Months/k/heatmaps/feature_heatmap_k.png
    "monthly_heatmap_png": False,

    # Features/config text dump:
    # → <Model>/Months/k/csv/featuresconfigused_k.txt
    "featuresconfig_txt": False,  # can turn on if you want the heavy dumps
}

# ---------------------------------------------------------------------
# Global defaults (single source of truth) — module-level
# ---------------------------------------------------------------------
CLASS_DEFAULTS = {
    "features": {
        # --- Sessions & leakage control (AFML-consistent) ---
        "session_filter_mode": "both",
        "session_filter_on_train": True,
        "final_embargo_bars": 0,
        "enforce_day1_start": True,

        # --- Feature pipeline / lags ---
        "lag_depth": 1,
        "roll_windows": [5, 10, 30, 60],
        "include_hour": True,
        "include_hour_cyclic": True,
        
        # Table prints
        "eval_print_causality_debug": False,

        # --- Feature slice cache (per-run df_out cache) ---
        "slice_cache_enabled": False,

        # --- Realized-volatility features (short/long windows) ---
        "use_rv_features": True,
        "rv_window_short": 48,
        "rv_window_long": 240,

        # --- Indicator state features (oscillators & volatility regimes) ---
        "use_indicator_states": False,
        "rsi_overbought_level": 70,
        "rsi_oversold_level": 30,
        "stoch_overbought_level": 80,
        "stoch_oversold_level": 20,
        "bbw_compress_threshold": 0.05,
        "bbw_expand_threshold": 0.20,

        # --- Donchian-style price channels / breakouts ---
        "use_donchian": True,
        "donchian_window_short": 20,
        "donchian_window_long": 60,

        # --- Fractional differencing (AFML-style) ---
        "use_fracdiff": False,
        "fracdiff_d": 0.5,

        # --- MTF housekeeping ---
        "mtf_fillna_method": "ffill",

        # --- Canonical indicators (toggled by strategies) ---
        "use_rsi": False,
        "use_macd": False,
        "use_ema": False,
        "use_adx": False,
        "use_bbands": False,
        "use_stoch": False,
        "use_atr": False,
        "use_mtf_ma": False,

        # --- Indicator windows (standard TA defaults) ---
        "indicator_windows": {
            "sma": 20,
            "ema": 20,
            "rsi": 14,
            "atr": 14,
            "adx": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_window": 20,
            "bb_dev": 2.0,
            "stoch_k": 14,
            "stoch_d": 3,
            "mtf_ma_fast_window": 10,
            "mtf_ma_slow_window": 50,
        },

        # --- Macro feature hook (daily / lower-frequency) ---
        "use_macro_features": False,
        "macro_sources": {},
        "macro_lag_days": 1,

        # --- Global HPO policy ---
        "tune_once": True,

        # --- Deep / windowing knobs ---
        "cnn_use_seq_windows": True,
        "lstm_use_seq_windows": True,
        "transformer_train_stride": 1,
        "cnn_train_stride": 1,
        "lstm_train_stride": 1,
        "deep_max_train_windows": 30000,

        # --- Runtime coverage calibration mode ---
        "runtime_coverage_mode": "rolling_quantile",

        # --- Spread/slippage protection baseline ---
        "eval_use_spread_guard": True,
        "eval_spread_cap": 0.00040,
        "slippage_factor": 1.0,
        "eval_impact_eta": 0.0,

        # --- Position sizing (Sprint 2 — S2.1) ---
        "sizing_method": "fixed",
        "sizing_risk_fraction": 0.02,
        "sizing_kelly_fraction": 0.5,
        "sizing_kelly_min_trades": 10,
        "sizing_atr_risk_pct": 0.02,
        "sizing_atr_sl_mult": 2.0,
        "sizing_initial_equity": 10000.0,
        "sizing_max_leverage": 5.0,

        # --- Stop-loss / take-profit (Sprint 2 — S2.2) ---
        "stop_method": "none",
        "stop_sl_pips": 30.0,
        "stop_tp_pips": 60.0,
        "stop_sl_atr_mult": 2.0,
        "stop_tp_atr_mult": 3.0,
        "stop_sl_sigma_mult": 2.0,
        "stop_tp_sigma_mult": 3.0,
        "stop_pip_value": 0.0001,
        "stop_use_be": False,
        "stop_be_trigger_pips": 20.0,
        "stop_use_partial_close": False,
        "stop_tp1_ratio": 0.5,
        "stop_tp1_pips": 30.0,
        "stop_tp2_pips": 0.0,

        # --- Trailing stops (Sprint 2 — S2.3) ---
        "trailing_method": "none",
        "trailing_pips": 30.0,
        "trailing_atr_mult": 3.0,
        "trailing_chandelier_atr_mult": 3.0,
        "trailing_chandelier_lookback": 22,
        "trailing_activation_pips": 10.0,
        "trailing_pip_value": 0.0001,

        # --- Risk management (Sprint 2 — S2.4) ---
        "risk_use_dd_breaker": False,
        "risk_max_drawdown_pct": 0.20,
        "risk_dd_resume": "session_end",
        "risk_dd_cooloff_bars": 48,
        "risk_use_daily_loss": False,
        "risk_max_daily_loss_pct": 0.03,
        "risk_max_daily_loss_sigma": 3.0,
        "risk_daily_loss_mode": "pct",
        "risk_use_consec_loss": False,
        "risk_max_consecutive_losses": 5,
        "risk_consec_resume": "session_end",
        "risk_consec_cooloff_bars": 48,
        "risk_initial_equity": 10000.0,
        "risk_max_open_positions": 1,

        # --- CV-time caps for keras models (early stopping regime) ---
        "deep_cv_max_epochs": 12,
        "deep_cv_batch_size": 256,
        "deep_cv_patience": 6,

        # --- Per-model CV caps (multi-fidelity HPO) ---
        "cnn_cv_max_epochs": 6,
        "cnn_cv_batch_size": 256,
        "cnn_cv_train_stride": 3,
        "cnn_cv_max_train_windows": 4000,

        "lstm_cv_max_epochs": 8,
        "lstm_cv_batch_size": 256,
        "lstm_cv_train_stride": 2,
        "lstm_cv_max_train_windows": 5000,

        "transformer_cv_max_epochs": 6,
        "transformer_cv_batch_size": 256,
        "transformer_cv_train_stride": 2,
        "transformer_cv_max_train_windows": 7000,

        # --- Extra-strict CV caps for ensemble deep heads (CNN/LSTM inside ensemble) ---
        "cnn_ens_cv_max_epochs": 6,
        "lstm_ens_cv_max_epochs": 8,
        "cnn_ens_cv_max_train_windows": 4000,
        "lstm_ens_cv_max_train_windows": 5000,

        # --- Probability calibration (classical + deep) ---
        "calibrate_method": "isotonic",
        "deep_calibrate": True,
        "deep_calibration_method": "temperature",
        "deep_calibration_frac": 0.10,
        "deep_calibration_min_samples": 500,

        # --- Trade gating — PSR/DSR-based reliability filters ---
        "gating_mode": "bets_psr",

        # --- Final experiment coverage policy (fixed; comparable across models) ---
        "target_active_rate": 0.15,
        "runtime_active_band_margin": 0.08,
        "runtime_conf_nudge": 0.005,
        "runtime_coverage_window": 192,


        # Fallback floor when coverage intent is OFF.
        # When coverage intent is ON (target_active_rate/target_coverage > 0),
        # train-anchored coverage calibration overrides this.
        "confidence_threshold": 0.80,

        # Never "force trades" by lowering conf_thr unless explicitly enabled.
        "allow_conf_backoff_cv": False,
        "allow_conf_backoff_eval": False,

        # Real-sim bump applied on top of target_active_rate (opt-in only)
        "real_sim_target_active_mult": 1.00,
        "real_sim_target_active_cap": 0.25,
        "allow_real_sim_target_active_mult": False,

        # --- Reliability / pruning helpers ---
        "min_trades_per_block": 0,
        "min_independent_bets": 20,
        "psr_alpha": 0.24,
        "dsr_prune": True,
        "floor_cv_final": -6.0,

        # --- Per-model confidence tweaks (architectural bias) ---
        "lstm_conf_relax": 0.15,
        "lstm_conf_floor": 0.40,

        # --- Labeling guards & triple-barrier events ---
        "use_triple_barrier": True,
        "tb_pt_mult": 2.0,
        "tb_sl_mult": 2.0,
        "tb_max_holding": 48,
        "tb_neutral_zone": 1.0,
        "tb_neutral_zone_is_sigma": True,
        "print_labeling_debug": False,

        # --- Prefilter / stability selection helpers ---
        "use_prefilter": True,
        "prefilter_min_unique_frac": 0.005,
        "prefilter_min_std": 1e-6,
        "prefilter_max_corr": 0.96,
        "prefilter_prefer_prefixes": ["rv", "ema", "sma", "macd", "adx"],
        "mutual_info_top_k": "sqrt",
        "prefilter_random_state": 42,

        # --- Ensemble throttles & fusion ---
        "ensemble_train_stride": 1,
        "ensemble_deep_max_train_windows": 15000,
        "fusion_alpha": 0.6,

        # --- Regime threshold defaults for AdaptiveRegimeStrategy ---
        "adx_thresh_q": 0.70,

        # --- Reporting / artifact controls ---
        "save_monthly_equity_plots": True,
        "save_monthly_feature_heatmaps": False,

        # --- Dynamic edge-vs-cost gating coefficients ---
        "alpha_vol_z": 0.004,
        "beta_spread_norm": 0.008,
        "gamma_slip_norm": 0.004,
        "slip_norm_bps": 0.25,
        "min_slip_norm_bps": 0.05,
        "vol_z_cap": 6.0,
        "spread_norm_cap": 5.0,
        "slip_ratio_cap": 6.0,
        "min_conf_thr": 0.33,
        "max_conf_thr": 0.90,
        "min_conf_thr_cov": 0.0,
        "max_conf_thr_cov": 0.90,

        # --- Top-N consensus & meta-analysis (runtime) ---
        "deploy_topN_consensus": False,

        # IMPORTANT: turn this OFF for the consensus experiment so results are not “adaptive mode”
        # (adaptive top3 can switch behavior / fall back and muddy your thesis comparison)
        "use_adaptive_top3_for_main_results": False,

        "topN_classical": 3,
        "topN_deep": 3,
        "topN_ensemble": 3,
        "topN_default": 3,

        "consensus_pool_max_trials": 0,
        "topN_style_lock": False,

        # Make it accept your Top-3 even if #2/#3 are worse than #1
        "topN_min_perf_frac": 0.00,

        # Make “geometry similarity” basically never reject members (Top-3 only -> don’t over-filter)
        "topN_geom_radius": 9.0,

        # Keep the tolerances (they won’t matter much once geom_radius is huge, but harmless)
        "topN_lags_tol": 4.0,
        "topN_depth_tol": 1.0,
        "topN_target_tol": 0.05,

        # Make correlation filter basically never drop a member
        "topN_max_corr": 0.9999,

        "print_topN_debug": True,

        "deploy_param_heatmaps": False,
        "topN_for_heatmaps": 5,
        "deploy_feature_freq": False,
        "top_feature_percent": 1.0,

        # --- Evaluation / risk / execution defaults ---
        "eval_use_vol_target": True,
        "eval_vol_target_ann": 0.10,
        "eval_vol_floor": 1e-6,
        "eval_vol_lookback": 96,
        "eval_max_leverage": 1.5,

        "eval_use_scaleout_trail": True,
        "eval_tp1_z": 1.5,
        "eval_trail_k": 3.0,
        "eval_trail_dynamic_vol": True,
        "eval_move_stop_to_be": True,
        "eval_max_holding_bars": 0,

        # Reduce hyper-churn: block flips for the first N bars after entry (exits-to-flat allowed).
        "eval_min_holding_bars": 3,

        "eval_use_twap_execution": True,
        "eval_twap_span_bars": 2,
        "eval_twap_freeze_size_at_entry": True,

        "eval_use_regime_adaptive": True,
        "eval_regime_source": "sigma",
        "eval_regime_q_low": 0.33,
        "eval_regime_q_high": 0.66,
        "eval_tp1_z_calm": 1.2,
        "eval_tp1_z_normal": 1.5,
        "eval_tp1_z_volatile": 1.8,
        "eval_trail_k_calm": 2.5,
        "eval_trail_k_normal": 3.0,
        "eval_trail_k_volatile": 3.5,
        "eval_print_regime_debug": False,
        "eval_print_trail_debug": False,
        "eval_twap_print_debug": False,

        "eval_use_kill_switch": True,
        "eval_kill_mode": "sigma",
        "eval_kill_limit_pct": 0.02,
        "eval_kill_sigma": 3.0,
        "eval_kill_until_session_end": True,
        "eval_cooloff_bars": 30,
        "eval_kill_min_limit_pct": 0.005,
        "eval_kill_min_sigma": 1.0,
        "eval_kill_max_sigma": 6.0,
        "eval_kill_max_cooloff_bars": 480,
        "eval_kill_print_debug": False,

        # --- Output / plotting profile ---
        "output_profile": "thesis",
        "light_output": False,

        "enable_pbo_mcs_analysis": False,

        "allow_param_fallback": False,
        "min_trades_for_wfo": 0,

        # --- Regime features ---
        "use_regime_features": True,
        "regime_num_states": 3,
        "regime_trend_quantile": 0.7,
        "regime_vol_high_quantile": 0.7,
        "regime_vol_low_quantile": 0.4,
        "regime_vol_window": 20,

        # --- DQN reward shaping ---
        "env_cost_scale_dqn": 1.0,
        "env_turnover_penalty_dqn": 0.0002,
    },

    "cv": {
        
        "use_cached_global_hpo": True,
        "n_trials": 0,
        
        # --- CV geometry ---
        "cv_mode": "mini_block",
        "cv_blocks": 5,
        "cv_min_train_frac": 0.75,
        "cv_val_frac": 0.05,
        "cv_embargo_bars": 0,
        "cv_embargo_frac": 0.01,
        "cv_fit_blocks_exact": True,
        "cv_tail_anchor": True,

        # --- Monthly-roll legacy knobs kept for compatibility ---
        "cv_target_folds": 5,
        "cv_val_months": 1.0,
        "cv_train_months": None,
        "bars_per_month_hint": 1000,
        "cv_sliding_stride_frac": None,

        # --- Fold aggregation / robustness ---
        "cv_fold_aggregator": "ivw_sharpe_capped",
        "cv_sr_cap": 4.0,
        "cv_sr_var_floor": 0.75,
        "cv_weight_blend_neff": 0.30,
        "cv_neff_mode": "trades",
        "cv_min_eff_n": 0.0,
        "cv_tail_weight": 1.00,
        "cv_z_weights": "sqrt_n",
        "cv_z_cap": 8.0,
        "cv_sr_ref": 0.0,
        "cv_huber_delta": 1.50,
        "cv_catoni_alpha": 0.50,
        "cv_std_penalty": 0.0,
        "cv_coverage_gamma": 1.50,

        # --- CV validity gates (stop 1-fold gaming) ---
        "cv_min_coverage": 0.80,
        "cv_min_valid_fraction": 0.80,
        "cv_prune_on_low_valid_fraction": True,

        # --- reliability / activity gates ---
        "cv_min_trades_per_block": 30,
        "cv_min_indep_bets_per_block": 12,
        "cv_gate_min_folds": 4,
        "cv_gate_min_active_rate": 0.02,
        "cv_gate_min_sr": 0.00,

        # --- active-rate hygiene ---
        "cv_min_active_rate": 0.005,
        "cv_active_rate_low": 0.00,
        "cv_active_rate_high": 1.00,
        "cv_active_rate_margin": 0.03,
        "cv_low_active_lambda": 0.25,

        # Soft penalties for missing active-rate band in CV
        "cv_soft_active_low_lambda": 1.0,
        "cv_soft_active_high_lambda": 1.0,

        # Soft penalties for turnover outside family band
        "cv_turnover_low": None,
        "cv_turnover_high": None,
        "cv_turnover_low_lambda": 1.0,
        "cv_turnover_high_lambda": 2.0,
        "cv_trade_shortfall_lambda": 0.0,

        # --- numeric stability ---
        "cv_min_volatility": 1e-6,

        "cv_invalid_share_penalty": 5.0,

        # --- Evaluation cost model knobs (CV) ---
        "eval_use_trading_costs": False,
        "eval_spread_pips": 0.8,
        "eval_slip_mode": "tworegime",
        "eval_slip_bps_lo": 0.08,
        "eval_slip_bps_med": 0.16,
        "eval_slip_bps_hi": 0.30,
        "vol_window_bars": 96,
        "high_vol_q": 0.85,
        "high_vol_conf_bump": 0.0,

        "turnover_penalty_lambda": 0.1,

        # --- Debug/log controls ---
        "print_cv_debug": False,
        "print_cv_fold_scores": False,
        "cv_log_precision": 8,
        "cv_use_psr_trim": False,

        # --- Pruning controls ---
        "prune_min_folds": 3,
        "prune_iqr_mult": 1.0125,
        "prune_abs_floor_sr": -8.0,

        # --- Trade cap controls ---
        "cv_dynamic_trades_cap_frac": 0.675,
        "cv_max_trades_per_block": 500,

        # --- Alternative aggregation knobs (kept for compatibility) ---
        "cv_agg_mode": "tanh_mean",
        "cv_tanh_s": 10.0,
        "cv_trim_frac": 0.20,
        "cv_psr_power": 1.0,
        "cv_use_recency_weight": False,
        "cv_recency_power": 1.0,

        # --- CSCV / PBO-related knobs (kept for compatibility) ---
        "cv_cscv_penalty_weight": 0.30,
        "cv_cscv_min_rank_corr": 0.20,
        "cv_cscv_disqualify": False,
        "cv_strict_pruning": False,
        "cv_prune_relax": 0.50,

        "cv_prune_precision_intent": False,

        # --- Optuna plateau stopping ---
        "plateau_min_trials": 20,
        "plateau_patience": 15,
        "plateau_delta": 0.02,

        # --- Disable extra stages: mini-fold → consensus → real trading (only) ---
        "robustness_eval": False,
        "robust_seeds": [1111, 2222, 3333],
        "robust_require_pass": False,
        "verify_topn_monthly_roll": False,
    },
}

# Convenience mirrors to avoid NameError and accidental mutation
DEFAULT_FEATURES = deepcopy(CLASS_DEFAULTS["features"])
DEFAULT_CV       = deepcopy(CLASS_DEFAULTS["cv"])


