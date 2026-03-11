"""
Configuration management using Pydantic models.
Extracts CLASS_DEFAULTS and Oanda credentials into validated, serializable config.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from copy import deepcopy


class IndicatorWindows(BaseModel):
    """Technical indicator window parameters"""
    sma: int = 20
    ema: int = 20
    rsi: int = 14
    atr: int = 14
    adx: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_dev: float = 2.0
    stoch_k: int = 14
    stoch_d: int = 3
    mtf_ma_fast_window: int = 10
    mtf_ma_slow_window: int = 50


class FeaturesConfig(BaseModel):
    """Feature engineering and indicator configuration"""
    
    # Sessions & leakage control
    session_filter_mode: str = "both"
    session_filter_on_train: bool = True
    final_embargo_bars: int = 0
    enforce_day1_start: bool = True
    
    # Feature pipeline / lags
    lag_depth: int = 1
    roll_windows: List[int] = Field(default_factory=lambda: [5, 10, 30, 60])
    include_hour: bool = True
    include_hour_cyclic: bool = True
    
    # Table prints
    eval_print_causality_debug: bool = False
    
    # Feature slice cache
    slice_cache_enabled: bool = False
    
    # Realized-volatility features
    use_rv_features: bool = True
    rv_window_short: int = 48
    rv_window_long: int = 240
    
    # Indicator state features
    use_indicator_states: bool = False
    rsi_overbought_level: int = 70
    rsi_oversold_level: int = 30
    stoch_overbought_level: int = 80
    stoch_oversold_level: int = 20
    bbw_compress_threshold: float = 0.05
    bbw_expand_threshold: float = 0.20
    
    # Donchian-style price channels
    use_donchian: bool = True
    donchian_window_short: int = 20
    donchian_window_long: int = 60
    
    # Fractional differencing
    use_fracdiff: bool = False
    fracdiff_d: float = 0.5
    
    # MTF housekeeping
    mtf_fillna_method: str = "ffill"
    
    # Canonical indicators
    use_sma: bool = False
    use_rsi: bool = False
    use_macd: bool = False
    use_ema: bool = False
    use_adx: bool = False
    use_bbands: bool = False
    use_stoch: bool = False
    use_atr: bool = False
    use_mtf_ma: bool = False
    
    # Indicator windows
    indicator_windows: IndicatorWindows = Field(default_factory=IndicatorWindows)
    
    # Macro feature hook
    use_macro_features: bool = False
    macro_sources: Dict[str, Any] = Field(default_factory=dict)
    macro_lag_days: int = 1
    
    # Global HPO policy
    tune_once: bool = True
    
    # Deep / windowing knobs
    cnn_use_seq_windows: bool = True
    lstm_use_seq_windows: bool = True
    transformer_train_stride: int = 1
    cnn_train_stride: int = 1
    lstm_train_stride: int = 1
    deep_max_train_windows: int = 30000
    
    # Runtime coverage calibration mode
    runtime_coverage_mode: str = "rolling_quantile"
    
    # Spread/slippage protection baseline
    eval_use_spread_guard: bool = True
    eval_spread_cap: float = 0.00040
    slippage_factor: float = 1.0
    eval_impact_eta: float = 0.0
    
    # CV-time caps for keras models
    deep_cv_max_epochs: int = 12
    deep_cv_batch_size: int = 256
    deep_cv_patience: int = 6
    
    # Per-model CV caps
    cnn_cv_max_epochs: int = 6
    cnn_cv_batch_size: int = 256
    cnn_cv_train_stride: int = 3
    cnn_cv_max_train_windows: int = 4000
    
    lstm_cv_max_epochs: int = 8
    lstm_cv_batch_size: int = 256
    lstm_cv_train_stride: int = 2
    lstm_cv_max_train_windows: int = 5000
    
    transformer_cv_max_epochs: int = 6
    transformer_cv_batch_size: int = 256
    transformer_cv_train_stride: int = 2
    transformer_cv_max_train_windows: int = 7000
    
    # Extra-strict CV caps for ensemble deep heads
    cnn_ens_cv_max_epochs: int = 6
    lstm_ens_cv_max_epochs: int = 8
    cnn_ens_cv_max_train_windows: int = 4000
    lstm_ens_cv_max_train_windows: int = 5000
    
    # Probability calibration
    calibrate_method: str = "isotonic"
    deep_calibrate: bool = True
    deep_calibration_method: str = "temperature"
    deep_calibration_frac: float = 0.10
    deep_calibration_min_samples: int = 500
    
    # Trade gating
    gating_mode: str = "bets_psr"
    
    # Final experiment coverage policy
    target_active_rate: float = 0.15
    runtime_active_band_margin: float = 0.08
    runtime_conf_nudge: float = 0.005
    runtime_coverage_window: int = 192
    
    # Fallback floor when coverage intent is OFF
    confidence_threshold: float = 0.80
    
    # Never force trades by lowering conf_thr unless explicitly enabled
    allow_conf_backoff_cv: bool = False
    allow_conf_backoff_eval: bool = False
    
    # Real-sim bump applied on top of target_active_rate
    real_sim_target_active_mult: float = 1.00
    real_sim_target_active_cap: float = 0.25
    allow_real_sim_target_active_mult: bool = False
    
    # Reliability / pruning helpers
    min_trades_per_block: int = 0
    min_independent_bets: int = 20
    psr_alpha: float = 0.24
    dsr_prune: bool = True
    floor_cv_final: float = -6.0
    
    # Per-model confidence tweaks
    lstm_conf_relax: float = 0.15
    lstm_conf_floor: float = 0.40
    
    # Labeling guards & triple-barrier events
    use_triple_barrier: bool = True
    tb_pt_mult: float = 2.0
    tb_sl_mult: float = 2.0
    tb_max_holding: int = 48
    tb_neutral_zone: float = 1.0
    tb_neutral_zone_is_sigma: bool = True
    print_labeling_debug: bool = False
    
    # Prefilter / stability selection helpers
    use_prefilter: bool = True
    prefilter_min_unique_frac: float = 0.005
    prefilter_min_std: float = 1e-6
    prefilter_max_corr: float = 0.96
    prefilter_prefer_prefixes: List[str] = Field(default_factory=lambda: ["rv", "ema", "sma", "macd", "adx"])
    mutual_info_top_k: str = "sqrt"
    prefilter_random_state: int = 42
    
    # Ensemble throttles & fusion
    ensemble_train_stride: int = 1
    ensemble_deep_max_train_windows: int = 15000
    fusion_alpha: float = 0.6
    
    # Regime threshold defaults for AdaptiveRegimeStrategy
    adx_thresh_q: float = 0.70
    
    # Reporting / artifact controls
    save_monthly_equity_plots: bool = True
    save_monthly_feature_heatmaps: bool = False
    
    # Dynamic edge-vs-cost gating coefficients
    alpha_vol_z: float = 0.004
    beta_spread_norm: float = 0.008
    gamma_slip_norm: float = 0.004
    slip_norm_bps: float = 0.25
    min_slip_norm_bps: float = 0.05
    vol_z_cap: float = 6.0
    spread_norm_cap: float = 5.0
    slip_ratio_cap: float = 6.0
    min_conf_thr: float = 0.33
    max_conf_thr: float = 0.90
    min_conf_thr_cov: float = 0.0
    max_conf_thr_cov: float = 0.90
    
    # Top-N consensus & meta-analysis
    deploy_topN_consensus: bool = False
    use_adaptive_top3_for_main_results: bool = False
    topN_classical: int = 3
    topN_deep: int = 3
    topN_ensemble: int = 3
    topN_default: int = 3
    consensus_pool_max_trials: int = 0
    topN_style_lock: bool = False
    topN_min_perf_frac: float = 0.00
    topN_geom_radius: float = 9.0
    topN_lags_tol: float = 4.0
    topN_depth_tol: float = 1.0
    topN_target_tol: float = 0.05
    topN_max_corr: float = 0.9999
    print_topN_debug: bool = True
    deploy_param_heatmaps: bool = False
    topN_for_heatmaps: int = 5
    deploy_feature_freq: bool = False
    top_feature_percent: float = 1.0
    
    # Evaluation / risk / execution defaults
    eval_use_vol_target: bool = True
    eval_vol_target_ann: float = 0.10
    eval_vol_floor: float = 1e-6
    eval_vol_lookback: int = 96
    eval_max_leverage: float = 1.5
    
    eval_use_scaleout_trail: bool = True
    eval_tp1_z: float = 1.5
    eval_trail_k: float = 3.0
    eval_trail_dynamic_vol: bool = True
    eval_move_stop_to_be: bool = True
    eval_max_holding_bars: int = 0
    eval_min_holding_bars: int = 3
    
    eval_use_twap_execution: bool = True
    eval_twap_span_bars: int = 2
    eval_twap_freeze_size_at_entry: bool = True
    
    eval_use_regime_adaptive: bool = True
    eval_regime_source: str = "sigma"
    eval_regime_q_low: float = 0.33
    eval_regime_q_high: float = 0.66
    eval_tp1_z_calm: float = 1.2
    eval_tp1_z_normal: float = 1.5
    eval_tp1_z_volatile: float = 1.8
    eval_trail_k_calm: float = 2.5
    eval_trail_k_normal: float = 3.0
    eval_trail_k_volatile: float = 3.5
    eval_print_regime_debug: bool = False
    eval_print_trail_debug: bool = False
    eval_twap_print_debug: bool = False
    
    eval_use_kill_switch: bool = True
    eval_kill_mode: str = "sigma"
    eval_kill_limit_pct: float = 0.02
    eval_kill_sigma: float = 3.0
    eval_kill_until_session_end: bool = True
    eval_cooloff_bars: int = 30
    eval_kill_min_limit_pct: float = 0.005
    eval_kill_min_sigma: float = 1.0
    eval_kill_max_sigma: float = 6.0
    eval_kill_max_cooloff_bars: int = 480
    eval_kill_print_debug: bool = False
    
    # Output / plotting profile
    output_profile: str = "thesis"
    light_output: bool = False
    enable_pbo_mcs_analysis: bool = False
    allow_param_fallback: bool = False
    min_trades_for_wfo: int = 0
    
    # Regime features
    use_regime_features: bool = True
    regime_num_states: int = 3
    regime_trend_quantile: float = 0.7
    regime_vol_high_quantile: float = 0.7
    regime_vol_low_quantile: float = 0.4
    regime_vol_window: int = 20
    
    # DQN reward shaping
    env_cost_scale_dqn: float = 1.0
    env_turnover_penalty_dqn: float = 0.0002


class HPOConfig(BaseModel):
    """
    Hyperparameter Optimization configuration.
    
    Controls Optuna search behavior and parameter selection.
    """
    # Enable/disable HPO (SMOKE TEST: Enabled by default)
    enable_hpo: bool = True
    
    # HPO execution mode (SMOKE TEST: Most complex routing)
    hpo_mode: str = Field(
        default='continuous_wfo',
        pattern='^(single_time|mini_folds|continuous_wfo)$'
    )
    
    # Number of Optuna trials (SMOKE TEST: Absolute minimum)
    n_trials: int = 3
    
    # Active parameters to tune (empty = tune all available)
    active_params: List[str] = Field(default_factory=list)
    
    # Optuna study settings
    direction: str = Field(
        default='maximize',
        pattern='^(maximize|minimize)$'
    )
    sampler: str = Field(
        default='tpe',
        pattern='^(tpe|random|grid)$'
    )
    pruner: str = Field(
        default='median',
        pattern='^(median|hyperband|none)$'
    )
    
    # Study persistence
    study_name: Optional[str] = None
    storage_url: Optional[str] = None  # e.g., "sqlite:///optuna.db"
    
    # Logging and diagnostics
    verbose: bool = True
    save_trial_history: bool = True
    track_boundary_hits: bool = True
    
    # Timeout settings
    timeout_seconds: Optional[int] = None
    
    # Parallel execution
    n_jobs: int = 1  # Number of parallel trials


class WFOConfig(BaseModel):
    """Walk-Forward Optimization configuration with strict sizing."""
    
    # EXPLICIT SIZING (SMOKE TEST DEFAULTS)
    training_duration: int = 500      # Very small lookback for instant testing
    test_period_duration: int = 50    # Small step size for quick validation
    n_mini_folds: Optional[int] = 2   # Minimum to prove rolling logic works
    
    # WINDOW TYPE
    train_window_type: Literal['expanding', 'rolling'] = 'expanding'
    
    # HPO VALIDATION
    hpo_validation_split: float = 0.20
    
    # LEGACY (DEPRECATED - kept for backward compatibility)
    train_window_size: Optional[int] = None
    test_window_size: int = 100
    
    # CONSTRAINTS
    min_train_size: int = 500
    log_fold_details: bool = True
    
    class Config:
        frozen = True


class CVConfig(BaseModel):
    """Cross-validation configuration"""
    
    use_cached_global_hpo: bool = True
    n_trials: int = 0
    
    # CV geometry
    cv_mode: str = "mini_block"
    cv_blocks: int = 5
    cv_min_train_frac: float = 0.75
    cv_val_frac: float = 0.05
    cv_embargo_bars: int = 0
    cv_embargo_frac: float = 0.01
    cv_fit_blocks_exact: bool = True
    cv_tail_anchor: bool = True
    
    # Monthly-roll legacy knobs
    cv_target_folds: int = 5
    cv_val_months: float = 1.0
    cv_train_months: Optional[float] = None
    bars_per_month_hint: int = 1000
    cv_sliding_stride_frac: Optional[float] = None
    
    # Fold aggregation / robustness
    cv_fold_aggregator: str = "ivw_sharpe_capped"
    cv_sr_cap: float = 4.0
    cv_sr_var_floor: float = 0.75
    cv_weight_blend_neff: float = 0.30
    cv_neff_mode: str = "trades"
    cv_min_eff_n: float = 0.0
    cv_tail_weight: float = 1.00
    cv_z_weights: str = "sqrt_n"
    cv_z_cap: float = 8.0
    cv_sr_ref: float = 0.0
    cv_huber_delta: float = 1.50
    cv_catoni_alpha: float = 0.50
    cv_std_penalty: float = 0.0
    cv_coverage_gamma: float = 1.50
    
    # CV validity gates
    cv_min_coverage: float = 0.80
    cv_min_valid_fraction: float = 0.80
    cv_prune_on_low_valid_fraction: bool = True
    
    # Reliability / activity gates
    cv_min_trades_per_block: int = 30
    cv_min_indep_bets_per_block: int = 12
    cv_gate_min_folds: int = 4
    cv_gate_min_active_rate: float = 0.02
    cv_gate_min_sr: float = 0.00
    
    # Active-rate hygiene
    cv_min_active_rate: float = 0.005
    cv_active_rate_low: float = 0.00
    cv_active_rate_high: float = 1.00
    cv_active_rate_margin: float = 0.03
    cv_low_active_lambda: float = 0.25
    
    # Soft penalties for missing active-rate band in CV
    cv_soft_active_low_lambda: float = 1.0
    cv_soft_active_high_lambda: float = 1.0
    
    # Soft penalties for turnover outside family band
    cv_turnover_low: Optional[float] = None
    cv_turnover_high: Optional[float] = None
    cv_turnover_low_lambda: float = 1.0
    cv_turnover_high_lambda: float = 2.0
    cv_trade_shortfall_lambda: float = 0.0
    
    # Numeric stability
    cv_min_volatility: float = 1e-6
    cv_invalid_share_penalty: float = 5.0
    
    # Evaluation cost model knobs (CV)
    eval_use_trading_costs: bool = False
    eval_spread_pips: float = 0.8
    eval_slip_mode: str = "tworegime"
    eval_slip_bps_lo: float = 0.08
    eval_slip_bps_med: float = 0.16
    eval_slip_bps_hi: float = 0.30
    vol_window_bars: int = 96
    high_vol_q: float = 0.85
    high_vol_conf_bump: float = 0.0
    turnover_penalty_lambda: float = 0.1
    
    # Debug/log controls
    print_cv_debug: bool = False
    print_cv_fold_scores: bool = False
    cv_log_precision: int = 8
    cv_use_psr_trim: bool = False
    
    # Pruning controls
    prune_min_folds: int = 3
    prune_iqr_mult: float = 1.0125
    prune_abs_floor_sr: float = -8.0
    
    # Trade cap controls
    cv_dynamic_trades_cap_frac: float = 0.675
    cv_max_trades_per_block: int = 500
    
    # Alternative aggregation knobs
    cv_agg_mode: str = "tanh_mean"
    cv_tanh_s: float = 10.0
    cv_trim_frac: float = 0.20
    cv_psr_power: float = 1.0
    cv_use_recency_weight: bool = False
    cv_recency_power: float = 1.0
    
    # CSCV / PBO-related knobs
    cv_cscv_penalty_weight: float = 0.30
    cv_cscv_min_rank_corr: float = 0.20
    cv_cscv_disqualify: bool = False
    cv_strict_pruning: bool = False
    cv_prune_relax: float = 0.50
    cv_prune_precision_intent: bool = False
    
    # Optuna plateau stopping
    plateau_min_trials: int = 20
    plateau_patience: int = 15
    plateau_delta: float = 0.02
    
    # Disable extra stages
    robustness_eval: bool = False
    robust_seeds: List[int] = Field(default_factory=lambda: [1111, 2222, 3333])
    robust_require_pass: bool = False
    verify_topn_monthly_roll: bool = False


class DataConfig(BaseModel):
    """Data source configuration"""
    csv_path: str = Field(default="csv_data/EURUSD_10_years_H1_OANDA.csv", description="Path to CSV data file")
    instrument: str = Field(default="EURUSD", description="Trading instrument")
    timeframe: str = Field(default="H1", description="Data timeframe")


class OandaConfig(BaseSettings):
    """Oanda API credentials and connection settings"""
    
    account_id: str = Field(default="", description="Oanda account ID")
    access_token: str = Field(default="", description="Oanda API access token")
    account_type: str = Field(default="practice", description="Account type: practice or live")
    environment: str = Field(default="practice", description="API environment")
    
    class Config:
        env_prefix = "OANDA_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def from_config_file(cls, filepath: str = "oanda.cfg") -> "OandaConfig":
        """Load credentials from oanda.cfg file"""
        import configparser
        config = configparser.ConfigParser()
        config.read(filepath)
        
        if "oanda" not in config:
            raise ValueError(f"No [oanda] section found in {filepath}")
        
        return cls(
            account_id=config["oanda"].get("account_id", ""),
            access_token=config["oanda"].get("access_token", ""),
            account_type=config["oanda"].get("account_type", "practice")
        )


class AppConfig(BaseModel):
    """Complete application configuration"""
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    wfo: WFOConfig = Field(default_factory=WFOConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    oanda: OandaConfig = Field(default_factory=OandaConfig)
    
    # Application metadata
    version: str = "0.1.0"
    
    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Export configuration to JSON format.
        
        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        json_str = self.model_dump_json(indent=indent)
        
        if filepath:
            Path(filepath).write_text(json_str, encoding="utf-8")
        
        return json_str
    
    @classmethod
    def from_json(cls, source: str) -> "AppConfig":
        """
        Load configuration from JSON file or string.
        
        Args:
            source: JSON string or filepath
            
        Returns:
            AppConfig instance
        """
        # Check if source is a file path
        source_path = Path(source)
        if source_path.exists() and source_path.is_file():
            json_str = source_path.read_text(encoding="utf-8")
        else:
            json_str = source
        
        data = json.loads(json_str)
        return cls(**data)
    
    def to_class_defaults(self) -> Dict[str, Any]:
        """
        Convert to CLASS_DEFAULTS-compatible dictionary format.
        Useful for backward compatibility with existing MLBacktester code.
        
        Returns:
            Dictionary matching CLASS_DEFAULTS structure
        """
        return {
            "features": self.features.model_dump(),
            "cv": self.cv.model_dump()
        }
    
    @classmethod
    def from_class_defaults(cls, defaults: Dict[str, Any]) -> "AppConfig":
        """
        Create AppConfig from CLASS_DEFAULTS-style dictionary.
        
        Args:
            defaults: Dictionary with 'features' and 'cv' keys
            
        Returns:
            AppConfig instance
        """
        return cls(
            features=FeaturesConfig(**defaults.get("features", {})),
            cv=CVConfig(**defaults.get("cv", {}))
        )
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from nested dictionary.
        
        Args:
            updates: Dictionary with section keys (features, cv, data, oanda)
        """
        if "features" in updates:
            for key, value in updates["features"].items():
                if hasattr(self.features, key):
                    setattr(self.features, key, value)
        
        if "cv" in updates:
            for key, value in updates["cv"].items():
                if hasattr(self.cv, key):
                    setattr(self.cv, key, value)
        
        if "data" in updates:
            for key, value in updates["data"].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if "oanda" in updates:
            for key, value in updates["oanda"].items():
                if hasattr(self.oanda, key):
                    setattr(self.oanda, key, value)
    
    def clone(self) -> "AppConfig":
        """Create a deep copy of the configuration"""
        return AppConfig(**self.model_dump())


def load_default_config() -> AppConfig:
    """
    Load default configuration with Oanda credentials from file if available.
    
    Returns:
        AppConfig with defaults
    """
    config = AppConfig()
    
    # Try to load Oanda credentials from config file
    try:
        config.oanda = OandaConfig.from_config_file("oanda.cfg")
    except (FileNotFoundError, ValueError):
        pass
    
    return config
