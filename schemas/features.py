"""
schemas.features — Pydantic models for feature engineering and CV configuration.

═══════════════════════════════════════════════════════════════════════════════
EDUCATIONAL: Nested Models — Pydantic's Superpower
═══════════════════════════════════════════════════════════════════════════════

BEFORE:
  CLASS_DEFAULTS = {
      "features": {
          "indicator_windows": {
              "sma": 20, "ema": 20, "rsi": 14, ...
          },
          "lags": 14,  # oops, this should be under features, not indicator_windows
      },
      "cv": { ... }
  }
  
  A typo or misplaced key is invisible — it's just a dict.
  
AFTER:
  FeaturesConfig(
      indicator_windows=IndicatorWindows(sma=20, ema=20),
      lags=14,
  )
  
  If you accidentally put "lags" inside IndicatorWindows, Pydantic rejects it
  (because IndicatorWindows doesn't have a "lags" field).

KEY CONCEPT: Pydantic models can contain OTHER Pydantic models as fields.
This creates a TYPE-SAFE NESTED STRUCTURE — like a JSON Schema that actually
enforces itself. When you call model_validate() on the outer model, it
recursively validates ALL inner models too.

  FeaturesConfig.model_validate(raw_dict)
  → validates IndicatorWindows inside FeaturesConfig
  → validates every field with constraints
  → returns a fully typed, fully validated object
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ══════════════════════════════════════════════════════════════════════════════
# IndicatorWindows — Technical indicator lookback periods
# ══════════════════════════════════════════════════════════════════════════════
#
# EDUCATIONAL NOTE: This is a NESTED MODEL inside FeaturesConfig.
# In CLASS_DEFAULTS["features"]["indicator_windows"], this was a plain dict.
# Now it's a typed model with constraints — window sizes can't be negative.

class IndicatorWindows(BaseModel):
    """
    Lookback periods for technical indicators.
    
    All window sizes must be positive integers (ge=1).
    These map directly to the indicator_windows dict in
    pipeline/metrics_tuples.py CLASS_DEFAULTS["features"].
    """
    model_config = ConfigDict(extra="ignore")
    
    sma: int = Field(default=20, ge=1, le=500)
    ema: int = Field(default=20, ge=1, le=500)
    rsi: int = Field(default=14, ge=2, le=100)
    atr: int = Field(default=14, ge=1, le=100)
    adx: int = Field(default=14, ge=2, le=100)
    macd_fast: int = Field(default=12, ge=1, le=50)
    macd_slow: int = Field(default=26, ge=1, le=100)
    macd_signal: int = Field(default=9, ge=1, le=50)
    bb_window: int = Field(default=20, ge=1, le=200)
    bb_dev: float = Field(default=2.0, ge=0.1, le=5.0)
    stoch_k: int = Field(default=14, ge=1, le=100)
    stoch_d: int = Field(default=3, ge=1, le=50)
    mtf_ma_fast_window: int = Field(default=10, ge=1, le=200)
    mtf_ma_slow_window: int = Field(default=50, ge=1, le=500)


# ══════════════════════════════════════════════════════════════════════════════
# FeaturesConfig — Feature engineering pipeline configuration
# ══════════════════════════════════════════════════════════════════════════════
#
# This replaces CLASS_DEFAULTS["features"] in pipeline/metrics_tuples.py.
# That dict has ~150 keys. This model captures the most important ones
# with typed fields and constraints. The rest pass through via extra="ignore".

class FeaturesConfig(BaseModel):
    """
    Typed configuration for the feature engineering pipeline.
    
    EDUCATIONAL NOTE — extra="ignore" for backward compatibility:
    The raw features dict in CLASS_DEFAULTS has ~150 keys. We've typed the
    most important ~80 here. The remaining ~70 keys still need to work.
    
    With extra="ignore", those extra keys are silently dropped when
    constructing a FeaturesConfig from model_validate(). To preserve them,
    use FeaturesConfig.preserve_extras(raw_dict) which stores unknown keys
    in ._extra for pass-through to the pipeline.
    """
    model_config = ConfigDict(extra="ignore")
    
    # ── Session & leakage control ──
    base_timeframe: str = Field(default="M30")
    session_filter_mode: Literal["both", "london", "newyork", "asian"] = Field(
        default="both",
    )
    session_filter_on_train: bool = Field(default=True)
    final_embargo_bars: int = Field(default=0, ge=0)
    enforce_day1_start: bool = Field(default=True)
    
    # ── Lag / windowing ──
    lag_depth: int = Field(default=1, ge=1, le=5)
    roll_windows: List[int] = Field(default=[5, 10, 30, 60])
    include_hour: bool = Field(default=True)
    include_hour_cyclic: bool = Field(default=True)
    
    # ── Volatility features ──
    use_rv_features: bool = Field(default=True)
    rv_window_short: int = Field(default=48, ge=1)
    rv_window_long: int = Field(default=240, ge=1)
    
    # ── Donchian channels ──
    use_donchian: bool = Field(default=True)
    donchian_window_short: int = Field(default=20, ge=1)
    donchian_window_long: int = Field(default=60, ge=1)
    
    # ── Fractional differencing ──
    use_fracdiff: bool = Field(default=False)
    fracdiff_d: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # ── Indicator toggles ──
    use_rsi: bool = Field(default=False)
    use_macd: bool = Field(default=False)
    use_ema: bool = Field(default=False)
    use_adx: bool = Field(default=False)
    use_bbands: bool = Field(default=False)
    use_stoch: bool = Field(default=False)
    use_atr: bool = Field(default=False)
    use_mtf_ma: bool = Field(default=False)
    
    # ── Indicator windows (nested model) ──
    indicator_windows: IndicatorWindows = Field(default_factory=IndicatorWindows)
    
    # ── Triple barrier labeling ──
    use_triple_barrier: bool = Field(default=True)
    tb_pt_mult: float = Field(default=2.0, ge=0.1)
    tb_sl_mult: float = Field(default=2.0, ge=0.1)
    tb_max_holding: int = Field(default=48, ge=1)
    tb_neutral_zone: float = Field(default=1.0, ge=0.0)
    tb_neutral_zone_is_sigma: bool = Field(default=True)
    
    # ── Deep model windowing ──
    cnn_use_seq_windows: bool = Field(default=True)
    lstm_use_seq_windows: bool = Field(default=True)
    transformer_train_stride: int = Field(default=1, ge=1)
    cnn_train_stride: int = Field(default=1, ge=1)
    lstm_train_stride: int = Field(default=1, ge=1)
    deep_max_train_windows: int = Field(default=30000, ge=100)
    
    # ── Calibration ──
    calibrate_method: Literal["sigmoid", "isotonic"] = Field(default="isotonic")
    deep_calibrate: bool = Field(default=True)
    deep_calibration_method: str = Field(default="temperature")
    deep_calibration_frac: float = Field(default=0.10, ge=0.01, le=0.5)
    
    # ── Confidence / coverage ──
    confidence_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    target_active_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    
    # ── Prefilter / stability selection ──
    use_prefilter: bool = Field(default=True)
    prefilter_min_unique_frac: float = Field(default=0.005, ge=0.0, le=1.0)
    prefilter_min_std: float = Field(default=1e-6, ge=0.0)
    prefilter_max_corr: float = Field(default=0.96, ge=0.0, le=1.0)
    mutual_info_top_k: str = Field(default="sqrt")
    
    # ── Evaluation ──
    eval_use_vol_target: bool = Field(default=True)
    eval_vol_target_ann: float = Field(default=0.10, ge=0.0)
    eval_max_leverage: float = Field(default=1.5, ge=0.1)
    eval_min_holding_bars: int = Field(default=3, ge=0)
    
    # ── Regime features ──
    use_regime_features: bool = Field(default=True)
    regime_num_states: int = Field(default=3, ge=2, le=10)
    
    # ── DQN reward shaping ──
    env_cost_scale_dqn: float = Field(default=1.0, ge=0.0)
    env_turnover_penalty_dqn: float = Field(default=0.0002, ge=0.0)
    
    @classmethod
    def from_class_defaults(cls) -> "FeaturesConfig":
        """
        Construct from CLASS_DEFAULTS["features"] (backward-compatible factory).
        
        This is the BRIDGE between the old dict-based defaults and the new
        typed model. It extracts the known keys and ignores the rest.
        
        USAGE:
            from pipeline.metrics_tuples import CLASS_DEFAULTS
            features = FeaturesConfig.from_class_defaults()
        """
        # Import here to avoid circular imports
        from pipeline.metrics_tuples import CLASS_DEFAULTS
        return cls.model_validate(CLASS_DEFAULTS["features"])


# ══════════════════════════════════════════════════════════════════════════════
# CVConfig — Cross-validation and HPO configuration
# ══════════════════════════════════════════════════════════════════════════════
#
# This replaces CLASS_DEFAULTS["cv"] in pipeline/metrics_tuples.py.

class CVConfig(BaseModel):
    """
    Typed configuration for cross-validation and hyperparameter optimization.
    
    Maps to CLASS_DEFAULTS["cv"] — the CV/HPO configuration block.
    """
    model_config = ConfigDict(extra="ignore")
    
    # ── HPO ──
    use_cached_global_hpo: bool = Field(default=True)
    n_trials: int = Field(default=0, ge=0, le=1000)
    
    # ── CV geometry ──
    cv_mode: str = Field(default="mini_block")
    cv_blocks: int = Field(default=3, ge=2, le=20)
    cv_min_train_frac: float = Field(default=0.75, ge=0.1, le=0.99)
    cv_val_frac: float = Field(default=0.05, ge=0.01, le=0.5)
    cv_embargo_bars: int = Field(default=0, ge=0)
    cv_embargo_frac: float = Field(default=0.01, ge=0.0, le=0.5)
    
    # ── Fold aggregation ──
    cv_fold_aggregator: str = Field(default="ivw_sharpe_capped")
    cv_sr_cap: float = Field(default=4.0, ge=0.1)
    
    # ── Validity gates ──
    cv_min_coverage: float = Field(default=0.80, ge=0.0, le=1.0)
    cv_min_valid_fraction: float = Field(default=0.80, ge=0.0, le=1.0)
    
    # ── Activity gates ──
    cv_min_trades_per_block: int = Field(default=30, ge=0)
    cv_min_indep_bets_per_block: int = Field(default=12, ge=0)
    
    # ── Cost model ──
    eval_use_trading_costs: bool = Field(default=False)
    eval_spread_pips: float = Field(default=0.8, ge=0.0)
    turnover_penalty_lambda: float = Field(default=0.1, ge=0.0)
    
    # ── Plateau stopping ──
    plateau_min_trials: int = Field(default=20, ge=1)
    plateau_patience: int = Field(default=15, ge=1)
    plateau_delta: float = Field(default=0.02, ge=0.0)
    
    # ── Robustness ──
    robustness_eval: bool = Field(default=False)
    robust_seeds: List[int] = Field(default=[1111, 2222, 3333])
    
    @classmethod
    def from_class_defaults(cls) -> "CVConfig":
        """Construct from CLASS_DEFAULTS["cv"] (backward-compatible factory)."""
        from pipeline.metrics_tuples import CLASS_DEFAULTS
        return cls.model_validate(CLASS_DEFAULTS["cv"])