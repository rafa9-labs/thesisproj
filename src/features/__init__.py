"""
Feature engineering layer: modular, vectorized indicators and pipeline
"""

from .indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_adx,
    compute_stochastic,
    compute_sar,
    compute_donchian,
    compute_realized_volatility,
)

from .composites import (
    compute_ma_spread,
    compute_price_ma_zscore,
    compute_crossover_bins,
    compute_reentry_momentum,
    compute_squeeze_expansion,
    compute_atr_channel_breaks,
    compute_trend_confirmation,
    compute_mtf_alignment,
    compute_vol_managed_momentum,
    compute_macd_atr_ratio,
)

from .regimes import compute_regime_features

from .pipeline import FeaturePipeline

from .utils import rolling_slope, fracdiff

__all__ = [
    "compute_sma",
    "compute_ema",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_adx",
    "compute_stochastic",
    "compute_sar",
    "compute_donchian",
    "compute_realized_volatility",
    "compute_ma_spread",
    "compute_price_ma_zscore",
    "compute_crossover_bins",
    "compute_reentry_momentum",
    "compute_squeeze_expansion",
    "compute_atr_channel_breaks",
    "compute_trend_confirmation",
    "compute_mtf_alignment",
    "compute_vol_managed_momentum",
    "compute_macd_atr_ratio",
    "compute_regime_features",
    "FeaturePipeline",
    "rolling_slope",
    "fracdiff",
]
