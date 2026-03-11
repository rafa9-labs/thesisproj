"""
Composite feature builders - combinations of base indicators.
Extracted from MLBacktesterNoWFO.py for modularity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .utils import rolling_slope, safe_divide


def compute_ma_spread(ema: pd.Series, sma: pd.Series) -> pd.Series:
    """
    EMA-SMA spread (momentum extension).
    
    Args:
        ema: EMA series
        sma: SMA series
        
    Returns:
        Spread series (float32)
    """
    result = (ema - sma).astype(np.float32)
    return result


def compute_price_ma_zscore(
    price: pd.Series,
    ma: pd.Series,
    window: int
) -> pd.Series:
    """
    Z-score of price relative to moving average.
    
    Args:
        price: Price series
        ma: Moving average series
        window: Window for standard deviation calculation
        
    Returns:
        Z-score series (float32)
    """
    min_periods = max(5, window // 3)
    sd = price.rolling(window, min_periods=min_periods).std(ddof=0)
    result = safe_divide(price - ma, sd, fill_value=0.0)
    return result.astype(np.float32)


def compute_crossover_bins(
    price: pd.Series,
    sma: Optional[pd.Series] = None,
    ema: Optional[pd.Series] = None,
    macd_diff: Optional[pd.Series] = None
) -> Dict[str, pd.Series]:
    """
    Binary crossover indicators.
    
    Args:
        price: Price series
        sma: SMA series (optional)
        ema: EMA series (optional)
        macd_diff: MACD diff series (optional)
        
    Returns:
        Dictionary with crossover binary features (int8)
    """
    features = {}
    
    if sma is not None:
        features["price_gt_sma"] = (price > sma).astype(np.int8)
    
    if ema is not None:
        features["price_gt_ema"] = (price > ema).astype(np.int8)
    
    trend_proxy = macd_diff
    if trend_proxy is None and ema is not None and sma is not None:
        trend_proxy = ema - sma
    
    if trend_proxy is not None:
        features["ma_cross_up"] = (trend_proxy > 0).astype(np.int8)
        features["ma_cross_dn"] = (trend_proxy < 0).astype(np.int8)
    
    return features


def compute_slope_differential(
    ema: pd.Series,
    sma: pd.Series,
    window: int,
    macd_diff: Optional[pd.Series] = None
) -> pd.Series:
    """
    Slope of MA spread (momentum acceleration).
    
    Args:
        ema: EMA series
        sma: SMA series
        window: Window for slope calculation
        macd_diff: Optional MACD diff to use instead
        
    Returns:
        Slope series (float32)
    """
    x = macd_diff if macd_diff is not None else (ema - sma)
    result = rolling_slope(pd.Series(x).ffill(), window)
    return result.astype(np.float32)


def compute_reentry_momentum(
    price: pd.Series,
    rsi: pd.Series,
    bb_upper: pd.Series,
    bb_lower: pd.Series
) -> pd.Series:
    """
    Re-entry momentum: detects bounce from lower BB with positive RSI slope.
    
    Args:
        price: Price series
        rsi: RSI series
        bb_upper: Bollinger upper band
        bb_lower: Bollinger lower band
        
    Returns:
        Re-entry momentum feature (float32)
    """
    bb_pct = safe_divide(price - bb_lower, bb_upper - bb_lower, fill_value=0.5)
    reenter = ((bb_pct.shift(1) < 0) & (bb_pct >= 0)).astype(np.float32)
    rsi_slope = rolling_slope(rsi.ffill(), 5)
    result = reenter * rsi_slope.clip(lower=0.0)
    return result.astype(np.float32)


def compute_ext_atr_low_adx(
    price: pd.Series,
    ema: pd.Series,
    atr: pd.Series,
    adx: pd.Series
) -> pd.Series:
    """
    Extension/ATR ratio weighted by low ADX (range-bound opportunity).
    
    Args:
        price: Price series
        ema: EMA series
        atr: ATR series
        adx: ADX series
        
    Returns:
        Extension feature (float32)
    """
    ext_atr = safe_divide((price - ema).abs(), atr, fill_value=0.0)
    adx_norm = (adx / 50.0).clip(0.0, 1.0)
    result = ext_atr * (1.0 - adx_norm)
    return result.astype(np.float32)


def compute_squeeze_expansion(
    bbw: pd.Series,
    adx: pd.Series,
    window: int = 300,
    quantile: float = 0.10
) -> pd.Series:
    """
    Squeeze-to-expansion signal: low BBW percentile rank with rising ADX.
    
    Args:
        bbw: Bollinger Band width
        adx: ADX series
        window: Window for percentile rank
        quantile: Low percentile threshold
        
    Returns:
        Squeeze expansion feature (float32)
    """
    min_periods = max(30, window // 5)
    
    def _pct_rank_last(x: np.ndarray) -> float:
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1]) if len(s) else np.nan
    
    bbw_rank = bbw.rolling(window, min_periods=min_periods).apply(_pct_rank_last, raw=True)
    adx_slope = rolling_slope(adx.ffill(), 5).clip(lower=0.0)
    result = ((quantile - bbw_rank).clip(lower=0.0)) * adx_slope
    return result.astype(np.float32)


def compute_atr_channel_breaks(
    price: pd.Series,
    ema: pd.Series,
    atr: pd.Series,
    mult: float = 1.5
) -> Dict[str, pd.Series]:
    """
    ATR channel breakout signals.
    
    Args:
        price: Price series
        ema: EMA series
        atr: ATR series
        mult: ATR multiplier for channel width
        
    Returns:
        Dictionary with 'atr_ch_up', 'atr_ch_dn' (float32)
    """
    deviation = safe_divide(price - ema, atr, fill_value=0.0)
    
    return {
        "atr_ch_up": (deviation - mult).astype(np.float32),
        "atr_ch_dn": ((ema - price) / (atr + 1e-8) - mult).astype(np.float32)
    }


def compute_trend_confirmation(
    price: pd.Series,
    ema: pd.Series,
    adx: pd.Series,
    macd_diff: Optional[pd.Series] = None
) -> pd.Series:
    """
    Trend confirmation: price > EMA, MACD positive, rising ADX.
    
    Args:
        price: Price series
        ema: EMA series
        adx: ADX series
        macd_diff: Optional MACD diff
        
    Returns:
        Trend confirmation feature (float32)
    """
    adx_slope = rolling_slope(adx.ffill(), 5).clip(lower=0.0)
    macd_ok = (macd_diff > 0).astype(np.float32) if macd_diff is not None else 1.0
    price_ok = (price > ema).astype(np.float32)
    result = price_ok * macd_ok * adx_slope
    return result.astype(np.float32)


def compute_mtf_alignment(
    price: pd.Series,
    ema: pd.Series,
    mtf_ma_fast: pd.Series
) -> pd.Series:
    """
    Multi-timeframe alignment: price > EMA and MTF MA rising.
    
    Args:
        price: Price series
        ema: EMA series
        mtf_ma_fast: Higher timeframe fast MA
        
    Returns:
        MTF alignment feature (float32)
    """
    mtf_slope = rolling_slope(mtf_ma_fast.ffill(), 5)
    result = ((price > ema).astype(np.float32)) * ((mtf_slope > 0).astype(np.float32))
    return result.astype(np.float32)


def compute_vol_managed_momentum(
    price: pd.Series,
    ema: pd.Series,
    atr: pd.Series
) -> pd.Series:
    """
    Volatility-managed momentum: (price - EMA) / ATR.
    
    Args:
        price: Price series
        ema: EMA series
        atr: ATR series
        
    Returns:
        Volatility-managed momentum (float32)
    """
    result = safe_divide(price - ema, atr, fill_value=0.0)
    return result.astype(np.float32)


def compute_macd_atr_ratio(
    macd_diff: pd.Series,
    atr: pd.Series
) -> pd.Series:
    """
    MACD/ATR ratio (volatility-normalized momentum).
    
    Args:
        macd_diff: MACD diff series
        atr: ATR series
        
    Returns:
        MACD/ATR ratio (float32)
    """
    result = safe_divide(macd_diff, atr, fill_value=0.0)
    return result.astype(np.float32)
