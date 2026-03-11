"""
Base technical indicators - all vectorized, no loops.
Extracted from MLBacktesterNoWFO.py for modularity and reusability.
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, Optional
from .utils import get_hlc, safe_divide


def compute_sma(price: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average.
    
    Args:
        price: Price series
        window: Rolling window size
        
    Returns:
        SMA series (float32)
    """
    result = ta.trend.sma_indicator(price, window=window)
    return result.astype(np.float32)


def compute_ema(price: pd.Series, window: int = 20) -> pd.Series:
    """
    Exponential Moving Average.
    
    Args:
        price: Price series
        window: Rolling window size
        
    Returns:
        EMA series (float32)
    """
    result = ta.trend.ema_indicator(price, window=window)
    return result.astype(np.float32)


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Args:
        price: Price series
        window: RSI period
        
    Returns:
        RSI series (float32)
    """
    result = ta.momentum.RSIIndicator(price, window=window).rsi()
    return result.astype(np.float32)


def compute_macd(
    price: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Dict[str, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).
    
    Args:
        price: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Dictionary with 'macd_line', 'macd_signal', 'macd_diff' (float32)
    """
    macd_obj = ta.trend.MACD(
        price,
        window_slow=slow,
        window_fast=fast,
        window_sign=signal
    )
    
    return {
        "macd_line": macd_obj.macd().astype(np.float32),
        "macd_signal": macd_obj.macd_signal().astype(np.float32),
        "macd_diff": macd_obj.macd_diff().astype(np.float32)
    }


def compute_bollinger_bands(
    price: pd.Series,
    window: int = 20,
    dev: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Bollinger Bands with %B and bandwidth.
    
    Args:
        price: Price series
        window: Rolling window size
        dev: Number of standard deviations
        
    Returns:
        Dictionary with 'bb_upper', 'bb_lower', 'bb_pct', 'bbw' (float32)
    """
    bb = ta.volatility.BollingerBands(price, window=window, window_dev=dev)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    
    bb_pct = safe_divide(price - lower, upper - lower, fill_value=0.5)
    
    return {
        "bb_upper": upper.astype(np.float32),
        "bb_lower": lower.astype(np.float32),
        "bb_pct": bb_pct.astype(np.float32),
        "bbw": bb.bollinger_wband().astype(np.float32)
    }


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ATR period
        
    Returns:
        ATR series (float32)
    """
    result = ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()
    return result.astype(np.float32)


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Average Directional Index.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ADX period
        
    Returns:
        ADX series (float32)
    """
    result = ta.trend.ADXIndicator(high, low, close, window=window).adx()
    return result.astype(np.float32)


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3
) -> Dict[str, pd.Series]:
    """
    Stochastic Oscillator (%K and %D).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_window: %K period
        d_window: %D smoothing period
        
    Returns:
        Dictionary with 'stoch_k', 'stoch_d' (float32)
    """
    Hh = high.rolling(k_window, min_periods=k_window).max()
    Ll = low.rolling(k_window, min_periods=k_window).min()
    
    stoch_k = 100.0 * safe_divide(close - Ll, Hh - Ll, fill_value=50.0)
    stoch_d = stoch_k.rolling(d_window, min_periods=d_window).mean()
    
    return {
        "stoch_k": stoch_k.astype(np.float32),
        "stoch_d": stoch_d.astype(np.float32)
    }


def compute_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Parabolic SAR.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        
    Returns:
        SAR series (float32)
    """
    result = ta.trend.PSARIndicator(high, low, close).psar()
    return result.astype(np.float32)


def compute_donchian(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20
) -> Dict[str, pd.Series]:
    """
    Donchian Channels with breakout signals.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Channel period
        
    Returns:
        Dictionary with 'donchian_up', 'donchian_dn', 'donchian_break_up', 'donchian_break_dn' (float32)
    """
    min_periods = max(5, window // 3)
    
    dc_high = high.rolling(window, min_periods=min_periods).max()
    dc_low = low.rolling(window, min_periods=min_periods).min()
    
    break_up = (close >= dc_high).astype(np.float32)
    break_dn = (close <= dc_low).astype(np.float32)
    
    return {
        "donchian_up": dc_high.astype(np.float32),
        "donchian_dn": dc_low.astype(np.float32),
        "donchian_break_up": break_up,
        "donchian_break_dn": break_dn
    }


def compute_realized_volatility(
    returns: pd.Series,
    window_short: int = 48,
    window_long: int = 240
) -> Dict[str, pd.Series]:
    """
    Realized Volatility and Bipower Variation.
    
    Args:
        returns: Returns series
        window_short: Short window for RV
        window_long: Long window for RV
        
    Returns:
        Dictionary with 'rv_short', 'rv_long', 'bpv_short', 'bpv_long', 'rv_roc_short', 'rv_roc_long' (float32)
    """
    def _rv(s: pd.Series, w: int) -> pd.Series:
        rv2 = s.pow(2).rolling(w, min_periods=max(5, w//3)).sum()
        return np.sqrt(rv2).astype(np.float32)
    
    def _bpv(s: pd.Series, w: int) -> pd.Series:
        abs_r = s.abs()
        prod = abs_r * abs_r.shift(1)
        bpv = (np.pi / 2.0) * prod.rolling(w, min_periods=max(5, w//3)).sum()
        return np.sqrt(bpv.clip(lower=0)).astype(np.float32)
    
    rv_short = _rv(returns, window_short)
    rv_long = _rv(returns, window_long)
    bpv_short = _bpv(returns, window_short)
    bpv_long = _bpv(returns, window_long)
    
    return {
        f"rv_{window_short}": rv_short,
        f"rv_{window_long}": rv_long,
        f"bpv_{window_short}": bpv_short,
        f"bpv_{window_long}": bpv_long,
        f"rv_roc_{window_short}": rv_short.pct_change().astype(np.float32),
        f"rv_roc_{window_long}": rv_long.pct_change().astype(np.float32)
    }


def compute_indicator_states(
    rsi: Optional[pd.Series] = None,
    stoch_k: Optional[pd.Series] = None,
    bbw: Optional[pd.Series] = None,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    stoch_overbought: float = 80,
    stoch_oversold: float = 20,
    bbw_compress: float = 0.05,
    bbw_expand: float = 0.20
) -> Dict[str, pd.Series]:
    """
    Indicator state features (oscillator levels and volatility regimes).
    
    Args:
        rsi: RSI series
        stoch_k: Stochastic %K series
        bbw: Bollinger Band width series
        rsi_overbought: RSI overbought level
        rsi_oversold: RSI oversold level
        stoch_overbought: Stochastic overbought level
        stoch_oversold: Stochastic oversold level
        bbw_compress: BBW compression threshold
        bbw_expand: BBW expansion threshold
        
    Returns:
        Dictionary with state indicators (int8)
    """
    states = {}
    
    if rsi is not None:
        rsi_state = pd.Series(0, index=rsi.index, dtype=np.int8)
        rsi_state[rsi >= rsi_overbought] = 1
        rsi_state[rsi <= rsi_oversold] = -1
        states["rsi_state"] = rsi_state
    
    if stoch_k is not None:
        stoch_state = pd.Series(0, index=stoch_k.index, dtype=np.int8)
        stoch_state[stoch_k >= stoch_overbought] = 1
        stoch_state[stoch_k <= stoch_oversold] = -1
        states["stoch_state"] = stoch_state
    
    if bbw is not None:
        vol_state = pd.Series(0, index=bbw.index, dtype=np.int8)
        vol_state[bbw <= bbw_compress] = -1
        vol_state[bbw >= bbw_expand] = 1
        states["vol_state_bbw"] = vol_state
    
    return states


def compute_spread_over_atr(
    spread: pd.Series,
    atr: pd.Series
) -> pd.Series:
    """
    Spread normalized by ATR.
    
    Args:
        spread: Spread series (ask - bid)
        atr: ATR series
        
    Returns:
        Spread/ATR ratio (float32)
    """
    result = safe_divide(spread, atr, fill_value=0.0)
    return result.astype(np.float32)
