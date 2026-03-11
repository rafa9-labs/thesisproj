"""
Utility functions for feature engineering.
Extracted from MLBacktesterNoWFO.py for modularity.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Efficient O(n) rolling slope using cumulative sums.
    Much faster than per-window polyfit.
    
    Args:
        series: Input time series
        window: Rolling window size
        
    Returns:
        Series with rolling slope values (NaN for first window-1 points)
    """
    x = np.arange(window, dtype=np.float32)
    Sx = x.sum()
    Sxx = (x * x).sum()
    n = window
    den = n * Sxx - Sx * Sx
    
    y = series.astype(np.float32).to_numpy()
    y_filled = np.where(np.isfinite(y), y, 0.0)
    
    csum_y = np.cumsum(y_filled)
    csum_xy = np.cumsum(y_filled * np.arange(len(y), dtype=np.float32))
    
    Sy = csum_y[window-1:] - np.concatenate(([0.0], csum_y[:-window]))
    Sxy = csum_xy[window-1:] - np.concatenate(([0.0], csum_xy[:-window]))
    
    num = n * (Sxy - np.arange(window-1, len(y), dtype=np.float32) * Sy) - Sx * Sy
    slope = num / den
    
    out = np.full_like(y, np.nan, dtype=np.float32)
    out[window-1:] = slope
    return pd.Series(out, index=series.index, dtype=np.float32)


def fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
    """
    Compute fractional differencing weights.
    
    Args:
        d: Differencing order (0 < d < 1)
        size: Maximum number of weights
        thresh: Threshold for truncating small weights
        
    Returns:
        Array of weights
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w, dtype=np.float32)


def fracdiff(
    series: pd.Series,
    d: float = 0.4,
    max_size: int = 2000,
    thresh: float = 1e-4
) -> pd.Series:
    """
    Apply fractional differencing to make series stationary while preserving memory.
    
    Args:
        series: Input time series
        d: Differencing order (0 < d < 1)
        max_size: Maximum window size for weights
        thresh: Threshold for truncating small weights
        
    Returns:
        Fractionally differenced series
    """
    s = series.astype(np.float32)
    w = fracdiff_weights(d, min(max_size, len(s)), thresh=thresh)
    out = np.full(len(s), np.nan, dtype=np.float32)
    kmax = len(w) - 1
    vals = s.values
    
    for t in range(kmax, len(s)):
        window = vals[t - kmax : t + 1]
        out[t] = float(np.dot(w[::-1], window))
    
    return pd.Series(out, index=s.index, name=f"fd_{getattr(series, 'name', 'x')}_d{d:.2f}", dtype=np.float32)


def get_hlc(
    df: pd.DataFrame,
    price_col: str = "close"
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Extract high, low, close columns from DataFrame.
    Falls back to price_col if high/low not available.
    
    Args:
        df: DataFrame with price data
        price_col: Fallback column name for price
        
    Returns:
        Tuple of (high, low, close) Series or None if not available
    """
    high = df.get("high", df.get(price_col))
    low = df.get("low", df.get(price_col))
    close = df.get("close", df.get(price_col))
    return high, low, close


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0,
    eps: float = 1e-8
) -> pd.Series:
    """
    Safely divide two series, handling division by zero and inf values.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use for invalid results
        eps: Small value to add to denominator to avoid division by zero
        
    Returns:
        Division result with inf/nan replaced by fill_value
    """
    result = numerator / (denominator + eps)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(fill_value)
    return result.astype(np.float32)


def apply_shift_for_signals(
    df: pd.DataFrame,
    columns: list,
    shift_periods: int = 1
) -> pd.DataFrame:
    """
    Shift indicator columns to prevent look-ahead bias.
    Use this for indicators that use "current close" as signals.
    
    Args:
        df: DataFrame with indicators
        columns: List of column names to shift
        shift_periods: Number of periods to shift (default 1)
        
    Returns:
        DataFrame with shifted columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].shift(shift_periods)
    return df


def downcast_to_float32(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Downcast float64 columns to float32 for memory efficiency.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from downcasting
        
    Returns:
        DataFrame with float32 columns
    """
    exclude_cols = exclude_cols or []
    df = df.copy()
    
    for col in df.select_dtypes(include=[np.float64]).columns:
        if col not in exclude_cols:
            df[col] = df[col].astype(np.float32)
    
    return df
