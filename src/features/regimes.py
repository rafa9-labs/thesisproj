"""
Regime classification features.
IMPORTANT: Reuses columns already calculated by indicators.py - does NOT recalculate.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_regime_features(
    df: pd.DataFrame,
    adx_col: str = "adx_14",
    vol_col: str = "rv_48",
    adx_thresh: float = 20.0,
    vol_thresh: float = 0.001
) -> pd.DataFrame:
    """
    Classify market regimes based on trend and volatility.
    
    CRITICAL: This function REUSES existing columns from the DataFrame.
    It does NOT recalculate ADX or volatility - those must already exist.
    
    Regime Classification:
        - 0 = SIDEWAYS (low trend, low volatility)
        - 1 = TREND (high trend, low volatility)
        - 2 = VOLATILE/CHOPPY (high volatility)
    
    Args:
        df: DataFrame with pre-calculated indicators
        adx_col: Column name for ADX (must already exist in df)
        vol_col: Column name for volatility proxy (must already exist in df)
        adx_thresh: ADX threshold for trend detection
        vol_thresh: Volatility threshold for high-vol detection
        
    Returns:
        DataFrame with added regime columns (does NOT modify input)
    """
    df = df.copy()
    
    # Guard: if required columns don't exist, return with default regime
    if adx_col not in df.columns or vol_col not in df.columns:
        df["regime_id"] = np.int8(1)
        df["trend_score"] = np.float32(0.0)
        df["vol_score"] = np.float32(0.0)
        df["regime_trend"] = np.int8(0)
        df["regime_sideways"] = np.int8(1)
        df["regime_volatile"] = np.int8(0)
        return df
    
    # REUSE existing columns (no recalculation)
    trend_score = df[adx_col].astype(np.float32)
    vol_score = df[vol_col].astype(np.float32)
    
    # Regime classification
    regime = np.full(len(df), 0, dtype=np.int8)
    
    trend_mask = trend_score >= adx_thresh
    vol_high = vol_score > vol_thresh
    
    regime[trend_mask & ~vol_high] = 1  # TREND
    regime[~trend_mask & vol_high] = 2  # VOLATILE
    regime[trend_mask & vol_high] = 2   # Strong but wild -> VOLATILE
    
    # Add regime columns
    df["trend_score"] = trend_score
    df["vol_score"] = vol_score
    df["regime_id"] = regime
    
    # One-hot encodings (helps classical models)
    df["regime_trend"] = (regime == 1).astype(np.int8)
    df["regime_sideways"] = (regime == 0).astype(np.int8)
    df["regime_volatile"] = (regime == 2).astype(np.int8)
    
    return df


def get_regime_column_names() -> list:
    """
    Get list of regime column names for feature selection.
    
    Returns:
        List of regime column names
    """
    return [
        "trend_score",
        "vol_score",
        "regime_id",
        "regime_trend",
        "regime_sideways",
        "regime_volatile"
    ]
