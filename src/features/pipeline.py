"""
FeaturePipeline: Orchestrates feature engineering with caching and multi-timeframe support.
Integrates with Phase 1 (AppConfig, DataFactory) for a complete modular system.
"""

import logging
import hashlib
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path

# Suppress pandas performance warnings for DataFrame fragmentation
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from ..core.config import AppConfig
from ..data.factory import DataFactory

from .indicators import (
    compute_sma, compute_ema, compute_rsi, compute_macd,
    compute_bollinger_bands, compute_atr, compute_adx,
    compute_stochastic, compute_sar, compute_donchian,
    compute_realized_volatility, compute_indicator_states,
    compute_spread_over_atr
)

from .composites import (
    compute_ma_spread, compute_price_ma_zscore, compute_crossover_bins,
    compute_slope_differential, compute_reentry_momentum,
    compute_ext_atr_low_adx, compute_squeeze_expansion,
    compute_atr_channel_breaks, compute_trend_confirmation,
    compute_mtf_alignment, compute_vol_managed_momentum,
    compute_macd_atr_ratio
)

from .regimes import compute_regime_features, get_regime_column_names

from .utils import (
    get_hlc, fracdiff, rolling_slope, downcast_to_float32,
    apply_shift_for_signals
)


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline with caching and multi-timeframe support.
    
    Key Features:
    - Uses AppConfig for all indicator toggles and windows
    - Uses DataFactory for multi-timeframe data loading
    - Caching to avoid redundant calculations
    - Left-join alignment for MTF (strict anti-look-ahead)
    - Feature integrity validation (NaN, inf detection)
    - Memory efficiency (float32 by default)
    """
    
    def __init__(self, config: AppConfig, data_factory: Optional[DataFactory] = None):
        """
        Initialize FeaturePipeline.
        
        Args:
            config: AppConfig instance with feature settings
            data_factory: Optional DataFactory for multi-timeframe loading
        """
        self.config = config
        self.data_factory = data_factory
        self._cache: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
        
        logger.info("FeaturePipeline initialized")
    
    def build_features(
        self,
        df: pd.DataFrame,
        include_lags: bool = True,
        include_rolling: bool = True,
        shift_signals: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Main feature engineering pipeline.
        
        Args:
            df: Input DataFrame with OHLC data
            include_lags: Whether to add lagged features
            include_rolling: Whether to add rolling statistics
            shift_signals: Whether to shift indicators by 1 bar (anti-look-ahead)
            
        Returns:
            Tuple of (df_with_features, feature_names)
        """
        df = df.copy()
        cfg = self.config.features
        ind_win = cfg.indicator_windows
        
        price_col = "close" if "close" in df.columns else "price"
        high, low, close = get_hlc(df, price_col)
        
        base_features = []
        
        logger.debug(f"Building features for {len(df)} rows")
        
        # 1) Base indicators
        if "returns" in df:
            df["rolling_std_20"] = df["returns"].rolling(20).std().astype(np.float32)
            base_features.append("rolling_std_20")
        
        # SMA
        if cfg.use_sma and price_col in df:
            df[f"sma_{ind_win.sma}"] = compute_sma(df[price_col], ind_win.sma)
            base_features.append(f"sma_{ind_win.sma}")
        
        # EMA
        if cfg.use_ema and price_col in df:
            df[f"ema_{ind_win.ema}"] = compute_ema(df[price_col], ind_win.ema)
            base_features.append(f"ema_{ind_win.ema}")
        
        # MACD
        if cfg.use_macd and price_col in df:
            macd_dict = compute_macd(
                df[price_col],
                ind_win.macd_fast,
                ind_win.macd_slow,
                ind_win.macd_signal
            )
            for key, val in macd_dict.items():
                df[key] = val
                base_features.append(key)
        
        # RSI
        if cfg.use_rsi and price_col in df:
            df[f"rsi_{ind_win.rsi}"] = compute_rsi(df[price_col], ind_win.rsi)
            base_features.append(f"rsi_{ind_win.rsi}")
        
        # Bollinger Bands
        if cfg.use_bbands and price_col in df:
            bb_dict = compute_bollinger_bands(df[price_col], ind_win.bb_window, ind_win.bb_dev)
            for key, val in bb_dict.items():
                df[key] = val
                base_features.append(key)
        
        # ATR
        if cfg.use_atr and high is not None and low is not None and close is not None:
            df[f"atr_{ind_win.atr}"] = compute_atr(high, low, close, ind_win.atr)
            base_features.append(f"atr_{ind_win.atr}")
            
            # Spread/ATR if spread column exists
            if "spread" in df.columns:
                df[f"spread_atr_{ind_win.atr}"] = compute_spread_over_atr(
                    df["spread"], df[f"atr_{ind_win.atr}"]
                )
                base_features.append(f"spread_atr_{ind_win.atr}")
        
        # ADX
        if cfg.use_adx and high is not None and low is not None and close is not None:
            df[f"adx_{ind_win.adx}"] = compute_adx(high, low, close, ind_win.adx)
            base_features.append(f"adx_{ind_win.adx}")
        
        # Stochastic
        if cfg.use_stoch and high is not None and low is not None and close is not None:
            stoch_dict = compute_stochastic(high, low, close, ind_win.stoch_k, ind_win.stoch_d)
            for key, val in stoch_dict.items():
                df[key] = val
                base_features.append(key)
        
        # SAR
        if high is not None and low is not None and close is not None:
            df["sar"] = compute_sar(high, low, close)
            base_features.append("sar")
        
        # Donchian Channels
        if cfg.use_donchian and high is not None and low is not None and close is not None:
            for window in [cfg.donchian_window_short, cfg.donchian_window_long]:
                dc_dict = compute_donchian(high, low, close, window)
                for key, val in dc_dict.items():
                    df[key] = val
                    base_features.append(key)
        
        # Realized Volatility
        if cfg.use_rv_features and "returns" in df:
            rv_dict = compute_realized_volatility(
                df["returns"],
                cfg.rv_window_short,
                cfg.rv_window_long
            )
            for key, val in rv_dict.items():
                df[key] = val
                base_features.append(key)
        
        # Fractional Differencing
        if cfg.use_fracdiff and price_col in df:
            fd = fracdiff(df[price_col], d=cfg.fracdiff_d)
            df[fd.name] = fd
            base_features.append(fd.name)
        
        # Indicator States
        if cfg.use_indicator_states:
            states_dict = compute_indicator_states(
                rsi=df.get(f"rsi_{ind_win.rsi}"),
                stoch_k=df.get("stoch_k"),
                bbw=df.get("bbw"),
                rsi_overbought=cfg.rsi_overbought_level,
                rsi_oversold=cfg.rsi_oversold_level,
                stoch_overbought=cfg.stoch_overbought_level,
                stoch_oversold=cfg.stoch_oversold_level,
                bbw_compress=cfg.bbw_compress_threshold,
                bbw_expand=cfg.bbw_expand_threshold
            )
            for key, val in states_dict.items():
                df[key] = val
                base_features.append(key)
        
        # 2) Composite features
        sma_col = f"sma_{ind_win.sma}"
        ema_col = f"ema_{ind_win.ema}"
        
        # MA spread
        if ema_col in df and sma_col in df:
            df["ema_sma_spread"] = compute_ma_spread(df[ema_col], df[sma_col])
            base_features.append("ema_sma_spread")
        
        # Price-MA z-scores
        if sma_col in df and price_col in df:
            df[f"price_sma_z_{ind_win.sma}"] = compute_price_ma_zscore(
                df[price_col], df[sma_col], ind_win.sma
            )
            base_features.append(f"price_sma_z_{ind_win.sma}")
        
        if ema_col in df and price_col in df:
            df[f"price_ema_z_{ind_win.ema}"] = compute_price_ma_zscore(
                df[price_col], df[ema_col], ind_win.ema
            )
            base_features.append(f"price_ema_z_{ind_win.ema}")
        
        # Crossover bins
        crossover_dict = compute_crossover_bins(
            df[price_col] if price_col in df else None,
            df.get(sma_col),
            df.get(ema_col),
            df.get("macd_diff")
        )
        for key, val in crossover_dict.items():
            df[key] = val
            base_features.append(key)
        
        # Slope differential
        if ema_col in df and sma_col in df:
            window_sd = max(5, min(ind_win.ema, ind_win.sma) // 2)
            df[f"ma_spread_slope{window_sd}"] = compute_slope_differential(
                df[ema_col], df[sma_col], window_sd, df.get("macd_diff")
            )
            base_features.append(f"ma_spread_slope{window_sd}")
        
        # Re-entry momentum
        if all(k in df for k in [price_col, f"rsi_{ind_win.rsi}", "bb_upper", "bb_lower"]):
            df["reentry_mom"] = compute_reentry_momentum(
                df[price_col], df[f"rsi_{ind_win.rsi}"],
                df["bb_upper"], df["bb_lower"]
            )
            base_features.append("reentry_mom")
        
        # Extension/ATR with low ADX
        if all(k in df for k in [price_col, ema_col, f"atr_{ind_win.atr}", f"adx_{ind_win.adx}"]):
            df["ext_atr_low_adx"] = compute_ext_atr_low_adx(
                df[price_col], df[ema_col],
                df[f"atr_{ind_win.atr}"], df[f"adx_{ind_win.adx}"]
            )
            base_features.append("ext_atr_low_adx")
        
        # Squeeze expansion
        if "bbw" in df and f"adx_{ind_win.adx}" in df:
            df["squeeze_expansion"] = compute_squeeze_expansion(
                df["bbw"], df[f"adx_{ind_win.adx}"]
            )
            base_features.append("squeeze_expansion")
        
        # ATR channel breaks
        if all(k in df for k in [price_col, ema_col, f"atr_{ind_win.atr}"]):
            atr_ch_dict = compute_atr_channel_breaks(
                df[price_col], df[ema_col], df[f"atr_{ind_win.atr}"]
            )
            for key, val in atr_ch_dict.items():
                df[key] = val
                base_features.append(key)
        
        # Trend confirmation
        if all(k in df for k in [price_col, ema_col, f"adx_{ind_win.adx}"]):
            df["trend_confirm"] = compute_trend_confirmation(
                df[price_col], df[ema_col],
                df[f"adx_{ind_win.adx}"], df.get("macd_diff")
            )
            base_features.append("trend_confirm")
        
        # MTF alignment
        if all(k in df for k in [price_col, ema_col, "mtf_ma_fast"]):
            df["mtf_align"] = compute_mtf_alignment(
                df[price_col], df[ema_col], df["mtf_ma_fast"]
            )
            base_features.append("mtf_align")
        
        # Volatility-managed momentum
        if all(k in df for k in [price_col, ema_col, f"atr_{ind_win.atr}"]):
            df["mom_vmm"] = compute_vol_managed_momentum(
                df[price_col], df[ema_col], df[f"atr_{ind_win.atr}"]
            )
            base_features.append("mom_vmm")
        
        # MACD/ATR ratio
        if "macd_diff" in df and f"atr_{ind_win.atr}" in df:
            df["macd_atr"] = compute_macd_atr_ratio(
                df["macd_diff"], df[f"atr_{ind_win.atr}"]
            )
            base_features.append("macd_atr")
        
        # 3) Regime features (reuses existing columns)
        if cfg.use_regime_features:
            df = compute_regime_features(
                df,
                adx_col=f"adx_{ind_win.adx}",
                vol_col=f"rv_{cfg.rv_window_short}",
                adx_thresh=20.0,
                vol_thresh=0.001
            )
            for col in get_regime_column_names():
                if col in df.columns and col not in base_features:
                    base_features.append(col)
        
        # 4) Lags and rolling expansions
        expanded_features = []
        
        if include_lags and "returns" in df:
            for lag in range(1, cfg.lag_depth + 1):
                col_name = f"returns_lag{lag}"
                df[col_name] = df["returns"].shift(lag).astype(np.float32)
                expanded_features.append(col_name)
        
        if include_lags or include_rolling:
            for feat in base_features:
                if feat not in df.columns:
                    continue
                
                if include_lags:
                    for k in range(1, cfg.lag_depth + 1):
                        col_name = f"{feat}_lag{k}"
                        df[col_name] = df[feat].shift(k).astype(np.float32)
                        expanded_features.append(col_name)
                
                if include_rolling:
                    for w in cfg.roll_windows:
                        df[f"{feat}_rollmean{w}"] = df[feat].rolling(w).mean().astype(np.float32)
                        df[f"{feat}_rollstd{w}"] = df[feat].rolling(w).std().astype(np.float32)
                        df[f"{feat}_rollslope{w}"] = rolling_slope(df[feat], w)
                        expanded_features.extend([
                            f"{feat}_rollmean{w}",
                            f"{feat}_rollstd{w}",
                            f"{feat}_rollslope{w}"
                        ])
        
        # 5) Hour features
        if cfg.include_hour:
            if hasattr(df.index, 'hour'):
                df["hour"] = df.index.hour.astype(np.int8)
                expanded_features.append("hour")
        
        if cfg.include_hour_cyclic:
            if hasattr(df.index, 'hour'):
                hour_vals = df.index.hour.to_numpy(dtype=np.float32)
                hour_rad = 2.0 * np.pi * hour_vals / 24.0
                df["hour_sin"] = np.sin(hour_rad).astype(np.float32)
                df["hour_cos"] = np.cos(hour_rad).astype(np.float32)
                expanded_features.extend(["hour_sin", "hour_cos"])
        
        # 6) Combine all features
        all_features = base_features + expanded_features
        
        # 7) Apply signal shift if requested (anti-look-ahead for MA-based signals)
        if shift_signals:
            signal_cols = [f"sma_{ind_win.sma}", f"ema_{ind_win.ema}"]
            df = apply_shift_for_signals(df, signal_cols, shift_periods=1)
        
        # 8) MTF fillna
        if cfg.mtf_fillna_method == "ffill":
            for mtf_col in ["mtf_ma_fast", "mtf_ma_slow"]:
                if mtf_col in df:
                    df[mtf_col] = df[mtf_col].ffill()
        
        # 9) Drop NaN rows
        valid_features = [f for f in all_features if f in df.columns]
        if valid_features:
            df = df.dropna(subset=valid_features)
        
        # 10) Memory optimization
        df = downcast_to_float32(df, exclude_cols=["time"])
        
        logger.info(f"Built {len(all_features)} features, {len(df)} rows after cleaning")
        
        return df, all_features
    
    def build_multi_timeframe_features(
        self,
        timeframes: List[str],
        source: str = "csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load multiple timeframes and merge using LEFT JOIN (strict anti-look-ahead).
        
        Args:
            timeframes: List of timeframes (e.g., ["M5", "H1", "D1"])
            source: Data source ("csv" or "oanda")
            **kwargs: Arguments for DataFactory.load_multi_timeframe
            
        Returns:
            DataFrame with merged multi-timeframe features
        """
        if self.data_factory is None:
            raise ValueError("DataFactory not configured. Pass it to __init__.")
        
        logger.info(f"Building multi-timeframe features: {timeframes}")
        
        # Load all timeframes
        mtf_data = self.data_factory.load_multi_timeframe(
            timeframes=timeframes,
            source=source,
            **kwargs
        )
        
        # Use first (highest frequency) as base
        base_tf = timeframes[0]
        base_df = mtf_data[base_tf]
        
        if base_df.empty:
            logger.error(f"Base timeframe {base_tf} is empty")
            return pd.DataFrame()
        
        # Build features for base timeframe
        base_df, base_features = self.build_features(base_df, shift_signals=True)
        
        # Merge higher timeframes using LEFT JOIN (backward fill)
        for tf in timeframes[1:]:
            if tf not in mtf_data or mtf_data[tf].empty:
                logger.warning(f"Skipping empty timeframe {tf}")
                continue
            
            higher_df = mtf_data[tf]
            higher_df, higher_features = self.build_features(
                higher_df,
                include_lags=False,
                include_rolling=False,
                shift_signals=True
            )
            
            # Rename columns to include timeframe suffix
            rename_map = {col: f"{col}_{tf}" for col in higher_features}
            higher_df = higher_df.rename(columns=rename_map)
            
            # LEFT JOIN with backward fill (strict anti-look-ahead)
            base_df = pd.merge_asof(
                base_df.sort_index(),
                higher_df[list(rename_map.values())].sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            logger.info(f"Merged {tf}: added {len(rename_map)} features")
        
        # Memory optimization
        base_df = downcast_to_float32(base_df)
        
        logger.info(f"Multi-timeframe merge complete: {base_df.shape}")
        
        return base_df
    
    def check_feature_integrity(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate features for NaN, inf, and data quality issues.
        
        Args:
            df: DataFrame with features
            features: List of feature column names
            
        Returns:
            Tuple of (cleaned_df, diagnostics_dict)
        """
        diagnostics = {
            "nan_counts": {},
            "inf_counts": {},
            "zero_variance": [],
            "rows_before": len(df),
            "rows_after": 0,
            "rows_dropped": 0
        }
        
        df = df.copy()
        
        # Check for NaN
        for feat in features:
            if feat in df.columns:
                nan_count = df[feat].isna().sum()
                if nan_count > 0:
                    diagnostics["nan_counts"][feat] = int(nan_count)
        
        # Check for inf
        for feat in features:
            if feat in df.columns and df[feat].dtype in [np.float32, np.float64]:
                inf_count = np.isinf(df[feat]).sum()
                if inf_count > 0:
                    diagnostics["inf_counts"][feat] = int(inf_count)
                    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
        
        # Check for zero variance
        for feat in features:
            if feat in df.columns and df[feat].dtype in [np.float32, np.float64]:
                if df[feat].std() == 0:
                    diagnostics["zero_variance"].append(feat)
        
        # Drop rows with NaN in feature columns
        valid_features = [f for f in features if f in df.columns]
        if valid_features:
            df = df.dropna(subset=valid_features)
        
        diagnostics["rows_after"] = len(df)
        diagnostics["rows_dropped"] = diagnostics["rows_before"] - diagnostics["rows_after"]
        
        logger.info(
            f"Feature integrity check: {diagnostics['rows_dropped']} rows dropped, "
            f"{len(diagnostics['nan_counts'])} features had NaN, "
            f"{len(diagnostics['inf_counts'])} features had inf"
        )
        
        return df, diagnostics
    
    def cache_features(self, key: str, df: pd.DataFrame, features: List[str]):
        """Store computed features in cache."""
        self._cache[key] = (df.copy(), list(features))
        logger.debug(f"Cached features with key: {key[:50]}...")
    
    def get_cached_features(self, key: str) -> Optional[Tuple[pd.DataFrame, List[str]]]:
        """Retrieve cached features if available."""
        if key in self._cache:
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return self._cache[key]
        return None
    
    def clear_cache(self):
        """Clear feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")
    
    def generate_cache_key(self, df: pd.DataFrame, **params) -> str:
        """Generate cache key from DataFrame and parameters."""
        key_parts = [
            str(df.index[0]) if len(df) > 0 else "",
            str(df.index[-1]) if len(df) > 0 else "",
            str(len(df)),
            str(sorted(params.items()))
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
