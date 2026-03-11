"""
Example usage of Phase 2: Feature Engineering Layer

Demonstrates:
1. Single-timeframe feature engineering
2. Multi-timeframe feature merging (M5 + H1 + D1)
3. Feature integrity validation
4. Caching performance
5. Individual indicator usage
"""

import pandas as pd
from datetime import datetime, timezone

from src.core.config import load_default_config
from src.data.factory import DataFactory, configure_logging as configure_data_logging
from src.features.pipeline import FeaturePipeline
from src.features.indicators import compute_rsi, compute_macd, compute_bollinger_bands
from src.features.composites import compute_ma_spread, compute_trend_confirmation
from src.features.regimes import compute_regime_features

import logging


def configure_logging():
    """Setup logging for examples"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def example_single_indicator():
    """Example: Use individual indicators directly"""
    print("=" * 60)
    print("EXAMPLE 1: Individual Indicator Usage")
    print("=" * 60)
    
    # Create sample price data
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    prices = pd.Series(
        1.0 + 0.01 * pd.Series(range(100)).apply(lambda x: x + 10 * (x % 10)),
        index=dates
    )
    
    # Compute individual indicators
    rsi = compute_rsi(prices, window=14)
    macd_dict = compute_macd(prices, fast=12, slow=26, signal=9)
    bb_dict = compute_bollinger_bands(prices, window=20, dev=2.0)
    
    print(f"✓ RSI computed: {len(rsi)} values")
    print(f"  Latest RSI: {rsi.iloc[-1]:.2f}")
    
    print(f"✓ MACD computed: {len(macd_dict)} components")
    print(f"  Latest MACD diff: {macd_dict['macd_diff'].iloc[-1]:.6f}")
    
    print(f"✓ Bollinger Bands computed: {len(bb_dict)} components")
    print(f"  Latest BB %B: {bb_dict['bb_pct'].iloc[-1]:.2f}")
    
    print()


def example_single_timeframe_pipeline():
    """Example: Full feature pipeline on single timeframe"""
    print("=" * 60)
    print("EXAMPLE 2: Single-Timeframe Feature Pipeline")
    print("=" * 60)
    
    config = load_default_config()
    
    # Enable some indicators
    config.features.use_rsi = True
    config.features.use_macd = True
    config.features.use_bbands = True
    config.features.use_atr = True
    config.features.use_adx = True
    config.features.use_rv_features = True
    config.features.use_regime_features = True
    
    factory = DataFactory(config)
    pipeline = FeaturePipeline(config, factory)
    
    try:
        # Load H4 data
        df = factory.load_csv("csv_data/EURUSD_10_years_H4_OANDA.csv")
        
        # Ensure required columns
        if "close" not in df.columns and "mid_close" in df.columns:
            df["close"] = df["mid_close"]
            df["high"] = df["mid_high"]
            df["low"] = df["mid_low"]
            df["open"] = df["mid_open"]
        
        # Add returns
        df["returns"] = df["close"].pct_change()
        df["price"] = df["close"]
        
        print(f"✓ Loaded {len(df)} rows of H4 data")
        
        # Build features
        df_features, features = pipeline.build_features(
            df,
            include_lags=True,
            include_rolling=True,
            shift_signals=True
        )
        
        print(f"✓ Built {len(features)} features")
        print(f"  Rows after cleaning: {len(df_features)}")
        print(f"  Sample features: {features[:10]}")
        
        # Check integrity
        df_clean, diagnostics = pipeline.check_feature_integrity(df_features, features)
        
        print(f"✓ Feature integrity check:")
        print(f"  Rows dropped: {diagnostics['rows_dropped']}")
        print(f"  Features with NaN: {len(diagnostics['nan_counts'])}")
        print(f"  Features with inf: {len(diagnostics['inf_counts'])}")
        print(f"  Zero variance features: {len(diagnostics['zero_variance'])}")
        
    except FileNotFoundError:
        print("✗ CSV file not found (expected for demo)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_multi_timeframe_pipeline():
    """Example: Multi-timeframe feature merging"""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Timeframe Feature Pipeline")
    print("=" * 60)
    
    config = load_default_config()
    config.features.use_rsi = True
    config.features.use_ema = True
    config.features.use_atr = True
    
    factory = DataFactory(config)
    pipeline = FeaturePipeline(config, factory)
    
    try:
        # Load and merge M5, H1, D1
        df_mtf = pipeline.build_multi_timeframe_features(
            timeframes=["H1", "H4"],  # H1 as base, H4 as higher TF
            source="csv",
            instrument="EURUSD",
            period="10_years",
            base_path="csv_data"
        )
        
        print(f"✓ Multi-timeframe merge complete")
        print(f"  Shape: {df_mtf.shape}")
        print(f"  Columns: {len(df_mtf.columns)}")
        
        # Show H4 features merged into H1
        h4_features = [col for col in df_mtf.columns if col.endswith("_H4")]
        print(f"  H4 features merged: {len(h4_features)}")
        if h4_features:
            print(f"  Sample H4 features: {h4_features[:5]}")
        
    except FileNotFoundError:
        print("✗ CSV files not found (expected for demo)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_composite_features():
    """Example: Composite feature builders"""
    print("=" * 60)
    print("EXAMPLE 4: Composite Features")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'close': 1.0 + 0.01 * pd.Series(range(100)).apply(lambda x: x + 10 * (x % 10)),
        'high': 1.01 + 0.01 * pd.Series(range(100)).apply(lambda x: x + 10 * (x % 10)),
        'low': 0.99 + 0.01 * pd.Series(range(100)).apply(lambda x: x + 10 * (x % 10)),
    }, index=dates)
    
    # Compute base indicators
    from src.features.indicators import compute_ema, compute_sma, compute_adx, compute_atr
    
    ema = compute_ema(df['close'], 20)
    sma = compute_sma(df['close'], 20)
    adx = compute_adx(df['high'], df['low'], df['close'], 14)
    atr = compute_atr(df['high'], df['low'], df['close'], 14)
    
    # Compute composites
    ma_spread = compute_ma_spread(ema, sma)
    trend_conf = compute_trend_confirmation(df['close'], ema, adx)
    
    print(f"✓ MA Spread computed: {len(ma_spread)} values")
    print(f"  Latest spread: {ma_spread.iloc[-1]:.6f}")
    
    print(f"✓ Trend Confirmation computed: {len(trend_conf)} values")
    print(f"  Latest value: {trend_conf.iloc[-1]:.6f}")
    
    print()


def example_regime_classification():
    """Example: Regime features (reusing indicators)"""
    print("=" * 60)
    print("EXAMPLE 5: Regime Classification")
    print("=" * 60)
    
    # Create sample data with pre-computed indicators
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'close': 1.0 + 0.01 * pd.Series(range(100)),
        'adx_14': 15 + 20 * pd.Series(range(100)) / 100,  # Rising ADX
        'rv_48': 0.0005 + 0.001 * pd.Series(range(100)) / 100,  # Rising volatility
    }, index=dates)
    
    # Compute regimes (REUSES existing columns, doesn't recalculate)
    df_regimes = compute_regime_features(
        df,
        adx_col='adx_14',
        vol_col='rv_48',
        adx_thresh=20.0,
        vol_thresh=0.001
    )
    
    print(f"✓ Regime features computed")
    print(f"  Regime distribution:")
    print(f"    SIDEWAYS: {(df_regimes['regime_id'] == 0).sum()}")
    print(f"    TREND: {(df_regimes['regime_id'] == 1).sum()}")
    print(f"    VOLATILE: {(df_regimes['regime_id'] == 2).sum()}")
    
    print()


def example_caching():
    """Example: Feature caching performance"""
    print("=" * 60)
    print("EXAMPLE 6: Feature Caching")
    print("=" * 60)
    
    config = load_default_config()
    config.features.use_rsi = True
    config.features.use_ema = True
    
    factory = DataFactory(config)
    pipeline = FeaturePipeline(config, factory)
    
    try:
        df = factory.load_csv("csv_data/EURUSD_10_years_H4_OANDA.csv")
        
        if "close" not in df.columns and "mid_close" in df.columns:
            df["close"] = df["mid_close"]
        df["returns"] = df["close"].pct_change()
        
        # Generate cache key
        cache_key = pipeline.generate_cache_key(df, config_hash="test")
        
        # First run (cache miss)
        import time
        start = time.time()
        df_feat1, feats1 = pipeline.build_features(df)
        time1 = time.time() - start
        
        # Cache the result
        pipeline.cache_features(cache_key, df_feat1, feats1)
        
        # Second run (cache hit)
        start = time.time()
        cached = pipeline.get_cached_features(cache_key)
        time2 = time.time() - start
        
        if cached:
            print(f"✓ Cache hit successful")
            print(f"  First run: {time1:.3f}s")
            print(f"  Cache retrieval: {time2:.3f}s")
            if time2 > 0:
                print(f"  Speedup: {time1/time2:.1f}x")
            else:
                print(f"  Speedup: ∞x (cache retrieval too fast to measure)")
        else:
            print("✗ Cache miss")
        
    except FileNotFoundError:
        print("✗ CSV file not found (expected for demo)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_memory_efficiency():
    """Example: Memory efficiency with float32"""
    print("=" * 60)
    print("EXAMPLE 7: Memory Efficiency (float32)")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=10000, freq='5min')
    df = pd.DataFrame({
        'close': 1.0 + 0.01 * pd.Series(range(10000)) / 100,
    }, index=dates)
    
    # Check memory before
    mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Downcast to float32
    from src.features.utils import downcast_to_float32
    df_optimized = downcast_to_float32(df)
    
    # Check memory after
    mem_after = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"✓ Memory optimization:")
    print(f"  Before (float64): {mem_before:.2f} MB")
    print(f"  After (float32): {mem_after:.2f} MB")
    print(f"  Savings: {(1 - mem_after/mem_before)*100:.1f}%")
    
    print()


def main():
    """Run all examples"""
    configure_logging()
    configure_data_logging()
    
    print("\n" + "=" * 60)
    print("PHASE 2: FEATURE ENGINEERING LAYER - EXAMPLES")
    print("=" * 60 + "\n")
    
    example_single_indicator()
    example_single_timeframe_pipeline()
    example_multi_timeframe_pipeline()
    example_composite_features()
    example_regime_classification()
    example_caching()
    example_memory_efficiency()
    
    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Modular indicator functions (vectorized)")
    print("✓ Full feature pipeline with config integration")
    print("✓ Multi-timeframe merging (LEFT JOIN, anti-look-ahead)")
    print("✓ Composite features from base indicators")
    print("✓ Regime classification (reuses existing columns)")
    print("✓ Feature caching for performance")
    print("✓ Memory efficiency with float32")
    print("\nNext Steps:")
    print("1. Test with your actual data")
    print("2. Customize indicator toggles in AppConfig")
    print("3. Experiment with multi-timeframe strategies")
    print("4. Ready for Phase 3: Model Integration Layer")
    print()


if __name__ == "__main__":
    main()
