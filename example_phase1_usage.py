"""
Example usage of Phase 1: Data & Configuration Layer

Demonstrates:
1. Loading and saving configuration
2. Loading data from CSV
3. Fetching data from Oanda
4. Multi-timeframe data loading
"""

from datetime import datetime, timezone
from src.core.config import AppConfig, load_default_config
from src.data.factory import DataFactory, configure_logging


def example_config_usage():
    """Example: Configuration management"""
    print("=" * 60)
    print("EXAMPLE 1: Configuration Management")
    print("=" * 60)
    
    # Load default configuration
    config = load_default_config()
    print(f"✓ Loaded default config (version {config.version})")
    
    # Modify some settings
    config.features.use_rsi = True
    config.features.use_macd = True
    config.features.indicator_windows.rsi = 21
    config.cv.n_trials = 100
    print("✓ Modified indicator settings")
    
    # Save to JSON
    config.to_json("config_example.json")
    print("✓ Saved to config_example.json")
    
    # Load from JSON
    loaded_config = AppConfig.from_json("config_example.json")
    print(f"✓ Loaded from JSON: RSI enabled = {loaded_config.features.use_rsi}")
    
    # Convert to CLASS_DEFAULTS format (backward compatibility)
    class_defaults = config.to_class_defaults()
    print(f"✓ Converted to CLASS_DEFAULTS format ({len(class_defaults)} sections)")
    
    print()


def example_csv_loading():
    """Example: Load data from CSV"""
    print("=" * 60)
    print("EXAMPLE 2: CSV Data Loading")
    print("=" * 60)
    
    config = load_default_config()
    factory = DataFactory(config)
    
    try:
        # Load single CSV file
        df = factory.load_csv("csv_data/EURUSD_10_years_H4_OANDA.csv", timeframe="H4")
        print(f"✓ Loaded H4 data: {len(df)} candles")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("✗ CSV file not found (expected for demo)")
    
    print()


def example_multi_timeframe_csv():
    """Example: Load multiple timeframes from CSV"""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Timeframe CSV Loading")
    print("=" * 60)
    
    config = load_default_config()
    factory = DataFactory(config)
    
    try:
        # Load M5, H1, D1 simultaneously
        data = factory.load_multi_timeframe(
            timeframes=["M5", "H1", "D1"],
            source="csv",
            instrument="EURUSD",
            period="10_years",
            base_path="csv_data"
        )
        
        for tf, df in data.items():
            if not df.empty:
                print(f"✓ {tf}: {len(df)} candles")
    except Exception as e:
        print(f"✗ Multi-timeframe loading failed: {e}")
    
    print()


def example_oanda_fetch():
    """Example: Fetch data from Oanda API"""
    print("=" * 60)
    print("EXAMPLE 4: Oanda API Data Fetching")
    print("=" * 60)
    
    config = load_default_config()
    
    if not config.oanda.access_token:
        print("✗ Oanda credentials not configured (skipping)")
        print("  Configure oanda.cfg or set OANDA_ACCESS_TOKEN environment variable")
        print()
        return
    
    factory = DataFactory(config)
    
    try:
        # Fetch 1 week of H1 data
        start = datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 11, 8, 0, 0, 0, tzinfo=timezone.utc)
        
        df = factory.fetch_oanda(
            instrument="EUR_USD",
            granularity="H1",
            start=start,
            end=end,
            is_live=False
        )
        
        print(f"✓ Fetched {len(df)} H1 candles from Oanda")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        
    except Exception as e:
        print(f"✗ Oanda fetch failed: {e}")
    
    print()


def example_live_mode_preparation():
    """Example: Prepare for live mode (future use)"""
    print("=" * 60)
    print("EXAMPLE 5: Live Mode Preparation")
    print("=" * 60)
    
    config = load_default_config()
    factory = DataFactory(config)
    
    print("Live mode interface ready:")
    print("  factory.get_data(source='oanda', is_live=True, ...)")
    print("  Note: Streaming not yet implemented, will fetch historical data")
    print()


def main():
    """Run all examples"""
    # Configure logging to see DataFactory messages
    configure_logging()
    
    print("\n" + "=" * 60)
    print("PHASE 1: DATA & CONFIGURATION LAYER - EXAMPLES")
    print("=" * 60 + "\n")
    
    example_config_usage()
    example_csv_loading()
    example_multi_timeframe_csv()
    example_oanda_fetch()
    example_live_mode_preparation()
    
    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Review generated config_example.json")
    print("2. Test with your actual CSV files")
    print("3. Verify Oanda API connection")
    print("4. Ready for Phase 2: Feature Engineering Layer")
    print()


if __name__ == "__main__":
    main()
