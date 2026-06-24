"""Quick smoke test for multi-currency pipeline."""
from pipeline.backtester.composed import MLBacktester
from copy import deepcopy
from pipeline.metrics.metrics_tuples import CLASS_DEFAULTS

cfg = deepcopy(CLASS_DEFAULTS["features"])

for pair in ["GBPUSD", "USDJPY"]:
    print(f"\n--- Testing {pair} ---")
    bt = MLBacktester(
        symbol=pair,
        start="2024-01-01 00:00:00",
        end="2024-03-01 00:00:00",
        trading_costs=False,
        features_config=cfg,
    )
    print(f"  Data loaded: {len(bt.data)} rows")
    print(f"  Pip value: {bt._pair_config.pip_value}")
    print(f"  Stop pip_value: {bt.features_config.get('stop_pip_value')}")
    print(f"  Trailing pip_value: {bt.features_config.get('trailing_pip_value')}")
    print(f"  {pair} PASSED")

print("\nAll multi-currency smoke tests PASSED")
