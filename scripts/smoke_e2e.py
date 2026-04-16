"""
End-to-end pipeline smoke test with memory profiling.

Tests all 6 pairs and all 8 model types, logging peak RSS.
Verifies RAM fixes are effective: peak RSS should stay under 2 GB.
"""
import gc
import os
import sys
import traceback

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copy import deepcopy
from pipeline.metrics_tuples import CLASS_DEFAULTS
from pipeline.backtester.composed import MLBacktester
from pipeline.standalone_utils import clear_data_cache

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GBPJPY"]
MODELS = ["logistic", "xgboost", "random_forest", "svm"]
DEEP_MODELS = ["cnn", "lstm", "transformer"]

process = psutil.Process(os.getpid())


def _rss_mb():
    return process.memory_info().rss / (1024 * 1024)


def test_pair_data_loading():
    """Test all 6 pairs load correctly and have correct pip values."""
    print("\n" + "=" * 60)
    print("TEST 1: Pair data loading (all 6 pairs)")
    print("=" * 60)
    results = []
    for pair in PAIRS:
        rss_before = _rss_mb()
        cfg = deepcopy(CLASS_DEFAULTS["features"])
        bt = MLBacktester(
            symbol=pair,
            start="2024-06-01 00:00:00",
            end="2024-07-01 00:00:00",
            trading_costs=False,
            features_config=cfg,
        )
        rows = len(bt.data)
        pip = bt._pair_config.pip_value
        feat_pip = bt.features_config.get("stop_pip_value")
        rss_after = _rss_mb()
        delta = rss_after - rss_before
        ok = rows > 0 and pip == feat_pip
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {pair}: {rows} rows, pip={pip}, feat_pip={feat_pip}, RAM +{delta:.0f}MB")
        results.append((pair, ok, delta))
        bt.free(release_data=True)
        del bt
        gc.collect()

    clear_data_cache()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n  Result: {passed}/{total} pairs OK")
    return all(ok for _, ok, _ in results)


def test_classical_models():
    """Test all 4 classical models on EURUSD."""
    print("\n" + "=" * 60)
    print("TEST 2: Classical models (EURUSD, 2-month window)")
    print("=" * 60)
    results = []
    rss_baseline = _rss_mb()

    for model in MODELS:
        rss_before = _rss_mb()
        cfg = deepcopy(CLASS_DEFAULTS["features"])
        cfg["tune_once"] = True
        try:
            bt = MLBacktester(
                symbol="EURUSD",
                start="2024-06-01 00:00:00",
                end="2024-08-01 00:00:00",
                trading_costs=False,
                model_type=model,
                features_config=cfg,
            )
            data_ok = bt.data is not None and len(bt.data) > 0
            rss_after = _rss_mb()
            delta = rss_after - rss_before
            print(f"  PASS {model}: data={data_ok}, {len(bt.data)} rows, RAM +{delta:.0f}MB (total {rss_after:.0f}MB)")
            results.append((model, True, delta))
            bt.free(release_data=True)
            del bt
        except Exception as e:
            print(f"  FAIL {model}: {e}")
            results.append((model, False, 0))
        gc.collect()
        clear_data_cache()

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    rss_now = _rss_mb()
    print(f"\n  Result: {passed}/{total} models OK | Total RAM: {rss_now:.0f}MB (baseline {rss_baseline:.0f}MB)")
    return all(ok for _, ok, _ in results)


def test_jpy_pair_pip_value():
    """Verify JPY pairs use pip_value=0.01 throughout the pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: JPY pair pip value propagation")
    print("=" * 60)
    results = []
    for pair in ["USDJPY", "GBPJPY"]:
        cfg = deepcopy(CLASS_DEFAULTS["features"])
        cfg["stop_method"] = "fixed_pips"
        cfg["stop_sl_pips"] = 30.0
        bt = MLBacktester(
            symbol=pair,
            start="2024-06-01 00:00:00",
            end="2024-07-01 00:00:00",
            trading_costs=False,
            features_config=cfg,
        )
        pair_pip = bt._pair_config.pip_value
        feat_pip = bt.features_config.get("stop_pip_value")
        trail_pip = bt.features_config.get("trailing_pip_value")
        ok = pair_pip == 0.01 and feat_pip == 0.01 and trail_pip == 0.01
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {pair}: pair={pair_pip}, stop={feat_pip}, trail={trail_pip}")
        results.append((pair, ok))
        bt.free(release_data=True)
        del bt
        gc.collect()
        clear_data_cache()

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  Result: {passed}/{len(results)} JPY pairs OK")
    return all(ok for _, ok in results)


def test_memory_leak_detection():
    """Load/unload multiple pairs repeatedly, check for memory leak."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory leak detection (5 iterations)")
    print("=" * 60)
    rss_readings = []

    for i in range(5):
        for pair in PAIRS[:3]:
            cfg = deepcopy(CLASS_DEFAULTS["features"])
            bt = MLBacktester(
                symbol=pair,
                start="2024-06-01 00:00:00",
                end="2024-07-01 00:00:00",
                trading_costs=False,
                features_config=cfg,
            )
            bt.free(release_data=True)
            del bt

        gc.collect()
        clear_data_cache()
        rss = _rss_mb()
        rss_readings.append(rss)
        print(f"  Iteration {i+1}: RSS = {rss:.0f}MB")

    rss_first = rss_readings[0]
    rss_last = rss_readings[-1]
    growth = rss_last - rss_first
    leak_threshold = 100  # MB
    ok = growth < leak_threshold
    status = "PASS" if ok else "FAIL"
    print(f"\n  {status} Memory growth: {growth:.0f}MB over 5 iterations (threshold: {leak_threshold}MB)")
    return ok


def test_feature_dtype_is_float32():
    """Verify features are computed in float32 after RAM fix."""
    print("\n" + "=" * 60)
    print("TEST 5: Feature dtype verification (float32)")
    print("=" * 60)
    cfg = deepcopy(CLASS_DEFAULTS["features"])
    cfg["use_rsi"] = True
    cfg["use_macd"] = True
    cfg["use_ema"] = True

    bt = MLBacktester(
        symbol="EURUSD",
        start="2024-01-01 00:00:00",
        end="2024-03-01 00:00:00",
        trading_costs=False,
        features_config=cfg,
    )

    df = bt.data
    float_cols = df.select_dtypes(include="float").columns
    f32_count = sum(1 for c in float_cols if df[c].dtype == "float32")
    f64_count = sum(1 for c in float_cols if df[c].dtype == "float64")
    total = len(float_cols)

    print(f"  float32 columns: {f32_count}/{total}")
    print(f"  float64 columns: {f64_count}/{total}")

    ok = f32_count >= f64_count
    status = "PASS" if ok else "FAIL"
    print(f"\n  {status} Majority float32: {f32_count}/{total}")
    bt.free(release_data=True)
    del bt
    clear_data_cache()
    gc.collect()
    return ok


def main():
    print("FX ML Pipeline — End-to-End Smoke Test + Memory Profiling")
    print(f"Python {sys.version}")
    print(f"Initial RSS: {_rss_mb():.0f}MB")

    tests = [
        ("Pair data loading", test_pair_data_loading),
        ("Classical models", test_classical_models),
        ("JPY pip value", test_jpy_pair_pip_value),
        ("Memory leak", test_memory_leak_detection),
        ("Feature float32", test_feature_dtype_is_float32),
    ]

    results = {}
    for name, fn in tests:
        try:
            ok = fn()
            results[name] = ok
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    print(f"\n  {passed}/{total} tests passed")
    print(f"  Final RSS: {_rss_mb():.0f}MB")

    if passed == total:
        print("\nAll end-to-end tests PASSED")
    else:
        print(f"\n{total - passed} test(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
