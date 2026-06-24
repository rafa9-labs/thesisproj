"""Stage 2: Full pipeline smoke test via pipeline.main_cli.main()

Runs logistic + xgboost for 3 months as configured in main_cli.py.
This exercises the entire mixin chain: init -> data -> features ->
strategy -> evaluation -> real_trading_simulation.
"""
import sys
import os
import time

# ── Force unbuffered output (works regardless of how script is invoked) ──
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

SEPARATOR = "=" * 60

# ── Enable smoke-test mode (1 model, 1 month, 2 trials) ──
os.environ["SMOKE_TEST"] = "1"

# ── Pre-flight diagnostics ──
print(SEPARATOR, flush=True)
print("STAGE 2: Full Pipeline Smoke Test (SMOKE_TEST=1)", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD:    {os.getcwd()}", flush=True)
print(f"ROOT:   {_project_root}", flush=True)
print(SEPARATOR, flush=True)

# Check TensorFlow / GPU
print("\n── Pre-flight: TensorFlow & GPU ──", flush=True)
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU devices found: {gpus}", flush=True)
    else:
        print("No GPU devices visible to TensorFlow.", flush=True)
        print("  Tip: On Windows, TF >= 2.11 has no native GPU support.", flush=True)
        print("  Options: tensorflow-directml, WSL2, or use only sklearn models.", flush=True)
except ImportError:
    print("TensorFlow not installed — deep models (LSTM/CNN/Transformer) unavailable.", flush=True)
except Exception as e:
    print(f"TensorFlow import error: {e}", flush=True)

# Check critical function availability
print("\n── Pre-flight: Critical imports ──", flush=True)
try:
    from pipeline.metrics.metrics_tuples import _safe_metrics_return, _empty_metrics, N_METRICS
    print(f"OK: _safe_metrics_return (arity={N_METRICS})", flush=True)
except Exception as e:
    print(f"FAIL: cannot import _safe_metrics_return: {e}", flush=True)
    sys.exit(1)

try:
    from pipeline.metrics.metrics_eval import compute_full_evaluation_metrics
    print("OK: compute_full_evaluation_metrics", flush=True)
except Exception as e:
    print(f"FAIL: cannot import compute_full_evaluation_metrics: {e}", flush=True)
    sys.exit(1)

try:
    from pipeline._imports import CLASS_DEFAULTS
    print("OK: CLASS_DEFAULTS", flush=True)
except Exception as e:
    print(f"FAIL: cannot import CLASS_DEFAULTS: {e}", flush=True)
    sys.exit(1)

print("\n── Starting pipeline ──", flush=True)
print("Model: logistic only | Months: 1 | Trials: 2 | Seed: 33333", flush=True)
print(SEPARATOR, flush=True)

from pipeline.main_cli import main  # noqa: E402

print("\n[STARTING] Calling main() ...", flush=True)
print("-" * 60, flush=True)

t0 = time.perf_counter()
try:
    main()
    elapsed = time.perf_counter() - t0
    print("-" * 60, flush=True)
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(SEPARATOR, flush=True)
    print("STAGE 2 PASSED - Full pipeline completed", flush=True)
    print(SEPARATOR, flush=True)
except Exception as e:
    elapsed = time.perf_counter() - t0
    print("-" * 60, flush=True)
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(SEPARATOR, flush=True)
    print("STAGE 2 FAILED", flush=True)
    print("Error: {}".format(e), flush=True)
    print(SEPARATOR, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)