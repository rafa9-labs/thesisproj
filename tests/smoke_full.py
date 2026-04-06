"""Stage 2: Full pipeline smoke test via pipeline.main_cli.main()

Runs logistic + xgboost for 3 months as configured in main_cli.py.
This exercises the entire mixin chain: init -> data -> features ->
strategy -> evaluation -> real_trading_simulation.
"""
import sys
import os

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

SEPARATOR = "=" * 60

# ── Enable smoke-test mode (1 model, 1 month, 2 trials) ──
os.environ["SMOKE_TEST"] = "1"

print(SEPARATOR)
print("STAGE 2: Full Pipeline Smoke Test (SMOKE_TEST=1)")
print("Model: logistic only | Months: 1 | Trials: 2 | Seed: 33333")
print(SEPARATOR)

from pipeline.main_cli import main  # noqa: E402

print("\n[STARTING] Calling main() ...")
print("-" * 60)

try:
    main()
    print("-" * 60)
    print("\n" + SEPARATOR)
    print("STAGE 2 PASSED - Full pipeline completed")
    print(SEPARATOR)
except Exception as e:
    print("-" * 60)
    print("\n" + SEPARATOR)
    print("STAGE 2 FAILED")
    print("Error: {}".format(e))
    print(SEPARATOR)
    import traceback
    traceback.print_exc()
    sys.exit(1)