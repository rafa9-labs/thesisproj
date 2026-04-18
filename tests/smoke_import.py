"""Stage 1: verify MLBacktester imports and composes correctly."""
import sys
import os

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

SEPARATOR = "=" * 60
print(SEPARATOR)
print("STAGE 1: Import and Instantiation Smoke Test")
print(SEPARATOR)

# Step 1: Import the composed class
print("\n[1/3] Importing MLBacktester...")
from pipeline.backtester.composed import MLBacktester  # noqa: E402
print("      OK - MLBacktester imported successfully")

# Step 2: Check class composition
print("\n[2/3] Checking class composition...")
public_methods = [m for m in dir(MLBacktester) if not m.startswith("_")]
print("      OK - {} public methods found".format(len(public_methods)))

required = [
    "__init__", "__repr__", "get_data", "prepare_features",
    "scale_features", "test_strategy", "test_ensemble_strategy",
    "test_dqn_strategy", "get_model", "evaluate_strategy",
    "run_strategy", "real_trading_simulation",
]
missing = [m for m in required if not hasattr(MLBacktester, m)]
if missing:
    print("      FAIL - Missing methods: {}".format(missing))
    sys.exit(1)
print("      OK - All {} required methods present".format(len(required)))

# Step 3: Check MRO includes all 11 mixins
print("\n[3/3] Checking MRO (mixin composition)...")
mro_names = [cls.__name__ for cls in MLBacktester.__mro__]
expected_mixins = [
    "CoreMixin", "DataMixin", "FeaturesMixin", "DeepMixin",
    "StrategyMixin", "EnsembleMixin", "DQNMixin", "ModelFactoryMixin",
    "EvaluationMixin", "RunMixin", "RealTradingMixin",
]
missing_mixins = [m for m in expected_mixins if m not in mro_names]
if missing_mixins:
    print("      FAIL - Missing mixins in MRO: {}".format(missing_mixins))
    sys.exit(1)
print("      OK - All {} mixins in MRO".format(len(expected_mixins)))

print("\n" + SEPARATOR)
print("STAGE 1 PASSED - All checks OK")
print(SEPARATOR)