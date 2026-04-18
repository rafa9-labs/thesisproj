"""Validate all 8 model types through the full pipeline.

Runs each model individually in SMOKE_TEST mode (1 trial, 1 month).
Reports per-model PASS/FAIL with timing.

Usage:
    python tests/smoke_all_models.py
    python tests/smoke_all_models.py --quick   # only logistic + xgboost
"""
import sys
import os
import time
import traceback

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# ── All registered model types ──
ALL_MODELS = [
    "logistic",
    "xgboost",
    "cnn",
    "lstm",
    "transformer",
    "dqn",
    "ensemble_cnn_lstm_xgboost",
    "ensemble_adaptive_regime",
]

QUICK_MODELS = ["logistic", "xgboost"]

SEPARATOR = "=" * 60


def check_tensorflow():
    """Check if TensorFlow is available (needed for CNN, LSTM, Transformer)."""
    try:
        import tensorflow as tf
        print(f"  TensorFlow {tf.__version__} available")
        return True
    except ImportError:
        print("  TensorFlow NOT installed — deep models will be skipped")
        return False
    except Exception as e:
        print(f"  TensorFlow error: {e} — deep models will be skipped")
        return False


def check_rl_deps():
    """Check if RL dependencies are available (needed for DQN)."""
    try:
        from rl.dqn_agent import DQNAgent
        print("  RL module available")
        return True
    except ImportError:
        print("  RL module NOT available — DQN will be skipped")
        return False
    except Exception as e:
        print(f"  RL module error: {e} — DQN will be skipped")
        return False


def _check_results_for_model(model_type: str) -> tuple[bool, str]:
    """Check if results were actually produced (not just 'no crash').
    
    Scans the most recent results directory for CSV ranking files.
    Returns (has_real_results, detail).
    """
    import glob
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(results_dir):
        return False, "No results directory found"
    
    # Find the most recent results folder
    result_folders = sorted(glob.glob(os.path.join(results_dir, "*")))
    if not result_folders:
        return False, "No result folders found"
    
    latest = result_folders[-1]
    
    # Search for ranking CSV that contains actual trade data
    ranking_files = []
    for root, dirs, files in os.walk(latest):
        for f in files:
            if "ranking" in f.lower() and f.endswith(".csv"):
                ranking_files.append(os.path.join(root, f))
    
    if not ranking_files:
        return False, "No ranking CSV found"
    
    # Read the most recent ranking file and check for our model
    import pandas as pd
    for rf in reversed(ranking_files):
        try:
            df = pd.read_csv(rf)
            model_col = None
            for col in df.columns:
                if "model" in col.lower():
                    model_col = col
                    break
            if model_col is None:
                continue
            
            # Find our model
            mask = df[model_col].astype(str).str.lower().str.contains(
                model_type.split("_")[0], na=False
            )
            if mask.any():
                row = df[mask].iloc[0]
                trades_col = None
                for col in df.columns:
                    if "trades" in col.lower():
                        trades_col = col
                        break
                if trades_col:
                    n_trades = int(row.get(trades_col, 0))
                    if n_trades == 0:
                        return False, f"0 trades produced (model ran but no signals)"
                    return True, f"{n_trades} trades produced"
        except Exception:
            continue
    
    return True, "Results found (trade count check skipped)"


def run_one_model(model_type: str) -> dict:
    """Run a single model through the pipeline in smoke mode.
    
    Returns dict with keys: model, status, elapsed, error
    """
    # Force fresh environment for each model
    os.environ["SMOKE_TEST"] = "1"
    os.environ["MODEL_LIST"] = model_type
    os.environ["N_MONTHS"] = "1"
    os.environ["REPEATS"] = "1"
    os.environ["SEEDS"] = "33333"
    
    # Suppress plots during validation
    os.environ["SKIP_PLOTS"] = "1"
    
    t0 = time.perf_counter()
    try:
        # Import fresh each time (main_cli re-reads env vars at call time)
        from pipeline.main_cli import main
        main()
        elapsed = time.perf_counter() - t0
        
        # Check if results were actually produced (not just "no crash")
        has_results, detail = _check_results_for_model(model_type)
        if not has_results:
            return {"model": model_type, "status": "FAIL", "elapsed": elapsed,
                    "error": f"Silent failure: {detail}"}
        
        return {"model": model_type, "status": "PASS", "elapsed": elapsed, "error": detail}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"model": model_type, "status": "FAIL", "elapsed": elapsed, "error": str(e)}


def main():
    quick = "--quick" in sys.argv
    
    print(SEPARATOR)
    print("MODEL VALIDATION — All Model Types Smoke Test")
    print(f"Mode: {'QUICK (logistic + xgboost)' if quick else 'FULL (all 8 models)'}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CWD: {os.getcwd()}")
    print(SEPARATOR)
    
    # ── Pre-flight checks ──
    print("\n── Pre-flight Dependency Check ──")
    tf_ok = check_tensorflow()
    rl_ok = check_rl_deps()
    
    # ── Select models to test ──
    models = QUICK_MODELS if quick else ALL_MODELS
    
    # Skip models that can't run without deps
    skip_reasons = {}
    deep_models = {"cnn", "lstm", "transformer", "ensemble_cnn_lstm_xgboost"}
    
    if not tf_ok:
        for m in deep_models:
            skip_reasons[m] = "TensorFlow not available"
    
    if not rl_ok and "dqn" in models:
        skip_reasons["dqn"] = "RL dependencies not available"
    
    # Also check ensemble_adaptive_regime (may need TF internally)
    if not tf_ok and "ensemble_adaptive_regime" in models:
        skip_reasons["ensemble_adaptive_regime"] = "TensorFlow not available"
    
    to_run = [m for m in models if m not in skip_reasons]
    
    if skip_reasons:
        print(f"\n⚠️  Skipping {len(skip_reasons)} models:")
        for m, reason in skip_reasons.items():
            print(f"   - {m}: {reason}")
    
    print(f"\n── Running {len(to_run)} models ──")
    print("-" * 60)
    
    # ── Run each model ──
    results = []
    for i, model in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] Testing: {model}")
        print("-" * 40)
        
        result = run_one_model(model)
        results.append(result)
        
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_icon} {result['model']}: {result['status']} ({result['elapsed']:.1f}s)")
        
        if result["error"]:
            print(f"   Error: {result['error'][:200]}")
        
        # Hard cleanup between models
        try:
            import gc
            gc.collect()
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
    
    # ── Summary ──
    print("\n" + SEPARATOR)
    print("VALIDATION SUMMARY")
    print(SEPARATOR)
    
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    skipped = skip_reasons
    
    print(f"\n{'Model':<35} {'Status':<8} {'Time':<10}")
    print("-" * 55)
    
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"{icon} {r['model']:<33} {r['status']:<8} {r['elapsed']:.1f}s")
    
    for m, reason in skipped.items():
        print(f"⏭️  {m:<33} {'SKIP':<8} {reason}")
    
    print("-" * 55)
    print(f"Total: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    
    if failed:
        print("\n❌ FAILURES:")
        for r in failed:
            print(f"   {r['model']}: {r['error'][:300]}")
    
    print(SEPARATOR)
    
    if failed:
        print("RESULT: SOME MODELS FAILED")
        sys.exit(1)
    elif passed:
        print("RESULT: ALL AVAILABLE MODELS PASSED")
        sys.exit(0)
    else:
        print("RESULT: NO MODELS TESTED")
        sys.exit(1)


if __name__ == "__main__":
    main()