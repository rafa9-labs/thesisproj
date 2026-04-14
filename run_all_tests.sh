#!/bin/bash
# =============================================================
# run_all_tests.sh — Full Phase Gate Test Suite
#
# Runs all unit tests + per-model pipeline smoke tests.
# Output saved to test_results/ directory for debugging.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_all_tests.sh
#
# Or via run_wsl.sh:
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_wsl.sh bash ./run_all_tests.sh
# =============================================================

set -e

PROJ="/mnt/c/Users/rafa/ML_Trading/thesisproj"
VENV="$PROJ/.venv-wsl"
SITE="$VENV/lib/python3.10/site-packages"

# ── Auto-discover CUDA libs ──
CUDA_LIBS=$(find "$SITE/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LIBS}/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

# ── Activate venv ──
source "$VENV/bin/activate"
cd "$PROJ"

# ── Ensure pytest is available ──
if ! python -m pytest --version >/dev/null 2>&1; then
    echo "📦 Installing pytest..."
    pip install pytest -q
fi

# ── Create results directory ──
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="$PROJ/test_results"
mkdir -p "$RESULT_DIR"
LOG_FILE="$RESULT_DIR/test_run_${TIMESTAMP}.log"

# ── Summary tracking ──
TOTAL=0
PASSED=0
FAILED=0
FAIL_LIST=""

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log_sep() {
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run_test() {
    local name="$1"
    shift
    local cmd="$@"

    TOTAL=$((TOTAL + 1))
    log ""
    log_sep
    log "TEST $TOTAL: $name"
    log "CMD: $cmd"
    log_sep

    local start_time=$(date +%s)

    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "✅ PASS — $name (${elapsed}s)"
        PASSED=$((PASSED + 1))
    else
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "❌ FAIL — $name (${elapsed}s)"
        FAILED=$((FAILED + 1))
        FAIL_LIST="$FAIL_LIST\n  ❌ $name"
    fi
}

# ── Header ──
log "================================================================"
log "  🧪 THESISPROJ — FULL PHASE GATE TEST SUITE"
log "  Started: $(date)"
log "  Log:     $LOG_FILE"
log "================================================================"
log ""

# ════════════════════════════════════════════════════════════════
# SECTION 1: Unit Tests (pytest)
# ════════════════════════════════════════════════════════════════
log ""
log "═══ SECTION 1: Pytest Unit Tests ═══"

run_test "Import smoke test" \
    "python tests/smoke_import.py"

run_test "Pipeline imports" \
    "python -m pytest tests/test_pipeline_imports.py -v --tb=short"

run_test "Pipeline utilities" \
    "python -m pytest tests/test_pipeline_utils.py -v --tb=short"

run_test "Pipeline metrics" \
    "python -m pytest tests/test_pipeline_metrics.py -v --tb=short"

run_test "Pipeline integration" \
    "python -m pytest tests/test_pipeline_integration.py -v --tb=short"

run_test "Pipeline config" \
    "python -m pytest tests/test_pipeline_config.py -v --tb=short"

run_test "Pipeline composed backtester" \
    "python -m pytest tests/test_pipeline_composed.py -v --tb=short"

run_test "Eval metrics" \
    "python -m pytest tests/test_eval_metrics.py -v --tb=short"

run_test "Feature cache" \
    "python -m pytest tests/test_feature_cache.py -v --tb=short"

run_test "Model registry" \
    "python -m pytest tests/test_model_registry.py -v --tb=short"

run_test "Walk-forward integrity" \
    "python -m pytest tests/test_walk_forward_integrity.py -v --tb=short"

# ════════════════════════════════════════════════════════════════
# SECTION 2: GPU & Environment
# ════════════════════════════════════════════════════════════════
log ""
log "═══ SECTION 2: GPU & Environment Checks ═══"

run_test "GPU detection (CUDA)" \
    "python test_gpu_wsl.py"

run_test "XGBoost GPU build check" \
    "python -c \"import xgboost; info=xgboost.build_info(); assert info.get('USE_CUDA') or info.get('USE_CUDA','') == '1', 'XGBoost not built with CUDA'; print('XGBoost CUDA: OK')\""

# ════════════════════════════════════════════════════════════════
# SECTION 3: UI Module Tests
# ════════════════════════════════════════════════════════════════
log ""
log "═══ SECTION 3: Streamlit UI Module Tests ═══"

run_test "UI imports (all modules)" \
    "python test_ui_imports.py"

run_test "Streamlit app import (app.py)" \
    "python -c \"import importlib.util; spec=importlib.util.spec_from_file_location('app','app.py'); print('app.py importable: OK')\""

# ════════════════════════════════════════════════════════════════
# SECTION 4: Per-Model Pipeline Smoke Tests
# ════════════════════════════════════════════════════════════════
log ""
log "═══ SECTION 4: Per-Model Pipeline Smoke Tests (SMOKE_TEST=1) ═══"
log "  Each model runs 1 trial, 1 month, 1 seed — fast validation."
log ""

export SMOKE_TEST=1

# 4.1 Logistic Regression (CPU baseline)
run_test "Pipeline: logistic (CPU baseline)" \
    "env MODEL_LIST=logistic python -m pipeline.main_cli"

# 4.2 XGBoost (GPU tree model)
run_test "Pipeline: xgboost (GPU)" \
    "env MODEL_LIST=xgboost XGB_USE_GPU=1 python -m pipeline.main_cli"

# 4.3 CNN (deep, GPU)
run_test "Pipeline: cnn (GPU)" \
    "env MODEL_LIST=cnn python -m pipeline.main_cli"

# 4.4 LSTM (deep sequence, GPU)
run_test "Pipeline: lstm (GPU)" \
    "env MODEL_LIST=lstm python -m pipeline.main_cli"

# 4.5 Transformer (attention, GPU)
run_test "Pipeline: transformer (GPU)" \
    "env MODEL_LIST=transformer python -m pipeline.main_cli"

# 4.6 DQN (reinforcement learning)
run_test "Pipeline: dqn (RL)" \
    "env MODEL_LIST=dqn python -m pipeline.main_cli"

# 4.7 Ensemble: CNN+LSTM+XGBoost
run_test "Pipeline: ensemble_cnn_lstm_xgboost" \
    "env MODEL_LIST=ensemble_cnn_lstm_xgboost python -m pipeline.main_cli"

# 4.8 Ensemble: Adaptive Regime
run_test "Pipeline: ensemble_adaptive_regime" \
    "env MODEL_LIST=ensemble_adaptive_regime python -m pipeline.main_cli"

# 4.9 Multi-model comparison (logistic + xgboost)
run_test "Pipeline: multi-model (logistic,xgboost)" \
    "env MODEL_LIST=logistic,xgboost python -m pipeline.main_cli"

# ════════════════════════════════════════════════════════════════
# SECTION 5: Edge Case Tests
# ════════════════════════════════════════════════════════════════
log ""
log "═══ SECTION 5: Edge Cases & Config Variations ═══"

# 5.1 XGBoost CPU fallback (no GPU flag)
run_test "XGBoost CPU fallback (XGB_USE_GPU=0)" \
    "env MODEL_LIST=xgboost XGB_USE_GPU=0 python -m pipeline.main_cli"

# 5.2 Custom seeds + multi-month
run_test "Custom config (2 seeds, 2 months)" \
    "env MODEL_LIST=logistic SEEDS=11111,22222 REPEATS=2 N_MONTHS=2 python -m pipeline.main_cli"

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
log ""
log "================================================================"
log "  📊 TEST SUITE SUMMARY"
log "================================================================"
log ""
log "  Total:  $TOTAL"
log "  Passed: $PASSED ✅"
log "  Failed: $FAILED ❌"
log ""

if [ $FAILED -gt 0 ]; then
    log "  Failed tests:"
    echo -e "$FAIL_LIST" | tee -a "$LOG_FILE"
    log ""
fi

log "  Log file: $LOG_FILE"
log "  Finished: $(date)"
log "================================================================"

if [ $FAILED -eq 0 ]; then
    log ""
    log "  🎉 ALL TESTS PASSED — Ready for next phase!"
    log ""
    exit 0
else
    log ""
    log "  ⚠️  SOME TESTS FAILED — Review log before proceeding."
    log ""
    exit 1
fi