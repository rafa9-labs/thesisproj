#!/bin/bash
# Run smoke tests via WSL with GPU acceleration (RTX 3090)
# Usage: wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_smoke_gpu.sh [model]
#   model: logistic|xgboost|cnn|lstm|transformer|ensemble_cnn_lstm_xgboost|ensemble_adaptive_regime|dqn
#   If no model specified, runs ALL models.

set -e
VENV="/mnt/c/Users/rafa/ML_Trading/thesisproj/.venv-wsl"
SITE="$VENV/lib/python3.10/site-packages"
CUDA_LIBS=$(find "$SITE/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LIBS}/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
echo "=== GPU Smoke Test Runner ==="
echo "LD_LIBRARY_PATH configured"
source "$VENV/bin/activate"
cd /mnt/c/Users/rafa/ML_Trading/thesisproj

# Verify GPU
echo ""
echo "--- GPU Detection ---"
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'GPUs: {gpus}') if gpus else print('WARNING: No GPU detected!')"

echo ""
echo "--- Running Smoke Tests ---"
MODEL="${1:-all}"

if [ "$MODEL" = "all" ]; then
    python -c "
import sys; sys.path.insert(0,'.')
from tests.smoke_all_models import run_all_models
results = run_all_models()
print()
print('=== FINAL RESULTS ===')
for m, r in results.items():
    status = 'PASS' if r else 'FAIL'
    print(f'  {m}: {status}')
passed = sum(1 for v in results.values() if v)
print(f'\n{passed}/{len(results)} models passed')
"
else
    python -c "
import sys; sys.path.insert(0,'.')
from tests.smoke_all_models import run_one_model
r = run_one_model('$MODEL')
print(f'RESULT: {\"PASS\" if r else \"FAIL\"}')"
fi