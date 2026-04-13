#!/bin/bash
# =============================================================
# run_wsl.sh — Launch thesisproj with GPU acceleration in WSL2
#
# Usage:
#   wsl -e bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_wsl.sh [command]
#
# Examples:
#   ./run_wsl.sh                    # Activate venv with GPU support
#   ./run_wsl.sh python app.py      # Run Streamlit app
#   ./run_wsl.sh python -m pipeline.main_cli --model cnn --months 3
# =============================================================

set -e

PROJ="/mnt/c/Users/rafa/ML_Trading/thesisproj"
VENV="$PROJ/.venv-wsl"
SITE="$VENV/lib/python3.10/site-packages"

# ── Auto-discover CUDA libs from pip nvidia packages ──
CUDA_LIBS=$(find "$SITE/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LIBS}/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

# ── Activate venv ──
source "$VENV/bin/activate"
cd "$PROJ"

# ── Auto-detect GPU for XGBoost ──
if python -c "import xgboost; xgboost.build_info()['USE_CUDA']" 2>/dev/null; then
    export XGB_USE_GPU=1
    export XGB_DEVICE=cuda
    GPU_STATUS="✅ XGBoost GPU enabled (cuda)"
else
    export XGB_USE_GPU=0
    GPU_STATUS="⚪ XGBoost CPU-only (no CUDA)"
fi

echo "🚀 WSL2 GPU Environment Ready"
echo "   CUDA libs: $(echo $CUDA_LIBS | tr ':' '\n' | wc -l) paths"
echo "   XGBoost:   $GPU_STATUS"
echo "   Project:   $PROJ"
echo ""

# ── Run command or drop into shell ──
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "Interactive shell with GPU support. Type 'exit' to leave."
    echo "Quick tests:"
    echo "  python test_gpu_wsl.py          # Verify GPU"
    echo "  python -m streamlit run app.py  # Launch UI"
    exec bash
fi