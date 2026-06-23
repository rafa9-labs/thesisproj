"""
Runtime performance knobs -- thread budgets, GPU init, BLAS caps.

Extracted from MLBacktesterNoWFO.py lines 196-294.
Imports config.py for centralized settings.
"""
import os
import sys
import glob

from config import get_settings, apply_global_env

# -- WSL2 CUDA auto-configuration --
# TF on WSL2 needs LD_LIBRARY_PATH pointing to pip-installed nvidia CUDA libs.
# This block auto-discovers them from the active venv's site-packages.
def _configure_wsl2_cuda():
    """Set LD_LIBRARY_PATH for CUDA libs in WSL2 environments."""
    try:
        is_wsl = "microsoft" in os.uname().release.lower()
    except (AttributeError, OSError):
        return  # Windows native -- no uname
    if not is_wsl:
        return
    if os.environ.get("LD_LIBRARY_PATH", "").count("nvidia") >= 3:
        return  # Already configured

    site = os.path.join(os.path.dirname(sys.executable), "..", "lib",
                        f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages")
    site = os.path.normpath(site)
    nvidia_base = os.path.join(site, "nvidia")
    if not os.path.isdir(nvidia_base):
        return

    cuda_dirs = glob.glob(os.path.join(nvidia_base, "*", "lib"))
    if not cuda_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_path = ":".join(cuda_dirs) + ":/usr/lib/wsl/lib:" + existing
    os.environ["LD_LIBRARY_PATH"] = new_path

_configure_wsl2_cuda()

# -- pandas pref --
try:
    import pandas as pd
    pd.options.mode.copy_on_write = True
except Exception:
    pass

# -- Apply centralized config --
_settings = get_settings()
apply_global_env(_settings)

# -- psutil --
try:
    import psutil
except Exception:
    psutil = None

# -- threadpoolctl --
try:
    from threadpoolctl import threadpool_limits as _tp_limits
except Exception:
    _tp_limits = None

# -- GPU allow-growth --
# Lazy: skip TF import entirely when only sklearn models are used.
# Set TF_SKIP_INIT=1 to avoid the heavy TF load for logistic/xgboost runs.
_TF_SKIP = os.environ.get("TF_SKIP_INIT", "0") == "1"

if not _TF_SKIP:
    _vram_limit = os.environ.get("CUDA_VRAM_LIMIT_MB", "").strip()
    if not _vram_limit:
        try:
            import tensorflow as _tf
            for g in _tf.config.list_physical_devices("GPU"):
                _tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # -- Force-CPU escape --
    if os.environ.get("TF_FORCE_CPU", "0") == "1":
        try:
            import tensorflow as _tf2
            _tf2.config.set_visible_devices([], "GPU")
        except Exception:
            pass

# -- Compute safe core count --
BLAS_THREADS = _settings.compute.blas_threads
SAFE_CORES = _settings.compute.safe_cores
CPU_TOTAL = _settings.compute.cpu_total

# -- Apply BLAS / OpenMP cap --
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=BLAS_THREADS)
except Exception:
    pass

# -- TensorFlow thread tuning --
if not _TF_SKIP:
    try:
        import tensorflow as _tf3
        _tf3.config.threading.set_intra_op_parallelism_threads(BLAS_THREADS)
        _tf3.config.threading.set_inter_op_parallelism_threads(max(2, BLAS_THREADS // 4))
    except Exception:
        pass

try:
    print(f"Trial thread budget = {BLAS_THREADS} cores active per model fit.")
except UnicodeEncodeError:
    pass


# -- Public GPU status helper --
# Heavy models (CNN, LSTM, Transformer, DQN) should run on GPU when available.
# Usage:  from pipeline.runtime import gpu_status, GPU_DEVICES
#         info = gpu_status()

GPU_AVAILABLE = False
GPU_DEVICES: list[str] = []

def _detect_gpu():
    """Detect available GPUs without importing TF eagerly."""
    global GPU_AVAILABLE, GPU_DEVICES
    try:
        import tensorflow as _tf_g
        GPU_DEVICES = [d.name for d in _tf_g.config.list_physical_devices("GPU")]
        GPU_AVAILABLE = len(GPU_DEVICES) > 0
    except Exception:
        GPU_AVAILABLE = False
        GPU_DEVICES = []

# Lazy detection -- only when first queried
_gpu_checked = False

def gpu_status() -> dict:
    """Return GPU status dict.  Heavy models should warn if GPU_AVAILABLE is False."""
    global _gpu_checked
    if not _gpu_checked:
        _detect_gpu()
        _gpu_checked = True
    return {
        "available": GPU_AVAILABLE,
        "devices": GPU_DEVICES,
        "mode": "GPU" if GPU_AVAILABLE else "CPU",
    }

# Models that benefit significantly from GPU acceleration
GPU_RECOMMENDED_MODELS = {"cnn", "lstm", "transformer", "dqn"}


def apply_vram_lock():
    """Apply CUDA_VRAM_LIMIT_MB as a hard logical device memory bound.

    Called once per process before any TF model is loaded.
    If CUDA_VRAM_LIMIT_MB is not set, falls back to memory growth.

    This ensures a process cannot exceed its allocated VRAM budget,
    preventing OOM from concurrent GPU backtests.
    """
    vram_limit = os.environ.get("CUDA_VRAM_LIMIT_MB", "").strip()
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return

        if vram_limit:
            limit_mb = int(vram_limit)
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
                )
                print(f"[VRAM] GPU locked to {limit_mb} MB via logical device config")
                return
            except Exception as exc:
                print(f"[VRAM] Logical device config failed ({exc}), falling back to memory growth")
                os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass
