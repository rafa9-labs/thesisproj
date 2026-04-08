"""
Runtime performance knobs — thread budgets, GPU init, BLAS caps.

Extracted from MLBacktesterNoWFO.py lines 196-294.
Imports config.py for centralized settings.
"""
import os
import sys
import glob
import multiprocessing

from config import get_settings, apply_global_env

# ── WSL2 CUDA auto-configuration ──
# TF on WSL2 needs LD_LIBRARY_PATH pointing to pip-installed nvidia CUDA libs.
# This block auto-discovers them from the active venv's site-packages.
def _configure_wsl2_cuda():
    """Set LD_LIBRARY_PATH for CUDA libs in WSL2 environments."""
    try:
        is_wsl = "microsoft" in os.uname().release.lower()
    except (AttributeError, OSError):
        return  # Windows native — no uname
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

# ── pandas pref ──
try:
    import pandas as pd
    pd.options.mode.copy_on_write = True
except Exception:
    pass

# ── Apply centralized config ──
_settings = get_settings()
apply_global_env(_settings)

# ── psutil ──
try:
    import psutil
except Exception:
    psutil = None

# ── threadpoolctl ──
try:
    from threadpoolctl import threadpool_limits as _tp_limits
except Exception:
    _tp_limits = None

# ── GPU allow-growth ──
# Lazy: skip TF import entirely when only sklearn models are used.
# Set TF_SKIP_INIT=1 to avoid the heavy TF load for logistic/xgboost runs.
_TF_SKIP = os.environ.get("TF_SKIP_INIT", "0") == "1"

if not _TF_SKIP:
    try:
        import tensorflow as _tf
        for g in _tf.config.list_physical_devices("GPU"):
            _tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    # ── Force-CPU escape ──
    if os.environ.get("TF_FORCE_CPU", "0") == "1":
        try:
            import tensorflow as _tf2
            _tf2.config.set_visible_devices([], "GPU")
        except Exception:
            pass

# ── Compute safe core count ──
SAFE_CORES = _settings.compute.safe_cores
CPU_TOTAL = _settings.compute.cpu_total

# ── Apply BLAS / OpenMP cap ──
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=SAFE_CORES)
except Exception:
    pass

# ── TensorFlow thread tuning ──
if not _TF_SKIP:
    try:
        import tensorflow as _tf3
        _tf3.config.threading.set_intra_op_parallelism_threads(SAFE_CORES)
        _tf3.config.threading.set_inter_op_parallelism_threads(min(4, SAFE_CORES // 2))
    except Exception:
        pass

try:
    print(f"Trial thread budget = {SAFE_CORES} cores active per model fit.")
except UnicodeEncodeError:
    pass
