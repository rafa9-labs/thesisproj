"""
Runtime performance knobs — thread budgets, GPU init, BLAS caps.

Extracted from MLBacktesterNoWFO.py lines 196-294.
Imports config.py for centralized settings.
"""
import os
import multiprocessing

from config import get_settings, apply_global_env

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
try:
    import tensorflow as _tf3
    _tf3.config.threading.set_intra_op_parallelism_threads(SAFE_CORES)
    _tf3.config.threading.set_inter_op_parallelism_threads(min(4, SAFE_CORES // 2))
except Exception:
    pass

print(f"🧩 Trial thread budget = {SAFE_CORES} cores active per model fit.")
