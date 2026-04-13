"""I/O helpers: run directories, model output paths, and file management.

Thin re-export shim — canonical source is utilsNoWFO.py.
"""
from __future__ import annotations

from utilsNoWFO import (  # noqa: F401
    _safe_mean,
    _lisbon_now,
    _format_run_stamp,
    make_results_run_dir,
    ensure_model_dirs,
    comparison_dirs,
    month_dir_path,
    _mk_flat_compat,
)