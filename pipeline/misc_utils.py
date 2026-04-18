"""Miscellaneous shared utilities.

Thin re-export shim — canonical source is utilsNoWFO.py.
"""
from __future__ import annotations

from utilsNoWFO import (  # noqa: F401
    SKIP_PLOTS,
    set_global_determinism,
    rolling_slope,
    ensure_list,
    ensure_dict,
    filter_params,
    print_feature_stats,
    print_conf_stats,
    _ensure_dt,
    _max_drawdown_from_equity,
    _mode_safe,
    _coerce_direction_labels,
    _compute_drawdown,
    _features_from_params_names_only,
    _set_even_time_ticks,
    _pretty_bar_label_global,
    estimate_frequency_per_year,
    _auto_nw_lag,
    hac_std,
    estimate_bars_per_day,
)