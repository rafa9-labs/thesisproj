"""Plotting and visualization outputs for model comparison.

Thin re-export shim — canonical source is utilsNoWFO.py.
"""
from __future__ import annotations

from utilsNoWFO import (  # noqa: F401
    save_model_bar_comparison_outputs,
    _short_model_label,
    apply_academic_style,
    set_paper_style,
    PAPER_STYLE_DEFAULTS,
    save_model_underwater_outputs,
    save_model_rolling_performance_outputs,
    save_group_equity_curves,
    save_monthly_model_stats,
    build_model_bar_compare_df,
    _coalesce_bh_series,
    _neutral_fill_before_first_trade,
    _extend_index_to_calendar_start,
    save_month_equity_graph,
    save_feature_heatmap_for_single_month,
)