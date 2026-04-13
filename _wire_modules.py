"""
Automated module wiring script.
Phase 1: Extract remaining functions from utilsNoWFO.py into new modules.
Phase 2: Convert utilsNoWFO.py into a re-export shim.
Phase 3: Update pipeline/_imports.py.
"""
import re, os, sys

# ── Read source ──────────────────────────────────────────────────────────
with open("utilsNoWFO.py", "r", encoding="utf-8") as f:
    LINES = f.readlines()

SRC = "".join(LINES)

def extract(start_line, end_line):
    """Extract lines [start_line, end_line) (1-indexed)."""
    return "".join(LINES[start_line - 1 : end_line - 1])

def find_def_end(start_line):
    """Find the end of a function/class definition starting at start_line (1-indexed)."""
    indent = len(LINES[start_line - 1]) - len(LINES[start_line - 1].lstrip())
    i = start_line  # next line
    while i < len(LINES):
        line = LINES[i]
        stripped = line.rstrip()
        if stripped == "":
            i += 1
            continue
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent <= indent and stripped and not stripped.startswith("#"):
            return i  # exclusive end
        i += 1
    return len(LINES)

# ── Module definitions ──────────────────────────────────────────────────
# Each module: (path, header, [(symbol, start_line, end_line)])

# ═════════════════════════════════════════════════════════════
# Module: pipeline/coverage.py  (NEW)
# ═════════════════════════════════════════════════════════════
coverage_defs = []

# is_coverage_intent (L125-150)
coverage_defs.append(("is_coverage_intent", 125, find_def_end(125)))
# freeze_confidence_threshold (L152-210)
coverage_defs.append(("freeze_confidence_threshold", 152, find_def_end(152)))
# target_coverage_policy (L550-565)
coverage_defs.append(("target_coverage_policy", 550, find_def_end(550)))
# enforce_target_coverage_policy (L567-575)
coverage_defs.append(("enforce_target_coverage_policy", 567, find_def_end(567)))

coverage_src = '''"""Coverage policy, confidence threshold, and target-rate helpers.

Extracted from utilsNoWFO.py — canonical home for coverage-related logic.
"""
from __future__ import annotations

import numpy as np

''' + "\n".join(extract(s, e) for _, s, e in coverage_defs)

with open("pipeline/coverage.py", "w", encoding="utf-8") as f:
    f.write(coverage_src)
print(f"✅ pipeline/coverage.py  ({len(coverage_src)} bytes, {len(coverage_defs)} defs)")

# ═════════════════════════════════════════════════════════════
# Module: pipeline/optuna_utils.py  (NEW)
# ═════════════════════════════════════════════════════════════
optuna_defs = []

# TRAIN_TEST_MONTHS (L713-748)
optuna_defs.append(("TRAIN_TEST_MONTHS", 713, 749))
# TRAIN_TEST_MONTHS_DEBUG (L734-749) — included above, extract separately
# _norm_optuna_direction (L4615)
optuna_defs.append(("_norm_optuna_direction", 4615, find_def_end(4615)))
# _bad_objective_for_direction (L4623)
optuna_defs.append(("_bad_objective_for_direction", 4623, find_def_end(4623)))

optuna_src = '''"""Optuna configuration, direction helpers, and train/test month schedules.

Extracted from utilsNoWFO.py.
"""
from __future__ import annotations

''' + "\n".join(extract(s, e) for _, s, e in optuna_defs)

with open("pipeline/optuna_utils.py", "w", encoding="utf-8") as f:
    f.write(optuna_src)
print(f"✅ pipeline/optuna_utils.py  ({len(optuna_src)} bytes, {len(optuna_defs)} defs)")

# ═════════════════════════════════════════════════════════════
# Module: pipeline/model_utils.py  (NEW)
# ═════════════════════════════════════════════════════════════
model_utils_defs = []

# model_category (L532)
model_utils_defs.append(("model_category", 532, find_def_end(532)))
# _infer_family (L577)
model_utils_defs.append(("_infer_family", 577, find_def_end(577)))
# friendly_model_name (L587)
model_utils_defs.append(("friendly_model_name", 587, find_def_end(587)))
# _fmt_table_ascii (L3451)
model_utils_defs.append(("_fmt_table_ascii", 3451, find_def_end(3451)))
# _build_bar_compare_dict (L3796)
model_utils_defs.append(("_build_bar_compare_dict", 3796, find_def_end(3796)))
# build_model_ranking (L855)
model_utils_defs.append(("build_model_ranking", 855, find_def_end(855)))
# save_model_ranking_csv (L982)
model_utils_defs.append(("save_model_ranking_csv", 982, find_def_end(982)))
# build_model_monthly_pivots (L754)
model_utils_defs.append(("build_model_monthly_pivots", 754, find_def_end(754)))

model_utils_src = '''"""Model naming, ranking, and comparison utilities.

Extracted from utilsNoWFO.py.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

''' + "\n".join(extract(s, e) for _, s, e in model_utils_defs)

with open("pipeline/model_utils.py", "w", encoding="utf-8") as f:
    f.write(model_utils_src)
print(f"✅ pipeline/model_utils.py  ({len(model_utils_src)} bytes, {len(model_utils_defs)} defs)")

# ═════════════════════════════════════════════════════════════
# Module: pipeline/io_utils.py  (NEW)
# ═════════════════════════════════════════════════════════════
io_defs = []

# _safe_mean (L481)
io_defs.append(("_safe_mean", 481, find_def_end(481)))
# _lisbon_now (L485)
io_defs.append(("_lisbon_now", 485, find_def_end(485)))
# _format_run_stamp (L499)
io_defs.append(("_format_run_stamp", 499, find_def_end(499)))
# make_results_run_dir (L518)
io_defs.append(("make_results_run_dir", 518, find_def_end(518)))
# ensure_model_dirs (L606)
io_defs.append(("ensure_model_dirs", 606, find_def_end(606)))
# comparison_dirs (L646)
io_defs.append(("comparison_dirs", 646, find_def_end(646)))
# month_dir_path (L3424)
io_defs.append(("month_dir_path", 3424, find_def_end(3424)))
# _mk_flat_compat (L3410)
io_defs.append(("_mk_flat_compat", 3410, find_def_end(3410)))

io_src = '''"""I/O helpers: run directories, model output paths, and file management.

Extracted from utilsNoWFO.py.
"""
from __future__ import annotations

import os
import json
import shutil
import numpy as np
import pandas as pd

''' + "\n".join(extract(s, e) for _, s, e in io_defs)

with open("pipeline/io_utils.py", "w", encoding="utf-8") as f:
    f.write(io_src)
print(f"✅ pipeline/io_utils.py  ({len(io_src)} bytes, {len(io_defs)} defs)")

# ═════════════════════════════════════════════════════════════
# Module: pipeline/plotting.py  (NEW — for large viz functions)
# ═════════════════════════════════════════════════════════════
plotting_defs = []

# save_model_bar_comparison_outputs (L1318-1540)
plotting_defs.append(("save_model_bar_comparison_outputs", 1318, find_def_end(1318)))
# _short_model_label (L1542)
plotting_defs.append(("_short_model_label", 1542, find_def_end(1542)))
# apply_academic_style (L1577)
plotting_defs.append(("apply_academic_style", 1577, find_def_end(1577)))
# set_paper_style (L1798)
plotting_defs.append(("set_paper_style", 1798, find_def_end(1798)))
# PAPER_STYLE_DEFAULTS (L1788-1798)
# save_model_underwater_outputs (L3915)
plotting_defs.append(("save_model_underwater_outputs", 3915, find_def_end(3915)))
# save_model_rolling_performance_outputs (L4136)
plotting_defs.append(("save_model_rolling_performance_outputs", 4136, find_def_end(4136)))
# save_group_equity_curves (L2196)
plotting_defs.append(("save_group_equity_curves", 2196, find_def_end(2196)))
# save_monthly_model_stats (L1194)
plotting_defs.append(("save_monthly_model_stats", 1194, find_def_end(1194)))
# build_model_bar_compare_df (L1267)
plotting_defs.append(("build_model_bar_compare_df", 1267, find_def_end(1267)))
# _coalesce_bh_series (L1245)
plotting_defs.append(("_coalesce_bh_series", 1245, find_def_end(1245)))
# _neutral_fill_before_first_trade (L4431)
plotting_defs.append(("_neutral_fill_before_first_trade", 4431, find_def_end(4431)))
# _extend_index_to_calendar_start (L4437 — actually L4431 area)
# save_month_equity_graph (L3172)
plotting_defs.append(("save_month_equity_graph", 3172, find_def_end(3172)))
# save_feature_heatmap_for_single_month (L3260)
plotting_defs.append(("save_feature_heatmap_for_single_month", 3260, find_def_end(3260)))

plotting_src = '''"""Plotting and visualization outputs for model comparison.

Extracted from utilsNoWFO.py.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

''' + "\n".join(extract(s, e) for _, s, e in plotting_defs)

with open("pipeline/plotting.py", "w", encoding="utf-8") as f:
    f.write(plotting_src)
print(f"✅ pipeline/plotting.py  ({len(plotting_src)} bytes, {len(plotting_defs)} defs)")

# ═════════════════════════════════════════════════════════════
# Module: pipeline/misc_utils.py  (NEW — small helpers)
# ═════════════════════════════════════════════════════════════
misc_defs = []

# set_global_determinism (L309)
misc_defs.append(("set_global_determinism", 309, find_def_end(309)))
# rolling_slope (L323)
misc_defs.append(("rolling_slope", 323, find_def_end(323)))
# ensure_list (L331)
misc_defs.append(("ensure_list", 331, find_def_end(331)))
# ensure_dict (L339)
misc_defs.append(("ensure_dict", 339, find_def_end(339)))
# filter_params (L362)
misc_defs.append(("filter_params", 362, find_def_end(362)))
# print_feature_stats (L348)
misc_defs.append(("print_feature_stats", 348, find_def_end(348)))
# print_conf_stats (L374)
misc_defs.append(("print_conf_stats", 374, find_def_end(374)))
# _ensure_dt (L750)
misc_defs.append(("_ensure_dt", 750, find_def_end(750)))
# _max_drawdown_from_equity (L827)
misc_defs.append(("_max_drawdown_from_equity", 827, find_def_end(827)))
# _mode_safe (L842)
misc_defs.append(("_mode_safe", 842, find_def_end(842)))
# SKIP_PLOTS (L19)
# estimate_frequency_per_year (L2906)
misc_defs.append(("estimate_frequency_per_year", 2906, find_def_end(2906)))
# _auto_nw_lag (L2937)
misc_defs.append(("_auto_nw_lag", 2937, find_def_end(2937)))
# hac_std (L2963)
misc_defs.append(("hac_std", 2963, find_def_end(2963)))
# estimate_bars_per_day (L3366)
misc_defs.append(("estimate_bars_per_day", 3366, find_def_end(3366)))
# _coerce_direction_labels (L32)
misc_defs.append(("_coerce_direction_labels", 32, find_def_end(32)))
# _compute_drawdown (L1052)
misc_defs.append(("_compute_drawdown", 1052, find_def_end(1052)))
# _features_from_params_names_only (L1819)
misc_defs.append(("_features_from_params_names_only", 1819, find_def_end(1819)))
# _set_even_time_ticks (L991)
misc_defs.append(("_set_even_time_ticks", 991, find_def_end(991)))
# _pretty_bar_label_global (L3895)
misc_defs.append(("_pretty_bar_label_global", 3895, find_def_end(3895)))

misc_src = '''"""Miscellaneous shared utilities.

Extracted from utilsNoWFO.py.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SKIP_PLOTS = bool(int(os.environ.get("SKIP_PLOTS", "0")))

''' + "\n".join(extract(s, e) for _, s, e in misc_defs)

with open("pipeline/misc_utils.py", "w", encoding="utf-8") as f:
    f.write(misc_src)
print(f"✅ pipeline/misc_utils.py  ({len(misc_src)} bytes, {len(misc_defs)} defs)")

# ═════════════════════════════════════════════════════════════
# Module: pipeline/hpo_io.py  (NEW — HPO persistence & saving)
# ═════════════════════════════════════════════════════════════
hpo_io_defs = []

# save_optuna_progress_from_study (L1663)
hpo_io_defs.append(("save_optuna_progress_from_study", 1663, find_def_end(1663)))
# save_feature_frequency_from_trials (L1929)
hpo_io_defs.append(("save_feature_frequency_from_trials", 1929, find_def_end(1929)))
# save_optuna_learning_summary (L3031)
hpo_io_defs.append(("save_optuna_learning_summary", 3031, find_def_end(3031)))
# PAPER_STYLE_DEFAULTS (L1788)
# save_hpo_config_to_disk (L4547)
hpo_io_defs.append(("save_hpo_config_to_disk", 4547, find_def_end(4547)))
# load_hpo_config_from_disk (L4584)
hpo_io_defs.append(("load_hpo_config_from_disk", 4584, find_def_end(4584)))
# get_hpo_config_dir (L4500)
hpo_io_defs.append(("get_hpo_config_dir", 4500, find_def_end(4500)))
# _hpo_config_path (L4505)
hpo_io_defs.append(("_hpo_config_path", 4505, find_def_end(4505)))
# _sanitize_for_json (L4510)
hpo_io_defs.append(("_sanitize_for_json", 4510, find_def_end(4510)))
# _REPO_ROOT (L4491), _HPO_BASE_DIR (L4493)
# save_feature_frequency_from_monthly_results (L2040)
hpo_io_defs.append(("save_feature_frequency_from_monthly_results", 2040, find_def_end(2040)))

hpo_io_src = '''"""HPO persistence: saving/loading Optuna studies and feature frequencies.

Extracted from utilsNoWFO.py.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__).replace("\\\\", "/"))
if "pipeline" in _REPO_ROOT:
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)

_HPO_BASE_DIR = os.environ.get(
    "HPO_BASE_DIR",
    os.path.join(_REPO_ROOT, "hpo"),
)

''' + "\n".join(extract(s, e) for _, s, e in hpo_io_defs)

with open("pipeline/hpo_io.py", "w", encoding="utf-8") as f:
    f.write(hpo_io_src)
print(f"✅ pipeline/hpo_io.py  ({len(hpo_io_src)} bytes, {len(hpo_io_defs)} defs)")

print("\n✅ Phase 1 complete: All new modules extracted")