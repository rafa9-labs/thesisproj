"""Backport Phase 4: Extract functions from utilsNoWFO.py into pipeline modules."""
import os

SRC = "utilsNoWFO.py"

def extract_lines(start, end):
    """Extract lines [start, end] inclusive from utilsNoWFO.py."""
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Convert to 0-based
    return "".join(lines[start-1:end])

# ═══════════════════════════════════════════════════════════════
# Module 1: pipeline/feature_utils.py
# ═══════════════════════════════════════════════════════════════
feature_utils = '''"""Feature engineering utilities backported from utilsNoWFO.py.

Phase 4.2a — feature builders, selectors, and transformers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


'''
# add_cyclic_hour_features (L2325-2330)
feature_utils += extract_lines(2325, 2330) + "\n\n"
# fracdiff (L2495-2507)
feature_utils += extract_lines(2495, 2507) + "\n\n"
# triple_barrier_labels (L2510-2575)
feature_utils += extract_lines(2510, 2575) + "\n\n"
# attach_macro_features (L2578-2696)
feature_utils += extract_lines(2578, 2696) + "\n\n"
# build_features_from_params (L2332-2467)
feature_utils += extract_lines(2332, 2467) + "\n\n"
# select_topk_by_mutual_info (L2787-2798)
feature_utils += extract_lines(2787, 2798) + "\n\n"
# drop_near_constant_features (L3482-3497)
feature_utils += extract_lines(3482, 3497) + "\n\n"
# drop_high_corr_features (L3499-3539)
feature_utils += extract_lines(3499, 3539) + "\n\n"
# prefilter_features_train (L3541-3615)
feature_utils += extract_lines(3541, 3615) + "\n\n"
# realized_vol (L3617-3619)
feature_utils += extract_lines(3617, 3619) + "\n"

with open("pipeline/feature_utils.py", "w", encoding="utf-8") as f:
    f.write(feature_utils)
print(f"✅ pipeline/feature_utils.py ({len(feature_utils)} chars)")


# ═══════════════════════════════════════════════════════════════
# Module 2: pipeline/calibration.py
# ═══════════════════════════════════════════════════════════════
calibration = '''"""Model calibration and uncertainty quantification.

Phase 4.3 — probability calibration, temperature scaling, conformal prediction.
Backported from utilsNoWFO.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


'''
# RollingStandardizer (L2703-2735)
calibration += extract_lines(2703, 2735) + "\n\n"
# calibrate_prefit_and_predict_proba (L2738-2754)
calibration += extract_lines(2738, 2754) + "\n\n"
# ConformalClassifier (L2757-2782)
calibration += extract_lines(2757, 2782) + "\n\n"
# predict_decisions (is a method of ConformalClassifier, already included)
# fit_coverage_threshold_on_calibration (L3096-3119)
calibration += extract_lines(3096, 3119) + "\n\n"
# apply_temperature_to_proba (L3122-3129)
calibration += extract_lines(3122, 3129) + "\n\n"
# fit_temperature_from_proba (L3132-3146)
calibration += extract_lines(3132, 3146) + "\n\n"
# sanitize_proba (L681-711)
calibration += extract_lines(681, 711) + "\n"

with open("pipeline/calibration.py", "w", encoding="utf-8") as f:
    f.write(calibration)
print(f"✅ pipeline/calibration.py ({len(calibration)} chars)")


# ═══════════════════════════════════════════════════════════════
# Module 3: pipeline/metrics_extra.py
# ═══════════════════════════════════════════════════════════════
metrics_extra = '''"""Additional metrics backported from utilsNoWFO.py.

Phase 4.4 — PSR, DSR, Brier/NLL, drawdown curves, rolling stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


'''
# compute_brier_and_nll (L426-471)
metrics_extra += extract_lines(426, 471) + "\n\n"
# _cliffs_delta (L3060-3067)
metrics_extra += extract_lines(3060, 3067) + "\n\n"
# probabilistic_sharpe_ratio (L3153-3170)
metrics_extra += extract_lines(3153, 3170) + "\n\n"
# compute_dsr_scores (L2999-3028)
metrics_extra += extract_lines(2999, 3028) + "\n\n"
# compute_rolling_hit_rate (L3379-3396)
metrics_extra += extract_lines(3379, 3396) + "\n\n"
# compute_rolling_sharpe_series (L3842-3892)
metrics_extra += extract_lines(3842, 3892) + "\n\n"
# compute_drawdown_curve (L3828-3839)
metrics_extra += extract_lines(3828, 3839) + "\n"

with open("pipeline/metrics_extra.py", "w", encoding="utf-8") as f:
    f.write(metrics_extra)
print(f"✅ pipeline/metrics_extra.py ({len(metrics_extra)} chars)")


# ═══════════════════════════════════════════════════════════════
# Module 4: pipeline/execution_utils.py
# ═══════════════════════════════════════════════════════════════
execution_utils = '''"""Walk-forward execution helpers backported from utilsNoWFO.py.

Phase 4.2b — test-bar alignment, warmup, trade logs, day-1 anchor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


'''
# first_tradable_test_bar (L2800-2830)
execution_utils += extract_lines(2800, 2830) + "\n\n"
# compute_required_test_warmup_bars (L2833-2902)
execution_utils += extract_lines(2833, 2902) + "\n\n"
# enforce_day1_eval_anchor (L3349-3362)
execution_utils += extract_lines(3349, 3362) + "\n\n"
# find_hit_rate_switch_idx (L3398-3407)
execution_utils += extract_lines(3398, 3407) + "\n\n"
# build_trade_log_from_df (L1066-1191) — includes nested close_trade()
execution_utils += extract_lines(1066, 1191) + "\n"

with open("pipeline/execution_utils.py", "w", encoding="utf-8") as f:
    f.write(execution_utils)
print(f"✅ pipeline/execution_utils.py ({len(execution_utils)} chars)")


# ═══════════════════════════════════════════════════════════════
# Module 5: rl/wrappers.py
# ═══════════════════════════════════════════════════════════════
rl_wrappers = '''"""RL environment wrappers backported from utilsNoWFO.py.

Phase 4.5 — cost-aware and reward-shaping wrappers for gym environments.
"""
from __future__ import annotations

import numpy as np


'''
# CostAwareWrapper (L3621-3714)
rl_wrappers += extract_lines(3621, 3714) + "\n\n"
# RewardProcessWrapper (L3717-3773)
rl_wrappers += extract_lines(3717, 3773) + "\n"

with open("rl/wrappers.py", "w", encoding="utf-8") as f:
    f.write(rl_wrappers)
print(f"✅ rl/wrappers.py ({len(rl_wrappers)} chars)")

print("\n🎉 All 5 modules extracted. Now run syntax check.")