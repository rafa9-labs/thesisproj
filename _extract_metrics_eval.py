"""Extract compute_full_evaluation_metrics from utilsNoWFO.py into pipeline/metrics_eval.py"""
import os

lines = open("utilsNoWFO.py", "rb").read().decode("utf-8", "ignore").splitlines()
print("utilsNoWFO.py has", len(lines), "lines")

# Function boundaries (1-based line numbers from _find_func_lines.py)
# _macro_prec_f1_from_confusion: L422-431 (0-based: 421-430)
# compute_full_evaluation_metrics: L433-1519 (0-based: 432-1518)

parts = []

# Header
parts.append('"""')
parts.append('Full evaluation metrics - compute_full_evaluation_metrics and helpers.')
parts.append('')
parts.append('Extracted from utilsNoWFO.py (Phase 3, step 3.1).')
parts.append('"""')
parts.append('')
parts.append('from __future__ import annotations')
parts.append('')
parts.append('import numpy as np')
parts.append('import pandas as pd')
parts.append('from scipy.stats import kurtosis')
parts.append('from sklearn.metrics import confusion_matrix')
parts.append('')

# _macro_prec_f1_from_confusion (L422-431, 0-based: 421-430)
for i in range(421, 431):
    parts.append(lines[i])
parts.append('')

# compute_full_evaluation_metrics (L433-1519, 0-based: 432-1518)
for i in range(432, 1519):
    parts.append(lines[i])
parts.append('')

# Write new module
dst = "pipeline/metrics_eval.py"
with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))
print("Wrote", dst, "with", len(parts), "lines")

# Patch utilsNoWFO.py: replace L420-1519 with re-export
# L420 = 0-based 419 = `from sklearn.metrics import confusion_matrix`
# L431 = blank line after _macro_prec_f1
# L432 = blank line
# L433-1519 = compute_full_evaluation_metrics
keep_before = lines[:419]  # lines 1-419 (0-based: 0-418)
keep_after = lines[1519:]  # lines 1520-end (0-based: 1519+)

reexport = [
    "# --- compute_full_evaluation_metrics extracted to pipeline/metrics_eval.py ---",
    "from pipeline.metrics_eval import (  # noqa: F401",
    "    compute_full_evaluation_metrics,",
    "    _macro_prec_f1_from_confusion,",
    ")",
    "",
]

new_lines = keep_before + reexport + keep_after
with open("utilsNoWFO.py", "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))
print("Patched utilsNoWFO.py:", len(new_lines), "lines (was", len(lines), ")")