"""Find line ranges of key functions in utilsNoWFO.py."""
import re

lines = open("utilsNoWFO.py", "rb").read().decode("utf-8", "ignore").splitlines()
starts = [i + 1 for i, l in enumerate(lines) if re.match(r"^(def |class |@)", l)]

# Build {func_name: (start, end)}
funcs = {}
prev_name = None
prev_start = None
for s in starts:
    m = re.match(r"(?:def |class )(\w+)", lines[s - 1])
    if m:
        if prev_name:
            funcs[prev_name] = (prev_start, s - 1)
        prev_name = m.group(1)
        prev_start = s

if prev_name:
    funcs[prev_name] = (prev_start, len(lines))

# Print key functions
targets = [
    "compute_full_evaluation_metrics",
    "combine_block_scores",
    "enforce_day1_eval_anchor",
    "first_tradable_test_bar",
    "compute_metrics",
]
for t in targets:
    if t in funcs:
        s, e = funcs[t]
        print(f"{t}: lines {s}-{e} ({e - s + 1} lines)")
    else:
        print(f"{t}: NOT FOUND")