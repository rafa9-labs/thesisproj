---
name: data-leakage-auditor
description: Run the walk-forward integrity test suite and grep for common data leakage patterns in feature engineering. Checks look-ahead bias, label leakage, execution delay, and chronological split integrity. Reports PASS/FAIL per check with file:line references.
---

# Skill: /data-leakage-auditor

**Trigger:** User types `/data-leakage-auditor` or asks to audit for data leakage.

**Objective:** Detect and report any data leakage or look-ahead bias in the pipeline. This is the single most critical quality gate for a backtesting system.

**Protocol:**

1. **Run integrity test suite:**
   `powershell
   python -m pytest tests/test_walk_forward_integrity.py -v 2>&1
   `
   All 16 tests must pass. Record any failures.

2. **Static analysis checks (grep across pipeline/):**

   | Check | Pattern | Fail Condition |
   |-------|---------|---------------|
   | Future data in features | `shift(-` or `.iloc[i+` | Any positive look-ahead in feature computation |
   | Label uses future bars | Labels computed after features for same bar | Label must use only past/current data |
   | Train/test overlap | `train_end > test_start` or shared data | Any temporal overlap |
   | Missing execution delay | `close[i]` without `close[i-1]` in execution | Trade executed on signal bar instead of next bar |
   | Rolling window leakage | `rolling(N).mean()` on unsorted data | Data not sorted by timestamp before rolling |
   | Imputation with future | `fillna(method='bfill')` or `interpolate(method='spline')` | Backward-fill uses future data |
   | Feature cache staleness | Cache key missing mtime or size | Stale features may include future data |

3. **Verify feature cache keys:**
   - Read `pipeline/feature_cache.py`.
   - Confirm SHA256 includes: data file path, file size, file mtime, canonical feature config.
   - If any component is missing: FAIL.

4. **Verify execution delay:**
   - Read `pipeline/backtester/execution_patches.py`.
   - Confirm signal at bar `t` executes at bar `t+1` (1-bar delay).
   - Search for any path that executes at bar `t` (same-bar execution = leak).

5. **Output format:**
`
## Data Leakage Audit

| Check | Status | Details |
|-------|--------|---------|
| Walk-forward tests (16/16) | PASS | All tests green |
| Look-ahead in features | PASS | No shift(-N) found |
| Label leakage | PASS | Labels use only past data |
| Train/test overlap | PASS | Strict chronological split |
| Execution delay | PASS | 1-bar delay enforced |
| Backward-fill imputation | FAIL | features_mixin.py:412 uses bfill |
| Cache key integrity | PASS | SHA256 includes mtime+size |

**Verdict: FAIL** -- 1 check failed

### Fix Required
- `pipeline/backtester/features_mixin.py:412` -- Replace `bfill()` with `ffill()` (forward-fill only uses past data).
`

6. **Critical rule:** If ANY check fails, the pipeline is unreliable for financial decisions. All failures must be fixed before proceeding.