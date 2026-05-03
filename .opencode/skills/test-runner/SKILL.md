---
name: test-runner
description: Run pnpm build, pnpm lint, and pnpm test sequentially (or project-equivalent commands). Summarize pass/fail per step with error count and key messages. Surface actionable fixes. Never mark ROADMAP items complete unless all three pass.
---

# Skill: /test-runner

**Trigger:** User types `/test-runner`.

**Objective:** Run the project's build, lint, and test commands sequentially. Summarize results and surface actionable fixes.

**Protocol:**

1. **Detect project type:**
   - If `package.json` with scripts exists: use `pnpm build`, `pnpm lint`, `pnpm test` (or npm/yarn equivalents).
   - If Python project (`pyproject.toml` or `requirements.txt`): use `python -m pytest tests/`, `ruff check .`, and relevant build command.
   - If both (monorepo): run frontend commands then backend commands.

2. **Run sequentially:**
   Step 1: Build (`pnpm build` or `pip install -e .`)
   Step 2: Lint (`pnpm lint` or `ruff check .`)
   Step 3: Test (`pnpm test` or `python -m pytest tests/`)

   Each step depends on the previous passing. If build fails, skip lint and test.

3. **For this project (Forex ML Pipeline):**
   The primary test runner is Python-based. Execute:

   `powershell
   # Step 1: Check Python environment
   python -c ""import pipeline; print('Pipeline imports OK')""

   # Step 2: Lint
   ruff check . 2>&1

   # Step 3: Tests
   python -m pytest tests/ -v --tb=short 2>&1
   `

   For the frontend (if touched):
   `powershell
   cd frontend ; pnpm build 2>&1
   cd frontend ; pnpm lint 2>&1
   cd frontend ; pnpm test 2>&1
   `

4. **Summarize Results:**
`
## Test Runner Results

| Step | Status | Duration | Errors | Key Messages |
|------|--------|----------|--------|--------------|
| Build | PASS | 12s | 0 | - |
| Lint | FAIL | 3s | 5 | unused imports in 3 files |
| Tests | FAIL | 45s | 2 | test_walk_forward: AssertionError on line 42 |

**Verdict: FAIL** — 2/3 steps passed.

### Actionable Fixes
1. Fix unused imports in `pipeline/backtester/strategy_mixin.py`, `models/xgboost_model.py`
2. Investigate `test_walk_forward_integrity` assertion — likely data leakage in split logic
`

5. **ROADMAP Rule:** Never mark a ROADMAP item as complete unless ALL three steps pass. If any step fails, report which sprint items are blocked.
