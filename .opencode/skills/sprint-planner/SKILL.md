---
name: sprint-planner
description: Read ROADMAP.md, identify current phase/sprint, list unchecked items, show progress percentage, suggest next task, and output a branch name per project conventions.
---

# Skill: /sprint-planner

**Trigger:** User types `/sprint-planner`.

**Objective:** Quickly assess the project roadmap status and determine what to work on next.

**Protocol:**

1. **Read ROADMAP.md** from the project root.

2. **Identify Current Phase/Sprint:**
   - Find the first sprint/phases where unchecked items `[ ]` remain.
   - Mark completed items (`[x]`) as done.

3. **Calculate Progress:**
   - For each incomplete sprint, count total items vs completed items.
   - Compute overall progress percentage across the entire roadmap.
   - Show per-sprint breakdown.

4. **List Unchecked Items:**
   - Show every `[ ]` item with its sprint ID and description.
   - Highlight the *next* recommended task (first unchecked item in the earliest incomplete sprint).

5. **Suggest Branch Name:**
   - Follow the project git convention from CLAUDE.md.
   - Current branch pattern: `feature/phase<N>-<descriptive-kebab-case>` or `sprint<N>/<sub-task-id>`.
   - Output the exact `git checkout -b` command.

6. **Output Format:**
`
## Sprint Status

| Sprint | Title | Done | Total | Progress |
|--------|-------|------|-------|----------|
| S1     | ...   | 5    | 5     | 100%     |
| S2     | ...   | 2    | 4     | 50%      |

**Overall Progress:** 72% (47/65)

## Next Task
- **Sprint:** S2
- **Item:** S2.3 — Description
- **Files:** related/files.py
- **Branch:** `git checkout -b feature/phase2-api-bridge`

## All Unchecked Items
- [ ] S2.3 — Description
- [ ] S2.4 — Description
- [ ] S5.1 — Description
...
`

**Edge Cases:**
- If ROADMAP.md does not exist, report: `No ROADMAP.md found. Create one or specify a planning document.`
- If all items are checked, report: `All roadmap items complete!` and suggest reviewing or extending the roadmap.
