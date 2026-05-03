---
name: review
description: Rigorous code review biased toward simplicity and surgical changes. Checks for bloat, unnecessary abstractions, drive-by refactors, missed vectorization, and style mismatches. Outputs CRITICAL / NITPICK / POSITIVE categories.
---

# Skill: /review (Karpathy-style Review)

**Trigger:** User types `/review` or `/karpathy-review`.

**Objective:** Perform a rigorous code review of the currently open files or recent changes. Bias toward ""Simplicity First"" and ""Surgical Changes.""

**Review Checklist:**

1. **The ""Bloat"" Check (Simplicity First):**
   - Are there unnecessary abstractions?
   - Is there code that serves no immediate purpose?
   - *Action:* If 50 lines can do the job of 200, demand a rewrite.

2. **The ""Surgical"" Check:**
   - Did the code touch lines that didn't need changing?
   - Did it refactor unrelated code without permission?
   - *Action:* Flag all ""drive-by"" refactors as errors.

3. **Vectorization & Performance (Karpathy Style):**
   - Are there explicit loops that could be vectorized (NumPy/Torch)?
   - Are there inefficient data structures?

4. **Contextual Correctness:**
   - Does the code match the existing project style?
   - Are imports valid based on the project file structure?

**Output Format:**
- **CRITICAL:** Issues that break logic or violate strict simplicity rules.
- **NITPICK:** Style preferences or minor readability improvements.
- **POSITIVE:** Acknowledge good, clean, ""boring"" code.
