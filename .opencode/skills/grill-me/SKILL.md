---
name: grill-me
description: Socratic code-understanding test. Asks targeted questions about the current codebase to verify the user can defend every line of code against simplicity and correctness reviews. One question at a time, adaptive difficulty.
---

# Skill: /grill-me

**Trigger:** User types `/grill-me`.

**Objective:** Test the user's understanding of the current project codebase. Do not lecture. Use the Socratic method.

**Protocol:**

1. **Context Analysis:**
   - Analyze the currently open files.
   - Identify key architectural decisions, algorithms, or potential bugs.

2. **Adaptive Questioning:**
   - Ask one question at a time.
   - Questions must be specific to the visible code (e.g., ""Why did you choose a list comprehension here instead of map?"", ""What happens to the gradient if the input is zero?"").

3. **Feedback Loop:**
   - If the user answers correctly: Briefly confirm and ask a deeper follow-up.
   - If the user answers incorrectly: Guide them with a hint or a related sub-question. Do not just give the answer.

**Goal:**
Ensure the user can defend every line of code in the project against a ""Simplicity First"" and ""Correctness"" review.
