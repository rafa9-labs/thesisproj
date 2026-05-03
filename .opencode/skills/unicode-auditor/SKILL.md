---
name: unicode-auditor
description: Grep all Python source files for non-ASCII characters that could crash Windows cp1252 console encoding. Reports file:line references with hex codes and suggested ASCII replacements. Should run before every commit to prevent production UnicodeEncodeError crashes.
---

# Skill: /unicode-auditor

**Trigger:** User types `/unicode-auditor` or before committing Python changes.

**Objective:** Detect any non-ASCII characters in Python source files that could cause `UnicodeEncodeError` on Windows cp1252 console. This project had a production crash from emoji/Unicode in print/log statements.

**Protocol:**

1. **Scan all Python files** in the project (excluding `.git`, `node_modules`, `__pycache__`, `dist`, `build`, `release`, `.venv`, `csv_data`, `hpo`, `frontend`):

   ```python
   import os
   NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]')
   ```

2. **For each file**, check every line for characters with `ord(ch) > 127`.

3. **Common replacement table** (the project already did a nuclear cleanup):

   | Unicode | ASCII Replacement |
   |---------|-------------------|
   | \u2713 (check) | [OK] |
   | \u2717 (cross) | [ERR] |
   | \u26A0/FE0F (warning) | [WARN] |
   | \u2192 (right arrow) | -> |
   | \u2014 (em dash) | -- |
   | \u2013 (en dash) | - |
   | \u2019 (right quote) | ' |
   | \u201C/\u201D (smart quotes) | " |
   | Emoji (any) | Descriptive [TAG] |
   | Greek letters (\u03B1 etc.) | Name (alpha, sigma, etc.) |
   | Degree sign (\u00B0) | deg |

4. **Output format:**
   ```
   ## Unicode Audit Results

   | File | Line | Char | Hex | Suggestion |
   |------|------|------|-----|-------------|
   | pipeline/backtester/run_mixin.py | 55 | emoji | U+1F680 | [LAUNCH] |
   | pipeline/backtester/strategy_mixin.py | 64 | emoji | U+1F6E1 | [SHIELD] |

   **Total: 2 files, 2 non-ASCII characters**

   **Verdict: FAIL** - 2 non-ASCII characters found.
   ```

5. **Auto-fix option:** If the user asks `/unicode-auditor fix`, replace all non-ASCII characters with their ASCII equivalents from the replacement table. Report what was changed.

6. **Edge cases:**
   - Skip files in `csv_data/` (data files may legitimately contain Unicode).
   - Skip `.pyc`, `.pyo`, binary files.
   - Allow `# -*- coding: utf-8 -*-` declarations (they're ASCII themselves).
   - Allow docstrings with intentionally international content (rare; flag for review).

7. **Pre-commit hook suggestion:**
   Recommend adding a pre-commit hook that runs this check. The project's CLAUDE.md already warns about this:
   > Never use emoji or non-ASCII characters in Python source files. Windows cp1252 encoding will crash.

**Historical context:** This project had a production crash where emoji in `print()` and `logging.StreamHandler` caused `UnicodeEncodeError: 'cp1252' codec can't encode character` on Windows. A nuclear cleanup removed ~850+ non-ASCII characters across 66 files. This skill prevents regression.