---
name: design-audit
description: Run the quality gate from the design system. Grep for outlawed patterns (borderRadius 9999, rounded-lg/xl, hardcoded rgba/hex opacity, DOM mutations in event handlers, backdrop-blur outside allowed components, diamond motifs). Run designmd lint if DESIGN.md exists. Report PASS/FAIL per check with file:line references.
---

# Skill: /design-audit

**Trigger:** User types `/design-audit`.

**Objective:** Enforce the project design system quality gate. Detect outlawed patterns in frontend code and report PASS/FAIL per check.

**Protocol:**

1. **Determine project type:**
   - If this is the Deep Focus project (has `DESIGN.md` in root): run the full Deep Focus quality gate.
   - Otherwise: adapt checks to this project's conventions (see Step 3).

2. **Deep Focus Quality Gate (if DESIGN.md exists):**
   Run each grep check across `frontend/src/` and report findings:

   | Check | Pattern | Fail Condition |
   |-------|---------|---------------|
   | Pill buttons | `borderRadius:\s*9999` or `rounded-full` | Any match outside allowed pill components |
   | Rounded classes | `rounded-lg` or `rounded-xl` | Any match (only 2px or sharp corners allowed) |
   | Hardcoded colors | `rgba\(255` or hex with opacity suffix `COLORS.X + "..."` | Any match (must use `COLORS` tokens) |
   | DOM mutations in events | `onMouseEnter` or `onMouseLeave` that modify DOM directly | Any match (use Framer Motion instead) |
   | Backdrop blur misplacement | `backdrop-blur` outside `CurriculumCard` | Any match outside allowed component |
   | Diamond motif misplacement | Diamond SVG motif outside allowed components | Any match outside allowed components |

   Then run: `npx designmd lint DESIGN.md` (skip if not installed).

3. **Adapted Quality Gate (no DESIGN.md):**
   This project is a Forex ML Pipeline with a React frontend. Apply these checks:

   | Check | Pattern | Fail Condition |
   |-------|---------|---------------|
   | Hardcoded colors | `rgba\(` or inline hex like `#[0-9a-fA-F]{6}` in TSX files | Any match that bypasses `COLORS` from `@/lib/constants` |
   | Hardcoded border values | `borderRadius` with values other than design tokens | Any non-token border radius in TSX |
   | Console logs left in production | `console\.(log|warn|error)` in TSX files outside components with explicit debug flags | More than 3 matches |
   | Any `any` type | `: any` in TypeScript | Any match (use proper types) |
   | Inline styles with magic numbers | `style={{` with numeric literals | Any match (use Tailwind tokens) |

4. **Output Format:**
`
## Design Audit Results

| Check | Status | Matches |
|-------|--------|---------|
| Hardcoded colors | FAIL | colors.ts:42, Dashboard.tsx:108 |
| Rounded classes | PASS | 0 matches |
| ... | ... | ... |

**Verdict: FAIL** (2/5 checks failed)
`

5. **Auto-fix option:** If the user asks `/design-audit fix`, attempt to auto-fix each FAIL by replacing with the correct token/pattern.
