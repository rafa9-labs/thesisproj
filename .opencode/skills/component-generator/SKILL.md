---
name: component-generator
description: Generate React components following project conventions. Checks DESIGN.md and lib/constants.ts for design tokens. Reuses existing components/shared primitives. Uses use client directive, Framer Motion for animations, COLORS from @/lib/constants, and JetBrains Mono for numeric values. Matches existing component patterns exactly.
---

# Skill: /component-generator

**Trigger:** User asks to create a new React component (e.g., ""create a new chart component"", ""add a settings panel"").

**Objective:** Generate a React component that exactly matches the project's existing patterns, design tokens, and conventions.

**Protocol:**

1. **Read design context:**
   - If `DESIGN.md` exists in root: read it for design system rules.
   - Read `frontend/src/lib/constants.ts` for `COLORS`, `DEFAULTS`, `RANGES`, and `SELECT_OPTIONS` tokens.
   - Read `frontend/package.json` to understand dependencies (especially: UI library, animation library, chart library).

2. **Check for reusable primitives:**
   - Scan `frontend/src/components/shared/` for existing primitives (MetricCard, ExportBar, ParamSlider, ParamSelect, ParamToggle, StatusDot, etc.)
   - Scan `frontend/src/components/charts/` for chart patterns (ChartCard, CumulativePnlChart, etc.)
   - Scan `frontend/src/components/layout/` for layout components (AppShell, TerminalPanel)
   - Never reimplement what already exists — extend or compose.

3. **Component conventions:**
   - Use `""use client""` directive for interactive components.
   - Use Framer Motion for animations (`motion.div`, `AnimatePresence`) — no CSS transitions for complex animations.
   - Colors: Import from `@/lib/constants` — never hardcode hex values or rgba.
   - Border radius: Use 2px (`rounded-sm` or `borderRadius: 2`) — never `rounded-lg`, `rounded-xl`, or `rounded-full` unless explicitly for pill buttons.
   - Numeric displays: Use `font-mono` (JetBrains Mono) for numbers, data values, prices.
   - TypeScript: Strict types, no `any`. Define proper props interfaces.
   - Imports: Use `@/` path aliases. Match existing import ordering patterns.

4. **Pattern matching:**
   - Read 2-3 existing components that are similar to what's being built.
   - Mirror their structure: hooks placement, error boundaries, loading states, responsive patterns.
   - If chart component: use Recharts (existing dependency) and wrap in `ChartCard`.
   - If form component: use controlled components with `DEFAULTS` values.

5. **Generate the component:**
   - Output the full file content.
   - Include proper TypeScript interfaces.
   - Include JSDoc comments for props.
   - Export as named export and default export (matching project pattern).

6. **Post-generation checklist:**
   - [ ] No hardcoded colors (all from constants)
   - [ ] No `any` types
   - [ ] Uses existing shared components where applicable
   - [ ] Animations use Framer Motion
   - [ ] Numeric values use JetBrains Mono
   - [ ] Border radius is 2px or sharp
   - [ ] Matches file naming convention (PascalCase.tsx)
