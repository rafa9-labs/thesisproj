# KodaQuant Design System

Single source of truth for all AI-assisted UI work. Read this before touching any TSX/CSS.

## Colors
- **Use CSS variables only.** All color, border, and background values must reference `var(--color-*)` from `frontend/src/index.css`.
- **Never hardcode** hex (`#FFFFFF`) or rgba values in TSX files. Exceptions: chart data palette arrays and dynamic alpha computed at runtime.
- Brand: `var(--color-brand)` = #00E5FF
- Surfaces: `var(--color-app)` / `var(--color-surface)` / `var(--color-elevated)` / `var(--color-glass)`
- Text: `var(--color-text-primary)` / `var(--color-text-secondary)` / `var(--color-text-muted)`
- Semantics: `var(--color-accent-success)` / `var(--color-accent-danger)` / `var(--color-accent-warning)`

## Spacing
- **Use Tailwind spacing classes** (`p-4`, `px-6`, `py-3`, `gap-5`). Tailwind's 4px-base scale is the canonical spacing system.
- Do not use hardcoded `style={{ padding: "Npx" }}`. Use `className="p-N"` or `className="px-N"`.
- **Page containers**: all pages render inside AppShell which provides `px-6 py-4` (24px horizontal, 16px vertical). Do NOT add redundant outer padding to page root elements.
- **Section gap**: use `gap-5` (20px) as the canonical vertical gap between major page sections.

## Layout
- Tailwind utility classes for flex/grid/spacing (`flex`, `grid`, `gap-*`, `p-*`, `m-*`).
- Inline `style={{}}` ONLY for colors, borders, backgrounds, fonts (via CSS variables).
- Grid: prefer `grid grid-cols-{n} gap-4` for multi-column layouts.

## Typography
- **Inter** for labels, headers, body text (`var(--font-sans)`).
- **JetBrains Mono** for numbers, prices, data values (`var(--font-mono)`).
- Sizes: labels 10-11px uppercase, body 12-13px, metrics 24px.

## Border Radius
- **2px only** (`rounded-sm` or `borderRadius: 2`).
- NO `rounded-lg`, `rounded-xl`, or `rounded-full` unless explicitly for pill badges/buttons.

## Animations
- **Framer Motion** for animated entry/exit transitions (`motion.div`, `AnimatePresence`).
- CSS transitions only for simple hover states.

## Components
- Always check `components/shared/` for existing primitives before writing new code.
- `MetricCard`, `TabBar`, `EmptyState`, `StatusDot`, `ValidationBar`, `ExportBar`, `ParamSelect`, `ParamSlider`, `ParamToggle` are all available.
- Reuse, extend, or compose — never duplicate.

## TypeScript
- Strict types. No `any`. Define proper props interfaces.
- Imports: use `@/` path aliases.
