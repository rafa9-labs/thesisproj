# KodaQuant — Design System & Styling Conventions

## Core Principle

> Every design token lives in `src/index.css` inside `@theme`. If a value comes from a CSS variable defined there, reference it via a **Tailwind utility**, never an inline `style` attribute.

---

## Background

This project uses **Tailwind CSS v4** with a **CSS-first configuration** model. The single source of truth for all visual tokens is the `@theme` block in `src/index.css`. Because Tailwind v4 reads `@theme` tokens at build time, it auto-generates utility classes for every token — meaning there is **zero need for inline `style={{}}`** when referencing these tokens.

---

## DOs

### Reference theme tokens via Tailwind utilities

```tsx
// ✅ GOOD — uses Tailwind utility, purge-safe, responsive/variant support
<div className="bg-(--color-surface) text-(--color-text-primary) border-(--color-border)">
  Institutional content
</div>

// ❌ BAD — inline style bypasses Tailwind, harder to maintain, no variant support
<div style={{
  backgroundColor: "var(--color-surface)",
  color: "var(--color-text-primary)",
  borderColor: "var(--color-border)",
}}>
  Institutional content
</div>
```

### Token → Utility Mapping

| CSS Variable | Tailwind Utility |
|---|---|
| `--color-brand` | `text-(--color-brand)` / `bg-(--color-brand)` / `border-(--color-brand)` |
| `--color-surface` | `bg-(--color-surface)` |
| `--color-border` | `border-(--color-border)` |
| `--color-text-primary` | `text-(--color-text-primary)` |
| `--color-text-muted` | `text-(--color-text-muted)` |
| `--font-sans` | `font-sans` |
| `--font-mono` | `font-mono` |
| `--radius-[value]` (future) | `rounded-[value]` |

### Use `cn()` for conditional classes

```tsx
import { cn } from "@/lib/utils";

<button className={cn(
  "flex items-center gap-2 rounded-md px-4 py-2",
  "transition-colors duration-150",
  isActive
    ? "bg-(--color-brand) text-(--color-text-inverse)"
    : "bg-(--color-surface) text-(--color-text-secondary)",
)}>
  {label}
</button>
```

### Use CVA for variant-heavy components

```tsx
import { cva } from "class-variance-authority";

const badge = cva("inline-flex items-center rounded px-2 py-0.5 text-xs font-medium", {
  variants: {
    intent: {
      success: "bg-(--color-accent-success) text-white",
      danger: "bg-(--color-accent-danger) text-white",
      warning: "bg-(--color-accent-warning) text-black",
      info: "bg-(--color-accent-info) text-black",
    },
  },
});

<span className={badge({ intent: severity })}>{label}</span>
```

---

## DON'Ts

### No inline styles for static design tokens

```tsx
// ❌ NEVER
<span style={{ color: "var(--color-text-muted)" }}>Muted text</span>

// ✅ ALWAYS
<span className="text-(--color-text-muted)">Muted text</span>
```

### No hardcoded hex/rgba values

```tsx
// ❌ NEVER
<div style={{ color: "#00E5FF" }}>Brand text</div>

// ✅ ALWAYS
<div className="text-(--color-brand)">Brand text</div>
```

### No dynamic class string concatenation

```tsx
// ❌ NEVER — Tailwind will purge these classes in production
<div className={`bg-${color}-500`}>Bad</div>

// ✅ ALWAYS — static map keeps classes detectable
const colorMap = {
  success: "bg-(--color-accent-success)",
  danger: "bg-(--color-accent-danger)",
};
<div className={colorMap[variant]}>Good</div>
```

### No `@apply` for everyday components

Extract a React component instead. `@apply` is only acceptable for third-party markup you cannot control.

---

## When Inline Styles ARE Acceptable

| Scenario | Pattern |
|---|---|
| Value from API / database | `style={{ backgroundColor: userBrandColor }}` |
| Complex `calc()` or `grid-*` that is unreadable as a utility | `style={{ gridTemplateColumns: "2fr minmax(0, 1fr)" }}` |
| CSS Variable Bridge (dynamic → static) | `style={{ "--row-count": n }}` + `className="grid-rows-(--row-count)"` |

If you use an inline style, **document why** with a comment.

---

## Tooling

| Tool | Purpose |
|---|---|
| `prettier-plugin-tailwindcss` | Auto-orders Tailwind classes on save (layout → sizing → typography → colors → states) |
| Tailwind CSS IntelliSense (VS Code) | Autocomplete, hover preview, lint warnings for all `@theme` tokens |
| `cn()` from `@/lib/utils` | Merge conditional classes with conflict resolution |
| `class-variance-authority` (CVA) | Typed variant definitions for complex components |

### VS Code Settings

The project includes `.vscode/settings.json` with IntelliSense class regex for `cn()` and `cva()`. Format-on-save is enabled with Prettier as the default formatter for TypeScript, TSX, and CSS files.

---

## Class Ordering Convention

`prettier-plugin-tailwindcss` enforces the **official Tailwind class order** automatically:

1. Layout (`flex`, `grid`, `hidden`, `overflow-*`)
2. Sizing (`w-*`, `h-*`, `max-w-*`)
3. Spacing (`p-*`, `m-*`, `gap-*`)
4. Typography (`text-*`, `font-*`, `leading-*`)
5. Visual (`bg-*`, `text-(--color-*)`, `border-*`, `rounded-*`)
6. Effects (`shadow-*`, `opacity-*`)
7. Transitions (`transition-*`, `duration-*`, `ease-*`)
8. States (`hover:*`, `focus:*`, `active:*`, `group-hover:*`)
9. Responsive (`sm:*`, `md:*`, `lg:*`)

**Do not manually reorder classes.** Prettier handles this.

---

## How to Add a New Design Token

1. Add the variable in `@theme { }` block in `src/index.css`
2. Tailwind v4 auto-generates utility classes — no config restart needed
3. Use it in components: `className="bg-(--color-new-token) text-(--color-new-token)"`

### Replace, Don't Extend

```css
/* ✅ DO — declare your full palette up front */
@theme {
  --color-*: initial;
  --color-brand: #00E5FF;
  --color-surface: #1E222D;
  ...
}
```

---

## Animation Convention

All animations are defined as `@keyframes` in `src/index.css` with corresponding `.animate-*` utility classes. To add a new animation:

1. Define `@keyframes` in `src/index.css`
2. Register it in `@theme`: `--animate-my-name: my-name <duration> <easing>;`
3. Use: `className="animate-my-name"`
