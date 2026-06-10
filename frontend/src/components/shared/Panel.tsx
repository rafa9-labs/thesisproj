import type { ReactNode, CSSProperties } from "react";

/* ──────────────────────────────────────────────────────────────────────────
 * Unified panel primitives for the Backtest workspace (and beyond).
 *
 * One source of truth for padding, radius, spacing, borders and typography so
 * every surface reads as a single coherent system instead of a stack of
 * differently-styled boxes.
 *
 * Hierarchy:
 *   <Panel>            ── the outer glass card (one per logical grouping)
 *     <PanelHeader>    ── title + subtitle + optional right-aligned accessory
 *     <Section>        ── an inner titled group (accent bar + title + explainer)
 * ────────────────────────────────────────────────────────────────────────── */

const PANEL_STYLE: CSSProperties = {
  backgroundColor: "var(--color-glass)",
  borderColor: "var(--color-glass-border)",
  backdropFilter: "blur(12px)",
};

const SECTION_STYLE: CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};

/** Outer card. Standard radius (lg), padding (24px) and vertical rhythm (24px). */
export function Panel({
  children,
  className = "",
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={`flex flex-col gap-6 rounded-lg border p-6 ${className}`}
      style={{ ...PANEL_STYLE, ...style }}
    >
      {children}
    </div>
  );
}

/** Panel-level heading. Consistent 15px bold title + 12px muted subtitle. */
export function PanelHeader({
  title,
  subtitle,
  accessory,
}: {
  title: string;
  subtitle?: string;
  accessory?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex flex-col gap-1">
        <h3
          className="text-[15px] font-bold tracking-tight"
          style={{ color: "var(--color-text-primary)" }}
        >
          {title}
        </h3>
        {subtitle && (
          <p className="text-[12px] leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
            {subtitle}
          </p>
        )}
      </div>
      {accessory && <div className="shrink-0">{accessory}</div>}
    </div>
  );
}

/**
 * Inner titled group. Accent bar + uppercase label + optional explainer.
 * `accent` overrides the default brand color for the bar/label (used to
 * colour-code execution sub-systems, model categories, etc).
 */
export function Section({
  title,
  description,
  accent = "var(--color-brand)",
  children,
  className = "",
}: {
  title: string;
  description?: string;
  accent?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-md border p-5 ${className}`} style={SECTION_STYLE}>
      <div className="mb-1 flex items-center gap-2">
        <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: accent }} />
        <h4
          className="text-[11px] font-semibold uppercase tracking-[0.12em]"
          style={{ color: accent }}
        >
          {title}
        </h4>
      </div>
      {description && (
        <p
          className="mb-4 max-w-[640px] text-[11px] font-light leading-relaxed"
          style={{ color: "var(--color-text-muted)" }}
        >
          {description}
        </p>
      )}
      {children}
    </section>
  );
}
