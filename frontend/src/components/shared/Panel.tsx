import { cn } from "@/lib/utils";
import type { ReactNode, CSSProperties } from "react";

/** Outer card. Standard radius (lg), padding (24px) and vertical rhythm (24px). */
export function Panel({
  children,
  className,
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-6 rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-6 backdrop-blur-[12px]",
        className,
      )}
      style={style}
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
        <h3 className="text-[15px] font-bold tracking-tight text-(--color-text-primary)">
          {title}
        </h3>
        {subtitle && (
          <p className="text-[12px] leading-relaxed text-(--color-text-muted)">{subtitle}</p>
        )}
      </div>
      {accessory && <div className="shrink-0">{accessory}</div>}
    </div>
  );
}

/**
 * Inner titled group. Accent bar + uppercase label.
 * Borderless — the outer Panel provides the visual container.
 * Sub-labels use structural vertical spacing (space-y-*) instead of nested cards.
 * `accent` overrides the default brand color for the bar/label.
 */
export function Section({
  title,
  description,
  accent = "var(--color-brand)",
  headerExtra,
  children,
  className,
}: {
  title: string;
  description?: string;
  accent?: string;
  headerExtra?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("pt-2", className)}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: accent }} />
          <h4
            className="text-[11px] font-semibold tracking-[0.12em] uppercase"
            style={{ color: accent }}
          >
            {title}
          </h4>
        </div>
        {headerExtra}
      </div>
      {description && (
        <p className="mb-4 max-w-[640px] text-[11px] leading-relaxed font-light text-(--color-text-muted)">
          {description}
        </p>
      )}
      {children}
    </section>
  );
}
