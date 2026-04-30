import type { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export function ChartCard({ title, subtitle, children }: ChartCardProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          {title}
        </h3>
        {subtitle && (
          <span
            className="text-[10px]"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
          >
            {subtitle}
          </span>
        )}
      </div>
      <div
        className="rounded-lg border p-3"
        style={{
          borderColor: "var(--color-border)",
          backgroundColor: "var(--color-surface)",
        }}
      >
        {children}
      </div>
    </div>
  );
}