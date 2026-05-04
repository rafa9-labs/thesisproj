import type { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export function ChartCard({ title, subtitle, children }: ChartCardProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          {title}
        </h3>
        {subtitle && (
          <span
            className="text-[10px] font-light"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
          >
            {subtitle}
          </span>
        )}
      </div>
      <div
        className="rounded-lg border p-4 transition-all duration-300"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
          backdropFilter: "blur(12px)",
        }}
      >
        {children}
      </div>
    </div>
  );
}
