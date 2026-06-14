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
        <h3 className="text-[11px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
          {title}
        </h3>
        {subtitle && (
          <span className="font-mono text-[10px] font-light text-(--color-text-muted)">
            {subtitle}
          </span>
        )}
      </div>
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-4 backdrop-blur-[12px] transition-all duration-300">
        {children}
      </div>
    </div>
  );
}
