import { type ReactNode, useMemo } from "react";

interface MetricCardProps {
  label: string;
  value: string | number | null;
  delta?: string | null;
  deltaType?: "positive" | "negative" | "neutral";
  icon?: ReactNode;
  sparklineData?: number[];
}

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  const path = useMemo(() => {
    if (data.length < 2) return "";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const h = 20;
    const w = 64;
    const step = w / (data.length - 1);
    return data
      .map((v, i) => {
        const x = i * step;
        const y = h - ((v - min) / range) * h;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [data]);

  return (
    <svg width={64} height={20} viewBox={`0 0 64 20`} style={{ display: "block" }}>
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={1.2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function MetricCard({
  label,
  value,
  delta,
  deltaType = "neutral",
  icon,
  sparklineData,
}: MetricCardProps) {
  const deltaColor =
    deltaType === "positive"
      ? "var(--color-accent-success)"
      : deltaType === "negative"
        ? "var(--color-accent-danger)"
        : "var(--color-text-muted)";

  const sparkColor =
    deltaType === "positive"
      ? "var(--color-accent-success)"
      : deltaType === "negative"
        ? "var(--color-accent-danger)"
        : "var(--color-text-secondary)";

  return (
    <div className="flex flex-col gap-2 rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-5 backdrop-blur-[12px] transition-all duration-300 hover:border-[var(--color-border-active)]">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
          {label}
        </span>
        {icon && (
          <span className="text-(--color-text-muted)" style={{ opacity: 0.6 }}>
            {icon}
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-2">
        <span
          className="font-mono text-2xl font-semibold text-(--color-text-primary)"
          style={{ letterSpacing: "-0.02em" }}
        >
          {value ?? "—"}
        </span>
        {sparklineData && sparklineData.length >= 2 && (
          <MiniSparkline data={sparklineData} color={sparkColor} />
        )}
      </div>
      {delta && (
        <span className="text-[11px] font-medium" style={{ color: deltaColor }}>
          {delta}
        </span>
      )}
    </div>
  );
}
