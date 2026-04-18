import { type ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string | number | null;
  delta?: string | null;
  deltaType?: "positive" | "negative" | "neutral";
  icon?: ReactNode;
  sparklineData?: number[];
}

export function MetricCard({
  label,
  value,
  delta,
  deltaType = "neutral",
  icon,
}: MetricCardProps) {
  const deltaColor =
    deltaType === "positive"
      ? "var(--color-accent-success)"
      : deltaType === "negative"
        ? "var(--color-accent-danger)"
        : "var(--color-text-muted)";

  return (
    <div
      className="flex flex-col gap-1 rounded-lg border p-4 transition-colors duration-150"
      style={{
        backgroundColor: "var(--color-surface)",
        borderColor: "var(--color-border)",
      }}
    >
      <div className="flex items-center justify-between">
        <span
          className="text-[11px] font-semibold uppercase"
          style={{ color: "var(--color-text-secondary)", letterSpacing: "0.08em" }}
        >
          {label}
        </span>
        {icon && <span style={{ color: "var(--color-text-muted)" }}>{icon}</span>}
      </div>
      <span
        className="text-2xl font-bold"
        style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
      >
        {value ?? "—"}
      </span>
      {delta && (
        <span className="text-xs" style={{ color: deltaColor }}>
          {delta}
        </span>
      )}
    </div>
  );
}
