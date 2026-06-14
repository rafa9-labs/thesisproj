import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { ChartCard } from "./ChartCard";
import type { EquityPoint } from "@/api/schemas";

interface DrawdownChartProps {
  drawdownCurve: EquityPoint[] | null;
}

function formatDate(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function DrawdownChart({ drawdownCurve }: DrawdownChartProps) {
  const data = useMemo(() => {
    if (!drawdownCurve || drawdownCurve.length === 0) return [];
    return drawdownCurve.map((p) => ({
      date: formatDate(p.time),
      drawdown: +((p.value ?? 0) * 100).toFixed(2),
    }));
  }, [drawdownCurve]);

  if (data.length === 0) return null;

  return (
    <ChartCard title="Drawdown" subtitle={`${data.length} points`}>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-elevated)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            tickFormatter={(v: number) => `${v}%`}
            domain={["dataMin", 0]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-elevated)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 6,
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
            labelStyle={{ color: "var(--color-text-muted)" }}
            itemStyle={{ color: "var(--color-accent-danger)" }}
            formatter={(value: number) => [
              value == null ? "—" : `${value.toFixed(2)}%`,
              "Drawdown",
            ]}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="var(--color-accent-danger)"
            fill="rgba(242, 54, 69, 0.25)"
            strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
