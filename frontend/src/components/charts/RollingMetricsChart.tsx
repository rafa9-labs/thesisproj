import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ChartCard } from "./ChartCard";
import { computeRollingSharpe, computeRollingReturn } from "@/lib/chartUtils";
import type { EquityPoint } from "@/api/schemas";

interface RollingMetricsChartProps {
  equityCurve: EquityPoint[] | null;
  windowSize?: number;
}

function formatDate(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function RollingMetricsChart({ equityCurve, windowSize = 30 }: RollingMetricsChartProps) {
  const data = useMemo(() => {
    if (!equityCurve || equityCurve.length < windowSize + 1) return [];

    const sharpeData = computeRollingSharpe(equityCurve, windowSize);
    const returnData = computeRollingReturn(equityCurve, windowSize);

    const len = Math.min(sharpeData.length, returnData.length);
    if (len === 0) return [];

    const merged: { date: string; sharpe: number; rollingReturn: number }[] = [];
    for (let i = 0; i < len; i++) {
      merged.push({
        date: formatDate(sharpeData[i].time),
        sharpe: +sharpeData[i].sharpe.toFixed(3),
        rollingReturn: +returnData[i].returnPct.toFixed(2),
      });
    }
    return merged;
  }, [equityCurve, windowSize]);

  if (data.length === 0) return null;

  return (
    <ChartCard
      title={`Rolling Metrics (${windowSize}-bar window)`}
      subtitle={`${data.length} points`}
    >
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-elevated)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            yAxisId="sharpe"
            tick={{
              fill: "var(--color-accent-success)",
              fontSize: 10,
              fontFamily: "var(--font-mono)",
            }}
          />
          <YAxis
            yAxisId="return"
            orientation="right"
            tick={{ fill: "var(--color-accent)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            tickFormatter={(v: number) => `${v}%`}
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
            formatter={(value: number, name: string) => {
              if (value == null) return ["—", name === "sharpe" ? "Sharpe" : "Return"];
              if (name === "sharpe") return [value.toFixed(3), "Sharpe"];
              return [`${value.toFixed(2)}%`, "Return"];
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="sharpe"
            stroke="var(--color-accent-success)"
            strokeWidth={1.5}
            dot={false}
            name="Sharpe"
          />
          <Line
            yAxisId="return"
            type="monotone"
            dataKey="rollingReturn"
            stroke="var(--color-accent)"
            strokeWidth={1.5}
            dot={false}
            name="Return"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
