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
    <ChartCard title={`Rolling Metrics (${windowSize}-bar window)`} subtitle={`${data.length} points`}>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            yAxisId="sharpe"
            tick={{ fill: "#089981", fontSize: 10, fontFamily: "JetBrains Mono" }}
          />
          <YAxis
            yAxisId="return"
            orientation="right"
            tick={{ fill: "#2962FF", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#2A2E39",
              border: "1px solid #363A45",
              borderRadius: 6,
              fontSize: 12,
              fontFamily: "JetBrains Mono",
            }}
            labelStyle={{ color: "#80899F" }}
            formatter={(value: number, name: string) => {
              if (name === "sharpe") return [value.toFixed(3), "Sharpe"];
              return [`${value.toFixed(2)}%`, "Return"];
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
          />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="sharpe"
            stroke="#089981"
            strokeWidth={1.5}
            dot={false}
            name="Sharpe"
          />
          <Line
            yAxisId="return"
            type="monotone"
            dataKey="rollingReturn"
            stroke="#2962FF"
            strokeWidth={1.5}
            dot={false}
            name="Return"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}