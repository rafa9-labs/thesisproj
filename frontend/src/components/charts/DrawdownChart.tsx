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
      drawdown: +(p.value * 100).toFixed(2),
    }));
  }, [drawdownCurve]);

  if (data.length === 0) return null;

  return (
    <ChartCard title="Drawdown" subtitle={`${data.length} points`}>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickFormatter={(v: number) => `${v}%`}
            domain={["dataMin", 0]}
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
            itemStyle={{ color: "#F23645" }}
            formatter={(value: number) => [`${value.toFixed(2)}%`, "Drawdown"]}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#F23645"
            fill="rgba(242, 54, 69, 0.25)"
            strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}