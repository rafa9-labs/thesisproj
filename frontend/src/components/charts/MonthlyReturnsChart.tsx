import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { ChartCard } from "./ChartCard";
import type { MonthlyResult } from "@/api/schemas";

interface MonthlyReturnsChartProps {
  monthlyResults: MonthlyResult[] | null;
  height?: number;
}

export function MonthlyReturnsChart({ monthlyResults, height = 280 }: MonthlyReturnsChartProps) {
  const data = useMemo(() => {
    if (!monthlyResults || monthlyResults.length === 0) return [];
    return monthlyResults.map((m) => ({
      month: (m.month ?? "").slice(0, 7),
      returnPct: +((m.return_pct ?? 0) * 100).toFixed(2),
      winRate: m.win_rate != null ? +(m.win_rate * 100).toFixed(1) : null,
      sharpe: m.sharpe != null ? +m.sharpe.toFixed(2) : null,
      trades: m.trades,
      positive: (m.return_pct ?? 0) >= 0,
    }));
  }, [monthlyResults]);

  if (data.length === 0) return null;

  return (
    <ChartCard title="Monthly Returns" subtitle={`${data.length} months`}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
          <XAxis
            dataKey="month"
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
            angle={-45}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
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
              if (value == null) return ["—", name];
              if (name === "returnPct") return [`${value.toFixed(2)}%`, "Return"];
              return [value, name];
            }}
            labelFormatter={(label: string) => `${label}`}
          />
          <Bar dataKey="returnPct" name="returnPct" radius={[3, 3, 0, 0]}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill={entry.positive ? "#089981" : "#F23645"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}