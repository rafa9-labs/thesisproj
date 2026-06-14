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
import { binTradeReturns } from "@/lib/chartUtils";
import type { TradeRecord } from "@/api/schemas";

interface TradeDistributionChartProps {
  trades: TradeRecord[] | null;
}

export function TradeDistributionChart({ trades }: TradeDistributionChartProps) {
  const bins = useMemo(() => binTradeReturns(trades ?? []), [trades]);

  if (bins.length === 0) return null;

  return (
    <ChartCard title="Trade P&L Distribution" subtitle={`${(trades ?? []).length} trades`}>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={bins} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-elevated)" />
          <XAxis
            dataKey="bin"
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
            angle={-45}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            tickFormatter={(v: number) => String(v)}
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
            formatter={(value: number) => [String(value), "Trades"]}
            labelFormatter={(label: string) => `Return: ${label}%`}
          />
          <Bar dataKey="count" name="count" radius={[2, 2, 0, 0]}>
            {bins.map((entry, idx) => (
              <Cell key={idx} fill={entry.positive ? "#089981" : "#F23645"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
