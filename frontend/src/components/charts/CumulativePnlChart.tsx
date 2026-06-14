import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { ChartCard } from "./ChartCard";
import { cumulativePnlFromTrades } from "@/lib/chartUtils";
import type { TradeRecord } from "@/api/schemas";

interface CumulativePnlChartProps {
  trades: TradeRecord[] | null;
}

export function CumulativePnlChart({ trades }: CumulativePnlChartProps) {
  const data = useMemo(() => cumulativePnlFromTrades(trades ?? []), [trades]);

  if (data.length === 0) return null;

  const finalPnl = data[data.length - 1].cumPnl;

  return (
    <ChartCard
      title="Cumulative P&L"
      subtitle={`${data.length} trades · Final: ${finalPnl >= 0 ? "+" : ""}${finalPnl?.toFixed?.(1) ?? "0"}%`}
    >
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-elevated)" />
          <XAxis
            dataKey="tradeNum"
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            label={{
              value: "Trade #",
              position: "insideBottomRight",
              offset: -5,
              style: { fill: "var(--color-text-muted)", fontSize: 10 },
            }}
          />
          <YAxis
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            tickFormatter={(v: number) => (v == null ? "" : `${v.toFixed(0)}%`)}
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
            formatter={(value: number) => [
              value == null ? "—" : `${value.toFixed(2)}%`,
              "Cum. P&L",
            ]}
            labelFormatter={(label: number) => `Trade #${label}`}
          />
          <ReferenceLine y={0} stroke="var(--color-glass-border)" strokeDasharray="3 3" />
          <Line
            type="stepAfter"
            dataKey="cumPnl"
            stroke={finalPnl >= 0 ? "#089981" : "#F23645"}
            strokeWidth={1.5}
            dot={false}
            name="Cum. P&L"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
