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
    <ChartCard title="Cumulative P&L" subtitle={`${data.length} trades · Final: ${finalPnl >= 0 ? "+" : ""}${finalPnl.toFixed(1)}%`}>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
          <XAxis
            dataKey="tradeNum"
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
            label={{ value: "Trade #", position: "insideBottomRight", offset: -5, style: { fill: "#787B86", fontSize: 10 } }}
          />
          <YAxis
            tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
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
            formatter={(value: number) => [`${value.toFixed(2)}%`, "Cum. P&L"]}
            labelFormatter={(label: number) => `Trade #${label}`}
          />
          <ReferenceLine y={0} stroke="#363A45" strokeDasharray="3 3" />
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