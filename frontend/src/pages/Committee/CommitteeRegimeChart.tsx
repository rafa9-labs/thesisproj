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
  LabelList,
} from "recharts";
import type { FullCycleResultsResponse } from "@/api/schemas";

interface Props {
  racecarBacktest: FullCycleResultsResponse["racecar_backtest"];
}

function formatRegime(name: string): string {
  return name.replace(/_/g, " ");
}

export function CommitteeRegimeChart({ racecarBacktest }: Props) {
  const chartData = useMemo(() => {
    const perRegime = racecarBacktest?.per_regime_summary ?? {};
    const entries = Object.entries(perRegime);
    if (entries.length === 0) return [];
    return entries
      .map(([regime, data]) => ({
        regime: formatRegime(regime),
        sharpe: data.sharpe ?? 0,
        trades: data.trades ?? 0,
        folds: data.folds_active ?? 0,
      }))
      .sort((a, b) => b.sharpe - a.sharpe);
  }, [racecarBacktest]);

  if (chartData.length === 0) {
    return (
      <p className="text-[11px] text-(--color-text-dim)">No per-regime performance data available</p>
    );
  }

  return (
    <div style={{ height: 220 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" vertical={false} />
          <XAxis
            dataKey="regime"
            tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            angle={-25}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-elevated)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 6,
              fontSize: 11,
              fontFamily: "var(--font-mono)",
            }}
            formatter={(value: number, name: string) => {
              if (name === "sharpe") return [value.toFixed(3), "Sharpe"];
              return [value, name];
            }}
          />
          <Bar dataKey="sharpe" radius={[3, 3, 0, 0]} maxBarSize={48}>
            {chartData.map((entry) => (
              <Cell
                key={entry.regime}
                fill={
                  entry.sharpe >= 1
                    ? "var(--color-accent-success)"
                    : entry.sharpe >= 0.5
                      ? "var(--color-accent-warning)"
                      : entry.sharpe >= 0
                        ? "#F29136"
                        : "var(--color-accent-danger)"
                }
              />
            ))}
            <LabelList
              dataKey="trades"
              position="top"
              style={{
                fill: "var(--color-text-muted)",
                fontSize: 9,
                fontFamily: "var(--font-mono)",
              }}
              formatter={(v: number) => `${v}t`}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
