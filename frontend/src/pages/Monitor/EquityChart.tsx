import { useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import type { OosPeriodResult } from "@/api/schemas";

const MODEL_COLORS = [
  "var(--color-brand)",
  "var(--color-accent-success)",
  "var(--color-accent-warning)",
  "var(--color-accent)",
  "var(--color-accent-danger)",
  "#a855f7",
  "#ec4899",
  "#06b6d4",
];

interface Props {
  models: string[];
  oosPeriods: OosPeriodResult[];
  oosEquity: { period: number; modelName: string; equity: number; bh: number }[];
}

type YMode = "pct" | "raw";

interface ChartRow {
  label: string;
  bh: number | null;
  [modelName: string]: number | string | null;
}

export function EquityChart({ models, oosPeriods, oosEquity }: Props) {
  const [yMode, setYMode] = useState<YMode>("pct");
  const chartWrapperRef = useRef<HTMLDivElement>(null);

  const chartData = useMemo(() => {
    const byPeriod = new Map<number, ChartRow>();
    const toChartVal = (v: number | null) =>
      v != null ? (yMode === "pct" ? (v - 1) * 100 : v) : null;
    const modelNames = new Set<string>();

    for (const pt of oosEquity) {
      if (!byPeriod.has(pt.period)) {
        byPeriod.set(pt.period, { label: `M${pt.period}`, bh: toChartVal(pt.bh) });
      }
      const row = byPeriod.get(pt.period)!;
      if (row.bh === null && pt.bh !== null) row.bh = toChartVal(pt.bh);
      row[pt.modelName] = toChartVal(pt.equity);
      modelNames.add(pt.modelName);
    }

    if (byPeriod.size > 0) {
      const zeroRow: ChartRow = { label: "0", bh: 0 };
      for (const m of modelNames) zeroRow[m] = 0;
      byPeriod.set(0, zeroRow);
    }

    return [...byPeriod.values()].sort((a, b) => {
      const an = parseInt(a.label === "0" ? "0" : a.label.slice(1));
      const bn = parseInt(b.label === "0" ? "0" : b.label.slice(1));
      return an - bn;
    });
  }, [oosEquity, yMode]);

  const yLabel = yMode === "pct" ? "%" : "$";
  const hasData = chartData.length > 0;

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="flex-1" />
        <button
          className="rounded border border-(--color-border) px-2 py-0.5 text-[10px] transition-colors"
          style={{
            color: yMode === "pct" ? "var(--color-brand)" : "var(--color-text-muted)",
            backgroundColor: yMode === "pct" ? "rgba(59,130,246,0.08)" : "transparent",
          }}
          onClick={() => setYMode(yMode === "pct" ? "raw" : "pct")}
        >
          {yMode === "pct" ? "%" : "$"}
        </button>
      </div>

      <div ref={chartWrapperRef} className="min-h-0 w-full min-w-0 flex-1">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace", fill: "#64748b" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace", fill: "#64748b" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v: number) => `${v.toFixed(1)}${yLabel}`}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="rounded-md border border-(--color-glass-border) bg-(--color-surface) px-3 py-2 font-mono text-xs shadow-2xl">
                      <div className="mb-1 text-[10px] text-(--color-text-dim)">Test {label}</div>
                      {payload.map((entry) => (
                        <div key={entry.name} className="flex items-center gap-2">
                          <span
                            className="inline-block h-1.5 w-1.5 rounded-full"
                            style={{ backgroundColor: entry.color }}
                          />
                          <span className="text-(--color-text-secondary)">{entry.name}:</span>
                          <span className="text-(--color-text-primary)">
                            {typeof entry.value === "number"
                              ? entry.value.toFixed(2) + yLabel
                              : entry.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              <Legend
                wrapperStyle={{
                  fontSize: 10,
                  fontFamily: "JetBrains Mono, monospace",
                  color: "#787b86",
                }}
              />
              <Line
                type="monotone"
                dataKey="bh"
                name="Buy & Hold"
                stroke="#475569"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                connectNulls
              />
              {models.map((m, i) => (
                <Line
                  key={m}
                  type="monotone"
                  dataKey={m}
                  name={m}
                  stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                  strokeWidth={3}
                  dot={false}
                  activeDot={{ r: 6, fill: "#06b6d4", strokeWidth: 0 }}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-[200px] items-center justify-center rounded-sm border border-(--color-border) text-(--color-text-muted)">
            <span className="text-xs">Waiting for simulation data...</span>
          </div>
        )}
      </div>
    </div>
  );
}
