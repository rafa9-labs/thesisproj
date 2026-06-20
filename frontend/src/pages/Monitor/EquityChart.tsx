import { useMemo, useRef } from "react";
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
  yMode: YMode;
}

type YMode = "pct" | "raw";

interface ChartRow {
  label: string;
  bh: number | null;
  [modelName: string]: number | string | null;
}

export function EquityChart({ models, oosPeriods, oosEquity, yMode }: Props) {
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

      {oosPeriods.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[10px] font-medium uppercase tracking-[0.06em] text-(--color-text-muted)">
            Per-Month Summary
          </span>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse rounded-sm border border-(--color-border) overflow-hidden">
              <thead className="bg-(--color-elevated)">
                <tr>
                  {["Model", "Period", "Sharpe", "Return", "Trades", "DD", "Win Rate", "Gate", "Risk"].map(
                    (h) => (
                      <th
                        key={h}
                        className={`px-2 py-1.5 text-[10px] font-medium uppercase tracking-[0.06em] text-(--color-text-muted) border-b border-(--color-border) ${h === "Model" || h === "Period" ? "text-left" : h === "Gate" || h === "Risk" ? "text-center" : "text-right"}`}
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const groups = new Map<string, OosPeriodResult[]>();
                  for (const p of oosPeriods) {
                    const m = p.model ?? "";
                    if (!groups.has(m)) groups.set(m, []);
                    groups.get(m)!.push(p);
                  }
                  const rows: React.ReactNode[] = [];
                  let first = true;
                  for (const [modelName, periods] of groups) {
                    if (!first) {
                      rows.push(
                        <tr key={`sep-${modelName}`}>
                          <td colSpan={9} className="p-0">
                            <div className="h-1.5" />
                          </td>
                        </tr>,
                      );
                    }
                    first = false;
                    for (const p of periods) {
                      rows.push(
                        <tr
                          key={`${modelName}-${p.period}`}
                          className="border-b border-(--color-border-subtle)"
                        >
                          <td className="px-2 py-1 font-mono text-[10px] text-(--color-brand)">
                            {modelName}
                          </td>
                          <td className="px-2 py-1 font-mono text-[10px] text-(--color-text-secondary)">
                            M{p.period}
                            {p.flat ? " (flat)" : ""}
                          </td>
                          <td
                            className="px-2 py-1 text-right font-mono text-[10px] text-(--color-text-primary)"
                            title={`Train: ${p.train_sharpe?.toFixed(2) ?? "?"} | Test: ${p.sharpe?.toFixed(2) ?? "?"} | Gap: ${(p.sharpe_gap_pct ?? 0).toFixed(0)}%`}
                          >
                            {p.sharpe?.toFixed(2) ?? "-"}
                          </td>
                          <td className="px-2 py-1 text-right font-mono text-[10px] text-(--color-text-primary)">
                            {p.return_pct != null ? `${p.return_pct.toFixed(2)}%` : "-"}
                          </td>
                          <td className="px-2 py-1 text-right font-mono text-[10px] text-(--color-text-primary)">
                            {p.trades ?? "-"}
                          </td>
                          <td className="px-2 py-1 text-right font-mono text-[10px] text-(--color-text-primary)">
                            {p.drawdown?.toFixed(2) ?? "-"}
                          </td>
                          <td className="px-2 py-1 text-right font-mono text-[10px] text-(--color-text-primary)">
                            {p.win_rate != null ? `${(p.win_rate * 100).toFixed(1)}%` : "-"}
                          </td>
                          <td
                            className="px-2 py-1 text-center font-mono text-[10px] text-(--color-text-primary)"
                            title={`Signals: ${p.signals_passed_gate ?? 0} passed / ${p.signals_raw ?? 0} raw`}
                          >
                            {(p.signals_raw ?? 0) > 0
                              ? `${Math.round(((p.signals_passed_gate ?? 0) / (p.signals_raw ?? 1)) * 100)}%`
                              : "-"}
                          </td>
                          <td className="px-2 py-1 text-center text-[10px]">
                            {p.sharpe_gap_pct != null ? (
                              <span
                                className="inline-block h-2 w-2 rounded-full"
                                style={{
                                  backgroundColor:
                                    p.sharpe_gap_pct > 40
                                      ? "var(--color-accent-danger)"
                                      : p.sharpe_gap_pct > 15
                                        ? "var(--color-accent-warning)"
                                        : "var(--color-accent-success)",
                                }}
                                title={`Train/OOS gap: ${p.sharpe_gap_pct.toFixed(0)}%`}
                              />
                            ) : (
                              <span className="text-(--color-text-muted)">-</span>
                            )}
                          </td>
                        </tr>,
                      );
                    }
                  }
                  return rows;
                })()}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
