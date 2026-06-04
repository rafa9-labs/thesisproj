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
    const toChartVal = (v: number | null) => v != null ? (yMode === "pct" ? (v - 1) * 100 : v) : null;
    const modelNames = new Set<string>();

    for (const pt of oosEquity) {
      if (!byPeriod.has(pt.period)) {
        byPeriod.set(pt.period, { label: `M${pt.period}`, bh: toChartVal(pt.bh) });
      }
      const row = byPeriod.get(pt.period)!;
      if (row.bh === null && pt.bh !== null) {
        row.bh = toChartVal(pt.bh);
      }
      row[pt.modelName] = toChartVal(pt.equity);
      modelNames.add(pt.modelName);
    }

    if (byPeriod.size > 0) {
      const zeroRow: ChartRow = { label: "0", bh: 0 };
      for (const m of modelNames) {
        zeroRow[m] = 0;
      }
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
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <span
          className="text-[11px] font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Walk-Forward Equity
        </span>
        <div className="flex-1" />
        <button
          className="px-2 py-0.5 text-[10px] rounded border transition-colors"
          style={{
            borderColor: "var(--color-border)",
            color: yMode === "pct" ? "var(--color-brand)" : "var(--color-text-muted)",
            backgroundColor: yMode === "pct" ? "rgba(59,130,246,0.08)" : "transparent",
          }}
          onClick={() => setYMode(yMode === "pct" ? "raw" : "pct")}
        >
          {yMode === "pct" ? "%" : "$"}
        </button>
      </div>

      <div ref={chartWrapperRef} style={{ width: "100%", height: 320, minWidth: 0 }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" strokeOpacity={0.3} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)" }}
                tickLine={false}
                axisLine={{ stroke: "var(--color-border)" }}
              />
              <YAxis
                tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-text-muted)" }}
                tickLine={false}
                axisLine={{ stroke: "var(--color-border)" }}
                tickFormatter={(v: number) => `${v.toFixed(1)}${yLabel}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-surface)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 6,
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                }}
                labelFormatter={(l: string) => `Test ${l}`}
              />
              <Legend
                wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono)" }}
              />
              <Line
                type="monotone"
                dataKey="bh"
                name="Buy & Hold"
                stroke="var(--color-text-muted)"
                strokeWidth={1}
                strokeDasharray="6 3"
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
                  strokeWidth={2}
                  dot={{ r: 3, strokeWidth: 0 }}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div
            className="flex items-center justify-center rounded-sm border"
            style={{ height: 320, borderColor: "var(--color-border)", color: "var(--color-text-muted)" }}
          >
            <span className="text-xs">Waiting for simulation data...</span>
          </div>
        )}
      </div>

      {oosPeriods.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
            Per-Month Summary
          </span>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse rounded-sm overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
              <thead style={{ backgroundColor: "var(--color-elevated)" }}>
                <tr>
                  <th
                    className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Model
                  </th>
                  <th
                    className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Period
                  </th>
                  <th
                    className="px-2 py-1.5 text-right text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Sharpe
                  </th>
                  <th
                    className="px-2 py-1.5 text-right text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Return
                  </th>
                  <th
                    className="px-2 py-1.5 text-right text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Trades
                  </th>
                  <th
                    className="px-2 py-1.5 text-right text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    DD
                  </th>
                  <th
                    className="px-2 py-1.5 text-right text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Win Rate
                  </th>
                  <th
                    className="px-2 py-1.5 text-center text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Gate
                  </th>
                  <th
                    className="px-2 py-1.5 text-center text-[10px] font-medium uppercase tracking-[0.06em]"
                    style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                  >
                    Risk
                  </th>
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const groups = new Map<string, typeof oosPeriods>();
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
                          <td colSpan={9} style={{ padding: 0 }}>
                            <div style={{ height: 6 }} />
                          </td>
                        </tr>
                      );
                    }
                    first = false;
                    for (const p of periods) {
                      rows.push(
                        <tr key={`${modelName}-${p.period}`} style={{ borderBottom: "1px solid var(--color-border-subtle)" }}>
                          <td
                            className="px-2 py-1 text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-brand)" }}
                          >
                            {modelName}
                          </td>
                          <td
                            className="px-2 py-1 text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}
                          >
                            M{p.period}
                            {p.flat ? " (flat)" : ""}
                          </td>
                          <td
                            className="px-2 py-1 text-right text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                            title={`Train: ${p.train_sharpe?.toFixed(2) ?? "?"} | Test: ${p.sharpe?.toFixed(2) ?? "?"} | Gap: ${(p.sharpe_gap_pct ?? 0).toFixed(0)}%`}
                          >
                            {p.sharpe?.toFixed(2) ?? "-"}
                          </td>
                          <td
                            className="px-2 py-1 text-right text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                          >
                            {p.return_pct != null ? `${p.return_pct.toFixed(2)}%` : "-"}
                          </td>
                          <td
                            className="px-2 py-1 text-right text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                          >
                            {p.trades ?? "-"}
                          </td>
                          <td
                            className="px-2 py-1 text-right text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                          >
                            {p.drawdown?.toFixed(2) ?? "-"}
                          </td>
                          <td
                            className="px-2 py-1 text-right text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                          >
                            {p.win_rate != null ? `${(p.win_rate * 100).toFixed(1)}%` : "-"}
                          </td>
                          <td
                            className="px-2 py-1 text-center text-[10px]"
                            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                            title={`Signals: ${p.signals_passed_gate ?? 0} passed / ${p.signals_raw ?? 0} raw`}
                          >
                            {(p.signals_raw ?? 0) > 0
                              ? `${Math.round(((p.signals_passed_gate ?? 0) / (p.signals_raw ?? 1)) * 100)}%`
                              : "-"}
                          </td>
                          <td className="px-2 py-1 text-center text-[10px]">
                            {p.sharpe_gap_pct != null ? (
                              <span
                                className="inline-block w-2 h-2 rounded-full"
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
                              <span style={{ color: "var(--color-text-muted)" }}>-</span>
                            )}
                          </td>
                        </tr>
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
