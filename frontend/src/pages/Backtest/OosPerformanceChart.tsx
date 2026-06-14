import { memo, useMemo, useState } from "react";
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

interface EquityPoint {
  time: number;
  model: number;
  bh: number;
}

interface Props {
  model: string;
  equity: EquityPoint[];
  totalPeriods: number;
  currentPeriod: number;
  bestTrial: { trial_number: number; score: number | null } | null;
  periods?: OosPeriodResult[];
}

function OosPerformanceChartInner({
  model,
  equity,
  totalPeriods,
  currentPeriod,
  bestTrial,
  periods,
}: Props) {
  const [yMode, setYMode] = useState<"pct" | "raw">("pct");

  const chartData = useMemo(() => {
    return equity.map((p) => ({
      period: p.time,
      model: yMode === "pct" ? (p.model - 1) * 100 : p.model,
      bh: yMode === "pct" ? (p.bh - 1) * 100 : p.bh,
    }));
  }, [equity, yMode]);

  const yLabel = yMode === "pct" ? "%" : "$";

  const pct =
    totalPeriods > 0 ? Math.min(Math.round((currentPeriod / totalPeriods) * 100), 100) : 0;

  const lastModel = equity.length > 0 ? equity[equity.length - 1].model : null;
  const lastBh = equity.length > 0 ? equity[equity.length - 1].bh : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Context bar */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs tracking-[0.06em] text-(--color-text-muted) uppercase">
          {model}
        </span>
        {bestTrial && (
          <span className="font-mono text-xs text-(--color-text-secondary)">
            Trial #{bestTrial.trial_number}
            {bestTrial.score !== null && ` | Score: ${bestTrial.score.toFixed(4)}`}
          </span>
        )}
        <div className="flex-1" />
        <button
          className="rounded border border-(--color-border) px-2 py-0.5 text-[10px]"
          style={{
            color: yMode === "pct" ? "var(--color-brand)" : "var(--color-text-muted)",
            backgroundColor: yMode === "pct" ? "rgba(59,130,246,0.08)" : "transparent",
          }}
          onClick={() => setYMode(yMode === "pct" ? "raw" : "pct")}
        >
          {yMode === "pct" ? "%" : "$"}
        </button>
      </div>

      {/* Equity chart */}
      <div style={{ width: "100%", height: 260 }}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--color-border)"
                strokeOpacity={0.3}
              />
              <XAxis
                dataKey="period"
                tick={{
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                  fill: "var(--color-text-muted)",
                }}
                tickLine={false}
                axisLine={{ stroke: "var(--color-border)" }}
              />
              <YAxis
                tick={{
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                  fill: "var(--color-text-muted)",
                }}
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
                labelFormatter={(l: number) => `Period ${l}`}
              />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono)" }} />
              <Line
                type="monotone"
                dataKey="bh"
                name="Buy & Hold"
                stroke="var(--color-text-muted)"
                strokeWidth={1.5}
                strokeDasharray="6 3"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="model"
                name={model}
                stroke="var(--color-brand)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-sm border border-(--color-border) text-(--color-text-muted)">
            <span className="text-xs">Waiting for simulation data...</span>
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-(--color-elevated)">
          <div
            className="h-full rounded-full bg-(--color-brand) transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="min-w-[60px] text-right font-mono text-[10px] text-(--color-text-secondary)">
          {pct}% | Period {currentPeriod}/{totalPeriods}
        </span>
      </div>

      {/* Stats row */}
      <div className="flex flex-wrap gap-4">
        {lastModel !== null && (
          <div className="flex flex-col">
            <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
              Model {yMode === "pct" ? "Return" : "Equity"}
            </span>
            <span className="font-mono text-sm font-semibold text-(--color-text-primary)">
              {yMode === "pct"
                ? `${((lastModel! - 1) * 100).toFixed(2)}%`
                : `$${(lastModel! * 10000).toFixed(0)}`}
            </span>
          </div>
        )}
        {lastBh !== null && (
          <div className="flex flex-col">
            <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
              B&H {yMode === "pct" ? "Return" : "Equity"}
            </span>
            <span className="font-mono text-sm font-semibold text-(--color-text-secondary)">
              {yMode === "pct"
                ? `${((lastBh! - 1) * 100).toFixed(2)}%`
                : `$${(lastBh! * 10000).toFixed(0)}`}
            </span>
          </div>
        )}
      </div>

      {/* Latest period stat cards */}
      {periods && periods.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {(() => {
            const lp = periods[periods.length - 1];
            const cards = [
              { label: "Sharpe", value: lp.sharpe?.toFixed(2) ?? "-" },
              {
                label: "Return",
                value: lp.return_pct != null ? `${lp.return_pct.toFixed(2)}%` : "-",
              },
              { label: "Trades", value: lp.trades?.toString() ?? "-" },
              { label: "Drawdown", value: lp.drawdown?.toFixed(2) ?? "-" },
              {
                label: "Win Rate",
                value: lp.win_rate != null ? `${(lp.win_rate * 100).toFixed(1)}%` : "-",
              },
            ];
            return cards.map((c) => (
              <div
                key={c.label}
                className="flex flex-col rounded-md border border-(--color-border) bg-(--color-elevated) px-2.5 py-1.5"
                style={{ minWidth: 72 }}
              >
                <span className="text-[9px] tracking-[0.08em] text-(--color-text-muted) uppercase">
                  {c.label}
                </span>
                <span className="font-mono text-xs font-semibold text-(--color-text-primary)">
                  {c.value}
                </span>
              </div>
            ));
          })()}
        </div>
      )}
    </div>
  );
}

export const OosPerformanceChart = memo(OosPerformanceChartInner);
