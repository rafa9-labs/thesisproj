import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { formatMetric, formatPercent } from "@/lib/formatters";
import type { WalkForwardPeriod } from "@/api/schemas";

interface Props {
  periods: WalkForwardPeriod[] | null;
}

const REGIME_COLORS = {
  sideways: "var(--color-accent-classical)",
  trend: "#22c55e",
  volatile: "#f97316",
};

type ViewMode = "sharpe" | "return" | "signals";

interface PeriodData {
  idx: number;
  label: string;
  testSharpe: number | null;
  trainSharpe: number | null;
  strategyReturn: number | null;
  bhReturn: number | null;
  trades: number;
  signalsRaw: number;
  signalsGated: number;
  signalRatio: number;
  pctSideways: number | null;
  pctTrend: number | null;
  pctVolatile: number | null;
  periodStart: string;
  periodEnd: string;
}

function KpiCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 rounded-sm border border-(--color-glass-border) px-4 py-2.5">
      <span className="text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
        {label}
      </span>
      <span className="font-mono text-lg font-bold text-(--color-text-primary) tabular-nums">
        {value}
      </span>
      {sub && <span className="text-[10px] text-(--color-text-muted)">{sub}</span>}
    </div>
  );
}

function WalkForwardKpiCards({ data }: { data: PeriodData[] }) {
  const cnt = data.length;
  const avgSharpe =
    data.filter((d) => d.testSharpe != null).length > 0
      ? data.reduce((s, d) => s + (d.testSharpe ?? 0), 0) / cnt
      : 0;
  const totalTrades = data.reduce((s, d) => s + d.trades, 0);
  const avgReturn =
    data.filter((d) => d.strategyReturn != null).length > 0
      ? data.reduce((s, d) => s + (d.strategyReturn ?? 0), 0) / cnt
      : 0;
  const positive = data.filter((d) => (d.testSharpe ?? 0) > 0).length;
  const gateRate = data.reduce((s, d) => s + d.signalRatio, 0) / cnt;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <KpiCard label="Avg Sharpe" value={avgSharpe.toFixed(2)} />
      <KpiCard label="Total Trades" value={String(totalTrades)} />
      <KpiCard label="Avg Return" value={`${avgReturn >= 0 ? "+" : ""}${avgReturn.toFixed(1)}%`} />
      <KpiCard label="Positive" value={`${positive}/${cnt}`} />
      <KpiCard label="Gate Rate" value={formatPercent(gateRate)} />
    </div>
  );
}

const tooltipStyle = {
  backgroundColor: "var(--color-elevated)",
  border: "1px solid var(--color-glass-border)",
  borderRadius: 6,
  fontSize: 11,
  fontFamily: "var(--font-mono)",
} as const;

function WalkForwardCompactChart({ data }: { data: PeriodData[] }) {
  const [viewMode, setViewMode] = useState<ViewMode>("sharpe");

  const barData = useMemo(() => {
    if (viewMode === "sharpe") {
      return data.map((d) => ({ ...d, value: d.testSharpe, value2: d.trainSharpe, positive: (d.testSharpe ?? 0) >= 0 }));
    } else if (viewMode === "return") {
      return data.map((d) => ({ ...d, value: d.strategyReturn, value2: d.bhReturn, positive: (d.strategyReturn ?? 0) >= 0 }));
    } else {
      return data.map((d) => ({ ...d, value: d.signalRatio * 100, value2: null, positive: true }));
    }
  }, [data, viewMode]);

  const viewLabel =
    viewMode === "sharpe" ? "Test Sharpe" : viewMode === "return" ? "Return %" : "Signal Gate %";
  const viewLabel2 =
    viewMode === "sharpe" ? "Train Sharpe" : viewMode === "return" ? "B&H Return %" : null;

  return (
    <div className="flex flex-col gap-2">
      {/* View mode toggle */}
      <div className="flex items-center gap-1">
        {(["sharpe", "return", "signals"] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setViewMode(m)}
            className="cursor-pointer rounded px-2 py-0.5 text-[9px] font-medium tracking-wider uppercase transition-all"
            style={{
              backgroundColor: viewMode === m ? "var(--color-brand-glow)" : "transparent",
              color: viewMode === m ? "var(--color-brand)" : "var(--color-text-muted)",
              border: viewMode === m ? "1px solid var(--color-brand)" : "1px solid transparent",
            }}
          >
            {m === "sharpe" ? "Sharpe" : m === "return" ? "Return" : "Signals"}
          </button>
        ))}
      </div>

      {/* Compact chart */}
      <div style={{ height: 130 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={barData}
            margin={{ top: 2, right: 4, left: -10, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
            <XAxis
              dataKey="label"
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
              angle={-45}
              textAnchor="end"
              height={40}
            />
            <YAxis
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
              tickFormatter={(v) => (viewMode === "signals" ? `${v}%` : `${v}`)}
              width={36}
            />
            <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value, name) => {
                const n = typeof value === "number" ? value : 0;
                if (viewMode === "signals") return [`${n.toFixed(1)}%`, name];
                return [n.toFixed(2), name];
              }}
              labelFormatter={(l) => `Period ${l}`}
            />
            <Bar dataKey="value" name={viewLabel} radius={[3, 3, 0, 0]} maxBarSize={22}>
              {barData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={
                    viewMode === "signals"
                      ? "var(--color-brand)"
                      : entry.positive
                        ? "rgba(34,197,94,0.7)"
                        : "rgba(239,68,68,0.7)"
                  }
                />
              ))}
            </Bar>
            {viewMode !== "signals" && (
              <Bar
                dataKey="value2"
                name={viewLabel2!}
                radius={[3, 3, 0, 0]}
                maxBarSize={12}
                fill="rgba(99,102,241,0.5)"
              />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Regime strip */}
      {data.some((d) => d.pctSideways != null || d.pctTrend != null || d.pctVolatile != null) && (
        <div>
          <div className="flex h-2 gap-0.5">
            {data.map((d, i) => (
              <div
                key={i}
                className="flex min-w-1 flex-1 overflow-hidden rounded-sm"
                title={`${d.label}: sideways=${formatPercent(d.pctSideways)} trend=${formatPercent(d.pctTrend)} volatile=${formatPercent(d.pctVolatile)}`}
              >
                <div style={{ width: `${(d.pctSideways ?? 0) * 100}%`, backgroundColor: REGIME_COLORS.sideways }} />
                <div style={{ width: `${(d.pctTrend ?? 0) * 100}%`, backgroundColor: REGIME_COLORS.trend }} />
                <div style={{ width: `${(d.pctVolatile ?? 0) * 100}%`, backgroundColor: REGIME_COLORS.volatile }} />
              </div>
            ))}
          </div>
          <div className="mt-1 flex items-center gap-3 text-[9px] text-(--color-text-muted)">
            <span className="flex items-center gap-1">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-(--color-accent-classical)" /> Sideways
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#22c55e]" /> Trending
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#f97316]" /> Volatile
            </span>
          </div>
        </div>
      )}

      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        <div className="rounded-sm bg-(--color-elevated) px-2 py-1.5">
          <span className="block text-[9px] tracking-[0.06em] text-(--color-text-muted) uppercase">Avg Sharpe</span>
          <span className="font-mono text-[13px] font-semibold">
            {formatMetric(data.reduce((s, d) => s + (d.testSharpe ?? 0), 0) / data.length)}
          </span>
        </div>
        <div className="rounded-sm bg-(--color-elevated) px-2 py-1.5">
          <span className="block text-[9px] tracking-[0.06em] text-(--color-text-muted) uppercase">Total Trades</span>
          <span className="font-mono text-[13px] font-semibold">{data.reduce((s, d) => s + d.trades, 0)}</span>
        </div>
        <div className="rounded-sm bg-(--color-elevated) px-2 py-1.5">
          <span className="block text-[9px] tracking-[0.06em] text-(--color-text-muted) uppercase">Gate Rate</span>
          <span className="font-mono text-[13px] font-semibold">
            {formatPercent(data.reduce((s, d) => s + d.signalRatio, 0) / data.length)}
          </span>
        </div>
        <div className="rounded-sm bg-(--color-elevated) px-2 py-1.5">
          <span className="block text-[9px] tracking-[0.06em] text-(--color-text-muted) uppercase">Pos Periods</span>
          <span className="font-mono text-[13px] font-semibold">
            {data.filter((d) => (d.testSharpe ?? 0) > 0).length}/{data.length}
          </span>
        </div>
      </div>
    </div>
  );
}

export function WalkForwardPanel({ periods }: Props) {
  const data = useMemo(() => {
    if (!periods || periods.length === 0) return [];
    return periods.map((p, i) => ({
      idx: i,
      label: (p.period_start ?? "").slice(0, 7) || `P${i + 1}`,
      testSharpe: p.test_sharpe,
      trainSharpe: p.train_sharpe,
      strategyReturn: p.strategy_return != null ? +(p.strategy_return * 100).toFixed(2) : null,
      bhReturn: p.bh_return != null ? +(p.bh_return * 100).toFixed(2) : null,
      trades: p.trades,
      signalsRaw: p.signals_raw,
      signalsGated: p.signals_passed_gate,
      signalRatio: p.signals_raw > 0 ? p.signals_passed_gate / p.signals_raw : 0,
      pctSideways: p.pct_sideways,
      pctTrend: p.pct_trend,
      pctVolatile: p.pct_volatile,
      periodStart: p.period_start,
      periodEnd: p.period_end,
    }));
  }, [periods]);

  if (data.length <= 1) {
    return null;
  }

  if (data.length <= 3) {
    return <WalkForwardKpiCards data={data} />;
  }

  return <WalkForwardCompactChart data={data} />;
}
