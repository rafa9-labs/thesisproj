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
import { Activity, BarChart3, TrendingUp, TrendingDown, Eye } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { formatMetric, formatPercent } from "@/lib/formatters";
import type { WalkForwardPeriod } from "@/api/schemas";

interface Props {
  periods: WalkForwardPeriod[] | null;
  modelName: string;
}

const REGIME_COLORS = {
  sideways: "#6366f1",
  trend: "#22c55e",
  volatile: "#f97316",
};

type ViewMode = "sharpe" | "return" | "signals";

export function WalkForwardPanel({ periods, modelName }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("sharpe");
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

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
      hasTrain: p.train_start != null && p.train_end != null,
      trainStart: p.train_start,
      trainEnd: p.train_end,
      periodStart: p.period_start,
      periodEnd: p.period_end,
    }));
  }, [periods]);

  if (!data.length) {
    return (
      <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
        <div className="flex items-center gap-2 mb-4">
          <Eye size={16} style={{ color: "var(--color-text-muted)" }} />
          <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
            Walk-Forward Transparency
          </h3>
        </div>
        <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          No walk-forward period data available for {modelName}.
        </p>
      </div>
    );
  }

  const barData = useMemo(() => {
    if (viewMode === "sharpe") {
      return data.map((d) => ({
        ...d,
        value: d.testSharpe,
        value2: d.trainSharpe,
        positive: (d.testSharpe ?? 0) >= 0,
      }));
    } else if (viewMode === "return") {
      return data.map((d) => ({
        ...d,
        value: d.strategyReturn,
        value2: d.bhReturn,
        positive: (d.strategyReturn ?? 0) >= 0,
      }));
    } else {
      return data.map((d) => ({
        ...d,
        value: d.signalRatio * 100,
        value2: null,
        positive: true,
      }));
    }
  }, [data, viewMode]);

  const viewLabel = viewMode === "sharpe" ? "Test Sharpe" : viewMode === "return" ? "Return %" : "Signal Gate %";
  const viewLabel2 = viewMode === "sharpe" ? "Train Sharpe" : viewMode === "return" ? "B&H Return %" : null;

  const hovered = hoveredIdx !== null ? data[hoveredIdx] : null;

  return (
    <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Eye size={16} style={{ color: "var(--color-brand)" }} />
          <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
            Walk-Forward Transparency
          </h3>
          <span className="text-[10px] font-mono" style={{ color: "var(--color-text-muted)" }}>
            {data.length} periods
          </span>
        </div>
        <div className="flex items-center gap-1">
          {(["sharpe", "return", "signals"] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className="rounded px-2 py-1 text-[10px] font-medium uppercase tracking-wider transition-all"
              style={{
                backgroundColor: viewMode === m ? "var(--color-brand-glow)" : "transparent",
                color: viewMode === m ? "var(--color-brand)" : "var(--color-text-muted)",
                border: viewMode === m ? "1px solid var(--color-brand)" : "1px solid transparent",
                cursor: "pointer",
              }}
            >
              {m === "sharpe" ? "Sharpe" : m === "return" ? "Return" : "Signals"}
            </button>
          ))}
        </div>
      </div>

      <ChartCard title="" subtitle="" height={240}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={barData} margin={{ top: 5, right: 10, left: -10, bottom: 30 }} onMouseLeave={() => setHoveredIdx(null)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
            <XAxis
              dataKey="label"
              tick={{ fill: "#787B86", fontSize: 9, fontFamily: "JetBrains Mono" }}
              angle={-45}
              textAnchor="end"
              height={50}
            />
            <YAxis
              tick={{ fill: "#787B86", fontSize: 10, fontFamily: "JetBrains Mono" }}
              tickFormatter={(v: number) => viewMode === "signals" ? `${v}%` : `${v}`}
            />
            <ReferenceLine y={0} stroke="#787B86" strokeDasharray="3 3" />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1E222D",
                border: "1px solid #363A45",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "JetBrains Mono",
              }}
              labelStyle={{ color: "#80899F" }}
              formatter={(value: number | null, name: string) => {
                if (value === null || value === undefined) return ["—", name];
                if (viewMode === "signals") return [`${value.toFixed(1)}%`, name];
                return [value.toFixed(2), name];
              }}
              labelFormatter={(label: string) => `Period ${label}`}
            />
            <Bar
              dataKey="value"
              name={viewLabel}
              radius={[3, 3, 0, 0]}
              maxBarSize={28}
              onMouseEnter={(_, idx) => setHoveredIdx(idx)}
            >
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
                  opacity={hoveredIdx === null || hoveredIdx === i ? 1 : 0.4}
                />
              ))}
            </Bar>
            {viewMode !== "signals" && (
              <Bar
                dataKey="value2"
                name={viewLabel2!}
                radius={[3, 3, 0, 0]}
                maxBarSize={16}
                fill="rgba(99,102,241,0.5)"
                onMouseEnter={(_, idx) => setHoveredIdx(idx)}
              />
            )}
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Regime strip */}
      {data.some((d) => d.pctSideways != null || d.pctTrend != null || d.pctVolatile != null) && (
        <div className="mt-3">
          <span className="text-[10px] uppercase tracking-[0.06em] block mb-1.5" style={{ color: "var(--color-text-muted)" }}>
            Regime Distribution
          </span>
          <div className="flex gap-0.5" style={{ height: 10 }}>
            {data.map((d, i) => (
              <div
                key={i}
                className="flex-1 rounded-sm overflow-hidden flex"
                style={{ minWidth: 4, cursor: "pointer", opacity: hoveredIdx === null || hoveredIdx === i ? 1 : 0.4 }}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                title={`${d.label}: sideways=${formatPercent(d.pctSideways)} trend=${formatPercent(d.pctTrend)} volatile=${formatPercent(d.pctVolatile)}`}
              >
                <div style={{ width: `${(d.pctSideways ?? 0) * 100}%`, backgroundColor: REGIME_COLORS.sideways }} />
                <div style={{ width: `${(d.pctTrend ?? 0) * 100}%`, backgroundColor: REGIME_COLORS.trend }} />
                <div style={{ width: `${(d.pctVolatile ?? 0) * 100}%`, backgroundColor: REGIME_COLORS.volatile }} />
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 mt-1.5">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: REGIME_COLORS.sideways }} />
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Sideways</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: REGIME_COLORS.trend }} />
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Trending</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: REGIME_COLORS.volatile }} />
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Volatile</span>
            </div>
          </div>
        </div>
      )}

      {/* Hovered period detail */}
      {hovered && (
        <div
          className="mt-3 rounded-lg p-3 text-xs"
          style={{ backgroundColor: "var(--color-elevated)", fontFamily: "var(--font-mono)" }}
          onMouseLeave={() => setHoveredIdx(null)}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              {hovered.periodStart?.slice(0, 10)} → {hovered.periodEnd?.slice(0, 10)}
            </span>
            {hovered.hasTrain && (
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                Train: {hovered.trainStart?.slice(0, 10)} → {hovered.trainEnd?.slice(0, 10)}
              </span>
            )}
          </div>
          <div className="grid grid-cols-4 gap-3">
            <div>
              <span className="text-[10px] block" style={{ color: "var(--color-text-muted)" }}>Test Sharpe</span>
              <span className="font-semibold" style={{ color: (hovered.testSharpe ?? 0) >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
                {formatMetric(hovered.testSharpe)}
              </span>
            </div>
            <div>
              <span className="text-[10px] block" style={{ color: "var(--color-text-muted)" }}>Train Sharpe</span>
              <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>
                {formatMetric(hovered.trainSharpe)}
              </span>
            </div>
            <div>
              <span className="text-[10px] block" style={{ color: "var(--color-text-muted)" }}>Trades</span>
              <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>{hovered.trades}</span>
            </div>
            <div>
              <span className="text-[10px] block" style={{ color: "var(--color-text-muted)" }}>Signals</span>
              <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>
                {hovered.signalsGated}/{hovered.signalsRaw}
              </span>
            </div>
          </div>
          {(hovered.pctSideways != null || hovered.pctTrend != null || hovered.pctVolatile != null) && (
            <div className="grid grid-cols-3 gap-3 mt-2 pt-2" style={{ borderTop: "1px solid var(--color-glass-border)" }}>
              <div className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: REGIME_COLORS.sideways }} />
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Sideways {formatPercent(hovered.pctSideways)}</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: REGIME_COLORS.trend }} />
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Trend {formatPercent(hovered.pctTrend)}</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: REGIME_COLORS.volatile }} />
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Volatile {formatPercent(hovered.pctVolatile)}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Summary stats */}
      {!hovered && data.length > 0 && (
        <div className="mt-3 grid grid-cols-4 gap-2">
          <div className="rounded-lg p-2" style={{ backgroundColor: "var(--color-elevated)" }}>
            <span className="text-[10px] uppercase tracking-[0.06em] block" style={{ color: "var(--color-text-muted)" }}>
              Avg Sharpe
            </span>
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)" }}>
              {formatMetric(data.reduce((s, d) => s + (d.testSharpe ?? 0), 0) / data.length)}
            </span>
          </div>
          <div className="rounded-lg p-2" style={{ backgroundColor: "var(--color-elevated)" }}>
            <span className="text-[10px] uppercase tracking-[0.06em] block" style={{ color: "var(--color-text-muted)" }}>
              Total Trades
            </span>
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)" }}>
              {data.reduce((s, d) => s + d.trades, 0)}
            </span>
          </div>
          <div className="rounded-lg p-2" style={{ backgroundColor: "var(--color-elevated)" }}>
            <span className="text-[10px] uppercase tracking-[0.06em] block" style={{ color: "var(--color-text-muted)" }}>
              Gate Rate
            </span>
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)" }}>
              {formatPercent(data.reduce((s, d) => s + d.signalRatio, 0) / data.length)}
            </span>
          </div>
          <div className="rounded-lg p-2" style={{ backgroundColor: "var(--color-elevated)" }}>
            <span className="text-[10px] uppercase tracking-[0.06em] block" style={{ color: "var(--color-text-muted)" }}>
              Pos Periods
            </span>
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)" }}>
              {data.filter((d) => (d.testSharpe ?? 0) > 0).length}/{data.length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}