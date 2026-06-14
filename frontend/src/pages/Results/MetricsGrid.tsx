import { useMemo, useState } from "react";
import { ZeroLeakageBadge } from "@/components/shared/ZeroLeakageBadge";
import { formatMetric, formatPercent, formatInt } from "@/lib/formatters";
import type { Metrics, MonthlyResult } from "@/api/schemas";

interface MetricsGridProps {
  metrics: Metrics;
  modelName: string;
  warnings?: string[];
  monthlyResults?: MonthlyResult[] | null;
}

interface KpiDef {
  key: string;
  label: string;
  value: string;
  sub: string | null;
  subType: "positive" | "negative" | "neutral" | "muted";
}

function getSubType(
  val: number | null | undefined,
  thresholds: [number, number],
): "positive" | "negative" | "neutral" {
  if (val == null || !Number.isFinite(val)) return "neutral";
  if (val >= thresholds[0]) return "positive";
  if (val >= thresholds[1]) return "neutral";
  return "negative";
}

const SUB_COLORS = {
  positive: "var(--color-accent-success)",
  negative: "var(--color-accent-danger)",
  neutral: "var(--color-accent-warning)",
  muted: "var(--color-text-muted)",
} as const;

function KpiCell({ kpi }: { kpi: KpiDef }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 border-r border-(--color-glass-border) px-4 py-2 last:border-r-0">
      <span
        className="truncate text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase"
        style={{ fontFamily: "Inter, sans-serif" }}
      >
        {kpi.label}
      </span>
      <span className="font-mono text-sm leading-none font-bold text-(--color-text-primary) tabular-nums">
        {kpi.value}
      </span>
      {kpi.sub && (
        <span
          className="truncate text-[10px] leading-none"
          style={{
            color: SUB_COLORS[kpi.subType],
            fontFamily: "Inter, sans-serif",
          }}
        >
          {kpi.sub}
        </span>
      )}
    </div>
  );
}

export function MetricsGrid({
  metrics,
  modelName,
  warnings,
}: MetricsGridProps) {
  const [showVif, setShowVif] = useState(false);

  const kpis: KpiDef[] = useMemo(
    () => [
      {
        key: "sharpe",
        label: "Sharpe",
        value: formatMetric(metrics.sharpe),
        sub:
          metrics.sharpe !== null
            ? metrics.sharpe >= 1
              ? "Excellent"
              : metrics.sharpe >= 0.5
                ? "Good"
                : "Weak"
            : null,
        subType: getSubType(metrics.sharpe, [1, 0.5]),
      },
      {
        key: "total_return",
        label: "Total Return",
        value: formatPercent(metrics.total_return_pct),
        sub: null,
        subType: (metrics.total_return_pct ?? 0) >= 0 ? "positive" : "negative",
      },
      {
        key: "max_dd",
        label: "Max Drawdown",
        value: metrics.max_drawdown !== null ? formatPercent(metrics.max_drawdown) : "—",
        sub:
          metrics.max_drawdown !== null
            ? Math.abs(metrics.max_drawdown) < 0.1
              ? "Low risk"
              : Math.abs(metrics.max_drawdown) < 0.2
                ? "Moderate"
                : "High risk"
            : null,
        subType:
          metrics.max_drawdown !== null
            ? Math.abs(metrics.max_drawdown) < 0.1
              ? "positive"
              : Math.abs(metrics.max_drawdown) < 0.2
                ? "neutral"
                : "negative"
            : "muted",
      },
      {
        key: "cagr",
        label: "CAGR",
        value: formatPercent(metrics.cagr),
        sub: null,
        subType: (metrics.cagr ?? 0) >= 0 ? "positive" : "negative",
      },
      {
        key: "win_rate",
        label: "Win Rate",
        value: formatPercent(metrics.win_rate, 1),
        sub:
          metrics.win_rate !== null
            ? metrics.win_rate >= 0.55
              ? "Above avg"
              : metrics.win_rate >= 0.5
                ? "Avg"
                : "Below avg"
            : null,
        subType: getSubType(metrics.win_rate, [0.55, 0.5]),
      },
      {
        key: "profit_factor",
        label: "Profit Factor",
        value: formatMetric(metrics.profit_factor),
        sub:
          metrics.profit_factor !== null
            ? metrics.profit_factor >= 2
              ? "Strong"
              : metrics.profit_factor >= 1.5
                ? "Good"
                : "Weak"
            : null,
        subType: getSubType(metrics.profit_factor, [2, 1.5]),
      },
      {
        key: "total_trades",
        label: "Trades",
        value: formatInt(metrics.total_trades),
        sub: null,
        subType: "muted",
      },
      {
        key: "sortino",
        label: "Sortino",
        value: formatMetric(metrics.sortino),
        sub: null,
        subType: getSubType(metrics.sortino, [1.5, 0.8]),
      },
      {
        key: "calmar",
        label: "Calmar",
        value: formatMetric(metrics.calmar_ratio),
        sub: null,
        subType: getSubType(metrics.calmar_ratio, [3, 1]),
      },
      {
        key: "active_rate",
        label: "Active Rate",
        value: formatPercent(metrics.active_rate),
        sub: null,
        subType: "muted",
      },
      {
        key: "dir_accuracy",
        label: "Dir. Accuracy",
        value: formatPercent(metrics.directional_accuracy),
        sub:
          metrics.directional_accuracy !== null
            ? metrics.directional_accuracy >= 0.55
              ? "Above avg"
              : "Below avg"
            : null,
        subType: (metrics.directional_accuracy ?? 0) >= 0.55 ? "positive" : "negative",
      },
      {
        key: "avg_trade",
        label: "Avg Trade",
        value: formatPercent(metrics.avg_trade),
        sub: null,
        subType: (metrics.avg_trade ?? 0) >= 0 ? "positive" : "negative",
      },
    ],
    [metrics],
  );

  return (
    <div className="flex flex-col gap-0 rounded-[8px] border border-(--color-glass-border) bg-(--color-surface)">
      {/* Header row */}
      <div className="flex items-center justify-between border-b border-(--color-glass-border) px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm font-bold text-(--color-text-primary)">
            {modelName}
          </span>
          <ZeroLeakageBadge />

          {/* VIF warning badge */}
          {warnings && warnings.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowVif((v) => !v)}
                className="flex items-center gap-1 rounded border border-amber-500/[0.25] bg-amber-500/[0.1] px-2 py-0.5 text-[10px] font-medium tracking-wider text-(--color-accent-warning) uppercase transition-all"
                style={{
                  cursor: "pointer",
                }}
                title="Collinear features detected"
              >
                <span className="text-[11px]">&#9888;</span>
                Collinear Features (VIF &gt; 10)
              </button>
              {showVif && (
                <div className="absolute top-full left-0 z-50 mt-1 min-w-[280px] rounded-sm border border-amber-500/[0.3] bg-(--color-elevated) p-3 font-mono text-[11px] whitespace-pre-wrap text-(--color-accent-warning) shadow-[0_8px_24px_rgba(0,0,0,0.4)]">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-[10px] font-semibold tracking-wide uppercase">
                      VIF Warning — Collinear Features
                    </span>
                    <button
                      onClick={() => setShowVif(false)}
                      className="cursor-pointer border-none bg-transparent text-(--color-text-muted)"
                    >
                      &#10005;
                    </button>
                  </div>
                  <div className="flex flex-col gap-1">
                    {warnings.map((w, i) => (
                      <span key={i} className="text-(--color-text-primary)">
                        {w}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* KPI ticker tape */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(12, minmax(0, 1fr))",
        }}
      >
        {kpis.map((kpi) => (
          <KpiCell key={kpi.key} kpi={kpi} />
        ))}
      </div>
    </div>
  );
}
