import { useMemo, useState } from "react";
import { ZeroLeakageBadge } from "@/components/shared/ZeroLeakageBadge";
import { formatMetric, formatPercent, formatInt } from "@/lib/formatters";
import type { Metrics, MonthlyResult } from "@/api/schemas";
import { Info } from "lucide-react";

interface MetricsGridProps {
  metrics: Metrics;
  modelName: string;
  warnings?: string[];
  monthlyResults?: MonthlyResult[] | null;
  onShowSummary?: () => void;
  onShowOverfitting?: () => void;
  overfittingScore?: number | null;
  overfittingColor?: string | null;
}

interface KpiDef {
  key: string;
  label: string;
  value: string;
  sub: string | null;
  subType: "positive" | "negative" | "neutral" | "muted";
}

function getSubType(val: number | null | undefined, thresholds: [number, number]): "positive" | "negative" | "neutral" {
  if (val == null) return "neutral";
  if (val >= thresholds[0]) return "positive";
  if (val >= thresholds[1]) return "neutral";
  return "negative";
}

const SUB_COLORS = {
  positive: "#089981",
  negative: "#F23645",
  neutral: "#F59E0B",
  muted: "#787B86",
} as const;

function KpiCell({ kpi }: { kpi: KpiDef }) {
  return (
    <div
      className="flex flex-col gap-0.5 px-4 py-2 border-r last:border-r-0"
      style={{ borderColor: "#2A2E39", minWidth: 0 }}
    >
      <span
        className="text-[10px] font-medium uppercase tracking-[0.1em] truncate"
        style={{ color: "#787B86", fontFamily: "Inter, sans-serif" }}
      >
        {kpi.label}
      </span>
      <span
        className="text-sm font-bold leading-none tabular-nums"
        style={{ color: "#E8ECF1", fontFamily: "JetBrains Mono, monospace" }}
      >
        {kpi.value}
      </span>
      {kpi.sub && (
        <span
          className="text-[10px] leading-none truncate"
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
  overfittingScore,
  overfittingColor,
  onShowSummary,
  onShowOverfitting,
}: MetricsGridProps) {
  const [showVif, setShowVif] = useState(false);

  const RISK_COLORS: Record<string, string> = {
    green: "#089981",
    yellow: "#F59E0B",
    red: "#F23645",
  };

  const kpis: KpiDef[] = useMemo(() => [
    {
      key: "sharpe",
      label: "Sharpe",
      value: formatMetric(metrics.sharpe),
      sub: metrics.sharpe !== null
        ? metrics.sharpe >= 1 ? "Excellent" : metrics.sharpe >= 0.5 ? "Good" : "Weak"
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
      sub: metrics.max_drawdown !== null
        ? Math.abs(metrics.max_drawdown) < 0.1 ? "Low risk" : Math.abs(metrics.max_drawdown) < 0.2 ? "Moderate" : "High risk"
        : null,
      subType: metrics.max_drawdown !== null
        ? Math.abs(metrics.max_drawdown) < 0.1 ? "positive" : Math.abs(metrics.max_drawdown) < 0.2 ? "neutral" : "negative"
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
      sub: metrics.win_rate !== null
        ? metrics.win_rate >= 0.55 ? "Above avg" : metrics.win_rate >= 0.5 ? "Avg" : "Below avg"
        : null,
      subType: getSubType(metrics.win_rate, [0.55, 0.5]),
    },
    {
      key: "profit_factor",
      label: "Profit Factor",
      value: formatMetric(metrics.profit_factor),
      sub: metrics.profit_factor !== null
        ? metrics.profit_factor >= 2 ? "Strong" : metrics.profit_factor >= 1.5 ? "Good" : "Weak"
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
      sub: metrics.directional_accuracy !== null
        ? metrics.directional_accuracy >= 0.55 ? "Above avg" : "Below avg"
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
  ], [metrics]);

  return (
    <div className="flex flex-col gap-0" style={{ backgroundColor: "#1E222D", border: "1px solid #2A2E39", borderRadius: 8 }}>
      {/* Header row */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b"
        style={{ borderColor: "#2A2E39" }}
      >
        <div className="flex items-center gap-3">
          <span
            className="text-sm font-bold"
            style={{ color: "#E8ECF1", fontFamily: "JetBrains Mono, monospace" }}
          >
            {modelName}
          </span>
          <ZeroLeakageBadge />

          {/* VIF warning badge */}
          {warnings && warnings.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowVif((v) => !v)}
                className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider transition-all"
                style={{
                  backgroundColor: "rgba(245,158,11,0.1)",
                  border: "1px solid rgba(245,158,11,0.25)",
                  color: "#F59E0B",
                  cursor: "pointer",
                }}
                title="Collinear features detected"
              >
                <span style={{ fontSize: 11 }}>&#9888;</span>
                Collinear Features (VIF &gt; 10)
              </button>
              {showVif && (
                <div
                  className="absolute left-0 top-full mt-1 z-50 rounded-lg p-3 text-[11px]"
                  style={{
                    backgroundColor: "#252934",
                    border: "1px solid rgba(245,158,11,0.3)",
                    color: "#F59E0B",
                    fontFamily: "JetBrains Mono, monospace",
                    minWidth: 280,
                    boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold uppercase tracking-wide text-[10px]">VIF Warning — Collinear Features</span>
                    <button onClick={() => setShowVif(false)} style={{ color: "#787B86", cursor: "pointer", background: "none", border: "none" }}>
                      &#10005;
                    </button>
                  </div>
                  <div className="flex flex-col gap-1">
                    {warnings.map((w, i) => (
                      <span key={i} style={{ color: "#E8ECF1" }}>{w}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right side: overfitting + info buttons */}
        <div className="flex items-center gap-3">
          {overfittingScore != null && (
            <button
              onClick={onShowOverfitting}
              className="flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider transition-all hover:opacity-80"
              style={{
                backgroundColor: "rgba(255,255,255,0.04)",
                border: "1px solid #2A2E39",
                color: RISK_COLORS[overfittingColor ?? "yellow"] ?? "#F59E0B",
                cursor: onShowOverfitting ? "pointer" : "default",
                fontFamily: "JetBrains Mono, monospace",
              }}
              title="Overfitting score — click to expand"
            >
              <span
                className="w-2 h-2 rounded-full inline-block"
                style={{ backgroundColor: RISK_COLORS[overfittingColor ?? "yellow"] ?? "#F59E0B" }}
              />
              CV FOLD &sigma; {overfittingScore.toFixed(3)}
              {onShowOverfitting && <Info size={10} />}
            </button>
          )}
          {onShowSummary && (
            <button
              onClick={onShowSummary}
              className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider transition-all hover:opacity-80"
              style={{
                backgroundColor: "rgba(255,255,255,0.04)",
                border: "1px solid #2A2E39",
                color: "#787B86",
                cursor: "pointer",
              }}
            >
              <Info size={10} />
              Summary
            </button>
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
