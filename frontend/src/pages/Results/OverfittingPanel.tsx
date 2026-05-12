import { Shield, AlertTriangle, TrendingUp, TrendingDown, Activity, BarChart3, Zap } from "lucide-react";
import { useMemo } from "react";
import { formatMetric, formatPercent } from "@/lib/formatters";
import type { OverfittingReportData, OverfittingCI } from "@/api/schemas";

interface Props {
  data: OverfittingReportData | null;
  modelName: string;
}

function CIStrip({ ci, label, formatFn }: { ci: OverfittingCI | null; label: string; formatFn: (v: number | null) => string }) {
  if (!ci || ci.mean === null) return null;
  const hasRange = ci.low !== null && ci.high !== null;
  const barPct = hasRange && ci.mean !== 0 && ci.mean !== null
    ? Math.min(((ci.high! - ci.low!) / Math.abs(ci.mean!)) * 100, 100)
    : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-[0.06em] min-w-[56px]" style={{ color: "var(--color-text-muted)" }}>
        {label}
      </span>
      <span className="text-xs font-semibold min-w-[48px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
        {formatFn(ci.mean)}
      </span>
      {hasRange && (
        <div className="flex items-center gap-1">
          <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
            [{formatFn(ci.low)}, {formatFn(ci.high)}]
          </span>
          <div className="h-1 rounded-full flex-1 min-w-[40px] max-w-[80px]" style={{ backgroundColor: "var(--color-elevated)" }}>
            <div
              className="h-full rounded-full"
              style={{
                width: `${barPct}%`,
                backgroundColor:
                  barPct > 60 ? "var(--color-accent-danger)" :
                  barPct > 30 ? "var(--color-accent-warning)" :
                  "var(--color-accent-success)",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export function OverfittingPanel({ data, modelName }: Props) {
  const scoreColor = data?.risk_level === "low" ? "var(--color-accent-success)"
    : data?.risk_level === "medium" ? "var(--color-accent-warning)"
    : "var(--color-accent-danger)";

  const scoreBg = data?.risk_level === "low" ? "rgba(34,197,94,0.08)"
    : data?.risk_level === "medium" ? "rgba(234,179,8,0.08)"
    : "rgba(239,68,68,0.08)";

  const scoreBorder = data?.risk_level === "low" ? "rgba(34,197,94,0.2)"
    : data?.risk_level === "medium" ? "rgba(234,179,8,0.2)"
    : "rgba(239,68,68,0.2)";

  const Icon = data?.risk_level === "low" ? Shield
    : data?.risk_level === "medium" ? AlertTriangle
    : AlertTriangle;

  const gapColor = (data?.train_oos_gap_pct ?? 0) > 40 ? "var(--color-accent-danger)"
    : (data?.train_oos_gap_pct ?? 0) > 20 ? "var(--color-accent-warning)"
    : "var(--color-accent-success)";

  if (!data) {
    return (
      <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
        <div className="flex items-center gap-2 mb-4">
          <Shield size={16} style={{ color: "var(--color-text-muted)" }} />
          <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
            Overfitting Analysis
          </h3>
        </div>
        <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          No overfitting data available for {modelName}. Run a backtest to generate this report.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <Icon size={16} style={{ color: scoreColor }} />
          <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
            Overfitting Analysis
          </h3>
        </div>
        <div
          className="flex items-center gap-1.5 rounded-full border px-3 py-1"
          style={{ borderColor: scoreBorder, backgroundColor: scoreBg }}
        >
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: scoreColor }} />
          <span className="text-[10px] font-semibold uppercase tracking-[0.06em]" style={{ color: scoreColor }}>
            {data.risk_level} risk
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-5">
        <div className="rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
          <span className="text-[10px] uppercase tracking-[0.06em] block mb-1" style={{ color: "var(--color-text-muted)" }}>
            Overfit Score
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold" style={{ fontFamily: "var(--font-mono)", color: scoreColor }}>
              {data.overfit_score.toFixed(0)}
            </span>
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
              /100
            </span>
          </div>
        </div>
        <div className="rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
          <span className="text-[10px] uppercase tracking-[0.06em] block mb-1" style={{ color: "var(--color-text-muted)" }}>
            Train / OOS Sharpe
          </span>
          <div className="flex items-center gap-2">
            <TrendingUp size={12} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
              {data.is_mean_sharpe !== null ? data.is_mean_sharpe.toFixed(2) : "—"}
            </span>
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>/</span>
            <TrendingDown size={12} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
              {data.oos_mean_sharpe !== null ? data.oos_mean_sharpe.toFixed(2) : "—"}
            </span>
            <span className="text-[10px] font-medium rounded px-1.5 py-0.5 ml-auto" style={{ color: gapColor, backgroundColor: `${gapColor}15` }}>
              {data.train_oos_gap_pct > 0 ? "+" : ""}{data.train_oos_gap_pct.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      <div className="mb-5">
        <span className="text-[10px] uppercase tracking-[0.06em] block mb-2" style={{ color: "var(--color-text-muted)" }}>
          Bootstrap Confidence Intervals (90%)
        </span>
        <div className="flex flex-col gap-2 rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
          <CIStrip ci={data.sharpe_ci} label="Sharpe" formatFn={(v) => formatMetric(v)} />
          <CIStrip ci={data.return_ci} label="Return" formatFn={(v) => formatPercent(v)} />
          <CIStrip ci={data.maxdd_ci} label="Max DD" formatFn={(v) => formatPercent(v)} />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
            CV Sharpe
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
              {data.cv_sharpe_mean !== null ? formatMetric(data.cv_sharpe_mean) : "—"}
            </span>
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
              &plusmn; {data.cv_sharpe_std !== null ? data.cv_sharpe_std.toFixed(2) : "—"}
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
            Temporal Decay
          </span>
          <div className="flex items-center gap-1.5">
            <Activity size={12} style={{ color: data.temporal_degradation_pct > 30 ? "var(--color-accent-danger)" : data.temporal_degradation_pct > 10 ? "var(--color-accent-warning)" : "var(--color-accent-success)" }} />
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
              {data.temporal_degradation_pct.toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
            Signal Gap
          </span>
          <div className="flex items-center gap-1.5">
            <Zap size={12} style={{ color: data.signal_gap_pct > 60 ? "var(--color-accent-danger)" : data.signal_gap_pct > 30 ? "var(--color-accent-warning)" : "var(--color-accent-success)" }} />
            <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
              {data.n_signal_periods}/{data.n_periods}
            </span>
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
              ({data.signal_gap_pct.toFixed(0)}%)
            </span>
          </div>
        </div>
      </div>

      <div className="rounded-lg p-2.5 flex items-center gap-2" style={{ backgroundColor: "var(--color-elevated)" }}>
        <BarChart3 size={12} style={{ color: "var(--color-text-muted)" }} />
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
          {data.sufficient_trades
            ? `MinTRL satisfied (>=${data.min_trl_trades} trades)`
            : `MinTRL NOT satisfied (<${data.min_trl_trades} trades)`}
        </span>
      </div>
    </div>
  );
}
