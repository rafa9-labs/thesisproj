import { useMemo } from "react";
import { formatMetric, formatPercent, formatInt } from "@/lib/formatters";
import type { FullCycleResultsResponse } from "@/api/schemas";

interface Props {
  results: FullCycleResultsResponse;
}

interface KpiDef {
  key: string;
  label: string;
  value: string;
  sub: string | null;
  subType: "positive" | "negative" | "neutral" | "muted";
}

const SUB_COLORS: Record<string, string> = {
  positive: "var(--color-accent-success)",
  negative: "var(--color-accent-danger)",
  neutral: "var(--color-accent-warning)",
  muted: "var(--color-text-muted)",
};

function getSubType(val: number | null | undefined, thresholds: [number, number]): "positive" | "negative" | "neutral" {
  if (val == null || !Number.isFinite(val)) return "neutral";
  if (val >= thresholds[0]) return "positive";
  if (val >= thresholds[1]) return "neutral";
  return "negative";
}

function KpiCell({ kpi }: { kpi: KpiDef }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 border-r border-(--color-glass-border) px-4 py-2 last:border-r-0">
      <span
        className="truncate text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase"
        style={{ fontFamily: "Inter, sans-serif" }}
      >
        {kpi.label}
      </span>
      <span className="font-mono text-sm font-bold tabular-nums leading-none text-(--color-text-primary)">
        {kpi.value}
      </span>
      {kpi.sub != null && (
        <span
          className="truncate text-[10px] leading-none"
          style={{ color: SUB_COLORS[kpi.subType], fontFamily: "Inter, sans-serif" }}
        >
          {kpi.sub}
        </span>
      )}
    </div>
  );
}

export function CommitteeMetricsGrid({ results }: Props) {
  const rb = results.racecar_backtest;
  const avgSharpe = rb?.avg_sharpe ?? null;
  const avgReturn = rb?.avg_return ?? null;
  const avgWinRate = rb?.avg_win_rate ?? null;
  const avgActiveRate = rb?.avg_active_rate ?? null;
  const avgTrades = rb?.avg_trades ?? null;
  const avgDrawdown = rb?.avg_drawdown ?? null;
  const totalFolds = rb?.folds ?? 0;

  // Derive Calmar from avg return / max dd
  const calmar = avgReturn != null && avgDrawdown != null && avgDrawdown !== 0
    ? Math.abs(avgReturn / avgDrawdown)
    : null;
  // CAGR approximation: avg monthly return * 12
  const cagr = avgReturn != null ? avgReturn * 12 : null;

  const kpis: KpiDef[] = useMemo(
    () => [
      {
        key: "avg_sharpe",
        label: "Avg Sharpe",
        value: formatMetric(avgSharpe),
        sub: avgSharpe != null
          ? avgSharpe >= 1 ? "Excellent" : avgSharpe >= 0.5 ? "Good" : "Weak"
          : null,
        subType: getSubType(avgSharpe, [1, 0.5]),
      },
      {
        key: "cagr",
        label: "CAGR",
        value: cagr != null ? formatPercent(cagr) : "—",
        sub: null,
        subType: (cagr ?? 0) >= 0 ? "positive" : "negative",
      },
      {
        key: "calmar",
        label: "Calmar",
        value: calmar != null ? formatMetric(calmar) : "—",
        sub: null,
        subType: getSubType(calmar, [3, 1]),
      },
      {
        key: "max_dd",
        label: "Max DD",
        value: avgDrawdown != null ? formatPercent(avgDrawdown) : "—",
        sub: avgDrawdown != null
          ? Math.abs(avgDrawdown) < 0.1
            ? "Low risk"
            : Math.abs(avgDrawdown) < 0.2 ? "Moderate" : "High risk"
          : null,
        subType: avgDrawdown != null
          ? Math.abs(avgDrawdown) < 0.1
            ? "positive" : Math.abs(avgDrawdown) < 0.2 ? "neutral" : "negative"
          : "muted",
      },
      {
        key: "win_rate",
        label: "Win Rate",
        value: avgWinRate != null ? formatPercent(avgWinRate, 1) : "—",
        sub: avgWinRate != null
          ? avgWinRate >= 0.55 ? "Above avg" : avgWinRate >= 0.5 ? "Avg" : "Below avg"
          : null,
        subType: getSubType(avgWinRate, [0.55, 0.5]),
      },
      {
        key: "avg_trades",
        label: "Trades",
        value: formatInt(avgTrades),
        sub: null,
        subType: "muted",
      },
      {
        key: "active_rate",
        label: "Active Rate",
        value: avgActiveRate != null ? formatPercent(avgActiveRate) : "—",
        sub: null,
        subType: "muted",
      },
      {
        key: "total_folds",
        label: "Folds",
        value: formatInt(totalFolds),
        sub: null,
        subType: "muted",
      },
      {
        key: "fold_cv",
        label: "Fold CV",
        value: results.final_fold_consistency_cv
          ? formatMetric(results.final_fold_consistency_cv)
          : results.phase3_fold_consistency_cv
            ? formatMetric(results.phase3_fold_consistency_cv)
            : "—",
        sub: (results.final_fold_consistency_cv || results.phase3_fold_consistency_cv)
          ? (results.final_fold_consistency_cv || results.phase3_fold_consistency_cv) < 0.5
            ? "Consistent"
            : (results.final_fold_consistency_cv || results.phase3_fold_consistency_cv) < 1.0
              ? "Moderate"
              : "Unstable"
          : null,
        subType: getSubType(
          -((results.final_fold_consistency_cv || results.phase3_fold_consistency_cv) ?? 0),
          [-0.5, -1.0],
        ),
      },
      {
        key: "pbo",
        label: "PBO",
        value: formatMetric(results.pbo),
        sub: results.pbo < 0.1 ? "Low risk" : results.pbo < 0.3 ? "Medium" : "High risk",
        subType: results.pbo < 0.1 ? "positive" : results.pbo < 0.3 ? "neutral" : "negative",
      },
      {
        key: "dsr",
        label: "DSR",
        value: formatMetric(results.dsr),
        sub: results.dsr > 0.95 ? "Significant" : results.dsr > 0.9 ? "Likely" : "Uncertain",
        subType: getSubType(results.dsr, [0.95, 0.9]),
      },
    ],
    [avgSharpe, cagr, calmar, avgDrawdown, avgWinRate, avgTrades, avgActiveRate, totalFolds, results],
  );

  return (
    <div className="flex flex-col gap-0 rounded-sm border border-(--color-glass-border) bg-(--color-surface)">
      <div className="flex items-center border-b border-(--color-glass-border) px-4 py-2">
        <span className="font-mono text-sm font-bold text-(--color-text-primary)">
          Committee
        </span>
        {results.trust_score && (
          <span
            className="ml-3 rounded-md px-2 py-0.5 text-[10px] font-semibold tracking-[0.06em] uppercase"
            style={{
              background: results.trust_score.action === "deploy"
                ? "rgba(8,153,129,0.12)"
                : results.trust_score.action === "proceed"
                  ? "rgba(242,180,54,0.12)"
                  : "rgba(242,54,69,0.12)",
              color: results.trust_score.action === "deploy"
                ? "var(--color-accent-success)"
                : results.trust_score.action === "proceed"
                  ? "var(--color-accent-warning)"
                  : "var(--color-accent-danger)",
            }}
          >
            {results.trust_score.action}
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[660px]" style={{ gridTemplateColumns: "repeat(11, minmax(0, 1fr))" }}>
        {kpis.map((kpi) => (
          <KpiCell key={kpi.key} kpi={kpi} />
        ))}
        </div>
      </div>
    </div>
  );
}
