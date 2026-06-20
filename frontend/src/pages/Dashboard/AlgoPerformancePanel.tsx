import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BarChart3, ArrowRight } from "lucide-react";
import { formatPercent, formatMetric, formatRelativeTime } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { EquityThumbnail } from "@/components/charts/EquityThumbnail";
import type { JobSummary, EquityPoint, FullCycleHistoryEntry } from "@/api/schemas";
import type { DashboardKPIValues, CommitteeKPIValues } from "@/hooks/useDashboardWaterfall";

type ActivityTab = "backtests" | "committees";

const STATUS_STYLES: Record<string, { dot: string; bg: string; text: string; label: string }> = {
  completed: {
    dot: "var(--color-accent-success)",
    bg: "rgba(34,197,94,0.10)",
    text: "var(--color-accent-success)",
    label: "Completed",
  },
  running: {
    dot: "var(--color-brand)",
    bg: "var(--color-brand-glow)",
    text: "var(--color-brand)",
    label: "Running",
  },
  pending: {
    dot: "var(--color-accent-warning)",
    bg: "rgba(245,158,11,0.10)",
    text: "var(--color-accent-warning)",
    label: "Pending",
  },
  failed: {
    dot: "var(--color-accent-danger)",
    bg: "rgba(239,68,68,0.10)",
    text: "var(--color-accent-danger)",
    label: "Failed",
  },
};

function KpiCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-slate-400 font-semibold tracking-wider font-sans uppercase">
        {label}
      </span>
      <span className={cn("text-xl font-mono tabular-nums", color)}>{value}</span>
    </div>
  );
}

function KpiSkeleton({ label }: { label: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-slate-500 font-semibold tracking-wider font-sans uppercase">
        {label}
      </span>
      <div className="h-7 w-16 animate-pulse rounded bg-white/5" />
    </div>
  );
}

function TabToggle({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded px-3 py-1 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all",
        active
          ? "bg-cyan-500/15 text-cyan-400"
          : "text-slate-500 hover:text-slate-300",
      )}
    >
      {label}
    </button>
  );
}

interface AlgoPerformancePanelProps {
  kpis: DashboardKPIValues;
  jobs: JobSummary[];
  totalJobCount?: number;
  equityData?: Record<string, EquityPoint[] | null>;
  isLoading: boolean;
  committeeKpis: CommitteeKPIValues;
  committeeJobs: FullCycleHistoryEntry[];
  totalCommitteeCount?: number;
  isCommitteesLoading: boolean;
}

export function AlgoPerformancePanel({
  kpis,
  jobs,
  totalJobCount = 0,
  equityData,
  isLoading,
  committeeKpis,
  committeeJobs,
  totalCommitteeCount = 0,
  isCommitteesLoading,
}: AlgoPerformancePanelProps) {
  const navigate = useNavigate();
  const [tab, setTab] = useState<ActivityTab>("backtests");

  const isBacktestsTab = tab === "backtests";
  const hasBacktests = jobs.length > 0;
  const hasCommittees = committeeJobs.length > 0;
  const showViewAll = isBacktestsTab
    ? totalJobCount > 5
    : totalCommitteeCount > 5;
  const currentIsLoading = isBacktestsTab ? isLoading : isCommitteesLoading;
  const hasData = isBacktestsTab ? hasBacktests : hasCommittees;

  return (
    <div className="rounded-lg border border-(--color-glass-border) bg-(--color-glass) backdrop-blur-[12px]">
      <div className="flex items-center justify-between border-b border-(--color-glass-border) px-5 py-3.5">
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-medium tracking-[0.12em] text-slate-400 uppercase">
            Algo Performance &amp; Activity
          </span>
          <div className="flex items-center gap-1">
            <TabToggle
              label="Backtests"
              active={isBacktestsTab}
              onClick={() => setTab("backtests")}
            />
            <TabToggle
              label="Committees"
              active={!isBacktestsTab}
              onClick={() => setTab("committees")}
            />
          </div>
        </div>
        {showViewAll && (
          <button
            onClick={() =>
              navigate(isBacktestsTab ? "/results" : "/committee")
            }
            className="flex items-center gap-1 text-[10px] font-medium tracking-[0.06em] text-slate-500 uppercase transition-colors hover:text-cyan-400"
          >
            View all
            <ArrowRight size={10} />
          </button>
        )}
      </div>

      {/* KPI cards */}
      <div className="px-5 pt-4 pb-1">
        {currentIsLoading ? (
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4 w-full">
            <KpiSkeleton label="WIN RATE" />
            <KpiSkeleton label="TOTAL PROFIT" />
            <KpiSkeleton label="PROFIT FACTOR" />
            <KpiSkeleton label="MAX DRAWDOWN" />
          </div>
        ) : !hasData ? (
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-4">
              <BarChart3 size={24} className="text-slate-500" />
              <span className="text-[11px] leading-relaxed text-slate-400">
                {isBacktestsTab
                  ? "No backtest data populated for current workspace."
                  : "No committee runs completed yet."}
              </span>
            </div>
            <button
              onClick={() =>
                navigate(isBacktestsTab ? "/backtest" : "/committee")
              }
              className="rounded bg-cyan-400 px-5 py-2 text-[11px] font-bold tracking-[0.06em] text-black uppercase transition hover:brightness-110"
            >
              {isBacktestsTab ? "Run First Backtest" : "Run Full Cycle"}
            </button>
          </div>
        ) : isBacktestsTab ? (
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4 w-full">
            <KpiCard
              label="WIN RATE"
              value={kpis.winRate != null ? formatPercent(kpis.winRate, 1) : "\u2014"}
              color={(kpis.winRate ?? 0) >= 0.5 ? "text-emerald-400" : "text-red-400"}
            />
            <KpiCard
              label="TOTAL PROFIT"
              value={kpis.totalProfit != null ? formatPercent(kpis.totalProfit, 2) : "\u2014"}
              color={(kpis.totalProfit ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}
            />
            <KpiCard
              label="PROFIT FACTOR"
              value={kpis.profitFactor != null ? formatMetric(kpis.profitFactor) : "\u2014"}
              color={
                (kpis.profitFactor ?? 0) >= 1.5
                  ? "text-emerald-400"
                  : (kpis.profitFactor ?? 0) >= 1.0
                    ? "text-amber-400"
                    : "text-red-400"
              }
            />
            <KpiCard
              label="MAX DRAWDOWN"
              value={kpis.maxDrawdown != null ? formatPercent(kpis.maxDrawdown, 2) : "\u2014"}
              color="text-red-400"
            />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4 w-full">
            <KpiCard
              label="AVG SHARPE"
              value={
                committeeKpis.avgSharpe != null
                  ? (committeeKpis.avgSharpe >= 0 ? "+" : "") + committeeKpis.avgSharpe.toFixed(2)
                  : "\u2014"
              }
              color={
                (committeeKpis.avgSharpe ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
              }
            />
            <KpiCard
              label="TRUST SCORE"
              value={
                committeeKpis.trustScore != null
                  ? (committeeKpis.trustScore * 100).toFixed(0) + "%"
                  : "\u2014"
              }
              color={
                (committeeKpis.trustScore ?? 0) >= 0.8
                  ? "text-emerald-400"
                  : (committeeKpis.trustScore ?? 0) >= 0.6
                    ? "text-amber-400"
                    : "text-red-400"
              }
            />
            <KpiCard
              label="SURVIVORS"
              value={
                committeeKpis.survivors != null
                  ? String(committeeKpis.survivors)
                  : "\u2014"
              }
              color="text-slate-200"
            />
            <KpiCard
              label="FACTORY SHARPE"
              value={
                committeeKpis.factorySharpe != null
                  ? (committeeKpis.factorySharpe >= 0 ? "+" : "") +
                    committeeKpis.factorySharpe.toFixed(2)
                  : "\u2014"
              }
              color={
                (committeeKpis.factorySharpe ?? 0) >= 0
                  ? "text-emerald-400"
                  : "text-red-400"
              }
            />
          </div>
        )}
      </div>

      {/* Table */}
      {hasData && !currentIsLoading && (
        <div className="px-3 pb-3 pt-2">
          <div className="overflow-hidden rounded-sm border border-(--color-glass-border) bg-(--color-elevated)">
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-(--color-glass-border) bg-(--color-surface) text-(--color-text-muted)">
                  <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                    Job
                  </th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                    {isBacktestsTab ? "Equity" : "Status"}
                  </th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                    {isBacktestsTab ? "Pair" : "Survivors"}
                  </th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                    {isBacktestsTab ? "Models" : "Avg Sharpe"}
                  </th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                    {isBacktestsTab ? "Status" : "Trust"}
                  </th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody>
                {isBacktestsTab
                  ? jobs.map((job) => {
                      const st = STATUS_STYLES[job.status] ?? {
                        dot: "var(--color-text-muted)",
                        bg: "var(--color-glass-hover)",
                        text: "var(--color-text-muted)",
                        label: job.status,
                      };
                      return (
                        <tr
                          key={job.job_id}
                          className="group cursor-pointer border-b border-(--color-glass-border) transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
                          onClick={() => {
                            if (job.status === "completed") {
                              navigate(`/results/${job.job_id}`);
                            }
                          }}
                        >
                          <td className="px-3 py-2.5 font-mono text-(--color-text-primary)">
                            {job.job_id.slice(0, 8)}&hellip;
                          </td>
                          <td className="px-3 py-2.5">
                            {equityData?.[job.job_id] ? (
                              <EquityThumbnail data={equityData[job.job_id]!} />
                            ) : (
                              <div className="h-[36px] w-[120px] rounded-[4px] bg-(--color-glass-hover)" />
                            )}
                          </td>
                          <td className="px-3 py-2.5 font-mono text-(--color-text-primary)">
                            {job.pair ?? "\u2014"}
                          </td>
                          <td className="max-w-[200px] truncate px-3 py-2.5 text-(--color-text-secondary)">
                            {job.models?.join(", ") ?? "\u2014"}
                          </td>
                          <td className="px-3 py-2.5">
                            <div
                              className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-[0.08em] uppercase"
                              style={{ backgroundColor: st.bg, color: st.text }}
                            >
                              <span
                                className="h-[5px] w-[5px] rounded-full"
                                style={{ backgroundColor: st.dot }}
                              />
                              {st.label}
                            </div>
                          </td>
                          <td className="px-3 py-2.5 font-mono text-(--color-text-muted)">
                            {formatRelativeTime(job.created_at)}
                          </td>
                        </tr>
                      );
                    })
                  : committeeJobs.map((c) => {
                      const st = STATUS_STYLES[c.status] ?? {
                        dot: "var(--color-text-muted)",
                        bg: "var(--color-glass-hover)",
                        text: "var(--color-text-muted)",
                        label: c.status,
                      };
                      return (
                        <tr
                          key={c.job_id}
                          className="group cursor-pointer border-b border-(--color-glass-border) transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
                          onClick={() => {
                            navigate(`/committee/full-cycle/${c.job_id}`);
                          }}
                        >
                          <td className="px-3 py-2.5 font-mono text-(--color-text-primary)">
                            {c.job_id.includes("_")
                              ? c.job_id.slice(-8)
                              : c.job_id.slice(0, 8)}&hellip;
                          </td>
                          <td className="px-3 py-2.5">
                            <div
                              className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-[0.08em] uppercase"
                              style={{
                                backgroundColor: st.bg,
                                color: st.text,
                              }}
                            >
                              <span
                                className="h-[5px] w-[5px] rounded-full"
                                style={{ backgroundColor: st.dot }}
                              />
                              {st.label}
                            </div>
                          </td>
                          <td className="max-w-[180px] truncate px-3 py-2.5 text-(--color-text-secondary)">
                            {c.survivors.length > 0
                              ? c.survivors.join(", ")
                              : `${c.survivors_count} models`}
                          </td>
                          <td
                            className="px-3 py-2.5 font-mono text-(--color-text-primary)"
                            style={{
                              color: (c.avg_sharpe ?? 0) >= 0 ? "#089981" : "#F23645",
                            }}
                          >
                            {c.avg_sharpe != null
                              ? (c.avg_sharpe >= 0 ? "+" : "") + c.avg_sharpe.toFixed(2)
                              : "\u2014"}
                          </td>
                          <td
                            className="px-3 py-2.5 font-mono text-(--color-text-primary)"
                            style={{
                              color:
                                (c.trust_score ?? 0) >= 0.8
                                  ? "#089981"
                                  : (c.trust_score ?? 0) >= 0.6
                                    ? "#F2B436"
                                    : "#F23645",
                            }}
                          >
                            {c.trust_score != null
                              ? (c.trust_score * 100).toFixed(0) + "%"
                              : "\u2014"}
                          </td>
                          <td className="px-3 py-2.5 font-mono text-(--color-text-muted)">
                            {c.started_at
                              ? formatRelativeTime(c.started_at)
                              : "\u2014"}
                          </td>
                        </tr>
                      );
                    })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
