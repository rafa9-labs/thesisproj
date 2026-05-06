import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, Activity, TrendingUp } from "lucide-react";
import { MetricCard } from "@/components/shared/MetricCard";
import { EmptyState } from "@/components/shared/EmptyState";
import { useJobHistory, usePairs } from "@/api/queries";
import apiClient from "@/api/client";
import type { JobResults } from "@/api/schemas";
import { formatMetric, formatPercent } from "@/lib/formatters";
import { useDashboardKPIs } from "./DashboardKPIs";
import { RecentJobsTable } from "./RecentJobsTable";
import { QuickActions } from "./QuickActions";
import { MarketPulsePanel } from "./MarketPulsePanel";
import { PriceTicker } from "./PriceTicker";
import { CandlestickChart } from "@/components/charts/CandlestickChart";

function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-6 animate-pulse">
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 3 }, (_, i) => (
          <div key={i} className="h-28 rounded-lg" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        ))}
      </div>
      <div className="h-[200px] rounded-lg" style={{ backgroundColor: "var(--color-glass-hover)" }} />
    </div>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const { data: pairs } = usePairs();
  const { data: jobs, isLoading: jobsLoading } = useJobHistory(50);

  const availablePairs = useMemo(
    () => (pairs ?? []).map((p) => p.symbol),
    [pairs],
  );

  const [activePair, setActivePair] = useState(() => availablePairs[0] ?? "EURUSD");

  const completedJobs = useMemo(
    () => (jobs ?? []).filter((j) => j.status === "completed").slice(0, 5),
    [jobs],
  );

  const completedIds = useMemo(
    () => completedJobs.map((j) => j.job_id),
    [completedJobs],
  );

  const allResults = useQuery({
    queryKey: ["dashboard-aggregate", completedIds],
    queryFn: async (): Promise<JobResults[]> => {
      const results = await Promise.allSettled(
        completedIds.map((id) =>
          apiClient.get<JobResults>(`/backtest/${id}/results`).then((r) => r.data),
        ),
      );
      const successful: JobResults[] = [];
      for (const r of results) {
        if (r.status === "fulfilled") successful.push(r.value);
      }
      return successful;
    },
    enabled: completedIds.length > 0,
    staleTime: 60_000,
  });

  const kpis = useDashboardKPIs(allResults.data ?? []);

  const equityDataMap = useMemo(() => {
    const map: Record<string, import("@/api/schemas").EquityPoint[] | null> = {};
    if (allResults.data) {
      for (const r of allResults.data) {
        const curves = r.metrics?.flatMap((m) => m.equity_curve ? [m.equity_curve] : []) ?? [];
        map[r.job_id] = curves.length > 0 ? curves[0] : null;
      }
    }
    return map;
  }, [allResults.data]);

  if (jobsLoading) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <DashboardSkeleton />
      </div>
    );
  }

  const hasCompleted = completedJobs.length > 0;

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <QuickActions />
        {availablePairs.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
              Pair
            </span>
            <select
              value={activePair}
              onChange={(e) => setActivePair(e.target.value)}
              className="rounded-md border px-2.5 py-1 text-xs transition focus:outline-none"
              style={{
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-glass)",
                color: "var(--color-text-primary)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {availablePairs.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      <PriceTicker pairs={availablePairs.length > 0 ? [activePair, ...availablePairs.filter((p) => p !== activePair).slice(0, 2)] : ["EURUSD", "GBPUSD", "USDJPY"]} />

      <CandlestickChart pair={activePair} timeframe="M30" limit={150} height={360} />

      <div>
        <h3
          className="text-[11px] font-medium uppercase tracking-[0.12em] mb-3"
          style={{ color: "var(--color-text-muted)" }}
        >
          Market Pulse
        </h3>
        <MarketPulsePanel pair={activePair} />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="Avg Sharpe"
          value={formatMetric(kpis.avgSharpe)}
          icon={<BarChart3 size={16} strokeWidth={1.5} />}
          delta={kpis.avgSharpe !== null ? (kpis.avgSharpe >= 1 ? "Strong" : kpis.avgSharpe >= 0.5 ? "Moderate" : "Weak") : null}
          deltaType={(kpis.avgSharpe ?? 0) >= 1 ? "positive" : (kpis.avgSharpe ?? 0) >= 0.5 ? "neutral" : "negative"}
        />
        <MetricCard
          label="Avg Win Rate"
          value={formatPercent(kpis.avgWinRate, 1)}
          icon={<Activity size={16} strokeWidth={1.5} />}
          delta="Fraction of winning trades across all completed model-runs"
          deltaType="neutral"
        />
        <MetricCard
          label="Profitable Months"
          value={formatPercent(kpis.profitableMonthsPct, 0)}
          icon={<TrendingUp size={16} strokeWidth={1.5} />}
          delta="Months with positive return across all runs and models"
          deltaType="neutral"
        />
      </div>

      <RecentJobsTable jobs={completedJobs} equityData={equityDataMap} />

      {!hasCompleted && (
        <EmptyState
          title="No backtests yet"
          description="Run your first backtest to populate the dashboard with performance data."
          actionLabel="Run First Backtest"
          onAction={() => navigate("/backtest")}
        />
      )}
    </div>
  );
}