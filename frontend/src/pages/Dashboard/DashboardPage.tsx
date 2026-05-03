import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, TrendingUp, Activity, Trophy } from "lucide-react";
import { MetricCard } from "@/components/shared/MetricCard";
import { EmptyState } from "@/components/shared/EmptyState";
import { useJobHistory, useHeatmap } from "@/api/queries";
import apiClient from "@/api/client";
import type { JobResults } from "@/api/schemas";
import { formatMetric, formatPercent } from "@/lib/formatters";
import { useDashboardKPIs } from "./DashboardKPIs";
import { RecentJobsTable } from "./RecentJobsTable";
import { PerformanceHeatmapSection } from "./PerformanceHeatmapSection";

function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-5 animate-pulse">
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-24 rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
        ))}
      </div>
      <div className="h-[300px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
    </div>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const { data: jobs, isLoading: jobsLoading } = useJobHistory(50);

  const completedJobs = useMemo(
    () => (jobs ?? []).filter((j) => j.status === "completed").slice(0, 10),
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

  const { data: heatmapData, isLoading: heatmapLoading } = useHeatmap();

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
      <div className="flex flex-col gap-5">
        <DashboardSkeleton />
      </div>
    );
  }

  const hasCompleted = completedJobs.length > 0;

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total Runs" value={kpis.totalRuns} icon={<BarChart3 size={16} />} />
        <MetricCard
          label="Best Sharpe"
          value={formatMetric(kpis.bestSharpe)}
          icon={<Trophy size={16} />}
          delta={kpis.bestSharpe !== null ? (kpis.bestSharpe >= 1 ? "Excellent" : kpis.bestSharpe >= 0.5 ? "Good" : "Weak") : null}
          deltaType={(kpis.bestSharpe ?? 0) >= 1 ? "positive" : "neutral"}
        />
        <MetricCard
          label="Avg Win Rate"
          value={formatPercent(kpis.avgWinRate, 1)}
          icon={<Activity size={16} />}
        />
        <MetricCard
          label="Best Return"
          value={formatPercent(kpis.bestReturn)}
          icon={<TrendingUp size={16} />}
          deltaType={(kpis.bestReturn ?? 0) >= 0 ? "positive" : "negative"}
        />
      </div>

      <RecentJobsTable jobs={completedJobs.slice(0, 10)} equityData={equityDataMap} />

      <PerformanceHeatmapSection data={heatmapData} isLoading={heatmapLoading} />

      {!hasCompleted && (
        <EmptyState
          title="No backtests yet"
          description="Run your first backtest to see performance data, model comparisons, and historical results here."
          actionLabel="Run First Backtest"
          onAction={() => navigate("/backtest")}
        />
      )}
    </div>
  );
}
