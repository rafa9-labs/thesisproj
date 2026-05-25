import { useMemo, useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { EmptyState } from "@/components/shared/EmptyState";
import { useJobHistory, usePairs } from "@/api/queries";
import apiClient from "@/api/client";
import type { JobResults } from "@/api/schemas";
import { formatMetric, formatPercent } from "@/lib/formatters";
import { useDashboardKPIs } from "./DashboardKPIs";
import { RecentJobsTable } from "./RecentJobsTable";
import { QuickActions } from "./QuickActions";
import { MarketPulsePanel, NewsArticlesPanel } from "./MarketPulsePanel";
import { PriceTicker } from "./PriceTicker";
import { CandlestickChart } from "@/components/charts/CandlestickChart";

function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-6 animate-pulse">
      <div className="h-10" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="h-[80px] rounded-lg" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="h-[360px] rounded-lg" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 3 }, (_, i) => (
          <div key={i} className="h-[300px] rounded-lg" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        ))}
      </div>
    </div>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const { data: pairs } = usePairs();
  const { data: jobs, isLoading: jobsLoading } = useJobHistory(50);

  const availablePairs = useMemo(
    () => (pairs ?? []).map((p) => p.pair?.symbol ?? "").filter((s) => s !== ""),
    [pairs],
  );

  const [activePair, setActivePair] = useState("EURUSD");
  const activePairRef = useRef(activePair);

  useEffect(() => {
    activePairRef.current = activePair;
  }, [activePair]);

  useEffect(() => {
    if (availablePairs.length > 0 && !availablePairs.includes(activePairRef.current)) {
      setActivePair(availablePairs[0]);
    }
  }, [availablePairs]);

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
      {/* ── Utility bar: pair selector + quick actions (no title) ───────────── */}
      <div className="flex items-center justify-end gap-2">
        <select
          value={activePair}
          onChange={(e) => setActivePair(e.target.value)}
          className="rounded border px-2 py-1 text-[11px] transition focus:outline-none"
          style={{
            borderColor: "var(--color-glass-border)",
            backgroundColor: "var(--color-glass)",
            color: "var(--color-text-primary)",
            fontFamily: "var(--font-mono)",
            height: 28,
          }}
        >
          {(availablePairs.length > 0 ? availablePairs : ["EURUSD", "GBPUSD", "USDJPY"]).map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <QuickActions />
      </div>

      {/* ── Ticker ribbon ──────────────────────────────────────────────────── */}
      <PriceTicker pairs={availablePairs.length > 0 ? [activePair, ...availablePairs.filter((p) => p !== activePair).slice(0, 2)] : ["EURUSD", "GBPUSD", "USDJPY"]} />

      {/* ── Full-width candlestick chart ────────────────────────────────────── */}
      <div
        className="rounded-lg border"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
          padding: "16px",
        }}
      >
        <CandlestickChart pair={activePair} timeframe="M30" limit={150} height={440} />
      </div>

      {/* ── 3-column widget grid ───────────────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4">
        {/* Col 1: ALGO Metrics — flat label/value list, no nested boxes */}
        <div
          className="rounded-lg border flex flex-col"
          style={{
            borderColor: "var(--color-glass-border)",
            backgroundColor: "var(--color-glass)",
            padding: "16px",
          }}
        >
          <span
            className="text-[10px] font-medium uppercase tracking-[0.12em] mb-3"
            style={{ color: "var(--color-text-muted)" }}
          >
            ALGO Metrics
          </span>
          <div className="flex flex-col divide-y" style={{ borderColor: "var(--color-glass-border)" }}>
            {[
              {
                label: "AVG SHARPE",
                value: formatMetric(kpis.avgSharpe),
                color: (kpis.avgSharpe ?? 0) >= 1
                  ? "var(--color-accent-success)"
                  : (kpis.avgSharpe ?? 0) >= 0.5
                  ? "var(--color-accent-warning)"
                  : "var(--color-accent-danger)",
              },
              {
                label: "WIN RATE",
                value: formatPercent(kpis.avgWinRate, 1),
                color: (kpis.avgWinRate ?? 0) >= 0.5
                  ? "var(--color-accent-success)"
                  : "var(--color-accent-danger)",
              },
              {
                label: "PROFIT. MONTHS",
                value: formatPercent(kpis.profitableMonthsPct, 0),
                color: (kpis.profitableMonthsPct ?? 0) >= 0.6
                  ? "var(--color-accent-success)"
                  : "var(--color-text-secondary)",
              },
              {
                label: "AVG RETURN",
                value: formatPercent(kpis.avgReturn, 2),
                color: (kpis.avgReturn ?? 0) >= 0
                  ? "var(--color-accent-success)"
                  : "var(--color-accent-danger)",
              },
              {
                label: "MAX DRAWDOWN",
                value: formatPercent(kpis.maxDrawdown, 2),
                color: "var(--color-accent-danger)",
              },
            ].map(({ label, value, color }) => (
              <div
                key={label}
                className="flex items-center justify-between py-2"
                style={{ borderColor: "var(--color-glass-border)" }}
              >
                <span
                  className="text-[10px] font-medium uppercase tracking-[0.08em]"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  {label}
                </span>
                <span
                  className="text-[13px] font-semibold tabular-nums"
                  style={{ fontFamily: "var(--font-mono)", color }}
                >
                  {value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Col 2: Market Sentiment */}
        <div
          className="rounded-lg border flex flex-col"
          style={{
            borderColor: "var(--color-glass-border)",
            backgroundColor: "var(--color-glass)",
            padding: "16px",
          }}
        >
          <span
            className="text-[10px] font-medium uppercase tracking-[0.12em] mb-3"
            style={{ color: "var(--color-text-muted)" }}
          >
            Market Sentiment
          </span>
          <MarketPulsePanel pair={activePair} />
        </div>

        {/* Col 3: Top News Articles */}
        <div
          className="rounded-lg border flex flex-col"
          style={{
            borderColor: "var(--color-glass-border)",
            backgroundColor: "var(--color-glass)",
            padding: "16px",
          }}
        >
          <NewsArticlesPanel pair={activePair} />
        </div>
      </div>

      {/* ── Recent Activity table ─────────────────────────────────────────── */}
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
