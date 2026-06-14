import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Database, BarChart3 } from "lucide-react";
import { useJobHistory, usePairs } from "@/api/queries";
import { useAppStore } from "@/stores/useAppStore";
import apiClient from "@/api/client";
import type { JobResults } from "@/api/schemas";
import { formatMetric, formatPercent } from "@/lib/formatters";
import { useDashboardKPIs } from "./DashboardKPIs";
import { RecentJobsTable } from "./RecentJobsTable";
import { MarketPulsePanel, NewsArticlesPanel } from "./MarketPulsePanel";
import { PriceTicker } from "./PriceTicker";
import { CandlestickChart } from "@/components/charts/CandlestickChart";

function DashboardSkeleton() {
  return (
    <div className="flex animate-pulse flex-col gap-6">
      <div className="h-10 bg-(--color-glass-hover)" />
      <div className="h-[80px] rounded-sm bg-(--color-glass-hover)" />
      <div className="h-[360px] rounded-sm bg-(--color-glass-hover)" />
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 3 }, (_, i) => (
          <div key={i} className="h-[300px] rounded-sm bg-(--color-glass-hover)" />
        ))}
      </div>
    </div>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const { demoMode } = useAppStore();
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

  const completedIds = useMemo(() => completedJobs.map((j) => j.job_id), [completedJobs]);

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
        const curves = r.metrics?.flatMap((m) => (m.equity_curve ? [m.equity_curve] : [])) ?? [];
        map[r.job_id] = curves.length > 0 ? curves[0] : null;
      }
    }
    return map;
  }, [allResults.data]);

  const [chartFraction, setChartFraction] = useState(0.6);
  const splitDragRef = useRef({ active: false, startY: 0, startFrac: 0.6 });
  const pageContentRef = useRef<HTMLDivElement>(null);

  const onSplitMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      splitDragRef.current = { active: true, startY: e.clientY, startFrac: chartFraction };
    },
    [chartFraction],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = splitDragRef.current;
      if (!d.active || !pageContentRef.current) return;
      const rect = pageContentRef.current.getBoundingClientRect();
      const total = rect.height;
      if (total <= 0) return;
      const dy = e.clientY - d.startY;
      const newFrac = Math.max(0.3, Math.min(0.8, d.startFrac + dy / total));
      setChartFraction(newFrac);
    };
    const onUp = () => {
      splitDragRef.current.active = false;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, []);

  if (jobsLoading) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <DashboardSkeleton />
      </div>
    );
  }

  const hasCompleted = completedJobs.length > 0;

  return (
    <div ref={pageContentRef} className="flex flex-1 flex-col gap-4 overflow-hidden">
      {/* Page header */}
      <div className="flex shrink-0 flex-col gap-1">
        <h1 className="text-[28px] font-bold tracking-tight text-(--color-text-primary)">
          Dashboard
        </h1>
        <p className="text-[13px] text-(--color-text-muted)">
          Live market overview and recent backtest performance.
        </p>
      </div>

      {demoMode && (
        <div
          className="flex shrink-0 items-center gap-3 rounded-sm px-4 py-2.5 text-xs text-(--color-brand)"
          style={{
            backgroundColor: "rgba(0, 229, 255, 0.08)",
            border: "1px solid rgba(0, 229, 255, 0.2)",
          }}
        >
          <Database size={14} />
          <span className="font-semibold">Demo Mode</span>
          <span className="text-(--color-text-muted)">
            &mdash; Using pre-loaded sample market data. Run a real backtest or connect OANDA for
            live data.
          </span>
        </div>
      )}

      <PriceTicker
        pairs={
          availablePairs.length > 0
            ? [activePair, ...availablePairs.filter((p) => p !== activePair).slice(0, 2)]
            : ["EURUSD", "GBPUSD", "USDJPY"]
        }
      />

      <div
        className="min-h-[280px] shrink-0 overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-glass)"
        style={{ height: `${chartFraction * 100}%` }}
      >
        <div className="flex h-10 items-center justify-between border-b border-(--color-glass-border) px-4">
          <span className="text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
            Price Chart
          </span>
          <select
            value={activePair}
            onChange={(e) => setActivePair(e.target.value)}
            aria-label="Select chart pair"
            className="h-7 rounded border border-(--color-glass-border) bg-(--color-elevated) px-2 font-mono text-[11px] text-(--color-text-primary) transition focus:outline-none"
          >
            {(availablePairs.length > 0 ? availablePairs : ["EURUSD", "GBPUSD", "USDJPY"]).map(
              (p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ),
            )}
          </select>
        </div>
        <div className="p-4">
          <CandlestickChart pair={activePair} timeframe="M30" limit={150} height={440} />
        </div>
      </div>

      <div
        className="h-1 flex-shrink-0 cursor-row-resize bg-(--color-glass-border) transition-colors hover:bg-(--color-brand)"
        onMouseDown={onSplitMouseDown}
      />

      <div className="min-h-0 flex-1 overflow-hidden">
        <div className="flex flex-col gap-4">
          {/* ── Top Row: 50/50 Sentiment + Articles ── */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="flex flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
              <span className="mb-3 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
                Market Sentiment
              </span>
              <MarketPulsePanel pair={activePair} />
            </div>

            <div className="flex flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
              <NewsArticlesPanel pair={activePair} />
            </div>
          </div>

          {/* ── Bottom Row: Full-width ALGO Metrics ── */}
          <div className="rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
            <span className="mb-3 block text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
              ALGO Metrics
            </span>
            {allResults.isLoading || jobsLoading ? (
              <div className="flex flex-col divide-y border-(--color-glass-border)">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex animate-pulse items-center justify-between py-2">
                    <div className="h-3 w-20 rounded bg-(--color-glass-hover)" />
                    <div className="h-4 w-14 rounded bg-(--color-glass-hover)" />
                  </div>
                ))}
              </div>
            ) : completedIds.length === 0 ? (
              <div className="flex items-center justify-between rounded-lg border border-(--color-glass-border) bg-(--color-glass-hover) px-5 py-4">
                <div className="flex items-center gap-4">
                  <BarChart3 size={24} className="text-(--color-text-dim)" />
                  <span className="text-[11px] leading-relaxed text-(--color-text-dim)">
                    No backtest data populated for current workspace.
                  </span>
                </div>
                <button
                  onClick={() => navigate("/backtest")}
                  className="rounded px-5 py-2 text-[11px] font-bold tracking-[0.06em] text-(--color-text-inverse) uppercase transition hover:brightness-110"
                  style={{ backgroundColor: "var(--color-brand)" }}
                >
                  Run First Backtest
                </button>
              </div>
            ) : (
              <div className="flex flex-wrap gap-x-12 gap-y-0 divide-y border-(--color-glass-border)">
                {[
                  {
                    label: "AVG SHARPE",
                    value: formatMetric(kpis.avgSharpe),
                    color:
                      (kpis.avgSharpe ?? 0) >= 1
                        ? "var(--color-accent-success)"
                        : (kpis.avgSharpe ?? 0) >= 0.5
                          ? "var(--color-accent-warning)"
                          : "var(--color-accent-danger)",
                  },
                  {
                    label: "WIN RATE",
                    value: formatPercent(kpis.avgWinRate, 1),
                    color:
                      (kpis.avgWinRate ?? 0) >= 0.5
                        ? "var(--color-accent-success)"
                        : "var(--color-accent-danger)",
                  },
                  {
                    label: "PROFIT. MONTHS",
                    value: formatPercent(kpis.profitableMonthsPct, 0),
                    color:
                      (kpis.profitableMonthsPct ?? 0) >= 0.6
                        ? "var(--color-accent-success)"
                        : "var(--color-text-secondary)",
                  },
                  {
                    label: "AVG RETURN",
                    value: formatPercent(kpis.avgReturn, 2),
                    color:
                      (kpis.avgReturn ?? 0) >= 0
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
                    className="flex items-center justify-between border-(--color-glass-border) py-2"
                  >
                    <span className="text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
                      {label}
                    </span>
                    <span
                      className="font-mono text-[13px] font-semibold tabular-nums"
                      style={{ color }}
                    >
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <RecentJobsTable jobs={completedJobs} equityData={equityDataMap} />
          </div>
        </div>
      </div>
    </div>
  );
}
