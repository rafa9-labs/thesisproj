import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Database, AlertTriangle } from "lucide-react";
import { useAppStore } from "@/stores/useAppStore";
import { useDashboardStore } from "@/stores/useDashboardStore";
import apiClient from "@/api/client";
import { useDashboardWaterfall } from "@/hooks/useDashboardWaterfall";
import { AlgoPerformancePanel } from "./AlgoPerformancePanel";
import { SentimentNewsWidget } from "./SentimentNewsWidget";
import { PriceTicker } from "./PriceTicker";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

function WidgetErrorFallback({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-6">
      <AlertTriangle size={16} className="text-(--color-accent-warning)" />
      <span className="text-[10px] text-(--color-text-muted)">{label} unavailable</span>
    </div>
  );
}

function WidgetSkeleton({ height = "80px", label }: { height?: string; label: string }) {
  return (
    <div
      className="animate-pulse rounded-lg border border-(--color-glass-border) bg-(--color-glass)"
      style={{ height }}
    >
      <div className="flex h-full items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <div className="h-3 w-20 rounded bg-(--color-glass-hover)" />
          <span className="text-[10px] text-(--color-text-muted)">{label} loading&hellip;</span>
        </div>
      </div>
    </div>
  );
}

function AlgoPanelSkeleton() {
  return (
    <div className="animate-pulse rounded-lg border border-(--color-glass-border) bg-(--color-glass)">
      <div className="border-b border-(--color-glass-border) px-5 py-3.5">
        <div className="h-3 w-48 rounded bg-(--color-glass-hover)" />
      </div>
      <div className="px-5 py-4">
        <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="flex flex-col gap-1">
              <div className="h-3 w-20 rounded bg-(--color-glass-hover)" />
              <div className="h-7 w-16 rounded bg-(--color-glass-hover)" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function DashboardPage() {
  const queryClient = useQueryClient();
  const { demoMode } = useAppStore();
  const activePair = useDashboardStore((s) => s.activePair);
  const activeTimeframe = useDashboardStore((s) => s.activeTimeframe);
  const setActivePair = useDashboardStore((s) => s.setActivePair);
  const setActiveTimeframe = useDashboardStore((s) => s.setActiveTimeframe);

  const {
    isPricesReady,
    isCandlesReady,
    isSentimentReady,
    isPerformanceReady,
    top3Pairs,
    pairs,
    completedJobs,
    totalJobCount,
    kpis,
    equityDataMap,
    isJobsLoading,
    completedCommittees,
    totalCommitteeCount,
    committeeKpis,
    isCommitteesLoading,
  } = useDashboardWaterfall(activePair, !demoMode);

  const availablePairs = useMemo(
    () => pairs.map((p) => p.pair?.symbol ?? "").filter((s) => s !== ""),
    [pairs],
  );

  const activePairRef = useRef(activePair);
  useEffect(() => {
    activePairRef.current = activePair;
  }, [activePair]);

  useEffect(() => {
    if (availablePairs.length > 0 && !availablePairs.includes(activePairRef.current)) {
      setActivePair(availablePairs[0]);
    }
  }, [availablePairs, setActivePair]);

  useEffect(() => {
    if (availablePairs.length === 0) return;
    for (const pair of availablePairs) {
      queryClient.prefetchQuery({
        queryKey: ["candles", pair, "M30", 150],
        queryFn: async () => {
          const { data } = await apiClient.get(`/candles/${pair}/M30`, {
            params: { limit: 150 },
          });
          return data;
        },
        staleTime: 15_000,
      });
    }
  }, [availablePairs, queryClient]);

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

  const fallbackPairs =
    availablePairs.length > 0 ? availablePairs : top3Pairs;

  return (
    <div ref={pageContentRef} className="flex flex-1 flex-col gap-3 overflow-hidden">
      {demoMode && (
        <div className="flex shrink-0 items-center gap-3 rounded-sm bg-[rgba(0,229,255,0.08)] border border-[rgba(0,229,255,0.2)] px-4 py-2.5 text-xs text-(--color-brand)">
          <Database size={14} />
          <span className="font-semibold">Local Data Mode</span>
          <span className="text-(--color-text-muted)">
            &mdash; Viewing pre-loaded historical seed data. Connect OANDA for live market data.
          </span>
        </div>
      )}

      {!isPricesReady ? (
        <WidgetSkeleton height="56px" label="Price Ticker" />
      ) : (
        <ErrorBoundary fallback={<WidgetErrorFallback label="Price Ticker" />}>
          <PriceTicker pairs={fallbackPairs} activePair={activePair} />
        </ErrorBoundary>
      )}

      {!isCandlesReady ? (
        <WidgetSkeleton height="360px" label="Price Chart" />
      ) : (
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
              {fallbackPairs.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div className="p-4">
            <ErrorBoundary fallback={<WidgetErrorFallback label="Price Chart" />}>
              <CandlestickChart
                pair={activePair}
                timeframe={activeTimeframe}
                limit={150}
                height={440}
                onTimeframeChange={setActiveTimeframe}
              />
            </ErrorBoundary>
          </div>
        </div>
      )}

      <div
        className="h-1 flex-shrink-0 cursor-row-resize bg-(--color-glass-border) transition-colors hover:bg-(--color-brand)"
        onMouseDown={onSplitMouseDown}
      />

      <div className="min-h-0 flex-1 overflow-hidden">
        <div className="flex flex-col gap-4">
          {!isSentimentReady ? (
            <WidgetSkeleton height="200px" label="Market Sentiment" />
          ) : (
            <div className="rounded-lg border border-(--color-glass-border) bg-(--color-glass)">
              <ErrorBoundary fallback={<WidgetErrorFallback label="Market Sentiment" />}>
                <SentimentNewsWidget pair={activePair} />
              </ErrorBoundary>
            </div>
          )}

          {!isPerformanceReady ? (
            <AlgoPanelSkeleton />
          ) : (
            <ErrorBoundary fallback={<WidgetErrorFallback label="Algo Performance" />}>
              <AlgoPerformancePanel
                kpis={kpis}
                jobs={completedJobs}
                totalJobCount={totalJobCount}
                equityData={equityDataMap}
                isLoading={isJobsLoading}
                committeeKpis={committeeKpis}
                committeeJobs={completedCommittees}
                totalCommitteeCount={totalCommitteeCount}
                isCommitteesLoading={isCommitteesLoading}
              />
            </ErrorBoundary>
          )}
        </div>
      </div>
    </div>
  );
}
