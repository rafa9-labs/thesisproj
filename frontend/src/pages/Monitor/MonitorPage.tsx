import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, FlaskConical, Square, Wifi, WifiOff, ChevronDown, ChevronUp } from "lucide-react";
import { useActiveBacktests, useForceStopJob, useJobStatus } from "@/api/queries";
import { useJobStore } from "@/stores/useJobStore";
import { useBacktestWebSocket } from "@/hooks/useBacktestWebSocket";
import { useBacktestProgress } from "@/api/queries";
import { JobPillStrip } from "./JobPillStrip";
import { EquityChart } from "./EquityChart";
import { MonthlyHeatmap } from "./MonthlyHeatmap";
import { TradeLogFeed } from "./TradeLogFeed";
import { HpoScatterChart } from "./HpoScatterChart";
import { HpoTrialFeed } from "./HpoTrialFeed";
import { ModelHealthTable } from "./ModelHealthTable";
import { wsManager } from "@/api/websocket";
import type { JobSummary } from "@/api/schemas";

export function MonitorPage() {
  const navigate = useNavigate();
  const { data: activeData, isLoading } = useActiveBacktests();
  const activeJobs = useJobStore((s) => s.activeJobs);
  const selectedJobId = useJobStore((s) => s.selectedJobId);
  const completedJobIds = useJobStore((s) => s.completedJobIds);
  const selectJob = useJobStore((s) => s.selectJob);
  const getJob = useJobStore((s) => s.getJob);
  const setActiveTab = useJobStore((s) => s.setActiveTab);
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);
  const markCompleted = useJobStore((s) => s.markCompleted);
  const clearCompletedJobs = useJobStore((s) => s.clearCompletedJobs);
  const forceStop = useForceStopJob();
  const ensureJob = useJobStore((s) => s.ensureJob);

  const [wsConnected, setWsConnected] = useState(false);
  const [split, setSplit] = useState(50);
  const [hpoModelFilter, setHpoModelFilter] = useState<string | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null);
  const splitDragRef = useRef({ active: false, startY: 0, startSplit: 50 });
  const mainRef = useRef<HTMLDivElement>(null);

  const activeList = useMemo(() => activeData?.jobs ?? [], [activeData?.jobs]);
  const runningList = useMemo(
    () => activeList.filter((j) => j.status === "pending" || j.status === "running"),
    [activeList],
  );

  useEffect(() => {
    for (const job of runningList) {
      if (!activeJobs.has(job.job_id)) {
        ensureJob(job.job_id, job.pair, job.models);
      }
    }
  }, [runningList, activeJobs, ensureJob]);

  const runningIds = useMemo(() => new Set(runningList.map((j) => j.job_id)), [runningList]);

  const prevRunningIds = useRef<{ ids: Set<string>; initialized: boolean }>({
    ids: new Set(),
    initialized: false,
  });

  useEffect(() => {
    const currentIds = runningIds;
    if (!prevRunningIds.current.initialized) {
      prevRunningIds.current = { ids: currentIds, initialized: true };
      return;
    }
    const newIds = [...currentIds].filter((id) => !prevRunningIds.current.ids.has(id));
    if (newIds.length > 0) {
      clearCompletedJobs();
    }
    prevRunningIds.current.ids = currentIds;
  }, [runningIds, clearCompletedJobs]);

  const completedSummaries: JobSummary[] = useMemo(() => {
    if (completedJobIds.size === 0) return [];
    return [...completedJobIds]
      .filter((id) => activeJobs.has(id))
      .map((id) => {
        const j = activeJobs.get(id)!;
        return {
          job_id: id,
          type: "backtest",
          status: j.status as "completed" | "failed",
          pair: j.pair,
          models: j.models,
          created_at: j.createdAt.toISOString(),
        };
      });
  }, [completedJobIds, activeJobs]);

  const allJobs = useMemo(
    () => [...activeList, ...completedSummaries],
    [activeList, completedSummaries],
  );
  const visibleIds = useMemo(
    () => new Set([...runningIds, ...completedJobIds]),
    [runningIds, completedJobIds],
  );

  useEffect(() => {
    if (visibleIds.size > 0) {
      const selectedStillVisible =
        selectedJobId && visibleIds.has(selectedJobId) && activeJobs.has(selectedJobId);
      if (!selectedStillVisible) {
        const firstRunning = [...visibleIds].find((id) => runningIds.has(id) && activeJobs.has(id));
        const first = firstRunning ?? [...visibleIds].find((id) => activeJobs.has(id)) ?? null;
        if (first) {
          selectJob(first);
          setActiveTab(first, "hpo-and-results");
        }
      }
    }
  }, [visibleIds, selectedJobId, selectJob, setActiveTab, activeJobs, runningIds]);

  const selectedJobStatus = selectedJobId ? getJob(selectedJobId)?.status : null;
  const wsJobId =
    selectedJobId && (selectedJobStatus === "pending" || selectedJobStatus === "running")
      ? selectedJobId
      : null;
  useBacktestWebSocket(wsJobId);

  useEffect(() => {
    if (!selectedJobId) return;
    const check = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 1000);
    return () => clearInterval(check);
  }, [selectedJobId]);

  const shouldPoll = selectedJobId != null;
  useBacktestProgress(shouldPoll ? selectedJobId : null);
  const { data: restStatus } = useJobStatus(shouldPoll ? selectedJobId : null);

  useEffect(() => {
    if (!restStatus || !selectedJobId) return;
    const local = getJob(selectedJobId);
    if (!local) return;
    if (restStatus.status === "failed" && local.status !== "failed") {
      handleWsEvent({
        event: "job_failed",
        job_id: selectedJobId,
        error: restStatus.error ?? "Job failed on server",
      });
      markCompleted(selectedJobId);
    }
    if (restStatus.status === "completed" && local.status !== "completed") {
      handleWsEvent({ event: "job_complete", job_id: selectedJobId, metrics: [] });
      markCompleted(selectedJobId);
    }
  }, [restStatus, selectedJobId, getJob, handleWsEvent, markCompleted]);

  const handleForceStop = async () => {
    if (!selectedJobId) return;
    const jobId = selectedJobId;
    try {
      await forceStop.mutateAsync(jobId);
      handleWsEvent({ event: "job_failed", job_id: jobId, error: "Stopped by user" });
    } catch {
      /* ignore */
    }
  };

  const selectedJob = selectedJobId ? getJob(selectedJobId) : undefined;
  const allOosPeriods = selectedJob?.oosPeriods ?? [];
  const allOosEquity = selectedJob?.oosEquity ?? [];
  const models = selectedJob?.models ?? [];
  const progress = selectedJob?.progress ?? 0;
  const progressText = selectedJob?.progressText ?? "";
  const status = selectedJob?.status;
  const isDone = status === "completed" || status === "failed";

  useEffect(() => {
    if (import.meta.env.DEV && allOosPeriods.length > 0) {
      const latest = allOosPeriods[allOosPeriods.length - 1];
      console.log("[Monitor] allOosPeriods count:", allOosPeriods.length);
      console.log("[Monitor] latest oos_result:", JSON.parse(JSON.stringify(latest)));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allOosPeriods.length]);

  const allHpoTrials = (selectedJob?.cycles ?? []).flatMap((c) =>
    c.hpoTrials.map((t) => ({ model: c.model, trial: t })),
  );

  const onSplitMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      splitDragRef.current = { active: true, startY: e.clientY, startSplit: split };
    },
    [split],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = splitDragRef.current;
      if (!d.active || !mainRef.current) return;
      const rect = mainRef.current.getBoundingClientRect();
      const total = rect.height;
      if (total <= 0) return;
      const dy = e.clientY - d.startY;
      setSplit(Math.max(20, Math.min(80, d.startSplit + (dy / total) * 100)));
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

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center bg-(--color-app)">
        <span className="text-xs text-(--color-text-muted)">Checking for backtests...</span>
      </div>
    );
  }

  if (runningList.length === 0 && completedJobIds.size === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded-sm border border-(--color-glass-border) bg-(--color-surface) py-16">
        <Eye size={40} strokeWidth={1} className="text-(--color-text-muted)" />
        <span className="text-sm font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
          No Active Backtests
        </span>
        <span className="text-xs text-(--color-text-muted)">
          Configure and deploy one from the Backtest tab
        </span>
        <button
          onClick={() => navigate("/backtest")}
          className="flex items-center gap-1.5 rounded-md bg-(--color-brand) px-4 py-2 text-[11px] font-medium tracking-[0.06em] text-(--color-text-inverse) uppercase transition-all hover:brightness-110"
        >
          <FlaskConical size={14} strokeWidth={2} />
          Go to Backtest
        </button>
      </div>
    );
  }

  return (
    <div ref={mainRef} className="flex h-full flex-col overflow-hidden px-4 lg:px-6">
      {/* ── Slim Header ── */}
      <div className="flex shrink-0 items-center gap-4 border-b border-(--color-glass-border) bg-(--color-surface) py-2">
        <JobPillStrip
          jobs={allJobs}
          selectedJobId={selectedJobId}
          onSelect={(id) => {
            selectJob(id);
            setActiveTab(id, "hpo-and-results");
          }}
        />

        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-(--color-elevated)">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${Math.min(progress, 100)}%`,
              backgroundColor: isDone
                ? selectedJob?.status === "failed"
                  ? "var(--color-accent-danger)"
                  : "var(--color-accent-success)"
                : "var(--color-brand)",
            }}
          />
        </div>
        <span className="font-mono text-[10px] text-(--color-text-secondary) tabular-nums">
          {Math.round(progress)}%
        </span>

        {selectedJobId && (
          <span
            className="flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium"
            style={{
              backgroundColor: wsConnected ? "rgba(34,197,94,0.08)" : "rgba(242,54,69,0.08)",
              color: wsConnected ? "var(--color-accent-success)" : "var(--color-accent-danger)",
            }}
          >
            {wsConnected ? <Wifi size={10} /> : <WifiOff size={10} />}
            {wsConnected ? "LIVE" : "DISCONNECTED"}
          </span>
        )}

        {!isDone && selectedJobId && (
          <button
            onClick={handleForceStop}
            disabled={forceStop.isPending}
            className="flex shrink-0 items-center gap-1 rounded border border-(--color-accent-danger) px-2 py-1 text-[10px] font-semibold tracking-[0.06em] text-(--color-accent-danger) uppercase transition hover:brightness-110"
          >
            <Square size={10} strokeWidth={2} fill="currentColor" />
            Force Stop
          </button>
        )}
      </div>

      {selectedJob ? (
        <>
          {/* ── Top Pane: Optimization Phase ── */}
          <div className="shrink-0 overflow-hidden border-b border-(--color-glass-border)">
            <div className="grid h-full grid-cols-1 gap-4 p-4 lg:grid-cols-3">
              <div className="flex h-full min-h-0 flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4 lg:col-span-2">
                <div className="mb-3 flex shrink-0 items-center gap-3">
                  <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                    HPO Convergence
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setHpoModelFilter(null)}
                      className="rounded px-2 py-0.5 text-[9px] font-medium uppercase transition"
                      style={{
                        backgroundColor: !hpoModelFilter
                          ? "var(--color-brand-glow)"
                          : "transparent",
                        color: !hpoModelFilter ? "var(--color-brand)" : "var(--color-text-muted)",
                      }}
                    >
                      All
                    </button>
                    {models.map((m) => (
                      <button
                        key={m}
                        onClick={() => setHpoModelFilter(m)}
                        className="rounded px-2 py-0.5 text-[9px] font-medium uppercase transition"
                        style={{
                          backgroundColor:
                            hpoModelFilter === m ? "var(--color-brand-glow)" : "transparent",
                          color:
                            hpoModelFilter === m ? "var(--color-brand)" : "var(--color-text-muted)",
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto [scrollbar-width:thin]">
                  <div className="min-h-[180px]">
                    <HpoScatterChart allTrials={allHpoTrials} filterModel={hpoModelFilter} />
                  </div>
                  <HpoTrialFeed
                    trials={
                      hpoModelFilter
                        ? allHpoTrials.filter((t) => t.model === hpoModelFilter)
                        : allHpoTrials
                    }
                  />
                </div>
              </div>

              <div className="flex h-full min-h-0 flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
                <span className="mb-3 shrink-0 text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  Model Health
                </span>
                <div className="min-h-0 flex-1 overflow-y-auto [scrollbar-width:thin]">
                  {selectedJob && <ModelHealthTable job={selectedJob} />}
                  <div className="mt-3 border-t border-(--color-glass-border) pt-3">
                    <span className="mb-2 block text-[9px] font-medium tracking-[0.08em] text-(--color-text-dim) uppercase">
                      OOS Period Performance
                    </span>
                    {(() => {
                      const total = allOosPeriods.length;
                      const longCount = allOosPeriods.filter((p) => (p.return_pct ?? 0) > 0).length;
                      const flatCount = allOosPeriods.filter(
                        (p) => (p.return_pct ?? 0) === 0,
                      ).length;
                      const shortCount = allOosPeriods.filter(
                        (p) => (p.return_pct ?? 0) < 0,
                      ).length;
                      const longPct = total > 0 ? (longCount / total) * 100 : 0;
                      const flatPct = total > 0 ? (flatCount / total) * 100 : 0;
                      const shortPct = total > 0 ? (shortCount / total) * 100 : 0;

                      const totalRawSignals = allOosPeriods.reduce(
                        (s, p) => s + (p.signals_raw ?? 0),
                        0,
                      );
                      const totalPassedSignals = allOosPeriods.reduce(
                        (s, p) => s + (p.signals_passed_gate ?? 0),
                        0,
                      );
                      const gateRate =
                        totalRawSignals > 0 ? (totalPassedSignals / totalRawSignals) * 100 : 0;

                      return (
                        <>
                          <div className="mb-1.5 flex h-5 overflow-hidden rounded-full bg-(--color-glass-hover)">
                            {longPct > 0 && (
                              <div
                                className="flex items-center justify-center text-[8px] font-bold text-white/80 transition-all"
                                style={{
                                  width: `${longPct}%`,
                                  backgroundColor: "var(--color-accent-success)",
                                }}
                              >
                                {longPct >= 12 ? `${Math.round(longPct)}%` : ""}
                              </div>
                            )}
                            {flatPct > 0 && (
                              <div
                                className="flex items-center justify-center text-[8px] font-bold text-white/60 transition-all"
                                style={{
                                  width: `${flatPct}%`,
                                  backgroundColor: "var(--color-text-dim)",
                                }}
                              >
                                {flatPct >= 12 ? `${Math.round(flatPct)}%` : ""}
                              </div>
                            )}
                            {shortPct > 0 && (
                              <div
                                className="flex items-center justify-center text-[8px] font-bold text-white/80 transition-all"
                                style={{
                                  width: `${shortPct}%`,
                                  backgroundColor: "var(--color-accent-danger)",
                                }}
                              >
                                {shortPct >= 12 ? `${Math.round(shortPct)}%` : ""}
                              </div>
                            )}
                          </div>
                          <div className="flex items-center justify-between font-mono text-[9px]">
                            <span className="text-(--color-accent-success)">
                              POS {Math.round(longPct)}%
                            </span>
                            <span className="text-(--color-text-dim)">
                              NEUT {Math.round(flatPct)}%
                            </span>
                            <span className="text-(--color-accent-danger)">
                              NEG {Math.round(shortPct)}%
                            </span>
                          </div>
                          {totalRawSignals > 0 && (
                            <div className="mt-2 font-mono text-[8px] text-(--color-text-dim)">
                              Gate: {totalPassedSignals}/{totalRawSignals} signals (
                              {Math.round(gateRate)}%)
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ── Resize Handle ── */}
          <div className="relative flex h-1 flex-shrink-0 items-center justify-center py-1">
            <div
              className="absolute inset-0 cursor-row-resize bg-(--color-glass-border) transition-colors hover:bg-(--color-brand)"
              onMouseDown={onSplitMouseDown}
            />
            <button
              onClick={() => setSplit(split > 50 ? 33 : 66)}
              className="relative z-10 flex h-5 w-5 items-center justify-center rounded-full border border-(--color-glass-border) bg-(--color-surface) text-(--color-text-muted) transition hover:border-(--color-brand) hover:text-(--color-brand)"
              aria-label={split > 50 ? "Expand bottom pane" : "Expand top pane"}
            >
              {split > 50 ? <ChevronDown size={10} /> : <ChevronUp size={10} />}
            </button>
          </div>

          {/* ── Bottom Pane: Execution Phase ── */}
          <div
            className="overflow-hidden p-6 transition-[flex] duration-300 ease-in-out"
            style={{ flex: `${100 - split} 0 0px`, minHeight: 350 }}
          >
            <div className="flex h-full flex-col gap-6 overflow-hidden lg:flex-row">
              <div className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
                <div className="flex items-center justify-between pt-4 lg:pt-0">
                  <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                    Walk-Forward Equity
                  </span>
                  <span className="font-mono text-[10px] text-(--color-text-muted)">
                    {progressText}
                  </span>
                </div>
                <div className="min-h-[400px] min-w-0 flex-1">
                  {selectedJob.cycles.length === 0 ? (
                    <div className="flex h-full items-center justify-center">
                      <span className="text-xs text-(--color-text-muted)">
                        Waiting for cycles...
                      </span>
                    </div>
                  ) : (
                    <EquityChart
                      models={models}
                      oosPeriods={allOosPeriods}
                      oosEquity={allOosEquity}
                    />
                  )}
                </div>
                {allOosPeriods.length > 0 && (
                  <div>
                    <MonthlyHeatmap periods={allOosPeriods} />
                  </div>
                )}
              </div>

              <div className="flex h-full shrink-0 flex-col overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-glass) lg:w-[320px]">
                <div className="flex shrink-0 items-center gap-2 border-b border-(--color-glass-border) px-3 py-2">
                  <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                    Period Results
                  </span>
                  <span className="font-mono text-[9px] text-(--color-text-dim)">
                    {Math.min(allOosPeriods.filter((p) => p.trades && p.trades > 0).length, 50)}{" "}
                    active
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-1.5 border-b border-[rgba(42,46,57,0.3)] px-3 py-1 font-mono text-[8px] tracking-[0.05em] text-(--color-text-dim) uppercase">
                  <span className="w-12 shrink-0">Period</span>
                  <span className="w-8 shrink-0 text-right">#</span>
                  <span className="flex-1 text-right">Return</span>
                  <span className="w-12 shrink-0 text-right">Sharpe</span>
                </div>
                <div className="flex-1 overflow-hidden">
                  <TradeLogFeed periods={allOosPeriods} models={models} />
                </div>
              </div>
            </div>

            <div className="shrink-0 border-t border-(--color-glass-border) pb-4">
              <div className="mt-4 flex flex-col gap-3 overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                    Per-Month Summary
                  </span>
                  {allOosPeriods.length > 0 &&
                    (() => {
                      const uniqueMonths = [...new Set(allOosPeriods.map((p) => p.period))].sort(
                        (a, b) => a - b,
                      );
                      return (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => setSelectedMonth(null)}
                            className="rounded px-2 py-0.5 text-[9px] font-medium uppercase transition"
                            style={{
                              backgroundColor:
                                selectedMonth === null ? "var(--color-brand-glow)" : "transparent",
                              color:
                                selectedMonth === null
                                  ? "var(--color-brand)"
                                  : "var(--color-text-dim)",
                            }}
                          >
                            All
                          </button>
                          {uniqueMonths.map((m) => (
                            <button
                              key={m}
                              onClick={() => setSelectedMonth(m)}
                              className="rounded px-2 py-0.5 font-mono text-[9px] font-medium transition"
                              style={{
                                backgroundColor:
                                  selectedMonth === m ? "var(--color-brand-glow)" : "transparent",
                                color:
                                  selectedMonth === m
                                    ? "var(--color-brand)"
                                    : "var(--color-text-dim)",
                              }}
                            >
                              Month {m}
                            </button>
                          ))}
                        </div>
                      );
                    })()}
                </div>
                <div className="overflow-x-auto">
                  <div className="min-w-[640px]">
                    <table className="w-full border-collapse overflow-hidden rounded-sm border border-(--color-glass-border)">
                      <thead className="bg-(--color-elevated)">
                        <tr>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-left text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Model
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-left text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Period
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-right text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Sharpe
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-right text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Return
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-right text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Trades
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-right text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            DD
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-right text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Win Rate
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-center text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Gate
                          </th>
                          <th className="border-b border-(--color-glass-border) px-2 py-1.5 text-center text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase">
                            Risk
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {allOosPeriods.length === 0 ? (
                          <tr>
                            <td
                              colSpan={9}
                              className="py-8 text-center text-[10px] text-(--color-text-muted)"
                            >
                              Awaiting monthly completion...
                            </td>
                          </tr>
                        ) : (
                          (() => {
                            const filtered =
                              selectedMonth != null
                                ? allOosPeriods.filter((p) => p.period === selectedMonth)
                                : allOosPeriods;
                            const groups = new Map<string, typeof filtered>();
                            for (const p of filtered) {
                              const m = p.model ?? "";
                              if (!groups.has(m)) groups.set(m, []);
                              groups.get(m)!.push(p);
                            }
                            const rows: React.ReactNode[] = [];
                            let first = true;
                            for (const [modelName, periods] of groups) {
                              if (!first)
                                rows.push(
                                  <tr key={`sep-${modelName}`}>
                                    <td colSpan={9} className="p-0">
                                      <div className="h-1.5" />
                                    </td>
                                  </tr>,
                                );
                              first = false;
                              for (const p of periods) {
                                rows.push(
                                  <tr
                                    key={`${modelName}-${p.period}`}
                                    className="border-b border-[rgba(42,46,57,0.4)]"
                                  >
                                    <td className="px-2 py-1 font-mono text-[10px] whitespace-nowrap text-(--color-brand)">
                                      {modelName}
                                    </td>
                                    <td className="px-2 py-1 font-mono text-[10px] whitespace-nowrap text-(--color-text-secondary)">
                                      M{p.period}
                                      {p.flat ? " (flat)" : ""}
                                    </td>
                                    <td
                                      className="px-2 py-1 text-right font-mono text-[10px] whitespace-nowrap text-(--color-text-primary)"
                                      title={`Train: ${p.train_sharpe?.toFixed(2) ?? "?"} | Test: ${p.sharpe?.toFixed(2) ?? "?"} | Gap: ${(p.sharpe_gap_pct ?? 0).toFixed(0)}%`}
                                    >
                                      {p.sharpe?.toFixed(2) ?? "-"}
                                    </td>
                                    <td className="px-2 py-1 text-right font-mono text-[10px] whitespace-nowrap text-(--color-text-primary)">
                                      {p.return_pct != null ? `${p.return_pct.toFixed(2)}%` : "-"}
                                    </td>
                                    <td className="px-2 py-1 text-right font-mono text-[10px] whitespace-nowrap text-(--color-text-primary)">
                                      {p.trades ?? "-"}
                                    </td>
                                    <td className="px-2 py-1 text-right font-mono text-[10px] whitespace-nowrap text-(--color-text-primary)">
                                      {p.drawdown?.toFixed(2) ?? "-"}
                                    </td>
                                    <td className="px-2 py-1 text-right font-mono text-[10px] whitespace-nowrap text-(--color-text-primary)">
                                      {p.win_rate != null
                                        ? `${(p.win_rate * 100).toFixed(1)}%`
                                        : "-"}
                                    </td>
                                    <td
                                      className="px-2 py-1 text-center font-mono text-[10px] whitespace-nowrap text-(--color-text-primary)"
                                      title={`Signals: ${p.signals_passed_gate ?? 0} passed / ${p.signals_raw ?? 0} raw`}
                                    >
                                      {(p.signals_raw ?? 0) > 0
                                        ? `${Math.round(((p.signals_passed_gate ?? 0) / (p.signals_raw ?? 1)) * 100)}%`
                                        : "-"}
                                    </td>
                                    <td className="px-2 py-1 text-center whitespace-nowrap">
                                      {p.sharpe_gap_pct != null ? (
                                        <span
                                          className="inline-block h-2 w-2 rounded-full"
                                          style={{
                                            backgroundColor:
                                              p.sharpe_gap_pct > 40
                                                ? "var(--color-accent-danger)"
                                                : p.sharpe_gap_pct > 15
                                                  ? "var(--color-accent-warning)"
                                                  : "var(--color-accent-success)",
                                          }}
                                          title={`Train/OOS gap: ${p.sharpe_gap_pct.toFixed(0)}%`}
                                        />
                                      ) : (
                                        <span className="text-(--color-text-muted)">-</span>
                                      )}
                                    </td>
                                  </tr>,
                                );
                              }
                            }
                            return rows.length > 0 ? (
                              rows
                            ) : (
                              <tr>
                                <td
                                  colSpan={9}
                                  className="py-4 text-center text-[10px] text-(--color-text-muted)"
                                >
                                  No data for selected month
                                </td>
                              </tr>
                            );
                          })()
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="flex flex-1 items-center justify-center">
          <span className="text-xs text-(--color-text-muted)">Select a job to monitor</span>
        </div>
      )}

      {selectedJob?.error && (
        <div className="flex shrink-0 items-center gap-2 border-t border-(--color-accent-danger) bg-[rgba(242,54,69,0.05)] px-6 py-1.5">
          <span className="font-mono text-[10px] text-(--color-accent-danger)">
            {selectedJob.error.length > 200
              ? selectedJob.error.slice(0, 200) + "\u2026"
              : selectedJob.error}
          </span>
        </div>
      )}
    </div>
  );
}
