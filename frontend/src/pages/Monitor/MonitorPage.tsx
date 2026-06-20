import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Square, Wifi, WifiOff } from "lucide-react";
import { useActiveBacktests, useForceStopJob, useJobStatus, useFullCycleStatus, useFullCycleHistory } from "@/api/queries";
import { useJobStore } from "@/stores/useJobStore";
import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import { useBacktestProgress } from "@/api/queries";
import { JobPillStrip } from "./JobPillStrip";
import { EquityChart } from "./EquityChart";
import { HpoScatterChart } from "./HpoScatterChart";
import { HpoTrialFeed } from "./HpoTrialFeed";
import { ModelHealthTable } from "./ModelHealthTable";
import { MonthlyHeatmap } from "./MonthlyHeatmap";
import { wsManager } from "@/api/websocket";
import type { JobSummary } from "@/api/schemas";
import { CommitteeJobHeader } from "./committee/CommitteeJobHeader";
import { PipelineNavigator } from "./committee/PipelineNavigator";
import { CommitteeLogConsole } from "./committee/CommitteeLogConsole";
import { FeatureSweepView } from "./committee/FeatureSweepView";
import { HpoTuningView } from "./committee/HpoTuningView";
import { AssemblyView } from "./committee/AssemblyView";
import { ValidationView } from "./committee/ValidationView";
import { FactoryView } from "./committee/FactoryView";
import { FinalSnapshotView } from "./committee/FinalSnapshotView";

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
  const removeJob = useJobStore((s) => s.removeJob);
  const forceStop = useForceStopJob();
  const ensureJob = useJobStore((s) => s.ensureJob);

  const [wsConnected, setWsConnected] = useState(false);
  const [hpoModelFilter, setHpoModelFilter] = useState<string | null>(null);
  const [yMode, setYMode] = useState<"pct" | "raw">("pct");

  const activeList = useMemo(() => activeData?.jobs ?? [], [activeData?.jobs]);
  const runningList = useMemo(
    () => activeList.filter((j) => j.status === "pending" || j.status === "running"),
    [activeList],
  );

  useEffect(() => {
    const jobs = activeJobs instanceof Map ? activeJobs : new Map();
    for (const job of runningList) {
      if (!jobs.has(job.job_id)) {
        ensureJob(job.job_id, job.pair ?? "", job.models ?? []);
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

  // Rehydrate sync: reconcile persisted jobs with server state.
  // On server restart the job store (localStorage) outlives the backend.
  // Any persisted job that is NOT in the active list is stale and should be
  // removed entirely to avoid phantom 404 polling.
  const rehydrateChecked = useRef(false);
  useEffect(() => {
    if (rehydrateChecked.current) return;
    if (!activeData || !activeData.jobs) return;
    rehydrateChecked.current = true;

    const serverIds = new Set(activeData.jobs.map((j) => j.job_id));
    if (activeJobs instanceof Map) {
      for (const [id] of activeJobs) {
        if (!serverIds.has(id)) {
          removeJob(id);
        }
      }
    }
  }, [activeData, activeJobs, removeJob]);

  const completedSummaries: JobSummary[] = useMemo(() => {
    if (completedJobIds.size === 0) return [];
    if (!(activeJobs instanceof Map)) return [];
    if (!(completedJobIds instanceof Set)) return [];
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
    () => new Set([...runningIds, ...(completedJobIds instanceof Set ? completedJobIds : [])]),
    [runningIds, completedJobIds],
  );

  useEffect(() => {
    if (!(activeJobs instanceof Map)) return;
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

  useEffect(() => {
    if (!selectedJobId) return;
    const check = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 1000);
    return () => clearInterval(check);
  }, [selectedJobId]);

  const shouldPoll = selectedJobId != null;
  useBacktestProgress(shouldPoll ? selectedJobId : null);
  const { data: restStatus, error: restError } = useJobStatus(shouldPoll ? selectedJobId : null);

  // Clean up persisted jobs that return 404 (server restart / old data)
  useEffect(() => {
    if (!restError || !selectedJobId) return;
    const status = (restError as { response?: { status?: number } })?.response?.status;
    if (status === 404) {
      removeJob(selectedJobId);
    }
  }, [restError, selectedJobId, removeJob]);

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
  const models: string[] = Array.isArray(selectedJob?.models) ? selectedJob.models : [];
  const status = selectedJob?.status;
  const isDone = status === "completed" || status === "failed";

  const modelPhases = selectedJob?.modelPhases;
  const jobProgressText = selectedJob?.progressText;

  /* eslint-disable react-hooks/preserve-manual-memoization */
  const { progress, progressText, phase } = useMemo(() => {
    const phases = modelPhases ? [...modelPhases.values()] : [];
    if (phases.length === 0) return { progress: 0, progressText: "Initializing...", phase: "idle" as const };

    let hpoTotal = 0, hpoDone = 0, wfoTotal = 0, wfoDone = 0;
    for (const p of phases) {
      hpoTotal += p.hpoTotalTrials;
      hpoDone += Math.min(p.hpoTrial, p.hpoTotalTrials);
      wfoTotal += p.simTotalMonths;
      wfoDone += Math.min(p.simMonth, p.simTotalMonths);
    }

    const isWfo = phases.some((p) => p.phase === "simulation");

    if (hpoTotal + wfoTotal === 0) {
      return { progress: 0, progressText: "Initializing...", phase: "hpo" as const };
    }

    if (!isWfo) {
      const pct = hpoTotal > 0 ? (hpoDone / hpoTotal) * 50 : 0;
      return {
        progress: Math.min(pct, 50),
        progressText: hpoTotal > 0 ? `HPO ${hpoDone}/${hpoTotal}` : "HPO...",
        phase: "hpo" as const,
      };
    }

    const wfoPct = wfoTotal > 0 ? (wfoDone / wfoTotal) * 50 : 0;
    const parts: string[] = [];
    if (hpoTotal > 0) parts.push(`HPO ${hpoDone}/${hpoTotal}`);
    if (wfoTotal > 0) parts.push(`WF ${wfoDone}/${wfoTotal}`);

    return {
      progress: 50 + Math.min(wfoPct, 50),
      progressText: parts.length > 0 ? parts.join(" · ") : (jobProgressText ?? ""),
      phase: "wfo" as const,
    };
  }, [modelPhases, jobProgressText]);
  /* eslint-enable react-hooks/preserve-manual-memoization */

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

  const [expandedPeriod, setExpandedPeriod] = useState<string | null>(null);
  const periodRows = allOosPeriods
    .filter((p) => (p.trades ?? 0) > 0)
    .slice(-50)
    .reverse();

  // ── Committee tab state ──
  const [searchParams] = useSearchParams();
  const monitorTab: "backtests" | "committee" =
    searchParams.get("tab") === "committee" ? "committee" : "backtests";
  const switchTab = (tab: "backtests" | "committee") => {
    navigate(tab === "committee" ? "/monitor?tab=committee" : "/monitor", { replace: true });
  };
  const cmJobId = useCommitteeMonitorStore((s) => s.selectedJobId);
  const cmUpdateFromStatus = useCommitteeMonitorStore((s) => s.updateFromStatus);
  const cmViewPhase = useCommitteeMonitorStore((s) => s.viewPhase);
  const cmSelectJob = useCommitteeMonitorStore((s) => s.selectJob);
  const { data: cmStatus } = useFullCycleStatus(monitorTab === "committee" ? cmJobId : null);
  const { data: fcHistory } = useFullCycleHistory();
  useEffect(() => {
    if (cmStatus) cmUpdateFromStatus(cmStatus);
  }, [cmStatus, cmUpdateFromStatus]);

  // Auto-select first running committee job when entering committee tab
  useEffect(() => {
    if (monitorTab !== "committee" || cmJobId) return;
    const running = (fcHistory?.entries ?? [])
      .filter((e) =>
        e.status !== "completed" &&
        e.status !== "failed" &&
        e.status !== "validation_failed" &&
        e.status !== "cancelled" &&
        e.status !== "orphaned" &&
        e.status !== "unknown",
      )
      .sort((a, b) => b.started_at.localeCompare(a.started_at));
    if (running.length > 0) {
      cmSelectJob(running[0].job_id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [monitorTab]);

  // Deselect orphaned/cancelled/unknown committee jobs — prevents broken dead-study UI
  const cmReset = useCommitteeMonitorStore((s) => s.reset);
  useEffect(() => {
    if (!cmJobId || monitorTab !== "committee") return;
    const entry = (fcHistory?.entries ?? []).find((e) => e.job_id === cmJobId);
    if (
      entry &&
      (entry.status === "orphaned" ||
        entry.status === "cancelled" ||
        entry.status === "unknown")
    ) {
      cmReset();
    }
  }, [cmJobId, fcHistory, monitorTab, cmReset]);

  // Auto-redirect to committee tab when only committee jobs are running
  useEffect(() => {
    if (monitorTab !== "backtests" || isLoading || runningList.length > 0) return;
    const hasRunningCommittee = (fcHistory?.entries ?? []).some(
      (e) =>
        e.status !== "completed" &&
        e.status !== "failed" &&
        e.status !== "validation_failed" &&
        e.status !== "cancelled" &&
        e.status !== "orphaned" &&
        e.status !== "unknown",
    );
    if (hasRunningCommittee) {
      navigate("/monitor?tab=committee", { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [monitorTab, isLoading, runningList.length, fcHistory]);

  // ── Committee tab ──
  if (monitorTab === "committee") {
    const cmEntry = (fcHistory?.entries ?? []).find((e) => e.job_id === cmJobId);
    const isCmTerminal =
      !cmEntry ||
      cmEntry.status === "orphaned" ||
      cmEntry.status === "cancelled" ||
      cmEntry.status === "completed" ||
      cmEntry.status === "failed" ||
      cmEntry.status === "validation_failed" ||
      cmEntry.status === "unknown";

    return (
      <div className="flex h-full flex-col overflow-hidden">
        <div className="flex shrink-0 items-center gap-1 pt-3 pb-2">
          <TabButton label="Backtests" active={false} onClick={() => switchTab("backtests")} />
          <TabButton label="Committee" active={true} onClick={() => {}} />
        </div>
        {cmJobId && !isCmTerminal ? (
          <div className="flex flex-1 flex-col overflow-hidden">
            <CommitteeJobHeader />
            <PipelineNavigator />
            <div className="flex-1 overflow-y-auto pb-4 [scrollbar-width:thin]">
              {cmViewPhase === 1 && <FeatureSweepView />}
              {cmViewPhase === 2 && <HpoTuningView />}
              {cmViewPhase === 3 && <AssemblyView />}
              {cmViewPhase === 4 && <ValidationView />}
              {cmViewPhase === 5 && <FactoryView />}
              {cmViewPhase === 6 && <FinalSnapshotView />}
            </div>
            <div className="shrink-0 border-t border-(--color-glass-border)">
              <CommitteeLogConsole />
            </div>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <span className="text-xs text-(--color-text-muted)">
              No active committee jobs. Click a committee pill in the running bar above.
            </span>
          </div>
        )}
      </div>
    );
  }

  if (monitorTab === "backtests" && isLoading) {
    return (
      <div className="flex h-full items-center justify-center bg-(--color-app)">
        <span className="text-xs text-(--color-text-muted)">Checking for backtests...</span>
      </div>
    );
  }

  if (runningList.length === 0 && completedJobIds instanceof Set && completedJobIds.size === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded-sm border border-(--color-glass-border) bg-(--color-surface) py-16">
        <span className="text-sm font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
          No Active Studies
        </span>
        <span className="text-xs text-(--color-text-muted)">
          Configure and deploy a backtest or committee pipeline
        </span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1.5 rounded-md bg-(--color-brand) px-4 py-2 text-[11px] font-medium tracking-[0.06em] text-(--color-text-inverse) uppercase transition-all hover:brightness-110"
          >
            Go to Backtest
          </button>
          <button
            onClick={() => navigate("/committee")}
            className="flex items-center gap-1.5 rounded-md border border-(--color-brand) bg-(--color-brand-glow) px-4 py-2 text-[11px] font-medium tracking-[0.06em] text-(--color-brand) uppercase transition-all hover:brightness-110"
          >
            Go to Committee
          </button>
        </div>
      </div>
    );
  }

  if (runningList.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded-sm border border-(--color-glass-border) bg-(--color-surface) py-16">
        <span className="text-sm font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
          All Studies Complete
        </span>
        <span className="text-xs text-(--color-text-muted)">
          View results or start a new study
        </span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1.5 rounded-md bg-(--color-brand) px-4 py-2 text-[11px] font-medium tracking-[0.06em] text-(--color-text-inverse) uppercase transition-all hover:brightness-110"
          >
            New Backtest
          </button>
          <button
            onClick={() => navigate("/committee")}
            className="flex items-center gap-1.5 rounded-md border border-(--color-brand) bg-(--color-brand-glow) px-4 py-2 text-[11px] font-medium tracking-[0.06em] text-(--color-brand) uppercase transition-all hover:brightness-110"
          >
            New Committee
          </button>
          <button
            onClick={() => navigate("/results")}
            className="flex items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-(--color-glass) px-4 py-2 text-[11px] font-medium tracking-[0.06em] text-(--color-text-secondary) uppercase transition-all hover:brightness-110"
          >
            View Results
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* ── Tab bar ── */}
      <div className="flex shrink-0 items-center gap-1 pt-3 pb-2">
        <TabButton label="Backtests" active={true} onClick={() => {}} />
        <TabButton label="Committee" active={false} onClick={() => switchTab("committee")} />
      </div>

      {/* ── Top Bar ── */}
      <div className="flex flex-row shrink-0 items-center gap-4 w-full rounded-lg border border-(--color-glass-border) bg-(--color-glass) px-4 py-2.5">
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
        {progressText && (
          <span className="shrink-0 font-mono text-[9px] text-(--color-text-dim) tabular-nums">
            {progressText}
          </span>
        )}

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

        {isDone && selectedJobId && (
          <button
            onClick={() => navigate(`/results/${selectedJobId}`)}
            className="flex shrink-0 items-center gap-1 rounded border border-(--color-accent-success) px-2 py-1 text-[10px] font-semibold tracking-[0.06em] text-(--color-accent-success) uppercase transition hover:brightness-110"
          >
            View Results
          </button>
        )}
      </div>

      {selectedJob ? (
        <>
          {/* ── Top Pane: HPO + Model Health (50%) ── */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <div className="grid h-full grid-cols-1 gap-4 p-4 lg:grid-cols-3">
              {/* HPO Convergence — 2/3 */}
              <div className="flex min-h-0 flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4 lg:col-span-2">
                <div className="mb-3 flex shrink-0 items-center gap-3">
                  <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                    HPO Convergence
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setHpoModelFilter(null)}
                      className="rounded px-2 py-0.5 text-[9px] font-medium uppercase transition"
                      style={{
                        backgroundColor: !hpoModelFilter ? "var(--color-brand-glow)" : "transparent",
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
                          backgroundColor: hpoModelFilter === m ? "var(--color-brand-glow)" : "transparent",
                          color: hpoModelFilter === m ? "var(--color-brand)" : "var(--color-text-muted)",
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto [scrollbar-width:thin]">
                  <div className="h-[260px]">
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

              {/* Model Health — 1/3 */}
              <div className="flex min-h-0 flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
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
                      const flatCount = allOosPeriods.filter((p) => (p.return_pct ?? 0) === 0).length;
                      const shortCount = allOosPeriods.filter((p) => (p.return_pct ?? 0) < 0).length;
                      const longPct = total > 0 ? (longCount / total) * 100 : 0;
                      const flatPct = total > 0 ? (flatCount / total) * 100 : 0;
                      const shortPct = total > 0 ? (shortCount / total) * 100 : 0;
                      const totalRawSignals = allOosPeriods.reduce((s, p) => s + (p.signals_raw ?? 0), 0);
                      const totalPassedSignals = allOosPeriods.reduce((s, p) => s + (p.signals_passed_gate ?? 0), 0);
                      const gateRate = totalRawSignals > 0 ? (totalPassedSignals / totalRawSignals) * 100 : 0;

                      return (
                        <>
                          <div className="mb-1.5 flex h-5 overflow-hidden rounded-full bg-(--color-glass-hover)">
                            {longPct > 0 && (
                              <div className="flex items-center justify-center text-[8px] font-bold text-white/80 transition-all" style={{ width: `${longPct}%`, backgroundColor: "var(--color-accent-success)" }}>
                                {longPct >= 12 ? `${Math.round(longPct)}%` : ""}
                              </div>
                            )}
                            {flatPct > 0 && (
                              <div className="flex items-center justify-center text-[8px] font-bold text-white/60 transition-all" style={{ width: `${flatPct}%`, backgroundColor: "var(--color-text-dim)" }}>
                                {flatPct >= 12 ? `${Math.round(flatPct)}%` : ""}
                              </div>
                            )}
                            {shortPct > 0 && (
                              <div className="flex items-center justify-center text-[8px] font-bold text-white/80 transition-all" style={{ width: `${shortPct}%`, backgroundColor: "var(--color-accent-danger)" }}>
                                {shortPct >= 12 ? `${Math.round(shortPct)}%` : ""}
                              </div>
                            )}
                          </div>
                          <div className="flex items-center justify-between font-mono text-[9px]">
                            <span className="text-(--color-accent-success)">POS {Math.round(longPct)}%</span>
                            <span className="text-(--color-text-dim)">NEUT {Math.round(flatPct)}%</span>
                            <span className="text-(--color-accent-danger)">NEG {Math.round(shortPct)}%</span>
                          </div>
                          {totalRawSignals > 0 && (
                            <div className="mt-2 font-mono text-[8px] text-(--color-text-dim)">
                              Gate: {totalPassedSignals}/{totalRawSignals} signals ({Math.round(gateRate)}%)
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

          {/* ── Static Divider ── */}
          <div className="shrink-0 border-t border-(--color-glass-border)" />

          {/* ── Bottom Pane: Equity + Period Results (50%) ── */}
          <div className="flex-1 min-h-0 p-4">
            <div className="flex h-full gap-4 overflow-hidden">
              {/* Left 50%: Walk-Forward Equity */}
              <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4">
                <div className="mb-3 flex shrink-0 items-center gap-3">
                  <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                    Walk-Forward Equity
                  </span>
                  <button
                    onClick={() => setYMode((y) => (y === "pct" ? "raw" : "pct"))}
                    className="rounded border border-(--color-border) px-2 py-0.5 text-[10px] transition-colors"
                    style={{
                      color: yMode === "pct" ? "var(--color-brand)" : "var(--color-text-muted)",
                      backgroundColor: yMode === "pct" ? "rgba(59,130,246,0.08)" : "transparent",
                    }}
                  >
                    {yMode === "pct" ? "%" : "$"}
                  </button>
                </div>
                <div className="flex-1 min-h-0">
                  {phase === "hpo" ? (
                    <div className="flex h-full items-center justify-center">
                      <span className="text-sm text-slate-500">Awaiting HPO convergence to calculate walk-forward equity...</span>
                    </div>
                  ) : selectedJob.cycles.length > 0 ? (
                    <EquityChart models={models} oosPeriods={allOosPeriods} oosEquity={allOosEquity} yMode={yMode} />
                  ) : (
                    <div className="flex h-full items-center justify-center">
                      <span className="text-xs text-(--color-text-muted)">Waiting for cycles...</span>
                    </div>
                  )}
                </div>
                {phase === "hpo" && allOosPeriods.length === 0 ? (
                  <div className="mt-2 flex h-6 shrink-0 items-center justify-center rounded border border-(--color-glass-border) bg-(--color-glass-hover)">
                    <span className="text-[9px] text-slate-500">Awaiting walk-forward periods...</span>
                  </div>
                ) : allOosPeriods.length > 0 ? (
                  <div className="mt-2 shrink-0">
                    <MonthlyHeatmap periods={allOosPeriods} />
                  </div>
                ) : null}
              </div>

              {/* Right 50%: Unified Period Results + Trades */}
              <div className="flex w-1/2 min-w-0 shrink-0 flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass)">
                <div className="flex shrink-0 items-center gap-2 border-b border-(--color-glass-border) px-3 py-2">
                  <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                    Period Results
                  </span>
                  <span className="font-mono text-[9px] text-(--color-text-dim)">
                    {periodRows.length} periods
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-2 border-b border-[rgba(42,46,57,0.3)] px-3 py-1 font-mono text-[8px] tracking-[0.05em] text-(--color-text-dim) uppercase">
                  <span className="w-10 shrink-0">Month</span>
                  <span className="w-8 shrink-0 text-right">Model</span>
                  <span className="w-14 shrink-0 text-right">Return</span>
                  <span className="w-14 shrink-0 text-right">Sharpe</span>
                  <span className="w-8 shrink-0 text-right">#</span>
                </div>
                <div className="flex-1 overflow-y-auto [scrollbar-width:thin]">
                  {periodRows.length === 0 ? (
                    <div className="flex h-full items-center justify-center">
                      <span className="text-[10px] text-(--color-text-muted)">No trade data yet</span>
                    </div>
                  ) : (
                    periodRows.map((p) => {
                      const key = `${p.period}-${p.model ?? ""}`;
                      const isExpanded = expandedPeriod === key;
                      const isPositive = (p.return_pct ?? 0) >= 0;
                      const dirColor = isPositive ? "var(--color-accent-success)" : "var(--color-accent-danger)";
                      const shortModel = (p.model ?? "").slice(0, 6);

                      return (
                        <div key={key} className="border-b border-[rgba(42,46,57,0.3)]">
                          <button
                            onClick={() => setExpandedPeriod(isExpanded ? null : key)}
                            className="flex w-full items-center gap-2 px-3 py-1.5 text-[10px] transition-colors hover:bg-(--color-glass-hover)"
                          >
                            <span className="w-10 shrink-0 whitespace-nowrap font-mono text-(--color-text-dim)">
                              M{p.period}
                            </span>
                            <span className="w-8 shrink-0 text-right font-mono text-(--color-brand) tabular-nums">
                              {shortModel}
                            </span>
                            <span className="w-14 shrink-0 text-right font-mono tabular-nums" style={{ color: dirColor }}>
                              {isPositive ? "+" : ""}{(p.return_pct ?? 0).toFixed(2)}%
                            </span>
                            <span className="w-14 shrink-0 text-right font-mono tabular-nums" style={{ color: dirColor }}>
                              {p.sharpe != null ? p.sharpe.toFixed(2) : "\u2014"}
                            </span>
                            <span className="w-8 shrink-0 text-right font-mono tabular-nums text-(--color-text-dim)">
                              {p.trades ?? 0}
                            </span>
                          </button>
                          {isExpanded && (
                            <div className="border-t border-[rgba(42,46,57,0.2)] bg-[rgba(0,229,255,0.02)] px-3 py-2">
                              <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[9px]">
                                <span className="text-(--color-text-dim)">Win Rate</span>
                                <span className="text-right text-(--color-text-secondary)">
                                  {p.win_rate != null ? `${(p.win_rate * 100).toFixed(1)}%` : "\u2014"}
                                </span>
                                <span className="text-(--color-text-dim)">Max DD</span>
                                <span className="text-right text-(--color-accent-danger)">
                                  {p.drawdown != null ? `${p.drawdown.toFixed(2)}%` : "\u2014"}
                                </span>
                                <span className="text-(--color-text-dim)">Gate Rate</span>
                                <span className="text-right text-(--color-text-secondary)">
                                  {(p.signals_raw ?? 0) > 0
                                    ? `${Math.round(((p.signals_passed_gate ?? 0) / (p.signals_raw ?? 1)) * 100)}% (${p.signals_passed_gate ?? 0}/${p.signals_raw ?? 0})`
                                    : "\u2014"}
                                </span>
                                <span className="text-(--color-text-dim)">Overfit Risk</span>
                                <span className="text-right" style={{
                                  color: (p.sharpe_gap_pct ?? 0) > 40
                                    ? "var(--color-accent-danger)"
                                    : (p.sharpe_gap_pct ?? 0) > 15
                                      ? "var(--color-accent-warning)"
                                      : "var(--color-accent-success)",
                                }}>
                                  {p.sharpe_gap_pct != null ? `${p.sharpe_gap_pct.toFixed(0)}% gap` : "\u2014"}
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}
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

function TabButton({
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
      className="rounded-md px-3 py-1.5 text-[11px] font-medium transition-colors"
      style={{
        backgroundColor: active ? "var(--color-brand)" : "transparent",
        color: active ? "white" : "var(--color-text-muted)",
        border: active ? "1px solid var(--color-brand)" : "1px solid var(--color-border)",
      }}
      onClick={onClick}
    >
      {label}
    </button>
  );
}
