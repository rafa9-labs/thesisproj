import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, FlaskConical, ChevronRight, Square, Wifi, WifiOff } from "lucide-react";
import { useActiveBacktests, useForceStopJob, useJobStatus } from "@/api/queries";
import { useJobStore } from "@/stores/useJobStore";
import { useBacktestWebSocket } from "@/hooks/useBacktestWebSocket";
import { useBacktestProgress } from "@/api/queries";
import { JobPillStrip } from "./JobPillStrip";
import { CycleCard } from "./CycleCard";
import { EquityChart } from "./EquityChart";
import { DebugOverlay } from "./DebugOverlay";
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
  const removeJob = useJobStore((s) => s.removeJob);
  const forceStop = useForceStopJob();

  const [wsConnected, setWsConnected] = useState(false);

  const ensureJob = useJobStore((s) => s.ensureJob);

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

  const prevRunningIds = useRef<{ ids: Set<string>; initialized: boolean }>({ ids: new Set(), initialized: false });

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
      const selectedStillVisible = selectedJobId && visibleIds.has(selectedJobId) && activeJobs.has(selectedJobId);
      if (!selectedStillVisible) {
        const firstRunning = [...visibleIds].find((id) => runningIds.has(id) && activeJobs.has(id));
        const first = firstRunning ?? ([...visibleIds].find((id) => activeJobs.has(id)) ?? null);
        if (first) {
          selectJob(first);
          setActiveTab(first, "hpo-and-results");
        }
      }
    }
  }, [visibleIds, selectedJobId, selectJob, setActiveTab, activeJobs, runningIds]);

  const selectedJobStatus = selectedJobId ? getJob(selectedJobId)?.status : null;
  const wsJobId = selectedJobId && (selectedJobStatus === "pending" || selectedJobStatus === "running") ? selectedJobId : null;
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
      handleWsEvent({
        event: "job_complete",
        job_id: selectedJobId,
        metrics: [],
      });
      markCompleted(selectedJobId);
    }
  }, [restStatus, selectedJobId, getJob, handleWsEvent, markCompleted]);

  const handleForceStop = async () => {
    if (!selectedJobId) return;
    const jobId = selectedJobId;
    try {
      await forceStop.mutateAsync(jobId);
      handleWsEvent({
        event: "job_failed",
        job_id: jobId,
        error: "Stopped by user",
      });
      removeJob(jobId);
    } catch (err) {
      console.error("Force stop failed:", err);
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

  if (isLoading) {
    return (
      <div
        className="flex h-full items-center justify-center"
        style={{ backgroundColor: "var(--color-app)" }}
      >
        <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          Checking for backtests...
        </span>
      </div>
    );
  }

  if (runningList.length === 0 && completedJobIds.size === 0) {
    return (
      <div className="flex flex-col gap-6">
        <div
          className="flex flex-col items-center justify-center gap-4 rounded-xl border py-16"
          style={{
            borderColor: "var(--color-glass-border)",
            backgroundColor: "var(--color-surface)",
          }}
        >
          <Eye size={40} strokeWidth={1} style={{ color: "var(--color-text-muted)" }} />
          <div className="flex flex-col items-center gap-1">
            <span
              className="text-sm font-semibold uppercase tracking-[0.08em]"
              style={{ color: "var(--color-text-secondary)" }}
            >
              No Active Backtests
            </span>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              Configure and deploy one from the Backtest tab
            </span>
          </div>
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1.5 rounded-md px-4 py-2 text-[11px] font-medium uppercase tracking-[0.06em] transition-all hover:brightness-110"
            style={{
              backgroundColor: "var(--color-brand)",
              color: "var(--color-text-inverse)",
            }}
          >
            <FlaskConical size={14} strokeWidth={2} />
            Go to Backtest
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <span
          className="text-xs font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Monitor{completedJobIds.size > 0 ? ` — ${completedJobIds.size} completed` : ""}
        </span>
        {selectedJobId && (
          <span
            className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium"
            style={{
              backgroundColor: wsConnected ? "rgba(34,197,94,0.08)" : "rgba(242,54,69,0.08)",
              color: wsConnected ? "var(--color-accent-success)" : "var(--color-accent-danger)",
            }}
          >
            {wsConnected ? <Wifi size={10} /> : <WifiOff size={10} />}
            {wsConnected ? "Live" : "Disconnected"}
          </span>
        )}
        <div className="flex-1" />
      </div>

      <JobPillStrip
        jobs={allJobs}
        selectedJobId={selectedJobId}
        onSelect={(id) => {
          selectJob(id);
          setActiveTab(id, "hpo-and-results");
        }}
      />

      {selectedJob && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="flex flex-col gap-3" style={{ maxHeight: "calc(100vh - 240px)", overflowY: "auto" }}>
              {selectedJob.cycles.length === 0 && (
                <div
                  className="flex items-center justify-center rounded-lg border py-12"
                  style={{ borderColor: "var(--color-border)", color: "var(--color-text-muted)" }}
                >
                  <span className="text-xs">Waiting for cycles to start...</span>
                </div>
              )}
              {selectedJob.cycles.map((c) => (
                <CycleCard key={`${c.model}-${c.cycleNumber}`} cycle={c} />
              ))}
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
            >
              <EquityChart
                models={models}
                oosPeriods={allOosPeriods}
                oosEquity={allOosEquity}
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <div
                className="h-2 flex-1 overflow-hidden rounded-full"
                style={{ backgroundColor: "var(--color-elevated)" }}
              >
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min(progress, 100)}%`,
                    backgroundColor: isDone
                      ? selectedJob.status === "failed"
                        ? "var(--color-accent-danger)"
                        : "var(--color-accent-success)"
                      : "var(--color-brand)",
                  }}
                />
              </div>
              <span
                className="text-xs min-w-[40px] text-right"
                style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
              >
                {Math.round(progress)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-xs" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                {progressText}
              </p>
              {!isDone && selectedJobId && (
                <button
                  onClick={handleForceStop}
                  disabled={forceStop.isPending}
                  className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.06em] transition-all hover:brightness-110"
                  style={{
                    borderColor: "var(--color-accent-danger)",
                    backgroundColor: "rgba(242,54,69,0.08)",
                    color: "var(--color-accent-danger)",
                    cursor: forceStop.isPending ? "not-allowed" : "pointer",
                    opacity: forceStop.isPending ? 0.6 : 1,
                  }}
                >
                  <Square size={12} strokeWidth={2} fill="currentColor" />
                  {forceStop.isPending ? "Stopping…" : "Force Stop"}
                </button>
              )}
              {isDone && selectedJob.status === "completed" && (
                <button
                  onClick={() => navigate(`/results/${selectedJobId}`)}
                  className="flex items-center gap-1 rounded-md px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.06em] transition-all hover:brightness-110"
                  style={{
                    backgroundColor: "var(--color-brand)",
                    color: "var(--color-text-inverse)",
                    boxShadow: "0 0 16px rgba(0,229,255,0.2)",
                  }}
                >
                  View Full Results
                  <ChevronRight size={14} strokeWidth={2} />
                </button>
              )}
            </div>
            {selectedJob.error && (
              <div
                className="rounded-md border px-3 py-1.5"
                style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(242,54,69,0.05)" }}
              >
                <span className="text-xs" style={{ color: "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
                  {selectedJob.error.length > 200 ? selectedJob.error.slice(0, 200) + "…" : selectedJob.error}
                </span>
              </div>
            )}
          </div>
        </>
      )}
      <DebugOverlay jobId={selectedJobId} pollCursor={0} />
    </div>
  );
}
