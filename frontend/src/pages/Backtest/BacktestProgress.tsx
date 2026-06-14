import { useEffect } from "react";
import { useJobStore } from "@/stores/useJobStore";
import { useJobStatus, useBacktestProgress } from "@/api/queries";
import { HpoMonitor } from "./HpoMonitor";
import { OosPerformanceChart } from "./OosPerformanceChart";
import { TradeLog } from "./TradeLog";

type Tab = "hpo-and-results" | "trade";

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

function ProgressBar({ pct, text }: { pct: number; text: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-3">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-(--color-elevated)">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${pct}%`,
              backgroundColor: pct >= 100 ? "var(--color-accent-success)" : "var(--color-brand)",
            }}
          />
        </div>
        <span className="min-w-[36px] text-right font-mono text-xs text-(--color-text-primary)">
          {Math.round(pct)}%
        </span>
      </div>
      <p className="text-xs text-(--color-text-secondary)">{text}</p>
    </div>
  );
}

function ModelPill({
  model,
  mp,
  done,
  current,
}: {
  model: string;
  mp?: {
    phase: string;
    hpoTrial: number;
    hpoTotalTrials: number;
    simMonth: number;
    simTotalMonths: number;
  };
  done: boolean;
  current: boolean;
}) {
  return (
    <div
      className="flex flex-col gap-1 rounded-md border px-2.5 py-1.5"
      style={{
        borderColor: done
          ? "var(--color-accent-success)"
          : current
            ? "var(--color-brand)"
            : "var(--color-border)",
        backgroundColor: done ? "rgba(34,197,94,0.08)" : "transparent",
        minWidth: 120,
      }}
    >
      <div className="flex items-center gap-1.5">
        <div
          className="h-1.5 w-1.5 rounded-full"
          style={{
            backgroundColor: done
              ? "var(--color-accent-success)"
              : current
                ? "var(--color-brand)"
                : "var(--color-text-muted)",
          }}
        />
        <span className="font-mono text-[11px] font-medium text-(--color-text-primary)">
          {model}
        </span>
      </div>
      {mp && !done && (
        <div className="flex gap-2 pl-3">
          {mp.phase === "hpo" && (
            <span className="font-mono text-[10px] text-(--color-accent)">
              HPO {mp.hpoTrial}/{mp.hpoTotalTrials}
            </span>
          )}
          {mp.phase === "simulation" && (
            <span className="font-mono text-[10px] text-(--color-accent-success)">
              Sim {mp.simMonth}/{mp.simTotalMonths}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function BacktestProgress({ jobId }: { jobId: string | null }) {
  const job = useJobStore((s) => (jobId ? s.activeJobs.get(jobId) : undefined));
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);
  const setActiveTab = useJobStore((s) => s.setActiveTab);

  const jobStatus = job?.status;
  const shouldPoll = jobId != null && (jobStatus === "pending" || jobStatus === "running");
  const { data: restStatus } = useJobStatus(shouldPoll ? jobId : null);

  useBacktestProgress(shouldPoll ? jobId : null);

  useEffect(() => {
    if (!restStatus || !jobId) return;
    if (restStatus.status === "completed") {
      handleWsEvent({ event: "job_complete", job_id: jobId, metrics: [] });
    } else if (restStatus.status === "failed") {
      handleWsEvent({
        event: "job_failed",
        job_id: jobId,
        error: restStatus.error ?? "Unknown error",
      });
    }
  }, [restStatus?.status, restStatus?.error, jobId, handleWsEvent]);

  if (!job) return null;

  const activeTab: Tab = job.activeTab || "hpo-and-results";

  return (
    <div className="rounded-sm border border-(--color-border) bg-(--color-surface) p-4">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
          Running Backtest
        </h3>
        <span className="font-mono text-xs text-(--color-text-muted)">{job.pair}</span>
      </div>

      {/* Progress bar */}
      <ProgressBar pct={job.progress} text={job.progressText} />

      {/* Model pills */}
      <div className="mt-3 flex flex-wrap gap-2">
        {(job.models ?? []).map((m) => {
          const done = (job.completedModels ?? []).includes(m);
          const current = job.currentModel === m;
          const mp = job.modelPhases.get(m) as
            | {
                phase: string;
                hpoTrial: number;
                hpoTotalTrials: number;
                simMonth: number;
                simTotalMonths: number;
              }
            | undefined;
          return <ModelPill key={m} model={m} mp={mp} done={done} current={current} />;
        })}
      </div>

      {/* Tab bar */}
      <div className="mt-4 flex gap-2">
        <TabButton
          label="HPO and Results"
          active={activeTab === "hpo-and-results"}
          onClick={() => setActiveTab(jobId!, "hpo-and-results")}
        />
        <TabButton
          label="Trade Log"
          active={activeTab === "trade"}
          onClick={() => setActiveTab(jobId!, "trade")}
        />
      </div>

      {/* Tab content */}
      <div className="mt-3">
        {activeTab === "hpo-and-results" && (
          <div className="flex flex-col gap-4">
            <HpoMonitor
              model={job.currentModel ?? ""}
              trials={job.hpoTrials}
              bestTrial={job.bestTrial}
              totalTrials={job.modelPhases.get(job.currentModel ?? "")?.hpoTotalTrials}
            />
            <OosPerformanceChart
              model={job.currentModel ?? ""}
              equity={job.oosEquity}
              totalPeriods={
                job.oosPeriods.length > 0
                  ? job.oosPeriods[job.oosPeriods.length - 1].total_periods
                  : 0
              }
              currentPeriod={job.oosPeriods.length}
              bestTrial={job.bestTrial}
              periods={job.oosPeriods}
            />
          </div>
        )}
        {activeTab === "trade" && <TradeLog />}
      </div>

      {/* Error */}
      {job.error && (
        <div className="mt-3 rounded-md border border-(--color-accent-danger) bg-red-500/[0.05] p-2">
          <p className="text-xs text-(--color-accent-danger)">{job.error}</p>
        </div>
      )}
    </div>
  );
}
