import { useEffect } from "react";
import { useJobStore } from "@/stores/useJobStore";
import { useJobStatus } from "@/api/queries";
import type { ModelProgress } from "@/stores/useJobStore";

function PhaseBar({
  label,
  current,
  total,
  color,
}: {
  label: string;
  current: number;
  total: number;
  color: string;
}) {
  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-8 text-[10px] uppercase" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
        {label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ backgroundColor: "var(--color-elevated)" }}>
        <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="w-8 text-right text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
        {current}/{total}
      </span>
    </div>
  );
}

function ModelPill({ model, mp, done, current }: { model: string; mp?: ModelProgress; done: boolean; current: boolean }) {
  return (
    <div
      className="flex flex-col gap-1 rounded-md border px-2.5 py-1.5"
      style={{
        borderColor: done ? "var(--color-accent-success)" : current ? "var(--color-brand)" : "var(--color-border)",
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
        <span className="text-[11px] font-medium" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
          {model}
        </span>
      </div>
      {mp && !done && (
        <div className="flex gap-2 pl-3">
          {mp.phase === "hpo" && (
            <span className="text-[10px]" style={{ color: "var(--color-accent)", fontFamily: "var(--font-mono)" }}>
              HPO {mp.hpoTrial}/{mp.hpoTotalTrials}
            </span>
          )}
          {mp.phase === "simulation" && (
            <span className="text-[10px]" style={{ color: "var(--color-accent-success)", fontFamily: "var(--font-mono)" }}>
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

  const jobStatus = job?.status;
  const shouldPoll = jobId != null && (jobStatus === "pending" || jobStatus === "running");
  const { data: restStatus } = useJobStatus(shouldPoll ? jobId : null);

  useEffect(() => {
    if (!restStatus || !jobId) return;
    if (restStatus.status === "completed") {
      handleWsEvent({ event: "job_complete", job_id: jobId, metrics: [] });
    } else if (restStatus.status === "failed") {
      handleWsEvent({ event: "job_failed", job_id: jobId, error: restStatus.error ?? "Unknown error" });
    }
  }, [restStatus?.status, restStatus?.error, jobId, handleWsEvent]);

  if (!job) return null;

  const currentModelPhase = job.currentModel ? job.modelPhases.get(job.currentModel) : undefined;

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <div className="mb-2 flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Running Backtest
        </h3>
        <span className="text-xs" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {job.pair}
        </span>
      </div>

      {/* Main progress bar */}
      <div className="mb-2 flex items-center gap-3">
        <div
          className="h-2.5 flex-1 overflow-hidden rounded-full"
          style={{ backgroundColor: "var(--color-elevated)" }}
        >
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${job.progress}%`,
              backgroundColor:
                job.status === "failed"
                  ? "var(--color-accent-danger)"
                  : job.status === "completed"
                    ? "var(--color-accent-success)"
                    : "var(--color-brand)",
            }}
          />
        </div>
        <span className="text-xs tabular-nums" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", minWidth: 36, textAlign: "right" }}>
          {Math.round(job.progress)}%
        </span>
      </div>

      {/* Two-segment phase bars */}
      {currentModelPhase && job.status === "running" && (
        <div className="mb-2 flex flex-col gap-1 pl-1">
          <PhaseBar
            label="HPO"
            current={currentModelPhase.hpoTrial}
            total={currentModelPhase.hpoTotalTrials}
            color="var(--color-accent-warning)"
          />
          <PhaseBar
            label="SIM"
            current={currentModelPhase.simMonth}
            total={currentModelPhase.simTotalMonths}
            color="var(--color-accent-success)"
          />
        </div>
      )}

      <p className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
        {job.progressText}
      </p>

      {/* Model status pills with sub-progress */}
      <div className="mt-3 flex flex-wrap gap-2">
        {(job.models ?? []).map((m) => {
          const done = (job.completedModels ?? []).includes(m);
          const current = job.currentModel === m;
          const mp = job.modelPhases.get(m);
          return <ModelPill key={m} model={m} mp={mp} done={done} current={current} />;
        })}
      </div>

      {job.error && (
        <div className="mt-3 rounded-md border p-2" style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(242,54,69,0.05)" }}>
          <p className="text-xs" style={{ color: "var(--color-accent-danger)" }}>{job.error}</p>
        </div>
      )}
    </div>
  );
}