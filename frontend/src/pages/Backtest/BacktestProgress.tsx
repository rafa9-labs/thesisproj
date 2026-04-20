import { useEffect } from "react";
import { useJobStore } from "@/stores/useJobStore";
import { useJobStatus } from "@/api/queries";

export function BacktestProgress({ jobId }: { jobId: string | null }) {
  const job = useJobStore((s) => (jobId ? s.activeJobs.get(jobId) : undefined));
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);

  const jobStatus = job?.status;
  const shouldPoll = jobId != null && (jobStatus === "pending" || jobStatus === "running");
  const { data: restStatus } = useJobStatus(shouldPoll ? jobId : null);

  useEffect(() => {
    if (!restStatus || !jobId) return;
    if (restStatus.status === "completed") {
      handleWsEvent({ event: "job_complete", job_id: jobId });
    } else if (restStatus.status === "failed") {
      handleWsEvent({ event: "job_failed", job_id: jobId, error: restStatus.error ?? "Unknown error" });
    }
  }, [restStatus?.status, jobId, handleWsEvent]);

  if (!job) return null;

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

      {/* Progress bar */}
      <div className="mb-3 flex items-center gap-3">
        <div
          className="h-2 flex-1 overflow-hidden rounded-full"
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
                    : "var(--color-accent)",
            }}
          />
        </div>
        <span className="text-xs" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
          {job.progress}%
        </span>
      </div>

      <p className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
        {job.progressText}
      </p>

      {/* Model status pills */}
      <div className="mt-3 flex flex-wrap gap-2">
        {(job.models ?? []).map((m) => {
          const done = (job.completedModels ?? []).includes(m);
          const current = job.currentModel === m;
          return (
            <div
              key={m}
              className="flex items-center gap-1.5 rounded-md border px-2 py-1"
              style={{
                borderColor: done
                  ? "var(--color-accent-success)"
                  : current
                    ? "var(--color-accent)"
                    : "var(--color-border)",
                backgroundColor: done ? "rgba(8,153,129,0.08)" : "transparent",
              }}
            >
              <div
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: done
                    ? "var(--color-accent-success)"
                    : current
                      ? "var(--color-accent-warning)"
                      : "var(--color-text-muted)",
                }}
              />
              <span className="text-[11px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                {m}
              </span>
            </div>
          );
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
