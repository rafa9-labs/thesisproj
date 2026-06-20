import { useNavigate } from "react-router-dom";
import { ChevronRight, X } from "lucide-react";
import type { JobSummary } from "@/api/schemas";
import { useJobStore } from "@/stores/useJobStore";

interface Props {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "var(--color-text-muted)",
  running: "var(--color-brand)",
  completed: "var(--color-accent-success)",
  failed: "var(--color-accent-danger)",
};

export function JobPillStrip({ jobs, selectedJobId, onSelect }: Props) {
  const navigate = useNavigate();
  const jobStore = useJobStore((s) => s.activeJobs);
  const removeJob = useJobStore((s) => s.removeJob);

  if (jobs.length === 0) return null;

  return (
    <div
      className="flex items-center gap-2 overflow-x-auto"
      style={{ scrollbarWidth: "none" }}
    >
      {jobs.map((j) => {
        const isSelected = j.job_id === selectedJobId;
        const local = jobStore.get(j.job_id);
        const progress = local?.progress ?? 0;
        const isDone = j.status === "completed" || j.status === "failed";

        return (
          <button
            key={j.job_id}
            onClick={() => onSelect(j.job_id)}
            className="flex shrink-0 items-center gap-2 rounded-sm border px-3 py-2 transition-all"
            style={{
              borderColor: isSelected ? "var(--color-brand)" : "var(--color-glass-border)",
              backgroundColor: isSelected ? "rgba(0,229,255,0.08)" : "var(--color-glass-hover)",
            }}
          >
            <div
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: STATUS_COLORS[j.status] ?? "var(--color-text-muted)" }}
            />
            <span
              className="max-w-[120px] truncate font-mono text-[11px] font-medium tracking-[0.04em] uppercase"
              style={{
                color: isSelected ? "var(--color-text-primary)" : "var(--color-text-secondary)",
              }}
            >
              {(Array.isArray(j.models) ? j.models : []).slice(0, 2).join("+")}
              {((Array.isArray(j.models) ? j.models.length : 0) > 2 ? "..." : "")}
            </span>
            {!isDone && (
              <span className="min-w-[28px] text-right font-mono text-[10px] text-(--color-brand)">
                {Math.round(progress)}%
              </span>
            )}
            {isDone && (
              <>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeJob(j.job_id);
                  }}
                  title="Dismiss"
                  aria-label="Dismiss job"
                  className="flex items-center justify-center rounded p-0.5 text-(--color-text-dim) transition-colors hover:bg-[rgba(242,54,69,0.12)] hover:text-(--color-accent-danger)"
                >
                  <X size={12} strokeWidth={2} />
                </button>
                <span
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(`/results/${j.job_id}`);
                  }}
                  className="flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors hover:brightness-110"
                  style={{
                    backgroundColor:
                      j.status === "completed" ? "rgba(34,197,94,0.12)" : "rgba(242,54,69,0.12)",
                    color:
                      j.status === "completed"
                        ? "var(--color-accent-success)"
                        : "var(--color-accent-danger)",
                  }}
                >
                  Results
                  <ChevronRight size={10} strokeWidth={2} />
                </span>
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}
