import { useNavigate } from "react-router-dom";
import { Eye, Trash2, ArrowRight } from "lucide-react";
import { formatRelativeTime } from "@/lib/formatters";
import { useDeleteJob } from "@/api/queries";
import { EquityThumbnail } from "@/components/charts/EquityThumbnail";
import type { JobSummary, EquityPoint } from "@/api/schemas";

const STATUS_STYLES: Record<string, { dot: string; bg: string; text: string; label: string }> = {
  completed: {
    dot: "var(--color-accent-success)",
    bg: "rgba(34,197,94,0.10)",
    text: "var(--color-accent-success)",
    label: "Completed",
  },
  running: {
    dot: "var(--color-brand)",
    bg: "var(--color-brand-glow)",
    text: "var(--color-brand)",
    label: "Running",
  },
  pending: {
    dot: "var(--color-accent-warning)",
    bg: "rgba(245,158,11,0.10)",
    text: "var(--color-accent-warning)",
    label: "Pending",
  },
  failed: {
    dot: "var(--color-accent-danger)",
    bg: "rgba(239,68,68,0.10)",
    text: "var(--color-accent-danger)",
    label: "Failed",
  },
};

interface RecentJobsTableProps {
  jobs: JobSummary[];
  equityData?: Record<string, EquityPoint[] | null>;
}

export function RecentJobsTable({ jobs, equityData }: RecentJobsTableProps) {
  const navigate = useNavigate();
  const deleteJob = useDeleteJob();

  if (jobs.length === 0) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
          Recent Activity
        </h3>
        <button
          onClick={() => navigate("/results")}
          className="flex items-center gap-1 text-[11px] font-medium text-(--color-text-muted) transition-colors duration-200 hover:text-[var(--color-text-primary)]"
        >
          View all
          <ArrowRight size={12} strokeWidth={1.5} />
        </button>
      </div>

      <div className="overflow-hidden rounded-sm border border-(--color-glass-border) bg-(--color-glass) backdrop-blur-[12px]">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="border-b border-(--color-glass-border) bg-(--color-surface) text-(--color-text-muted)">
              <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                Job
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                Equity
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                Pair
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                Models
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                Status
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-medium tracking-[0.1em] uppercase">
                Created
              </th>
              <th className="px-3 py-2.5 text-right text-[10px] font-medium tracking-[0.1em] uppercase">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => {
              const status = STATUS_STYLES[job.status] ?? {
                dot: "var(--color-text-muted)",
                bg: "var(--color-glass-hover)",
                text: "var(--color-text-muted)",
                label: job.status,
              };
              return (
                <tr
                  key={job.job_id}
                  className="group cursor-pointer border-b border-(--color-glass-border) transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
                  onClick={() => {
                    if (job.status === "completed") {
                      navigate(`/results/${job.job_id}`);
                    }
                  }}
                >
                  <td className="px-3 py-2.5 font-mono text-(--color-text-primary)">
                    {job.job_id.slice(0, 8)}…
                  </td>
                  <td className="px-3 py-2.5">
                    {equityData?.[job.job_id] ? (
                      <EquityThumbnail data={equityData[job.job_id]} />
                    ) : (
                      <div className="h-[36px] w-[120px] rounded-[4px] bg-(--color-glass-hover)" />
                    )}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-(--color-text-primary)">
                    {job.pair ?? "—"}
                  </td>
                  <td className="max-w-[200px] truncate px-3 py-2.5 text-(--color-text-secondary)">
                    {job.models?.join(", ") ?? "—"}
                  </td>
                  <td className="px-3 py-2.5">
                    <div
                      className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-medium tracking-[0.08em] uppercase"
                      style={{
                        backgroundColor: status.bg,
                        color: status.text,
                      }}
                    >
                      <span
                        className="h-[5px] w-[5px] rounded-full"
                        style={{ backgroundColor: status.dot }}
                      />
                      {status.label}
                    </div>
                  </td>
                  <td className="px-3 py-2.5 font-mono text-(--color-text-muted)">
                    {formatRelativeTime(job.created_at)}
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {job.status === "completed" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/results/${job.job_id}`);
                          }}
                          className="rounded p-1.5 text-(--color-text-muted) transition-colors duration-200 hover:bg-[var(--color-primary-glow)]"
                          className="cursor-pointer"
                          title="View results"
                        >
                          <Eye size={14} strokeWidth={1.5} />
                        </button>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteJob.mutate(job.job_id);
                        }}
                        className="rounded p-1.5 text-(--color-text-muted) transition-colors duration-200 hover:bg-[var(--color-accent-danger)]"
                        className="cursor-pointer"
                        title="Delete job"
                      >
                        <Trash2 size={14} strokeWidth={1.5} />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
