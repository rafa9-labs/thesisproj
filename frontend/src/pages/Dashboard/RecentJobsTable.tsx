import { useNavigate } from "react-router-dom";
import { Eye, Trash2, ArrowRight } from "lucide-react";
import { formatRelativeTime } from "@/lib/formatters";
import { useDeleteJob } from "@/api/queries";
import { EquityThumbnail } from "@/components/charts/EquityThumbnail";
import type { JobSummary, EquityPoint } from "@/api/schemas";

const STATUS_STYLES: Record<
  string,
  { dot: string; bg: string; text: string; label: string }
> = {
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
        <h3
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          Recent Activity
        </h3>
        <button
          onClick={() => navigate("/results")}
          className="flex items-center gap-1 text-[11px] font-medium transition-colors duration-200 hover:text-[var(--color-text-primary)]"
          style={{ color: "var(--color-text-muted)" }}
        >
          View all
          <ArrowRight size={12} strokeWidth={1.5} />
        </button>
      </div>

      <div
        className="rounded-sm border overflow-hidden"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
          backdropFilter: "blur(12px)",
        }}
      >
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr
              style={{
                backgroundColor: "var(--color-surface)",
                color: "var(--color-text-muted)",
                borderBottom: "1px solid var(--color-glass-border)",
              }}
            >
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Job</th>
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Equity</th>
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Pair</th>
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Models</th>
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Status</th>
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Created</th>
              <th className="px-3 py-2.5 text-right font-medium uppercase tracking-[0.1em] text-[10px]">Actions</th>
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
                  className="group transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
                  style={{
                    borderBottom: "1px solid var(--color-glass-border)",
                    cursor: "pointer",
                  }}
                  onClick={() => {
                    if (job.status === "completed") {
                      navigate(`/results/${job.job_id}`);
                    }
                  }}
                >
                  <td
                    className="px-3 py-2.5"
                    style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                  >
                    {job.job_id.slice(0, 8)}…
                  </td>
                  <td className="px-3 py-2.5">
                    {equityData?.[job.job_id] ? (
                      <EquityThumbnail data={equityData[job.job_id]} />
                    ) : (
                      <div
                        style={{
                          width: 120,
                          height: 36,
                          borderRadius: 4,
                          backgroundColor: "var(--color-glass-hover)",
                        }}
                      />
                    )}
                  </td>
                  <td
                    className="px-3 py-2.5"
                    style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                  >
                    {job.pair ?? "—"}
                  </td>
                  <td
                    className="px-3 py-2.5 max-w-[200px] truncate"
                    style={{ color: "var(--color-text-secondary)" }}
                  >
                    {job.models?.join(", ") ?? "—"}
                  </td>
                  <td className="px-3 py-2.5">
                    <div
                      className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.08em]"
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
                  <td
                    className="px-3 py-2.5"
                    style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
                  >
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
                          className="rounded p-1.5 transition-colors duration-200 hover:bg-[var(--color-primary-glow)]"
                          style={{
                            color: "var(--color-text-muted)",
                            cursor: "pointer",
                          }}
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
                        className="rounded p-1.5 transition-colors duration-200 hover:bg-[var(--color-accent-danger)]"
                        style={{
                          color: "var(--color-text-muted)",
                          cursor: "pointer",
                        }}
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
