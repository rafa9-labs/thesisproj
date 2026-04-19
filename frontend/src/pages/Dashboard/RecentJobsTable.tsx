import { useNavigate } from "react-router-dom";
import { Eye, Trash2 } from "lucide-react";
import { StatusDot } from "@/components/shared/StatusDot";
import { formatRelativeTime } from "@/lib/formatters";
import { useDeleteJob } from "@/api/queries";
import type { JobSummary } from "@/api/schemas";

const STATUS_COLORS: Record<string, string> = {
  completed: "var(--color-accent-success)",
  running: "var(--color-accent)",
  pending: "var(--color-accent-warning)",
  failed: "var(--color-accent-danger)",
};

interface RecentJobsTableProps {
  jobs: JobSummary[];
}

export function RecentJobsTable({ jobs }: RecentJobsTableProps) {
  const navigate = useNavigate();
  const deleteJob = useDeleteJob();

  if (jobs.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      <h3
        className="text-xs font-semibold uppercase tracking-[0.08em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Recent Jobs
      </h3>
      <div
        className="rounded-lg border overflow-hidden"
        style={{ borderColor: "var(--color-border)" }}
      >
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr
              style={{
                backgroundColor: "var(--color-elevated)",
                color: "var(--color-text-secondary)",
              }}
            >
              <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Job</th>
              <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Pair</th>
              <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Models</th>
              <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Status</th>
              <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Created</th>
              <th className="px-3 py-2 text-right font-semibold uppercase tracking-wide">Actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job, i) => {
              const color = STATUS_COLORS[job.status] ?? "var(--color-text-muted)";
              return (
                <tr
                  key={job.job_id}
                  className="group"
                  style={{
                    backgroundColor: i % 2 === 0 ? "var(--color-surface)" : "var(--color-app)",
                    borderBottom: "1px solid var(--color-border)",
                  }}
                >
                  <td
                    className="px-3 py-2"
                    style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                  >
                    {job.job_id.slice(0, 8)}…
                  </td>
                  <td
                    className="px-3 py-2"
                    style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                  >
                    {job.pair ?? "—"}
                  </td>
                  <td
                    className="px-3 py-2 max-w-[200px] truncate"
                    style={{ color: "var(--color-text-secondary)" }}
                  >
                    {job.models?.join(", ") ?? "—"}
                  </td>
                  <td className="px-3 py-2">
                    <StatusDot color={color} label={job.status} />
                  </td>
                  <td
                    className="px-3 py-2"
                    style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
                  >
                    {formatRelativeTime(job.created_at)}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {job.status === "completed" && (
                        <button
                          onClick={() => navigate(`/results/${job.job_id}`)}
                          className="rounded p-1 transition-colors"
                          style={{
                            color: "var(--color-text-muted)",
                            cursor: "pointer",
                          }}
                          title="View results"
                        >
                          <Eye size={14} />
                        </button>
                      )}
                      <button
                        onClick={() => deleteJob.mutate(job.job_id)}
                        className="rounded p-1 transition-colors"
                        style={{
                          color: "var(--color-text-muted)",
                          cursor: "pointer",
                        }}
                        title="Delete job"
                      >
                        <Trash2 size={14} />
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
