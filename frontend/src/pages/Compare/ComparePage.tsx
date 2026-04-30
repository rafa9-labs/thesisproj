import { useState, useMemo } from "react";
import { GitCompare } from "lucide-react";
import { EmptyState } from "@/components/shared/EmptyState";
import { useJobHistory, useJobResults } from "@/api/queries";
import { LeaderboardTable } from "./LeaderboardTable";
import { EquityOverlayChart } from "./EquityOverlayChart";
import { SignificanceMatrix } from "./SignificanceMatrix";
import { ParameterSensitivityChart } from "@/components/charts/ParameterSensitivityChart";
import { CrossPairSection } from "./CrossPairSection";

function CompareSkeleton() {
  return (
    <div className="flex flex-col gap-6 animate-pulse">
      <div className="h-10 w-64 rounded" style={{ backgroundColor: "var(--color-elevated)" }} />
      <div className="h-[280px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
      <div className="h-[400px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
      <div className="h-[300px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
    </div>
  );
}

export function ComparePage() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const { data: jobs, isLoading: jobsLoading } = useJobHistory(50);

  const multiModelJobs = useMemo(
    () =>
      (jobs ?? [])
        .filter((j) => j.status === "completed" && (j.models?.length ?? 0) > 1)
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [jobs],
  );

  const { data: results, isLoading: resultsLoading } = useJobResults(selectedJobId);

  const metrics = results?.metrics ?? [];
  const modelCurves = useMemo(() => {
    if (!metrics.length) return [];
    return metrics
      .filter((m) => m.equity_curve && m.equity_curve.length > 0)
      .map((m) => ({
        model: m.model,
        data: m.equity_curve!,
      }));
  }, [metrics]);

  if (jobsLoading) {
    return (
      <div className="flex flex-col gap-6">
        <h2
          className="text-base font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model Comparison
        </h2>
        <CompareSkeleton />
      </div>
    );
  }

  if (multiModelJobs.length === 0) {
    return (
      <div className="flex flex-col gap-6">
        <h2
          className="text-base font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model Comparison
        </h2>
        <EmptyState
          icon={<GitCompare size={48} />}
          title="No multi-model jobs"
          description="Run a backtest with 2+ models to generate a leaderboard, equity overlay, and significance testing matrix."
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2
          className="text-base font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model Comparison
        </h2>
        <select
          value={selectedJobId ?? ""}
          onChange={(e) => setSelectedJobId(e.target.value || null)}
          className="rounded-md border px-3 py-2 text-xs"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-surface)",
            color: "var(--color-text-primary)",
            fontFamily: "var(--font-mono)",
            cursor: "pointer",
            minWidth: 280,
          }}
        >
          <option value="">Select a multi-model job…</option>
          {multiModelJobs.map((job) => (
            <option key={job.job_id} value={job.job_id}>
              {job.job_id.slice(0, 8)}… — {job.pair} — {job.models?.join(", ")}
            </option>
          ))}
        </select>
      </div>

      {!selectedJobId && (
        <EmptyState
          icon={<GitCompare size={48} />}
          title="Select a job to compare"
          description={`${multiModelJobs.length} multi-model job${multiModelJobs.length !== 1 ? "s" : ""} available. Choose one from the dropdown above.`}
        />
      )}

      {selectedJobId && resultsLoading && <CompareSkeleton />}

      {selectedJobId && !resultsLoading && results && (
        <>
          <LeaderboardTable metrics={metrics} sortMetric="sharpe" />

          <EquityOverlayChart curves={modelCurves} />

          <SignificanceMatrix
            models={metrics.map((m) => m.model)}
            pValues={null}
          />

          {metrics.length > 0 && metrics[0].hpo_trials && (
            <ParameterSensitivityChart trials={metrics[0].hpo_trials} />
          )}
        </>
      )}

      <CrossPairSection />

      {selectedJobId && !resultsLoading && !results && (
        <div
          className="flex items-center justify-center rounded-lg border p-8"
          style={{
            backgroundColor: "var(--color-surface)",
            borderColor: "var(--color-border)",
            color: "var(--color-text-muted)",
          }}
        >
          <span className="text-sm" style={{ fontFamily: "var(--font-mono)" }}>
            Could not load results for this job.
          </span>
        </div>
      )}
    </div>
  );
}
