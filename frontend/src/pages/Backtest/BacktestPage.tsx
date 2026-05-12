import { useState } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { useValidation } from "@/hooks/useValidation";
import { useActiveBacktests, useForceStopJob, useSubmitBacktest } from "@/api/queries";
import { useBacktestWebSocket } from "@/hooks/useBacktestWebSocket";
import { AssetSelector } from "./AssetSelector";
import { ModelSelector } from "./ModelSelector";
import { FeaturesPanel } from "./FeaturesPanel";
import { LabelsPanel } from "./LabelsPanel";
import { HpoPanel } from "./HpoPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { RunSummary } from "./RunSummary";
import { BacktestProgress } from "./BacktestProgress";
import { QuickTestBar } from "./QuickTestBar";
import { ValidationAlert } from "@/components/shared/ValidationAlert";
import { RuntimeEstimate } from "@/components/shared/RuntimeEstimate";

export function BacktestPage() {
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const toPayload = useBacktestStore((s) => s.toRequestPayload);
  const startJob = useJobStore((s) => s.startJob);
  const { warnings, errors, ok } = useValidation();
  const submit = useSubmitBacktest();
  const { data: activeBacktestsData } = useActiveBacktests();
  const forceStop = useForceStopJob();

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [advancedMode, setAdvancedMode] = useState(false);

  useBacktestWebSocket(activeJobId);

  const activeJob = useJobStore((s) => (activeJobId ? s.activeJobs.get(activeJobId) : undefined));

  const activeBacktests = activeBacktestsData?.jobs ?? [];
  const hasActiveJob =
    activeBacktests.length > 0 ||
    activeJob?.status === "pending" ||
    activeJob?.status === "running";

  // Keep user on Backtest Setup page when job completes (results stay visible in HPO and Results tab)

  const handleDeploy = async () => {
    try {
      const payload = toPayload();
      const result = await submit.mutateAsync(payload);
      setActiveJobId(result.job_id);
      startJob(result.job_id, payload.pair, payload.models);
      setSummaryOpen(false);
    } catch (err) {
      console.error("Deploy failed:", err);
    }
  };

  const handleForceStop = async () => {
    const targetId = activeJobId || activeBacktests[0]?.job_id;
    if (!targetId) return;
    try {
      await forceStop.mutateAsync(targetId);
      if (activeJobId === targetId) {
        setActiveJobId(null);
      }
    } catch (err) {
      console.error("Force stop failed:", err);
    }
  };

  const hasModels = selectedModels.length > 0;

  return (
    <div className="flex flex-col gap-6">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <RuntimeEstimate />
          <button
            onClick={() => hasModels && ok && !hasActiveJob && setSummaryOpen(true)}
            disabled={!hasModels || submit.isPending || hasActiveJob}
            className="rounded-md px-7 py-2.5 text-xs font-semibold uppercase transition-all duration-300 hover:brightness-110"
            style={{
              background: hasModels && !hasActiveJob
                ? "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)"
                : "var(--color-glass-border)",
              color: hasModels && !hasActiveJob ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              letterSpacing: "0.08em",
              cursor: hasModels && !hasActiveJob ? "pointer" : "not-allowed",
              opacity: submit.isPending || hasActiveJob ? 0.6 : 1,
              boxShadow: hasModels && !hasActiveJob
                ? "0 0 24px rgba(0,229,255,0.2)"
                : "none",
            }}
          >
            {hasActiveJob
              ? "Backtest in progress — wait for completion."
              : submit.isPending
                ? "Submitting…"
                : "Deploy Backtest"}
          </button>
          {hasActiveJob && (
            <button
              onClick={handleForceStop}
              disabled={forceStop.isPending}
              className="text-[11px] font-medium uppercase tracking-[0.06em] transition-colors hover:brightness-110"
              style={{ color: "var(--color-accent-danger)", opacity: forceStop.isPending ? 0.5 : 1 }}
            >
              {forceStop.isPending ? "Stopping…" : "Force Stop"}
            </button>
          )}
        </div>
      </div>

      {/* Quick Start */}
      <QuickTestBar />

      {/* Active progress */}
      {activeJobId && (
        <BacktestProgress jobId={activeJobId} />
      )}

      {/* Validation alerts */}
      {(warnings.length > 0 || errors.length > 0) && hasModels && (
        <ValidationAlert warnings={warnings} errors={errors} />
      )}

      {/* Config sections */}
      <AssetSelector />
      <ModelSelector />

      {advancedMode ? (
        <>
          <FeaturesPanel />
          <LabelsPanel />
          <HpoPanel advancedMode={advancedMode} onToggleAdvanced={() => setAdvancedMode(!advancedMode)} />
          <ExecutionPanel />
        </>
      ) : (
        <HpoPanel advancedMode={advancedMode} onToggleAdvanced={() => setAdvancedMode(!advancedMode)} />
      )}

      {/* Run summary modal */}
      <RunSummary
        open={summaryOpen}
        onClose={() => setSummaryOpen(false)}
        onDeploy={handleDeploy}
        warnings={warnings.length}
        errors={errors.length}
      />
    </div>
  );
}
