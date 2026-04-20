import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { useValidation } from "@/hooks/useValidation";
import { useSubmitBacktest } from "@/api/queries";
import { useBacktestWebSocket } from "@/hooks/useBacktestWebSocket";
import { AssetSelector } from "./AssetSelector";
import { ModelSelector } from "./ModelSelector";
import { FeaturesPanel } from "./FeaturesPanel";
import { LabelsPanel } from "./LabelsPanel";
import { HpoPanel } from "./HpoPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { RunSummary } from "./RunSummary";
import { BacktestProgress } from "./BacktestProgress";
import { ValidationAlert } from "@/components/shared/ValidationAlert";

export function BacktestPage() {
  const navigate = useNavigate();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const toPayload = useBacktestStore((s) => s.toRequestPayload);
  const startJob = useJobStore((s) => s.startJob);
  const { warnings, errors, ok } = useValidation();
  const submit = useSubmitBacktest();

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  useBacktestWebSocket(activeJobId);

  const activeJob = useJobStore((s) => (activeJobId ? s.activeJobs.get(activeJobId) : undefined));

  useEffect(() => {
    if (activeJob?.status === "completed" && activeJobId) {
      const timer = setTimeout(() => navigate(`/results/${activeJobId}`), 800);
      return () => clearTimeout(timer);
    }
  }, [activeJob?.status, activeJobId, navigate]);

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

  const hasModels = selectedModels.length > 0;

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2
          className="text-base font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          New Backtest
        </h2>
        <button
          onClick={() => hasModels && ok && setSummaryOpen(true)}
          disabled={!hasModels || submit.isPending}
          className="rounded-md px-6 py-2 text-xs font-bold uppercase transition-colors duration-150"
          style={{
            backgroundColor: hasModels ? "var(--color-accent)" : "var(--color-border)",
            color: hasModels ? "var(--color-text-inverse)" : "var(--color-text-muted)",
            letterSpacing: "0.05em",
            cursor: hasModels ? "pointer" : "not-allowed",
            opacity: submit.isPending ? 0.6 : 1,
          }}
        >
          {submit.isPending ? "Submitting..." : "Deploy Backtest"}
        </button>
      </div>

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
      <FeaturesPanel />
      <LabelsPanel />
      <HpoPanel />
      <ExecutionPanel />

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
