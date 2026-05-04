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
import { QuickTestBar } from "./QuickTestBar";
import { ValidationAlert } from "@/components/shared/ValidationAlert";
import { RuntimeEstimate } from "@/components/shared/RuntimeEstimate";

export function BacktestPage() {
  const navigate = useNavigate();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const toPayload = useBacktestStore((s) => s.toRequestPayload);
  const startJob = useJobStore((s) => s.startJob);
  const { warnings, errors, ok } = useValidation();
  const submit = useSubmitBacktest();

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [advancedMode, setAdvancedMode] = useState(false);

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
    <div className="flex flex-col gap-6">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <RuntimeEstimate />
          <button
            onClick={() => hasModels && ok && setSummaryOpen(true)}
            disabled={!hasModels || submit.isPending}
            className="rounded-md px-7 py-2.5 text-xs font-semibold uppercase transition-all duration-300 hover:brightness-110"
            style={{
              background: hasModels
                ? "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)"
                : "var(--color-glass-border)",
              color: hasModels ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              letterSpacing: "0.08em",
              cursor: hasModels ? "pointer" : "not-allowed",
              opacity: submit.isPending ? 0.6 : 1,
              boxShadow: hasModels
                ? "0 0 24px rgba(0,229,255,0.2)"
                : "none",
            }}
          >
            {submit.isPending ? "Submitting…" : "Deploy Backtest"}
          </button>
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
