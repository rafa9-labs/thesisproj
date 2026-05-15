import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { useValidation } from "@/hooks/useValidation";
import { useSubmitBacktest } from "@/api/queries";
import { AssetSelector } from "./AssetSelector";
import { ModelSelector } from "./ModelSelector";
import { FeaturesPanel } from "./FeaturesPanel";
import { LabelsPanel } from "./LabelsPanel";
import { HpoPanel } from "./HpoPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { RunSummary } from "./RunSummary";
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
  const [advancedMode, setAdvancedMode] = useState(false);

  const isSubmitting = submit.isPending;

  const handleDeploy = async () => {
    try {
      const payload = toPayload();
      const result = await submit.mutateAsync(payload);
      startJob(result.job_id, payload.pair, payload.models);
      setSummaryOpen(false);
      navigate("/monitor");
    } catch (err) {
      console.error("Deploy failed:", err);
    }
  };

  const hasModels = selectedModels.length > 0;
  const canDeploy = hasModels && ok;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <RuntimeEstimate />
          <button
            onClick={() => canDeploy && setSummaryOpen(true)}
            disabled={!canDeploy || isSubmitting}
            className="rounded-md px-7 py-2.5 text-xs font-semibold uppercase transition-all duration-300 hover:brightness-110"
            style={{
              background: canDeploy
                ? "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)"
                : "var(--color-glass-border)",
              color: canDeploy ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              letterSpacing: "0.08em",
              cursor: canDeploy ? "pointer" : "not-allowed",
              opacity: isSubmitting ? 0.6 : 1,
              boxShadow: canDeploy
                ? "0 0 24px rgba(0,229,255,0.2)"
                : "none",
            }}
          >
            {isSubmitting ? "Submitting…" : "Deploy Backtest"}
          </button>
        </div>
      </div>

      <QuickTestBar />

      {(warnings.length > 0 || errors.length > 0) && hasModels && (
        <ValidationAlert warnings={warnings} errors={errors} />
      )}

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

      <RunSummary
        open={summaryOpen}
        onClose={() => setSummaryOpen(false)}
        onDeploy={handleDeploy}
        warnings={warnings.length}
        errors={errors.length}
        isSubmitting={isSubmitting}
      />
    </div>
  );
}
