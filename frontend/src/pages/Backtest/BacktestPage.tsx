import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { useValidation } from "@/hooks/useValidation";
import { useSubmitBacktest } from "@/api/queries";
import { TabBar } from "@/components/shared/TabBar";
import { ValidationBar } from "@/components/shared/ValidationBar";
import { RuntimeEstimate } from "@/components/shared/RuntimeEstimate";
import { AssetSelector } from "./AssetSelector";
import { ModelSelector } from "./ModelSelector";
import { QuickTestBar } from "./QuickTestBar";
import { HpoPanel } from "./HpoPanel";
import { FeaturesPanel } from "./FeaturesPanel";
import { LabelsPanel } from "./LabelsPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { RunSummary } from "./RunSummary";
import { STUDY_PRESETS } from "@/lib/constants";

const TABS = [
  { key: "quickstart", label: "Quick Start" },
  { key: "asset", label: "Asset & Model" },
  { key: "study", label: "Study & HPO" },
  { key: "features", label: "Features" },
  { key: "execution", label: "Execution" },
];

export function BacktestPage() {
  const navigate = useNavigate();
  const s = useBacktestStore.getState();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const toPayload = useBacktestStore((s) => s.toRequestPayload);
  const startJob = useJobStore((s) => s.startJob);
  const { warnings, errors, ok } = useValidation();
  const submit = useSubmitBacktest();

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("quickstart");
  const [advancedHpo, setAdvancedHpo] = useState(false);

  const isSubmitting = submit.isPending;
  const hasModels = selectedModels.length > 0;
  const canDeploy = hasModels && ok;

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

  const activePreset = s.activePreset;
  const presetInfo = activePreset ? STUDY_PRESETS[activePreset as keyof typeof STUDY_PRESETS] : null;
  const presetLabel = presetInfo?.label ?? s.hpoIntensity;

  // Build Quick Start summary string
  const summaryParts: string[] = [s.pair ?? "EURUSD", s.timeframe ?? "H1"];
  if (hasModels) {
    summaryParts.push(`${selectedModels.length} model${selectedModels.length > 1 ? "s" : ""}`);
  }
  if (presetLabel) {
    summaryParts.push(`${presetLabel} preset`);
  }
  const summaryText = summaryParts.join(" · ");

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <RuntimeEstimate />
      </div>

      {/* Tab navigation */}
      <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* ── Quick Start ── */}
      {activeTab === "quickstart" && (
        <div className="flex flex-col gap-5 pt-2">
          {/* Config summary */}
          <div
            className="rounded-xl border p-4"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] font-semibold uppercase tracking-[0.08em]" style={{ color: "var(--color-text-muted)" }}>
                Configuration
              </span>
            </div>
            <span
              className="text-sm font-mono"
              style={{ color: "var(--color-text-primary)" }}
            >
              {summaryText}
            </span>
          </div>

          {/* Quick presets */}
          <QuickTestBar />

          {/* Model selection */}
          <ModelSelector />

          {/* Deploy button */}
          {hasModels && (
            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSummaryOpen(true)}
                disabled={!canDeploy || isSubmitting}
                className="rounded-md px-8 py-3 text-sm font-bold uppercase transition-all duration-300 hover:brightness-110"
                style={{
                  background: canDeploy
                    ? "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)"
                    : "var(--color-glass-border)",
                  color: canDeploy ? "var(--color-text-inverse)" : "var(--color-text-muted)",
                  letterSpacing: "0.08em",
                  cursor: canDeploy ? "pointer" : "not-allowed",
                  boxShadow: canDeploy ? "0 0 24px rgba(0,229,255,0.2)" : "none",
                  opacity: isSubmitting ? 0.7 : 1,
                }}
              >
                {isSubmitting ? "Submitting..." : "Deploy Backtest"}
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Asset & Model ── */}
      {activeTab === "asset" && (
        <div className="flex flex-col gap-5 pt-2">
          <AssetSelector />
          <ModelSelector />
        </div>
      )}

      {/* ── Study & HPO ── */}
      {activeTab === "study" && (
        <div className="flex flex-col gap-5 pt-2">
          <HpoPanel advancedMode={advancedHpo} onToggleAdvanced={() => setAdvancedHpo(!advancedHpo)} />
        </div>
      )}

      {/* ── Features ── */}
      {activeTab === "features" && (
        <div className="flex flex-col gap-5 pt-2">
          <FeaturesPanel />
          <LabelsPanel />
        </div>
      )}

      {/* ── Execution ── */}
      {activeTab === "execution" && (
        <div className="flex flex-col gap-5 pt-2">
          <ExecutionPanel defaultOpen={true} />
        </div>
      )}

      {/* Sticky validation bar */}
      <ValidationBar
        warnings={warnings.length}
        errors={errors.length}
        canDeploy={canDeploy}
        isSubmitting={isSubmitting}
        hasModels={hasModels}
        hasPair={true}
        hasDates={true}
        onDeploy={() => setSummaryOpen(true)}
      />

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
