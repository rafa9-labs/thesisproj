import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { useValidation } from "@/hooks/useValidation";
import { useSubmitBacktest } from "@/api/queries";
import { ConfigSummaryBar } from "@/components/shared/ConfigSummaryBar";
import { TabBar } from "@/components/shared/TabBar";
import { ValidationBar } from "@/components/shared/ValidationBar";
import { AssetSelector } from "./AssetSelector";
import { ModelSelector } from "./ModelSelector";
import { HpoPanel } from "./HpoPanel";
import { FeaturesPanel } from "./FeaturesPanel";
import { LabelsPanel } from "./LabelsPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { HyperparamsTab } from "./HyperparamsTab";
import { RunSummary } from "./RunSummary";

const TABS = [
  { key: "asset", label: "Asset & Timeframe" },
  { key: "models", label: "Models" },
  { key: "study", label: "Study & HPO" },
  { key: "features", label: "Features" },
  { key: "hyperparams", label: "Hyperparameters" },
  { key: "execution", label: "Execution" },
];

export function BacktestPage() {
  const navigate = useNavigate();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const toPayload = useBacktestStore((s) => s.toRequestPayload);
  const saveCustomPreset = useBacktestStore((s) => s.saveCustomPreset);
  const startJob = useJobStore((s) => s.startJob);
  const { warnings, errors, ok } = useValidation();
  const submit = useSubmitBacktest();

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("asset");

  const [presetName, setPresetName] = useState("");
  const [showSavePreset, setShowSavePreset] = useState(false);

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

  const handleSavePreset = () => {
    if (!presetName.trim()) return;
    saveCustomPreset(presetName.trim(), "");
    setPresetName("");
    setShowSavePreset(false);
  };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Status strip */}
      <ConfigSummaryBar />

      {/* 24px gap between status strip and tab nav */}
      <div className="mt-6">
        <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      {/* 24px gap between tab nav and main content */}
      <div className="h-6 shrink-0" />

      {/* Content area — scrollable, bottom padding for sticky footer clearance */}
      <div className="flex flex-1 flex-col overflow-y-auto pb-6">
        {/* Asset & Timeframe */}
        {activeTab === "asset" && (
          <div className="flex flex-col gap-5 pt-4">
            <AssetSelector />
          </div>
        )}

        {/* Models */}
        {activeTab === "models" && (
          <div className="flex flex-col gap-5 pt-4">
            <ModelSelector />
          </div>
        )}

        {/* Study & HPO */}
        {activeTab === "study" && (
          <div className="flex flex-col gap-5 pt-4">
            <HpoPanel />
          </div>
        )}

        {/* Features */}
        {activeTab === "features" && (
          <div className="flex flex-col gap-5 pt-4">
            <FeaturesPanel />
            <LabelsPanel />
          </div>
        )}

        {/* Hyperparameters */}
        {activeTab === "hyperparams" && (
          <div className="flex flex-col gap-5 pt-4">
            <HyperparamsTab />
          </div>
        )}

        {/* Execution */}
        {activeTab === "execution" && (
          <div className="flex flex-col gap-5 pt-4">
            <ExecutionPanel defaultOpen={true} />
          </div>
        )}
      </div>

      {/* Validation bar — sticky footer */}
      <ValidationBar
        warnings={warnings.length}
        errors={errors.length}
        canDeploy={canDeploy}
        isSubmitting={isSubmitting}
        hasModels={hasModels}
        hasPair={true}
        hasDates={true}
        onDeploy={() => setSummaryOpen(true)}
        onSavePreset={hasModels ? () => setShowSavePreset(true) : undefined}
      />

      {/* Save Preset dialog */}
      {showSavePreset && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ backgroundColor: "rgba(0,0,0,0.7)" }}
          onClick={() => setShowSavePreset(false)}
        >
          <div
            className="flex w-[400px] flex-col gap-4 rounded-sm border border-(--color-border) bg-(--color-surface) p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-sm font-semibold text-(--color-text-primary)">
              Save as Quick Start Preset
            </h3>
            <input
              placeholder="Preset name"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSavePreset()}
              className="w-full rounded-md border border-(--color-border) bg-(--color-elevated) px-3 py-2 font-mono text-xs text-(--color-text-primary)"
              style={{ outline: "none" }}
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowSavePreset(false)}
                className="rounded-md px-4 py-1.5 text-xs font-semibold text-(--color-text-secondary) uppercase"
                style={{ border: "1px solid var(--color-border)" }}
              >
                Cancel
              </button>
              <button
                onClick={handleSavePreset}
                disabled={!presetName.trim()}
                className="rounded-md px-4 py-1.5 text-xs font-semibold uppercase"
                style={{
                  backgroundColor: presetName.trim() ? "var(--color-brand)" : "var(--color-border)",
                  color: presetName.trim()
                    ? "var(--color-text-inverse)"
                    : "var(--color-text-muted)",
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
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
