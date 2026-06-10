import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { useValidation } from "@/hooks/useValidation";
import { useSubmitBacktest } from "@/api/queries";
import { ConfigSummaryBar } from "@/components/shared/ConfigSummaryBar";
import { StepTracker } from "@/components/shared/StepTracker";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import { ValidationBar } from "@/components/shared/ValidationBar";
import { AssetSelector } from "./AssetSelector";
import { ModelSelector } from "./ModelSelector";
import { HpoPanel } from "./HpoPanel";
import { FeaturesPanel } from "./FeaturesPanel";
import { LabelsPanel } from "./LabelsPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { HyperparamsTab } from "./HyperparamsTab";
import { RunSummary } from "./RunSummary";

import { ForwardTestTab } from "./ForwardTestTab";

const TABS = [
  { key: "asset", label: "Asset & Model" },
  { key: "study", label: "Optimization" },
  { key: "design", label: "Features & Tuning" },
  { key: "execution", label: "Execution & Testing" },
];

const STEP_META: Record<string, { title: string; subtitle: string }> = {
  asset: { title: "Asset & Strategy Setup", subtitle: "Select the instrument and model architecture." },
  study: { title: "Study & Optimization", subtitle: "Configure hyperparameter search and walk-forward validation." },
  design: { title: "Features & Tuning", subtitle: "Engineer input signals, define labels, and tune model hyperparameters." },
  execution: { title: "Execution & Testing", subtitle: "Define trade simulation rules and validate on out-of-sample data." },
};

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
  const [designSeg, setDesignSeg] = useState("features");
  const [execSeg, setExecSeg] = useState("execution");

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
    <div className="flex flex-col flex-1" style={{ minHeight: "100%" }}>
      {/* Page title */}
      <div className="flex flex-col gap-1">
        <h1
          className="text-[28px] font-bold tracking-tight"
          style={{ color: "var(--color-text-primary)" }}
        >
          {STEP_META[activeTab]?.title ?? "Backtest Setup"}
        </h1>
        <p className="text-[13px]" style={{ color: "var(--color-text-muted)" }}>
          {STEP_META[activeTab]?.subtitle ?? ""}
        </p>
      </div>

      {/* Step progress tracker */}
      <div className="mt-5 max-w-[720px]">
        <StepTracker steps={TABS} activeStep={activeTab} onStepChange={setActiveTab} />
      </div>

      {/* Config summary strip */}
      <div className="mt-6">
        <ConfigSummaryBar />
      </div>

      {/* 24px gap between header and main content */}
      <div className="h-6 shrink-0" />

      {/* Content area grows to fill remaining space */}
      <div className="flex flex-col flex-1">

      {/* Asset & Model */}
      {activeTab === "asset" && (
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(300px,360px)_1fr] gap-5">
          <AssetSelector />
          <ModelSelector />
        </div>
      )}

      {/* Optimization */}
      {activeTab === "study" && (
        <div className="flex flex-col gap-5">
          <HpoPanel />
        </div>
      )}

      {/* Features & Tuning */}
      {activeTab === "design" && (
        <div className="flex flex-col gap-5">
          <SegmentedControl
            segments={[
              { key: "features", label: "Feature Engineering" },
              { key: "labels", label: "Labels" },
              { key: "hyperparams", label: "Hyperparameters", badge: selectedModels.length },
            ]}
            active={designSeg}
            onChange={setDesignSeg}
          />
          {designSeg === "features" && <FeaturesPanel />}
          {designSeg === "labels" && <LabelsPanel />}
          {designSeg === "hyperparams" && <HyperparamsTab />}
        </div>
      )}

      {/* Execution & Testing */}
      {activeTab === "execution" && (
        <div className="flex flex-col gap-5">
          <SegmentedControl
            segments={[
              { key: "execution", label: "Execution Settings" },
              { key: "forwardtest", label: "Forward Test" },
            ]}
            active={execSeg}
            onChange={setExecSeg}
          />
          {execSeg === "execution" && <ExecutionPanel defaultOpen={true} />}
          {execSeg === "forwardtest" && <ForwardTestTab />}
        </div>
      )}

      </div>{/* end flex-1 content area */}

      {/* Bottom spacer so content clears the sticky footer */}
      <div className="h-[72px] shrink-0" />

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
            className="flex w-[400px] flex-col gap-4 rounded-sm border p-5"
            style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
              Save as Quick Start Preset
            </h3>
            <input
              placeholder="Preset name"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSavePreset()}
              className="rounded-md border px-3 py-2 text-xs w-full"
              style={{ borderColor: "var(--color-border)", backgroundColor: "var(--color-elevated)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", outline: "none" }}
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowSavePreset(false)}
                className="rounded-md px-4 py-1.5 text-xs font-semibold uppercase"
                style={{ border: "1px solid var(--color-border)", color: "var(--color-text-secondary)" }}
              >
                Cancel
              </button>
              <button
                onClick={handleSavePreset}
                disabled={!presetName.trim()}
                className="rounded-md px-4 py-1.5 text-xs font-semibold uppercase"
                style={{
                  backgroundColor: presetName.trim() ? "var(--color-brand)" : "var(--color-border)",
                  color: presetName.trim() ? "var(--color-text-inverse)" : "var(--color-text-muted)",
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
