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

const TABS = [
  { key: "asset", label: "Asset & Model" },
  { key: "study", label: "Study & HPO" },
  { key: "features", label: "Features" },
  { key: "execution", label: "Labels & Execution" },
];

export function BacktestPage() {
  const navigate = useNavigate();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const toPayload = useBacktestStore((s) => s.toRequestPayload);
  const startJob = useJobStore((s) => s.startJob);
  const { warnings, errors, ok } = useValidation();
  const submit = useSubmitBacktest();

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("asset");
  const [advancedHpo, setAdvancedHpo] = useState(false);

  const isSubmitting = submit.isPending;
  const hasModels = selectedModels.length > 0;
  const canDeploy = hasModels && ok;
  const disabledTabs = new Set<string>();
  if (!hasModels) {
    disabledTabs.add("study");
    disabledTabs.add("features");
    disabledTabs.add("execution");
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <RuntimeEstimate />
      </div>

      <QuickTestBar />

      {/* Tab navigation */}
      <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} disabledTabs={disabledTabs} />

      {/* Tab content */}
      {activeTab === "asset" && (
        <div className="flex flex-col gap-5">
          <AssetSelector />
          <ModelSelector />
        </div>
      )}

      {activeTab === "study" && (
        <div className="flex flex-col gap-5">
          <HpoPanel advancedMode={advancedHpo} onToggleAdvanced={() => setAdvancedHpo(!advancedHpo)} />
        </div>
      )}

      {activeTab === "features" && (
        <div className="flex flex-col gap-5">
          <FeaturesPanel />
          <LabelsPanel />
        </div>
      )}

      {activeTab === "execution" && (
        <div className="flex flex-col gap-5">
          <ExecutionPanel defaultOpen={true} />
        </div>
      )}

      {/* Sticky validation bar */}
      <ValidationBar
        warnings={warnings.length}
        errors={errors.length}
        canDeploy={canDeploy}
        isSubmitting={isSubmitting}
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
