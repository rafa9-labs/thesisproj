import { useBacktestStore } from "@/stores/useBacktestStore";
import { ModelHyperparamsPanel } from "./ModelHyperparamsPanel";
import { ParameterGuideInline } from "./ParameterGuide";

export function HyperparamsTab() {
  const selectedModels = useBacktestStore((s) => s.selectedModels);

  if (selectedModels.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border py-16"
        style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-muted)" }}
      >
        <span className="text-xs">Select a model first to see hyperparameters.</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 pt-1">
      {/* Model-specific hyperparameter controls */}
      <ModelHyperparamsPanel />

      {/* Parameter guidance per model */}
      <ParameterGuideInline />
    </div>
  );
}
