import { useBacktestStore } from "@/stores/useBacktestStore";
import { ModelHyperparamsPanel } from "./ModelHyperparamsPanel";
import { Panel, PanelHeader } from "@/components/shared/Panel";

export function HyperparamsTab() {
  const selectedModels = useBacktestStore((s) => s.selectedModels);

  if (selectedModels.length === 0) {
    return (
      <Panel>
        <PanelHeader
          title="Model Hyperparameters"
          subtitle="Configure the underlying execution logic."
        />
        <div className="flex items-center justify-center rounded-lg border border-(--color-glass-border) py-16 text-(--color-text-muted)">
          <span className="text-xs">Select a model first to see its hyperparameters.</span>
        </div>
      </Panel>
    );
  }

  return (
    <Panel>
      <PanelHeader
        title="Model Hyperparameters"
        subtitle="Configure the underlying execution logic for each selected model."
      />

      {/* Model-specific hyperparameter controls */}
      <ModelHyperparamsPanel />
    </Panel>
  );
}
