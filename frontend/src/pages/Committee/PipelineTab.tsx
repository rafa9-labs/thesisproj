import { useState, useCallback } from "react";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { PhaseArchitectureDiagram } from "@/components/shared/PhaseArchitectureDiagram";
import { PipelineParamPanel } from "./PipelineParamPanel";

export function PipelineTab() {
  const store = useFullCycleStore();
  const [selectedPhase, setSelectedPhase] = useState<number | null>(null);

  const handlePhaseSelect = useCallback((phaseId: number | null) => {
    setSelectedPhase(phaseId);
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-[960px] flex-col gap-0">
      <PhaseArchitectureDiagram
        config={{
          proposer: store.proposer,
          llmBackend: store.llmBackend,
          hpoSampler: store.hpoSampler,
          skipFeatureSweep: store.skipFeatureSweep,
          enablePhase3: store.enablePhase3,
          enablePhase4: store.enablePhase4,
          enablePhase5: store.enablePhase5,
          enablePhase6: store.enablePhase6,
          maxIterations: store.maxIterations,
          committeeTopK: store.committeeTopK,
          selectedModelCount: store.selectedModels.length,
        }}
        selectedPhase={selectedPhase}
        onPhaseSelect={handlePhaseSelect}
      />
      {selectedPhase !== null && <PipelineParamPanel selectedPhase={selectedPhase} />}
    </div>
  );
}
