import { useFullCycleStore, ALL_MODELS } from "@/stores/useFullCycleStore";
import { modelCategories, modelDescriptions } from "@/lib/tokens";
import { ModelCard } from "@/components/shared/ModelCard";
import { Panel, PanelHeader } from "@/components/shared/Panel";
import type { ModelInfo } from "@/api/schemas";

const modelCategoryLookup: Record<string, string> = {};
for (const [catKey, cat] of Object.entries(modelCategories)) {
  for (const m of cat.models) {
    modelCategoryLookup[m] = catKey;
  }
}

function getCategory(modelName: string): string {
  return modelCategoryLookup[modelName] ?? "classical";
}

export function BaseModelsTab() {
  const selected = useFullCycleStore((s) => s.selectedModels);
  const toggleModel = useFullCycleStore((s) => s.toggleModel);

  const modelsByCategory: Record<string, ModelInfo[]> = {};
  for (const modelName of ALL_MODELS) {
    const cat = getCategory(modelName);
    const desc = modelDescriptions[modelName];
    if (!modelsByCategory[cat]) modelsByCategory[cat] = [];
    modelsByCategory[cat].push({
      name: modelName,
      display_name: desc?.name ?? modelName,
      category: cat as ModelInfo["category"],
      description: desc?.short ?? "",
    });
  }

  return (
    <div className="flex flex-col gap-6">
      <Panel>
        <PanelHeader
          title="Base Model Pool"
          subtitle="Select the models that will compete for a seat on the committee. You can choose any number of models."
          accessory={
            <span className="rounded-md border border-(--color-glass-border) px-3 py-1 font-mono text-[11px] font-semibold text-(--color-text-secondary)">
              {selected.length}/{ALL_MODELS.length} selected
            </span>
          }
        />

        <div className="flex flex-col gap-5">
          {Object.entries(modelCategories).map(([catKey, cat]) => {
            const catModels = modelsByCategory[catKey] ?? [];
            if (catModels.length === 0) return null;
            return (
              <div key={catKey}>
                <div className="mb-3 flex items-center gap-2">
                  <div
                    className="shrink-0 rounded-full"
                    style={{ width: 8, height: 8, backgroundColor: cat.color }}
                  />
                  <span className="text-[10px] font-medium tracking-[0.14em] whitespace-nowrap text-(--color-text-secondary) uppercase">
                    {cat.label}
                  </span>
                  <div className="h-px flex-1" style={{ backgroundColor: "#333" }} />
                </div>
                <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 2xl:grid-cols-3">
                  {catModels.map((m) => (
                    <ModelCard
                      key={m.name}
                      model={m}
                      isSelected={selected.includes(m.name)}
                      categoryKey={catKey}
                      categoryColor={cat.color}
                      onToggle={() => toggleModel(m.name)}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
