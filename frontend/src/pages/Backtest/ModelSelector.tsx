import { useModels } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { Panel, PanelHeader } from "@/components/shared/Panel";
import { ModelCard } from "@/components/shared/ModelCard";
import { modelCategories } from "@/lib/tokens";
import type { ModelInfo } from "@/api/schemas";

export function ModelSelector() {
  const { data: models, isLoading } = useModels();
  const selected = useBacktestStore((s) => s.selectedModels);
  const toggleModel = useBacktestStore((s) => s.toggleModel);

  const modelsSafe = Array.isArray(models) ? models : [];

  const modelsByCategory = categorizeModels(modelsSafe);

  return (
    <Panel>
      <PanelHeader
        title="Model Architecture"
        subtitle="Click any model to toggle it on or off. You can select multiple models."
        accessory={
          <span className="rounded-md border border-(--color-glass-border) px-3 py-1 font-mono text-[11px] font-semibold text-(--color-text-secondary)">
            {selected.length}/5 selected
          </span>
        }
      />

      {isLoading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-12 animate-skeleton rounded-sm bg-(--color-glass-hover)" />
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-5">
          {Object.entries(modelCategories).map(([catKey, cat]) => {
            const catModels = modelsByCategory[catKey] ?? [];
            if (catModels.length === 0) return null;
            return (
              <div key={catKey}>
                {/* Category divider: hairline + dot + off-white label */}
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
                      isFull={selected.length >= 5 && !selected.includes(m.name)}
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
      )}
    </Panel>
  );
}



function categorizeModels(models: ModelInfo[]): Record<string, ModelInfo[]> {
  const result: Record<string, ModelInfo[]> = {};
  for (const model of models) {
    const cat = model.category;
    if (!result[cat]) result[cat] = [];
    result[cat].push(model);
  }
  return result;
}
