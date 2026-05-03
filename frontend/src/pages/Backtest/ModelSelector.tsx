import { useModels } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelCategories, modelDescriptions } from "@/lib/tokens";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { AlertTriangle } from "lucide-react";
import type { ModelInfo } from "@/api/schemas";

const GPU_MODELS = new Set(["cnn", "lstm", "transformer", "dqn", "ensemble_adaptive_regime"]);

export function ModelSelector() {
  const { data: models, isLoading } = useModels();
  const selected = useBacktestStore((s) => s.selectedModels);
  const toggleModel = useBacktestStore((s) => s.toggleModel);
  const verbose = useSettingsStore((s) => s.verboseMode);

  const modelsByCategory = categorizeModels(models ?? []);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <div className="mb-3 flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model Selection
        </h3>
        <span className="text-xs" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {selected.length}/5 selected
        </span>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-24 animate-skeleton rounded-md"
              style={{ backgroundColor: "var(--color-elevated)" }}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-5">
          {Object.entries(modelCategories).map(([catKey, cat]) => {
            const catModels = modelsByCategory[catKey] ?? [];
            if (catModels.length === 0) return null;
            return (
              <div key={catKey}>
                <div className="mb-2 flex items-center gap-2">
                  <div className="h-px flex-1" style={{ backgroundColor: cat.color, opacity: 0.3 }} />
                  <span
                    className="text-xs font-semibold uppercase tracking-[0.1em]"
                    style={{ color: cat.color }}
                  >
                    {cat.label}
                  </span>
                  <div className="h-px flex-1" style={{ backgroundColor: cat.color, opacity: 0.3 }} />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  {catModels.map((m) => (
                    <ModelCard
                      key={m.name}
                      model={m}
                      isSelected={selected.includes(m.name)}
                      isFull={selected.length >= 5 && !selected.includes(m.name)}
                      categoryColor={cat.color}
                      verbose={verbose}
                      onToggle={() => toggleModel(m.name)}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ModelCard({
  model,
  isSelected,
  isFull,
  categoryColor,
  verbose,
  onToggle,
}: {
  model: ModelInfo;
  isSelected: boolean;
  isFull: boolean;
  categoryColor: string;
  verbose: boolean;
  onToggle: () => void;
}) {
  const desc = modelDescriptions[model.name];
  const needsGpu = GPU_MODELS.has(model.name);

  return (
    <button
      onClick={onToggle}
      disabled={isFull}
      className="flex flex-col gap-1.5 rounded-md border p-3 text-left transition-all duration-150"
      style={{
        borderColor: isSelected ? categoryColor : "var(--color-border)",
        backgroundColor: isSelected ? "var(--color-elevated)" : "var(--color-surface)",
        opacity: isFull ? 0.4 : 1,
        cursor: isFull ? "not-allowed" : "pointer",
        borderLeftWidth: isSelected ? 3 : 1,
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
          {desc?.name ?? model.display_name}
        </span>
        {isSelected && (
          <span
            className="text-xs font-bold"
            style={{ color: categoryColor, fontFamily: "var(--font-mono)" }}
          >
            ✓
          </span>
        )}
      </div>
      <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
        {desc?.short ?? model.description}
      </span>
      {needsGpu && !isSelected && (
        <div className="flex items-center gap-1">
          <AlertTriangle size={10} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-[10px]" style={{ color: "var(--color-accent-warning)" }}>
            GPU recommended
          </span>
        </div>
      )}
      {verbose && desc?.apprentice && (
        <p className="mt-1 text-[11px] leading-snug" style={{ color: "var(--color-text-secondary)" }}>
          {desc.apprentice}
        </p>
      )}
    </button>
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
