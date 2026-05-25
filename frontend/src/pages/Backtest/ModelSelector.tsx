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
      className="flex flex-col gap-4 rounded-lg border p-5"
      style={{
        backgroundColor: "var(--color-glass)",
        borderColor: "var(--color-glass-border)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="mb-1 flex items-center justify-between">
        <span
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          Model Selection
        </span>
        <span className="text-[11px] font-medium" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {selected.length}/5 selected
        </span>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-28 animate-skeleton rounded-lg"
              style={{ backgroundColor: "var(--color-glass-hover)" }}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-7">
          {Object.entries(modelCategories).map(([catKey, cat]) => {
            const catModels = modelsByCategory[catKey] ?? [];
            if (catModels.length === 0) return null;
            return (
              <div key={catKey}>
                {/* Category divider: hairline + dot + off-white label */}
                <div className="mb-3 flex items-center gap-2">
                  <div
                    className="rounded-full shrink-0"
                    style={{ width: 8, height: 8, backgroundColor: cat.color }}
                  />
                  <span
                    className="text-[10px] font-medium uppercase tracking-[0.14em] whitespace-nowrap"
                    style={{ color: "var(--color-text-secondary)" }}
                  >
                    {cat.label}
                  </span>
                  <div className="h-px flex-1" style={{ backgroundColor: "#333" }} />
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
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
      className="flex flex-col gap-2 rounded-lg border p-4 text-left transition-all duration-150"
      style={{
        borderColor: isSelected ? "var(--color-brand)" : "#333",
        backgroundColor: isSelected ? "rgba(168,224,99,0.05)" : "var(--color-glass)",
        opacity: isFull ? 0.35 : 1,
        cursor: isFull ? "not-allowed" : "pointer",
        borderLeftWidth: isSelected ? 2 : 1,
        boxShadow: "none",
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
          {desc?.name ?? model.display_name}
        </span>
        {isSelected && (
          <span
            className="text-xs font-semibold"
            style={{ color: "var(--color-brand)", fontFamily: "var(--font-mono)" }}
          >
            ✓
          </span>
        )}
      </div>
      <span className="text-[12px] font-light leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
        {desc?.short ?? model.description}
      </span>
      {needsGpu && !isSelected && (
        <div className="flex items-center gap-1.5">
          <AlertTriangle size={10} strokeWidth={1.5} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-[10px] font-medium" style={{ color: "var(--color-accent-warning)" }}>
            GPU recommended
          </span>
        </div>
      )}
      {verbose && desc?.apprentice && (
        <p className="mt-1 text-[11px] font-light leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
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
