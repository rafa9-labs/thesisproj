import { useModels } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelCategories, modelDescriptions } from "@/lib/tokens";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { Panel, PanelHeader } from "@/components/shared/Panel";
import { Cpu, Network, GitBranch, Layers, Boxes, Zap } from "lucide-react";
import type { ModelInfo } from "@/api/schemas";

const GPU_MODELS = new Set(["cnn", "lstm", "transformer", "gru", "gru_lstm", "dqn", "ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost", "meta_ensemble", "stacking_ensemble"]);

const CATEGORY_ICON: Record<string, React.ElementType> = {
  classical: GitBranch,
  deep: Network,
  rl: Zap,
  ensemble: Boxes,
};

const CATEGORY_BADGE: Record<string, string> = {
  classical: "Classical",
  deep: "Deep Learning",
  rl: "Reinforcement",
  ensemble: "Ensemble",
};

export function ModelSelector() {
  const { data: models, isLoading } = useModels();
  const selected = useBacktestStore((s) => s.selectedModels);
  const toggleModel = useBacktestStore((s) => s.toggleModel);
  const verbose = useSettingsStore((s) => s.verboseMode);

  const modelsByCategory = categorizeModels(models ?? []);

  return (
    <Panel>
      <PanelHeader
        title="Model Architecture"
        subtitle="Choose one or more models to train and compare."
        accessory={
          <span
            className="rounded-md border px-3 py-1 text-[11px] font-semibold"
            style={{
              borderColor: "var(--color-glass-border)",
              color: "var(--color-text-secondary)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {selected.length}/5 selected
          </span>
        }
      />

      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="h-28 animate-skeleton rounded-sm"
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
                      categoryKey={catKey}
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
    </Panel>
  );
}

function ModelCard({
  model,
  isSelected,
  isFull,
  categoryKey,
  categoryColor,
  verbose,
  onToggle,
}: {
  model: ModelInfo;
  isSelected: boolean;
  isFull: boolean;
  categoryKey: string;
  categoryColor: string;
  verbose: boolean;
  onToggle: () => void;
}) {
  const desc = modelDescriptions[model.name];
  const needsGpu = GPU_MODELS.has(model.name);
  const Icon = CATEGORY_ICON[categoryKey] ?? Layers;
  const badge = CATEGORY_BADGE[categoryKey] ?? categoryKey;

  return (
    <button
      onClick={onToggle}
      disabled={isFull}
      className="flex flex-col gap-3 rounded-lg border p-4 text-left transition-all duration-150"
      style={{
        borderColor: isSelected ? "var(--color-brand)" : "var(--color-glass-border)",
        backgroundColor: isSelected ? "rgba(0,229,255,0.05)" : "var(--color-input-bg)",
        opacity: isFull ? 0.35 : 1,
        cursor: isFull ? "not-allowed" : "pointer",
        boxShadow: isSelected ? "0 0 16px rgba(0,229,255,0.12)" : "none",
      }}
    >
      {/* Top row: icon box + category badge */}
      <div className="flex items-start justify-between">
        <div
          className="flex items-center justify-center rounded-lg"
          style={{
            width: 38,
            height: 38,
            backgroundColor: "var(--color-elevated)",
            border: "1px solid var(--color-glass-border)",
            color: categoryColor,
          }}
        >
          <Icon size={18} strokeWidth={1.75} />
        </div>
        <span
          className="rounded-full border px-2.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em]"
          style={{
            borderColor: `${categoryColor}40`,
            color: categoryColor,
            backgroundColor: `${categoryColor}12`,
          }}
        >
          {badge}
        </span>
      </div>

      {/* Name */}
      <span className="text-[15px] font-bold leading-tight" style={{ color: "var(--color-text-primary)" }}>
        {desc?.name ?? model.display_name}
      </span>

      {/* Description */}
      <span
        className="text-[12px] font-light leading-relaxed"
        style={{
          color: "var(--color-text-secondary)",
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}
      >
        {desc?.short ?? model.description}
      </span>

      {verbose && desc?.apprentice && (
        <p className="text-[11px] font-light leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          {desc.apprentice}
        </p>
      )}

      {/* Footer: GPU flag + radio selector */}
      <div className="mt-1 flex items-center justify-between border-t pt-3" style={{ borderColor: "var(--color-glass-border)" }}>
        {needsGpu ? (
          <div className="flex items-center gap-1.5">
            <Cpu size={11} strokeWidth={1.75} style={{ color: "var(--color-accent-warning)" }} />
            <span className="text-[10px] font-medium uppercase tracking-[0.04em]" style={{ color: "var(--color-accent-warning)" }}>
              GPU Optimized
            </span>
          </div>
        ) : (
          <span className="text-[10px] font-medium uppercase tracking-[0.06em]" style={{ color: "var(--color-text-dim)" }}>
            {desc?.short ? "" : "CPU"}
          </span>
        )}

        {/* Radio selector */}
        <span
          className="flex items-center justify-center rounded-full transition-all duration-150"
          style={{
            width: 18,
            height: 18,
            border: `1.5px solid ${isSelected ? "var(--color-brand)" : "var(--color-text-dim)"}`,
          }}
        >
          {isSelected && (
            <span
              className="rounded-full"
              style={{ width: 9, height: 9, backgroundColor: "var(--color-brand)" }}
            />
          )}
        </span>
      </div>
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
