import { TooltipIcon } from "@/components/shared/TooltipLabel";
import { modelDescriptions } from "@/lib/tokens";
import { Cpu, Network, GitBranch, Layers, Boxes, Zap, Lock } from "lucide-react";
import type { ModelInfo } from "@/api/schemas";

export const GPU_MODELS = new Set([
  "cnn",
  "lstm",
  "transformer",
  "gru",
  "gru_lstm",
  "dqn",
  "ensemble_adaptive_regime",
  "ensemble_cnn_lstm_xgboost",
  "meta_ensemble",
  "stacking_ensemble",
]);

export const CATEGORY_ICON: Record<string, React.ElementType> = {
  classical: GitBranch,
  deep: Network,
  rl: Zap,
  ensemble: Boxes,
};

export const CATEGORY_BADGE: Record<string, string> = {
  classical: "Classical",
  deep: "Deep Learning",
  rl: "Reinforcement",
  ensemble: "Ensemble",
};

export interface ModelCardProps {
  model: ModelInfo;
  isSelected: boolean;
  isFull?: boolean;
  isLocked?: boolean;
  categoryKey: string;
  categoryColor: string;
  onToggle: (e: React.MouseEvent) => void;
}

export function ModelCard({
  model,
  isSelected,
  isFull = false,
  isLocked = false,
  categoryKey,
  categoryColor,
  onToggle,
}: ModelCardProps) {
  const desc = modelDescriptions[model.name];
  const needsGpu = GPU_MODELS.has(model.name);
  const Icon = CATEGORY_ICON[categoryKey] ?? Layers;
  const badge = CATEGORY_BADGE[categoryKey] ?? categoryKey;
  const disabled = isFull || isLocked;

  return (
    <button
      onClick={(e) => onToggle(e)}
      disabled={disabled}
      className="flex flex-row items-center gap-3 rounded-lg border p-3 text-left transition-all duration-150 overflow-visible"
      style={{
        borderColor: isSelected ? "var(--color-brand)" : "var(--color-glass-border)",
        backgroundColor: isSelected ? "rgba(0,229,255,0.05)" : "var(--color-input-bg)",
        opacity: disabled ? 0.35 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        boxShadow: isSelected ? "0 0 16px rgba(0,229,255,0.12)" : "none",
      }}
      title={isLocked ? "Requires Pro license" : undefined}
    >
      <div
        className="flex shrink-0 items-center justify-center rounded-lg"
        style={{
          width: 36,
          height: 36,
          border: "1px solid var(--color-glass-border)",
          color: categoryColor,
          backgroundColor: "var(--color-elevated)",
        }}
      >
        <Icon size={17} strokeWidth={1.75} />
      </div>

      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <span className="flex items-center gap-1.5 text-[13px] leading-tight font-semibold text-(--color-text-primary)">
          <span className="truncate">{desc?.name ?? model.display_name}</span>
          {desc?.apprentice && <TooltipIcon text={desc.apprentice} />}
        </span>
        <span className="text-[11px] leading-snug text-(--color-text-muted) truncate">
          {desc?.short ?? model.description}
        </span>
      </div>

      <div className="flex shrink-0 items-center gap-1.5">
        {isLocked && (
          <span className="inline-flex items-center gap-1 rounded-full border border-(--color-accent-warning) bg-(color-mix(in srgb, var(--color-accent-warning) 10%, transparent)) px-1.5 py-0.5 text-[8px] font-semibold text-(--color-accent-warning)">
            <Lock size={8} />
            PRO
          </span>
        )}
        <span
          className="inline-flex rounded-full border px-1.5 py-0.5 text-[8px] font-semibold tracking-[0.06em] uppercase"
          style={{
            borderColor: `${categoryColor}40`,
            color: categoryColor,
            backgroundColor: `${categoryColor}12`,
          }}
        >
          {badge}
        </span>
        {needsGpu && (
          <span
            className="inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[8px] font-semibold tracking-[0.06em] uppercase"
            style={{
              borderColor: "rgba(245,158,11,0.25)",
              color: "var(--color-accent-warning)",
              backgroundColor: "rgba(245,158,11,0.08)",
            }}
          >
            <Cpu size={8} strokeWidth={1.75} />
            GPU
          </span>
        )}
        <span
          className="flex shrink-0 items-center justify-center rounded-full transition-all duration-150"
          style={{
            width: 16,
            height: 16,
            border: `1.5px solid ${isSelected ? "var(--color-brand)" : "var(--color-text-dim)"}`,
          }}
        >
          {isSelected && (
            <span className="rounded-full bg-(--color-brand)" style={{ width: 8, height: 8 }} />
          )}
        </span>
      </div>
    </button>
  );
}
