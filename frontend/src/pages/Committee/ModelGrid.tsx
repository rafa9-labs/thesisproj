import { ALL_MODELS } from "@/stores/useFullCycleStore";
import { modelDescriptions } from "@/lib/tokens";

interface Props {
  selected: string[];
  onToggle: (model: string) => void;
}

export function ModelGrid({ selected, onToggle }: Props) {
  return (
    <div className="flex flex-col gap-4 rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-6 backdrop-blur-[12px]">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[11px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
          Model Selection
        </span>
        <span className="font-mono text-[11px] font-medium text-(--color-text-muted)">
          {selected.length}/{ALL_MODELS.length} selected
        </span>
      </div>

      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
        {ALL_MODELS.map((m) => {
          const desc = (
            modelDescriptions as Record<string, { name: string; short: string; apprentice: string }>
          )[m];
          const isSelected = selected.includes(m);
          return (
            <button
              key={m}
              onClick={() => onToggle(m)}
              title={desc?.apprentice ?? m}
              className="font-mono"
              style={{
                background: isSelected ? "var(--color-brand-glow)" : "var(--color-elevated)",
                border: `1px solid ${isSelected ? "var(--color-brand)" : "var(--color-glass-border)"}`,
                borderRadius: 4,
                padding: "4px 10px",
                fontSize: 10,
                color: isSelected ? "var(--color-brand)" : "var(--color-text-muted)",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              {desc?.name ?? m}
            </button>
          );
        })}
      </div>
    </div>
  );
}
