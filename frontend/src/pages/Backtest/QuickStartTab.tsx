import { useCallback } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { QUICK_START_PRESETS } from "@/lib/constants";
import { modelDescriptions } from "@/lib/tokens";
import { Trash2, Plus } from "lucide-react";

interface Props {
  onDeploy: () => void;
  canDeploy: boolean;
  isSubmitting: boolean;
}

const PRESET_COLORS: Record<string, string> = {
  baseline: "var(--color-accent-success)",
  classical: "var(--color-brand)",
  deep: "var(--color-accent)",
  ensemble: "var(--color-accent-warning)",
  full: "var(--color-accent-danger)",
  signal: "var(--color-text-muted)",
};

export function QuickStartTab({ onDeploy, canDeploy, isSubmitting }: Props) {
  const applyQuickPreset = useBacktestStore((s) => s.applyQuickPreset);
  const removeCustomPreset = useBacktestStore((s) => s.removeCustomPreset);
  const customPresets = useBacktestStore((s) => s.customPresets);
  const selectedModels = useBacktestStore((s) => s.selectedModels);

  const handlePreset = useCallback((key: string) => {
    applyQuickPreset(key);
  }, [applyQuickPreset]);

  const handleDeployCustom = useCallback(() => {
    if (canDeploy) onDeploy();
  }, [canDeploy, onDeploy]);

  const modelName = (m: string) =>
    (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m;

  return (
    <div className="flex flex-col gap-5 pt-1">
      {/* Predefined Quick Tests */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <span className="text-[10px] font-semibold uppercase tracking-[0.08em]" style={{ color: "var(--color-text-secondary)" }}>
            Predefined Studies
          </span>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {QUICK_START_PRESETS.map((preset) => {
            const color = PRESET_COLORS[preset.key] ?? "var(--color-brand)";
            const hrs = preset.estMinutes >= 120;
            const timeStr = hrs ? `${(preset.estMinutes / 60).toFixed(0)}h` : `${preset.estMinutes}min`;
            return (
              <div
                key={preset.key}
                onClick={() => handlePreset(preset.key)}
                className="rounded-lg border p-3 cursor-pointer transition-all duration-150 hover:brightness-110"
                style={{
                  borderColor: color,
                  backgroundColor: `${color}08`,
                }}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.08em]" style={{ color }}>
                    {preset.label}
                  </span>
                  <span className="text-[9px] font-mono" style={{ color: "var(--color-text-muted)" }}>
                    ~{timeStr}
                  </span>
                </div>
                <p className="text-[10px] mb-1.5" style={{ color: "var(--color-text-muted)" }}>
                  {preset.subtitle}
                </p>
                <div className="flex flex-wrap gap-1">
                  {preset.models.map((m) => (
                    <span
                      key={m}
                      className="rounded px-1 py-0.5 text-[8px] font-semibold uppercase tracking-wider"
                      style={{
                        backgroundColor: `${color}15`,
                        color,
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {modelName(m)}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Custom Presets */}
      {Object.keys(customPresets).length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-[0.08em]" style={{ color: "var(--color-text-secondary)" }}>
              My Presets
            </span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(customPresets).map(([key, p]) => (
              <div
                key={key}
                className="rounded-lg border p-3 flex items-center justify-between"
                style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-elevated)" }}
              >
                <div>
                  <span className="text-[11px] font-semibold" style={{ color: "var(--color-text-primary)" }}>
                    {p.name}
                  </span>
                  {p.subtitle && (
                    <p className="text-[9px] mt-0.5" style={{ color: "var(--color-text-muted)" }}>
                      {p.subtitle}
                    </p>
                  )}
                  <span className="text-[8px] font-mono" style={{ color: "var(--color-text-muted)" }}>
                    Saved {p.date}
                  </span>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); removeCustomPreset(key); }}
                  className="rounded p-1"
                  style={{ color: "var(--color-accent-danger)", cursor: "pointer" }}
                  title="Delete preset"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Models from current selection — appears after applying a preset or manually */}
      {selectedModels.length > 0 && (
        <div className="flex justify-end pt-1">
          <button
            onClick={handleDeployCustom}
            disabled={!canDeploy || isSubmitting}
            className="rounded-md px-8 py-3 text-sm font-bold uppercase transition-all duration-300 hover:brightness-110"
            style={{
              background: canDeploy
                ? "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)"
                : "var(--color-glass-border)",
              color: canDeploy ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              letterSpacing: "0.08em",
              cursor: canDeploy ? "pointer" : "not-allowed",
              boxShadow: canDeploy ? "0 0 24px rgba(0,229,255,0.2)" : "none",
              opacity: isSubmitting ? 0.7 : 1,
            }}
          >
            {isSubmitting ? "Submitting..." : "Deploy Backtest"}
          </button>
        </div>
      )}
    </div>
  );
}
