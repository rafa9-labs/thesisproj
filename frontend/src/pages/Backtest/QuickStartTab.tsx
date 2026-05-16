import { useCallback } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { QUICK_START_CATEGORIES } from "@/lib/constants";
import { modelDescriptions } from "@/lib/tokens";
import { Trash2, Bug, Cpu, Network, Layers, Bot } from "lucide-react";
import { ParameterGuide } from "./ParameterGuide";

interface Props {
  onDeploy: () => void;
  canDeploy: boolean;
  isSubmitting: boolean;
}

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  debug: <Bug size={13} />,
  classical: <Cpu size={13} />,
  deep: <Network size={13} />,
  ensemble: <Layers size={13} />,
  rl: <Bot size={13} />,
};

const CATEGORY_COLORS: Record<string, string> = {
  debug: "var(--color-text-muted)",
  classical: "var(--color-brand)",
  deep: "var(--color-accent)",
  ensemble: "var(--color-accent-warning)",
  rl: "var(--color-accent-danger)",
};

export function QuickStartTab({ onDeploy, canDeploy, isSubmitting }: Props) {
  const applyQuickPreset = useBacktestStore((s) => s.applyQuickPreset);
  const removeCustomPreset = useBacktestStore((s) => s.removeCustomPreset);
  const customPresets = useBacktestStore((s) => s.customPresets);
  const selectedModels = useBacktestStore((s) => s.selectedModels);

  const handlePreset = useCallback((key: string) => {
    applyQuickPreset(key);
  }, [applyQuickPreset]);

  const modelName = (m: string) =>
    (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m;

  return (
    <div className="flex flex-col gap-5 pt-1">
      {/* Category blocks stacked vertically */}
      {QUICK_START_CATEGORIES.map((cat) => {
        const catColor = CATEGORY_COLORS[cat.key] ?? "var(--color-text-muted)";

        return (
          <div key={cat.key}>
            {/* Category header */}
            <div className="flex items-center gap-1.5 mb-2">
              <span style={{ color: catColor }}>{CATEGORY_ICONS[cat.key]}</span>
              <span
                className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                style={{ color: catColor }}
              >
                {cat.label}
              </span>
            </div>

            {/* Options in a horizontal row */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs ? `${(opt.estMinutes / 60).toFixed(0)}h` : `${opt.estMinutes}min`;
                return (
                  <div
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="rounded-lg border p-2.5 cursor-pointer transition-all duration-150 hover:brightness-110"
                    style={{
                      borderColor: catColor,
                      backgroundColor: `${catColor}06`,
                    }}
                  >
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.06em]" style={{ color: catColor }}>
                        {opt.label}
                      </span>
                      <span className="text-[8px] font-mono" style={{ color: "var(--color-text-muted)" }}>
                        ~{timeStr}
                      </span>
                    </div>
                    <p className="text-[9px] mb-1" style={{ color: "var(--color-text-muted)" }}>
                      {opt.subtitle}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="rounded px-1 py-0.5 text-[7px] font-semibold uppercase tracking-wider"
                          style={{
                            backgroundColor: `${catColor}12`,
                            color: catColor,
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span
                          className="text-[7px] font-mono"
                          style={{ color: "var(--color-text-muted)" }}
                        >
                          +{opt.models.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

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

      {/* Parameter Guide — collapsible per-model constraints */}
      <ParameterGuide />

      {/* Deploy */}
      {selectedModels.length > 0 && (
        <div className="flex justify-end pt-1">
          <button
            onClick={canDeploy ? onDeploy : undefined}
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
