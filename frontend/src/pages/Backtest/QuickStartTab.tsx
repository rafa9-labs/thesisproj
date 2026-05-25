import { useCallback } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { QUICK_START_CATEGORIES } from "@/lib/constants";
import { modelDescriptions } from "@/lib/tokens";
import { Trash2, Bug, Cpu, Network, Layers, Bot } from "lucide-react";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface Props {}

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

export function QuickStartTab(_props: Props) {
  const applyQuickPreset = useBacktestStore((s) => s.applyQuickPreset);
  const removeCustomPreset = useBacktestStore((s) => s.removeCustomPreset);
  const customPresets = useBacktestStore((s) => s.customPresets);

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
          <div key={cat.key} className="mb-10">
            {/* Category header */}
            <div className="flex items-center gap-1.5 mb-4">
              <span style={{ color: catColor }}>{CATEGORY_ICONS[cat.key]}</span>
              <span
                className="text-[11px] font-semibold uppercase tracking-[0.1em]"
                style={{ color: "#FFFFFF" }}
              >
                {cat.label}
              </span>
            </div>

            {/* Options grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs ? `${(opt.estMinutes / 60).toFixed(0)}h` : `${opt.estMinutes}min`;
                return (
                  <div
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="bg-[#1E1E1E] border border-[#333333] rounded-xl p-6 flex flex-col justify-between cursor-pointer transition-colors duration-150 hover:border-[#A8E063]"
                  >
                    {/* Header row */}
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-white text-lg font-bold tracking-wide leading-tight">
                        {opt.label}
                      </span>
                      <span className="text-[#9CA3AF] text-sm font-medium whitespace-nowrap ml-2 mt-0.5">
                        ~{timeStr}
                      </span>
                    </div>

                    {/* Description */}
                    <p className="text-[#D1D5DB] text-sm leading-relaxed flex-1 mb-0">
                      {opt.subtitle}
                    </p>

                    {/* Model badges */}
                    <div className="mt-5 flex flex-wrap gap-2">
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="inline-flex items-center px-2.5 py-1 bg-[#2A2A2A] rounded-md text-xs font-medium text-gray-300 uppercase tracking-wider"
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span className="inline-flex items-center text-xs text-[#9CA3AF]">
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


    </div>
  );
}
