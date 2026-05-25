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
    <div className="flex flex-col py-6 px-2">
      {/* Category blocks stacked vertically, separated by full-width rules */}
      {QUICK_START_CATEGORIES.map((cat, idx) => {
        const catColor = CATEGORY_COLORS[cat.key] ?? "var(--color-text-muted)";

        return (
          <div key={cat.key} className="pt-12">
            {/* Divider between categories (not before the first) */}
            {idx > 0 && (
              <div
                className="mb-12"
                style={{ borderTop: "1px solid #333333" }}
              />
            )}

            {/* Category header */}
            <div className="flex items-center gap-2 mt-8 mb-4 px-1">

            {/* Options grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs ? `${(opt.estMinutes / 60).toFixed(0)}h` : `${opt.estMinutes}min`;
                return (
                  <div
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-6 flex flex-col justify-between cursor-pointer transition-colors duration-150 hover:border-[#A8E063] hover:bg-[#1E1E1E]"
                  >
                    {/* Top content */}
                    <div>
                      {/* Title */}
                      <span
                        className="text-[13px] font-semibold tracking-wide mb-1 block"
                        style={{ color: "#FFFFFF", lineHeight: 1.4 }}
                      >
                        {opt.label}
                      </span>

                      {/* Time estimate */}
                      <span
                        className="text-[10px] tabular-nums mb-4 block"
                        style={{
                          color: "#A8E063",
                          fontFamily: "var(--font-mono)",
                          letterSpacing: "0.04em",
                        }}
                      >
                        est. {timeStr}
                      </span>

                      {/* Description */}
                      <p
                        className="text-[11px]"
                        style={{ color: "#9CA3AF", lineHeight: 1.75 }}
                      >
                        {opt.subtitle}
                      </p>
                    </div>

                    {/* Model badges */}
                    <div className="mt-4 flex flex-wrap gap-2 pt-3 border-t border-[#252525]">
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="inline-flex items-center px-2.5 py-1 bg-[#2A2A2A] rounded-md text-[10px] font-medium text-gray-300 uppercase tracking-wider"
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span className="inline-flex items-center text-[10px]" style={{ color: "#6B7280" }}>
                          +{opt.models.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Options grid — generous gap, internal padding */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs ? `${(opt.estMinutes / 60).toFixed(0)}h` : `${opt.estMinutes}min`;
                return (
                  <div
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg px-5 py-5 flex flex-col cursor-pointer transition-colors duration-150 hover:border-[#A8E063] hover:bg-[#1E1E1E]"
                  >
                    {/* Title */}
                    <span
                      className="text-[13px] font-semibold tracking-wide mb-1"
                      style={{ color: "#FFFFFF", lineHeight: 1.4 }}
                    >
                      {opt.label}
                    </span>

                    {/* Time estimate — own line, clearly separated from title */}
                    <span
                      className="text-[10px] tabular-nums mb-4"
                      style={{
                        color: "#A8E063",
                        fontFamily: "var(--font-mono)",
                        letterSpacing: "0.04em",
                      }}
                    >
                      est. {timeStr}
                    </span>

                    {/* Description */}
                    <p
                      className="text-[11px] flex-1 mb-4"
                      style={{ color: "#9CA3AF", lineHeight: 1.75 }}
                    >
                      {opt.subtitle}
                    </p>

                    {/* Model badges */}
                    <div className="flex flex-wrap gap-1.5 pt-3 border-t border-[#252525]">
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="inline-flex items-center px-2 py-0.5 bg-[#252525] rounded text-[9px] font-medium uppercase tracking-wider"
                          style={{ color: "#D1D5DB", fontFamily: "var(--font-mono)" }}
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span className="text-[9px]" style={{ color: "#6B7280" }}>
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
        <div className="pt-12">
          <div className="mb-12" style={{ borderTop: "1px solid #333333" }} />
          <div className="flex items-center gap-2 mb-6 px-1">
            <span className="text-[11px] font-semibold uppercase tracking-[0.1em]" style={{ color: "#FFFFFF" }}>
              My Presets
            </span>
            <span className="text-[10px] ml-1" style={{ color: "#4B5563" }}>
              {Object.keys(customPresets).length} saved
            </span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-8">
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
