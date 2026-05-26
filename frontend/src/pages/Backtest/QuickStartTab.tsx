import { useCallback, useState } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { QUICK_START_CATEGORIES } from "@/lib/constants";
import { modelDescriptions } from "@/lib/tokens";
import { Trash2, Bug, Cpu, Network, Layers, Bot, Info } from "lucide-react";

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  debug: <Bug size={11} />,
  classical: <Cpu size={11} />,
  deep: <Network size={11} />,
  ensemble: <Layers size={11} />,
  rl: <Bot size={11} />,
};

const CATEGORY_COLORS: Record<string, string> = {
  debug:    "#787B86",
  classical: "#22D3EE",
  deep:     "#A78BFA",
  ensemble: "#EC4899",
  rl:       "#F59E0B",
};

export function QuickStartTab() {
  const applyQuickPreset  = useBacktestStore((s) => s.applyQuickPreset);
  const removeCustomPreset = useBacktestStore((s) => s.removeCustomPreset);
  const customPresets      = useBacktestStore((s) => s.customPresets);
  const [tooltip, setTooltip] = useState<{ key: string; x: number; y: number } | null>(null);

  const handlePreset = useCallback((key: string) => {
    applyQuickPreset(key);
  }, [applyQuickPreset]);

  const modelBadgeName = (m: string) =>
    (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m;

  // Flatten all presets for tooltip lookup
  const allPresets = QUICK_START_CATEGORIES.flatMap((c) => c.options);
  const tooltipPreset = tooltip ? allPresets.find((p) => p.key === tooltip.key) : null;

  return (
    <div
      className="flex flex-col"
      style={{ paddingBottom: 8 }}
      onMouseLeave={() => setTooltip(null)}
    >
      {QUICK_START_CATEGORIES.map((cat, idx) => {
        const catColor = CATEGORY_COLORS[cat.key] ?? "#787B86";

        return (
          <div key={cat.key}>
            {/* Category separator */}
            {idx > 0 && (
              <div
                className="mx-0"
                style={{ height: 1, backgroundColor: "#2A2E39", margin: "12px 0" }}
              />
            )}

            {/* Category header row */}
            <div
              className="flex items-center gap-2 px-3 py-1.5"
              style={{ backgroundColor: "#1A1D27" }}
            >
              <span style={{ color: catColor, display: "flex", alignItems: "center" }}>
                {CATEGORY_ICONS[cat.key]}
              </span>
              <span
                className="text-[10px] font-semibold uppercase tracking-[0.12em]"
                style={{ color: catColor }}
              >
                {cat.label}
              </span>
              <span
                className="text-[10px] ml-1"
                style={{ color: "#4A5568", fontFamily: "var(--font-mono)" }}
              >
                {cat.options.length} presets
              </span>
            </div>

            {/* Preset rows */}
            {cat.options.map((opt, rowIdx) => {
              const hrs = opt.estMinutes >= 120;
              const timeStr = hrs
                ? `${(opt.estMinutes / 60).toFixed(0)}h`
                : `${opt.estMinutes}min`;
              const isOdd = rowIdx % 2 === 1;

              return (
                <div
                  key={opt.key}
                  className="flex items-center gap-3 px-3 transition-colors duration-100"
                  style={{
                    height: 40,
                    backgroundColor: isOdd ? "rgba(255,255,255,0.015)" : "transparent",
                    borderBottom: "1px solid #1E222D",
                  }}
                  onMouseEnter={() => setTooltip(null)}
                >
                  {/* Col 1: Name + info icon */}
                  <div className="flex items-center gap-1.5" style={{ minWidth: 200, maxWidth: 200 }}>
                    <span
                      className="text-[12px] font-medium truncate"
                      style={{ color: "#D1D4DC" }}
                    >
                      {opt.label}
                    </span>
                    <button
                      className="shrink-0 flex items-center justify-center rounded transition-colors"
                      style={{ color: "#4A5568", cursor: "default" }}
                      onMouseEnter={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setTooltip({ key: opt.key, x: rect.left, y: rect.bottom + 4 });
                      }}
                      onMouseLeave={() => setTooltip(null)}
                      tabIndex={-1}
                    >
                      <Info size={11} />
                    </button>
                  </div>

                  {/* Col 2: Model badges */}
                  <div className="flex flex-1 items-center gap-1.5 flex-wrap">
                    {opt.models.slice(0, 4).map((m) => (
                      <span
                        key={m}
                        className="inline-flex items-center px-1.5 rounded text-[9px] font-medium uppercase tracking-wider"
                        style={{
                          backgroundColor: "#1E222D",
                          color: "#787B86",
                          border: "1px solid #2A2E39",
                          lineHeight: "18px",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {modelBadgeName(m)}
                      </span>
                    ))}
                    {opt.models.length > 4 && (
                      <span
                        className="text-[9px]"
                        style={{ color: "#4A5568", fontFamily: "var(--font-mono)" }}
                      >
                        +{opt.models.length - 4}
                      </span>
                    )}
                  </div>

                  {/* Col 3: Time + Select */}
                  <div className="flex items-center gap-3 shrink-0">
                    <span
                      className="text-[10px] tabular-nums"
                      style={{ color: "#4A5568", fontFamily: "var(--font-mono)", minWidth: 46, textAlign: "right" }}
                    >
                      {timeStr}
                    </span>
                    <button
                      onClick={() => handlePreset(opt.key)}
                      className="rounded text-[10px] font-semibold uppercase tracking-wider transition-colors duration-100"
                      style={{
                        backgroundColor: "#1E222D",
                        border: "1px solid #2A2E39",
                        color: "#787B86",
                        padding: "3px 10px",
                        cursor: "pointer",
                        whiteSpace: "nowrap",
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLButtonElement).style.borderColor = "#089981";
                        (e.currentTarget as HTMLButtonElement).style.color = "#089981";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLButtonElement).style.borderColor = "#2A2E39";
                        (e.currentTarget as HTMLButtonElement).style.color = "#787B86";
                      }}
                    >
                      Select
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}

      {/* Custom Presets */}
      {Object.keys(customPresets).length > 0 && (
        <div>
          <div style={{ height: 1, backgroundColor: "#2A2E39", margin: "12px 0" }} />
          <div
            className="flex items-center gap-2 px-3 py-1.5"
            style={{ backgroundColor: "#1A1D27" }}
          >
            <span className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: "#787B86" }}>
              My Presets
            </span>
            <span className="text-[10px] ml-1" style={{ color: "#4A5568", fontFamily: "var(--font-mono)" }}>
              {Object.keys(customPresets).length} saved
            </span>
          </div>
          {Object.entries(customPresets).map(([key, p], rowIdx) => (
            <div
              key={key}
              className="flex items-center gap-3 px-3 transition-colors duration-100"
              style={{
                height: 40,
                backgroundColor: rowIdx % 2 === 1 ? "rgba(255,255,255,0.015)" : "transparent",
                borderBottom: "1px solid #1E222D",
              }}
            >
              <div className="flex items-center gap-1.5 flex-1" style={{ minWidth: 0 }}>
                <span className="text-[12px] font-medium truncate" style={{ color: "#D1D4DC" }}>
                  {p.name}
                </span>
                {p.subtitle && (
                  <span className="text-[10px] truncate" style={{ color: "#4A5568" }}>
                    — {p.subtitle}
                  </span>
                )}
              </div>
              <span
                className="text-[9px] tabular-nums shrink-0"
                style={{ color: "#4A5568", fontFamily: "var(--font-mono)" }}
              >
                {p.date}
              </span>
              <button
                onClick={() => removeCustomPreset(key)}
                className="shrink-0 rounded p-1 transition-colors"
                style={{ color: "#4A5568", cursor: "pointer" }}
                onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "#EF4444")}
                onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.color = "#4A5568")}
                title="Delete preset"
              >
                <Trash2 size={11} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Floating tooltip */}
      {tooltip && tooltipPreset && (
        <div
          className="fixed z-50 rounded border pointer-events-none"
          style={{
            top: tooltip.y,
            left: Math.min(tooltip.x, typeof window !== "undefined" ? window.innerWidth - 280 : tooltip.x),
            width: 260,
            backgroundColor: "#1E222D",
            borderColor: "#2A2E39",
            padding: "10px 12px",
            boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
          }}
        >
          <p
            className="text-[10px] font-semibold mb-1.5 uppercase tracking-wider"
            style={{ color: "#D1D4DC" }}
          >
            {tooltipPreset.label}
          </p>
          {tooltipPreset.subtitle.split("\n\n").map((para, i) => (
            <p
              key={i}
              className="text-[10px] leading-relaxed"
              style={{ color: "#787B86", marginTop: i > 0 ? 6 : 0 }}
            >
              {para}
            </p>
          ))}
          <div
            className="mt-2 pt-2 flex items-center gap-2"
            style={{ borderTop: "1px solid #2A2E39" }}
          >
            <span className="text-[9px] uppercase tracking-wider" style={{ color: "#4A5568" }}>
              Est.
            </span>
            <span
              className="text-[10px] tabular-nums"
              style={{ color: "#089981", fontFamily: "var(--font-mono)" }}
            >
              {tooltipPreset.estMinutes >= 120
                ? `${(tooltipPreset.estMinutes / 60).toFixed(0)}h`
                : `${tooltipPreset.estMinutes}min`}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
