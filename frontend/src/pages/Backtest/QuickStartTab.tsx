import { useCallback } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { QUICK_START_CATEGORIES } from "@/lib/constants";
import { modelDescriptions } from "@/lib/tokens";
import { Trash2, Bug, Cpu, Network, Layers, Bot } from "lucide-react";

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
    <div
      className="flex flex-col rounded-xl border p-6"
      style={{
        backgroundColor: "var(--color-glass)",
        borderColor: "var(--color-glass-border)",
        backdropFilter: "blur(12px)",
      }}
    >
      {QUICK_START_CATEGORIES.map((cat, idx) => {
        const catColor = CATEGORY_COLORS[cat.key] ?? "var(--color-text-muted)";

        return (
          <div key={cat.key}>
            {idx > 0 && (
              <div className="my-8" style={{ borderTop: "1px solid var(--color-glass-border)" }} />
            )}

            {/* Category header with left accent bar */}
            <div className="flex items-center gap-2 mt-10 mb-4 px-1">
              <div
                className="h-4 w-[2px] rounded-full shrink-0"
                style={{ backgroundColor: catColor }}
              />
              <span style={{ color: catColor }}>{CATEGORY_ICONS[cat.key]}</span>
              <span
                className="text-[11px] font-semibold uppercase tracking-[0.1em]"
                style={{ color: "var(--color-text-primary)" }}
              >
                {cat.label}
              </span>
              <span className="text-[10px] ml-1" style={{ color: "var(--color-text-muted)" }}>
                {cat.options.length} presets
              </span>
            </div>

            {/* Wider cards — 3 columns max */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs
                  ? `${(opt.estMinutes / 60).toFixed(0)}h`
                  : `${opt.estMinutes}min`;

                return (
                  <div
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="rounded-lg border p-5 flex flex-col cursor-pointer transition-all duration-150"
                    style={{
                      backgroundColor: "var(--color-elevated)",
                      borderColor: "var(--color-glass-border)",
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLDivElement).style.borderColor = "var(--color-brand)";
                      (e.currentTarget as HTMLDivElement).style.boxShadow = "0 0 0 1px rgba(0,229,255,0.15)";
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLDivElement).style.borderColor = "var(--color-glass-border)";
                      (e.currentTarget as HTMLDivElement).style.boxShadow = "none";
                    }}
                  >
                    {/* Title row: label + est time pill */}
                    <div className="flex items-start justify-between gap-3 mb-4">
                      <span
                        className="text-[13px] font-semibold tracking-wide"
                        style={{ color: "var(--color-text-primary)", lineHeight: 1.4 }}
                      >
                        {opt.label}
                      </span>
                      {(opt as any).isNew && (
                        <span
                          className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase tracking-[0.08em]"
                          style={{
                            color: "#A8E063",
                            backgroundColor: "rgba(168, 224, 99, 0.1)",
                            border: "1px solid rgba(168, 224, 99, 0.2)",
                          }}
                        >
                          NEW
                        </span>
                      )}
                      <span
                        className="shrink-0 inline-flex items-center px-2 py-0.5 rounded text-[10px] tabular-nums font-medium"
                        style={{
                          color: "var(--color-brand)",
                          backgroundColor: "rgba(0,229,255,0.08)",
                          border: "1px solid rgba(0,229,255,0.2)",
                          fontFamily: "var(--font-mono)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {timeStr}
                      </span>
                    </div>

                    {/* Stat pills */}
                    <div className="flex flex-wrap gap-1.5">
                      <StatPill label="train" value={`${opt.trainMonths}mo`} />
                      <StatPill label="test" value={`${opt.testMonths}mo`} />
                      <StatPill label="trials" value={`${opt.nTrials}`} />
                      <StatPill label="hpo" value={opt.hpoIntensity.toUpperCase()} accent />
                    </div>

                    {/* Model badges */}
                    <div className="mt-4 flex flex-wrap gap-2 pt-3" style={{ borderTop: "1px solid var(--color-glass-border)" }}>
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="inline-flex items-center px-2.5 py-1 rounded-md text-[10px] font-medium uppercase tracking-wider"
                          style={{
                            color: "var(--color-text-secondary)",
                            backgroundColor: "var(--color-glass)",
                            border: "1px solid var(--color-glass-border)",
                          }}
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span
                          className="inline-flex items-center text-[10px]"
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
          <div className="my-8" style={{ borderTop: "1px solid var(--color-glass-border)" }} />
          <div className="flex items-center gap-2 mt-8 mb-4 px-1">
            <span
              className="text-[11px] font-semibold uppercase tracking-[0.1em]"
              style={{ color: "var(--color-text-primary)" }}
            >
              My Presets
            </span>
            <span className="text-[10px] ml-1" style={{ color: "var(--color-text-muted)" }}>
              {Object.keys(customPresets).length} saved
            </span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(customPresets).map(([key, p]) => (
              <div
                key={key}
                className="rounded-lg border p-6 flex items-center justify-between"
                style={{
                  borderColor: "var(--color-glass-border)",
                  backgroundColor: "var(--color-elevated)",
                }}
              >
                <div>
                  <span
                    className="text-[11px] font-semibold"
                    style={{ color: "var(--color-text-primary)" }}
                  >
                    {p.name}
                  </span>
                  {p.subtitle && (
                    <p
                      className="text-[9px] mt-0.5"
                      style={{ color: "var(--color-text-muted)" }}
                    >
                      {p.subtitle}
                    </p>
                  )}
                  <span
                    className="text-[8px] font-mono"
                    style={{ color: "var(--color-text-muted)" }}
                  >
                    Saved {p.date}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeCustomPreset(key);
                  }}
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

// ── Stat Pill ─────────────────────────────────────────────────────────────────

function StatPill({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded px-2 py-0.5"
      style={{
        backgroundColor: accent
          ? "rgba(0,229,255,0.08)"
          : "var(--color-elevated)",
        border: `1px solid ${accent ? "rgba(0,229,255,0.2)" : "var(--color-glass-border)"}`,
      }}
    >
      <span
        className="text-[8px] uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
      >
        {label}
      </span>
      <span
        className="text-[10px] font-semibold tabular-nums"
        style={{
          color: accent ? "var(--color-brand)" : "var(--color-text-primary)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {value}
      </span>
    </span>
  );
}
