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

  const handlePreset = useCallback(
    (key: string) => {
      applyQuickPreset(key);
    },
    [applyQuickPreset],
  );

  const modelName = (m: string) =>
    (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m;

  return (
    <div className="flex flex-col rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-6 backdrop-blur-[12px]">
      {QUICK_START_CATEGORIES.map((cat, idx) => {
        const catColor = CATEGORY_COLORS[cat.key] ?? "var(--color-text-muted)";

        return (
          <div key={cat.key}>
            {idx > 0 && <div className="my-8 border-t border-(--color-glass-border)" />}

            {/* Category header with left accent bar */}
            <div className="mt-10 mb-4 flex items-center gap-2 px-1">
              <div
                className="h-4 w-[2px] shrink-0 rounded-full"
                style={{ backgroundColor: catColor }}
              />
              <span style={{ color: catColor }}>{CATEGORY_ICONS[cat.key]}</span>
              <span className="text-[11px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
                {cat.label}
              </span>
              <span className="ml-1 text-[10px] text-(--color-text-muted)">
                {cat.options.length} presets
              </span>
            </div>

            {/* Wider cards — 3 columns max */}
            <div className="grid grid-cols-1 gap-8 md:grid-cols-2 xl:grid-cols-3">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs
                  ? `${(opt.estMinutes / 60).toFixed(0)}h`
                  : `${opt.estMinutes}min`;

                return (
                  <div
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="flex cursor-pointer flex-col rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-5 transition-all duration-150 hover:border-(--color-brand) hover:shadow-[0_0_0_1px_rgba(0,229,255,0.15)]"
                  >
                    {/* Title row: label + est time pill */}
                    <div className="mb-4 flex items-start justify-between gap-3">
                      <span
                        className="text-[13px] font-semibold tracking-wide text-(--color-text-primary)"
                        style={{ lineHeight: 1.4 }}
                      >
                        {opt.label}
                      </span>
                      {(opt as any).isNew && (
                        <span className="inline-flex shrink-0 items-center rounded border border-[rgba(168,224,99,0.2)] bg-[rgba(168,224,99,0.1)] px-1.5 py-0.5 text-[9px] font-semibold tracking-[0.08em] text-[#A8E063] uppercase">
                          NEW
                        </span>
                      )}
                      <span className="inline-flex shrink-0 items-center rounded border border-[rgba(0,229,255,0.2)] bg-[rgba(0,229,255,0.08)] px-2 py-0.5 font-mono text-[10px] font-medium whitespace-nowrap text-(--color-brand) tabular-nums">
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
                    <div className="mt-4 flex flex-wrap gap-2 border-t border-(--color-glass-border) pt-3">
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="inline-flex items-center rounded-md border border-(--color-glass-border) bg-(--color-glass) px-2.5 py-1 text-[10px] font-medium tracking-wider text-(--color-text-secondary) uppercase"
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span className="inline-flex items-center text-[10px] text-(--color-text-muted)">
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
          <div className="my-8 border-t border-(--color-glass-border)" />
          <div className="mt-8 mb-4 flex items-center gap-2 px-1">
            <span className="text-[11px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
              My Presets
            </span>
            <span className="ml-1 text-[10px] text-(--color-text-muted)">
              {Object.keys(customPresets).length} saved
            </span>
          </div>
          <div className="grid grid-cols-2 gap-6 lg:grid-cols-3">
            {Object.entries(customPresets).map(([key, p]) => (
              <div
                key={key}
                className="flex items-center justify-between rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-6"
              >
                <div>
                  <span className="text-[11px] font-semibold text-(--color-text-primary)">
                    {p.name}
                  </span>
                  {p.subtitle && (
                    <p className="mt-0.5 text-[9px] text-(--color-text-muted)">{p.subtitle}</p>
                  )}
                  <span className="font-mono text-[8px] text-(--color-text-muted)">
                    Saved {p.date}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeCustomPreset(key);
                  }}
                  className="rounded p-1 text-(--color-accent-danger)"
                  className="cursor-pointer"
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
        backgroundColor: accent ? "rgba(0,229,255,0.08)" : "var(--color-elevated)",
        border: `1px solid ${accent ? "rgba(0,229,255,0.2)" : "var(--color-glass-border)"}`,
      }}
    >
      <span className="font-mono text-[8px] tracking-[0.1em] text-(--color-text-muted) uppercase">
        {label}
      </span>
      <span
        className="font-mono text-[10px] font-semibold tabular-nums"
        style={{ color: accent ? "var(--color-brand)" : "var(--color-text-primary)" }}
      >
        {value}
      </span>
    </span>
  );
}
