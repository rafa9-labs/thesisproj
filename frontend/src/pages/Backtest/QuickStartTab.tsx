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
  deep: "var(--color-accent-deep)",
  ensemble: "var(--color-accent-ensemble)",
  rl: "var(--color-accent-rl)",
};

// HPO intensity → readable label
const HPO_LABELS: Record<string, string> = {
  light: "Light",
  quick: "Quick",
  standard: "Std",
  deep: "Deep",
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
            {/* Divider between categories */}
            {idx > 0 && (
              <div className="my-10" style={{ borderTop: "1px solid var(--color-glass-border)" }} />
            )}

            {/* Category header */}
            <div className="flex items-center gap-2 mb-4 px-1">
              <div
                className="rounded-full shrink-0"
                style={{ width: 7, height: 7, backgroundColor: catColor }}
              />
              <span
                className="text-[10px] font-medium uppercase tracking-[0.14em] whitespace-nowrap"
                style={{ color: "var(--color-text-secondary)" }}
              >
                {cat.label}
              </span>
              <div className="h-px flex-1" style={{ backgroundColor: "var(--color-glass-border)" }} />
              <span
                className="text-[10px] shrink-0"
                style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
              >
                {cat.options.length} presets
              </span>
            </div>

            {/* Cards grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {cat.options.map((opt) => {
                const hrs = opt.estMinutes >= 120;
                const timeStr = hrs
                  ? `${(opt.estMinutes / 60).toFixed(0)}h`
                  : `${opt.estMinutes}min`;

                return (
                  <button
                    key={opt.key}
                    onClick={() => handlePreset(opt.key)}
                    className="flex flex-col gap-4 rounded-lg border p-5 text-left transition-all duration-150"
                    style={{
                      borderColor: "var(--color-glass-border)",
                      backgroundColor: "var(--color-glass)",
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--color-brand)";
                      (e.currentTarget as HTMLButtonElement).style.backgroundColor = "rgba(0,229,255,0.04)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--color-glass-border)";
                      (e.currentTarget as HTMLButtonElement).style.backgroundColor = "var(--color-glass)";
                    }}
                  >
                    {/* Title row */}
                    <div className="flex flex-col gap-1.5">
                      <span
                        className="text-[13px] font-semibold tracking-wide"
                        style={{ color: "var(--color-text-primary)", lineHeight: 1.35 }}
                      >
                        {opt.label}
                      </span>
                      <span
                        className="text-[10px] tabular-nums"
                        style={{
                          color: "var(--color-brand)",
                          fontFamily: "var(--font-mono)",
                          letterSpacing: "0.04em",
                        }}
                      >
                        est. {timeStr}
                      </span>
                    </div>

                    {/* Stat pills */}
                    <div className="flex flex-wrap gap-1.5">
                      {/* Train window */}
                      <StatPill
                        label="train"
                        value={`${opt.trainMonths}mo`}
                      />
                      {/* Test window */}
                      <StatPill
                        label="test"
                        value={`${opt.testMonths}mo`}
                      />
                      {/* HPO trials */}
                      <StatPill
                        label="trials"
                        value={`${opt.nTrials}`}
                      />
                      {/* HPO intensity */}
                      <StatPill
                        label="hpo"
                        value={HPO_LABELS[opt.hpoIntensity] ?? opt.hpoIntensity}
                        accent
                      />
                    </div>

                    {/* Model badges */}
                    <div
                      className="flex flex-wrap gap-1.5 pt-2.5"
                      style={{ borderTop: "1px solid var(--color-glass-border)" }}
                    >
                      {opt.models.slice(0, 3).map((m) => (
                        <span
                          key={m}
                          className="inline-flex items-center px-2 py-0.5 rounded text-[9px] font-medium uppercase tracking-wider"
                          style={{
                            backgroundColor: "var(--color-elevated)",
                            color: "var(--color-text-secondary)",
                            border: "1px solid var(--color-glass-border)",
                          }}
                        >
                          {modelName(m)}
                        </span>
                      ))}
                      {opt.models.length > 3 && (
                        <span
                          className="inline-flex items-center px-2 py-0.5 rounded text-[9px] font-medium"
                          style={{
                            backgroundColor: "var(--color-elevated)",
                            color: "var(--color-text-muted)",
                            border: "1px solid var(--color-glass-border)",
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          +{opt.models.length - 3}
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Custom Presets */}
      {Object.keys(customPresets).length > 0 && (
        <div>
          <div className="my-10" style={{ borderTop: "1px solid var(--color-glass-border)" }} />
          <div className="flex items-center gap-2 mb-4 px-1">
            <div
              className="rounded-full shrink-0"
              style={{ width: 7, height: 7, backgroundColor: "var(--color-text-muted)" }}
            />
            <span
              className="text-[10px] font-medium uppercase tracking-[0.14em]"
              style={{ color: "var(--color-text-secondary)" }}
            >
              My Presets
            </span>
            <div className="h-px flex-1" style={{ backgroundColor: "var(--color-glass-border)" }} />
            <span
              className="text-[10px] shrink-0"
              style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
            >
              {Object.keys(customPresets).length} saved
            </span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(customPresets).map(([key, p]) => (
              <div
                key={key}
                className="rounded-lg border p-4 flex items-center justify-between"
                style={{
                  borderColor: "var(--color-glass-border)",
                  backgroundColor: "var(--color-glass)",
                }}
              >
                <div className="flex flex-col gap-0.5">
                  <span
                    className="text-[12px] font-semibold"
                    style={{ color: "var(--color-text-primary)" }}
                  >
                    {p.name}
                  </span>
                  {p.subtitle && (
                    <p
                      className="text-[9px]"
                      style={{ color: "var(--color-text-muted)" }}
                    >
                      {p.subtitle}
                    </p>
                  )}
                  <span
                    className="text-[9px]"
                    style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
                  >
                    {p.date}
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
