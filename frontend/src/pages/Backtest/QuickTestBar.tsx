import { useBacktestStore } from "@/stores/useBacktestStore";
import type { HpoIntensity } from "@/api/schemas";

interface QuickTestPresetConfig {
  name: string;
  label: string;
  description: string;
  pair: string;
  timeframe: string;
  models: string[];
  months: number;
  hpo_intensity: HpoIntensity;
  seed: number;
}

const PRESETS: QuickTestPresetConfig[] = [
  {
    name: "validate",
    label: "Validate",
    description: "1 model, 1 month, light HPO (~30s)",
    pair: "EURUSD",
    timeframe: "H1",
    models: ["logistic"],
    months: 1,
    hpo_intensity: "light",
    seed: 42,
  },
  {
    name: "quick",
    label: "Quick Test",
    description: "2 models, 3 months, quick HPO (~3-5 min)",
    pair: "EURUSD",
    timeframe: "H1",
    models: ["logistic", "cnn"],
    months: 3,
    hpo_intensity: "quick",
    seed: 42,
  },
];

export function QuickTestBar() {
  const applyPreset = useBacktestStore((s) => s.applyPreset);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-accent)" }}
    >
      <div className="mb-3 flex items-center gap-2">
        <span
          className="text-xs font-bold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-accent)" }}
        >
          Quick Start
        </span>
        <span
          className="text-xs"
          style={{ color: "var(--color-text-muted)" }}
        >
          — pre-fill settings, then tweak & deploy
        </span>
      </div>
      <div className="flex gap-3">
        {PRESETS.map((p) => (
          <button
            key={p.name}
            onClick={() => applyPreset(p)}
            className="flex flex-col items-start gap-1 rounded-md border px-4 py-2.5 transition-colors duration-150"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-bg)",
              cursor: "pointer",
              minWidth: "180px",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--color-accent)";
              e.currentTarget.style.backgroundColor = "var(--color-surface)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--color-border)";
              e.currentTarget.style.backgroundColor = "var(--color-bg)";
            }}
          >
            <span
              className="text-sm font-semibold"
              style={{ color: "var(--color-text-primary)" }}
            >
              {p.label}
            </span>
            <span
              className="text-xs"
              style={{ color: "var(--color-text-muted)" }}
            >
              {p.description}
            </span>
            <div className="mt-1 flex gap-1.5 flex-wrap">
              {p.models.map((m) => (
                <span
                  key={m}
                  className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase"
                  style={{
                    backgroundColor: "var(--color-accent-classical)",
                    color: "var(--color-text-inverse)",
                  }}
                >
                  {m}
                </span>
              ))}
              <span
                className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: "var(--color-border)",
                  color: "var(--color-text-secondary)",
                }}
              >
                {p.pair}
              </span>
              <span
                className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: "var(--color-border)",
                  color: "var(--color-text-secondary)",
                }}
              >
                {p.hpo_intensity}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}