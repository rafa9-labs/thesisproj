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
      className="rounded-sm border bg-(--color-glass) p-5 backdrop-blur-[12px]"
      style={{ borderColor: "rgba(0,229,255,0.15)" }}
    >
      <div className="mb-3 flex items-center gap-2">
        <span className="text-[11px] font-medium tracking-[0.12em] text-(--color-brand) uppercase">
          Quick Start
        </span>
        <span className="text-[11px] font-light text-(--color-text-muted)">
          — pre-fill settings, then tweak &amp; deploy
        </span>
      </div>
      <div className="flex gap-3">
        {PRESETS.map((p) => (
          <button
            key={p.name}
            onClick={() => applyPreset(p)}
            className="flex flex-col items-start gap-1 rounded-sm border border-(--color-glass-border) bg-(--color-glass-hover) px-4 py-3 transition-all duration-300 hover:border-[var(--color-border-active)]"
            className="cursor-pointer"
            style={{ minWidth: "180px" }}
          >
            <span className="text-sm font-medium text-(--color-text-primary)">{p.label}</span>
            <span className="text-[11px] font-light text-(--color-text-muted)">
              {p.description}
            </span>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {p.models.map((m) => (
                <span
                  key={m}
                  className="rounded px-1.5 py-0.5 text-[10px] font-medium tracking-[0.04em] text-(--color-accent-classical) uppercase"
                  style={{
                    backgroundColor: "rgba(34,211,238,0.12)",
                    border: "1px solid rgba(34,211,238,0.15)",
                  }}
                >
                  {m}
                </span>
              ))}
              <span
                className="rounded bg-(--color-glass-hover) px-1.5 py-0.5 text-[10px] font-medium text-(--color-text-muted)"
                style={{ border: "1px solid var(--color-glass-border)" }}
              >
                {p.pair}
              </span>
              <span
                className="rounded bg-(--color-glass-hover) px-1.5 py-0.5 text-[10px] font-medium text-(--color-text-muted)"
                style={{ border: "1px solid var(--color-glass-border)" }}
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
