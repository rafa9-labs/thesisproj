import { useState } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelDescriptions } from "@/lib/tokens";
import { STUDY_PRESETS } from "@/lib/constants";

export function ConfigSummaryBar() {
  const s = useBacktestStore();
  const pair = s.pair ?? "EURUSD";
  const tf = s.timeframe ?? "H1";
  const models = s.selectedModels ?? [];
  const activePreset = s.activePreset;
  const presetInfo = activePreset
    ? STUDY_PRESETS[activePreset as keyof typeof STUDY_PRESETS]
    : null;
  const hpoIntensity = s.hpoIntensity ?? "quick";
  const nTrials = s.nTrials ?? 10;
  const repeats = s.repeats ?? 1;
  const trainMonths = s.trainMonths ?? 36;
  const testMonths = s.testMonths ?? 1;
  const applyStudyPreset = s.applyStudyPreset;

  const [presetOpen, setPresetOpen] = useState(false);

  const totalTrials = nTrials * repeats;
  const modelNames = models.map(
    (m) => (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m,
  );
  const modelText =
    modelNames.length === 0
      ? "—"
      : modelNames.length === 1
        ? modelNames[0]
        : `${modelNames[0]} +${modelNames.length - 1}`;

  const hasModels = models.length > 0;
  const studyLabel = presetInfo?.label ?? hpoIntensity;

  const readouts: { label: string; value: string; active?: boolean }[] = [
    { label: "ASSET", value: `${pair} · ${tf}` },
    { label: "MODELS", value: modelText, active: hasModels },
    { label: "HPO", value: `${nTrials} tri \u00d7 ${repeats} = ${totalTrials}` },
    { label: "WALK-FWD", value: `${trainMonths}mo / ${testMonths}mo` },
    { label: "STUDY", value: studyLabel, active: !!presetInfo },
  ];

  const presetEntries = Object.entries(STUDY_PRESETS) as [
    string,
    (typeof STUDY_PRESETS)[keyof typeof STUDY_PRESETS],
  ][];

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-(--color-glass-border) px-6 pb-4">
      {readouts.map(({ label, value, active }, i) => (
        <div
          key={label}
          className="flex flex-col"
          style={{
            paddingLeft: i === 0 ? 0 : 16,
            paddingRight: 16,
          }}
        >
          <span
            className="text-(--color-text-muted)"
            style={{
              fontSize: 9,
              fontWeight: 600,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              opacity: 0.7,
              marginBottom: 4,
            }}
          >
            {label}
          </span>
          <span
            className="font-mono"
            style={{
              fontSize: 12,
              fontWeight: 500,
              color: active ? "var(--color-brand)" : "var(--color-text-primary)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
            title={value}
          >
            {value}
          </span>
        </div>
      ))}

      <div className="relative ml-auto">
        <button
          onClick={() => setPresetOpen((v) => !v)}
          className="flex shrink-0 items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-(--color-glass) px-3 py-1.5 text-[10px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
        >
          Load Preset
        </button>
        {presetOpen && (
          <div
            className="absolute right-0 z-40 mt-1 w-56 rounded-lg border border-(--color-glass-border) bg-(--color-elevated) shadow-2xl"
            style={{ boxShadow: "0 12px 32px rgba(0,0,0,0.4)" }}
            onMouseLeave={() => setPresetOpen(false)}
          >
            {presetEntries.map(([key, preset]) => (
              <button
                key={key}
                onClick={() => {
                  applyStudyPreset(key);
                  setPresetOpen(false);
                }}
                className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-xs transition-colors hover:bg-(--color-glass-hover)"
              >
                <span
                  className="inline-block h-2 w-2 shrink-0 rounded-full"
                  style={{
                    backgroundColor:
                      preset.badgeColor === "green"
                        ? "var(--color-accent-success)"
                        : preset.badgeColor === "yellow"
                          ? "var(--color-accent-warning)"
                          : preset.badgeColor === "orange"
                            ? "#f59e0b"
                            : "var(--color-accent-danger)",
                  }}
                />
                <div className="flex flex-col gap-0.5">
                  <span className="font-semibold text-(--color-text-primary)">{preset.label}</span>
                  <span className="text-[9px] text-(--color-text-dim)">
                    {preset.desc.slice(0, 60)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
