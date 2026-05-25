import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelDescriptions } from "@/lib/tokens";
import { STUDY_PRESETS } from "@/lib/constants";

export function ConfigSummaryBar() {
  const s = useBacktestStore();
  const pair = s.pair ?? "EURUSD";
  const tf = s.timeframe ?? "H1";
  const models = s.selectedModels ?? [];
  const activePreset = s.activePreset;
  const presetInfo = activePreset ? STUDY_PRESETS[activePreset as keyof typeof STUDY_PRESETS] : null;
  const hpoIntensity = s.hpoIntensity ?? "quick";
  const nTrials = s.nTrials ?? 10;
  const repeats = s.repeats ?? 1;
  const trainMonths = s.trainMonths ?? 36;
  const testMonths = s.testMonths ?? 1;

  const totalTrials = nTrials * repeats;
  const modelNames = models.map(
    (m) => (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m
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
    { label: "HPO", value: `${nTrials} tri × ${repeats} = ${totalTrials}` },
    { label: "WALK-FWD", value: `${trainMonths}mo / ${testMonths}mo` },
    { label: "STUDY", value: studyLabel, active: !!presetInfo },
  ];

  return (
    <div
      className="flex items-center"
      style={{
        borderBottom: "1px solid var(--color-glass-border)",
        paddingBottom: 16,
        gap: 0,
      }}
    >
      {readouts.map(({ label, value, active }, i) => (
        <div
          key={label}
          className="flex flex-col"
          style={{
            flex: 1,
            paddingLeft: i === 0 ? 0 : 24,
            paddingRight: 24,
          }}
        >
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--color-text-muted)",
              opacity: 0.7,
              marginBottom: 4,
            }}
          >
            {label}
          </span>
          <span
            style={{
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              fontWeight: 500,
              color: active
                ? "var(--color-brand)"
                : "var(--color-text-primary)",
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
    </div>
  );
}
