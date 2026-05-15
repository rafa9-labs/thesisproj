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
  const modelLabels = models.length > 0
    ? models.map((m) => (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m).join(", ")
    : "No models selected";

  const sectionStyle: React.CSSProperties = {
    borderRight: "1px solid var(--color-border)",
    paddingRight: 14,
  };

  return (
    <div
      className="flex items-center gap-4 rounded-xl border px-5 py-3 text-[11px] overflow-x-auto"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-glass)",
      }}
    >
      {/* Asset */}
      <div className="flex items-center gap-2 shrink-0" style={sectionStyle}>
        <span className="font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          Asset
        </span>
        <span className="font-mono" style={{ color: "var(--color-text-primary)" }}>
          {pair}·{tf}
        </span>
      </div>

      {/* Models */}
      <div className="flex items-center gap-2 shrink-0" style={sectionStyle}>
        <span className="font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          Models
        </span>
        <span className="font-mono" style={{ color: "var(--color-brand)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={modelLabels}>
          {modelLabels}
        </span>
      </div>

      {/* HPO */}
      <div className="flex items-center gap-2 shrink-0" style={sectionStyle}>
        <span className="font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          HPO
        </span>
        <span className="font-mono" style={{ color: "var(--color-text-primary)" }}>
          {nTrials} tri × {repeats} run{repeats > 1 ? "s" : ""} = {totalTrials} total
        </span>
      </div>

      {/* Walk-Forward */}
      <div className="flex items-center gap-2 shrink-0" style={sectionStyle}>
        <span className="font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          Walk-Fwd
        </span>
        <span className="font-mono" style={{ color: "var(--color-text-primary)" }}>
          Train {trainMonths}mo · Test {testMonths}mo
        </span>
      </div>

      {/* Preset / Intensity */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          Study
        </span>
        <span className="font-mono" style={{ color: presetInfo?.badgeColor ? `var(--color-${presetInfo.badgeColor === "green" ? "accent-success" : presetInfo.badgeColor === "yellow" ? "accent-warning" : presetInfo.badgeColor === "orange" ? "accent" : "accent-danger"})` : "var(--color-text-primary)" }}>
          {presetInfo?.label ?? hpoIntensity}
        </span>
      </div>
    </div>
  );
}
