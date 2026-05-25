import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelDescriptions } from "@/lib/tokens";
import { STUDY_PRESETS } from "@/lib/constants";
import { Eye } from "lucide-react";

const labelStyle: React.CSSProperties = {
  color: "var(--color-text-muted)",
  fontSize: 9,
  fontWeight: 600,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  marginBottom: 3,
};
const valueStyle: React.CSSProperties = {
  color: "var(--color-text-primary)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  lineHeight: "1.3",
};
const presetColorMap: Record<string, string> = {
  green: "var(--color-accent-success)",
  yellow: "var(--color-accent-warning)",
  orange: "var(--color-accent)",
  red: "var(--color-accent-danger)",
};

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
  const modelNames = models.length > 0
    ? models.map((m) => (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m)
    : [];
  const modelText = modelNames.length > 0 ? modelNames.join(", ") : "No models selected";
  const showVertical = modelNames.length > 2 || modelText.length > 24;

  const colBorder: React.CSSProperties = { borderLeft: "1px solid var(--color-border)" };
  const presetColor = presetInfo ? presetColorMap[presetInfo.badgeColor] ?? "var(--color-text-primary)" : "var(--color-text-primary)";

  return (
    <div className="flex flex-col gap-2">
      {/* Title row */}
      <div className="flex items-center gap-1.5">
        <Eye size={12} style={{ color: "var(--color-text-muted)" }} />
        <span
          className="text-[10px] font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Backtest Preview
        </span>
      </div>

      {/* Summary bar */}
      <div
        className="flex rounded-xl border px-3 py-3"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
        }}
      >
        {/* Asset */}
        <div className="flex flex-col items-center justify-center" style={{ flex: 1 }}>
          <span style={labelStyle}>Asset</span>
          <span style={valueStyle}>{pair}·{tf}</span>
        </div>

        {/* Models */}
        <div className="flex flex-col items-center justify-center" style={{ flex: 1, ...colBorder }}>
          <span style={labelStyle}>Models</span>
          {showVertical ? (
            <div className="flex flex-col items-center gap-0.5" style={{ maxHeight: 48, overflowY: "auto" }}>
              {modelNames.length > 0 ? modelNames.map((name, i) => (
                <span key={i} style={{ ...valueStyle, color: "var(--color-brand)", fontSize: 10, lineHeight: 1.2 }}>
                  {name}
                </span>
              )) : (
                <span style={{ ...valueStyle, color: "var(--color-text-muted)", fontSize: 10 }}>
                  No models selected
                </span>
              )}
            </div>
          ) : (
            <span
              style={{ ...valueStyle, color: "var(--color-brand)", maxWidth: "90%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
              title={modelText}
            >
              {modelText}
            </span>
          )}
        </div>

        {/* HPO */}
        <div className="flex flex-col items-center justify-center" style={{ flex: 1, ...colBorder }}>
          <span style={labelStyle}>HPO</span>
          <span style={valueStyle}>{nTrials} tri × {repeats} run{repeats > 1 ? "s" : ""} = {totalTrials}</span>
        </div>

        {/* Walk-Forward */}
        <div className="flex flex-col items-center justify-center" style={{ flex: 1, ...colBorder }}>
          <span style={labelStyle}>Walk-Fwd</span>
          <span style={valueStyle}>Train {trainMonths}mo · Test {testMonths}mo</span>
        </div>

        {/* Study */}
        <div className="flex flex-col items-center justify-center" style={{ flex: 1, ...colBorder }}>
          <span style={labelStyle}>Study</span>
          <span style={{ ...valueStyle, color: presetColor }}>
            {presetInfo?.label ?? hpoIntensity}
          </span>
        </div>
      </div>
    </div>
  );
}
