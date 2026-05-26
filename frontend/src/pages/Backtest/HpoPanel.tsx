import { useCallback, useState, useRef, useEffect } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { RANGES, SELECT_OPTIONS, STUDY_PRESETS } from "@/lib/constants";
import { ModelHyperparamsPanel } from "./ModelHyperparamsPanel";

// ─── Preset dot colors ────────────────────────────────────────────────────────
const PRESET_DOT: Record<string, string> = {
  green:  "#4B5563", // gray  – Diagnostic
  yellow: "#92400E", // amber – Exploratory
  orange: "#7C3D12", // orange – Validation
  red:    "#7F1D1D", // red   – Production
};
const PRESET_DOT_ACTIVE: Record<string, string> = {
  green:  "#6B7280",
  yellow: "#D97706",
  orange: "#EA580C",
  red:    "#EF4444",
};

// ─── Tooltip ──────────────────────────────────────────────────────────────────
function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div ref={ref} className="relative inline-flex items-center">
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen((p) => !p)}
        className="flex items-center justify-center rounded-full transition-colors"
        style={{
          width: 14,
          height: 14,
          fontSize: 9,
          color: "#4B5563",
          border: "1px solid #2A2E39",
          backgroundColor: "#1E222D",
          lineHeight: 1,
          fontFamily: "var(--font-mono)",
        }}
        aria-label="More info"
      >
        i
      </button>
      {open && (
        <div
          className="absolute z-50 rounded px-2.5 py-2 text-[10px] leading-relaxed shadow-lg"
          style={{
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            width: 220,
            backgroundColor: "#1E222D",
            border: "1px solid #2A2E39",
            color: "#9598A1",
            fontFamily: "var(--font-sans)",
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}

// ─── Compact number input ─────────────────────────────────────────────────────
function NumInput({
  value,
  min,
  max,
  step = 1,
  onChange,
  width = 72,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  width?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className="rounded border px-2 text-right focus:outline-none"
      style={{
        width,
        height: 28,
        fontSize: 11,
        fontFamily: "var(--font-mono)",
        backgroundColor: "#131722",
        borderColor: "#2A2E39",
        color: "#D1D4DC",
      }}
    />
  );
}

// ─── Compact select ───────────────────────────────────────────────────────────
function CompactSelect({
  value,
  options,
  onChange,
  width = 120,
}: {
  value: string;
  options: readonly { value: string; label: string }[];
  onChange: (v: string) => void;
  width?: number;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border px-2 focus:outline-none"
      style={{
        width,
        height: 28,
        fontSize: 11,
        fontFamily: "var(--font-sans)",
        backgroundColor: "#131722",
        borderColor: "#2A2E39",
        color: "#D1D4DC",
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

// ─── Minimal slider (quality gates) ──────────────────────────────────────────
function ThinSlider({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex items-center gap-2 flex-1">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 cursor-pointer appearance-none rounded-full"
        style={{
          height: 3,
          accentColor: "#2D4A47",
          background: `linear-gradient(to right, #3D6B68 0%, #3D6B68 ${pct}%, #2A2E39 ${pct}%, #2A2E39 100%)`,
        }}
      />
      <span
        className="tabular-nums"
        style={{ width: 40, fontSize: 10, fontFamily: "var(--font-mono)", color: "#D1D4DC", textAlign: "right" }}
      >
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// ─── Field row label + info icon ──────────────────────────────────────────────
function FieldLabel({ label, tip }: { label: string; tip: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="uppercase tracking-[0.08em] whitespace-nowrap"
        style={{ fontSize: 10, color: "#787B86", fontFamily: "var(--font-sans)" }}
      >
        {label}
      </span>
      <InfoTooltip text={tip} />
    </div>
  );
}

// ─── Section header ───────────────────────────────────────────────────────────
function SectionHeader({ label, accentColor }: { label: string; accentColor: string }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: accentColor }} />
      <span
        className="uppercase tracking-[0.1em]"
        style={{ fontSize: 10, fontWeight: 600, color: "#787B86", fontFamily: "var(--font-sans)" }}
      >
        {label}
      </span>
    </div>
  );
}

// ─── Toggle switch ────────────────────────────────────────────────────────────
function CompactToggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="relative rounded-full transition-colors duration-200"
      style={{
        width: 36,
        height: 20,
        backgroundColor: checked ? "#2D4A47" : "#2A2E39",
        border: "1px solid",
        borderColor: checked ? "#3D6B68" : "#2A2E39",
        flexShrink: 0,
      }}
    >
      <span
        className="absolute rounded-full transition-transform duration-200"
        style={{
          top: 3,
          left: checked ? 17 : 3,
          width: 12,
          height: 12,
          backgroundColor: checked ? "#9CCDCA" : "#4B5563",
        }}
      />
    </button>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
const PRESET_KEYS = ["diagnostic", "exploratory", "validation", "production"] as const;

interface HpoPanelProps {
  advancedMode: boolean;
  onToggleAdvanced: () => void;
}

export function HpoPanel({ advancedMode, onToggleAdvanced }: HpoPanelProps) {
  const setField = useBacktestStore((s) => s.setField);
  const applyStudyPreset = useBacktestStore((s) => s.applyStudyPreset);
  const activePreset = useBacktestStore((s) => s.activePreset);
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const nTrials = useBacktestStore((s) => s.nTrials);
  const hpoManualOverride = useBacktestStore((s) => s.hpoManualOverride);
  const repeats = useBacktestStore((s) => s.repeats);
  const optunaDirection = useBacktestStore((s) => s.optunaDirection);
  const seed = useBacktestStore((s) => s.seed);
  const trainMonths = useBacktestStore((s) => s.trainMonths);
  const testMonths = useBacktestStore((s) => s.testMonths);
  const periodUnit = useBacktestStore((s) => s.periodUnit);
  const targetActiveRate = useBacktestStore((s) => s.targetActiveRate);
  const targetCoverage = useBacktestStore((s) => s.targetCoverage);
  const confidenceThreshold = useBacktestStore((s) => s.confidenceThreshold);
  const evalUseTradingCosts = useBacktestStore((s) => s.evalUseTradingCosts);
  const slipNormBps = useBacktestStore((s) => s.slipNormBps);

  const handlePresetClick = useCallback((key: string) => {
    applyStudyPreset(key);
  }, [applyStudyPreset]);

  const panelBase: React.CSSProperties = {
    backgroundColor: "#1E222D",
    border: "1px solid #2A2E39",
    borderRadius: 6,
  };

  return (
    <div
      className="flex flex-col rounded-lg"
      style={{ backgroundColor: "#1E222D", border: "1px solid #2A2E39" }}
    >
      {/* ── Header ── */}
      <div
        className="flex items-center justify-between px-4"
        style={{ height: 40, borderBottom: "1px solid #2A2E39" }}
      >
        <span
          className="uppercase tracking-[0.1em]"
          style={{ fontSize: 10, fontWeight: 600, color: "#787B86" }}
        >
          Walk-Forward &amp; HPO
        </span>

        {/* Pill toggle */}
        <div
          className="flex items-center rounded-full p-0.5"
          style={{ backgroundColor: "#131722", border: "1px solid #2A2E39" }}
        >
          {[
            { label: "PRESET", active: !advancedMode },
            { label: "ADVANCED", active: advancedMode },
          ].map(({ label, active }) => (
            <button
              key={label}
              onClick={onToggleAdvanced}
              className="rounded-full px-3 transition-all duration-150"
              style={{
                height: 22,
                fontSize: 9,
                fontWeight: 600,
                letterSpacing: "0.08em",
                fontFamily: "var(--font-sans)",
                backgroundColor: active ? "#2A2E39" : "transparent",
                color: active ? "#D1D4DC" : "#4B5563",
                border: "none",
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-5 p-4">

        {/* ══════════════ PRESET MODE ══════════════ */}
        {!advancedMode && (
          <div className="flex flex-col" style={{ gap: 2 }}>
            {PRESET_KEYS.map((key) => {
              const p = STUDY_PRESETS[key];
              const isSelected = activePreset === key;
              const dotColor = isSelected
                ? PRESET_DOT_ACTIVE[p.badgeColor]
                : PRESET_DOT[p.badgeColor];
              const tr = p.trialRange;
              const trialStr = tr.min === tr.max ? `${tr.min}` : `${tr.min}–${tr.max}`;
              const modelKey = (selectedModels as string[])[0] ?? "logistic";
              const estMin = (p.estMinutes as Record<string, number>)[modelKey] ?? 30;
              const estStr = estMin > 120 ? `~${(estMin / 60).toFixed(0)}h / model` : `~${estMin}min / model`;

              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => handlePresetClick(key)}
                  className="flex items-center gap-3 rounded px-3 text-left transition-all duration-100"
                  style={{
                    height: 44,
                    backgroundColor: isSelected ? "#2A2E39" : "transparent",
                    border: `1px solid ${isSelected ? "#3A3E4A" : "transparent"}`,
                  }}
                >
                  {/* Dot */}
                  <div
                    className="rounded-full flex-shrink-0"
                    style={{ width: 6, height: 6, backgroundColor: dotColor }}
                  />

                  {/* Name */}
                  <span
                    className="flex-shrink-0"
                    style={{
                      width: 88,
                      fontSize: 12,
                      fontWeight: isSelected ? 600 : 400,
                      color: isSelected ? "#D1D4DC" : "#787B86",
                      fontFamily: "var(--font-sans)",
                    }}
                  >
                    {p.label}
                  </span>

                  {/* Specs */}
                  <span
                    className="flex-1 tabular-nums"
                    style={{ fontSize: 11, color: "#4B5563", fontFamily: "var(--font-mono)" }}
                  >
                    {trialStr} tri&nbsp;&nbsp;·&nbsp;&nbsp;{p.repeats} run{p.repeats > 1 ? "s" : ""}&nbsp;&nbsp;·&nbsp;&nbsp;{p.trainMonths}mo train
                  </span>

                  {/* Est time */}
                  <span
                    className="tabular-nums flex-shrink-0"
                    style={{ fontSize: 10, color: isSelected ? "#6B7280" : "#374151", fontFamily: "var(--font-mono)" }}
                  >
                    {estStr}
                  </span>
                </button>
              );
            })}

            {/* Description strip */}
            {activePreset && (
              <div
                className="mt-2 px-3 py-2 rounded text-[11px] leading-relaxed"
                style={{ backgroundColor: "#131722", border: "1px solid #2A2E39", color: "#4B5563", fontFamily: "var(--font-sans)" }}
              >
                {STUDY_PRESETS[activePreset as keyof typeof STUDY_PRESETS]?.description}
              </div>
            )}
          </div>
        )}

        {/* ══════════════ ADVANCED MODE ══════════════ */}
        {advancedMode && (
          <div className="flex flex-col gap-4">

            {/* ── Group 1: Optimization Strategy ── */}
            <div style={{ ...panelBase, padding: "12px 14px" }}>
              <SectionHeader label="Optimization Strategy" accentColor="#3D6B68" />
              <div className="flex flex-wrap gap-x-6 gap-y-3 items-end">

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="HPO Trials"
                    tip="Number of Optuna trials per model. Set 0 for default params only. More trials = better configs but slower runs."
                  />
                  <NumInput
                    value={hpoManualOverride ? nTrials : 10}
                    min={RANGES.nTrials.min}
                    max={RANGES.nTrials.max}
                    onChange={(v) => {
                      setField("nTrials", v);
                      setField("hpoManualOverride", true);
                    }}
                    width={72}
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="Repeats / Seeds"
                    tip="Independent runs with different seeds. Higher values reduce seed-dependence and improve confidence in results."
                  />
                  <NumInput
                    value={repeats}
                    min={RANGES.repeats.min}
                    max={RANGES.repeats.max}
                    onChange={(v) => setField("repeats", v)}
                    width={72}
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="Direction"
                    tip="Whether to maximize a metric (e.g. Sharpe) or minimize one (e.g. drawdown) during HPO search."
                  />
                  <CompactSelect
                    value={optunaDirection}
                    options={SELECT_OPTIONS.optunaDirection}
                    onChange={(v) => setField("optunaDirection", v as "maximize" | "minimize")}
                    width={120}
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="Seed"
                    tip="Random seed for reproducibility. Same seed and same config will always produce identical results."
                  />
                  <NumInput
                    value={seed}
                    min={RANGES.seed.min}
                    max={RANGES.seed.max}
                    onChange={(v) => setField("seed", v)}
                    width={72}
                  />
                </div>

              </div>
            </div>

            {/* ── Group 2: Walk-Forward Windows ── */}
            <div style={{ ...panelBase, padding: "12px 14px" }}>
              <SectionHeader label="Walk-Forward Windows" accentColor="#4B5563" />
              <div className="flex flex-wrap gap-x-6 gap-y-3 items-end">

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="Train Window"
                    tip="Months of historical data used to train each walk-forward fold."
                  />
                  <NumInput
                    value={trainMonths}
                    min={RANGES.trainMonths.min}
                    max={RANGES.trainMonths.max}
                    onChange={(v) => setField("trainMonths", v)}
                    width={72}
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="Test Window"
                    tip="Months held out for out-of-sample validation after each training fold."
                  />
                  <NumInput
                    value={testMonths}
                    min={RANGES.testMonths.min}
                    max={RANGES.testMonths.max}
                    onChange={(v) => setField("testMonths", v)}
                    width={72}
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <FieldLabel
                    label="Period Unit"
                    tip="Granularity of the train and test window lengths (months, weeks, or days)."
                  />
                  <CompactSelect
                    value={periodUnit ?? "months"}
                    options={[
                      { value: "months", label: "Months" },
                      { value: "weeks",  label: "Weeks"  },
                      { value: "days",   label: "Days"   },
                    ]}
                    onChange={(v) => setField("periodUnit", v as "months" | "weeks" | "days")}
                    width={110}
                  />
                </div>

              </div>
            </div>

            {/* ── Group 3: Quality Gates ── */}
            <div style={{ ...panelBase, padding: "12px 14px" }}>
              <SectionHeader label="Quality Gates" accentColor="#92400E" />
              <div className="flex flex-col gap-3">

                <div className="flex items-center gap-3">
                  <FieldLabel
                    label="Active Rate"
                    tip="Minimum % of bars the model must signal trades on. Filters models that barely trade."
                  />
                  <ThinSlider
                    value={targetActiveRate}
                    min={RANGES.targetActiveRate.min}
                    max={RANGES.targetActiveRate.max}
                    step={RANGES.targetActiveRate.step}
                    onChange={(v) => setField("targetActiveRate", v)}
                  />
                </div>

                <div className="flex items-center gap-3">
                  <FieldLabel
                    label="Coverage"
                    tip="Minimum % of the test period the model must have exposure. Rejects models that go silent mid-window."
                  />
                  <ThinSlider
                    value={targetCoverage}
                    min={RANGES.targetCoverage.min}
                    max={RANGES.targetCoverage.max}
                    step={RANGES.targetCoverage.step}
                    onChange={(v) => setField("targetCoverage", v)}
                  />
                </div>

                <div className="flex items-center gap-3">
                  <FieldLabel
                    label="Confidence"
                    tip="Minimum prediction probability required to enter a trade. Higher = fewer but higher-conviction signals."
                  />
                  <ThinSlider
                    value={confidenceThreshold}
                    min={RANGES.confidenceThreshold.min}
                    max={RANGES.confidenceThreshold.max}
                    step={RANGES.confidenceThreshold.step}
                    onChange={(v) => setField("confidenceThreshold", v)}
                  />
                </div>

              </div>
            </div>

            {/* ── Group 4: Execution Reality ── */}
            <div style={{ ...panelBase, padding: "12px 14px" }}>
              <SectionHeader label="Execution Reality" accentColor="#7F1D1D" />
              <div className="flex flex-wrap gap-x-6 gap-y-3 items-center">

                <div className="flex items-center gap-3">
                  <FieldLabel
                    label="Trading Costs"
                    tip="Enables spread and slippage simulation during the backtest loop. Disabling inflates returns."
                  />
                  <CompactToggle
                    checked={evalUseTradingCosts}
                    onChange={(v) => setField("evalUseTradingCosts", v)}
                  />
                </div>

                {evalUseTradingCosts && (
                  <div className="flex flex-col gap-1.5">
                    <FieldLabel
                      label="Slippage BPS"
                      tip="Average execution slippage in basis points. 1 bps = 0.01%. Typical retail FX: 0.1–0.5 bps."
                    />
                    <NumInput
                      value={slipNormBps}
                      min={RANGES.slipNormBps.min}
                      max={RANGES.slipNormBps.max}
                      step={RANGES.slipNormBps.step}
                      onChange={(v) => setField("slipNormBps", v)}
                      width={72}
                    />
                  </div>
                )}

              </div>
            </div>

            {/* ── Per-Model Hyperparameters ── */}
            <ModelHyperparamsPanel />
          </div>
        )}

      </div>
    </div>
  );
}

// ─── Private helpers (retained for internal use) ──────────────────────────────
const _HPO_TRIALS: Record<string, Record<string, { r: number; b: number }>> = {
  light: { logistic: { r: 1, b: 1 }, svm: { r: 1, b: 1 }, decision_tree: { r: 1, b: 1 },
           random_forest: { r: 1, b: 1 }, xgboost: { r: 1, b: 1 },
           lstm: { r: 1, b: 1 }, cnn: { r: 1, b: 1 }, transformer: { r: 1, b: 1 },
           ensemble_adaptive_regime: { r: 1, b: 1 }, ensemble_cnn_lstm_xgboost: { r: 1, b: 1 }, dqn: { r: 1, b: 1 } },
  quick: { logistic: { r: 2, b: 2 }, svm: { r: 2, b: 2 }, decision_tree: { r: 2, b: 2 },
           random_forest: { r: 2, b: 2 }, xgboost: { r: 2, b: 2 },
           lstm: { r: 2, b: 2 }, cnn: { r: 2, b: 2 }, transformer: { r: 2, b: 2 },
           ensemble_adaptive_regime: { r: 2, b: 2 }, ensemble_cnn_lstm_xgboost: { r: 2, b: 2 }, dqn: { r: 1, b: 1 } },
  standard: { logistic: { r: 5, b: 5 }, svm: { r: 5, b: 5 }, decision_tree: { r: 5, b: 5 },
             random_forest: { r: 5, b: 10 }, xgboost: { r: 5, b: 15 },
             lstm: { r: 3, b: 7 }, cnn: { r: 3, b: 7 }, transformer: { r: 3, b: 7 },
             ensemble_adaptive_regime: { r: 2, b: 3 }, ensemble_cnn_lstm_xgboost: { r: 2, b: 3 }, dqn: { r: 2, b: 3 } },
  deep: { logistic: { r: 10, b: 10 }, svm: { r: 10, b: 10 }, decision_tree: { r: 5, b: 10 },
          random_forest: { r: 10, b: 20 }, xgboost: { r: 10, b: 30 },
          lstm: { r: 5, b: 15 }, cnn: { r: 5, b: 15 }, transformer: { r: 5, b: 15 },
          ensemble_adaptive_regime: { r: 3, b: 7 }, ensemble_cnn_lstm_xgboost: { r: 3, b: 7 }, dqn: { r: 3, b: 5 } },
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function _trialForModel(intensity: string, model: string): number {
  const m = _HPO_TRIALS[intensity]?.[model];
  return m ? Math.max(m.r + m.b, 10) : 10;
}
