import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { RANGES } from "@/lib/constants";

const sectionClass = "rounded-xl border p-6";
const sectionStyle: React.CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};
const sectionTitleClass = "mb-1 text-[11px] font-medium uppercase tracking-[0.12em]";
const sectionTitleStyle: React.CSSProperties = { color: "var(--color-text-secondary)" };
const explainerClass = "mb-5 text-[11px] font-light leading-relaxed max-w-[720px]";
const explainerStyle: React.CSSProperties = { color: "var(--color-text-muted)" };

export function LabelsPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const verbose = useSettingsStore((s) => s.verboseMode);
  const useTB = useBacktestStore((s) => s.useTripleBarrier);
  const s = useBacktestStore.getState();

  return (
    <div
      className="flex flex-col gap-6 rounded-xl border p-6"
      style={{
        backgroundColor: "var(--color-glass)",
        borderColor: "var(--color-glass-border)",
        backdropFilter: "blur(12px)",
      }}
    >
      <h3
        className="text-[11px] font-medium uppercase tracking-[0.12em]"
        style={{ color: "var(--color-text-muted)" }}
      >
        Labels &amp; Triple Barrier
      </h3>

      {/* Label Threshold */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>Label Configuration</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          Defines how price movements are converted into target labels for the model. Higher thresholds require larger moves to register as a signal.
        </p>
        <div className="max-w-sm">
          <ParamSlider
            label="Label Threshold"
            value={s.labelThreshold}
            min={RANGES.labelThreshold.min}
            max={RANGES.labelThreshold.max}
            step={RANGES.labelThreshold.step}
            description="Minimum price move (in %) to trigger a label. Smaller = more signals, more noise."
            onChange={(v) => setField("labelThreshold", v)}
          />
        </div>
      </section>

      {/* Triple Barrier */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-success)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>Triple Barrier</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          An advanced labeling method from quantitative finance. Each trade has a profit target (upper barrier), stop-loss (lower barrier), and time limit (vertical barrier). The first one hit determines the label.
        </p>
        <div className="flex flex-col gap-5">
          <ParamToggle
            label="Use Triple Barrier"
            checked={useTB}
            description={verbose
              ? "Classifies a trade as successful only if it hits a profit target before hitting stop-loss or timing out."
              : "Profit target / stop-loss / time limit labeling."}
            onChange={(v) => setField("useTripleBarrier", v)}
          />

          {useTB && (
            <div
              className="flex flex-col gap-6 rounded-xl border p-6"
              style={{
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-glass)",
                backdropFilter: "blur(8px)",
              }}
            >
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-6 gap-y-6">
                <ParamSlider
                  label="PT Mult"
                  value={s.tbPtMult}
                  min={RANGES.tbPtMult.min}
                  max={RANGES.tbPtMult.max}
                  step={RANGES.tbPtMult.step}
                  description="Profit-target distance as a volatility multiplier."
                  onChange={(v) => setField("tbPtMult", v)}
                />
                <ParamSlider
                  label="SL Mult"
                  value={s.tbSlMult}
                  min={RANGES.tbSlMult.min}
                  max={RANGES.tbSlMult.max}
                  step={RANGES.tbSlMult.step}
                  description="Stop-loss distance as a volatility multiplier."
                  onChange={(v) => setField("tbSlMult", v)}
                />
                <ParamSlider
                  label="Max Holding"
                  value={s.tbMaxHolding}
                  min={RANGES.tbMaxHolding.min}
                  max={RANGES.tbMaxHolding.max}
                  step={RANGES.tbMaxHolding.step}
                  description="Maximum bars to hold a position before the vertical barrier fires."
                  onChange={(v) => setField("tbMaxHolding", v)}
                />
              </div>
              <div className="max-w-sm">
                <ParamSlider
                  label="Neutral Zone"
                  value={s.tbNeutralZone}
                  min={RANGES.tbNeutralZone.min}
                  max={RANGES.tbNeutralZone.max}
                  step={RANGES.tbNeutralZone.step}
                  description="Price range around entry that is considered flat / no-direction. Filters sideways chop."
                  onChange={(v) => setField("tbNeutralZone", v)}
                />
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
