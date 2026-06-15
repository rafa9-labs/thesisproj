import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";
import { RANGES } from "@/lib/constants";

export function LabelsPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const verbose = useSettingsStore((s) => s.verboseMode);
  const useTB = useBacktestStore((s) => s.useTripleBarrier);
  const s = useBacktestStore.getState();

  return (
    <Panel>
      <PanelHeader
        title="Labels & Triple Barrier"
        subtitle="Define how price movements become model targets."
      />

      {/* Label Threshold */}
      <Section
        title="Label Configuration"
        description="Defines how price movements are converted into target labels for the model. Higher thresholds require larger moves to register as a signal."
      >
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
      </Section>

      {/* Triple Barrier */}
      <Section
        title="Triple Barrier"
        accent="var(--color-accent-success)"
        description="An advanced labeling method from quantitative finance. Each trade has a profit target (upper barrier), stop-loss (lower barrier), and time limit (vertical barrier). The first one hit determines the label."
      >
        <div className="flex flex-col gap-5">
          <ParamToggle
            label="Use Triple Barrier"
            checked={useTB}
            description={
              verbose
                ? "Classifies a trade as successful only if it hits a profit target before hitting stop-loss or timing out."
                : "Profit target / stop-loss / time limit labeling."
            }
            onChange={(v) => setField("useTripleBarrier", v)}
          />

          {useTB && (
            <div className="flex flex-col gap-6 rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-5 backdrop-blur-[8px]">
              <div className="grid grid-cols-1 gap-x-6 gap-y-6 sm:grid-cols-3">
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
      </Section>
    </Panel>
  );
}
