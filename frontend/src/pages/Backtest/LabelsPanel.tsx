import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";
import { RANGES } from "@/lib/constants";

export function LabelsPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const useTB = useBacktestStore((s) => s.useTripleBarrier);
  const s = useBacktestStore.getState();

  return (
    <Panel>
      <PanelHeader
        title="Labels & Triple Barrier"
        subtitle="Define how price movements become model targets."
      />

      {/* Label Threshold */}
      <Section title="Label Configuration">
        <div className="max-w-sm">
          <ParamSlider
            label="LABEL THRESHOLD"
            value={s.labelThreshold}
            min={RANGES.labelThreshold.min}
            max={RANGES.labelThreshold.max}
            step={RANGES.labelThreshold.step}
            tooltip="Minimum price move (in %) to trigger a label. Smaller = more signals, more noise."
            onChange={(v) => setField("labelThreshold", v)}
          />
        </div>
      </Section>

      {/* Triple Barrier */}
      <Section
        title="Triple Barrier"
        accent="var(--color-accent-classical)"
      >
        <div className="flex flex-col gap-5">
          <ParamToggle
            label="USE TRIPLE BARRIER"
            checked={useTB}
            tooltip="Classifies a trade as successful only if it hits a profit target before hitting stop-loss or timing out."
            onChange={(v) => setField("useTripleBarrier", v)}
          />

          {useTB && (
            <div className="flex flex-col gap-6 rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-5 backdrop-blur-[8px]">
              <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                <ParamSlider
                  label="PT MULT"
                  value={s.tbPtMult}
                  min={RANGES.tbPtMult.min}
                  max={RANGES.tbPtMult.max}
                  step={RANGES.tbPtMult.step}
                  tooltip="Profit-target distance as a volatility multiplier."
                  onChange={(v) => setField("tbPtMult", v)}
                />
                <ParamSlider
                  label="SL MULT"
                  value={s.tbSlMult}
                  min={RANGES.tbSlMult.min}
                  max={RANGES.tbSlMult.max}
                  step={RANGES.tbSlMult.step}
                  tooltip="Stop-loss distance as a volatility multiplier."
                  onChange={(v) => setField("tbSlMult", v)}
                />
                <ParamSlider
                  label="MAX HOLDING"
                  value={s.tbMaxHolding}
                  min={RANGES.tbMaxHolding.min}
                  max={RANGES.tbMaxHolding.max}
                  step={RANGES.tbMaxHolding.step}
                  tooltip="Maximum bars to hold a position before the vertical barrier fires."
                  onChange={(v) => setField("tbMaxHolding", v)}
                />
              </div>
              <div className="max-w-sm">
                <ParamSlider
                  label="NEUTRAL ZONE"
                  value={s.tbNeutralZone}
                  min={RANGES.tbNeutralZone.min}
                  max={RANGES.tbNeutralZone.max}
                  step={RANGES.tbNeutralZone.step}
                  tooltip="Price range around entry that is considered flat / no-direction. Filters sideways chop."
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
