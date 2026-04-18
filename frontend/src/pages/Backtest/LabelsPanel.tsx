import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { RANGES } from "@/lib/constants";

export function LabelsPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const verbose = useSettingsStore((s) => s.verboseMode);
  const useTB = useBacktestStore((s) => s.useTripleBarrier);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <h3
        className="mb-3 text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Labels &amp; Triple Barrier
      </h3>

      <div className="mb-4">
        <ParamSlider
          label="Label Threshold"
          paramKey="label_threshold"
          value={useBacktestStore.getState().labelThreshold}
          min={RANGES.labelThreshold.min}
          max={RANGES.labelThreshold.max}
          step={RANGES.labelThreshold.step}
          onChange={(v) => setField("labelThreshold", v)}
        />
      </div>

      <ParamToggle
        label="Use Triple Barrier"
        paramKey="use_triple_barrier"
        checked={useTB}
        description={verbose ? "Classifies a trade as successful only if it hits a profit target before hitting stop-loss or timing out" : undefined}
        onChange={(v) => setField("useTripleBarrier", v)}
      />

      {useTB && (
        <div className="mt-3 flex flex-col gap-4 rounded-md border p-3" style={{ borderColor: "var(--color-border-subtle)" }}>
          <div className="grid grid-cols-3 gap-4">
            <ParamSlider
              label="PT Mult"
              paramKey="tb_pt_mult"
              value={useBacktestStore.getState().tbPtMult}
              min={RANGES.tbPtMult.min}
              max={RANGES.tbPtMult.max}
              step={RANGES.tbPtMult.step}
              onChange={(v) => setField("tbPtMult", v)}
            />
            <ParamSlider
              label="SL Mult"
              paramKey="tb_sl_mult"
              value={useBacktestStore.getState().tbSlMult}
              min={RANGES.tbSlMult.min}
              max={RANGES.tbSlMult.max}
              step={RANGES.tbSlMult.step}
              onChange={(v) => setField("tbSlMult", v)}
            />
            <ParamSlider
              label="Max Holding"
              paramKey="tb_max_holding"
              value={useBacktestStore.getState().tbMaxHolding}
              min={RANGES.tbMaxHolding.min}
              max={RANGES.tbMaxHolding.max}
              step={RANGES.tbMaxHolding.step}
              onChange={(v) => setField("tbMaxHolding", v)}
            />
          </div>
          <ParamSlider
            label="Neutral Zone"
            paramKey="tb_neutral_zone"
            value={useBacktestStore.getState().tbNeutralZone}
            min={RANGES.tbNeutralZone.min}
            max={RANGES.tbNeutralZone.max}
            step={RANGES.tbNeutralZone.step}
            onChange={(v) => setField("tbNeutralZone", v)}
          />
        </div>
      )}
    </div>
  );
}
