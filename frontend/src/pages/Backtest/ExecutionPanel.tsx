import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export function ExecutionPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();
  const [open, setOpen] = useState(false);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <button
        className="flex w-full items-center gap-2 text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Execution Models (Advanced)
      </button>

      {open && (
        <div className="mt-4 flex flex-col gap-4">

          {/* Position Sizing */}
          <section className="rounded-md border p-3" style={{ borderColor: "var(--color-border-subtle)" }}>
            <h4 className="mb-2 text-xs font-semibold uppercase" style={{ color: "var(--color-accent-classical)" }}>
              Position Sizing
            </h4>
            <div className="mb-3">
              <ParamSelect
                label="Method"
                paramKey="sizing_method"
                value={s.sizingMethod}
                options={[...SELECT_OPTIONS.sizingMethod]}
                onChange={(v) => setField("sizingMethod", v as typeof s.sizingMethod)}
              />
            </div>
            {s.sizingMethod === "kelly" && (
              <ParamSlider
                label="Kelly Fraction"
                paramKey="kelly_fraction"
                value={s.kellyFraction}
                min={RANGES.kellyFraction.min}
                max={RANGES.kellyFraction.max}
                step={RANGES.kellyFraction.step}
                description="Mathematical aggression. Dynamically sizes based on win rate and risk/reward."
                onChange={(v) => setField("kellyFraction", v)}
              />
            )}
            {s.sizingMethod === "fixed_fractional" && (
              <ParamSlider
                label="Risk Fraction"
                paramKey="risk_fraction"
                value={s.riskFraction}
                min={RANGES.riskFraction.min}
                max={RANGES.riskFraction.max}
                step={RANGES.riskFraction.step}
                description="Risks a strict % of total equity per trade"
                onChange={(v) => setField("riskFraction", v)}
              />
            )}
            {s.sizingMethod === "atr" && (
              <ParamSlider
                label="ATR Risk %"
                paramKey="atr_risk_pct"
                value={s.atrRiskPct}
                min={0.01}
                max={0.05}
                step={0.005}
                onChange={(v) => setField("atrRiskPct", v)}
              />
            )}
          </section>

          {/* Trailing Stops */}
          <section className="rounded-md border p-3" style={{ borderColor: "var(--color-border-subtle)" }}>
            <h4 className="mb-2 text-xs font-semibold uppercase" style={{ color: "var(--color-accent-deep)" }}>
              Trailing Stops
            </h4>
            <div className="mb-3">
              <ParamSelect
                label="Method"
                paramKey="trailing_method"
                value={s.trailingMethod}
                options={[...SELECT_OPTIONS.trailingMethod]}
                onChange={(v) => setField("trailingMethod", v as typeof s.trailingMethod)}
              />
            </div>
            {s.trailingMethod !== "none" && (
              <ParamSlider
                label="Activation"
                paramKey="trailing_activation"
                value={s.trailingActivation}
                min={RANGES.trailingActivation.min}
                max={RANGES.trailingActivation.max}
                step={RANGES.trailingActivation.step}
                description="Profit threshold before trailing activates"
                onChange={(v) => setField("trailingActivation", v)}
              />
            )}
          </section>

          {/* Risk Management */}
          <section className="rounded-md border p-3" style={{ borderColor: "var(--color-border-subtle)" }}>
            <h4 className="mb-2 text-xs font-semibold uppercase" style={{ color: "var(--color-accent-danger)" }}>
              Risk Management
            </h4>
            <div className="flex flex-col gap-3">
              <ParamSlider
                label="Max Drawdown"
                paramKey="max_drawdown_pct"
                value={s.maxDrawdownPct}
                min={RANGES.maxDrawdownPct.min}
                max={RANGES.maxDrawdownPct.max}
                step={RANGES.maxDrawdownPct.step}
                description="Circuit breaker. Halts trading if equity drops this % from peak."
                onChange={(v) => setField("maxDrawdownPct", v)}
              />
              <div className="grid grid-cols-2 gap-4">
                <ParamSlider
                  label="Max Consec. Losses"
                  paramKey="max_consecutive_losses"
                  value={s.maxConsecutiveLosses}
                  min={RANGES.maxConsecutiveLosses.min}
                  max={RANGES.maxConsecutiveLosses.max}
                  step={RANGES.maxConsecutiveLosses.step}
                  onChange={(v) => setField("maxConsecutiveLosses", v)}
                />
                <ParamSlider
                  label="Daily Loss Limit"
                  paramKey="daily_loss_limit_pct"
                  value={s.dailyLossLimitPct}
                  min={RANGES.dailyLossLimitPct.min}
                  max={RANGES.dailyLossLimitPct.max}
                  step={RANGES.dailyLossLimitPct.step}
                  onChange={(v) => setField("dailyLossLimitPct", v)}
                />
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
