import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

const sectionClass = "rounded-xl border p-6";
const sectionStyle: React.CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};
const sectionTitleClass = "mb-1 text-[11px] font-medium uppercase tracking-[0.12em]";
const sectionTitleStyle: React.CSSProperties = { color: "var(--color-text-secondary)" };
const explainerClass = "mb-5 text-[11px] font-light leading-relaxed max-w-[720px]";
const explainerStyle: React.CSSProperties = { color: "var(--color-text-muted)" };

export function ExecutionPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();
  const [open, setOpen] = useState(false);

  return (
    <div
      className="flex flex-col gap-6 rounded-xl border p-6"
      style={{
        backgroundColor: "var(--color-glass)",
        borderColor: "var(--color-glass-border)",
        backdropFilter: "blur(12px)",
      }}
    >
      {/* Header with collapse toggle */}
      <button
        className="flex w-full items-center gap-2 text-[11px] font-medium uppercase tracking-[0.12em] transition-colors duration-200 hover:text-[var(--color-text-primary)]"
        style={{ color: "var(--color-text-muted)" }}
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown size={14} strokeWidth={1.5} /> : <ChevronRight size={14} strokeWidth={1.5} />}
        Execution Models (Advanced)
      </button>

      {open && (
        <div className="flex flex-col gap-6">

          {/* ── General ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>General</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Base account and margin settings that apply to every execution model.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
              <ParamSlider
                label="Initial Equity"
                value={s.initialEquity}
                min={RANGES.initialEquity.min}
                max={RANGES.initialEquity.max}
                step={RANGES.initialEquity.step}
                description="Starting account balance in base currency."
                onChange={(v) => setField("initialEquity", v)}
              />
              <ParamSlider
                label="Max Leverage"
                value={s.maxLeverage}
                min={RANGES.maxLeverage.min}
                max={RANGES.maxLeverage.max}
                step={RANGES.maxLeverage.step}
                description="Highest leverage allowed per position. Affects margin requirements and drawdown magnitude."
                onChange={(v) => setField("maxLeverage", v)}
              />
            </div>
          </section>

          {/* ── Position Sizing ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-classical)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Position Sizing</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Determines how large each trade should be relative to account equity. Different methods balance growth vs. risk of ruin.
            </p>
            <div className="flex flex-col gap-6">
              <div className="max-w-sm">
                <ParamSelect
                  label="Method"
                  value={s.sizingMethod}
                  options={[...SELECT_OPTIONS.sizingMethod]}
                  description="Fixed Lot = constant size. Kelly = optimal growth. Fixed Fractional = risk %. ATR = volatility-adjusted."
                  onChange={(v) => setField("sizingMethod", v as typeof s.sizingMethod)}
                />
              </div>

              {s.sizingMethod === "kelly" && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
                  <ParamSlider
                    label="Kelly Fraction"
                    value={s.kellyFraction}
                    min={RANGES.kellyFraction.min}
                    max={RANGES.kellyFraction.max}
                    step={RANGES.kellyFraction.step}
                    description="Fraction of full Kelly to use. 0.5 = Half-Kelly, safer but slower growth."
                    onChange={(v) => setField("kellyFraction", v)}
                  />
                  <ParamSlider
                    label="Min Trades"
                    value={s.kellyMinTrades}
                    min={RANGES.kellyMinTrades.min}
                    max={RANGES.kellyMinTrades.max}
                    step={RANGES.kellyMinTrades.step}
                    description="Minimum trade history before Kelly sizing activates. Prevents erratic early sizing."
                    onChange={(v) => setField("kellyMinTrades", v)}
                  />
                </div>
              )}

              {s.sizingMethod === "fixed_fractional" && (
                <div className="max-w-sm">
                  <ParamSlider
                    label="Risk Fraction"
                    value={s.riskFraction}
                    min={RANGES.riskFraction.min}
                    max={RANGES.riskFraction.max}
                    step={RANGES.riskFraction.step}
                    description="Percentage of equity risked on each trade. 2% = standard conservative sizing."
                    onChange={(v) => setField("riskFraction", v)}
                  />
                </div>
              )}

              {s.sizingMethod === "atr" && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
                  <ParamSlider
                    label="ATR Risk %"
                    value={s.atrRiskPct}
                    min={RANGES.atrRiskPct.min}
                    max={RANGES.atrRiskPct.max}
                    step={RANGES.atrRiskPct.step}
                    description="Equity percentage risked, scaled by current ATR volatility."
                    onChange={(v) => setField("atrRiskPct", v)}
                  />
                  <ParamSlider
                    label="ATR SL Mult"
                    value={s.atrSlMult}
                    min={RANGES.atrSlMult.min}
                    max={RANGES.atrSlMult.max}
                    step={RANGES.atrSlMult.step}
                    description="Stop-loss distance as a multiple of ATR. Higher = wider stops, fewer whipsaws."
                    onChange={(v) => setField("atrSlMult", v)}
                  />
                </div>
              )}
            </div>
          </section>

          {/* ── Trailing Stops ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-deep)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Trailing Stops</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Automatically moves the stop-loss in favor of the trade as price progresses, locking in profits while allowing runners.
            </p>
            <div className="flex flex-col gap-6">
              <div className="max-w-sm">
                <ParamSelect
                  label="Method"
                  value={s.trailingMethod}
                  options={[...SELECT_OPTIONS.trailingMethod]}
                  description="None = no trailing. Standard = fixed pip step. ATR = volatility-adjusted step. Chandelier = highest high / lowest low minus ATR multiple."
                  onChange={(v) => setField("trailingMethod", v as typeof s.trailingMethod)}
                />
              </div>
              {s.trailingMethod !== "none" && (
                <div className="max-w-sm">
                  <ParamSlider
                    label="Activation"
                    value={s.trailingActivation}
                    min={RANGES.trailingActivation.min}
                    max={RANGES.trailingActivation.max}
                    step={RANGES.trailingActivation.step}
                    description="Profit threshold (as a fraction of price) that must be reached before the trailing stop begins to move."
                    onChange={(v) => setField("trailingActivation", v)}
                  />
                </div>
              )}
            </div>
          </section>

          {/* ── Risk Management ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-danger)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Risk Management</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Circuit breakers and drawdown controls that halt trading when conditions become unfavorable. Essential for live deployment.
            </p>
            <div className="flex flex-col gap-6">
              <div className="max-w-sm">
                <ParamSlider
                  label="Max Drawdown"
                  value={s.maxDrawdownPct}
                  min={RANGES.maxDrawdownPct.min}
                  max={RANGES.maxDrawdownPct.max}
                  step={RANGES.maxDrawdownPct.step}
                  description="Peak-to-trough equity drop that triggers a full trading halt. 15% = common institutional limit."
                  onChange={(v) => setField("maxDrawdownPct", v)}
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
                <ParamSlider
                  label="Max Consec. Losses"
                  value={s.maxConsecutiveLosses}
                  min={RANGES.maxConsecutiveLosses.min}
                  max={RANGES.maxConsecutiveLosses.max}
                  step={RANGES.maxConsecutiveLosses.step}
                  description="Halt after this many consecutive losing trades. Prevents revenge-trading spirals."
                  onChange={(v) => setField("maxConsecutiveLosses", v)}
                />
                <ParamSlider
                  label="Daily Loss Limit"
                  value={s.dailyLossLimitPct}
                  min={RANGES.dailyLossLimitPct.min}
                  max={RANGES.dailyLossLimitPct.max}
                  step={RANGES.dailyLossLimitPct.step}
                  description="Maximum daily equity loss before pausing until the next session."
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
