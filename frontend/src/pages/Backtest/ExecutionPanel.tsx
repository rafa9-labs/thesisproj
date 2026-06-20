import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";

export function ExecutionPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();

  return (
    <Panel>
      <PanelHeader
        title="Execution Models"
        subtitle="Define account, sizing, and risk-control rules."
      />

      {/* ── General ── */}
      <Section title="General">
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          <ParamSlider
            label="INITIAL EQUITY"
            value={s.initialEquity}
            min={RANGES.initialEquity.min}
            max={RANGES.initialEquity.max}
            step={RANGES.initialEquity.step}
            tooltip="Starting account balance in base currency."
            onChange={(v) => setField("initialEquity", v)}
          />
          <ParamSlider
            label="MAX LEVERAGE"
            value={s.maxLeverage}
            min={RANGES.maxLeverage.min}
            max={RANGES.maxLeverage.max}
            step={RANGES.maxLeverage.step}
            tooltip="Highest leverage allowed per position. Affects margin requirements and drawdown magnitude."
            onChange={(v) => setField("maxLeverage", v)}
          />
        </div>
      </Section>

      {/* ── Position Sizing ── */}
      <Section title="Position Sizing" accent="var(--color-accent-classical)">
        <div className="flex flex-col gap-6">
          <div className="max-w-sm">
            <ParamSelect
              label="METHOD"
              value={s.sizingMethod}
              options={[...SELECT_OPTIONS.sizingMethod]}
              tooltip="Fixed Lot = constant size. Kelly = optimal growth. Fixed Fractional = risk %. ATR = volatility-adjusted."
              onChange={(v) => setField("sizingMethod", v as typeof s.sizingMethod)}
            />
          </div>

          {s.sizingMethod === "kelly" && (
            <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              <ParamSlider
                label="KELLY FRACTION"
                value={s.kellyFraction}
                min={RANGES.kellyFraction.min}
                max={RANGES.kellyFraction.max}
                step={RANGES.kellyFraction.step}
                tooltip="Fraction of full Kelly to use. 0.5 = Half-Kelly, safer but slower growth."
                onChange={(v) => setField("kellyFraction", v)}
              />
              <ParamSlider
                label="MIN TRADES"
                value={s.kellyMinTrades}
                min={RANGES.kellyMinTrades.min}
                max={RANGES.kellyMinTrades.max}
                step={RANGES.kellyMinTrades.step}
                tooltip="Minimum trade history before Kelly sizing activates. Prevents erratic early sizing."
                onChange={(v) => setField("kellyMinTrades", v)}
              />
            </div>
          )}

          {s.sizingMethod === "fixed_fractional" && (
            <div className="max-w-sm">
              <ParamSlider
                label="RISK FRACTION"
                value={s.riskFraction}
                min={RANGES.riskFraction.min}
                max={RANGES.riskFraction.max}
                step={RANGES.riskFraction.step}
                tooltip="Percentage of equity risked on each trade. 2% = standard conservative sizing."
                onChange={(v) => setField("riskFraction", v)}
              />
            </div>
          )}

          {s.sizingMethod === "atr" && (
            <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              <ParamSlider
                label="ATR RISK %"
                value={s.atrRiskPct}
                min={RANGES.atrRiskPct.min}
                max={RANGES.atrRiskPct.max}
                step={RANGES.atrRiskPct.step}
                tooltip="Equity percentage risked, scaled by current ATR volatility."
                onChange={(v) => setField("atrRiskPct", v)}
              />
              <ParamSlider
                label="ATR SL MULT"
                value={s.atrSlMult}
                min={RANGES.atrSlMult.min}
                max={RANGES.atrSlMult.max}
                step={RANGES.atrSlMult.step}
                tooltip="Stop-loss distance as a multiple of ATR. Higher = wider stops, fewer whipsaws."
                onChange={(v) => setField("atrSlMult", v)}
              />
            </div>
          )}
        </div>
      </Section>

      {/* ── Trailing Stops ── */}
      <Section title="Trailing Stops">
        <div className="flex flex-col gap-6">
          <div className="max-w-sm">
            <ParamSelect
              label="METHOD"
              value={s.trailingMethod}
              options={[...SELECT_OPTIONS.trailingMethod]}
              tooltip="None = no trailing. Standard = fixed pip step. ATR = volatility-adjusted step. Chandelier = highest high / lowest low minus ATR multiple."
              onChange={(v) => setField("trailingMethod", v as typeof s.trailingMethod)}
            />
          </div>
          {s.trailingMethod !== "none" && (
            <div className="max-w-sm">
              <ParamSlider
                label="ACTIVATION"
                value={s.trailingActivation}
                min={RANGES.trailingActivation.min}
                max={RANGES.trailingActivation.max}
                step={RANGES.trailingActivation.step}
                tooltip="Profit threshold (as a fraction of price) that must be reached before the trailing stop begins to move."
                onChange={(v) => setField("trailingActivation", v)}
              />
            </div>
          )}
        </div>
      </Section>

      {/* ── Risk Management ── */}
      <Section title="Risk Management">
        <div className="flex flex-col gap-6">
          <div className="max-w-sm">
            <ParamSlider
              label="MAX DRAWDOWN"
              value={s.maxDrawdownPct}
              min={RANGES.maxDrawdownPct.min}
              max={RANGES.maxDrawdownPct.max}
              step={RANGES.maxDrawdownPct.step}
              tooltip="Peak-to-trough equity drop that triggers a full trading halt. 15% = common institutional limit."
              onChange={(v) => setField("maxDrawdownPct", v)}
            />
          </div>
          <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            <ParamSlider
              label="MAX CONSEC. LOSSES"
              value={s.maxConsecutiveLosses}
              min={RANGES.maxConsecutiveLosses.min}
              max={RANGES.maxConsecutiveLosses.max}
              step={RANGES.maxConsecutiveLosses.step}
              tooltip="Halt after this many consecutive losing trades. Prevents revenge-trading spirals."
              onChange={(v) => setField("maxConsecutiveLosses", v)}
            />
            <ParamSlider
              label="DAILY LOSS LIMIT"
              value={s.dailyLossLimitPct}
              min={RANGES.dailyLossLimitPct.min}
              max={RANGES.dailyLossLimitPct.max}
              step={RANGES.dailyLossLimitPct.step}
              tooltip="Maximum daily equity loss before pausing until the next session."
              onChange={(v) => setField("dailyLossLimitPct", v)}
            />
          </div>
        </div>
      </Section>
    </Panel>
  );
}
