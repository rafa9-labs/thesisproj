import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

const CORE_INDICATORS = [
  { key: "useAdx" as const, label: "ADX" },
  { key: "useAtr" as const, label: "ATR" },
  { key: "useBbands" as const, label: "Bollinger" },
  { key: "useEma" as const, label: "EMA" },
  { key: "useSma" as const, label: "SMA" },
  { key: "useRsi" as const, label: "RSI" },
  { key: "useMacd" as const, label: "MACD" },
  { key: "useDonchian" as const, label: "Donchian" },
  { key: "useStoch" as const, label: "Stochastic" },
  { key: "useSar" as const, label: "SAR" },
];

const ADVANCED_TOGGLES = [
  { key: "useMtfMa" as const, label: "MTF MA", param: "use_mtf_ma" },
  { key: "useMtfAlignment" as const, label: "MTF Alignment", param: "use_mtf_alignment" },
  { key: "useMacdAtrRatio" as const, label: "MACD/ATR Ratio", param: "use_macd_atr_ratio" },
  { key: "useTripleConfirm" as const, label: "Triple Confirm", param: "use_triple_confirm" },
  { key: "useTrendConfirm" as const, label: "Trend Confirm", param: "use_trend_confirm" },
  { key: "useVolManagedMom" as const, label: "Vol-Managed Mom", param: "use_vol_managed_mom" },
  { key: "useMaSpread" as const, label: "MA Spread", param: "use_ma_spread" },
  { key: "useSlopeDiff" as const, label: "Slope Diff", param: "use_slope_diff" },
  { key: "useSqueezeBreakout" as const, label: "Squeeze Breakout", param: "use_squeeze_breakout" },
  { key: "useSqueezeExpansion" as const, label: "Squeeze Expansion", param: "use_squeeze_expansion" },
  { key: "useAtrChannelBreakout" as const, label: "ATR Channel Breakout", param: "use_atr_channel_breakout" },
  { key: "useExtAtrLowAdx" as const, label: "Ext ATR Low ADX", param: "use_ext_atr_low_adx" },
  { key: "useReentryMom" as const, label: "Re-entry Momentum", param: "use_reentry_mom" },
  { key: "useRvFeatures" as const, label: "RV Features", param: "use_rv_features" },
  { key: "useIndicatorStates" as const, label: "Indicator States", param: "use_indicator_states" },
];

export function FeaturesPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const verbose = useSettingsStore((s) => s.verboseMode);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <h3
        className="mb-3 text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Feature Engineering
      </h3>

      {/* Core indicators */}
      <div className="mb-4 grid grid-cols-5 gap-x-4 gap-y-2">
        {CORE_INDICATORS.map(({ key, label }) => (
          <ParamToggle
            key={key}
            label={label}
            paramKey={key}
            checked={useBacktestStore.getState()[key] as boolean}
            onChange={(v) => setField(key, v)}
          />
        ))}
      </div>

      {/* FracDiff + crossover */}
      <div className="mb-4 flex gap-6">
        <ParamToggle
          label="FracDiff"
          paramKey="use_fracdiff"
          checked={useBacktestStore.getState().useFracdiff}
          description={verbose ? "Fractional differentiation preserves memory while achieving stationarity" : undefined}
          onChange={(v) => setField("useFracdiff", v)}
        />
        <div className="flex-1">
          <ParamSlider
            label="fracdiff d"
            paramKey="fracdiff_d"
            value={useBacktestStore.getState().fracdiffD}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => setField("fracdiffD", v)}
          />
        </div>
      </div>

      <div className="mb-4 flex gap-4">
        <ParamToggle
          label="Crossover Bins"
          paramKey="use_crossover_bins"
          checked={useBacktestStore.getState().useCrossoverBins}
          onChange={(v) => setField("useCrossoverBins", v)}
        />
        <ParamToggle
          label="Price-MA Z-Score"
          paramKey="use_price_ma_z"
          checked={useBacktestStore.getState().usePriceMaZ}
          onChange={(v) => setField("usePriceMaZ", v)}
        />
      </div>

      {/* Lag features */}
      <div className="mb-4 flex gap-6">
        <div className="flex-1">
          <ParamSlider
            label="Lags"
            paramKey="lags"
            value={useBacktestStore.getState().lags}
            min={1}
            max={60}
            step={1}
            onChange={(v) => setField("lags", v)}
          />
        </div>
        <div className="flex-1">
          <ParamSlider
            label="Lag Depth"
            paramKey="lag_depth"
            value={useBacktestStore.getState().lagDepth}
            min={1}
            max={3}
            step={1}
            onChange={(v) => setField("lagDepth", v)}
          />
        </div>
      </div>

      {/* Advanced toggles */}
      <button
        className="flex w-full items-center gap-2 border-t pt-3 text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ borderColor: "var(--color-border-subtle)", color: "var(--color-text-muted)" }}
        onClick={() => setAdvancedOpen(!advancedOpen)}
      >
        {advancedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Advanced Toggles ({ADVANCED_TOGGLES.length} features)
      </button>
      {advancedOpen && (
        <div className="mt-3 grid grid-cols-3 gap-x-4 gap-y-2">
          {ADVANCED_TOGGLES.map(({ key, label, param }) => (
            <ParamToggle
              key={key}
              label={label}
              paramKey={param}
              checked={useBacktestStore.getState()[key] as boolean}
              onChange={(v) => setField(key, v)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
