import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { SELECT_OPTIONS } from "@/lib/constants";
import { useState } from "react";
import { ChevronDown, ChevronRight, Newspaper } from "lucide-react";

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

const sectionStyle: React.CSSProperties = {
  padding: "12px 16px",
  borderRadius: "6px",
  backgroundColor: "var(--color-elevated)",
  border: "1px solid var(--color-border-subtle)",
};

const sectionTitleStyle: React.CSSProperties = {
  color: "var(--color-text-muted)",
  fontSize: "10px",
  fontWeight: 600,
  textTransform: "uppercase" as const,
  letterSpacing: "0.08em",
  marginBottom: "10px",
};

export function FeaturesPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const verbose = useSettingsStore((s) => s.verboseMode);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <div
      className="flex flex-col gap-4 rounded-lg border p-5"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <h3
        className="text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Feature Engineering
      </h3>

      {/* Core indicators — stacked vertically */}
      <div style={sectionStyle}>
        <div style={sectionTitleStyle}>Core Indicators</div>
        <div className="flex flex-col gap-2">
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
      </div>

      {/* FracDiff + Crossover/Price-MA */}
      <div style={sectionStyle}>
        <div style={sectionTitleStyle}>Transformations</div>
        <div className="flex flex-col gap-3">
          <ParamToggle
            label="FracDiff"
            paramKey="use_fracdiff"
            checked={useBacktestStore.getState().useFracdiff}
            description={verbose ? "Fractional differentiation preserves memory while achieving stationarity" : undefined}
            onChange={(v) => setField("useFracdiff", v)}
          />
          {useBacktestStore.getState().useFracdiff && (
            <ParamSlider
              label="fracdiff d"
              paramKey="fracdiff_d"
              value={useBacktestStore.getState().fracdiffD}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setField("fracdiffD", v)}
            />
          )}
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
      </div>

      {/* Lag features */}
      <div style={sectionStyle}>
        <div style={sectionTitleStyle}>Lag Features</div>
        <div className="flex flex-col gap-3">
          <ParamSlider
            label="Lags"
            paramKey="lags"
            value={useBacktestStore.getState().lags}
            min={1}
            max={60}
            step={1}
            onChange={(v) => setField("lags", v)}
          />
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
      <div style={sectionStyle}>
        <button
          className="flex w-full items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-muted)" }}
          onClick={() => setAdvancedOpen(!advancedOpen)}
        >
          {advancedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Advanced Toggles ({ADVANCED_TOGGLES.length} features)
        </button>
        {advancedOpen && (
          <div className="mt-3 flex flex-col gap-2">
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

      {/* News & Sentiment */}
      <div
        className="rounded-md border p-4"
        style={{ borderColor: "var(--color-border-subtle)", backgroundColor: "var(--color-elevated)" }}
      >
        <div className="mb-3 flex items-center gap-2">
          <Newspaper size={14} style={{ color: "var(--color-accent-info)" }} />
          <h4 className="text-xs font-semibold uppercase" style={{ color: "var(--color-accent-info)" }}>
            News &amp; Sentiment
          </h4>
        </div>
        <div className="flex flex-col gap-3">
          <ParamToggle
            label="News Features"
            paramKey="use_news"
            checked={useBacktestStore.getState().useNews}
            description={verbose ? "RSS + economic calendar features: sentiment scores, event flags, news volume" : undefined}
            onChange={(v) => setField("useNews", v)}
          />
          {useBacktestStore.getState().useNews && (
            <div className="flex flex-col gap-3 pl-2">
              <ParamToggle
                label="Event Flags"
                paramKey="news_event_flags"
                checked={useBacktestStore.getState().newsEventFlags}
                description="NFP, FOMC, CPI proximity markers"
                onChange={(v) => setField("newsEventFlags", v)}
              />
              <ParamSelect
                label="Sentiment Engine"
                paramKey="news_sentiment_backend"
                value={useBacktestStore.getState().newsSentimentBackend}
                options={[...SELECT_OPTIONS.newsSentimentBackend]}
                description={useBacktestStore.getState().newsSentimentBackend === "finbert" ? "Requires HuggingFace transformers" : undefined}
                onChange={(v) => setField("newsSentimentBackend", v as "vader" | "finbert")}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
