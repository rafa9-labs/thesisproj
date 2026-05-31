import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { SELECT_OPTIONS } from "@/lib/constants";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

const CORE_INDICATORS = [
  { key: "useAdx" as const, label: "ADX", description: "Average Directional Index. Measures trend strength regardless of direction." },
  { key: "useAtr" as const, label: "ATR", description: "Average True Range. Volatility measure for position sizing and stops." },
  { key: "useBbands" as const, label: "Bollinger", description: "Bands around a moving average. Signals overbought/oversold conditions." },
  { key: "useEma" as const, label: "EMA", description: "Exponential Moving Average. Faster-reacting trend follower than SMA." },
  { key: "useSma" as const, label: "SMA", description: "Simple Moving Average. Classic trend smoothing at the chosen period." },
  { key: "useRsi" as const, label: "RSI", description: "Relative Strength Index. Momentum oscillator for reversal signals." },
  { key: "useMacd" as const, label: "MACD", description: "Moving Average Convergence Divergence. Trend-following momentum indicator." },
  { key: "useDonchian" as const, label: "Donchian", description: "Channel breakout indicator. Highest high / lowest low over N periods." },
  { key: "useStoch" as const, label: "Stochastic", description: "Compares closing price to its range over time. Reversal detector." },
  { key: "useSar" as const, label: "SAR", description: "Parabolic Stop and Reverse. Trailing stop that flips with trend changes." },
];

const ADVANCED_TOGGLES = [
  { key: "useMtfMa" as const, label: "MTF MA", param: "use_mtf_ma", description: "Multi-timeframe moving-average alignment filter." },
  { key: "useMtfAlignment" as const, label: "MTF Alignment", param: "use_mtf_alignment", description: "Confirms trend direction across multiple timeframes." },
  { key: "useMacdAtrRatio" as const, label: "MACD/ATR Ratio", param: "use_macd_atr_ratio", description: "Normalizes MACD momentum by volatility." },
  { key: "useTripleConfirm" as const, label: "Triple Confirm", param: "use_triple_confirm", description: "Requires three independent signals before triggering a trade." },
  { key: "useTrendConfirm" as const, label: "Trend Confirm", param: "use_trend_confirm", description: "Only trade when higher-timeframe trend agrees." },
  { key: "useVolManagedMom" as const, label: "Vol-Managed Mom", param: "use_vol_managed_mom", description: "Scales momentum exposure by inverse volatility." },
  { key: "useMaSpread" as const, label: "MA Spread", param: "use_ma_spread", description: "Distance between fast and slow MAs as a feature." },
  { key: "useSlopeDiff" as const, label: "Slope Diff", param: "use_slope_diff", description: "Difference in price slope across lookback windows." },
  { key: "useSqueezeBreakout" as const, label: "Squeeze Breakout", param: "use_squeeze_breakout", description: "Detects Bollinger-Band compression before volatility expansion." },
  { key: "useSqueezeExpansion" as const, label: "Squeeze Expansion", param: "use_squeeze_expansion", description: "Measures volatility expansion after a squeeze period." },
  { key: "useAtrChannelBreakout" as const, label: "ATR Channel Breakout", param: "use_atr_channel_breakout", description: "Price breaking ATR-derived channels." },
  { key: "useExtAtrLowAdx" as const, label: "Ext ATR Low ADX", param: "use_ext_atr_low_adx", description: "Combines extreme volatility with low trend strength." },
  { key: "useReentryMom" as const, label: "Re-entry Momentum", param: "use_reentry_mom", description: "Signals for re-entering after a pullback within a trend." },
  { key: "useRvFeatures" as const, label: "RV Features", param: "use_rv_features", description: "Realized volatility estimates from intraday ranges." },
  { key: "useIndicatorStates" as const, label: "Indicator States", param: "use_indicator_states", description: "Categorical buckets for each indicator's current regime." },
];

const sectionClass = "rounded-xl border p-6";
const sectionStyle: React.CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};
const sectionTitleClass = "mb-1 text-[11px] font-medium uppercase tracking-[0.12em]";
const sectionTitleStyle: React.CSSProperties = { color: "var(--color-text-secondary)" };
const explainerClass = "mb-3 text-[11px] font-light leading-relaxed max-w-[720px]";
const explainerStyle: React.CSSProperties = { color: "var(--color-text-muted)" };

export function FeaturesPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const verbose = useSettingsStore((s) => s.verboseMode);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const state = useBacktestStore.getState();

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
        Feature Engineering
      </h3>

      {/* Core Indicators */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>Core Indicators</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          Standard technical indicators that form the base signal set. Toggle each to include it in the model input vector.
        </p>
        <div className="flex flex-col">
          {CORE_INDICATORS.map(({ key, label, description }) => (
            <ParamToggle
              key={key}
              label={label}
              checked={state[key] as boolean}
              description={verbose ? description : undefined}
              onChange={(v) => setField(key, v)}
            />
          ))}
        </div>
      </section>

      {/* Transformations */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>Transformations</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          Derived features that preprocess raw price data into more informative representations.
        </p>
        <div className="flex flex-col">
          <ParamToggle
            label="FracDiff"
            checked={state.useFracdiff}
            description={verbose ? "Fractional differentiation preserves long-memory while achieving stationarity." : "Preserves memory while making series stationary."}
            onChange={(v) => setField("useFracdiff", v)}
          />
          {state.useFracdiff && (
            <div className="max-w-xs">
              <ParamSlider
                label="FracDiff d"
                value={state.fracdiffD}
                min={0}
                max={1}
                step={0.05}
                description="Differentiation degree. 0 = no change, 1 = full difference."
                onChange={(v) => setField("fracdiffD", v)}
              />
            </div>
          )}
          <ParamToggle
            label="Crossover Bins"
            checked={state.useCrossoverBins}
            description="Categorical bins for price crossing moving averages."
            onChange={(v) => setField("useCrossoverBins", v)}
          />
          <ParamToggle
            label="Price-MA Z-Score"
            checked={state.usePriceMaZ}
            description="Standardized distance from price to its moving average."
            onChange={(v) => setField("usePriceMaZ", v)}
          />
        </div>
      </section>

      {/* Lag Features */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>Lag Features</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          Lookback windows that feed historical values into the model as additional dimensions.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6 max-w-lg">
          <ParamSlider
            label="Lags"
            value={state.lags}
            min={1}
            max={60}
            step={1}
            description="Number of past bars to include as lagged features."
            onChange={(v) => setField("lags", v)}
          />
          <ParamSlider
            label="Lag Depth"
            value={state.lagDepth}
            min={1}
            max={3}
            step={1}
            description="Granularity step size between lagged values."
            onChange={(v) => setField("lagDepth", v)}
          />
        </div>
      </section>

      {/* Advanced Toggles */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>Advanced Toggles</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          Experimental features for power users. Each adds a specialized signal layer to the feature vector.
        </p>
        <button
          className="flex w-full items-center gap-2 text-[11px] font-medium uppercase tracking-[0.1em] transition-colors duration-200 hover:text-[var(--color-text-primary)] mb-4"
          style={{ color: "var(--color-text-muted)" }}
          onClick={() => setAdvancedOpen(!advancedOpen)}
        >
          {advancedOpen ? <ChevronDown size={14} strokeWidth={1.5} /> : <ChevronRight size={14} strokeWidth={1.5} />}
          {ADVANCED_TOGGLES.length} features {advancedOpen ? "(collapse)" : "(expand)"}
        </button>
        {advancedOpen && (
          <div className="flex flex-col">
            {ADVANCED_TOGGLES.map(({ key, label, param, description }) => (
              <ParamToggle
                key={key}
                label={label}
                paramKey={param}
                checked={state[key] as boolean}
                description={verbose ? description : undefined}
                onChange={(v) => setField(key, v)}
              />
            ))}
          </div>
        )}
      </section>

      {/* News & Sentiment */}
      <section className={sectionClass} style={sectionStyle}>
        <div className="flex items-center gap-2 mb-1">
          <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
          <h4 className={sectionTitleClass} style={sectionTitleStyle}>News &amp; Sentiment</h4>
        </div>
        <p className={explainerClass} style={explainerStyle}>
          External sentiment signals from RSS feeds and economic calendars. Adds a macro-awareness layer to the model.
        </p>
        <div className="flex flex-col">
          <ParamToggle
            label="News Features"
            checked={state.useNews}
            description={verbose ? "RSS + economic calendar features: sentiment scores, event flags, news volume." : "Enable sentiment and event-based features."}
            onChange={(v) => setField("useNews", v)}
          />
          {state.useNews && (
            <div className="flex flex-col gap-5 pl-2">
              <ParamToggle
                label="Event Flags"
                checked={state.newsEventFlags}
                description="NFP, FOMC, CPI proximity markers before/after releases."
                onChange={(v) => setField("newsEventFlags", v)}
              />
              <div className="max-w-sm">
                <ParamSelect
                  label="Sentiment Engine"
                  value={state.newsSentimentBackend}
                  options={[...SELECT_OPTIONS.newsSentimentBackend]}
                  description={state.newsSentimentBackend === "finbert" ? "Requires HuggingFace transformers installed." : "VADER = fast rule-based. finBERT = slower but more accurate."}
                  onChange={(v) => setField("newsSentimentBackend", v as "vader" | "finbert")}
                />
              </div>

              {/* LLM Sentiment */}
              {state.llmSentimentEnabled !== undefined && (
                <div className="flex flex-col gap-4 pl-4 border-l-2 border-cyan-500/30 mt-2">
                  <ParamToggle
                    label="LLM Sentiment"
                    checked={state.llmSentimentEnabled}
                    description="Use Ollama/OpenAI/Anthropic LLM to score news with directional sentiment. Falls back to VADER if unavailable."
                    onChange={(v) => setField("llmSentimentEnabled", v)}
                  />
                  {state.llmSentimentEnabled && (
                    <>
                      <div className="max-w-sm">
                        <ParamSelect
                          label="LLM Backend"
                          value={state.llmBackend}
                          options={[...SELECT_OPTIONS.llmBackend]}
                          description={state.llmBackend === "ollama" ? "Free, local, private. Requires Ollama running." : "Paid, cloud. Requires API key."}
                          onChange={(v) => setField("llmBackend", v as "ollama" | "openai" | "anthropic")}
                        />
                      </div>
                      {(state.llmBackend === "openai" || state.llmBackend === "anthropic") && (
                        <div className="max-w-sm">
                          <label className="text-sm text-gray-400 block mb-1">API Key</label>
                          <input
                            type="password"
                            value={state.llmApiKey || ""}
                            placeholder={state.llmBackend === "openai" ? "sk-..." : "sk-ant-..."}
                            onChange={(e) => setField("llmApiKey", e.target.value)}
                            className="w-full rounded-md border border-gray-600 bg-gray-800 px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                          />
                        </div>
                      )}
                      <div className="max-w-sm">
                        <label className="text-sm text-gray-400 block mb-1">Model</label>
                        <input
                          type="text"
                          value={state.llmModel}
                          placeholder={state.llmBackend === "ollama" ? "llama3" : state.llmBackend === "openai" ? "gpt-4o-mini" : "claude-3-haiku-20240307"}
                          onChange={(e) => setField("llmModel", e.target.value)}
                          className="w-full rounded-md border border-gray-600 bg-gray-800 px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                        />
                      </div>
                      <div className="max-w-sm">
                        <label className="text-sm text-gray-400 block mb-1">
                          LLM Weight: {Number(state.llmWeight).toFixed(1)} vs VADER {(((1 - Number(state.llmWeight)) * 100)).toFixed(0)}%
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={state.llmWeight}
                          onChange={(e) => setField("llmWeight", parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
