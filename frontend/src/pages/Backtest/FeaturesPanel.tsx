import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";
import { SELECT_OPTIONS } from "@/lib/constants";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

const CORE_INDICATORS = [
  {
    key: "useAdx" as const,
    label: "ADX",
    description: "Average Directional Index. Measures trend strength regardless of direction.",
  },
  {
    key: "useAtr" as const,
    label: "ATR",
    description: "Average True Range. Volatility measure for position sizing and stops.",
  },
  {
    key: "useBbands" as const,
    label: "BOLLINGER",
    description: "Bands around a moving average. Signals overbought/oversold conditions.",
  },
  {
    key: "useEma" as const,
    label: "EMA",
    description: "Exponential Moving Average. Faster-reacting trend follower than SMA.",
  },
  {
    key: "useSma" as const,
    label: "SMA",
    description: "Simple Moving Average. Classic trend smoothing at the chosen period.",
  },
  {
    key: "useRsi" as const,
    label: "RSI",
    description: "Relative Strength Index. Momentum oscillator for reversal signals.",
  },
  {
    key: "useMacd" as const,
    label: "MACD",
    description: "Moving Average Convergence Divergence. Trend-following momentum indicator.",
  },
  {
    key: "useDonchian" as const,
    label: "DONCHIAN",
    description: "Channel breakout indicator. Highest high / lowest low over N periods.",
  },
  {
    key: "useStoch" as const,
    label: "STOCHASTIC",
    description: "Compares closing price to its range over time. Reversal detector.",
  },
  {
    key: "useSar" as const,
    label: "SAR",
    description: "Parabolic Stop and Reverse. Trailing stop that flips with trend changes.",
  },
];

const ADVANCED_TOGGLES = [
  {
    key: "useMtfMa" as const,
    label: "MTF MA",
    param: "use_mtf_ma",
    description: "Multi-timeframe moving-average alignment filter.",
  },
  {
    key: "useMtfAlignment" as const,
    label: "MTF ALIGNMENT",
    param: "use_mtf_alignment",
    description: "Confirms trend direction across multiple timeframes.",
  },
  {
    key: "useMacdAtrRatio" as const,
    label: "MACD/ATR RATIO",
    param: "use_macd_atr_ratio",
    description: "Normalizes MACD momentum by volatility.",
  },
  {
    key: "useTripleConfirm" as const,
    label: "TRIPLE CONFIRM",
    param: "use_triple_confirm",
    description: "Requires three independent signals before triggering a trade.",
  },
  {
    key: "useTrendConfirm" as const,
    label: "TREND CONFIRM",
    param: "use_trend_confirm",
    description: "Only trade when higher-timeframe trend agrees.",
  },
  {
    key: "useVolManagedMom" as const,
    label: "VOL-MANAGED MOM",
    param: "use_vol_managed_mom",
    description: "Scales momentum exposure by inverse volatility.",
  },
  {
    key: "useMaSpread" as const,
    label: "MA SPREAD",
    param: "use_ma_spread",
    description: "Distance between fast and slow MAs as a feature.",
  },
  {
    key: "useSlopeDiff" as const,
    label: "SLOPE DIFF",
    param: "use_slope_diff",
    description: "Difference in price slope across lookback windows.",
  },
  {
    key: "useSqueezeBreakout" as const,
    label: "SQUEEZE BREAKOUT",
    param: "use_squeeze_breakout",
    description: "Detects Bollinger-Band compression before volatility expansion.",
  },
  {
    key: "useSqueezeExpansion" as const,
    label: "SQUEEZE EXPANSION",
    param: "use_squeeze_expansion",
    description: "Measures volatility expansion after a squeeze period.",
  },
  {
    key: "useAtrChannelBreakout" as const,
    label: "ATR CHANNEL BREAKOUT",
    param: "use_atr_channel_breakout",
    description: "Price breaking ATR-derived channels.",
  },
  {
    key: "useExtAtrLowAdx" as const,
    label: "EXT ATR LOW ADX",
    param: "use_ext_atr_low_adx",
    description: "Combines extreme volatility with low trend strength.",
  },
  {
    key: "useReentryMom" as const,
    label: "RE-ENTRY MOMENTUM",
    param: "use_reentry_mom",
    description: "Signals for re-entering after a pullback within a trend.",
  },
  {
    key: "useRvFeatures" as const,
    label: "RV FEATURES",
    param: "use_rv_features",
    description: "Realized volatility estimates from intraday ranges.",
  },
  {
    key: "useIndicatorStates" as const,
    label: "INDICATOR STATES",
    param: "use_indicator_states",
    description: "Categorical buckets for each indicator's current regime.",
  },
];

export function FeaturesPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const state = useBacktestStore.getState();

  return (
    <Panel>
      <PanelHeader title="Feature Engineering" subtitle="Select and configure input signals." />

      {/* Core Indicators */}
      <Section title="Core Indicators">
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 items-start">
          {CORE_INDICATORS.map(({ key, label, description }) => (
            <ParamToggle
              key={key}
              label={label}
              checked={state[key] as boolean}
              tooltip={description}
              compact
              onChange={(v) => setField(key, v)}
            />
          ))}
        </div>
      </Section>

      {/* Transformations */}
      <Section title="Transformations">
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 items-start">
          <div className="flex flex-col gap-3">
            <ParamToggle
              label="FRACDIFF"
              checked={state.useFracdiff}
              tooltip="Fractional differentiation preserves long-memory while achieving stationarity."
              compact
              onChange={(v) => setField("useFracdiff", v)}
            />
            {state.useFracdiff && (
              <div className="px-3">
                <ParamSlider
                  label="FRACDIFF D"
                  value={state.fracdiffD}
                  min={0}
                  max={1}
                  step={0.05}
                  tooltip="Differentiation degree. 0 = no change, 1 = full difference."
                  onChange={(v) => setField("fracdiffD", v)}
                />
              </div>
            )}
          </div>
          <ParamToggle
            label="CROSSOVER BINS"
            checked={state.useCrossoverBins}
            tooltip="Categorical bins for price crossing moving averages."
            compact
            onChange={(v) => setField("useCrossoverBins", v)}
          />
          <ParamToggle
            label="PRICE-MA Z-SCORE"
            checked={state.usePriceMaZ}
            tooltip="Standardized distance from price to its moving average."
            compact
            onChange={(v) => setField("usePriceMaZ", v)}
          />
        </div>
      </Section>

      {/* Lag Features */}
      <Section title="Lag Features">
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          <ParamSlider
            label="LAGS"
            value={state.lags}
            min={1}
            max={60}
            step={1}
            tooltip="Number of past bars to include as lagged features."
            onChange={(v) => setField("lags", v)}
          />
          <ParamSlider
            label="LAG DEPTH"
            value={state.lagDepth}
            min={1}
            max={3}
            step={1}
            tooltip="Granularity step size between lagged values."
            onChange={(v) => setField("lagDepth", v)}
          />
        </div>
      </Section>

      {/* Advanced Toggles */}
      <Section title="Advanced Toggles">
        <button
          className="mb-4 flex w-full items-center gap-2 text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase transition-colors duration-200 hover:text-[var(--color-text-primary)]"
          onClick={() => setAdvancedOpen(!advancedOpen)}
        >
          {advancedOpen ? (
            <ChevronDown size={14} strokeWidth={1.5} />
          ) : (
            <ChevronRight size={14} strokeWidth={1.5} />
          )}
          {ADVANCED_TOGGLES.length} features {advancedOpen ? "(collapse)" : "(expand)"}
        </button>
        {advancedOpen && (
          <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 items-start">
            {ADVANCED_TOGGLES.map(({ key, label, param, description }) => (
              <ParamToggle
                key={key}
                label={label}
                paramKey={param}
                checked={state[key] as boolean}
                tooltip={description}
                compact
                onChange={(v) => setField(key, v)}
              />
            ))}
          </div>
        )}
      </Section>

      {/* News & Sentiment */}
      <Section title="News & Sentiment">
        <div className="flex flex-col gap-2.5">
          <ParamToggle
            label="NEWS FEATURES"
            checked={state.useNews}
            tooltip="RSS + economic calendar features: sentiment scores, event flags, news volume."
            onChange={(v) => setField("useNews", v)}
          />
          {state.useNews && (
            <div className="flex flex-col gap-5 pl-2">
              <ParamToggle
                label="EVENT FLAGS"
                checked={state.newsEventFlags}
                tooltip="NFP, FOMC, CPI proximity markers before/after releases."
                onChange={(v) => setField("newsEventFlags", v)}
              />
              <div className="max-w-sm">
                <ParamSelect
                  label="SENTIMENT ENGINE"
                  value={state.newsSentimentBackend}
                  options={[...SELECT_OPTIONS.newsSentimentBackend]}
                  tooltip={
                    state.newsSentimentBackend === "finbert"
                      ? "Requires HuggingFace transformers installed."
                      : "VADER = fast rule-based. finBERT = slower but more accurate."
                  }
                  onChange={(v) => setField("newsSentimentBackend", v as "vader" | "finbert")}
                />
              </div>

              {/* LLM Sentiment */}
              {state.llmSentimentEnabled !== undefined && (
                <div className="mt-2 flex flex-col gap-4 border-l-2 border-cyan-500/30 pl-4">
                  <ParamToggle
                    label="LLM SENTIMENT"
                    checked={state.llmSentimentEnabled}
                    tooltip="Use Ollama/OpenAI/Anthropic LLM to score news with directional sentiment. Falls back to VADER if unavailable."
                    onChange={(v) => setField("llmSentimentEnabled", v)}
                  />
                  {state.llmSentimentEnabled && (
                    <>
                      <div className="max-w-sm">
                        <ParamSelect
                          label="LLM BACKEND"
                          value={state.llmBackend}
                          options={[...SELECT_OPTIONS.llmBackend]}
                          tooltip={
                            state.llmBackend === "ollama"
                              ? "Free, local, private. Requires Ollama running."
                              : "Paid, cloud. Requires API key."
                          }
                          onChange={(v) =>
                            setField("llmBackend", v as "ollama" | "openai" | "anthropic")
                          }
                        />
                      </div>
                      {(state.llmBackend === "openai" || state.llmBackend === "anthropic") && (
                        <div className="max-w-sm">
                          <label className="mb-1.5 block text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                            API KEY
                          </label>
                          <input
                            type="password"
                            value={state.llmApiKey || ""}
                            placeholder={state.llmBackend === "openai" ? "sk-..." : "sk-ant-..."}
                            onChange={(e) => setField("llmApiKey", e.target.value)}
                            className="w-full rounded-md border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-1.5 font-mono text-xs text-(--color-text-primary) focus:outline-none"
                          />
                        </div>
                      )}
                      <div className="max-w-sm">
                        <label className="mb-1.5 block text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                          MODEL
                        </label>
                        <input
                          type="text"
                          value={state.llmModel}
                          placeholder={
                            state.llmBackend === "ollama"
                              ? "llama3"
                              : state.llmBackend === "openai"
                                ? "gpt-4o-mini"
                                : "claude-3-haiku-20240307"
                          }
                          onChange={(e) => setField("llmModel", e.target.value)}
                          className="w-full rounded-md border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-1.5 font-mono text-xs text-(--color-text-primary) focus:outline-none"
                        />
                      </div>
                      <div className="max-w-sm">
                        <ParamSlider
                          label="LLM WEIGHT"
                          value={Number(state.llmWeight)}
                          min={0}
                          max={1}
                          step={0.1}
                          tooltip={`Weight of LLM sentiment vs VADER (${((1 - Number(state.llmWeight)) * 100).toFixed(0)}% VADER).`}
                          onChange={(v) => setField("llmWeight", v)}
                        />
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </Section>
    </Panel>
  );
}
