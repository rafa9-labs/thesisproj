import { useBacktestStore } from "@/stores/useBacktestStore";
import { SELECT_OPTIONS } from "@/lib/constants";
import { useState, useRef, useEffect } from "react";
import { Info, ChevronDown, ChevronRight } from "lucide-react";

// ---------------------------------------------------------------------------
// Tiny inline tooltip
// ---------------------------------------------------------------------------
function Tip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <span ref={ref} className="relative inline-flex items-center" style={{ lineHeight: 0 }}>
      <button
        type="button"
        className="flex items-center justify-center rounded transition-colors"
        style={{ color: "#4B5563", width: 14, height: 14 }}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        aria-label="Info"
      >
        <Info size={11} strokeWidth={1.5} />
      </button>
      {open && (
        <span
          className="absolute z-50 rounded border text-[10px] leading-relaxed"
          style={{
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "#1E222D",
            borderColor: "#2A2E39",
            color: "#9CA3AF",
            padding: "6px 9px",
            width: 220,
            pointerEvents: "none",
            whiteSpace: "normal",
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Compact checkbox
// ---------------------------------------------------------------------------
function Checkbox({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex items-center justify-center rounded-sm border flex-shrink-0 transition-colors"
      style={{
        width: 13,
        height: 13,
        borderColor: checked ? "#3B82F6" : "#374151",
        backgroundColor: checked ? "#3B82F618" : "transparent",
      }}
      aria-pressed={checked}
    >
      {checked && (
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
          <path d="M1 4l2 2 4-4" stroke="#3B82F6" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Compact toggle (on/off pill)
// ---------------------------------------------------------------------------
function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex-shrink-0 rounded-full transition-colors"
      style={{
        width: 28,
        height: 14,
        backgroundColor: checked ? "#1D4ED833" : "#1F2937",
        border: `1px solid ${checked ? "#3B82F6" : "#374151"}`,
        position: "relative",
      }}
      aria-pressed={checked}
    >
      <span
        className="absolute rounded-full transition-transform"
        style={{
          width: 8,
          height: 8,
          top: 2,
          left: checked ? 15 : 3,
          backgroundColor: checked ? "#3B82F6" : "#4B5563",
          transition: "left 0.15s ease",
        }}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Compact number input
// ---------------------------------------------------------------------------
function NumInput({
  value,
  min,
  max,
  step,
  onChange,
  width = 52,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  width?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v)) onChange(Math.min(max, Math.max(min, v)));
      }}
      className="rounded border text-right focus:outline-none"
      style={{
        width,
        height: 24,
        backgroundColor: "#131722",
        borderColor: "#2A2E39",
        color: "#E5E7EB",
        fontSize: 11,
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
        padding: "0 6px",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Compact select
// ---------------------------------------------------------------------------
function CompactSelect({
  value,
  options,
  onChange,
  width = 140,
}: {
  value: string;
  options: readonly { value: string; label: string }[];
  onChange: (v: string) => void;
  width?: number;
}) {
  return (
    <div className="relative flex-shrink-0" style={{ width }}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded border focus:outline-none pr-5"
        style={{
          height: 24,
          backgroundColor: "#131722",
          borderColor: "#2A2E39",
          color: "#E5E7EB",
          fontSize: 11,
          fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
          paddingLeft: 7,
          cursor: "pointer",
        }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} style={{ backgroundColor: "#1E222D" }}>
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown
        size={10}
        className="pointer-events-none absolute"
        style={{ right: 5, top: "50%", transform: "translateY(-50%)", color: "#6B7280" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Row label
// ---------------------------------------------------------------------------
function RowLabel({ children, tip }: { children: React.ReactNode; tip: string }) {
  return (
    <span className="flex items-center gap-1 flex-shrink-0" style={{ color: "#9CA3AF", fontSize: 11, minWidth: 130 }}>
      {children}
      <Tip text={tip} />
    </span>
  );
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------
function SectionHeader({ label, sub }: { label: string; sub?: string }) {
  return (
    <div
      className="flex items-center gap-2 pb-2"
      style={{ borderBottom: "1px solid #2A2E39" }}
    >
      <span
        className="text-[10px] font-semibold uppercase tracking-widest"
        style={{ color: "#6B7280", fontFamily: "inherit", letterSpacing: "0.12em" }}
      >
        {label}
      </span>
      {sub && (
        <span
          className="text-[10px]"
          style={{ color: "#374151", fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)" }}
        >
          {sub}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function FeaturesPanel() {
  const s = useBacktestStore.getState();
  const setField = useBacktestStore((st) => st.setField);
  const [advOpen, setAdvOpen] = useState(false);

  // Subscribe to specific fields so inputs stay reactive
  const useAdx = useBacktestStore((st) => st.useAdx);
  const useAtr = useBacktestStore((st) => st.useAtr);
  const useBbands = useBacktestStore((st) => st.useBbands);
  const useEma = useBacktestStore((st) => st.useEma);
  const useSma = useBacktestStore((st) => st.useSma);
  const useRsi = useBacktestStore((st) => st.useRsi);
  const useMacd = useBacktestStore((st) => st.useMacd);
  const useStoch = useBacktestStore((st) => st.useStoch);
  const useSar = useBacktestStore((st) => st.useSar);
  const useDonchian = useBacktestStore((st) => st.useDonchian);

  const useFracdiff = useBacktestStore((st) => st.useFracdiff);
  const fracdiffD = useBacktestStore((st) => st.fracdiffD);
  const useCrossoverBins = useBacktestStore((st) => st.useCrossoverBins);
  const usePriceMaZ = useBacktestStore((st) => st.usePriceMaZ);
  const useMaSpread = useBacktestStore((st) => st.useMaSpread);
  const useIndicatorStates = useBacktestStore((st) => st.useIndicatorStates);

  const lags = useBacktestStore((st) => st.lags);
  const lagDepth = useBacktestStore((st) => st.lagDepth);

  const useMtfMa = useBacktestStore((st) => st.useMtfMa);
  const useMtfAlignment = useBacktestStore((st) => st.useMtfAlignment);
  const useMacdAtrRatio = useBacktestStore((st) => st.useMacdAtrRatio);
  const useTripleConfirm = useBacktestStore((st) => st.useTripleConfirm);
  const useTrendConfirm = useBacktestStore((st) => st.useTrendConfirm);
  const useVolManagedMom = useBacktestStore((st) => st.useVolManagedMom);
  const useSqueezeBreakout = useBacktestStore((st) => st.useSqueezeBreakout);
  const useSqueezeExpansion = useBacktestStore((st) => st.useSqueezeExpansion);
  const useAtrChannelBreakout = useBacktestStore((st) => st.useAtrChannelBreakout);
  const useExtAtrLowAdx = useBacktestStore((st) => st.useExtAtrLowAdx);
  const useReentryMom = useBacktestStore((st) => st.useReentryMom);
  const useSlopeDiff = useBacktestStore((st) => st.useSlopeDiff);
  const useRvFeatures = useBacktestStore((st) => st.useRvFeatures);

  const useNews = useBacktestStore((st) => st.useNews);
  const newsEventFlags = useBacktestStore((st) => st.newsEventFlags);
  const newsSentimentBackend = useBacktestStore((st) => st.newsSentimentBackend);
  const llmSentimentEnabled = useBacktestStore((st) => st.llmSentimentEnabled);
  const llmBackend = useBacktestStore((st) => st.llmBackend);

  // Cell style for form grid rows
  const rowStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 12,
    paddingTop: 6,
    paddingBottom: 6,
    borderBottom: "1px solid #1A1E2A",
  };

  const panel: React.CSSProperties = {
    backgroundColor: "#1E222D",
    border: "1px solid #2A2E39",
    borderRadius: 6,
    padding: "10px 14px",
    display: "flex",
    flexDirection: "column",
    gap: 0,
  };

  return (
    <div
      className="flex flex-col gap-4"
      style={{
        backgroundColor: "#131722",
        border: "1px solid #2A2E39",
        borderRadius: 6,
        padding: "14px 16px",
      }}
    >
      {/* Section title */}
      <div className="flex items-center justify-between">
        <span
          className="text-[11px] font-semibold uppercase tracking-widest"
          style={{ color: "#6B7280", letterSpacing: "0.14em" }}
        >
          FEATURES
        </span>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* 1. CORE INDICATORS                                                  */}
      {/* ------------------------------------------------------------------ */}
      <div style={panel}>
        <SectionHeader label="Core Indicators" />
        <div style={{ ...rowStyle, flexWrap: "wrap", gap: 8, paddingTop: 8, borderBottom: "none" }}>
          {([
            { key: "useAdx", label: "ADX", val: useAdx, tip: "Average Directional Index. Measures trend strength regardless of direction." },
            { key: "useAtr", label: "ATR", val: useAtr, tip: "Average True Range. Volatility measure for position sizing and stops." },
            { key: "useBbands", label: "Bollinger", val: useBbands, tip: "Bands around a moving average. Signals overbought/oversold conditions." },
            { key: "useEma", label: "EMA", val: useEma, tip: "Exponential Moving Average. Faster-reacting trend follower than SMA." },
            { key: "useSma", label: "SMA", val: useSma, tip: "Simple Moving Average. Classic trend smoothing at the chosen period." },
            { key: "useRsi", label: "RSI", val: useRsi, tip: "Relative Strength Index. Momentum oscillator for reversal signals." },
            { key: "useMacd", label: "MACD", val: useMacd, tip: "Moving Average Convergence Divergence. Trend-following momentum indicator." },
            { key: "useDonchian", label: "Donchian", val: useDonchian, tip: "Channel breakout indicator. Highest high / lowest low over N periods." },
            { key: "useStoch", label: "Stochastic", val: useStoch, tip: "Compares closing price to its range over time. Reversal detector." },
            { key: "useSar", label: "SAR", val: useSar, tip: "Parabolic Stop and Reverse. Trailing stop that flips with trend changes." },
          ] as const).map(({ key, label, val, tip }) => (
            <button
              key={key}
              type="button"
              onClick={() => setField(key as Parameters<typeof setField>[0], !val)}
              className="flex items-center gap-1.5 rounded border transition-colors"
              style={{
                height: 24,
                padding: "0 8px",
                fontSize: 11,
                fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
                backgroundColor: val ? "#1D4ED818" : "#131722",
                borderColor: val ? "#3B82F655" : "#2A2E39",
                color: val ? "#60A5FA" : "#6B7280",
                cursor: "pointer",
              }}
            >
              {/* LED indicator: blue when active, dim red when inactive */}
              <span
                className="rounded-full flex-shrink-0"
                style={{
                  width: 5,
                  height: 5,
                  backgroundColor: val ? "#3B82F6" : "#7F1D1D",
                  boxShadow: val ? "0 0 4px #3B82F6" : "none",
                }}
              />
              {label}
              <Tip text={tip} />
            </button>
          ))}
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* 2. TRANSFORMATIONS & LAGS                                           */}
      {/* ------------------------------------------------------------------ */}
      <div style={panel}>
        <SectionHeader label="Transformations & Lags" />

        <div style={rowStyle}>
          <RowLabel tip="Fractional differentiation preserves long-memory while achieving stationarity.">FracDiff</RowLabel>
          <Toggle checked={useFracdiff} onChange={(v) => setField("useFracdiff", v)} />
          {useFracdiff && (
            <>
              <span style={{ fontSize: 10, color: "#4B5563", fontFamily: "var(--font-mono, monospace)" }}>d =</span>
              <NumInput value={fracdiffD} min={0} max={1} step={0.05} onChange={(v) => setField("fracdiffD", v)} width={52} />
            </>
          )}
        </div>

        <div style={rowStyle}>
          <RowLabel tip="Categorical bins for price crossing moving averages.">Crossover Bins</RowLabel>
          <Toggle checked={useCrossoverBins} onChange={(v) => setField("useCrossoverBins", v)} />
          <span style={{ width: 24 }} />
          <RowLabel tip="Standardized distance from price to its moving average.">Price-MA Z</RowLabel>
          <Toggle checked={usePriceMaZ} onChange={(v) => setField("usePriceMaZ", v)} />
          <span style={{ width: 24 }} />
          <RowLabel tip="Distance between fast and slow MAs as a raw feature.">MA Spread</RowLabel>
          <Toggle checked={useMaSpread} onChange={(v) => setField("useMaSpread", v)} />
          <span style={{ width: 24 }} />
          <RowLabel tip="Categorical buckets for each indicator's current regime.">Indicator States</RowLabel>
          <Toggle checked={useIndicatorStates} onChange={(v) => setField("useIndicatorStates", v)} />
        </div>

        <div style={{ ...rowStyle, borderBottom: "none" }}>
          <RowLabel tip="Number of past bars to include as lagged features.">Lags</RowLabel>
          <NumInput value={lags} min={1} max={60} step={1} onChange={(v) => setField("lags", v)} width={52} />
          <span style={{ width: 24 }} />
          <RowLabel tip="Granularity step size between lagged values. 1 = every bar, 2 = every other bar.">Lag Depth</RowLabel>
          <NumInput value={lagDepth} min={1} max={3} step={1} onChange={(v) => setField("lagDepth", v)} width={52} />
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* 3. ADVANCED SIGNAL FILTERS                                          */}
      {/* ------------------------------------------------------------------ */}
      <div style={panel}>
        <button
          type="button"
          className="flex items-center gap-2 w-full transition-colors"
          style={{ padding: "0 0 6px 0", borderBottom: "1px solid #2A2E39", background: "none", cursor: "pointer" }}
          onClick={() => setAdvOpen((v) => !v)}
        >
          {advOpen
            ? <ChevronDown size={12} style={{ color: "#4B5563", flexShrink: 0 }} />
            : <ChevronRight size={12} style={{ color: "#4B5563", flexShrink: 0 }} />
          }
          <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "#6B7280", letterSpacing: "0.12em" }}>
            Advanced Signal Filters
          </span>
          <span style={{ fontSize: 10, color: "#374151", fontFamily: "var(--font-mono, monospace)", marginLeft: "auto" }}>
            {[useMtfMa, useMtfAlignment, useMacdAtrRatio, useTripleConfirm, useTrendConfirm,
              useVolManagedMom, useSqueezeBreakout, useSqueezeExpansion, useAtrChannelBreakout,
              useExtAtrLowAdx, useReentryMom, useSlopeDiff, useRvFeatures].filter(Boolean).length} / 13 active
          </span>
        </button>

        {advOpen && (
          <div style={{ paddingTop: 6, display: "flex", flexDirection: "column", gap: 0 }}>
            {([
              { key: "useMtfMa", val: useMtfMa, label: "MTF MA", tip: "Multi-timeframe moving-average alignment filter." },
              { key: "useMtfAlignment", val: useMtfAlignment, label: "MTF Alignment", tip: "Confirms trend direction across multiple timeframes." },
              { key: "useMacdAtrRatio", val: useMacdAtrRatio, label: "MACD/ATR Ratio", tip: "Normalizes MACD momentum by volatility." },
              { key: "useTripleConfirm", val: useTripleConfirm, label: "Triple Confirm", tip: "Requires three independent signals before triggering a trade." },
              { key: "useTrendConfirm", val: useTrendConfirm, label: "Trend Confirm", tip: "Only trade when higher-timeframe trend agrees." },
              { key: "useVolManagedMom", val: useVolManagedMom, label: "Vol-Managed Mom", tip: "Scales momentum exposure by inverse volatility." },
              { key: "useSqueezeBreakout", val: useSqueezeBreakout, label: "Squeeze Breakout", tip: "Detects Bollinger-Band compression before volatility expansion." },
              { key: "useSqueezeExpansion", val: useSqueezeExpansion, label: "Squeeze Expansion", tip: "Measures volatility expansion after a squeeze period." },
              { key: "useAtrChannelBreakout", val: useAtrChannelBreakout, label: "ATR Channel Breakout", tip: "Price breaking ATR-derived channels." },
              { key: "useExtAtrLowAdx", val: useExtAtrLowAdx, label: "Ext ATR Low ADX", tip: "Combines extreme volatility with low trend strength." },
              { key: "useReentryMom", val: useReentryMom, label: "Re-entry Momentum", tip: "Signals for re-entering after a pullback within a trend." },
              { key: "useSlopeDiff", val: useSlopeDiff, label: "Slope Diff", tip: "Difference in price slope across lookback windows." },
              { key: "useRvFeatures", val: useRvFeatures, label: "RV Features", tip: "Realized volatility estimates from intraday ranges." },
            ] as const).map(({ key, val, label, tip }, i, arr) => (
              <div key={key} style={{ ...rowStyle, borderBottom: i < arr.length - 1 ? "1px solid #1A1E2A" : "none" }}>
                <Checkbox checked={val} onChange={(v) => setField(key as Parameters<typeof setField>[0], v)} />
                <span style={{ fontSize: 11, color: val ? "#D1D5DB" : "#6B7280", minWidth: 160, fontFamily: "inherit" }}>{label}</span>
                <Tip text={tip} />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* 4. NEWS & SENTIMENT                                                 */}
      {/* ------------------------------------------------------------------ */}
      <div style={panel}>
        <SectionHeader label="News & Sentiment" />

        <div style={rowStyle}>
          <RowLabel tip="RSS + economic calendar features: sentiment scores, event flags, news volume.">News Features</RowLabel>
          <Toggle checked={useNews} onChange={(v) => setField("useNews", v)} />
          {useNews && (
            <>
              <span style={{ width: 24 }} />
              <RowLabel tip="NFP, FOMC, CPI proximity markers before/after releases.">Event Flags</RowLabel>
              <Toggle checked={newsEventFlags} onChange={(v) => setField("newsEventFlags", v)} />
              <span style={{ width: 24 }} />
              <RowLabel tip="VADER = fast rule-based. finBERT = slower but more accurate.">Sentiment Engine</RowLabel>
              <CompactSelect
                value={newsSentimentBackend}
                options={SELECT_OPTIONS.newsSentimentBackend}
                onChange={(v) => setField("newsSentimentBackend", v as "vader" | "finbert")}
                width={160}
              />
            </>
          )}
        </div>

        {useNews && (
          <div style={{ ...rowStyle, borderBottom: "none" }}>
            <RowLabel tip="Use a local or cloud LLM to score news with directional sentiment. Falls back to VADER if unavailable.">LLM Sentiment</RowLabel>
            <Toggle checked={llmSentimentEnabled} onChange={(v) => setField("llmSentimentEnabled", v)} />
            {llmSentimentEnabled && (
              <>
                <span style={{ width: 24 }} />
                <RowLabel tip="Ollama = free, local, private. OpenAI / Anthropic = paid, cloud.">LLM Backend</RowLabel>
                <CompactSelect
                  value={llmBackend}
                  options={SELECT_OPTIONS.llmBackend}
                  onChange={(v) => setField("llmBackend", v as "ollama" | "openai" | "anthropic")}
                  width={160}
                />
              </>
            )}
          </div>
        )}
      </div>


    </div>
  );
}
