import { useBacktestStore } from "@/stores/useBacktestStore";
import { RANGES } from "@/lib/constants";
import { useState, useRef, useEffect } from "react";
import { Info } from "lucide-react";

// ---------------------------------------------------------------------------
// Shared primitives (same style as FeaturesPanel)
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
        className="absolute rounded-full"
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

function NumInput({
  value,
  min,
  max,
  step,
  onChange,
  width = 64,
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

function RowLabel({ children, tip }: { children: React.ReactNode; tip: string }) {
  return (
    <span className="flex items-center gap-1 flex-shrink-0" style={{ color: "#9CA3AF", fontSize: 11, minWidth: 120 }}>
      {children}
      <Tip text={tip} />
    </span>
  );
}

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 pb-2" style={{ borderBottom: "1px solid #2A2E39" }}>
      <span
        className="text-[10px] font-semibold uppercase tracking-widest"
        style={{ color: "#6B7280", letterSpacing: "0.12em" }}
      >
        {label}
      </span>
    </div>
  );
}

// Compact inline slider — no paragraph text, value shown as a number input
function InlineSlider({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2 flex-1" style={{ minWidth: 0 }}>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1"
        style={{
          accentColor: "#3B82F6",
          height: 3,
          cursor: "pointer",
        }}
      />
      <NumInput value={value} min={min} max={max} step={step} onChange={onChange} width={64} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function LabelsPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const useTB = useBacktestStore((s) => s.useTripleBarrier);
  const labelThreshold = useBacktestStore((s) => s.labelThreshold);
  const tbPtMult = useBacktestStore((s) => s.tbPtMult);
  const tbSlMult = useBacktestStore((s) => s.tbSlMult);
  const tbMaxHolding = useBacktestStore((s) => s.tbMaxHolding);
  const tbNeutralZone = useBacktestStore((s) => s.tbNeutralZone);

  const panel: React.CSSProperties = {
    backgroundColor: "#1E222D",
    border: "1px solid #2A2E39",
    borderRadius: 6,
    padding: "10px 14px",
    display: "flex",
    flexDirection: "column",
    gap: 0,
  };

  const rowStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 12,
    paddingTop: 6,
    paddingBottom: 6,
    borderBottom: "1px solid #1A1E2A",
  };

  const rowLast: React.CSSProperties = {
    ...rowStyle,
    borderBottom: "none",
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
      <span
        className="text-[11px] font-semibold uppercase tracking-widest"
        style={{ color: "#6B7280", letterSpacing: "0.14em" }}
      >
        LABELS &amp; TRIPLE BARRIER
      </span>

      {/* ------------------------------------------------------------------ */}
      {/* 1. LABEL CONFIGURATION                                              */}
      {/* ------------------------------------------------------------------ */}
      <div style={panel}>
        <SectionHeader label="Label Configuration" />
        <div style={rowLast}>
          <RowLabel tip="Minimum price move (%) to trigger a label. Smaller = more signals, more noise.">
            Label Threshold
          </RowLabel>
          <InlineSlider
            value={labelThreshold}
            min={RANGES.labelThreshold.min}
            max={RANGES.labelThreshold.max}
            step={RANGES.labelThreshold.step}
            onChange={(v) => setField("labelThreshold", v)}
          />
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* 2. TRIPLE BARRIER                                                   */}
      {/* ------------------------------------------------------------------ */}
      <div style={panel}>
        <SectionHeader label="Triple Barrier" />

        {/* Enable toggle */}
        <div style={rowStyle}>
          <RowLabel tip="Each trade has a profit target (upper barrier), stop-loss (lower barrier), and time limit (vertical barrier). The first one hit determines the label.">
            Use Triple Barrier
          </RowLabel>
          <Toggle checked={useTB} onChange={(v) => setField("useTripleBarrier", v)} />
        </div>

        {/* Barrier params — only visible when enabled */}
        {useTB && (
          <>
            <div style={rowStyle}>
              <RowLabel tip="Profit-target distance expressed as a multiple of ATR volatility.">
                PT Mult
              </RowLabel>
              <InlineSlider
                value={tbPtMult}
                min={RANGES.tbPtMult.min}
                max={RANGES.tbPtMult.max}
                step={RANGES.tbPtMult.step}
                onChange={(v) => setField("tbPtMult", v)}
              />
            </div>

            <div style={rowStyle}>
              <RowLabel tip="Stop-loss distance expressed as a multiple of ATR volatility.">
                SL Mult
              </RowLabel>
              <InlineSlider
                value={tbSlMult}
                min={RANGES.tbSlMult.min}
                max={RANGES.tbSlMult.max}
                step={RANGES.tbSlMult.step}
                onChange={(v) => setField("tbSlMult", v)}
              />
            </div>

            <div style={rowStyle}>
              <RowLabel tip="Maximum bars to hold a position before the vertical (time) barrier fires.">
                Max Holding
              </RowLabel>
              <InlineSlider
                value={tbMaxHolding}
                min={RANGES.tbMaxHolding.min}
                max={RANGES.tbMaxHolding.max}
                step={RANGES.tbMaxHolding.step}
                onChange={(v) => setField("tbMaxHolding", v)}
              />
            </div>

            <div style={rowLast}>
              <RowLabel tip="Price range around entry considered flat / no-direction. Filters sideways chop.">
                Neutral Zone
              </RowLabel>
              <InlineSlider
                value={tbNeutralZone}
                min={RANGES.tbNeutralZone.min}
                max={RANGES.tbNeutralZone.max}
                step={RANGES.tbNeutralZone.step}
                onChange={(v) => setField("tbNeutralZone", v)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
