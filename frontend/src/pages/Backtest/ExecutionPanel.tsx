import { useState, useRef, useEffect } from "react";
import { Info } from "lucide-react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { SELECT_OPTIONS } from "@/lib/constants";

// ─── Shared primitives ────────────────────────────────────────────────────────

const PANEL: React.CSSProperties = {
  backgroundColor: "#1E222D",
  border: "1px solid #2A2E39",
  borderRadius: 4,
  padding: "10px 14px",
};

const LABEL: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: "0.1em",
  textTransform: "uppercase" as const,
  color: "#787B86",
  whiteSpace: "nowrap" as const,
};

const INPUT: React.CSSProperties = {
  height: 26,
  padding: "0 8px",
  backgroundColor: "#131722",
  border: "1px solid #2A2E39",
  borderRadius: 3,
  color: "#D1D4DC",
  fontSize: 12,
  fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
  outline: "none",
  width: "100%",
};

const SELECT_STYLE: React.CSSProperties = {
  ...INPUT,
  cursor: "pointer",
  appearance: "none" as const,
  WebkitAppearance: "none" as const,
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%23787B86' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")`,
  backgroundRepeat: "no-repeat",
  backgroundPosition: "right 8px center",
  paddingRight: 24,
};

// ─── Tooltip ──────────────────────────────────────────────────────────────────

function Tip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    function handle(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [open]);

  return (
    <span ref={ref} style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        style={{ background: "none", border: "none", padding: 0, cursor: "pointer", display: "flex", alignItems: "center" }}
        aria-label="More information"
      >
        <Info size={11} color="#4B5563" strokeWidth={1.5} />
      </button>
      {open && (
        <span
          style={{
            position: "absolute",
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "#1E222D",
            border: "1px solid #2A2E39",
            borderRadius: 4,
            padding: "6px 10px",
            width: 220,
            fontSize: 11,
            color: "#9CA3AF",
            lineHeight: 1.5,
            zIndex: 50,
            whiteSpace: "normal",
            pointerEvents: "none",
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}

// ─── FieldRow: label + (i) stacked above an input ────────────────────────────

function Field({
  label,
  tip,
  children,
  style,
}: {
  label: string;
  tip: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5, ...style }}>
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span style={LABEL}>{label}</span>
        <Tip text={tip} />
      </div>
      {children}
    </div>
  );
}

// ─── NumericInput ─────────────────────────────────────────────────────────────

function NumericInput({
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
  const [raw, setRaw] = useState(String(value));

  // keep in sync when store changes externally
  useEffect(() => setRaw(String(value)), [value]);

  function commit(str: string) {
    const n = parseFloat(str);
    if (!isNaN(n)) {
      const clamped = Math.min(max, Math.max(min, n));
      onChange(clamped);
      setRaw(String(clamped));
    } else {
      setRaw(String(value));
    }
  }

  return (
    <input
      type="number"
      value={raw}
      min={min}
      max={max}
      step={step}
      style={INPUT}
      onChange={(e) => setRaw(e.target.value)}
      onBlur={(e) => commit(e.target.value)}
      onKeyDown={(e) => { if (e.key === "Enter") commit((e.target as HTMLInputElement).value); }}
    />
  );
}

// ─── SelectInput ─────────────────────────────────────────────────────────────

function SelectInput({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      style={SELECT_STYLE}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} style={{ backgroundColor: "#1E222D", color: "#D1D4DC" }}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

// ─── SectionHeader ────────────────────────────────────────────────────────────

function SectionHeader({ label }: { label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
      <span
        style={{
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "#4B5563",
        }}
      >
        {label}
      </span>
      <div style={{ flex: 1, height: 1, backgroundColor: "#2A2E39" }} />
    </div>
  );
}

// ─── Main panel ──────────────────────────────────────────────────────────────

export function ExecutionPanel({ defaultOpen: _defaultOpen = false }: { defaultOpen?: boolean }) {
  const setField = useBacktestStore((s) => s.setField);
  const initialEquity = useBacktestStore((s) => s.initialEquity as number);
  const maxLeverage = useBacktestStore((s) => s.maxLeverage as number);
  const sizingMethod = useBacktestStore((s) => s.sizingMethod as string);
  const trailingMethod = useBacktestStore((s) => s.trailingMethod as string);
  const maxDrawdownPct = useBacktestStore((s) => s.maxDrawdownPct as number);
  const maxConsecutiveLosses = useBacktestStore((s) => s.maxConsecutiveLosses as number);
  const dailyLossLimitPct = useBacktestStore((s) => s.dailyLossLimitPct as number);

  return (
    <div
      style={{
        backgroundColor: "#131722",
        border: "1px solid #2A2E39",
        borderRadius: 4,
        padding: "14px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      {/* ── Section 1: GENERAL ── */}
      <div style={PANEL}>
        <SectionHeader label="General" />
        <div style={{ display: "flex", gap: 16 }}>
          <Field
            label="Initial Equity"
            tip="Starting account balance in base currency used for all backtest simulations."
            style={{ flex: 1 }}
          >
            <NumericInput
              value={initialEquity}
              min={1000}
              max={100000}
              step={1000}
              onChange={(v) => setField("initialEquity", v)}
            />
          </Field>

          <Field
            label="Max Leverage"
            tip="Maximum leverage multiplier allowed per position. Higher leverage amplifies both gains and drawdowns."
            style={{ flex: 1 }}
          >
            <NumericInput
              value={maxLeverage}
              min={1}
              max={50}
              step={1}
              onChange={(v) => setField("maxLeverage", v)}
            />
          </Field>
        </div>
      </div>

      {/* ── Section 2: TRADE MANAGEMENT ── */}
      <div style={PANEL}>
        <SectionHeader label="Trade Management" />
        <div style={{ display: "flex", gap: 16 }}>
          <Field
            label="Position Sizing Method"
            tip="Fixed Lot = constant size. Kelly = optimal growth fraction. Fixed Fractional = % of equity per trade. ATR = volatility-adjusted."
            style={{ flex: 1 }}
          >
            <SelectInput
              value={sizingMethod}
              options={[...SELECT_OPTIONS.sizingMethod]}
              onChange={(v) => setField("sizingMethod", v as typeof sizingMethod)}
            />
          </Field>

          <Field
            label="Trailing Stops Method"
            tip="None = no trailing. Standard = fixed pip step. ATR = volatility-adjusted step. Chandelier = highest high/low minus ATR multiple."
            style={{ flex: 1 }}
          >
            <SelectInput
              value={trailingMethod}
              options={[...SELECT_OPTIONS.trailingMethod]}
              onChange={(v) => setField("trailingMethod", v as typeof trailingMethod)}
            />
          </Field>
        </div>
      </div>

      {/* ── Section 3: RISK MANAGEMENT ── */}
      <div style={PANEL}>
        <SectionHeader label="Risk Management" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
          <Field
            label="Max Drawdown"
            tip="Peak-to-trough equity drop that triggers a full trading halt. 0.15 = 15%, a common institutional circuit-breaker."
          >
            <NumericInput
              value={maxDrawdownPct}
              min={0.05}
              max={0.5}
              step={0.05}
              onChange={(v) => setField("maxDrawdownPct", v)}
            />
          </Field>

          <Field
            label="Max Consec. Losses"
            tip="Halt trading after this many consecutive losing trades. Prevents emotional drawdown spirals in live deployment."
          >
            <NumericInput
              value={maxConsecutiveLosses}
              min={2}
              max={20}
              step={1}
              onChange={(v) => setField("maxConsecutiveLosses", v)}
            />
          </Field>

          <Field
            label="Daily Loss Limit"
            tip="Maximum daily equity loss as a fraction before pausing until the next session. 0.03 = 3% of equity."
          >
            <NumericInput
              value={dailyLossLimitPct}
              min={0.01}
              max={0.1}
              step={0.01}
              onChange={(v) => setField("dailyLossLimitPct", v)}
            />
          </Field>
        </div>
      </div>
    </div>
  );
}
