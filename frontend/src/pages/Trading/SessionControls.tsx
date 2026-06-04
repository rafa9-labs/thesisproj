import { Play, Square, Loader2, ShieldOff } from "lucide-react";

export type TradingMode = "paper" | "live";

const SIZING_OPTIONS = [
  { value: "fixed", label: "Fixed (1 lot)" },
  { value: "fixed_fractional", label: "Fractional (% of equity)" },
  { value: "kelly", label: "Kelly Criterion" },
  { value: "atr", label: "ATR Volatility" },
  { value: "vol_target", label: "Vol Target" },
];

interface SessionControlsProps {
  mode: TradingMode;
  isRunning: boolean;
  isDeploying: boolean;
  isKilled: boolean;
  hasModel: boolean;
  positionSizing: string;
  initialEquity: number;
  onModeChange: (mode: TradingMode) => void;
  onSizingChange: (s: string) => void;
  onEquityChange: (e: number) => void;
  onDeploy: () => void;
  onStop: () => void;
  onEmergency: () => void;
}

function ModeToggle({ mode, disabled, onChange }: { mode: TradingMode; disabled: boolean; onChange: (m: TradingMode) => void }) {
  return (
    <div className="flex items-center rounded-sm border p-0.5" style={{ borderColor: "var(--color-glass-border)" }}>
      <button
        onClick={() => { if (!disabled) onChange("paper"); }}
        disabled={disabled}
        className="rounded-md px-3 py-1 text-[10px] font-semibold uppercase transition-all disabled:opacity-50"
        style={{
          backgroundColor: mode === "paper" ? "var(--color-brand)" : "transparent",
          color: mode === "paper" ? "var(--color-text-inverse)" : "var(--color-text-muted)",
        }}
      >
        Paper
      </button>
      <button
        onClick={() => { if (!disabled) onChange("live"); }}
        disabled={disabled}
        className="rounded-md px-3 py-1 text-[10px] font-semibold uppercase transition-all disabled:opacity-50"
        style={{
          backgroundColor: mode === "live" ? "var(--color-accent-danger)" : "transparent",
          color: mode === "live" ? "var(--color-text-inverse)" : "var(--color-text-muted)",
        }}
      >
        Live
      </button>
    </div>
  );
}

export function SessionControls({
  mode,
  isRunning,
  isDeploying,
  isKilled,
  hasModel,
  positionSizing,
  initialEquity,
  onModeChange,
  onSizingChange,
  onEquityChange,
  onDeploy,
  onStop,
  onEmergency,
}: SessionControlsProps) {
  const disabled = isRunning || isDeploying;

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <ModeToggle mode={mode} disabled={isRunning} onChange={onModeChange} />

      <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Sizing</span>
      <select
        value={positionSizing}
        onChange={(e) => onSizingChange(e.target.value)}
        disabled={disabled}
        className="rounded-md border px-2.5 py-1 text-xs transition focus:outline-none disabled:opacity-50"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
      >
        {SIZING_OPTIONS.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
      </select>

      <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Equity</span>
      <input
        type="number"
        value={initialEquity}
        onChange={(e) => onEquityChange(Number(e.target.value) || 10000)}
        disabled={disabled}
        min={100}
        max={1000000}
        className="rounded-md border px-2.5 py-1 text-xs w-20 transition focus:outline-none disabled:opacity-50"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
      />

      <div className="flex-1" />

      {mode === "live" && isRunning && (
        <button
          onClick={onEmergency}
          className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-bold uppercase animate-pulse"
          style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(239,68,68,0.15)", color: "var(--color-accent-danger)" }}
        >
          <ShieldOff size={14} />
          Emergency Kill
        </button>
      )}

      <button
        onClick={isRunning ? onStop : onDeploy}
        disabled={isDeploying || !hasModel || isKilled}
        className="flex items-center gap-1.5 rounded-md border px-4 py-1.5 text-[11px] font-semibold uppercase transition-all duration-200 disabled:opacity-50"
        style={{
          borderColor: isRunning ? "var(--color-accent-danger)" : "var(--color-accent-success)",
          backgroundColor: isRunning ? "rgba(239,68,68,0.1)" : "rgba(34,197,94,0.1)",
          color: isRunning ? "var(--color-accent-danger)" : "var(--color-accent-success)",
        }}
      >
        {isDeploying ? (
          <><Loader2 size={12} className="animate-spin" />Deploying</>
        ) : isRunning ? (
          <><Square size={12} />Stop</>
        ) : (
          <><Play size={12} />Deploy</>
        )}
      </button>
    </div>
  );
}
