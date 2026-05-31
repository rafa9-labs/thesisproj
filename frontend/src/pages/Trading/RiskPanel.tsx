interface RiskConfig {
  max_drawdown_pct: number;
  max_daily_loss_pct: number;
  max_consecutive_losses: number;
  min_confidence: number;
  max_position_pct: number;
  max_daily_trades: number;
}

interface RiskPanelProps {
  show: boolean;
  config: RiskConfig;
  onChange: (update: Partial<RiskConfig>) => void;
  onToggle: () => void;
}

const FIELDS: { key: keyof RiskConfig; label: string; min: number; max: number; step: number }[] = [
  { key: "max_drawdown_pct", label: "Max DD %", min: 5, max: 30, step: 1 },
  { key: "max_daily_loss_pct", label: "Daily Loss %", min: 1, max: 15, step: 1 },
  { key: "max_consecutive_losses", label: "Consec Losses", min: 2, max: 10, step: 1 },
  { key: "min_confidence", label: "Min Confidence", min: 50, max: 90, step: 5 },
  { key: "max_position_pct", label: "Max Pos %", min: 10, max: 50, step: 5 },
  { key: "max_daily_trades", label: "Daily Trades", min: 5, max: 50, step: 5 },
];

export function RiskPanel({ show, config, onChange, onToggle }: RiskPanelProps) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onToggle}
        className="text-[10px] font-medium uppercase tracking-[0.08em] rounded-md border px-2 py-1 transition"
        style={{
          borderColor: "var(--color-glass-border)",
          color: show ? "var(--color-brand)" : "var(--color-text-muted)",
          backgroundColor: show ? "var(--color-brand-glow)" : "transparent",
        }}
      >
        {show ? "Hide" : "Show"} Risk Config
      </button>
      {show && (
        <div className="flex flex-wrap gap-3">
          {FIELDS.map((rc) => (
            <div key={rc.key} className="flex items-center gap-1">
              <span className="text-[9px] uppercase" style={{ color: "var(--color-text-muted)" }}>
                {rc.label}
              </span>
              <input
                type="number"
                value={config[rc.key]}
                onChange={(e) => onChange({ [rc.key]: Number(e.target.value) })}
                min={rc.min}
                max={rc.max}
                step={rc.step}
                className="rounded-md border px-1.5 py-0.5 text-[10px] w-14 transition focus:outline-none"
                style={{
                  borderColor: "var(--color-glass-border)",
                  backgroundColor: "var(--color-glass)",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
