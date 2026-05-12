export function TradeLog() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-lg border p-8"
      style={{ borderColor: "var(--color-border)", backgroundColor: "var(--color-surface)" }}
    >
      <span className="text-sm" style={{ color: "var(--color-text-muted)" }}>
        Trade data available after simulation completes.
      </span>
      <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
        Individual trade details will appear here once the backtest finishes.
      </span>
    </div>
  );
}