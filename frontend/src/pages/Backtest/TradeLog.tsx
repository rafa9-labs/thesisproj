export function TradeLog() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-sm border border-(--color-border) bg-(--color-surface) p-8">
      <span className="text-sm text-(--color-text-muted)">
        Trade data available after simulation completes.
      </span>
      <span className="text-xs text-(--color-text-muted)">
        Individual trade details will appear here once the backtest finishes.
      </span>
    </div>
  );
}
