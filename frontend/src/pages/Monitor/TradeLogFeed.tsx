import type { OosPeriodResult } from "@/api/schemas";

interface Props {
  periods: OosPeriodResult[];
  models: string[];
}

export function TradeLogFeed({ periods, models }: Props) {
  const entries = periods
    .filter((p) => (p.trades ?? 0) > 0)
    .map((p) => ({
      period: p.period,
      model: p.model ?? models[0] ?? "",
      trades: p.trades ?? 0,
      returnPct: p.return_pct ?? 0,
      sharpe: p.sharpe ?? null,
      winRate: p.win_rate ?? null,
    }))
    .reverse()
    .slice(0, 50);

  if (entries.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="text-[10px] text-(--color-text-muted)">No trade data yet</span>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-0 overflow-y-auto font-mono [scrollbar-width:thin]">
      {entries.map((e, i) => {
        const isPositive = e.returnPct >= 0;
        const dirColor = isPositive ? "var(--color-accent-success)" : "var(--color-accent-danger)";
        const pnlColor = isPositive ? "var(--color-accent-success)" : "var(--color-accent-danger)";

        return (
          <div
            key={`${e.period}-${e.model}-${i}`}
            className="flex items-center gap-1.5 border-b border-[rgba(42,46,57,0.3)] px-3 py-1.5 text-[10px] transition-colors hover:bg-(--color-glass-hover)"
          >
            <span className="w-12 shrink-0 whitespace-nowrap text-(--color-text-dim)">
              M{e.period}
            </span>
            <span className="w-8 shrink-0 text-right text-(--color-brand) tabular-nums">
              {e.trades}
            </span>
            <span className="flex-1 text-right tabular-nums" style={{ color: pnlColor }}>
              {e.returnPct >= 0 ? "+" : ""}
              {e.returnPct.toFixed(2)}%
            </span>
            <span className="w-12 shrink-0 text-right tabular-nums" style={{ color: dirColor }}>
              {e.sharpe != null ? e.sharpe.toFixed(2) : "\u2014"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
