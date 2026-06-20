import { useMemo } from "react";

interface ModelContribution {
  model: string;
  delta_sharpe: number;
  active_pct: number;
}

interface Props {
  contributions: ModelContribution[] | undefined;
}

export function ModelContributionChart({ contributions }: Props) {
  const sorted = useMemo(() => {
    if (!contributions) return [];
    return [...contributions].sort((a, b) => b.delta_sharpe - a.delta_sharpe);
  }, [contributions]);

  if (sorted.length === 0) {
    return <p className="text-[11px] text-(--color-text-dim)">No model contribution data</p>;
  }

  const maxAbsDelta = Math.max(...sorted.map((c) => Math.abs(c.delta_sharpe)), 0.01);

  return (
    <div>
      <div className="mb-2">
        <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
          Leave-One-Out ΔSharpe
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {sorted.map((c) => {
          const barPct = Math.min(Math.abs(c.delta_sharpe) / maxAbsDelta * 100, 100);
          const isPositive = c.delta_sharpe >= 0;
          return (
            <div key={c.model} className="flex items-center gap-2">
              <span className="w-[100px] shrink-0 truncate text-right font-mono text-[10px] text-(--color-text-secondary)">
                {c.model.length > 12 ? c.model.slice(0, 10) + ".." : c.model}
              </span>
              <div className="h-[8px] flex-1 rounded-full bg-(--color-elevated)">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${barPct}%`,
                    backgroundColor: isPositive
                      ? "var(--color-accent-success)"
                      : "var(--color-accent-danger)",
                    opacity: 0.75,
                  }}
                />
              </div>
              <span
                className="w-[50px] text-right font-mono text-[10px] tabular-nums"
                style={{
                  color: isPositive
                    ? "var(--color-accent-success)"
                    : "var(--color-accent-danger)",
                }}
              >
                {isPositive ? "+" : ""}
                {c.delta_sharpe.toFixed(3)}
              </span>
              <span className="w-[32px] text-right font-mono text-[9px] text-(--color-text-muted)">
                {Math.round(c.active_pct * 100)}%
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-1 text-[9px] text-(--color-text-dim)">
        Positive Δ = model adds value. % = active bar fraction.
      </div>
    </div>
  );
}
