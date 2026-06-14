import type { CycleState, JobState } from "@/stores/useJobStore";

interface Props {
  job: JobState;
}

export function ModelHealthTable({ job }: Props) {
  const models = job.models ?? [];
  const cycles = job.cycles ?? [];

  const rows = models.map((model) => {
    const modelCycles = cycles.filter((c) => c.model === model);
    const lastCycle = modelCycles[modelCycles.length - 1];
    const phase = lastCycle?.phase ?? "pending";
    const bestTrial = lastCycle?.bestTrial;
    const testMonths = lastCycle?.testMonths ?? [];
    const avgSharpe =
      testMonths.length > 0
        ? testMonths.reduce((s, m) => s + (m.sharpe ?? 0), 0) / testMonths.length
        : null;
    const totalTrades = testMonths.reduce((s, m) => s + (m.trades ?? 0), 0);
    const avgWinRate =
      testMonths.length > 0
        ? testMonths.reduce((s, m) => s + (m.win_rate ?? 0), 0) / testMonths.length
        : null;
    const overfitRisk =
      testMonths.length > 0
        ? testMonths.some((m) => m.sharpe_gap_pct != null && m.sharpe_gap_pct > 0.5)
        : false;

    return { model, phase, bestTrial, avgSharpe, totalTrades, avgWinRate, overfitRisk };
  });

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-[10px] text-(--color-text-muted)">
        No models active
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-(--color-glass-border)">
            {["Model", "Phase", "Sharpe", "Trades", "Win %", "Risk"].map((col) => (
              <th
                key={col}
                className="px-1.5 py-1 text-left text-[9px] font-medium tracking-[0.06em] whitespace-nowrap text-(--color-text-muted) uppercase"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.model}
              className="border-b border-[rgba(42,46,57,0.4)] text-[10px] transition hover:bg-(--color-glass-hover)"
            >
              <td className="max-w-[80px] truncate px-1.5 py-1 font-mono font-semibold whitespace-nowrap text-(--color-text-primary)">
                {r.model}
              </td>
              <td className="px-1.5 py-1 whitespace-nowrap">
                <span
                  className="inline-flex items-center gap-1 rounded-full bg-(--color-glass-hover) px-1.5 py-0.5 text-[8px] font-medium uppercase"
                  style={{
                    color:
                      r.phase === "complete"
                        ? "var(--color-accent-success)"
                        : r.phase === "hpo"
                          ? "var(--color-accent)"
                          : r.phase === "simulation"
                            ? "var(--color-accent-success)"
                            : "var(--color-text-muted)",
                  }}
                >
                  {r.phase}
                </span>
              </td>
              <td className="px-1.5 py-1 font-mono whitespace-nowrap tabular-nums">
                <span
                  style={{
                    color:
                      r.avgSharpe != null
                        ? r.avgSharpe >= 0
                          ? "var(--color-accent-success)"
                          : "var(--color-accent-danger)"
                        : "var(--color-text-muted)",
                  }}
                >
                  {r.avgSharpe?.toFixed(2) ?? "\u2014"}
                </span>
              </td>
              <td className="px-1.5 py-1 font-mono whitespace-nowrap text-(--color-text-secondary) tabular-nums">
                {r.totalTrades || "\u2014"}
              </td>
              <td className="px-1.5 py-1 font-mono whitespace-nowrap tabular-nums">
                <span
                  style={{
                    color:
                      r.avgWinRate != null
                        ? r.avgWinRate >= 0.5
                          ? "var(--color-accent-success)"
                          : "var(--color-accent-danger)"
                        : "var(--color-text-muted)",
                  }}
                >
                  {r.avgWinRate != null ? `${(r.avgWinRate * 100).toFixed(0)}%` : "\u2014"}
                </span>
              </td>
              <td className="px-1.5 py-1">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  title={r.overfitRisk ? "Overfitting risk detected" : "No overfitting risk"}
                  style={{
                    backgroundColor: r.overfitRisk
                      ? "var(--color-accent-danger)"
                      : "var(--color-accent-success)",
                  }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
