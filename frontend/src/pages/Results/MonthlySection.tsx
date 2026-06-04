import { MonthlyReturnsChart } from "@/components/charts/MonthlyReturnsChart";
import type { MonthlyResult } from "@/api/schemas";
import { formatPercent, formatMetric, formatInt, colorForReturn } from "@/lib/formatters";

interface MonthlySectionProps {
  monthlyResults: MonthlyResult[] | null;
}

export function MonthlySection({ monthlyResults }: MonthlySectionProps) {
  const data = monthlyResults ?? [];

  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-sm border p-8"
        style={{
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        <span className="text-sm" style={{ fontFamily: "var(--font-mono)" }}>
          No monthly breakdown available
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <MonthlyReturnsChart monthlyResults={monthlyResults} height={260} />
      <div
        className="rounded-sm border overflow-hidden"
        style={{ borderColor: "var(--color-border)" }}
      >
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr
              style={{
                backgroundColor: "var(--color-elevated)",
                color: "var(--color-text-secondary)",
              }}
            >
              <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Month</th>
              <th className="px-3 py-2 text-right font-semibold uppercase tracking-wide">Return</th>
              <th className="px-3 py-2 text-right font-semibold uppercase tracking-wide">Win Rate</th>
              <th className="px-3 py-2 text-right font-semibold uppercase tracking-wide">Trades</th>
              <th className="px-3 py-2 text-right font-semibold uppercase tracking-wide">Sharpe</th>
            </tr>
          </thead>
          <tbody>
            {data.map((m, i) => (
              <tr
                key={m.month}
                style={{
                  backgroundColor: i % 2 === 0 ? "var(--color-surface)" : "var(--color-app)",
                  borderBottom: "1px solid var(--color-border)",
                }}
              >
                <td
                  className="px-3 py-2"
                  style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                >
                  {m.month}
                </td>
                <td
                  className="px-3 py-2 text-right"
                  style={{ color: colorForReturn(m.return_pct), fontFamily: "var(--font-mono)" }}
                >
                  {formatPercent(m.return_pct)}
                </td>
                <td
                  className="px-3 py-2 text-right"
                  style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                >
                  {formatPercent(m.win_rate, 0)}
                </td>
                <td
                  className="px-3 py-2 text-right"
                  style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
                >
                  {formatInt(m.trades)}
                </td>
                <td
                  className="px-3 py-2 text-right"
                  style={{
                    color: (m.sharpe ?? 0) >= 1 ? "var(--color-accent-success)" : "var(--color-text-primary)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {formatMetric(m.sharpe)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
