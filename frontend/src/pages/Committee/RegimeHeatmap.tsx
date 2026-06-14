import { useRegimeMatrix, useRegimeLabels } from "@/api/queries";
import type { RegimeMatrixEntry } from "@/api/schemas";

const REGIME_COLORS: Record<string, string> = {
  trend_up: "#089981",
  trend_down: "#F23645",
  mean_reverting: "#F59E0B",
  breakout: "#A78BFA",
  high_volatile: "#EC4899",
  quiet_squeeze: "#22D3EE",
  sideways: "#787B86",
};

function regimeColor(name: string): string {
  return REGIME_COLORS[name] ?? "#787B86";
}

function regimeLabel(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function RegimeHeatmap() {
  const { data: matrix, isLoading: matrixLoading } = useRegimeMatrix();
  const { data: labels, isLoading: labelsLoading } = useRegimeLabels("EURUSD", "H1", 500);

  const isLoading = matrixLoading || labelsLoading;

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-[60px]">
        <div className="h-6 w-6 animate-spin rounded-full border-[3px] border-(--color-glass-border) border-t-(--color-brand)" />
        <span className="mt-4 text-[11px] text-(--color-text-muted)">Loading regime data...</span>
      </div>
    );
  }

  if (!matrix?.entries?.length) {
    return (
      <div className="rounded border border-(--color-glass-border) bg-(--color-surface) p-[60px] text-center">
        <span className="mb-2 block text-xs text-(--color-text-dim)">
          No regime matrix available
        </span>
        <span className="text-[10px] text-(--color-text-muted)">
          Run the Full Cycle pipeline first to generate regime performance data.
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Regime Distribution Chart */}
      <div className="rounded border border-(--color-glass-border) bg-(--color-surface) p-5">
        <h2 className="mb-4 text-[13px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Market Regime Distribution
          <span className="ml-2 text-[10px] text-(--color-text-muted)">
            Last {labels?.count ?? 0} Bars
          </span>
        </h2>

        {isLoading ? (
          <div className="text-xs text-(--color-text-muted)">Loading...</div>
        ) : labels && labels.labels.length > 0 ? (
          <RegimeBarChart labels={labels.labels} />
        ) : (
          <div className="text-xs text-(--color-text-muted)">
            No regime data available. Run ExpertProfiler first.
          </div>
        )}
      </div>

      {/* Regime × Model Matrix */}
      <div className="rounded border border-(--color-glass-border) bg-(--color-surface) p-5">
        <h2 className="mb-4 text-[13px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Regime x Model Performance Matrix
          <span className="ml-2 text-[10px] text-(--color-text-muted)">Sharpe per Regime</span>
        </h2>

        {matrixLoading ? (
          <div className="text-xs text-(--color-text-muted)">Loading...</div>
        ) : matrix && matrix.entries.length > 0 ? (
          <MatrixGrid entries={matrix.entries} regimes={matrix.regimes} models={matrix.models} />
        ) : (
          <div className="text-xs text-(--color-text-muted)">
            No matrix data. Run ExpertProfiler (PROFILE=1) to generate.
          </div>
        )}
      </div>
    </div>
  );
}

function RegimeBarChart({ labels }: { labels: { regime_name: string }[] }) {
  const counts: Record<string, number> = {};
  for (const l of labels) {
    counts[l.regime_name] = (counts[l.regime_name] ?? 0) + 1;
  }
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const maxCount = Math.max(...entries.map(([, c]) => c), 1);
  const barHeight = 28;

  return (
    <div className="flex flex-col gap-1">
      {entries.map(([name, count]) => {
        const pct = (count / maxCount) * 100;
        return (
          <div key={name} className="flex items-center gap-2">
            <div
              className="w-[120px] shrink-0 text-right text-[11px] font-medium tracking-[0.06em] uppercase"
              style={{ color: regimeColor(name) }}
            >
              {regimeLabel(name)}
            </div>
            <div
              className="relative flex-1 overflow-hidden rounded-[2px] bg-(--color-elevated)"
              style={{ height: barHeight }}
            >
              <div
                className="h-full rounded-[2px] opacity-70"
                style={{
                  width: `${pct}%`,
                  background: regimeColor(name),
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <div className="w-12 text-right font-mono text-[11px] text-(--color-text-secondary)">
              {count}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MatrixGrid({
  entries,
  regimes,
  models,
}: {
  entries: RegimeMatrixEntry[];
  regimes: string[];
  models: string[];
}) {
  const lookup: Record<string, Record<string, RegimeMatrixEntry>> = {};
  for (const e of entries) {
    if (!lookup[e.regime]) lookup[e.regime] = {};
    lookup[e.regime][e.model] = e;
  }

  return (
    <div className="overflow-auto">
      <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th className="border-b border-(--color-glass-border) px-[10px] py-[6px] text-left text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
              Model
            </th>
            {regimes.map((r) => (
              <th
                key={r}
                className="border-b border-(--color-glass-border) px-[10px] py-[6px] text-right text-[10px] font-medium tracking-[0.06em] uppercase"
                style={{ color: regimeColor(r) }}
              >
                {regimeLabel(r)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model} className="border-b border-(--color-glass-border)">
              <td className="px-[10px] py-2 font-mono text-[11px] text-(--color-text-primary)">
                {model}
              </td>
              {regimes.map((regime) => {
                const e = lookup[regime]?.[model];
                const sharpe = e?.sharpe ?? NaN;
                const val = isNaN(sharpe) ? "--" : sharpe.toFixed(2);
                const color = isNaN(sharpe)
                  ? "var(--color-text-dim)"
                  : sharpe > 0.5
                    ? "#089981"
                    : sharpe > 0
                      ? "var(--color-text-secondary)"
                      : "#F23645";
                return (
                  <td key={regime} className="px-[10px] py-2 text-right font-mono text-[11px]">
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
