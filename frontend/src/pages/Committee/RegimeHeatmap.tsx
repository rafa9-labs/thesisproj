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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Regime Distribution Chart */}
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 20,
        }}
      >
        <h2
          style={{
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
            margin: "0 0 16px",
          }}
        >
          Market Regime Distribution
          <span
            style={{
              fontSize: 10,
              color: "var(--color-text-muted)",
              marginLeft: 8,
            }}
          >
            Last {labels?.count ?? 0} Bars
          </span>
        </h2>

        {isLoading ? (
          <div style={{ color: "var(--color-text-muted)", fontSize: 12 }}>
            Loading...
          </div>
        ) : labels && labels.labels.length > 0 ? (
          <RegimeBarChart labels={labels.labels} />
        ) : (
          <div style={{ color: "var(--color-text-muted)", fontSize: 12 }}>
            No regime data available. Run ExpertProfiler first.
          </div>
        )}
      </div>

      {/* Regime × Model Matrix */}
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 20,
        }}
      >
        <h2
          style={{
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
            margin: "0 0 16px",
          }}
        >
          Regime x Model Performance Matrix
          <span
            style={{
              fontSize: 10,
              color: "var(--color-text-muted)",
              marginLeft: 8,
            }}
          >
            Sharpe per Regime
          </span>
        </h2>

        {matrixLoading ? (
          <div style={{ color: "var(--color-text-muted)", fontSize: 12 }}>
            Loading...
          </div>
        ) : matrix && matrix.entries.length > 0 ? (
          <MatrixGrid entries={matrix.entries} regimes={matrix.regimes} models={matrix.models} />
        ) : (
          <div style={{ color: "var(--color-text-muted)", fontSize: 12 }}>
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
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {entries.map(([name, count]) => {
        const pct = (count / maxCount) * 100;
        return (
          <div
            key={name}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            <div
              style={{
                width: 120,
                fontSize: 11,
                fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: regimeColor(name),
                textAlign: "right",
                flexShrink: 0,
              }}
            >
              {regimeLabel(name)}
            </div>
            <div
              style={{
                flex: 1,
                height: barHeight,
                borderRadius: 2,
                background: "var(--color-elevated)",
                overflow: "hidden",
                position: "relative",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${pct}%`,
                  background: regimeColor(name),
                  opacity: 0.7,
                  borderRadius: 2,
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <div
              style={{
                width: 48,
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--color-text-secondary)",
                textAlign: "right",
              }}
            >
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
    <div style={{ overflow: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 11,
        }}
      >
        <thead>
          <tr>
            <th
              style={{
                padding: "6px 10px",
                textAlign: "left",
                color: "var(--color-text-muted)",
                fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                fontSize: 10,
                borderBottom: "1px solid var(--color-glass-border)",
              }}
            >
              Model
            </th>
            {regimes.map((r) => (
              <th
                key={r}
                style={{
                  padding: "6px 10px",
                  textAlign: "right",
                  color: regimeColor(r),
                  fontWeight: 500,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  fontSize: 10,
                  borderBottom: "1px solid var(--color-glass-border)",
                }}
              >
                {regimeLabel(r)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr
              key={model}
              style={{
                borderBottom: "1px solid var(--color-glass-border)",
              }}
            >
              <td
                style={{
                  padding: "8px 10px",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                }}
              >
                {model}
              </td>
              {regimes.map((regime) => {
                const e = lookup[regime]?.[model];
                const sharpe = e?.sharpe ?? NaN;
                const val = isNaN(sharpe) ? "--" : sharpe.toFixed(2);
                const color =
                  isNaN(sharpe)
                    ? "var(--color-text-dim)"
                    : sharpe > 0.5
                      ? "#089981"
                      : sharpe > 0
                        ? "var(--color-text-secondary)"
                        : "#F23645";
                return (
                  <td
                    key={regime}
                    style={{
                      padding: "8px 10px",
                      textAlign: "right",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      color,
                    }}
                  >
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
