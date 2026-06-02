import { useState } from "react";
import { useCommitteeBacktestResults, useSubmitCommitteeBacktest, useCommitteeConfig } from "@/api/queries";
import type { CommitteeBacktestRequest } from "@/api/schemas";

export function CommitteeResultsTab() {
  const [jobId, setJobId] = useState<string | null>(null);
  const submitMutation = useSubmitCommitteeBacktest();
  const { data: config } = useCommitteeConfig();
  const { data: results } = useCommitteeBacktestResults(jobId);

  function handleSubmit() {
    if (!config) return;
    submitMutation.mutate(
      { config, train_months: 4, test_months: 1, confidence_threshold: 0.5 },
      { onSuccess: (data) => setJobId(data.job_id) },
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Submit */}
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 20,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <h2
              style={{
                fontSize: 13,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
                margin: "0 0 4px",
              }}
            >
              Committee Backtest
            </h2>
            <span
              style={{ fontSize: 10, color: "var(--color-text-muted)" }}
            >
              4/1 months WFO, confidence threshold 0.5
            </span>
          </div>
          <button
            onClick={handleSubmit}
            disabled={submitMutation.isPending || !config}
            style={{
              background: "var(--color-brand)",
              color: "var(--color-text-inverse)",
              border: "none",
              borderRadius: 4,
              padding: "8px 20px",
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              cursor: "pointer",
              opacity: submitMutation.isPending || !config ? 0.5 : 1,
            }}
          >
            {submitMutation.isPending ? "Running..." : "Run Backtest"}
          </button>
        </div>
        {submitMutation.isError && (
          <div style={{ marginTop: 8, fontSize: 11, color: "var(--color-accent-danger)" }}>
            Submit failed. Check API logs.
          </div>
        )}
      </div>

      {/* Results */}
      {results && results.status === "completed" && (
        <>
          {/* Summary Cards */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
              gap: 12,
            }}
          >
            <MetricLabel value={results.avg_sharpe.toFixed(3)} label="Avg Sharpe" />
            <MetricLabel value={results.avg_trades.toFixed(0)} label="Avg Trades" />
            <MetricLabel value={String(results.total_folds)} label="Folds" />
            <MetricLabel value={results.execution_time_s.toFixed(1) + "s"} label="Time" />
          </div>

          {/* Per-fold table */}
          <div
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 4,
              padding: 20,
              overflow: "auto",
            }}
          >
            <h3
              style={{
                fontSize: 12,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
                margin: "0 0 12px",
              }}
            >
              Per-Fold Breakdown
            </h3>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 11,
              }}
            >
              <thead>
                <tr>
                  {["#", "Sharpe", "Trades", "Win Rate", "Active Rate", "Return", "Max DD"].map((h) => (
                    <th
                      key={h}
                      style={{
                        padding: "6px 8px",
                        textAlign: h === "#" ? "left" : "right",
                        color: "var(--color-text-muted)",
                        fontWeight: 500,
                        letterSpacing: "0.06em",
                        textTransform: "uppercase",
                        fontSize: 10,
                        borderBottom: "1px solid var(--color-glass-border)",
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.folds.map((fold) => (
                  <tr key={fold.fold_idx}>
                    <td style={{ padding: "6px 8px", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                      {fold.fold_idx + 1}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "var(--font-mono)", color: fold.sharpe >= 0 ? "#089981" : "#F23645" }}>
                      {fold.sharpe.toFixed(3)}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                      {fold.trades}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                      {(fold.win_rate * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                      {(fold.active_rate * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "var(--font-mono)", color: fold.return_val >= 0 ? "#089981" : "#F23645" }}>
                      {fold.return_val.toFixed(4)}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                      {(fold.drawdown * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Models */}
          <div style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
            Models:{" "}
            {results.models.map((m) => (
              <span
                key={m}
                style={{
                  fontFamily: "var(--font-mono)",
                  color: "var(--color-text-secondary)",
                  marginRight: 8,
                }}
              >
                {m}
              </span>
            ))}
          </div>

          {results.warnings.length > 0 && (
            <div style={{ fontSize: 11, color: "var(--color-accent-warning)" }}>
              Warnings: {results.warnings.join(", ")}
            </div>
          )}
        </>
      )}

      {results && results.status !== "completed" && (
        <div style={{ fontSize: 12, color: "var(--color-text-muted)" }}>
          Status: {results.status}
          {results.warnings.length > 0 && <> — {results.warnings[0]}</>}
        </div>
      )}
    </div>
  );
}

function MetricLabel({ value, label }: { value: string; label: string }) {
  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-glass-border)",
        borderRadius: 4,
        padding: "14px 16px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 600,
          fontFamily: "var(--font-mono)",
          color: "var(--color-brand)",
          marginBottom: 4,
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: 10,
          fontWeight: 500,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--color-text-muted)",
        }}
      >
        {label}
      </div>
    </div>
  );
}
