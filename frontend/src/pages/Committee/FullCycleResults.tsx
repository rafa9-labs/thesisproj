import { useFullCycleResults } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { formatStopReason } from "@/lib/constants";
import apiClient from "@/api/client";
import type { FactoryIterationRecord } from "@/api/schemas";

interface Props {
  jobId: string;
  onRunAgain: () => void;
}

function SectionHeader({ label }: { label: string }) {
  return (
    <span style={{
      display: "block",
      fontSize: 11,
      fontWeight: 500,
      letterSpacing: "0.06em",
      textTransform: "uppercase",
      color: "var(--color-text-muted)",
      marginBottom: 8,
    }}>
      {label}
    </span>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, padding: "14px 16px", textAlign: "center" }}>
      <div style={{ fontSize: 22, fontWeight: 600, fontFamily: "var(--font-mono)", color, marginBottom: 6 }}>
        {value}
      </div>
      <div style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
        {label}
      </div>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: "5px 8px",
  textAlign: "left",
  color: "var(--color-text-muted)",
  fontWeight: 500,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  fontSize: 10,
  borderBottom: "1px solid var(--color-glass-border)",
};

const tdStyle: React.CSSProperties = { padding: "5px 8px", color: "var(--color-text-secondary)", fontSize: 11 };
const tdStyleMono: React.CSSProperties = { ...tdStyle, fontFamily: "var(--font-mono)" };

export function FullCycleResults({ jobId, onRunAgain }: Props) {
  const { data: results } = useFullCycleResults(jobId);
  const { data: status } = { data: null };
  const store = useFullCycleStore();

  if (!results) {
    return (
      <div style={{ padding: 28, textAlign: "center", color: "var(--color-text-muted)", fontSize: 11 }}>
        Loading results...
      </div>
    );
  }

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `full_cycle_${jobId ?? "results"}.json`;
    a.click(); URL.revokeObjectURL(url);
  };

  const handleDeploy = () => {
    apiClient.post("/live/deploy-committee", {
      pair: store.deployedPair, timeframe: store.deployedTimeframe, initial_equity: 10000.0, confidence_threshold: 0.55,
      execution_mode: store.executionMode,
      full_cycle_job_id: jobId,
    }).then((r: { data: { session_id: string; pair: string; timeframe: string; models: string[] } }) => {
      store.setDeployedSession(
        r.data.session_id,
        r.data.pair || store.deployedPair,
        r.data.timeframe || store.deployedTimeframe,
      );
    }).catch((err: { response?: { data?: { detail?: string } }; message?: string }) => {
      console.error("Deploy failed:", err?.response?.data?.detail ?? err?.message);
    });
  };

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-glass-border)",
        borderRadius: 4,
        padding: 28,
      }}
    >
      <h3 style={{
        fontSize: 14, fontWeight: 600, letterSpacing: "0.08em",
        textTransform: "uppercase", color: "var(--color-text-primary)", margin: "0 0 20px",
      }}>
        Pipeline Results
        {results.status === "validation_failed" && (
          <span style={{ fontSize: 11, color: "#F2B436", marginLeft: 10, fontWeight: 500 }}>VALIDATION FAILED</span>
        )}
        <span style={{ fontSize: 11, color: "var(--color-text-muted)", marginLeft: 10, fontWeight: 400, fontFamily: "var(--font-mono)" }}>
          {Number(results.total_time_s).toFixed(0)}s
        </span>
      </h3>

      {/* Phase 1 Feature Sweep */}
      {results.locked_features_count !== undefined && results.locked_features_count > 0 && (
        <>
          <SectionHeader label="Phase 1: Feature Sweep" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
            <MetricCard label="Features Locked" value={String(results.locked_features_count)} color="#089981" />
            <MetricCard label="Features Pruned" value={String(results.pruned_features_count ?? 0)} color="#F23645" />
            <MetricCard label="Top Feature" value={results.top_importance_feature || "N/A"} color="var(--color-text-secondary)" />
          </div>
        </>
      )}

      {/* Phase 3 HPO Status */}
      {results.hpo_status && Object.keys(results.hpo_status).length > 0 && (
        <>
          <SectionHeader label="Phase 3: HPO Status" />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 20 }}>
            {Object.entries(results.hpo_status).map(([model, status]) => {
              const statusColor =
                status === "success" ? "#089981"
                : status === "timed_out" ? "#F2B436"
                : status === "crashed" ? "#F23645"
                : status === "no_folds" ? "#EF5350"
                : "var(--color-text-dim)";
              const statusBg =
                status === "success" ? "rgba(8,153,129,0.12)"
                : status === "timed_out" ? "rgba(242,180,54,0.12)"
                : status === "crashed" ? "rgba(242,54,69,0.12)"
                : status === "no_folds" ? "rgba(239,83,80,0.10)"
                : "var(--color-elevated)";
              return (
                <span key={model} style={{
                  background: statusBg,
                  border: `1px solid ${statusColor}33`,
                  borderRadius: 3,
                  padding: "4px 10px",
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                  color: statusColor,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}>
                  <span style={{ fontWeight: 500 }}>{model}</span>
                  <span style={{ opacity: 0.7, fontSize: 9 }}>{status.replace(/_/g, " ")}</span>
                </span>
              );
            })}
          </div>
        </>
      )}

      {/* Alternative: show survivors list when hpo_status not available */}
      {(!results.hpo_status || Object.keys(results.hpo_status).length === 0) && (results.phase0_survivors?.length ?? 0) > 0 && (
        <>
          <SectionHeader label="Phase 3: Survivors" />
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 20 }}>
            {(results.phase0_survivors ?? []).map((m: string) => (
              <span key={m} style={{ background: "rgba(8,153,129,0.12)", border: "1px solid rgba(8,153,129,0.25)", borderRadius: 3, padding: "3px 9px", fontSize: 10, fontFamily: "var(--font-mono)", color: "#089981" }}>{m}</span>
            ))}
          </div>
        </>
      )}

      {/* Phase 4: Trust Score + Validation */}
      {results.trust_score && (
        <>
          <SectionHeader label="Phase 4: Validation — Trust Score" />
          <div style={{ marginBottom: 20 }}>
            {/* Trust Score Gauge */}
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 16,
              marginBottom: 16,
              padding: 16,
              background: "var(--color-elevated)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 6,
            }}>
              <div style={{
                width: 64,
                height: 64,
                borderRadius: "50%",
                background: results.trust_score.trust_score >= 0.80 ? "rgba(8,153,129,0.15)"
                  : results.trust_score.trust_score >= 0.60 ? "rgba(242,180,54,0.15)"
                  : results.trust_score.trust_score >= 0.40 ? "rgba(242,145,54,0.15)"
                  : "rgba(242,54,69,0.15)",
                border: `3px solid ${
                  results.trust_score.trust_score >= 0.80 ? "#089981"
                  : results.trust_score.trust_score >= 0.60 ? "#F2B436"
                  : results.trust_score.trust_score >= 0.40 ? "#F29136"
                  : "#F23645"
                }`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}>
                <span style={{
                  fontSize: 22,
                  fontWeight: 700,
                  fontFamily: "var(--font-mono)",
                  color: results.trust_score.trust_score >= 0.80 ? "#089981"
                    : results.trust_score.trust_score >= 0.60 ? "#F2B436"
                    : results.trust_score.trust_score >= 0.40 ? "#F29136"
                    : "#F23645",
                }}>
                  {Number(results.trust_score.trust_score * 100).toFixed(0)}
                </span>
              </div>
              <div>
                <div style={{
                  fontSize: 14,
                  fontWeight: 600,
                  fontFamily: "var(--font-mono)",
                  color: "var(--color-text-primary)",
                  marginBottom: 2,
                }}>
                  Trust Score: {results.trust_score.trust_score.toFixed(3)}
                </div>
                <span style={{
                  fontSize: 11,
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  padding: "2px 10px",
                  borderRadius: 3,
                  background: results.trust_score.action === "deploy" ? "rgba(8,153,129,0.15)"
                    : results.trust_score.action === "proceed" ? "rgba(242,180,54,0.15)"
                    : results.trust_score.action === "flag" ? "rgba(242,145,54,0.15)"
                    : "rgba(242,54,69,0.15)",
                  color: results.trust_score.action === "deploy" ? "#089981"
                    : results.trust_score.action === "proceed" ? "#F2B436"
                    : results.trust_score.action === "flag" ? "#F29136"
                    : "#F23645",
                }}>
                  {results.trust_score.action.toUpperCase()}
                </span>
              </div>
            </div>

            {/* Sub-score metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10 }}>
              <MetricCard label="PBO" value={Number(results.pbo ?? 1).toFixed(3)} color={Number(results.pbo ?? 1) < 0.20 ? "#089981" : Number(results.pbo ?? 1) < 0.50 ? "#F2B436" : "#F23645"} />
              <MetricCard label="DSR" value={Number(results.dsr ?? 0).toFixed(3)} color={Number(results.dsr ?? 0) > 0.90 ? "#089981" : Number(results.dsr ?? 0) > 0.50 ? "#F2B436" : "#F23645"} />
              <MetricCard label="Fold CV" value={Number(results.phase3_fold_consistency_cv ?? 0).toFixed(3)} color={Number(results.phase3_fold_consistency_cv ?? 0) < 1.0 ? "#089981" : "#F2B436"} />
              <MetricCard label="Seed Sharpe" value={Number(results.phase3_seed_robustness_sharpe ?? 0).toFixed(3)} color={Number(results.phase3_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"} />
            </div>
          </div>
        </>
      )}

      {/* Fallback: show old-style Phase 5 gates if trust_score not available */}
      {!results.trust_score && (
        <>
          <SectionHeader label="Phase 4: Validation" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
            <MetricCard label="Fold CV" value={Number(results.phase3_fold_consistency_cv ?? 0).toFixed(3)} color={Number(results.phase3_fold_consistency_cv ?? 0) < 1.0 ? "#089981" : "#F2B436"} />
            <MetricCard label="Seed Sharpe" value={Number(results.phase3_seed_robustness_sharpe ?? 0).toFixed(3)} color={Number(results.phase3_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"} />
          </div>
        </>
      )}

      {results.status === "validation_failed" && (
        <div style={{ marginBottom: 20, padding: 14, background: "rgba(242,180,54,0.06)", border: "1px solid rgba(242,180,54,0.2)", borderRadius: 4, fontSize: 10, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
          <span style={{ fontWeight: 600, color: "#F2B436" }}>Validation halted pipeline.</span> Phase 5 was skipped. Review the diagnostics above and adjust parameters. Common fixes: increase train_months, try different model types, or adjust HPO trial budgets.
        </div>
      )}

      {/* Team Backtest */}
      {results.racecar_backtest && (
        <SectionHeader label="Team Backtest" />
      )}
      {results.racecar_backtest && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 24 }}>
          <MetricCard label="Avg Sharpe" value={Number((results.racecar_backtest as Record<string, unknown>)?.avg_sharpe ?? 0).toFixed(3)} color={Number((results.racecar_backtest as Record<string, unknown>)?.avg_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"} />
          <MetricCard label="Avg Trades" value={String((results.racecar_backtest as Record<string, unknown>)?.avg_trades ?? 0)} color="var(--color-text-secondary)" />
          <MetricCard label="Folds" value={String((results.racecar_backtest as Record<string, unknown>)?.total_folds ?? 0)} color="var(--color-text-secondary)" />
          <MetricCard label="Models in Config" value={String(((results.racecar_backtest as Record<string, unknown>)?.models as unknown[])?.length ?? 0)} color="var(--color-text-secondary)" />
        </div>
      )}

      {/* Final validation (10-year WFO) */}
      {results.final_fold_consistency_cv !== undefined && results.final_fold_consistency_cv > 0 && (
        <>
          <SectionHeader label="Final Validation (10-year WFO + 5-seed)" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
            <MetricCard label="Fold CV" value={Number(results.final_fold_consistency_cv ?? 0).toFixed(3)} color={results.final_fold_consistency_pass ? "#089981" : "#F23645"} />
            <MetricCard label="Regime Coverage" value={results.final_regime_coverage ? "PASS" : "FAIL"} color={results.final_regime_coverage ? "#089981" : "#F23645"} />
            <MetricCard label="Seed Robustness" value={results.final_seed_robustness_pass ? "PASS" : "FAIL"} color={results.final_seed_robustness_pass ? "#089981" : "#F23645"} />
            <MetricCard label="Seed Sharpe" value={Number(results.final_seed_robustness_sharpe ?? 0).toFixed(3)} color={Number(results.final_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"} />
          </div>
        </>
      )}

      {/* Factory results */}
      {(results.factory_total_iterations ?? 0) > 0 && (
        <>
          <SectionHeader label="Factory Optimization" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 24 }}>
            <MetricCard label="Best Sharpe" value={Number(results.factory_best_sharpe).toFixed(4)} color="#089981" />
            <MetricCard label="Iterations" value={String(results.factory_total_iterations)} color="var(--color-text-secondary)" />
            <MetricCard label="Accepted" value={String(results.factory_accepted_count)} color="var(--color-text-secondary)" />
          </div>
        </>
      )}

      {/* Best config after Factory */}
      {results.factory_best_config && (
        <div style={{ marginBottom: 20 }}>
          <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
            Optimized Committee Config
          </span>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 10 }}>
            {Object.entries(
              ((results.factory_best_config as Record<string, unknown>).regimes as Record<string, Record<string, unknown>>) ?? {}
            ).map(([regime, a]) => (
              <div key={regime} style={{ display: "flex", alignItems: "center", gap: 14, fontSize: 11 }}>
                <span style={{
                  width: 130,
                  fontWeight: 500,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: "var(--color-brand)",
                  flexShrink: 0,
                }}>
                  {regime.replace(/_/g, " ")}
                </span>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {(a.models as string[])?.map((m: string, i: number) => (
                    <span
                      key={`${m}-${i}`}
                      style={{
                        background: "var(--color-elevated)",
                        border: "1px solid var(--color-glass-border)",
                        borderRadius: 3,
                        padding: "3px 9px",
                        fontSize: 10,
                        fontFamily: "var(--font-mono)",
                        color: "var(--color-text-secondary)",
                      }}
                    >
                      {m} {((Number((a.weights as number[])?.[i] ?? 0) * 100)).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Factory iteration history */}
      {(results.factory_history?.length ?? 0) > 0 && (
        <div style={{ marginBottom: 16 }}>
          <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
            Optimization Log
          </span>
          <div style={{ marginTop: 8, overflow: "auto", maxHeight: 300 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
              <thead>
                <tr>
                  <th style={thStyle}>#</th>
                  <th style={thStyle}>Action</th>
                  <th style={thStyle}>Regime</th>
                  <th style={thStyle}>Model Change</th>
                  <th style={{ ...thStyle, textAlign: "right" }}>Sharpe Δ</th>
                  <th style={thStyle}></th>
                </tr>
              </thead>
              <tbody>
                {(results.factory_history ?? []).map((row: FactoryIterationRecord, i: number) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--color-glass-border)" }}>
                    <td style={tdStyleMono}>{row.iteration}</td>
                    <td style={{ ...tdStyleMono, color: "var(--color-text-secondary)" }}>{row.action_type.toUpperCase()}</td>
                    <td style={tdStyle}>{row.regime.replace(/_/g, " ")}</td>
                    <td style={{ ...tdStyleMono, color: "var(--color-text-dim)" }}>
                      {[row.model_add, row.model_remove].filter(Boolean).join(" / ")}
                    </td>
                    <td style={{ ...tdStyleMono, textAlign: "right", color: row.delta_sharpe >= 0 ? "#089981" : "#F23645" }}>
                      {row.delta_sharpe >= 0 ? "+" : ""}{Number(row.delta_sharpe).toFixed(4)}
                    </td>
                    <td style={{ ...tdStyle, textAlign: "center", fontSize: 12 }}>
                      {row.accepted ? <span style={{ color: "#089981" }}>✓</span> : <span style={{ color: "#F23645" }}>✗</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Stop reason */}
      {results.factory_stop_reason && (
        <div style={{ marginTop: 8, padding: 10, background: "rgba(0,229,255,0.05)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, fontSize: 11, color: "var(--color-brand)", letterSpacing: "0.04em" }}>
          Stopped — {formatStopReason(results.factory_stop_reason)}
        </div>
      )}

      <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
        <button onClick={onRunAgain} style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "8px 20px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>Run Again</button>
        <button onClick={handleDownload} style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "8px 20px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>Download JSON</button>
        {results.factory_best_config && results.status !== "validation_failed" && (
          <button onClick={handleDeploy} style={{ background: "var(--color-accent-success)", border: "none", borderRadius: 4, color: "var(--color-text-inverse)", padding: "8px 20px", fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>Deploy to Trading</button>
        )}
      </div>
    </div>
  );
}
