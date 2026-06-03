import { useState } from "react";
import {
  useStartFullCycle,
  useFullCycleStatus,
  useFullCycleResults,
} from "@/api/queries";
import type { FactoryIterationRecord, FullCycleRequest } from "@/api/schemas";

const DEFAULT_MODELS = [
  "logistic", "random_forest", "xgboost", "lightgbm", "decision_tree",
];

const ALL_MODELS = [
  "logistic", "svm", "random_forest", "decision_tree",
  "xgboost", "lightgbm", "catboost",
  "cnn", "lstm", "transformer", "gru", "gru_lstm",
  "meta_ensemble", "stacking_ensemble", "ensemble_adaptive_regime",
];

const STOP_REASONS: Record<string, string> = {
  budget: "Max iterations reached",
  patience: "Global best Sharpe has not improved",
  hard_gate: "All regimes above Sharpe floor",
  exhaustion: "No untested model improves any regime",
  divergence: "3 consecutive deteriorating iterations",
};

function formatStopReason(raw: string): string {
  for (const [key, label] of Object.entries(STOP_REASONS)) {
    if (raw.startsWith(key)) return label + (raw.includes(":") ? raw.slice(raw.indexOf(":")) : "");
  }
  return raw || "Optimization stopped";
}

const FC_PHASES = ["profiling", "building", "backtesting", "optimizing"] as const;
const PHASE_LABEL: Record<string, string> = {
  profiling: "Profiling",
  building: "Building",
  backtesting: "Backtesting",
  optimizing: "Optimizing",
};
const PHASE_DESC: Record<string, string> = {
  profiling: "Running all models across walk-forward folds to build regime x model matrix",
  building: "Selecting top models per regime and assigning blend weights",
  backtesting: "Validating committee with full walk-forward backtest",
  optimizing: "Iteratively improving committee via propose -> test -> accept/reject",
};

export function FullCycleTab() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>(DEFAULT_MODELS);
  const [proposer, setProposer] = useState("llm");
  const [llmBackend, setLlmBackend] = useState("deepseek");
  const [maxIter, setMaxIter] = useState(20);
  const [patience, setPatience] = useState(5);
  const [tolerance, setTolerance] = useState(0.02);

  const startMutation = useStartFullCycle();
  const { data: status } = useFullCycleStatus(jobId);
  const { data: results } = useFullCycleResults(
    status?.phase === "completed" || status?.phase === "failed" ? jobId : null,
  );

  const isRunning = status && status.phase !== "completed" && status.phase !== "failed";
  const currentPhaseIdx = FC_PHASES.indexOf(status?.phase as typeof FC_PHASES[number] ?? "");

  function handleStart() {
    const req: FullCycleRequest = {
      models,
      proposer,
      llm_backend: llmBackend,
      max_iterations: maxIter,
      patience,
      stopping_tolerance: tolerance,
      profile_trials: 5,
      committee_top_k: 3,
      train_months: 6,
    };
    startMutation.mutate(req, { onSuccess: (data) => setJobId(data.job_id) });
  }

  function handleReset() {
    setJobId(null);
  }

  function toggleModel(model: string) {
    setModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model],
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* ── Control Panel ── */}
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 24,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
          <div style={{ maxWidth: 520 }}>
            <h2
              style={{
                fontSize: 16,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
                margin: 0,
              }}
            >
              Full Cycle
            </h2>
            <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "6px 0 0", lineHeight: 1.5 }}>
              End-to-end pipeline: profile all models across market regimes, build the optimal
              committee, validate via walk-forward backtest, then iteratively optimize with
              {proposer === "llm" ? " LLM-guided" : ""} propose/test cycles. One click from
              raw data to deployable committee config.
            </p>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0 }}>
            {!isRunning && (
              <button
                onClick={handleStart}
                disabled={startMutation.isPending || models.length === 0}
                style={{
                  background: "var(--color-brand)",
                  color: "var(--color-text-inverse)",
                  border: "none",
                  borderRadius: 4,
                  padding: "12px 28px",
                  fontSize: 13,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  opacity: models.length === 0 ? 0.4 : 1,
                }}
              >
                {startMutation.isPending ? "Starting..." : "Start Full Cycle"}
              </button>
            )}
          </div>
        </div>

        {/* Model selection + settings */}
        {!isRunning && (
          <div style={{ marginTop: 20 }}>
            <span
              style={{
                fontSize: 10,
                fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: "var(--color-text-muted)",
              }}
            >
              Models to Run ({models.length} selected)
            </span>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 8 }}>
              {ALL_MODELS.map((m) => (
                <button
                  key={m}
                  onClick={() => toggleModel(m)}
                  style={{
                    background: models.includes(m) ? "var(--color-brand-glow)" : "var(--color-elevated)",
                    border: `1px solid ${models.includes(m) ? "var(--color-brand)" : "var(--color-glass-border)"}`,
                    borderRadius: 4,
                    padding: "4px 10px",
                    fontSize: 10,
                    fontFamily: "var(--font-mono)",
                    color: models.includes(m) ? "var(--color-brand)" : "var(--color-text-muted)",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  {m}
                </button>
              ))}
            </div>

            {/* Settings row */}
            <div style={{ marginTop: 16, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Proposer:{" "}
                <select
                  value={proposer}
                  onChange={(e) => setProposer(e.target.value)}
                  style={{
                    background: "var(--color-input-bg)",
                    border: "1px solid var(--color-glass-border)",
                    borderRadius: 4,
                    color: "var(--color-text-primary)",
                    padding: "3px 6px",
                    fontSize: 10,
                  }}
                >
                  <option value="llm">LLM (DeepSeek V4)</option>
                  <option value="ollama">LLM (Ollama)</option>
                  <option value="deterministic">Deterministic Greedy</option>
                </select>
              </label>
              {proposer !== "deterministic" && (
                <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                  Backend:{" "}
                  <select
                    value={llmBackend}
                    onChange={(e) => setLlmBackend(e.target.value)}
                    style={{
                      background: "var(--color-input-bg)",
                      border: "1px solid var(--color-glass-border)",
                      borderRadius: 4,
                      color: "var(--color-text-primary)",
                      padding: "3px 6px",
                      fontSize: 10,
                    }}
                  >
                    <option value="deepseek">DeepSeek</option>
                    <option value="ollama">Ollama</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                  </select>
                </label>
              )}
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Max Iter:{" "}
                <input
                  type="number" value={maxIter}
                  onChange={(e) => setMaxIter(Number(e.target.value))}
                  min={3} max={50}
                  style={{ width: 42, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }}
                />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Patience:{" "}
                <input
                  type="number" value={patience}
                  onChange={(e) => setPatience(Number(e.target.value))}
                  min={3} max={15}
                  style={{ width: 40, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }}
                />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Tolerance:{" "}
                <input
                  type="number" value={tolerance}
                  onChange={(e) => setTolerance(Number(e.target.value))}
                  min={0.005} max={0.1} step={0.005}
                  style={{ width: 52, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }}
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* ── Progress Panel ── */}
      {(isRunning || status?.phase === "completed" || status?.phase === "failed") && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 28,
          }}
        >
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
            <span
              style={{
                fontSize: 14,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
              }}
            >
              {status.phase === "completed" ? "Full Cycle Complete" : status.phase === "failed" ? "Pipeline Failed" : "Pipeline Progress"}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {isRunning && (
                <>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-brand)", animation: "pulse 1.5s infinite" }} />
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                    Running
                  </span>
                </>
              )}
              {status.phase === "completed" && (
                <span style={{ fontSize: 11, color: "var(--color-accent-success)", fontWeight: 500, letterSpacing: "0.06em" }}>
                  Complete
                </span>
              )}
            </div>
          </div>

          {/* 4-Phase progress bar */}
          <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
            {FC_PHASES.map((phase, idx) => {
              const isActive = phase === status.phase;
              const isDone = idx < currentPhaseIdx || status.phase === "completed";
              return (
                <div key={phase} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{
                    height: 5,
                    borderRadius: 2.5,
                    background: isDone
                      ? "var(--color-accent-success)"
                      : isActive
                        ? "var(--color-brand)"
                        : "var(--color-elevated)",
                    transition: "background 0.4s ease",
                  }} />
                  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                    <span style={{
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      color: isDone || isActive ? "var(--color-text-primary)" : "var(--color-text-dim)",
                    }}>
                      {PHASE_LABEL[phase]}
                    </span>
                    <span style={{
                      fontSize: 9,
                      color: isDone || isActive ? "var(--color-text-secondary)" : "var(--color-text-dim)",
                      lineHeight: 1.3,
                    }}>
                      {PHASE_DESC[phase]}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Current action / Factory iteration */}
          {isRunning && status?.phase === "optimizing" && status?.current_action && (
            <div
              style={{
                padding: 14,
                background: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 4,
                marginBottom: 16,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{
                  background: "var(--color-brand-glow)",
                  color: "var(--color-brand)",
                  borderRadius: 3,
                  padding: "2px 8px",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  fontFamily: "var(--font-mono)",
                }}>
                  Iteration {status.iteration}/{status.total_iterations}
                </span>
                <span style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>
                  {status.current_action}
                </span>
                {status.best_sharpe_so_far !== undefined && (
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", marginLeft: "auto" }}>
                    Best: {status.best_sharpe_so_far.toFixed(4)}
                  </span>
                )}
              </div>
            </div>
          )}

          {status?.phase === "failed" && (
            <div style={{ padding: 12, background: "rgba(242,54,69,0.08)", border: "1px solid rgba(242,54,69,0.2)", borderRadius: 4, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-accent-danger" }}>
              {status.error || "Unknown error"}
            </div>
          )}
        </div>
      )}

      {/* ── Results Panel ── */}
      {results && status?.phase === "completed" && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 28,
          }}
        >
          <h3 style={{
            fontSize: 14,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
            margin: "0 0 20px",
          }}>
            Pipeline Results
            <span style={{ fontSize: 11, color: "var(--color-text-muted)", marginLeft: 10, fontWeight: 400, fontFamily: "var(--font-mono)" }}>
              {Number(results.total_time_s).toFixed(0)}s
            </span>
          </h3>

          {/* Racecar backtest summary */}
          {results.racecar_backtest && (
            <SectionHeader label="Racecar Backtest" />
          )}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 24 }}>
            <MetricCard
              label="Avg Sharpe"
              value={Number((results.racecar_backtest as Record<string, unknown> | undefined)?.avg_sharpe ?? 0).toFixed(3)}
              color={Number((results.racecar_backtest as Record<string, unknown> | undefined)?.avg_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"}
            />
            <MetricCard
              label="Avg Trades"
              value={String((results.racecar_backtest as Record<string, unknown> | undefined)?.avg_trades ?? 0)}
              color="var(--color-text-secondary)"
            />
            <MetricCard
              label="Folds"
              value={String((results.racecar_backtest as Record<string, unknown> | undefined)?.total_folds ?? 0)}
              color="var(--color-text-secondary)"
            />
            <MetricCard
              label="Models in Config"
              value={String((results.racecar_backtest as Record<string, unknown> | undefined)?.models?.length ?? 0)}
              color="var(--color-text-secondary)"
            />
          </div>

          {/* Factory results */}
          <SectionHeader label="Factory Optimization" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 24 }}>
            <MetricCard label="Best Sharpe" value={Number(results.factory_best_sharpe).toFixed(4)} color="#089981" />
            <MetricCard label="Iterations" value={String(results.factory_total_iterations)} color="var(--color-text-secondary)" />
            <MetricCard label="Accepted" value={String(results.factory_accepted_count)} color="var(--color-text-secondary)" />
          </div>

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
          {results.factory_history?.length > 0 && (
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
                    {results.factory_history.map((row: FactoryIterationRecord, i: number) => (
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

          {/* Factory stop reason */}
          {results.factory_stop_reason && (
            <div style={{ marginTop: 8, padding: 10, background: "rgba(0,229,255,0.05)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, fontSize: 11, color: "var(--color-brand", letterSpacing: "0.04em" }}>
              Stopped — {formatStopReason(results.factory_stop_reason)}
            </div>
          )}

          <button onClick={handleReset} style={{ marginTop: 24, background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "8px 20px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>
            Run Again
          </button>
        </div>
      )}
    </div>
  );
}

function SectionHeader({ label }: { label: string }) {
  return (
    <span style={{
      display: "block",
      fontSize: 10,
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
      <div style={{ fontSize: 9, fontWeight: 500, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
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
  fontSize: 9,
  borderBottom: "1px solid var(--color-glass-border)",
};

const tdStyle: React.CSSProperties = { padding: "5px 8px", color: "var(--color-text-secondary)", fontSize: 10 };
const tdStyleMono: React.CSSProperties = { ...tdStyle, fontFamily: "var(--font-mono)" };
