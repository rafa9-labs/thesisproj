import { useState } from "react";
import {
  useStartFactory,
  useFactoryStatus,
  useFactoryResults,
} from "@/api/queries";
import type { FactoryIterationRecord, FactoryStartRequest } from "@/api/schemas";

const DEFAULT_MODELS = [
  "logistic", "random_forest", "xgboost", "lightgbm", "decision_tree",
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

export function FactoryTab() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>(DEFAULT_MODELS);
  const [proposer, setProposer] = useState("llm");
  const [llmBackend, setLlmBackend] = useState("deepseek");
  const [maxIter, setMaxIter] = useState(20);
  const [patience, setPatience] = useState(5);
  const [tolerance, setTolerance] = useState(0.02);
  const [configOpen, setConfigOpen] = useState(false);

  const startMutation = useStartFactory();
  const { data: status, isFetching: statusPolling } = useFactoryStatus(jobId);
  const { data: results } = useFactoryResults(
    status?.phase === "completed" || status?.phase === "failed" ? jobId : null,
  );

  const isRunning = status && status.phase !== "completed" && status.phase !== "failed";

  function handleStart() {
    const req: FactoryStartRequest = {
      models,
      proposer,
      llm_backend: llmBackend,
      max_iterations: maxIter,
      patience,
      stopping_tolerance: tolerance,
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
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
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
              Factory
            </h2>
            <p style={{ fontSize: 11, color: "var(--color-text-secondary)", margin: "6px 0 0" }}>
              Iteratively optimizes the committee through propose → test → accept/reject cycles.
              {proposer === "llm" && " LLM analyzes the regime matrix to find the best swaps."}
            </p>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {proposer === "llm" && (
              <span
                style={{
                  background: "var(--color-accent-ensemble)",
                  color: "var(--color-text-inverse)",
                  borderRadius: 4,
                  padding: "4px 10px",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                }}
              >
                LLM
              </span>
            )}
            {proposer !== "llm" && (
              <span
                style={{
                  background: "var(--color-accent-classical)",
                  color: "var(--color-text-inverse)",
                  borderRadius: 4,
                  padding: "4px 10px",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                }}
              >
                Deterministic
              </span>
            )}
            {!isRunning && (
              <button
                onClick={handleStart}
                disabled={startMutation.isPending || models.length === 0}
                style={{
                  background: "var(--color-brand)",
                  color: "var(--color-text-inverse)",
                  border: "none",
                  borderRadius: 4,
                  padding: "10px 24px",
                  fontSize: 12,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  opacity: models.length === 0 ? 0.4 : 1,
                }}
              >
                {startMutation.isPending ? "Starting..." : "Start Loop"}
              </button>
            )}
          </div>
        </div>

        {/* Config toggle */}
        {!isRunning && (
          <>
            <button
              onClick={() => setConfigOpen(!configOpen)}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--color-text-muted)",
                fontSize: 10,
                fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                cursor: "pointer",
                marginTop: 12,
                padding: 0,
              }}
            >
              {configOpen ? "▾" : "▸"} Configure
            </button>
            {configOpen && (
              <div
                style={{
                  marginTop: 12,
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                {/* Model selection */}
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span style={{ fontSize: 10, color: "var(--color-text-muted)", fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                    Models to Optimize
                  </span>
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    {[
                      "logistic", "svm", "random_forest", "decision_tree",
                      "xgboost", "lightgbm", "catboost",
                      "cnn", "lstm", "transformer", "gru", "gru_lstm",
                      "meta_ensemble", "stacking_ensemble", "ensemble_adaptive_regime",
                    ].map((m) => (
                      <button
                        key={m}
                        onClick={() => toggleModel(m)}
                        style={{
                          background: models.includes(m) ? "var(--color-brand-glow)" : "var(--color-elevated)",
                          border: `1px solid ${models.includes(m) ? "var(--color-brand)" : "var(--color-glass-border)"}`,
                          borderRadius: 4,
                          padding: "3px 8px",
                          fontSize: 10,
                          fontFamily: "var(--font-mono)",
                          color: models.includes(m) ? "var(--color-brand)" : "var(--color-text-muted)",
                          cursor: "pointer",
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Proposer */}
                <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
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
                      style={{ width: 40, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }}
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
                      style={{ width: 50, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    />
                  </label>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Progress Panel ── */}
      {(isRunning || status?.phase === "completed" || status?.phase === "failed") && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 24,
          }}
        >
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <span
              style={{
                fontSize: 13,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
              }}
            >
              {status.phase === "completed" ? "Optimization Complete" : status.phase === "failed" ? "Failed" : `Iteration ${status.iteration}/${status.total_iterations}`}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {isRunning && (
                <>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-brand)", animation: "pulse 1.5s infinite" }} />
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                    Processing
                  </span>
                </>
              )}
              {status.phase === "completed" && (
                <span style={{ fontSize: 10, color: "var(--color-accent-success)", fontWeight: 500, letterSpacing: "0.06em" }}>
                  Done
                </span>
              )}
            </div>
          </div>

          {/* Progress bar */}
          {isRunning && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ height: 4, borderRadius: 2, background: "var(--color-elevated)", overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${(status.iteration / Math.max(1, status.total_iterations)) * 100}%`,
                    background: "var(--color-brand)",
                    borderRadius: 2,
                    transition: "width 0.5s ease",
                  }}
                />
              </div>
            </div>
          )}

          {/* Current action */}
          {(isRunning || status.current_action) && (
            <div
              style={{
                padding: 14,
                background: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 4,
                marginBottom: 16,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span
                  style={{
                    background: status.accepted ? "rgba(8,153,129,0.15)" : "rgba(242,54,69,0.15)",
                    color: status.accepted ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                    borderRadius: 3,
                    padding: "2px 8px",
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: "0.06em",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {status.current_action || "Starting..."}
                </span>
                {status.current_regime && (
                  <span style={{ fontSize: 10, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                    in {status.current_regime.replace(/_/g, " ")}
                  </span>
                )}
                {isRunning && status.iteration > 0 && (
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: status.accepted ? "var(--color-accent-success)" : "var(--color-accent-danger)", marginLeft: "auto" }}>
                    {status.accepted ? "ACCEPTED" : "REJECTED"}
                  </span>
                )}
              </div>
              {status.iteration > 0 && (
                <div style={{ display: "flex", gap: 24, marginTop: 8 }}>
                  <MetricInline label="Before" value={status.before_sharpe.toFixed(4)} />
                  <MetricInline label="After" value={status.after_sharpe.toFixed(4)} />
                  <MetricInline
                    label="Delta"
                    value={`${status.delta_sharpe >= 0 ? "+" : ""}${status.delta_sharpe.toFixed(4)}`}
                    color={status.delta_sharpe >= 0 ? "#089981" : "#F23645"}
                  />
                  <MetricInline label="Best" value={status.best_sharpe_so_far.toFixed(4)} color="var(--color-brand" />
                </div>
              )}
            </div>
          )}

          {/* Iteration history */}
          {status.history.length > 0 && (
            <IterationLog history={status.history} />
          )}

          {/* Stopped reason */}
          {status.stopped && status.stop_reason && (
            <div style={{ marginTop: 12, padding: 10, background: "rgba(0,229,255,0.05)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, fontSize: 11, color: "var(--color-brand", letterSpacing: "0.04em" }}>
              Stopped — {formatStopReason(status.stop_reason)}
            </div>
          )}

          {status.phase === "failed" && (
            <div style={{ marginTop: 12, padding: 12, background: "rgba(242,54,69,0.08)", border: "1px solid rgba(242,54,69,0.2)", borderRadius: 4, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-accent-danger" }}>
              {status.stop_reason || "Unknown error"}
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
            padding: 24,
          }}
        >
          <h3 style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-primary)", margin: "0 0 16px" }}>
            Results
            <span style={{ fontSize: 10, color: "var(--color-text-muted)", marginLeft: 8, fontWeight: 400 }}>
              {Number(results.total_time_s).toFixed(0)}s
            </span>
          </h3>

          {/* Summary cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 10, marginBottom: 20 }}>
            <MetricBlock label="Best Sharpe" value={Number(results.best_sharpe).toFixed(4)} color="#089981" />
            <MetricBlock label="Iterations" value={String(results.total_iterations)} color="var(--color-text-secondary)" />
            <MetricBlock label="Accepted" value={String(results.accepted_count)} color="var(--color-text-secondary)" />
          </div>

          {/* Best config */}
          {results.best_config && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
                Best Committee Config
              </span>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 8 }}>
                {Object.entries(results.best_config.regimes as Record<string, Record<string, unknown>> ?? {}).map(([regime, a]) => (
                  <div key={regime} style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 11 }}>
                    <span style={{ width: 120, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-brand", flexShrink: 0 }}>
                      {regime.replace(/_/g, " ")}
                    </span>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                      {(a.models as string[])?.map((m: string, i: number) => (
                        <span key={`${m}-${i}`} style={{ background: "var(--color-elevated)", borderRadius: 3, padding: "3px 8px", fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                          {m} {((Number((a.weights as number[])?.[i] ?? 0) * 100)).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <button onClick={handleReset} style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "6px 16px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>
            Run Again
          </button>
        </div>
      )}
    </div>
  );
}

function IterationLog({ history }: { history: FactoryIterationRecord[] }) {
  return (
    <div>
      <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
        Iteration Log
      </span>
      <div style={{ marginTop: 8, overflow: "auto", maxHeight: 260 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr>
              <TH>#</TH>
              <TH>Action</TH>
              <TH>Regime</TH>
              <TH>Details</TH>
              <TH style={{ textAlign: "right" }}>Sharpe Δ</TH>
              <TH></TH>
            </tr>
          </thead>
          <tbody>
            {history.map((row) => (
              <tr
                key={row.iteration}
                style={{ borderBottom: "1px solid var(--color-glass-border)" }}
              >
                <td style={{ padding: "5px 6px", fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                  {row.iteration}
                </td>
                <td style={{ padding: "5px 6px", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  {row.action_type.toUpperCase()}
                </td>
                <td style={{ padding: "5px 6px", color: "var(--color-text-secondary)" }}>
                  {row.regime.replace(/_/g, " ")}
                </td>
                <td style={{ padding: "5px 6px", fontFamily: "var(--font-mono)", color: "var(--color-text-dim)", fontSize: 9 }}>
                  {[row.model_add, row.model_remove].filter(Boolean).join(" / ")}
                </td>
                <td style={{ padding: "5px 6px", textAlign: "right", fontFamily: "var(--font-mono)", color: row.delta_sharpe >= 0 ? "#089981" : "#F23645" }}>
                  {row.delta_sharpe >= 0 ? "+" : ""}{Number(row.delta_sharpe).toFixed(4)}
                </td>
                <td style={{ padding: "5px 4px", textAlign: "center", fontSize: 12 }}>
                  {row.accepted ? <span style={{ color: "#089981" }}>✓</span> : <span style={{ color: "#F23645" }}>✗</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TH({ children, style: extra }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <th
      style={{
        padding: "4px 6px",
        textAlign: "left",
        color: "var(--color-text-muted)",
        fontWeight: 500,
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        fontSize: 9,
        borderBottom: "1px solid var(--color-glass-border)",
        ...extra,
      }}
    >
      {children}
    </th>
  );
}

function MetricInline({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 9, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
        {label}
      </span>
      <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: color ?? "var(--color-text-primary)", fontWeight: 600 }}>
        {value}
      </span>
    </div>
  );
}

function MetricBlock({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, padding: "12px 14px", textAlign: "center" }}>
      <div style={{ fontSize: 20, fontWeight: 600, fontFamily: "var(--font-mono)", color, marginBottom: 4 }}>
        {value}
      </div>
      <div style={{ fontSize: 9, fontWeight: 500, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
        {label}
      </div>
    </div>
  );
}
