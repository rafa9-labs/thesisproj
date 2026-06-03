import { useState } from "react";
import {
  useStartFactory,
  useFactoryStatus,
  useFactoryResults,
  useStartFullCycle,
  useFullCycleStatus,
  useFullCycleResults,
} from "@/api/queries";
import type { FactoryIterationRecord, FactoryStartRequest, FullCycleRequest } from "@/api/schemas";

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

const PHASE_LABELS_FC: Record<string, string> = {
  profiling: "Profiling models across regimes",
  building: "Building committee config",
  backtesting: "Backtesting committee",
  optimizing: "Optimizing with Factory loop",
  completed: "Full cycle complete",
  failed: "Full cycle failed",
};

export function FactoryTab() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>(DEFAULT_MODELS);
  const [proposer, setProposer] = useState("llm");
  const [llmBackend, setLlmBackend] = useState("deepseek");
  const [maxIter, setMaxIter] = useState(20);
  const [patience, setPatience] = useState(5);
  const [tolerance, setTolerance] = useState(0.02);
  const [configOpen, setConfigOpen] = useState(false);
  const [isFullCycle, setIsFullCycle] = useState(false);

  // Factory-only hooks
  const startMutation = useStartFactory();
  const { data: status, isFetching: statusPolling } = useFactoryStatus(
    !isFullCycle ? jobId : null,
  );
  const { data: results } = useFactoryResults(
    !isFullCycle && (status?.phase === "completed" || status?.phase === "failed") ? jobId : null,
  );

  // Full-cycle hooks
  const startFcMutation = useStartFullCycle();
  const { data: fcStatus } = useFullCycleStatus(isFullCycle ? jobId : null);
  const { data: fcResults } = useFullCycleResults(
    isFullCycle && (fcStatus?.phase === "completed" || fcStatus?.phase === "failed") ? jobId : null,
  );

  // Active status/results depending on mode
  const activeStatus = isFullCycle ? fcStatus : status;
  const activeResults = isFullCycle ? fcResults : results;
  const activePending = isFullCycle ? startFcMutation.isPending : startMutation.isPending;
  const isRunning = activeStatus && activeStatus.phase !== "completed" && activeStatus.phase !== "failed";
  const phaseTitle = isFullCycle
    ? (PHASE_LABELS_FC[activeStatus?.phase ?? ""] ?? activeStatus?.phase ?? "")
    : (activeStatus?.phase === "completed" ? "Optimization Complete"
       : activeStatus?.phase === "failed" ? "Failed"
       : `Iteration ${activeStatus?.iteration ?? 0}/${activeStatus?.total_iterations ?? 0}`);

  function handleStart() {
    if (isFullCycle) {
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
      startFcMutation.mutate(req, { onSuccess: (data) => setJobId(data.job_id) });
    } else {
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
                {/* Full Cycle toggle */}
                <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
                  <label style={{ position: "relative", display: "inline-block", width: 32, height: 18, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={isFullCycle}
                      onChange={(e) => setIsFullCycle(e.target.checked)}
                      style={{ opacity: 0, width: 0, height: 0 }}
                    />
                    <span style={{
                      position: "absolute", inset: 0, borderRadius: 9,
                      background: isFullCycle ? "var(--color-brand)" : "var(--color-elevated)",
                      transition: "0.2s",
                      border: "1px solid var(--color-glass-border)",
                    }}>
                      <span style={{
                        position: "absolute", top: 2, left: isFullCycle ? 16 : 2,
                        width: 12, height: 12, borderRadius: "50%",
                        background: "white", transition: "0.2s",
                      }} />
                    </span>
                  </label>
                  <span style={{ fontSize: 10, color: "var(--color-text-muted)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                    Full Cycle
                  </span>
                  {isFullCycle && (
                    <span style={{ fontSize: 9, color: "var(--color-accent-warning)" }}>
                      (Racecar → Factory — takes longer)
                    </span>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
            )}
          </>
        )}
      </div>

      {/* ── Progress Panel ── */}
      {(isRunning || activeStatus?.phase === "completed" || activeStatus?.phase === "failed") && (
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
              {phaseTitle}
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
              {activeStatus?.phase === "completed" && (
                <span style={{ fontSize: 10, color: "var(--color-accent-success)", fontWeight: 500, letterSpacing: "0.06em" }}>
                  Done
                </span>
              )}
            </div>
          </div>

          {/* Phase progress bar (full cycle) */}
          {isFullCycle && isRunning && (
            <div style={{ marginBottom: 16, display: "flex", gap: 4 }}>
              {["profiling", "building", "backtesting", "optimizing"].map((phase) => {
                const current = activeStatus?.phase ?? "";
                const idx = ["profiling", "building", "backtesting", "optimizing"].indexOf(phase);
                const curIdx = ["profiling", "building", "backtesting", "optimizing"].indexOf(current);
                const isActive = phase === current;
                const isDone = idx < curIdx;
                return (
                  <div key={phase} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 3 }}>
                    <div style={{
                      height: 4, borderRadius: 2,
                      background: isDone ? "var(--color-accent-success)"
                        : isActive ? "var(--color-brand)"
                        : "var(--color-elevated)",
                      transition: "background 0.3s ease",
                    }} />
                    <span style={{
                      fontSize: 8, fontWeight: 500, letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      color: isDone || isActive ? "var(--color-text-primary)" : "var(--color-text-dim)",
                    }}>
                      {phase === "optimizing" ? "Factory" : phase.charAt(0).toUpperCase() + phase.slice(1, 3)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Progress bar (factory-only) */}
          {!isFullCycle && isRunning && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ height: 4, borderRadius: 2, background: "var(--color-elevated)", overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${((activeStatus?.iteration ?? 0) / Math.max(1, activeStatus?.total_iterations ?? 1)) * 100}%`,
                    background: "var(--color-brand)",
                    borderRadius: 2,
                    transition: "width 0.5s ease",
                  }}
                />
              </div>
            </div>
          )}

          {/* Current action */}
          {(isRunning || activeStatus?.current_action) && (
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
                    background: activeStatus?.accepted ? "rgba(8,153,129,0.15)" : "rgba(242,54,69,0.15)",
                    color: activeStatus?.accepted ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                    borderRadius: 3,
                    padding: "2px 8px",
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: "0.06em",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {activeStatus?.current_action || (isFullCycle ? "Running Racecar..." : "Starting...")}
                </span>
                {activeStatus?.current_regime && (
                  <span style={{ fontSize: 10, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                    in {activeStatus.current_regime.replace(/_/g, " ")}
                  </span>
                )}
                {isRunning && !isFullCycle && activeStatus?.iteration > 0 && (
                  <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: activeStatus?.accepted ? "var(--color-accent-success)" : "var(--color-accent-danger)", marginLeft: "auto" }}>
                    {activeStatus?.accepted ? "ACCEPTED" : "REJECTED"}
                  </span>
                )}
              </div>
              {!isFullCycle && activeStatus?.iteration > 0 && (
                <div style={{ display: "flex", gap: 24, marginTop: 8 }}>
                  <MetricInline label="Before" value={(activeStatus?.before_sharpe ?? 0).toFixed(4)} />
                  <MetricInline label="After" value={(activeStatus?.after_sharpe ?? 0).toFixed(4)} />
                  <MetricInline
                    label="Delta"
                    value={`${(activeStatus?.delta_sharpe ?? 0) >= 0 ? "+" : ""}${(activeStatus?.delta_sharpe ?? 0).toFixed(4)}`}
                    color={(activeStatus?.delta_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"}
                  />
                  <MetricInline label="Best" value={(activeStatus?.best_sharpe_so_far ?? 0).toFixed(4)} color="var(--color-brand" />
                </div>
              )}
              {isFullCycle && activeStatus?.best_sharpe_so_far !== undefined && (
                <div style={{ display: "flex", gap: 24, marginTop: 8 }}>
                  <MetricInline label="Best Sharpe" value={(activeStatus?.best_sharpe_so_far ?? 0).toFixed(4)} color="var(--color-brand" />
                </div>
              )}
            </div>
          )}

          {/* Job ID display for full cycle */}
          {isFullCycle && (
            <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)", marginBottom: 8 }}>
              Job: {activeStatus?.job_id}
            </div>
          )}

          {/* Factory iteration history */}
          {!isFullCycle && activeStatus?.history?.length > 0 && (
            <IterationLog history={activeStatus.history} />
          )}

          {/* Stopped reason */}
          {!isFullCycle && activeStatus?.stopped && activeStatus?.stop_reason && (
            <div style={{ marginTop: 12, padding: 10, background: "rgba(0,229,255,0.05)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, fontSize: 11, color: "var(--color-brand", letterSpacing: "0.04em" }}>
              Stopped — {formatStopReason(activeStatus.stop_reason)}
            </div>
          )}

          {activeStatus?.phase === "failed" && (
            <div style={{ marginTop: 12, padding: 12, background: "rgba(242,54,69,0.08)", border: "1px solid rgba(242,54,69,0.2)", borderRadius: 4, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-accent-danger" }}>
              {activeStatus?.error || activeStatus?.stop_reason || "Unknown error"}
            </div>
          )}
        </div>
      )}

      {/* ── Results Panel (Factory-only) ── */}
      {!isFullCycle && results && status?.phase === "completed" && (
        <FactoryResultsSection
          bestSharpe={results.best_sharpe}
          totalIterations={results.total_iterations}
          acceptedCount={results.accepted_count}
          totalTime={results.total_time_s}
          bestConfig={results.best_config}
          onReset={handleReset}
        />
      )}

      {/* ── Results Panel (Full Cycle) ── */}
      {isFullCycle && fcResults && fcStatus?.phase === "completed" && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 24,
          }}
        >
          <h3 style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-primary)", margin: "0 0 16px" }}>
            Full Cycle Complete
            <span style={{ fontSize: 10, color: "var(--color-text-muted)", marginLeft: 8, fontWeight: 400 }}>
              {Number(fcResults.total_time_s).toFixed(0)}s
            </span>
          </h3>

          {/* Racecar backtest summary */}
          {fcResults.racecar_backtest && (
            <div style={{ marginBottom: 20 }}>
              <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
                Racecar Backtest
              </span>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 10, marginTop: 8 }}>
                <MetricBlock label="Avg Sharpe" value={Number((fcResults.racecar_backtest as Record<string, unknown>).avg_sharpe ?? 0).toFixed(3)} color={(Number((fcResults.racecar_backtest as Record<string, unknown>).avg_sharpe ?? 0)) >= 0 ? "#089981" : "#F23645"} />
                <MetricBlock label="Avg Trades" value={String((fcResults.racecar_backtest as Record<string, unknown>).avg_trades ?? 0)} color="var(--color-text-secondary)" />
                <MetricBlock label="Folds" value={String((fcResults.racecar_backtest as Record<string, unknown>).total_folds ?? 0)} color="var(--color-text-secondary)" />
              </div>
            </div>
          )}

          {/* Factory results */}
          <div style={{ marginBottom: 20 }}>
            <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
              Factory Optimization
            </span>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 10, marginTop: 8 }}>
              <MetricBlock label="Best Sharpe" value={Number(fcResults.factory_best_sharpe).toFixed(4)} color="#089981" />
              <MetricBlock label="Iterations" value={String(fcResults.factory_total_iterations)} color="var(--color-text-secondary)" />
              <MetricBlock label="Accepted" value={String(fcResults.factory_accepted_count)} color="var(--color-text-secondary)" />
            </div>
          </div>

          {/* Factory best config */}
          {fcResults.factory_best_config && (
            <div style={{ marginBottom: 16 }}>
              <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
                Best Committee Config (after Factory)
              </span>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 8 }}>
                {Object.entries((fcResults.factory_best_config as Record<string, Record<string, unknown>>).regimes as Record<string, Record<string, unknown>> ?? {}).map(([regime, a]) => (
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

          {/* Factory iteration history */}
          {fcResults.factory_history?.length > 0 && (
            <IterationLog history={fcResults.factory_history.map((r) => ({
              iteration: r.iteration,
              action_type: r.action_type ?? r.action?.type ?? "",
              regime: r.regime ?? r.action?.regime ?? "",
              model_add: r.model_add ?? r.action?.model_add ?? "",
              model_remove: r.model_remove ?? r.action?.model_remove ?? "",
              before_sharpe: r.before_sharpe ?? 0,
              after_sharpe: r.after_sharpe ?? 0,
              delta_sharpe: r.delta_sharpe ?? 0,
              accepted: r.accepted ?? false,
              rationale: r.rationale ?? "",
            }))} />
          )}

          {/* Stop reason */}
          {fcResults.factory_stop_reason && (
            <div style={{ marginTop: 12, padding: 10, background: "rgba(0,229,255,0.05)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, fontSize: 11, color: "var(--color-brand", letterSpacing: "0.04em" }}>
              Stopped — {formatStopReason(fcResults.factory_stop_reason)}
            </div>
          )}

          <button onClick={handleReset} style={{ marginTop: 20, background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "6px 16px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>
            Run Again
          </button>
        </div>
      )}
    </div>
  );
}

function FactoryResultsSection({ bestSharpe, totalIterations, acceptedCount, totalTime, bestConfig, onReset }: {
  bestSharpe: number; totalIterations: number; acceptedCount: number; totalTime: number;
  bestConfig?: Record<string, unknown>; onReset: () => void;
}) {
  return (
    <div style={{ background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, padding: 24 }}>
      <h3 style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-primary)", margin: "0 0 16px" }}>
        Results
        <span style={{ fontSize: 10, color: "var(--color-text-muted)", marginLeft: 8, fontWeight: 400 }}>{Number(totalTime).toFixed(0)}s</span>
      </h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 10, marginBottom: 20 }}>
        <MetricBlock label="Best Sharpe" value={Number(bestSharpe).toFixed(4)} color="#089981" />
        <MetricBlock label="Iterations" value={String(totalIterations)} color="var(--color-text-secondary)" />
        <MetricBlock label="Accepted" value={String(acceptedCount)} color="var(--color-text-secondary)" />
      </div>
      {bestConfig && (
        <ConfigDisplay config={bestConfig} />
      )}
      <button onClick={onReset} style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "6px 16px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>
        Run Again
      </button>
    </div>
  );
}

function ConfigDisplay({ config }: { config: Record<string, unknown> }) {
  const regimes = (config.regimes as Record<string, Record<string, unknown>>) ?? {};
  return (
    <div style={{ marginBottom: 16 }}>
      <span style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
        Best Committee Config
      </span>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 8 }}>
        {Object.entries(regimes).map(([regime, a]) => (
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
