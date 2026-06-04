import { useState } from "react";
import {
  useStartFullCycle,
  useFullCycleStatus,
  useFullCycleResults,
  useFullCycleHistory,
} from "@/api/queries";
import apiClient from "@/api/client";
import type { FullCycleRequest, FullCycleHistoryEntry } from "@/api/schemas";

const CORE_MODELS = [
  "logistic", "svm", "random_forest", "xgboost",
  "lightgbm", "catboost", "lstm", "ensemble_adaptive_regime",
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

const FC_PHASES = ["feature_sweep", "phase0_profiling", "phase1_hpo", "phase2_assembly", "phase3_validation", "phase4_factory"] as const;
const PHASE_LABEL: Record<string, string> = {
  feature_sweep: "Phase -1: Sweep",
  phase0_profiling: "Phase 0: Pre-screen",
  phase1_hpo: "Phase 1: HPO",
  phase2_assembly: "Phase 2: Assemble",
  phase3_validation: "Phase 3: Validate",
  phase4_factory: "Phase 4: Factory",
};
const PHASE_DESC: Record<string, string> = {
  feature_sweep: "Grid-expand indicators, shallow RF + permutation importance -> lock features",
  phase0_profiling: "Quick-screening all models -> prune to surviving set with diversity enforcement",
  phase1_hpo: "Targeted hyperparameter optimization per survivor with TPE sampler",
  phase2_assembly: "Building committee config from best HPO models per regime",
  phase3_validation: "36-month WFO, fold consistency, regime coverage, 3-seed robustness",
  phase4_factory: "Iterative LLM-guided optimization with proxy WFO -> final 10-year WFO",
};

export function FullCycleTab() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>(CORE_MODELS);
  const [proposer, setProposer] = useState("llm");
  const [llmBackend, setLlmBackend] = useState("deepseek");
  const [maxIter, setMaxIter] = useState(20);
  const [patience, setPatience] = useState(5);
  const [tolerance, setTolerance] = useState(0.02);
  const [trainMonths, setTrainMonths] = useState(36);
  const [testMonths, setTestMonths] = useState(1);
  const [hpoSampler, setHpoSampler] = useState("tpe");
  const [cvBlocks, setCvBlocks] = useState(3);
  const [profileTrialsPhase0, setProfileTrialsPhase0] = useState(5);
  const [maxSurvivingModels, setMaxSurvivingModels] = useState(7);
  const [sweepNEstimators, setSweepNEstimators] = useState(100);
  const [sweepMaxDepth, setSweepMaxDepth] = useState(5);

  const startMutation = useStartFullCycle();
  const { data: status } = useFullCycleStatus(jobId);
  const { data: results } = useFullCycleResults(
    status?.phase === "completed" || status?.phase === "failed"
      || status?.phase === "validation_failed" ? jobId : null,
  );

  const terminalPhases = new Set(["completed", "failed", "validation_failed"]);
  const isRunning = status && !terminalPhases.has(status.phase);

  const [showHistory, setShowHistory] = useState(true);
  const { data: history } = useFullCycleHistory();

  function handleDownload() {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `full_cycle_${jobId ?? "results"}.json`;
    a.click(); URL.revokeObjectURL(url);
  }

  function handleDeploy() {
    apiClient.post("/api/v1/live/deploy-committee", {
      pair: "EURUSD", timeframe: "H1", initial_equity: 10000.0, confidence_threshold: 0.55,
    }).then((r: { data: { session_id: string } }) => {
      window.location.href = `/trading?session=${r.data.session_id}`;
    });
  }
  function handleStart() {
    const req: FullCycleRequest = {
      models,
      proposer,
      llm_backend: llmBackend,
      max_iterations: maxIter,
      patience,
      stopping_tolerance: tolerance,
      train_months: trainMonths,
      test_months: testMonths,
      hpo_sampler: hpoSampler,
      cv_blocks: cvBlocks,
      profile_trials_phase0: profileTrialsPhase0,
      max_surviving_models: maxSurvivingModels,
      sweep_n_estimators: sweepNEstimators,
      sweep_max_depth: sweepMaxDepth,
      committee_top_k: 3,
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

            {/* Settings row 1: Pipeline controls */}
            <div style={{ marginTop: 16, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Train Months:{" "}
                <input type="number" value={trainMonths} onChange={(e) => setTrainMonths(Number(e.target.value))} min={12} max={120} style={{ width: 42, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Test Months:{" "}
                <input type="number" value={testMonths} onChange={(e) => setTestMonths(Number(e.target.value))} min={1} max={12} style={{ width: 36, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                HPO Sampler:{" "}
                <select value={hpoSampler} onChange={(e) => setHpoSampler(e.target.value)} style={{ background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "3px 6px", fontSize: 10 }}>
                  <option value="tpe">TPE</option>
                  <option value="random">Random</option>
                  <option value="grid">Grid</option>
                </select>
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                CV Blocks:{" "}
                <input type="number" value={cvBlocks} onChange={(e) => setCvBlocks(Number(e.target.value))} min={2} max={10} style={{ width: 36, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Phase 0 Trials:{" "}
                <input type="number" value={profileTrialsPhase0} onChange={(e) => setProfileTrialsPhase0(Number(e.target.value))} min={2} max={50} style={{ width: 36, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Max Survivors:{" "}
                <input type="number" value={maxSurvivingModels} onChange={(e) => setMaxSurvivingModels(Number(e.target.value))} min={2} max={15} style={{ width: 36, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
            </div>

            {/* Settings row 2: Factory / Proposer */}
            <div style={{ marginTop: 12, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Proposer:{" "}
                <select value={proposer} onChange={(e) => setProposer(e.target.value)} style={{ background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "3px 6px", fontSize: 10 }}>
                  <option value="llm">LLM (DeepSeek V4)</option>
                  <option value="ollama">LLM (Ollama)</option>
                  <option value="deterministic">Deterministic Greedy</option>
                </select>
              </label>
              {proposer !== "deterministic" && (
                <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                  Backend:{" "}
                  <select value={llmBackend} onChange={(e) => setLlmBackend(e.target.value)} style={{ background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "3px 6px", fontSize: 10 }}>
                    <option value="deepseek">DeepSeek</option>
                    <option value="ollama">Ollama</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                  </select>
                </label>
              )}
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Max Iter:{" "}
                <input type="number" value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} min={3} max={50} style={{ width: 42, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Patience:{" "}
                <input type="number" value={patience} onChange={(e) => setPatience(Number(e.target.value))} min={3} max={15} style={{ width: 40, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Tolerance:{" "}
                <input type="number" value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} min={0.005} max={0.1} step={0.005} style={{ width: 52, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
            </div>

            {/* Settings row 3: Phase -1 Feature Sweep */}
            <div style={{ marginTop: 12, display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", padding: "8px 12px", background: "rgba(0,229,255,0.04)", border: "1px solid rgba(0,229,255,0.12)", borderRadius: 4 }}>
              <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-brand)" }}>Phase -1:</span>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                RF Trees:{" "}
                <input type="number" value={sweepNEstimators} onChange={(e) => setSweepNEstimators(Number(e.target.value))} min={30} max={300} step={10} style={{ width: 42, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Max Depth:{" "}
                <input type="number" value={sweepMaxDepth} onChange={(e) => setSweepMaxDepth(Number(e.target.value))} min={2} max={10} style={{ width: 32, background: "var(--color-input-bg)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-primary)", padding: "2px 6px", fontSize: 10, fontFamily: "var(--font-mono)" }} />
              </label>
              <span style={{ fontSize: 9, color: "var(--color-text-dim)" }}>(shallow RF {`->`} permutation importance {`->`} lock features)</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Run History Panel ── */}
      <div style={{ background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: history?.entries?.length ? 14 : 0 }}>
          <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-primary)" }}>
            Run History{history?.total_runs ? ` (${history.total_runs})` : ""}
          </span>
          <button onClick={() => setShowHistory(!showHistory)} style={{ background: "none", border: "none", color: "var(--color-text-muted)", cursor: "pointer", fontSize: 10, fontFamily: "var(--font-mono)" }}>
            {showHistory ? "Hide" : "Show"}
          </button>
        </div>
        {showHistory && (
          <>
            {history && history.entries.length === 0 && (
              <div style={{ fontSize: 10, color: "var(--color-text-dim)", textAlign: "center", padding: 16 }}>No history yet. Run your first Full Cycle to see results here.</div>
            )}
            {history && history.entries.length > 0 && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                <thead>
                  <tr>
                    <th style={histThStyle}>Started</th>
                    <th style={histThStyle}>Status</th>
                    <th style={histThStyle}>Time</th>
                    <th style={histThStyle}>Locked</th>
                    <th style={histThStyle}>Survivors</th>
                    <th style={{ ...histThStyle, textAlign: "right" }}>Sharpe</th>
                    <th style={histThStyle}></th>
                  </tr>
                </thead>
                <tbody>
                  {history.entries.map((entry: FullCycleHistoryEntry) => {
                    const isActive = entry.job_id === jobId;
                    const isTerminal = terminalPhases.has(entry.status);
                    const isInProgress = !isTerminal && entry.status !== "unknown";
                    const started = entry.started_at ? new Date(entry.started_at).toLocaleString(undefined, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }) : "--";
                    const timeStr = entry.total_time_s > 0 ? (entry.total_time_s >= 3600 ? `${Math.floor(entry.total_time_s / 3600)}h ${Math.floor((entry.total_time_s % 3600) / 60)}m` : `${Math.floor(entry.total_time_s / 60)}m ${Math.floor(entry.total_time_s % 60)}s`) : "--";
                    const sharpeStr = entry.avg_sharpe !== 0 ? entry.avg_sharpe.toFixed(2) : "--";
                    const sharpeColor = entry.avg_sharpe > 0 ? "#089981" : entry.avg_sharpe < 0 ? "#F23645" : "var(--color-text-dim)";
                    const statusColor = isInProgress ? "var(--color-brand)" : entry.status === "completed" ? "#089981" : entry.status === "validation_failed" ? "#F2B436" : "#F23645";
                    const survivorsStr = entry.survivors_count > 0 ? entry.survivors.slice(0, 3).join(",") + (entry.survivors_count > 3 ? ` +${entry.survivors_count - 3}` : "") : "--";
                    return (
                      <tr key={entry.job_id} style={{ borderBottom: "1px solid var(--color-glass-border)", background: isActive ? "rgba(0,229,255,0.04)" : "transparent" }}>
                        <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", opacity: isActive ? 1 : 0.7 }}>{started}</td>
                        <td style={histTdStyle}>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", display: "inline-block", background: statusColor, animation: isInProgress ? "pulse 1.5s infinite" : "none" }} />
                            <span style={{ color: "var(--color-text-secondary)" }}>{isInProgress ? "Running" : entry.status}</span>
                          </span>
                        </td>
                        <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>{timeStr}</td>
                        <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>{entry.locked_features_count > 0 ? entry.locked_features_count : "--"}</td>
                        <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)", maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis" }}>{survivorsStr}</td>
                        <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", textAlign: "right", color: sharpeColor }}>{sharpeStr}</td>
                        <td style={histTdStyle}>
                          <button onClick={() => setJobId(entry.job_id)} disabled={isActive} style={{ background: "none", border: "none", cursor: isActive ? "default" : "pointer", color: isActive ? "var(--color-text-dim)" : "var(--color-brand)", fontSize: 9, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
                            {isActive ? "Active" : "Load"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
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
              {status.phase === "completed" ? "Full Cycle Complete" : status.phase === "validation_failed" ? "Validation Failed" : status.phase === "failed" ? "Pipeline Failed" : "Pipeline Progress"}
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

          {/* 6-Phase progress bar */}
          <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
            {FC_PHASES.map((phase, idx) => {
              const isActive = phase === status.phase;
              const phaseNum = status.phase_number ?? 0;
              const isDone = idx < phaseNum || status.phase === "completed";
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

          {/* Phase -1 locked features */}
          {status.locked_features_count !== undefined && status.locked_features_count > 0 && (
            <div style={{ padding: 10, background: "rgba(0,229,255,0.06)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, marginBottom: 12, fontSize: 10 }}>
              <span style={{ fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-brand)" }}>
                Phase -1: {status.locked_features_count} features locked
              </span>
            </div>
          )}

          {/* Phase 0 survivors */}
          {status.pruned_models?.length > 0 && (
            <div style={{ padding: 10, background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, marginBottom: 12, fontSize: 10 }}>
              <span style={{ fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)" }}>
                Phase 0: Pruned {status.pruned_models.length} models
              </span>
              {status.surviving_models?.length > 0 && (
                <div style={{ marginTop: 6, display: "flex", gap: 4, flexWrap: "wrap" }}>
                  <span style={{ color: "var(--color-text-dim)" }}>Survivors:</span>
                  {status.surviving_models.map((m: string) => (
                    <span key={m} style={{ background: "rgba(8,153,129,0.12)", border: "1px solid rgba(8,153,129,0.25)", borderRadius: 3, padding: "2px 7px", color: "#089981", fontFamily: "var(--font-mono)" }}>{m}</span>
                  ))}
                </div>
              )}
            </div>
          )}

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

          {status?.phase === "validation_failed" && (
            <div style={{ padding: 14, background: "rgba(242,180,54,0.08)", border: "1px solid rgba(242,180,54,0.25)", borderRadius: 4, marginBottom: 12, fontSize: 11 }}>
              <span style={{ fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "#F2B436" }}>Phase 3 Validation Failed</span>
              <div style={{ marginTop: 8, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", lineHeight: 1.6 }}>
                {status.error || "One or more gates failed. Check the results for diagnostics."}
              </div>
              <div style={{ marginTop: 8, fontSize: 10, color: "var(--color-text-dim)" }}>
                Suggested: increase train_months, reduce max_surviving_models, or run with more Phase 0 trials.
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Results Panel ── */}
      {results && (status?.phase === "completed" || status?.phase === "validation_failed") && (
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
            {status?.phase === "validation_failed" && (
              <span style={{ fontSize: 11, color: "#F2B436", marginLeft: 10, fontWeight: 500 }}>VALIDATION FAILED</span>
            )}
            <span style={{ fontSize: 11, color: "var(--color-text-muted)", marginLeft: 10, fontWeight: 400, fontFamily: "var(--font-mono)" }}>
              {Number(results.total_time_s).toFixed(0)}s
            </span>
          </h3>

          {/* Phase -1 Feature Sweep */}
          {results.locked_features_count !== undefined && results.locked_features_count > 0 && (
            <>
              <SectionHeader label="Phase -1: Feature Sweep" />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
                <MetricCard label="Features Locked" value={String(results.locked_features_count)} color="#089981" />
                <MetricCard label="Features Pruned" value={String(results.pruned_features_count ?? 0)} color="#F23645" />
                <MetricCard label="Top Feature" value={results.top_importance_feature || "N/A"} color="var(--color-text-secondary)" />
              </div>
            </>
          )}

          {/* Phase 0 survivors */}
          {results.phase0_survivors?.length > 0 && (
            <>
              <SectionHeader label="Phase 0: Pre-screened Survivors" />
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 20 }}>
                {results.phase0_survivors.map((m: string) => (
                  <span key={m} style={{ background: "rgba(8,153,129,0.12)", border: "1px solid rgba(8,153,129,0.25)", borderRadius: 3, padding: "3px 9px", fontSize: 10, fontFamily: "var(--font-mono)", color: "#089981" }}>{m}</span>
                ))}
              </div>
              {results.phase0_pruned?.length > 0 && (
                <div style={{ marginBottom: 16, fontSize: 9, color: "var(--color-text-dim)" }}>
                  Pruned: {results.phase0_pruned.join(", ")}
                </div>
              )}
            </>
          )}

          {/* Phase 3 validation */}
          <SectionHeader label={`Phase 3: Validation${!results.phase3_fold_consistency_pass ? " (FAILED)" : ""}`} />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
            <MetricCard label="Fold CV" value={Number(results.phase3_fold_consistency_cv ?? 0).toFixed(3)} color={results.phase3_fold_consistency_pass ? "#089981" : "#F23645"} />
            <MetricCard label="Regime Coverage" value={results.phase3_regime_coverage ? "PASS" : "FAIL"} color={results.phase3_regime_coverage ? "#089981" : "#F23645"} />
            <MetricCard label="Seed Robustness" value={results.phase3_seed_robustness_pass ? "PASS" : "FAIL"} color={results.phase3_seed_robustness_pass ? "#089981" : "#F23645"} />
            <MetricCard label="Seed Sharpe" value={Number(results.phase3_seed_robustness_sharpe ?? 0).toFixed(3)} color={Number(results.phase3_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"} />
          </div>

          {status?.phase === "validation_failed" && (
            <div style={{ marginBottom: 20, padding: 14, background: "rgba(242,180,54,0.06)", border: "1px solid rgba(242,180,54,0.2)", borderRadius: 4, fontSize: 10, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
              <span style={{ fontWeight: 600, color: "#F2B436" }}>Validation halted pipeline.</span> Phase 4 Factory was skipped. Review the diagnostics above and adjust parameters. Common fixes: increase train_months, reduce max_surviving_models, or add more Phase 0 trials.
            </div>
          )}

          {/* Team Backtest */}
          {results.racecar_backtest && (
            <SectionHeader label="Team Backtest" />
          )}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 24 }}>
            <MetricCard label="Avg Sharpe" value={Number((results.racecar_backtest as Record<string, unknown> | undefined)?.avg_sharpe ?? 0).toFixed(3)} color={Number((results.racecar_backtest as Record<string, unknown> | undefined)?.avg_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"} />
            <MetricCard label="Avg Trades" value={String((results.racecar_backtest as Record<string, unknown> | undefined)?.avg_trades ?? 0)} color="var(--color-text-secondary)" />
            <MetricCard label="Folds" value={String((results.racecar_backtest as Record<string, unknown> | undefined)?.total_folds ?? 0)} color="var(--color-text-secondary)" />
            <MetricCard label="Models in Config" value={String((results.racecar_backtest as Record<string, unknown> | undefined)?.models?.length ?? 0)} color="var(--color-text-secondary)" />
          </div>

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

          <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
            <button onClick={handleReset} style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "8px 20px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>Run Again</button>
            <button onClick={handleDownload} style={{ background: "var(--color-elevated)", border: "1px solid var(--color-glass-border)", borderRadius: 4, color: "var(--color-text-secondary)", padding: "8px 20px", fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>Download JSON</button>
            {results.factory_best_config && status?.phase !== "validation_failed" && (
              <button onClick={handleDeploy} style={{ background: "var(--color-accent-success)", border: "none", borderRadius: 4, color: "var(--color-text-inverse)", padding: "8px 20px", fontSize: 10, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", cursor: "pointer" }}>Deploy to Trading</button>
            )}
          </div>
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

const histThStyle: React.CSSProperties = {
  padding: "4px 8px", textAlign: "left", color: "var(--color-text-muted)",
  fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase",
  fontSize: 9, borderBottom: "1px solid var(--color-glass-border)",
};

const histTdStyle: React.CSSProperties = {
  padding: "5px 8px", color: "var(--color-text-secondary)", fontSize: 10,
};
