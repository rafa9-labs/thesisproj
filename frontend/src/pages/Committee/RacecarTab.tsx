import { useState } from "react";
import {
  useStartRacecar,
  useRacecarStatus,
  useRacecarResults,
} from "@/api/queries";

const PHASE_LABELS: Record<string, string> = {
  profiling: "Profiling models across market regimes",
  building: "Building optimal committee configuration",
  backtesting: "Validating committee with walk-forward backtest",
  completed: "Auto-optimize complete",
  failed: "Auto-optimize failed",
};

const DEFAULT_MODELS = [
  "logistic", "random_forest", "xgboost", "lightgbm", "decision_tree",
];

export function RacecarTab() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>(DEFAULT_MODELS);
  const [profileTrials, setProfileTrials] = useState(5);
  const [topK, setTopK] = useState(3);

  const startMutation = useStartRacecar();
  const { data: status, isFetching: statusPolling } = useRacecarStatus(jobId);
  const { data: results } = useRacecarResults(
    status?.phase === "completed" || status?.phase === "failed" ? jobId : null,
  );

  const isRunning = status && status.phase !== "completed" && status.phase !== "failed";

  function handleStart() {
    startMutation.mutate(
      {
        models,
        profile_trials: profileTrials,
        committee_top_k: topK,
      },
      { onSuccess: (data) => setJobId(data.job_id) },
    );
  }

  function handleReset() {
    setJobId(null);
  }

  function toggleModel(model: string) {
    setModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model],
    );
  }

  const backtest = results?.backtest as Record<string, unknown> | undefined;
  const config = results?.committee_config as Record<string, unknown> | undefined;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Control Panel */}
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
              Racecar Auto-Optimize
            </h2>
            <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "6px 0 0" }}>
              Profiles all models across market regimes, builds the optimal committee, and
              validates it — all in one click.
            </p>
          </div>
          <button
            onClick={isRunning ? undefined : handleStart}
            disabled={isRunning || startMutation.isPending || models.length === 0}
            style={{
              background: isRunning
                ? "var(--color-elevated)"
                : "var(--color-brand)",
              color: isRunning ? "var(--color-text-muted)" : "var(--color-text-inverse)",
              border: "none",
              borderRadius: 4,
              padding: "10px 24px",
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              cursor: isRunning ? "default" : "pointer",
              opacity: models.length === 0 ? 0.4 : 1,
            }}
          >
            {isRunning ? "Running..." : startMutation.isPending ? "Starting..." : "Run Racecar"}
          </button>
        </div>

        {/* Model selection */}
        {!isRunning && (
          <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 8 }}>
            <span
              style={{
                fontSize: 10,
                fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: "var(--color-text-muted)",
              }}
            >
              Models to Profile
            </span>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {[
                "logistic", "svm", "random_forest", "decision_tree",
                "xgboost", "lightgbm", "catboost",
                "cnn", "lstm", "transformer", "gru", "gru_lstm",
                "meta_ensemble", "stacking_ensemble", "ensemble_adaptive_regime",
              ].map((model) => (
                <button
                  key={model}
                  onClick={() => toggleModel(model)}
                  style={{
                    background: models.includes(model)
                      ? "var(--color-brand-glow)"
                      : "var(--color-elevated)",
                    border: `1px solid ${models.includes(model) ? "var(--color-brand)" : "var(--color-glass-border)"}`,
                    borderRadius: 4,
                    padding: "4px 10px",
                    fontSize: 10,
                    fontFamily: "var(--font-mono)",
                    color: models.includes(model)
                      ? "var(--color-brand)"
                      : "var(--color-text-muted)",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  {model}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Trials per model:{" "}
                <input
                  type="number"
                  value={profileTrials}
                  onChange={(e) => setProfileTrials(Number(e.target.value))}
                  min={2}
                  max={20}
                  style={{
                    width: 48,
                    background: "var(--color-input-bg)",
                    border: "1px solid var(--color-glass-border)",
                    borderRadius: 4,
                    color: "var(--color-text-primary)",
                    padding: "2px 6px",
                    fontSize: 10,
                    fontFamily: "var(--font-mono)",
                  }}
                />
              </label>
              <label style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                Top-k per regime:{" "}
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  min={1}
                  max={5}
                  style={{
                    width: 40,
                    background: "var(--color-input-bg)",
                    border: "1px solid var(--color-glass-border)",
                    borderRadius: 4,
                    color: "var(--color-text-primary)",
                    padding: "2px 6px",
                    fontSize: 10,
                    fontFamily: "var(--font-mono)",
                  }}
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Progress */}
      {(isRunning || status?.phase === "completed" || status?.phase === "failed") && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 24,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 16,
            }}
          >
            <span
              style={{
                fontSize: 13,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
              }}
            >
              {status.phase === "completed"
                ? "Results"
                : status.phase === "failed"
                  ? "Failed"
                  : "Pipeline Progress"}
            </span>
            <span
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--color-text-muted)",
              }}
            >
              Job: {status.job_id}
            </span>
          </div>

          {/* Phase progress bar */}
          {isRunning && (
            <PhaseProgress currentPhase={status.phase} />
          )}

          {/* Status line */}
          {(isRunning || statusPolling) && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: status.phase === "failed" ? "var(--color-accent-danger)" : "var(--color-brand)",
                  animation: status.phase !== "failed" ? "pulse 1.5s infinite" : "none",
                }}
              />
              <span
                style={{ fontSize: 11, color: "var(--color-text-secondary)" }}
              >
                {PHASE_LABELS[status.phase] ?? status.phase}
              </span>
              {status.phase_progress && (
                <span
                  style={{ fontSize: 10, color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
                >
                  ({status.phase_progress})
                </span>
              )}
            </div>
          )}

          {/* Error */}
          {status.phase === "failed" && status.error && (
            <div
              style={{
                marginTop: 12,
                padding: 12,
                background: "rgba(242,54,69,0.08)",
                border: "1px solid rgba(242,54,69,0.2)",
                borderRadius: 4,
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--color-accent-danger)",
              }}
            >
              {status.error}
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {results && status?.phase === "completed" && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 24,
          }}
        >
          <h3
            style={{
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--color-text-primary)",
              margin: "0 0 16px",
            }}
          >
            Optimization Results
            <span
              style={{
                fontSize: 10,
                color: "var(--color-text-muted)",
                marginLeft: 8,
                fontWeight: 400,
              }}
            >
              Completed in {results.total_time_s.toFixed(0)}s
            </span>
          </h3>

          {/* Summary cards */}
          {backtest && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                gap: 10,
                marginBottom: 20,
              }}
            >
              {backtest.avg_sharpe !== undefined && (
                <MetricBlock
                  label="Avg Sharpe"
                  value={Number(backtest.avg_sharpe).toFixed(3)}
                  color={Number(backtest.avg_sharpe) >= 0 ? "#089981" : "#F23645"}
                />
              )}
              {backtest.avg_trades !== undefined && (
                <MetricBlock
                  label="Avg Trades"
                  value={String(backtest.avg_trades)}
                  color="var(--color-text-secondary)"
                />
              )}
              {backtest.folds !== undefined && (
                <MetricBlock
                  label="Folds"
                  value={String(backtest.folds)}
                  color="var(--color-text-secondary)"
                />
              )}
              {backtest.models && (
                <MetricBlock
                  label="Models"
                  value={String((backtest.models as string[]).length)}
                  color="var(--color-text-secondary)"
                />
              )}
            </div>
          )}

          {/* Committee Config Preview */}
          {config && (
            <div style={{ marginTop: 8 }}>
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 500,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: "var(--color-text-muted)",
                }}
              >
                Committee Configuration
              </span>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 8 }}>
                {Object.entries(config.regimes as Record<string, Record<string, unknown>> ?? {}).map(
                  ([regime, assignment]) => (
                    <div
                      key={regime}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 12,
                        fontSize: 11,
                      }}
                    >
                      <span
                        style={{
                          width: 120,
                          fontWeight: 500,
                          letterSpacing: "0.06em",
                          textTransform: "uppercase",
                          color: "var(--color-brand)",
                          flexShrink: 0,
                        }}
                      >
                        {regime.replace(/_/g, " ")}
                      </span>
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                        {(assignment.models as string[])?.map((model, idx) => (
                          <span
                            key={`${model}-${idx}`}
                            style={{
                              background: "var(--color-elevated)",
                              borderRadius: 3,
                              padding: "3px 8px",
                              fontSize: 10,
                              fontFamily: "var(--font-mono)",
                              color: "var(--color-text-secondary)",
                            }}
                          >
                            {model}{" "}
                            {((Number((assignment.weights as number[])?.[idx] ?? 0) * 100)).toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          )}

          {/* Reset */}
          <button
            onClick={handleReset}
            style={{
              marginTop: 20,
              background: "var(--color-elevated)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 4,
              color: "var(--color-text-secondary)",
              padding: "6px 16px",
              fontSize: 10,
              fontWeight: 500,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              cursor: "pointer",
            }}
          >
            Run Again
          </button>
        </div>
      )}
    </div>
  );
}

function PhaseProgress({ currentPhase }: { currentPhase: string }) {
  const phases = ["profiling", "building", "backtesting", "completed"];
  const currentIdx = phases.indexOf(currentPhase);

  return (
    <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
      {phases.slice(0, 3).map((phase, idx) => {
        const isActive = phase === currentPhase;
        const isDone = idx < (currentIdx >= 3 ? 3 : currentIdx);
        return (
          <div key={phase} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 4 }}>
            <div
              style={{
                height: 4,
                borderRadius: 2,
                background: isDone
                  ? "var(--color-accent-success)"
                  : isActive
                    ? "var(--color-brand)"
                    : "var(--color-elevated)",
                transition: "background 0.3s ease",
              }}
            />
            <span
              style={{
                fontSize: 9,
                fontWeight: 500,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: isDone || isActive ? "var(--color-text-primary)" : "var(--color-text-dim)",
              }}
            >
              {phase === "profiling" ? "Profile" : phase === "building" ? "Build" : "Backtest"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function MetricBlock({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div
      style={{
        background: "var(--color-elevated)",
        border: "1px solid var(--color-glass-border)",
        borderRadius: 4,
        padding: "12px 16px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 600,
          fontFamily: "var(--font-mono)",
          color,
          marginBottom: 4,
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: 9,
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
