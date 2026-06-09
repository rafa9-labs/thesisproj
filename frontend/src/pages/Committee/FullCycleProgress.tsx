import { useRef, useEffect, useState } from "react";
import {
  useFullCycleStatus,
  useFullCycleLogs,
} from "@/api/queries";
import type { LogEntry } from "@/api/schemas";

const FC_PHASES = ["feature_sweep", "phase1_hpo", "phase2_assembly", "phase3_validation", "phase4_factory"] as const;

const PHASE_LABEL: Record<string, string> = {
  feature_sweep: "Phase 1: Feature Sweep",
  phase1_hpo: "Phase 2: Hyperparameter Tuning",
  phase2_assembly: "Phase 3: Committee Assembly",
  phase3_validation: "Phase 4: Walk-Forward Validation",
  phase4_factory: "Phase 5: Iterative Optimization",
};

const PHASE_COLORS: Record<number, string> = {
  1: "#089981",
  2: "#2962FF",
  3: "#E5A014",
  4: "#A78BFA",
  5: "#F23645",
};

const PHASE_DESC: Record<string, string> = {
  feature_sweep: "Grid-expand indicators, BorutaSHAP shadow-feature validation with Purged K-Fold CV",
  phase1_hpo: "Targeted hyperparameter optimization per model with TPE sampler",
  phase2_assembly: "Build committee config from best HPO models per market regime",
  phase3_validation: "36-month walk-forward, fold consistency, regime coverage, 3-seed robustness",
  phase4_factory: "Iterative LLM-guided optimization with proxy WFO, final 10-year WFO",
};

interface Props {
  jobId: string;
  onCancel: () => void;
}

export function FullCycleProgress({ jobId, onCancel }: Props) {
  const { data: status } = useFullCycleStatus(jobId);
  const isRunning = status && !["completed", "failed", "validation_failed", "cancelled", "orphaned"].includes(status.phase);

  const [showLogs, setShowLogs] = useState(true);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [logNextIndex, setLogNextIndex] = useState(0);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const { data: logsData } = useFullCycleLogs(
    isRunning ? jobId : null,
    logNextIndex,
  );

  useEffect(() => {
    if (!logsData?.entries?.length) return;
    setLogEntries((prev) => [...prev, ...logsData.entries]);
    setLogNextIndex(logsData.next_index);
  }, [logsData]);

  useEffect(() => {
    if (!showLogs) return;
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logEntries, showLogs]);

  const prevJobId = useRef<string | null>(null);
  useEffect(() => {
    if (jobId && jobId !== prevJobId.current) {
      setLogEntries([]);
      setLogNextIndex(0);
      prevJobId.current = jobId;
    }
  }, [jobId]);

  if (!status) {
    return (
      <div style={{ padding: 28, textAlign: "center", color: "var(--color-text-muted)", fontSize: 11 }}>
        Loading status...
      </div>
    );
  }

  const phaseNum = status.phase_number ?? 0;
  const phaseLabel = status.phase === "completed" ? "Full Cycle Complete"
    : status.phase === "validation_failed" ? "Validation Failed"
    : status.phase === "failed" ? "Pipeline Failed"
    : status.phase === "cancelled" ? "Pipeline Cancelled"
    : "Pipeline Progress";

  return (
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
          {phaseLabel}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {isRunning && (
            <>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-brand)", animation: "pulse 1.5s infinite" }} />
              <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                Running
              </span>
              <button
                onClick={onCancel}
                style={{
                  marginLeft: 8,
                  background: "rgba(242,54,69,0.12)",
                  color: "var(--color-accent-danger)",
                  border: "1px solid rgba(242,54,69,0.25)",
                  borderRadius: 3,
                  padding: "3px 10px",
                  fontSize: 9,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
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
          const isActive = idx + 1 === phaseNum;
          const isDone = idx + 1 < phaseNum || status.phase === "completed";
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
                  fontSize: 11,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: isDone || isActive ? "var(--color-text-primary)" : "var(--color-text-dim)",
                }}>
                  {PHASE_LABEL[phase]}
                </span>
                <span style={{
                  fontSize: 10,
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

      {/* Intra-phase sub-progress bar */}
      {isRunning && (() => {
        const parts = (status.phase_progress || "").split("/");
        const cur = parseInt(parts[0]);
        const tot = parseInt(parts[1]);
        if (!tot || isNaN(cur) || isNaN(tot)) return null;
        const pct = Math.round((cur / tot) * 100);
        const label =
          status.phase === "feature_sweep" ? `Fold ${cur}/${tot}`
          : status.phase === "phase1_hpo" ? `HPO ${cur}/${tot}`
          : status.phase === "phase4_factory" ? `Iteration ${cur}/${tot}`
          : `${cur}/${tot}`;
        return (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                {label}
              </span>
              <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>{pct}%</span>
            </div>
            <div style={{ height: 4, borderRadius: 2, background: "var(--color-elevated)", overflow: "hidden" }}>
              <div style={{
                height: "100%", borderRadius: 2,
                width: `${pct}%`,
                background: "var(--color-brand-glow)",
                transition: "width 0.5s ease",
                minWidth: pct > 0 ? 8 : 0,
              }} />
            </div>
          </div>
        );
      })()}

      {/* Phase 1 locked features */}
      {status.locked_features_count !== undefined && status.locked_features_count > 0 && (
        <div style={{ padding: 10, background: "rgba(0,229,255,0.06)", border: "1px solid rgba(0,229,255,0.15)", borderRadius: 4, marginBottom: 12, fontSize: 10 }}>
          <span style={{ fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-brand)" }}>
            Phase 1: {status.locked_features_count} features locked
          </span>
        </div>
      )}

      {/* Current action */}
      {isRunning && status.current_action && (
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
            {status.phase === "optimizing" && (
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
            )}
            <span style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>
              {status.current_action}
            </span>
            {status.best_sharpe_so_far !== undefined && status.best_sharpe_so_far > 0 && (
              <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--color-text-muted)", marginLeft: "auto" }}>
                Best: {status.best_sharpe_so_far.toFixed(4)}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Error / failed */}
      {status.phase === "failed" && (
        <div style={{ padding: 12, background: "rgba(242,54,69,0.08)", border: "1px solid rgba(242,54,69,0.2)", borderRadius: 4, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-accent-danger)" }}>
          {status.error || "Unknown error"}
        </div>
      )}

      {/* Validation failed */}
      {status.phase === "validation_failed" && (
        <div style={{ padding: 14, background: "rgba(242,180,54,0.08)", border: "1px solid rgba(242,180,54,0.25)", borderRadius: 4, marginBottom: 12, fontSize: 11 }}>
          <span style={{ fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "#F2B436" }}>Phase 5 Validation Failed</span>
          <div style={{ marginTop: 8, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", lineHeight: 1.6 }}>
            {status.error || "One or more gates failed. Check the results for diagnostics."}
          </div>
          <div style={{ marginTop: 8, fontSize: 10, color: "var(--color-text-dim)" }}>
            Suggested: increase train_months, try different model types, or re-run with fewer models.
          </div>
        </div>
      )}

      {/* Live Log Viewer */}
      {isRunning && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 16,
            marginTop: 8,
          }}
        >
          <div
            onClick={() => setShowLogs((v) => !v)}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: "var(--color-text-primary)",
              }}
            >
              Live Log ({logEntries.length} lines)
            </span>
            <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
              {showLogs ? "[-]" : "[+]"}
            </span>
          </div>
          {showLogs && (
            <div
              style={{
                marginTop: 10,
                background: "#0a0e14",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 4,
                padding: 12,
                maxHeight: 320,
                overflowY: "auto",
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                lineHeight: 1.6,
              }}
            >
              {logEntries.length === 0 && (
                <span style={{ color: "var(--color-text-dim)" }}>Waiting for logs...</span>
              )}
              {logEntries.map((e, i) => {
                const prevPhase = i > 0 ? logEntries[i - 1].phase_number : 0;
                const phaseChanged = e.phase_number && e.phase_number !== prevPhase;
                return (
                  <div key={e.index}>
                    {phaseChanged && (
                      <div style={{
                        color: PHASE_COLORS[e.phase_number] || "var(--color-text-dim)",
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.04em",
                        padding: "6px 0 2px",
                        borderTop: `1px solid ${PHASE_COLORS[e.phase_number]}22`,
                        marginTop: 4,
                      }}>
                        {e.message}
                      </div>
                    )}
                    {!phaseChanged && (
                      <div
                        style={{
                          color:
                            e.level === "error"
                              ? "var(--color-accent-danger)"
                              : e.level === "warn"
                                ? "#F2B436"
                                : "#c9d1d9",
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-all",
                        }}
                      >
                        <span style={{ color: "#484f58" }}>[{e.timestamp}]</span>{" "}
                        {e.phase_number && (
                          <span style={{
                            display: "inline-block",
                            padding: "0 5px",
                            borderRadius: 2,
                            fontSize: 9,
                            fontWeight: 600,
                            marginRight: 4,
                            color: "#fff",
                            background: PHASE_COLORS[e.phase_number] || "var(--color-text-dim)",
                          }}>
                            P{e.phase_number}
                          </span>
                        )}
                        <span style={{ color: "#8b949e" }}>{e.message}</span>
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export { FC_PHASES, PHASE_LABEL, PHASE_DESC };
