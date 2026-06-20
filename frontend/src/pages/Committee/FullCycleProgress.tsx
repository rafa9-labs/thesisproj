import { useRef, useEffect, useState } from "react";
import { useFullCycleStatus, useFullCycleLogs } from "@/api/queries";
import type { LogEntry } from "@/api/schemas";

const FC_PHASES = [
  "feature_sweep",
  "phase1_hpo",
  "phase2_assembly",
  "phase3_validation",
  "phase4_factory",
] as const;

const PHASE_LABEL: Record<string, string> = {
  feature_sweep: "Phase 1: Feature Sweep",
  phase1_hpo: "Phase 2: Hyperparameter Tuning",
  phase2_assembly: "Phase 3: Committee Assembly",
  phase3_validation: "Phase 4: Walk-Forward Validation",
  phase4_factory: "Phase 5: Iterative Optimization",
};

const PHASE_DESC: Record<string, string> = {
  feature_sweep:
    "Grid-expand indicators, BorutaSHAP shadow-feature validation with Purged K-Fold CV",
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
  const isRunning =
    status &&
    !["completed", "failed", "validation_failed", "cancelled", "orphaned"].includes(status.phase);

  const [showLogs, setShowLogs] = useState(true);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [logNextIndex, setLogNextIndex] = useState(0);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const { data: logsData } = useFullCycleLogs(isRunning ? jobId : null, logNextIndex);

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
      <div className="p-[28px] text-center text-[11px] text-(--color-text-muted)">
        Loading status...
      </div>
    );
  }

  const phaseNum = status.phase_number ?? 0;
  const phaseLabel =
    status.phase === "completed"
      ? "Full Cycle Complete"
      : status.phase === "validation_failed"
        ? "Validation Failed"
        : status.phase === "failed"
          ? "Pipeline Failed"
          : status.phase === "cancelled"
            ? "Pipeline Cancelled"
            : "Pipeline Progress";

  return (
    <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[28px]">
      {/* Header */}
      <div className="mb-[20px] flex items-center justify-between">
        <span className="text-[14px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          {phaseLabel}
        </span>
        <div className="flex items-center gap-[8px]">
          {isRunning && (
            <>
              <div
                className="h-[8px] w-[8px] rounded-full bg-(--color-brand)"
                style={{ animation: "pulse 1.5s infinite" }}
              />
              <span className="font-mono text-[10px] text-(--color-text-muted)">Running</span>
              <button
                onClick={onCancel}
                className="ml-[8px] cursor-pointer rounded-[3px] border border-[rgba(242,54,69,0.25)] bg-[rgba(242,54,69,0.12)] px-[10px] py-[3px] text-[9px] font-semibold tracking-[0.06em] text-(--color-accent-danger) uppercase"
              >
                Cancel
              </button>
            </>
          )}
          {status.phase === "completed" && (
            <span className="text-[11px] font-medium tracking-[0.06em] text-(--color-accent-success)">
              Complete
            </span>
          )}
        </div>
      </div>

      {/* 6-Phase progress bar */}
      <div className="mb-[16px] flex gap-[6px]">
        {FC_PHASES.map((phase, idx) => {
          const isActive = idx + 1 === phaseNum;
          const isDone = idx + 1 < phaseNum || status.phase === "completed";
          return (
            <div key={phase} className="flex flex-1 flex-col gap-[6px]">
              <div
                className="h-[5px] rounded-[2.5px]"
                style={{
                  background: isDone
                    ? "var(--color-accent-success)"
                    : isActive
                      ? "var(--color-brand)"
                      : "var(--color-elevated)",
                  transition: "background 0.4s ease",
                }}
              />
              <div className="flex flex-col gap-[2px]">
                <span
                  className="text-[11px] font-semibold tracking-[0.06em] uppercase"
                  style={{
                    color:
                      isDone || isActive ? "var(--color-text-primary)" : "var(--color-text-dim)",
                  }}
                >
                  {PHASE_LABEL[phase]}
                </span>
                <span
                  className="text-[10px] leading-[1.3]"
                  style={{
                    color:
                      isDone || isActive ? "var(--color-text-secondary)" : "var(--color-text-dim)",
                  }}
                >
                  {PHASE_DESC[phase]}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Intra-phase sub-progress bar */}
      {isRunning &&
        (() => {
          const parts = (status.phase_progress || "").split("/");
          const cur = parseInt(parts[0]);
          const tot = parseInt(parts[1]);
          if (!tot || isNaN(cur) || isNaN(tot)) return null;
          const pct = Math.round((cur / tot) * 100);
          const label =
            status.phase === "feature_sweep"
              ? `Fold ${cur}/${tot}`
              : status.phase === "phase1_hpo"
                ? `HPO ${cur}/${tot}`
                : status.phase === "phase4_factory"
                  ? `Iteration ${cur}/${tot}`
                  : `${cur}/${tot}`;
          return (
            <div className="mb-[12px]">
              <div className="mb-[4px] flex justify-between">
                <span className="font-mono text-[10px] font-medium tracking-[0.04em] text-(--color-text-muted) uppercase">
                  {label}
                </span>
                <span className="font-mono text-[10px] text-(--color-text-dim)">{pct}%</span>
              </div>
              <div className="h-[4px] overflow-hidden rounded-[2px] bg-(--color-elevated)">
                <div
                  className="h-full rounded-[2px] bg-(--color-brand-glow)"
                  style={{
                    width: `${pct}%`,
                    transition: "width 0.5s ease",
                    minWidth: pct > 0 ? 8 : 0,
                  }}
                />
              </div>
            </div>
          );
        })()}

      {/* Phase 1 locked features */}
      {status.locked_features_count !== undefined && status.locked_features_count > 0 && (
        <div className="mb-[12px] rounded-[4px] border border-[rgba(0,229,255,0.15)] bg-[rgba(0,229,255,0.06)] p-[10px] text-[10px]">
          <span className="font-medium tracking-[0.06em] text-(--color-brand) uppercase">
            Phase 1: {status.locked_features_count} features locked
          </span>
        </div>
      )}

      {/* Current action */}
      {isRunning && status.current_action && (
        <div className="mb-[16px] rounded-[4px] border border-(--color-glass-border) bg-(--color-elevated) p-[14px]">
          <div className="flex items-center gap-[8px]">
            {status.phase === "optimizing" && (
              <span className="rounded-[3px] bg-(--color-brand-glow) px-[8px] py-[2px] font-mono text-[10px] font-semibold tracking-[0.06em] text-(--color-brand)">
                Iteration {status.iteration}/{status.total_iterations}
              </span>
            )}
            <span className="text-[10px] text-(--color-text-secondary)">
              {status.current_action}
            </span>
            {status.best_sharpe_so_far !== undefined && status.best_sharpe_so_far > 0 && (
              <span className="ml-auto font-mono text-[10px] text-(--color-text-muted)">
                Best: {status.best_sharpe_so_far.toFixed(4)}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Error / failed */}
      {status.phase === "failed" && (
        <div className="rounded-[4px] border border-[rgba(242,54,69,0.2)] bg-[rgba(242,54,69,0.08)] p-[12px] font-mono text-[11px] text-(--color-accent-danger)">
          {status.error || "Unknown error"}
        </div>
      )}

      {/* Validation failed */}
      {status.phase === "validation_failed" && (
        <div className="mb-[12px] rounded-[4px] border border-[rgba(242,180,54,0.25)] bg-[rgba(242,180,54,0.08)] p-[14px] text-[11px]">
          <span className="font-semibold tracking-[0.06em] text-[#F2B436] uppercase">
            Phase 5 Validation Failed
          </span>
          <div className="mt-[8px] font-mono leading-[1.6] text-(--color-text-secondary)">
            {status.error || "One or more gates failed. Check the results for diagnostics."}
          </div>
          <div className="mt-[8px] text-[10px] text-(--color-text-dim)">
            Suggested: increase train_months, try different model types, or re-run with fewer
            models.
          </div>
        </div>
      )}

      {/* Live Log Viewer */}
      {isRunning && (
        <div className="mt-[8px] rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[16px]">
          <div
            onClick={() => setShowLogs((v) => !v)}
            className="flex cursor-pointer items-center justify-between select-none"
          >
            <span className="text-[11px] font-semibold tracking-[0.06em] text-(--color-text-primary) uppercase">
              Live Log ({logEntries.length} lines)
            </span>
            <span className="text-[10px] text-(--color-text-muted)">
              {showLogs ? "[-]" : "[+]"}
            </span>
          </div>
          {showLogs && (
            <div className="mt-[10px] max-h-[320px] overflow-y-auto rounded-[4px] border border-[rgba(255,255,255,0.06)] bg-[#0a0e14] p-[12px] font-mono text-[10px] leading-[1.6]">
              {logEntries.length === 0 && (
                <span className="text-(--color-text-dim)">Waiting for logs...</span>
              )}
              {logEntries.map((e, i) => {
                const prevPhase = i > 0 ? logEntries[i - 1].phase_number : 0;
                const phaseChanged = e.phase_number && e.phase_number !== prevPhase;
                return (
                  <div key={e.index}>
                    {phaseChanged && (
                      <div
                        style={{
                          color: "var(--color-stepper-active)",
                          borderTop: `1px solid rgba(0,229,255,0.13)`,
                        }}
                        className="mt-[4px] pt-[6px] pb-[2px] text-[11px] font-bold tracking-[0.04em]"
                      >
                        {e.message}
                      </div>
                    )}
                    {!phaseChanged && (
                      <div
                        className="whitespace-pre-wrap"
                        style={{
                          color:
                            e.level === "error"
                              ? "var(--color-accent-danger)"
                              : e.level === "warn"
                                ? "#F2B436"
                                : "#c9d1d9",
                          wordBreak: "break-all",
                        }}
                      >
                        <span className="text-[#484f58]">[{e.timestamp}]</span>{" "}
                        {e.phase_number && (
                          <span
                            className="mr-[4px] inline-block rounded-[2px] px-[5px] text-[9px] font-semibold text-[#fff]"
                            style={{
                              background: "var(--color-stepper-active)",
                            }}
                          >
                            P{e.phase_number}
                          </span>
                        )}
                        <span className="text-[#8b949e]">{e.message}</span>
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
