import { useCallback } from "react";
import { Square, Clock, AlertTriangle } from "lucide-react";
import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";

const PHASE_LABEL_MAP: Record<number, string> = {
  1: "Feature Sweep",
  2: "HPO Tuning",
  3: "Committee Assembly",
  4: "Validation",
  5: "Factory Optimization",
};

function formatElapsed(startedAt: string): string {
  if (!startedAt) return "";
  const start = new Date(startedAt).getTime();
  const now = Date.now();
  const diff = Math.max(0, (now - start) / 1000);
  const mins = Math.floor(diff / 60);
  const secs = Math.floor(diff % 60);
  if (mins >= 60) {
    const hrs = Math.floor(mins / 60);
    const rem = mins % 60;
    return `${hrs}h ${rem}m`;
  }
  return `${mins}m ${secs}s`;
}

export function CommitteeJobHeader({ onForceStop }: { onForceStop?: () => void }) {
  const selectedJobId = useCommitteeMonitorStore((s) => s.selectedJobId);
  const phase = useCommitteeMonitorStore((s) => s.phase);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const phaseProgress = useCommitteeMonitorStore((s) => s.phaseProgress);
  const iteration = useCommitteeMonitorStore((s) => s.iteration);
  const totalIterations = useCommitteeMonitorStore((s) => s.totalIterations);
  const bestSharpeSoFar = useCommitteeMonitorStore((s) => s.bestSharpeSoFar);
  const error = useCommitteeMonitorStore((s) => s.error);
  const startedAt = useCommitteeMonitorStore((s) => s.startedAt);

  const isRunning = !["completed", "failed", "validation_failed", "cancelled", "orphaned", ""].includes(phase);
  const isFailed = phase === "failed" || phase === "validation_failed";

  const phaseLabel = PHASE_LABEL_MAP[phaseNumber] || (phase ? phase.replace(/_/g, " ") : "Initializing");

  const handleStop = useCallback(() => {
    if (onForceStop) onForceStop();
  }, [onForceStop]);

  if (!selectedJobId) return null;

  return (
    <div className="shrink-0 border-b border-(--color-glass-border) px-3 py-2 sm:px-6">
      {/* Top row: job identity + status + force stop */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {/* Job pill */}
          <span
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-semibold tracking-[0.04em] uppercase"
            style={{
              backgroundColor: isRunning
                ? "rgba(0,229,255,0.12)"
                : isFailed
                  ? "rgba(244,63,94,0.12)"
                  : "rgba(16,185,129,0.12)",
              color: isRunning ? "var(--color-brand)" : isFailed ? "var(--color-accent-danger)" : "var(--color-accent-success)",
              border: `1px solid ${
                isRunning ? "rgba(0,229,255,0.3)" : isFailed ? "rgba(244,63,94,0.3)" : "rgba(16,185,129,0.3)"
              }`,
            }}
          >
            <span
              className="inline-block h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: "currentColor" }}
            />
            {isRunning ? "RUNNING" : isFailed ? "FAILED" : phase === "completed" ? "COMPLETED" : phase.toUpperCase()}
          </span>

          {/* Job ID label */}
          <span className="truncate text-[11px] font-medium tracking-[0.02em] text-(--color-text-primary) font-mono">
            Committee {selectedJobId.slice(-8)}
          </span>

          {/* Phase indicator */}
          <span className="hidden text-[10px] tracking-[0.04em] text-(--color-text-muted) uppercase sm:inline">
            {phaseLabel}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Elapsed time */}
          {startedAt && (
            <span className="flex items-center gap-1 text-[10px] font-mono text-(--color-text-dim)">
              <Clock size={11} />
              <span>{formatElapsed(startedAt)}</span>
            </span>
          )}

          {/* Force Stop */}
          {isRunning && (
            <button
              onClick={handleStop}
              className="flex items-center gap-1 rounded-sm border border-[rgba(244,63,94,0.25)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.04em] transition-colors duration-150"
              style={{
                backgroundColor: "rgba(244,63,94,0.06)",
                color: "var(--color-accent-danger)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "rgba(244,63,94,0.12)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "rgba(244,63,94,0.06)";
              }}
            >
              <Square size={9} fill="currentColor" />
              FORCE STOP
            </button>
          )}
        </div>
      </div>

      {/* Second row: progress + metrics */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1">
        {/* Phase progress bar */}
        {phaseProgress && (
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-(--color-text-muted)">
            <span className="text-(--color-text-dim) uppercase tracking-[0.04em]">Progress</span>
            <span>{phaseProgress}</span>
          </div>
        )}

        {/* Best Sharpe */}
        {bestSharpeSoFar > 0 && (
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            <span className="text-(--color-text-dim) uppercase tracking-[0.04em]">Best Sharpe</span>
            <span style={{ color: bestSharpeSoFar >= 0.5 ? "var(--color-accent-success)" : "var(--color-accent-warning)" }}>
              {bestSharpeSoFar.toFixed(4)}
            </span>
          </div>
        )}

        {/* Iteration counter (Phase 5) */}
        {iteration > 0 && (
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            <span className="text-(--color-text-dim) uppercase tracking-[0.04em]">Iteration</span>
            <span className="text-(--color-text-secondary)">
              {iteration}/{totalIterations || "?"}
            </span>
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="mt-2 flex items-start gap-1.5 rounded-[4px] border border-[rgba(244,63,94,0.2)] bg-[rgba(244,63,94,0.06)] px-2 py-1.5 text-[10px] text-(--color-accent-danger)">
          <AlertTriangle size={11} className="mt-px shrink-0" />
          <span className="line-clamp-2">{error}</span>
        </div>
      )}
    </div>
  );
}
