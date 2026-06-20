import { Check, X, Loader2, SkipForward } from "lucide-react";
import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { PhaseStatus } from "@/stores/useCommitteeMonitorStore";
import { cn } from "@/lib/utils";

const PHASE_DEFS: { label: string; shortLabel: string }[] = [
  { label: "Feature Sweep", shortLabel: "Sweep" },
  { label: "HPO Tuning", shortLabel: "HPO" },
  { label: "Committee Assembly", shortLabel: "Assembly" },
  { label: "Validation", shortLabel: "Validate" },
  { label: "Factory Optimization", shortLabel: "Factory" },
  { label: "Snapshot", shortLabel: "Deploy" },
];

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function getStepStatus(
  stepIndex: number,
  phaseNumber: number,
  hasError: boolean,
  hasCache: boolean,
  phase: string,
): PhaseStatus {
  const step = stepIndex + 1;
  const isTerminal =
    phase === "completed" || phase === "failed" || phase === "validation_failed";
  const isCancelled = phase === "cancelled";

  if (isCancelled) {
    if (step <= phaseNumber) return "complete";
    return "skipped";
  }

  if (hasError && step === phaseNumber) return "failed";

  if (isTerminal) {
    if (phase === "completed") return "complete";
    if (step <= phaseNumber) return "complete";
    return "failed";
  }

  if (step < phaseNumber) return "complete";
  if (step === phaseNumber) return "active";
  return "pending";
}

export function PipelineNavigator() {
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const viewPhase = useCommitteeMonitorStore((s) => s.viewPhase);
  const phase = useCommitteeMonitorStore((s) => s.phase);
  const error = useCommitteeMonitorStore((s) => s.error);
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const selectedJobId = useCommitteeMonitorStore((s) => s.selectedJobId);
  const phaseTimings = useCommitteeMonitorStore((s) => s.phaseTimings);
  const setViewPhase = useCommitteeMonitorStore((s) => s.setViewPhase);

  if (!selectedJobId) return null;

  return (
    <nav className="flex w-full items-start px-2 py-1 sm:px-4" role="tablist">
      {PHASE_DEFS.map((def, i) => {
        const step = i + 1;
        const hasCache = phaseCache[step as keyof typeof phaseCache] !== null;
        const status = getStepStatus(i, phaseNumber, !!error, hasCache, phase);
        const isSelected = viewPhase === step;
        const isLast = i === PHASE_DEFS.length - 1;
        const canClick = status === "complete" || status === "active" || (status === "failed" && hasCache);
        const elapsed = phaseTimings[`phase${step}`] ?? 0;

        const circleBg =
          status === "active"
            ? "var(--color-brand)"
            : status === "complete"
              ? "rgba(16,185,129,0.12)"
              : status === "failed"
                ? "rgba(244,63,94,0.12)"
                : "transparent";

        const circleBorder =
          status === "active"
            ? "var(--color-brand)"
            : status === "complete"
              ? "var(--color-accent-success)"
              : status === "failed"
                ? "var(--color-accent-danger)"
                : "var(--color-text-dim)";

        const circleColor =
          status === "active"
            ? "var(--color-text-inverse)"
            : status === "complete"
              ? "var(--color-accent-success)"
              : status === "failed"
                ? "var(--color-accent-danger)"
                : "var(--color-text-muted)";

        return (
          <div
            key={step}
            className={cn("flex flex-1 items-start")}
            style={{ minWidth: 0 }}
          >
            <button
              onClick={() => canClick && setViewPhase(step)}
              disabled={!canClick}
              role="tab"
              aria-selected={isSelected}
              className="flex flex-col items-center gap-1.5 transition-opacity duration-150"
              style={{
                cursor: canClick ? "pointer" : "default",
                opacity: canClick ? 1 : 0.4,
                background: "transparent",
                border: "none",
                outline: "none",
                minWidth: 0,
              }}
            >
              <span
                className="flex shrink-0 items-center justify-center rounded-full font-mono text-[10px] font-bold transition-all duration-150"
                style={{
                  width: 28,
                  height: 28,
                  backgroundColor: circleBg,
                  border: `1.5px solid ${circleBorder}`,
                  color: circleColor,
                  boxShadow:
                    status === "active"
                      ? "0 0 12px rgba(0,229,255,0.35)"
                      : isSelected
                        ? "0 0 8px rgba(0,229,255,0.15)"
                        : "none",
                  outline: isSelected ? "1px solid rgba(0,229,255,0.25)" : "none",
                  outlineOffset: 2,
                }}
              >
                {status === "complete" && <Check size={13} strokeWidth={2.5} />}
                {status === "failed" && <X size={13} strokeWidth={2.5} />}
                {status === "active" && (
                  <Loader2
                    size={13}
                    strokeWidth={2.5}
                    className="animate-spin"
                  />
                )}
                {status === "pending" && step}
                {status === "skipped" && (
                  <SkipForward size={13} strokeWidth={2} />
                )}
              </span>
              <span
                className="hidden truncate text-center text-[9px] font-semibold tracking-[0.02em] sm:block sm:max-w-[72px]"
                style={{
                  color:
                    status === "active"
                      ? "var(--color-brand)"
                      : isSelected
                        ? "var(--color-text-primary)"
                        : status === "complete"
                          ? "var(--color-text-secondary)"
                          : "var(--color-text-muted)",
                }}
              >
                {def.label}
              </span>
              {/* Phase timing */}
              {elapsed > 0 && (
                <span className="font-mono text-[7px] text-(--color-text-dim) tracking-tight">
                  {formatElapsed(elapsed)}
                </span>
              )}
            </button>

            {!isLast && (
              <div
                className="mt-[14px] h-px flex-1"
                style={{
                  minWidth: 12,
                  backgroundColor:
                    status === "complete"
                      ? "var(--color-accent-success)"
                      : "var(--color-text-dim)",
                  opacity: status === "complete" ? 0.5 : 0.25,
                }}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
