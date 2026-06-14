import { CheckCircle2, Loader2, Zap, BarChart3, Trophy } from "lucide-react";
import type { CycleState } from "@/stores/useJobStore";
import { HpoTrialTable } from "./HpoTrialTable";

interface Props {
  cycle: CycleState;
}

function PhaseBadge({ phase }: { phase: string }) {
  const config: Record<string, { icon: React.ReactNode; label: string; color: string }> = {
    pending: {
      icon: <Loader2 size={10} className="animate-spin" />,
      label: "Pending",
      color: "var(--color-text-muted)",
    },
    hpo: {
      icon: <Zap size={10} />,
      label: "HPO",
      color: "var(--color-accent)",
    },
    simulation: {
      icon: <BarChart3 size={10} />,
      label: "Testing",
      color: "var(--color-accent-success)",
    },
    complete: {
      icon: <CheckCircle2 size={10} />,
      label: "Done",
      color: "var(--color-accent-success)",
    },
  };
  const c = config[phase] ?? config.pending;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full bg-(--color-glass-hover) px-2 py-0.5 text-[9px] font-medium tracking-[0.06em] uppercase"
      style={{ color: c.color }}
    >
      {c.icon}
      {c.label}
    </span>
  );
}

export function CycleCard({ cycle }: Props) {
  const isHpo = cycle.phase === "hpo";
  const isSim = cycle.phase === "simulation";
  const isDone = cycle.phase === "complete";

  return (
    <div
      className="flex flex-col rounded-sm border bg-(--color-surface)"
      style={{
        borderColor: isDone
          ? "var(--color-accent-success)"
          : isHpo
            ? "var(--color-accent)"
            : isSim
              ? "var(--color-accent-success)"
              : "var(--color-border)",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-(--color-border-subtle) bg-(--color-elevated) px-3 py-2">
        <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
          Cycle {cycle.cycleNumber}
        </span>
        <span className="font-mono text-[11px] font-medium text-(--color-text-primary)">
          {cycle.model}
        </span>
        <div className="flex-1" />
        <PhaseBadge phase={cycle.phase} />
      </div>

      {/* Body */}
      <div style={{ maxHeight: 320, overflowY: "auto" }}>
        {cycle.phase === "pending" && (
          <div className="flex items-center justify-center py-6">
            <span className="animate-pulse text-xs text-(--color-text-muted)">Waiting...</span>
          </div>
        )}

        {cycle.phase !== "pending" && (
          <div className="flex flex-col gap-3 p-3">
            {/* Best Trial banner — always visible once we have trials */}
            {cycle.bestTrial && (
              <div
                className="flex items-center gap-2 rounded-sm border px-3 py-2"
                style={{
                  borderColor: "rgba(234,179,8,0.2)",
                  backgroundColor: "rgba(234,179,8,0.06)",
                }}
              >
                <Trophy size={14} className="text-(--color-accent-warning)" />
                <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
                  Best Trial
                </span>
                <span className="font-mono text-[11px] font-semibold text-(--color-text-primary)">
                  #{cycle.bestTrial.trial_number}: {cycle.bestTrial.score?.toFixed(4) ?? "-"}
                </span>
              </div>
            )}

            {/* HPO trial table — persists after completion */}
            {cycle.hpoTrials.length > 0 && (
              <HpoTrialTable
                model={cycle.model}
                trials={cycle.hpoTrials}
                bestTrial={cycle.bestTrial}
                totalTrials={cycle.hpoTrialTotal}
              />
            )}

            {/* Per-month test results — persists after completion */}
            {cycle.testMonths.length > 0 && (
              <div className="flex flex-col gap-1">
                {cycle.testMonths.map((tm) => (
                  <div
                    key={tm.period}
                    className="flex items-center gap-2 rounded border border-(--color-border-subtle) bg-(--color-glass-hover) px-2 py-1"
                  >
                    <span className="font-mono text-[10px] font-medium text-(--color-text-secondary)">
                      M{tm.period}
                      {tm.flat ? " (flat)" : ""}
                    </span>
                    <span className="font-mono text-[10px] text-(--color-text-primary)">
                      Sharpe: {tm.sharpe?.toFixed(2) ?? "-"}
                    </span>
                    <span className="font-mono text-[10px] text-(--color-text-primary)">
                      Ret: {tm.return_pct != null ? `${tm.return_pct.toFixed(2)}%` : "-"}
                    </span>
                    <span className="font-mono text-[10px] text-(--color-text-primary)">
                      DD: {tm.drawdown?.toFixed(2) ?? "-"}
                    </span>
                    <span className="font-mono text-[10px] text-(--color-text-primary)">
                      Tr: {tm.trades ?? "-"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
