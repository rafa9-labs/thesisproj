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
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium uppercase tracking-[0.06em]"
      style={{ backgroundColor: "var(--color-glass-hover)", color: c.color }}
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
      className="flex flex-col rounded-lg border"
      style={{
        borderColor: isDone
          ? "var(--color-accent-success)"
          : isHpo
            ? "var(--color-accent)"
            : isSim
              ? "var(--color-accent-success)"
              : "var(--color-border)",
        backgroundColor: "var(--color-surface)",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-3 py-2 border-b"
        style={{ borderColor: "var(--color-border-subtle)", backgroundColor: "var(--color-elevated)" }}
      >
        <span className="text-[10px] font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          Cycle {cycle.cycleNumber}
        </span>
        <span className="text-[11px] font-medium" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
          {cycle.model}
        </span>
        <div className="flex-1" />
        <PhaseBadge phase={cycle.phase} />
      </div>

      {/* Body */}
      <div style={{ maxHeight: 320, overflowY: "auto" }}>
        {cycle.phase === "pending" && (
          <div className="flex items-center justify-center py-6">
            <span className="text-xs animate-pulse" style={{ color: "var(--color-text-muted)" }}>
              Waiting...
            </span>
          </div>
        )}

        {cycle.phase !== "pending" && (
          <div className="p-3 flex flex-col gap-3">
            {/* Best Trial banner — always visible once we have trials */}
            {cycle.bestTrial && (
              <div
                className="flex items-center gap-2 rounded-lg border px-3 py-2"
                style={{
                  borderColor: "rgba(234,179,8,0.2)",
                  backgroundColor: "rgba(234,179,8,0.06)",
                }}
              >
                <Trophy size={14} style={{ color: "var(--color-accent-warning)" }} />
                <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
                  Best Trial
                </span>
                <span className="text-[11px] font-semibold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
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
                    className="flex items-center gap-2 rounded border px-2 py-1"
                    style={{ borderColor: "var(--color-border-subtle)", backgroundColor: "var(--color-glass-hover)" }}
                  >
                    <span className="text-[10px] font-medium" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                      M{tm.period}
                      {tm.flat ? " (flat)" : ""}
                    </span>
                    <span className="text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                      Sharpe: {tm.sharpe?.toFixed(2) ?? "-"}
                    </span>
                    <span className="text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                      Ret: {tm.return_pct != null ? `${tm.return_pct.toFixed(2)}%` : "-"}
                    </span>
                    <span className="text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                      DD: {tm.drawdown?.toFixed(2) ?? "-"}
                    </span>
                    <span className="text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
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
