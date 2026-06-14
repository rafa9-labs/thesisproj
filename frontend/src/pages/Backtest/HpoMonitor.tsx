import { useMemo } from "react";
import { Trophy } from "lucide-react";
import type { HpoTrialRow } from "@/api/schemas";

interface Props {
  model: string;
  trials: HpoTrialRow[];
  bestTrial: HpoTrialRow | null;
  totalTrials?: number;
}

function ParamPill({ label, value }: { label: string; value: unknown }) {
  const display = typeof value === "number" ? value.toFixed(3) : String(value);
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[10px] text-(--color-brand)"
      style={{ backgroundColor: "rgba(59,130,246,0.08)" }}
    >
      {label}: {display}
    </span>
  );
}

export function HpoMonitor({ model, trials, bestTrial, totalTrials }: Props) {
  const currentTrial = trials.length > 0 ? trials[trials.length - 1] : null;

  const topParams = useMemo(() => {
    if (!currentTrial?.params) return [];
    return Object.entries(currentTrial.params).slice(0, 3);
  }, [currentTrial]);

  return (
    <div className="flex flex-col gap-3">
      {/* Best trial banner */}
      <div
        className="flex items-center gap-2 rounded-sm border px-3 py-2"
        style={{
          backgroundColor: "rgba(234,179,8,0.06)",
          borderColor: "rgba(234,179,8,0.2)",
        }}
      >
        <Trophy size={16} className="text-(--color-accent-warning)" />
        <span className="text-xs tracking-[0.06em] text-(--color-text-muted) uppercase">
          Best Trial
        </span>
        {bestTrial ? (
          <span className="font-mono text-sm font-semibold text-(--color-text-primary)">
            #{bestTrial.trial_number} | Sharpe: {bestTrial.score?.toFixed(4) ?? "-"}
          </span>
        ) : (
          <span className="text-xs text-(--color-text-muted)">Waiting for trials...</span>
        )}
      </div>

      {/* Current trial compact card */}
      {currentTrial ? (
        <div className="flex flex-col gap-2 rounded-sm border border-(--color-border) bg-(--color-surface) px-3 py-2">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs tracking-[0.06em] text-(--color-text-muted) uppercase">
              {model}
            </span>
            <span className="font-mono text-xs font-medium text-(--color-text-primary)">
              Trial {currentTrial.trial_number}
              {totalTrials ? ` / ${totalTrials}` : ""}
            </span>
            <span className="font-mono text-xs text-(--color-brand)">
              Sharpe: {currentTrial.score?.toFixed(4) ?? "-"}
            </span>
            <span
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]"
              style={{
                backgroundColor:
                  currentTrial.trial_state === "COMPLETE"
                    ? "rgba(34,197,94,0.1)"
                    : currentTrial.trial_state === "FAIL"
                      ? "rgba(239,68,68,0.1)"
                      : "rgba(59,130,246,0.1)",
                color:
                  currentTrial.trial_state === "COMPLETE"
                    ? "var(--color-accent-success)"
                    : currentTrial.trial_state === "FAIL"
                      ? "var(--color-accent-danger)"
                      : "var(--color-brand)",
              }}
            >
              {currentTrial.trial_state === "COMPLETE"
                ? "done"
                : currentTrial.trial_state === "FAIL"
                  ? "fail"
                  : "run"}
            </span>
          </div>

          {topParams.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {topParams.map(([k, v]) => (
                <ParamPill key={k} label={k} value={v} />
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-center rounded-sm border border-(--color-border) px-3 py-4 text-(--color-text-muted)">
          <span className="text-xs">HPO trials will appear here...</span>
        </div>
      )}
    </div>
  );
}
