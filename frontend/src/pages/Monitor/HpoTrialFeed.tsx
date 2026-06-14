import { useState } from "react";
import type { HpoTrialRow } from "@/api/schemas";

interface Props {
  trials: { model: string; trial: HpoTrialRow }[];
}

const IMPORTANT_PARAMS = [
  "lags",
  "confidence_threshold",
  "learning_rate",
  "max_depth",
  "n_estimators",
  "units",
  "num_layers",
  "dropout_rate",
  "filters",
  "d_model",
  "num_heads",
  "C",
  "gamma",
  "subsample",
  "colsample_bytree",
  "reg_alpha",
  "reg_lambda",
];

function parseState(raw: string): { short: string; color: string; bg: string } {
  const lower = raw.toLowerCase().trim();
  if (lower === "complete")
    return { short: "COMPLETE", color: "var(--color-accent-success)", bg: "rgba(8,153,129,0.1)" };
  if (lower.startsWith("fail"))
    return { short: "FAILED", color: "var(--color-accent-danger)", bg: "rgba(242,54,69,0.1)" };
  if (lower.startsWith("pruned")) {
    const parts = lower.replace("pruned:", "").split(":");
    const reason = parts[0]?.toUpperCase().replace(/_/g, " ") ?? "PRUNED";
    return {
      short: `PRUNED: ${reason}`,
      color: "var(--color-accent-warning)",
      bg: "rgba(245,158,11,0.1)",
    };
  }
  return {
    short: raw.toUpperCase(),
    color: "var(--color-text-muted)",
    bg: "rgba(120,123,134,0.08)",
  };
}

function StateBadge({ raw }: { raw: string | undefined }) {
  const [hover, setHover] = useState(false);
  const state = raw ?? "complete";
  const { short, color, bg } = parseState(state);
  const showTooltip = hover && state.length > short.length;

  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <span
        className="inline-block max-w-[100px] cursor-default truncate rounded px-1.5 py-0.5 text-[8px] font-medium whitespace-nowrap uppercase"
        style={{ backgroundColor: bg, color }}
      >
        {short}
      </span>
      {showTooltip && (
        <span className="pointer-events-none absolute right-0 bottom-full z-30 mb-1 max-w-[280px] rounded border border-(--color-glass-border) bg-(--color-surface) px-2 py-1 font-mono text-[9px] whitespace-normal text-(--color-text-secondary) shadow-2xl">
          {state}
        </span>
      )}
    </span>
  );
}

function ParamsCell({ params }: { params: Record<string, unknown> | undefined }) {
  const [hover, setHover] = useState(false);
  if (!params) return <span className="text-(--color-text-muted)">\u2014</span>;

  const entries = Object.entries(params).filter(([k]) => IMPORTANT_PARAMS.includes(k));
  const visible = entries.slice(0, 2);
  const remaining = entries.length - 2;
  const fullStr = entries
    .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(4) : String(v)}`)
    .join("\n");

  return (
    <span
      className="relative inline-flex items-center gap-1"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {visible.map(([k, v]) => (
        <span
          key={k}
          className="inline-block rounded bg-(--color-glass-hover) px-1.5 py-0.5 text-[8px] font-medium whitespace-nowrap"
          style={{ color: "var(--color-brand)" }}
        >
          {k}: {typeof v === "number" ? v.toFixed(4) : String(v)}
        </span>
      ))}
      {remaining > 0 && (
        <span className="inline-block rounded bg-(--color-glass-hover) px-1.5 py-0.5 text-[8px] font-medium whitespace-nowrap text-(--color-text-dim)">
          +{remaining} more
        </span>
      )}
      {hover && entries.length > 0 && (
        <span className="pointer-events-none absolute bottom-full left-0 z-30 mb-1 max-w-[320px] rounded border border-(--color-glass-border) bg-(--color-surface) px-2 py-1 font-mono text-[9px] whitespace-pre text-(--color-text-secondary) shadow-2xl">
          {fullStr}
        </span>
      )}
    </span>
  );
}

export function HpoTrialFeed({ trials }: Props) {
  if (trials.length === 0) {
    return (
      <div className="flex items-center justify-center py-4 text-[10px] text-(--color-text-muted)">
        No trials for this model
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-(--color-glass-border)">
            <th className="px-2 py-1 text-left text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
              #
            </th>
            <th className="px-2 py-1 text-right text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
              Score
            </th>
            <th className="px-2 py-1 text-left text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
              Parameters
            </th>
            <th className="px-2 py-1 text-center text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
              State
            </th>
          </tr>
        </thead>
        <tbody>
          {trials.map(({ model, trial }) => {
            const scoreColor =
              (trial.score ?? 0) >= 0.5
                ? "var(--color-accent-success)"
                : (trial.score ?? 0) >= 0
                  ? "var(--color-accent-warning)"
                  : "var(--color-accent-danger)";
            return (
              <tr
                key={`${model}-${trial.trial_number}`}
                className="border-b border-[rgba(42,46,57,0.3)] text-[10px] transition hover:bg-(--color-glass-hover)"
              >
                <td className="px-2 py-1 font-mono whitespace-nowrap text-(--color-text-dim) tabular-nums">
                  {trial.trial_number}
                </td>
                <td className="px-2 py-1 text-right font-mono font-semibold whitespace-nowrap tabular-nums">
                  <span style={{ color: scoreColor }}>{trial.score?.toFixed(4) ?? "\u2014"}</span>
                </td>
                <td className="px-2 py-1">
                  <ParamsCell params={trial.params as Record<string, unknown> | undefined} />
                </td>
                <td className="px-2 py-1 text-center">
                  <StateBadge raw={trial.trial_state} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
