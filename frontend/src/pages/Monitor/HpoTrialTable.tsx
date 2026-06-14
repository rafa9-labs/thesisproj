import { useRef, useEffect, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { HpoTrialRow } from "@/api/schemas";

interface Props {
  model: string;
  trials: HpoTrialRow[];
  bestTrial: HpoTrialRow | null;
  totalTrials: number;
}

const IMPORTANT_PARAMS = new Set([
  "lags",
  "lags_range",
  "confidence_threshold",
  "target_active_rate",
  "calibrate_method",
  "gating_mode",
  "label_threshold",
  "roll_windows_key",
  "alpha_vol_z",
  "beta_spread_norm",
  "gamma_slip_norm",
  "max_depth",
  "learning_rate",
  "n_estimators",
  "subsample",
  "C",
  "gamma",
  "units",
  "num_layers",
  "dropout_rate",
  "filters",
  "kernel_size",
  "d_model",
  "num_heads",
]);

function parseTrialState(state: string): { label: string; color: string; reason: string } {
  if (state.startsWith("PRUNED:")) {
    return { label: "prune", color: "var(--color-accent-warning)", reason: state.slice(7) };
  }
  if (state.startsWith("FAIL:")) {
    return { label: "fail", color: "var(--color-accent-danger)", reason: state.slice(5) };
  }
  if (state === "PRUNED")
    return { label: "prune", color: "var(--color-accent-warning)", reason: "No reason given" };
  if (state === "FAIL")
    return { label: "fail", color: "var(--color-accent-danger)", reason: "No reason given" };
  if (state === "COMPLETE")
    return { label: "done", color: "var(--color-accent-success)", reason: "" };
  return { label: state.toLowerCase(), color: "var(--color-text-muted)", reason: "" };
}

function topParams(params: Record<string, unknown>, limit = 3): [string, unknown][] {
  return Object.entries(params)
    .filter(([k]) => IMPORTANT_PARAMS.has(k))
    .slice(0, limit);
}

function allParamsList(params: Record<string, unknown>): [string, unknown][] {
  return Object.entries(params).filter(([k]) => !k.startsWith("_"));
}

function isBest(trial: HpoTrialRow, best: HpoTrialRow | null): boolean {
  return best?.trial_number === trial.trial_number;
}

export function HpoTrialTable({ trials, bestTrial, totalTrials }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const visible = trials.slice(-30);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [trials.length]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
          HPO Trials
        </span>
        <span className="font-mono text-[10px] text-(--color-text-muted)">
          {trials.length}
          {totalTrials > 0 ? `/${totalTrials}` : ""}
        </span>
        {bestTrial && (
          <span className="ml-1 font-mono text-[9px] text-(--color-accent-warning)">
            best: #{bestTrial.trial_number} — {bestTrial.score?.toFixed(4) ?? "-"}
          </span>
        )}
      </div>

      {visible.length === 0 && (
        <div className="flex items-center justify-center rounded-sm border border-(--color-border) px-3 py-4 text-(--color-text-muted)">
          <span className="text-xs">Waiting for trials...</span>
        </div>
      )}

      {visible.length > 0 && (
        <div
          ref={scrollRef}
          className="max-h-[340px] overflow-y-auto rounded-sm border border-(--color-border)"
        >
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-(--color-surface)">
              <tr>
                <th className="w-[28px] border-b border-(--color-border) px-2 py-1.5 text-left text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                  #
                </th>
                <th className="w-[60px] border-b border-(--color-border) px-2 py-1.5 text-left text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                  Score
                </th>
                <th className="border-b border-(--color-border) px-2 py-1.5 text-left text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                  Params
                </th>
                <th className="w-[70px] border-b border-(--color-border) px-2 py-1.5 text-center text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                  State
                </th>
              </tr>
            </thead>
            <tbody>
              {visible.map((t) => {
                const st = parseTrialState(t.trial_state);
                const top = topParams(t.params ?? {});
                const all = allParamsList(t.params ?? {});
                const best = isBest(t, bestTrial);
                const isExpanded = expanded === t.trial_number;

                return (
                  <tr
                    key={t.trial_number}
                    onClick={() => {
                      if (all.length > 0 || st.reason) {
                        setExpanded(isExpanded ? null : t.trial_number);
                      }
                    }}
                    className="border-b border-(--color-border-subtle)"
                    style={{
                      backgroundColor: best ? "rgba(234,179,8,0.04)" : "transparent",
                      cursor: all.length > 0 || st.reason ? "pointer" : "default",
                    }}
                  >
                    <td className="px-2 py-1 align-top font-mono text-[10px] text-(--color-text-secondary)">
                      {t.trial_number}
                      {isExpanded ? (
                        <ChevronUp size={8} className="ml-[2px]" />
                      ) : (
                        <ChevronDown size={8} className="ml-[2px]" />
                      )}
                    </td>
                    <td className="px-2 py-1 align-top font-mono text-[10px] text-(--color-text-primary)">
                      {t.score != null ? t.score.toFixed(4) : "-"}
                    </td>
                    <td className="px-2 py-1 align-top">
                      <div className="flex flex-wrap gap-1">
                        {top.length > 0 ? (
                          top.map(([k, v]) => (
                            <span
                              key={k}
                              className="inline-flex items-center rounded bg-[rgba(59,130,246,0.08)] px-1 font-mono text-[9px] text-(--color-brand)"
                            >
                              {k}: {typeof v === "number" ? v.toFixed(3) : String(v)}
                            </span>
                          ))
                        ) : (
                          <span className="text-[9px] text-(--color-text-muted)">—</span>
                        )}
                      </div>

                      {/* Expanded detail row (inline) */}
                      {isExpanded && (
                        <div className="mt-1.5 flex flex-col gap-1">
                          {st.reason && (
                            <div className="rounded bg-[rgba(242,54,69,0.05)] px-1.5 py-1 text-[9px] leading-tight text-(--color-accent-danger)">
                              <span className="font-medium tracking-[0.06em] uppercase">
                                Reason:{" "}
                              </span>
                              {st.reason}
                            </div>
                          )}
                          {all.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {all.map(([k, v]) => (
                                <span
                                  key={k}
                                  className="inline-flex items-center rounded bg-(--color-elevated) px-1 font-mono text-[8px] text-(--color-text-muted)"
                                >
                                  {k}: {typeof v === "number" ? v.toFixed(4) : String(v)}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </td>
                    <td className="px-2 py-1 text-center align-top">
                      <div className="flex flex-col items-center gap-0.5">
                        <span
                          className="inline-block rounded px-1.5 py-0.5 text-[9px] font-medium uppercase"
                          style={{
                            backgroundColor: st.reason ? "rgba(242,54,69,0.08)" : undefined,
                            color: st.color,
                          }}
                        >
                          {st.label}
                        </span>
                        {st.reason && (
                          <span className="max-w-[64px] text-center text-[7px] leading-tight break-words text-(--color-text-muted)">
                            {st.reason.length > 40 ? st.reason.slice(0, 38) + ".." : st.reason}
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
