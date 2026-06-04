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
  "lags", "lags_range", "confidence_threshold", "target_active_rate",
  "calibrate_method", "gating_mode", "label_threshold", "roll_windows_key",
  "alpha_vol_z", "beta_spread_norm", "gamma_slip_norm",
  "max_depth", "learning_rate", "n_estimators", "subsample", "C", "gamma",
  "units", "num_layers", "dropout_rate", "filters", "kernel_size",
  "d_model", "num_heads",
]);

function parseTrialState(state: string): { label: string; color: string; reason: string } {
  if (state.startsWith("PRUNED:")) {
    return { label: "prune", color: "var(--color-accent-warning)", reason: state.slice(7) };
  }
  if (state.startsWith("FAIL:")) {
    return { label: "fail", color: "var(--color-accent-danger)", reason: state.slice(5) };
  }
  if (state === "PRUNED") return { label: "prune", color: "var(--color-accent-warning)", reason: "No reason given" };
  if (state === "FAIL") return { label: "fail", color: "var(--color-accent-danger)", reason: "No reason given" };
  if (state === "COMPLETE") return { label: "done", color: "var(--color-accent-success)", reason: "" };
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
        <span
          className="text-[11px] font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          HPO Trials
        </span>
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {trials.length}{totalTrials > 0 ? `/${totalTrials}` : ""}
        </span>
        {bestTrial && (
          <span className="text-[9px] ml-1" style={{ color: "var(--color-accent-warning)", fontFamily: "var(--font-mono)" }}>
            best: #{bestTrial.trial_number} — {bestTrial.score?.toFixed(4) ?? "-"}
          </span>
        )}
      </div>

      {visible.length === 0 && (
        <div
          className="flex items-center justify-center rounded-sm border px-3 py-4"
          style={{ borderColor: "var(--color-border)", color: "var(--color-text-muted)" }}
        >
          <span className="text-xs">Waiting for trials...</span>
        </div>
      )}

      {visible.length > 0 && (
        <div
          ref={scrollRef}
          className="overflow-y-auto rounded-sm border"
          style={{ maxHeight: 340, borderColor: "var(--color-border)" }}
        >
          <table className="w-full border-collapse">
            <thead className="sticky top-0" style={{ backgroundColor: "var(--color-surface)" }}>
              <tr>
                <th
                  className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-[0.06em]"
                  style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)", width: 28 }}
                >
                  #
                </th>
                <th
                  className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-[0.06em]"
                  style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)", width: 60 }}
                >
                  Score
                </th>
                <th
                  className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-[0.06em]"
                  style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)" }}
                >
                  Params
                </th>
                <th
                  className="px-2 py-1.5 text-center text-[10px] font-medium uppercase tracking-[0.06em]"
                  style={{ color: "var(--color-text-muted)", borderBottom: "1px solid var(--color-border)", width: 70 }}
                >
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
                    style={{
                      backgroundColor: best ? "rgba(234,179,8,0.04)" : "transparent",
                      borderBottom: "1px solid var(--color-border-subtle)",
                      cursor: (all.length > 0 || st.reason) ? "pointer" : "default",
                    }}
                  >
                    <td
                      className="px-2 py-1 text-[10px] align-top"
                      style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}
                    >
                      {t.trial_number}
                      {isExpanded ? <ChevronUp size={8} style={{ marginLeft: 2 }} /> : <ChevronDown size={8} style={{ marginLeft: 2 }} />}
                    </td>
                    <td
                      className="px-2 py-1 text-[10px] align-top"
                      style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}
                    >
                      {t.score != null ? t.score.toFixed(4) : "-"}
                    </td>
                    <td className="px-2 py-1 align-top">
                      <div className="flex gap-1 flex-wrap">
                        {top.length > 0 ? (
                          top.map(([k, v]) => (
                            <span
                              key={k}
                              className="inline-flex items-center px-1 rounded text-[9px]"
                              style={{ backgroundColor: "rgba(59,130,246,0.08)", color: "var(--color-brand)", fontFamily: "var(--font-mono)" }}
                            >
                              {k}: {typeof v === "number" ? v.toFixed(3) : String(v)}
                            </span>
                          ))
                        ) : (
                          <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>—</span>
                        )}
                      </div>

                      {/* Expanded detail row (inline) */}
                      {isExpanded && (
                        <div className="mt-1.5 flex flex-col gap-1">
                          {st.reason && (
                            <div className="rounded px-1.5 py-1 text-[9px] leading-tight" style={{ backgroundColor: "rgba(242,54,69,0.05)", color: "var(--color-accent-danger)" }}>
                              <span className="font-medium uppercase tracking-[0.06em]">Reason: </span>
                              {st.reason}
                            </div>
                          )}
                          {all.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {all.map(([k, v]) => (
                                <span
                                  key={k}
                                  className="inline-flex items-center px-1 rounded text-[8px]"
                                  style={{ backgroundColor: "var(--color-elevated)", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
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
                          style={{ backgroundColor: st.reason ? "rgba(242,54,69,0.08)" : undefined, color: st.color }}
                        >
                          {st.label}
                        </span>
                        {st.reason && (
                          <span
                            className="text-[7px] leading-tight max-w-[64px] text-center"
                            style={{ color: "var(--color-text-muted)", wordBreak: "break-word" }}
                          >
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
