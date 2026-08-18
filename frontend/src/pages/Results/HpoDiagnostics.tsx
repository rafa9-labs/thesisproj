import { HorizontalBarChart } from "@/components/charts/SimpleCharts";
import type {
  HpoParamImportance,
  HpoTrial,
  BestStudy,
  HpoStudyMeta,
  HpoLearningSummary,
  HpoSensitivityEntry,
} from "@/api/schemas";

interface HpoDiagnosticsProps {
  paramImportance: HpoParamImportance[] | null;
  trials: HpoTrial[] | null;
}

export function OptimizationTrace({ trials }: { trials: HpoTrial[] }) {
  if (trials.length === 0) return null;

  const maxTrial = trials.length;
  const values = trials.map((t) => t.value);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;

  const chartW = 400;
  const chartH = 220;
  const padL = 50;
  const padR = 15;
  const padT = 15;
  const padB = 25;
  const plotW = chartW - padL - padR;
  const plotH = chartH - padT - padB;

  const pts = trials.map((t, i) => ({
    x: padL + (i / Math.max(maxTrial - 1, 1)) * plotW,
    y: padT + plotH - ((t.value - minVal) / range) * plotH,
  }));

  const { cumBests } = trials.reduce(
    (acc, t) => {
      const nextBest = Math.min(acc.minSoFar, t.value);
      acc.cumBests.push(nextBest);
      acc.minSoFar = nextBest;
      return acc;
    },
    { cumBests: [] as number[], minSoFar: trials[0].value },
  );

  const bestLine = cumBests.map((best, i) => ({
    x: padL + (i / Math.max(maxTrial - 1, 1)) * plotW,
    y: padT + plotH - ((best - minVal) / range) * plotH,
  }));

  const linePath = bestLine
    .map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`))
    .join(" ");

  return (
    <svg width="100%" height={chartH} viewBox={`0 0 ${chartW} ${chartH}`}>
      {Array.from({ length: 5 }, (_, i) => {
        const y = padT + (i / 4) * plotH;
        const val = maxVal - (i / 4) * range;
        return (
          <g key={i}>
            <line
              x1={padL}
              y1={y}
              x2={chartW - padR}
              y2={y}
              stroke="var(--color-border)"
              strokeWidth={0.5}
            />
            <text
              x={padL - 5}
              y={y + 3}
              textAnchor="end"
              fill="var(--color-text-muted)"
              fontSize={9}
              fontFamily="var(--font-mono)"
            >
              {val != null ? val.toFixed(2) : "—"}
            </text>
          </g>
        );
      })}
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3} fill="var(--color-text-muted)" opacity={0.5} />
      ))}
      <path d={linePath} fill="none" stroke="var(--color-accent-success)" strokeWidth={1.5} />
      {trials.length <= 30 &&
        trials.map((_, i) => (
          <text
            key={i}
            x={padL + (i / Math.max(maxTrial - 1, 1)) * plotW}
            y={chartH - 6}
            textAnchor="middle"
            fill="var(--color-text-muted)"
            fontSize={8}
            fontFamily="var(--font-mono)"
          >
            {i + 1}
          </text>
        ))}
      <text
        x={padL + plotW / 2}
        y={chartH}
        textAnchor="middle"
        fill="var(--color-text-muted)"
        fontSize={9}
      >
        Trial
      </text>
    </svg>
  );
}

export function HpoDiagnostics({ paramImportance, trials }: HpoDiagnosticsProps) {
  const importance = paramImportance ?? [];
  const trialData = trials ?? [];

  if (importance.length === 0 && trialData.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-8 text-(--color-text-muted)">
        <span className="font-mono text-sm">No HPO diagnostics available</span>
      </div>
    );
  }

  const barData = importance
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 20)
    .map((p) => ({ label: p.param, value: p.importance }));

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        HPO Diagnostics
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3">
          <span className="mb-2 block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
            Param Importance
          </span>
          <HorizontalBarChart data={barData} barColor="var(--color-accent-success)" />
        </div>
        <div className="rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3">
          <span className="mb-2 block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
            Optimization Trace — {trialData.length} trials
          </span>
          {trialData.length > 0 ? (
            <OptimizationTrace trials={trialData} />
          ) : (
            <span className="text-xs text-(--color-text-muted)">No trial data</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function BestStudyCard({ bestStudy }: { bestStudy: BestStudy | null }) {
  if (!bestStudy) return null;

  const params = Object.entries(bestStudy.best_params ?? {});
  if (params.length === 0 && bestStudy.best_value == null) return null;

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        Best Study
      </h3>
      <div className="rounded-sm border border-(--color-accent-success) bg-(--color-surface) p-4">
        <div className="grid grid-cols-2 gap-x-6 gap-y-3">
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Best Trial
            </span>
            <span className="font-mono text-sm text-(--color-text-primary)">
              #{bestStudy.best_trial}
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Objective (penalized CV)
            </span>
            <span className="font-mono text-sm text-(--color-accent-success)">
              {bestStudy.best_value != null ? bestStudy.best_value.toFixed(4) : "—"}
            </span>
          </div>
        </div>
        {params.length > 0 && (
          <div className="mt-3">
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase mb-1.5">
              Best Parameters
            </span>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1">
              {params.slice(0, 12).map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs">
                  <span className="truncate text-(--color-text-muted)">{k}</span>
                  <span className="ml-2 shrink-0 font-mono text-(--color-text-primary) tabular-nums">
                    {typeof v === "number" ? v.toFixed(4) : String(v)}
                  </span>
                </div>
              ))}
              {params.length > 12 && (
                <span className="text-xs text-(--color-text-muted)">
                  +{params.length - 12} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function StudyMetaCard({ meta }: { meta: HpoStudyMeta | null }) {
  if (!meta || meta.n_trials === 0) return null;

  const trialBadges = [
    { label: "Completed", value: meta.n_completed, color: "var(--color-accent-success)" },
    { label: "Pruned", value: meta.n_pruned, color: "var(--color-accent-warning)" },
    { label: "Failed", value: meta.n_failed, color: "var(--color-accent-danger)" },
  ];

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        Study Metadata
      </h3>
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Study
            </span>
            <span className="font-mono text-xs text-(--color-text-primary)">
              {meta.study_name ?? "—"}
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Sampler
            </span>
            <span className="font-mono text-xs text-(--color-text-primary)">
              {meta.sampler_type ?? "—"}
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Direction
            </span>
            <span className="font-mono text-xs text-(--color-text-primary)">
              {meta.direction ?? "—"}
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Total Duration
            </span>
            <span className="font-mono text-xs text-(--color-text-primary)">
              {meta.total_duration_sec != null
                ? `${(meta.total_duration_sec / 60).toFixed(1)}m`
                : "—"}
            </span>
          </div>
        </div>
        <div className="mt-2.5 flex gap-2">
          {trialBadges.map((b) => (
            <div
              key={b.label}
              className="flex items-center gap-1 rounded-sm px-2 py-0.5 border"
              style={{ borderColor: b.color }}
            >
              <span className="text-[10px] font-mono" style={{ color: b.color }}>
                {b.value}
              </span>
              <span className="text-[9px] text-(--color-text-muted)">{b.label}</span>
            </div>
          ))}
          <div className="flex items-center gap-1 rounded-sm px-2 py-0.5 border border-(--color-glass-border)">
            <span className="text-[10px] font-mono text-(--color-text-primary)">
              {meta.n_trials}
            </span>
            <span className="text-[9px] text-(--color-text-muted)">Total</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function LearningSummaryCard({ summary }: { summary: HpoLearningSummary | null }) {
  if (!summary) return null;
  if (
    summary.cliff_delta == null &&
    summary.best_uplift_pct == null &&
    summary.startup_trials === 0
  )
    return null;

  const cliffVal = summary.cliff_delta ?? 0;
  const cliffSign = cliffVal > 0 ? "+" : cliffVal < 0 ? "" : "";
  const cliffDir = cliffVal > 0 ? "improves" : cliffVal < 0 ? "degrades" : "unchanged";

  const deltaColors: Record<string, string> = {
    negligible: "var(--color-text-muted)",
    small: "var(--color-accent-warning)",
    medium: "var(--color-accent)",
    large: "var(--color-accent-success)",
  };

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        Learning Curve
      </h3>
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Cliff's Delta
            </span>
            <span
              className="font-mono text-xs"
              style={{
                color:
                  deltaColors[summary.delta_interpretation ?? ""] ??
                  "var(--color-text-primary)",
              }}
            >
              {cliffSign}
              {cliffVal.toFixed(3)}
              {summary.delta_interpretation ? ` (${summary.delta_interpretation})` : ""}
            </span>
            <span className="block text-[9px] text-(--color-text-dim)">
              Post-startup {cliffDir} over startup
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Best Uplift
            </span>
            <span
              className="font-mono text-xs"
              style={{
                color:
                  (summary.best_uplift_pct ?? 0) > 0
                    ? "var(--color-accent-success)"
                    : "var(--color-text-primary)",
              }}
            >
              {summary.best_uplift_pct != null
                ? `${summary.best_uplift_pct > 0 ? "+" : ""}${summary.best_uplift_pct.toFixed(1)}%`
                : "—"}
            </span>
            <span className="block text-[9px] text-(--color-text-dim)">
              Best score improvement
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Beating Startup
            </span>
            <span className="font-mono text-xs text-(--color-text-primary)">
              {summary.share_beating_startup != null
                ? `${(summary.share_beating_startup * 100).toFixed(0)}%`
                : "—"}
            </span>
            <span className="block text-[9px] text-(--color-text-dim)">
              of post-startup trials
            </span>
          </div>
          <div>
            <span className="block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
              Trial Split
            </span>
            <span className="font-mono text-xs text-(--color-text-primary)">
              {summary.startup_trials} startup / {summary.post_startup_trials} post
            </span>
            <span className="block text-[9px] text-(--color-text-dim)">
              Median:{" "}
              {summary.startup_median_score != null
                ? summary.startup_median_score.toFixed(4)
                : "—"}{" "}
              →{" "}
              {summary.post_startup_median_score != null
                ? summary.post_startup_median_score.toFixed(4)
                : "—"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function SensitivityChart({
  sensitivity,
}: {
  sensitivity: HpoSensitivityEntry[] | null;
}) {
  if (!sensitivity || sensitivity.length === 0) return null;

  const maxAbs = Math.max(...sensitivity.map((s) => Math.abs(s.index)), 0.01);
  const top = [...sensitivity].sort((a, b) => Math.abs(b.index) - Math.abs(a.index));

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        Parameter Sensitivity
      </h3>
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3">
        <span className="mb-2.5 block text-[10px] font-semibold tracking-wide text-(--color-text-muted) uppercase">
          Spearman &rho; (param vs objective)
        </span>
        <div className="flex flex-col gap-1">
          {top.slice(0, 12).map((s) => {
            const barPct = (Math.abs(s.index) / maxAbs) * 100;
            const isPositive = s.index > 0;
            return (
              <div key={s.param} className="flex items-center gap-2 text-[11px]">
                <span className="w-32 shrink-0 truncate text-(--color-text-muted)">
                  {s.param}
                </span>
                <div className="flex-1 h-2.5 rounded-sm overflow-hidden bg-(--color-glass-border)">
                  <div
                    className="h-full rounded-sm transition-all"
                    style={{
                      width: `${barPct}%`,
                      backgroundColor: isPositive
                        ? "var(--color-accent-success)"
                        : "var(--color-accent-danger)",
                    }}
                  />
                </div>
                <span className="w-14 text-right font-mono tabular-nums text-(--color-text-primary)">
                  {s.index > 0 ? "+" : ""}
                  {s.index.toFixed(3)}
                </span>
                <span className="w-16 text-right text-[9px] text-(--color-text-dim)">
                  {s.perturbation_direction ?? ""}
                </span>
              </div>
            );
          })}
          {top.length > 12 && (
            <span className="text-[10px] text-(--color-text-muted) mt-1">
              +{top.length - 12} more
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
