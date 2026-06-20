import { HorizontalBarChart } from "@/components/charts/SimpleCharts";
import type { HpoParamImportance, HpoTrial, BestStudy } from "@/api/schemas";

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
              Best Value
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
