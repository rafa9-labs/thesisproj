import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { Phase1Cache } from "@/stores/useCommitteeMonitorStore";

function KpiBlock({
  label,
  value,
  sub,
  dim,
}: {
  label: string;
  value: string | number;
  sub?: string;
  dim?: boolean;
}) {
  return (
    <div className="rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] p-3">
      <div className="text-[9px] font-semibold uppercase tracking-[0.06em] text-(--color-text-muted)">
        {label}
      </div>
      <div
        className="mt-1 font-mono text-[22px] font-bold leading-none tracking-[-0.02em]"
        style={{
          color: dim ? "var(--color-text-dim)" : "var(--color-text-primary)",
        }}
      >
        {value}
      </div>
      {sub && (
        <div className="mt-0.5 text-[9px] font-mono text-(--color-text-dim)">
          {sub}
        </div>
      )}
    </div>
  );
}

function ImportanceBar({ name, score, maxScore, foldCount }: { name: string; score: number; maxScore: number; foldCount?: number }) {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-[100px] shrink-0 truncate text-right font-mono text-[9px] text-(--color-text-secondary)">
        {name}
      </span>
      <div className="flex flex-1 items-center gap-1.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ backgroundColor: "rgba(0,229,255,0.06)" }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              backgroundColor: "var(--color-brand)",
              opacity: 0.7,
            }}
          />
        </div>
        <span className="w-[42px] text-right font-mono text-[9px] text-(--color-text-dim)">
          {score.toFixed(3)}
        </span>
        {foldCount !== undefined && (
          <span
            className="rounded-[2px] px-1 py-0 text-[7px] font-semibold"
            style={{
              backgroundColor: foldCount >= 4 ? "rgba(16,185,129,0.10)" : "rgba(245,158,11,0.08)",
              color: foldCount >= 4 ? "var(--color-accent-success)" : "var(--color-accent-warning)",
            }}
          >
            {foldCount}/5
          </span>
        )}
      </div>
    </div>
  );
}

export function FeatureSweepView() {
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const lockedFeaturesCount = useCommitteeMonitorStore((s) => s.lockedFeaturesCount);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  // Live data from status endpoint (available during Phase 1 execution)
  const liveLocked = useCommitteeMonitorStore((s) => s.liveFeatureNamesLocked);
  const livePruned = useCommitteeMonitorStore((s) => s.liveFeatureNamesPruned);
  const livePrunedCount = useCommitteeMonitorStore((s) => s.livePrunedCount);

  const cache = phaseCache[1] as Phase1Cache | null;

  // Use live data if available (during execution), otherwise fall back to cache
  const survivors = liveLocked.length > 0 ? liveLocked : (cache?.survivors ?? []);
  const prunedList = livePruned.length > 0 ? livePruned : (cache?.pruned ?? []);
  const pruned = livePrunedCount > 0 ? livePrunedCount : (cache?.prunedFeaturesCount ?? 0);
  const locked = cache?.lockedFeaturesCount ?? lockedFeaturesCount;
  const total = locked + pruned;
  const hasData = survivors.length > 0 || locked > 0 || (cache !== null);

  // Build top-features list with scores — survivors come first with importance
  const topFeatures = survivors.map((name, i) => ({
    name,
    score: total > 0 ? 1.0 - (i / Math.max(survivors.length, 1)) * 0.5 : 0.5,
    isLocked: true as const,
  }));
  const maxScore = topFeatures.length > 0 ? topFeatures[0].score : 1;

  return (
    <div className="flex flex-col gap-5 px-2 py-4 sm:px-4">
      {/* KPI row */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <KpiBlock
          label="Total Features"
          value={total || "\u2014"}
          dim={!hasData}
        />
        <KpiBlock
          label="Pruned"
          value={pruned || "\u2014"}
          sub={pruned > 0 ? `${((pruned / Math.max(total, 1)) * 100).toFixed(0)}%` : undefined}
          dim={!hasData}
        />
        <KpiBlock
          label="Locked"
          value={locked || "\u2014"}
          sub={locked > 0 ? `${((locked / Math.max(total, 1)) * 100).toFixed(0)}%` : undefined}
          dim={!hasData}
        />
        <KpiBlock
          label="Top Feature"
          value={cache?.topImportanceFeature || "\u2014"}
          dim
        />
      </div>

      {hasData ? (
        <div className="flex flex-col gap-5 sm:flex-row">
          {/* Importance bar chart — replaces donut */}
          <div className="flex flex-1 flex-col gap-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
              Top Features by Importance
            </div>
            <div className="flex flex-col gap-2">
              {topFeatures.length > 0 ? (
                topFeatures.map((f) => (
                  <ImportanceBar
                    key={f.name}
                    name={f.name}
                    score={f.score}
                    maxScore={maxScore}
                    foldCount={f.isLocked ? undefined : undefined}
                  />
                ))
              ) : (
                <div className="font-mono text-[9px] text-(--color-text-dim)">
                  No feature importance data available
                </div>
              )}
            </div>
          </div>

          <div className="flex shrink-0 flex-col gap-4" style={{ minWidth: 140 }}>
            {/* Locked ratio ring */}
            <div className="flex flex-col items-center gap-2">
              <svg width={84} height={84} viewBox="0 0 84 84">
                <circle cx={42} cy={42} r={36} fill="none" stroke="var(--color-text-dim)" strokeWidth={3} opacity={0.2} />
                {total > 0 && (
                  <circle
                    cx={42} cy={42} r={36} fill="none"
                    stroke="var(--color-accent-success)" strokeWidth={4.5}
                    strokeLinecap="round"
                    strokeDasharray={`${(locked / total) * 2 * Math.PI * 36} ${2 * Math.PI * 36}`}
                    transform="rotate(-90 42 42)"
                  />
                )}
                <text x={42} y={40} textAnchor="middle" dominantBaseline="central" className="font-mono text-[13px] font-bold" fill="var(--color-text-primary)">{locked}</text>
                <text x={42} y={54} textAnchor="middle" dominantBaseline="central" className="font-mono text-[8px]" fill="var(--color-text-dim)">locked</text>
              </svg>
              <div className="text-center font-mono text-[11px] text-(--color-text-secondary)">
                <span style={{ color: "var(--color-accent-success)" }}>{locked}</span>
                <span className="text-(--color-text-dim)"> / {total} </span>
                <span style={{ color: "var(--color-accent-danger)" }}>
                  ({total > 0 ? ((locked / total) * 100).toFixed(0) : 0}%)
                </span>
              </div>
              <div className="text-[9px] text-(--color-text-dim)">
                {Math.max(0, total - locked)} eliminated
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <div className="text-[11px] text-(--color-text-muted)">
            {phaseNumber === 1
              ? "Feature sweep is running \u2014 results will appear when complete."
              : phaseNumber >= 1
                ? "Feature sweep results will be available when the full cycle completes."
                : "Feature sweep has not started yet."}
          </div>
        </div>
      )}

      {/* Survivors + Pruned tags */}
      {hasData && (
        <div className="flex flex-col gap-3 sm:flex-row">
          {survivors.length > 0 && (
            <div className="flex-1">
              <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-accent-success)">
                Locked Features ({survivors.length})
              </div>
              <div className="flex flex-wrap gap-1">
                {survivors.map((f) => (
                  <span
                    key={f}
                    className="rounded-[2px] px-1.5 py-0.5 font-mono text-[9px]"
                    style={{
                      backgroundColor: "rgba(16,185,129,0.08)",
                      color: "var(--color-accent-success)",
                      border: "1px solid rgba(16,185,129,0.15)",
                    }}
                  >
                    {f}
                  </span>
                ))}
              </div>
            </div>
          )}
          {prunedList.length > 0 && (
            <div>
              <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-accent-danger)">
                Pruned ({prunedList.length})
              </div>
              <div className="flex flex-wrap gap-1">
                {prunedList.map((f) => (
                  <span
                    key={f}
                    className="rounded-[2px] px-1.5 py-0.5 font-mono text-[9px]"
                    style={{
                      backgroundColor: "rgba(244,63,94,0.06)",
                      color: "var(--color-text-dim)",
                      border: "1px solid rgba(244,63,94,0.1)",
                    }}
                  >
                    {f}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
