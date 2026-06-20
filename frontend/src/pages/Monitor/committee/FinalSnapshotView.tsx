import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { FinalCache, Phase1Cache, Phase2Cache, Phase4Cache, Phase5Cache } from "@/stores/useCommitteeMonitorStore";
import { Check, X, Box, Database } from "lucide-react";

function MetricCard({
  label,
  value,
  sub,
  good,
}: {
  label: string;
  value: string;
  sub?: string;
  good?: boolean;
}) {
  return (
    <div className="rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] p-3">
      <div className="text-[9px] font-semibold uppercase tracking-[0.06em] text-(--color-text-muted)">
        {label}
      </div>
      <div className="mt-1 font-mono text-[18px] font-bold leading-none tracking-[-0.02em] text-(--color-text-primary)">
        {value}
      </div>
      {sub && (
        <div
          className="mt-0.5 text-[10px] font-mono"
          style={{
            color:
              good === undefined
                ? "var(--color-text-dim)"
                : good
                  ? "var(--color-accent-success)"
                  : "var(--color-accent-danger)",
          }}
        >
          {good !== undefined && (
            <span className="mr-1">
              {good ? <Check size={10} className="inline" /> : <X size={10} className="inline" />}
            </span>
          )}
          {sub}
        </div>
      )}
    </div>
  );
}

function PhaseSummaryRow({
  phaseNum,
  label,
  metrics,
}: {
  phaseNum: number;
  label: string;
  metrics: { label: string; value: string; good?: boolean }[];
}) {
  return (
    <div className="rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] p-3">
      <div className="flex items-center gap-2 mb-2">
        <span
          className="flex h-5 w-5 items-center justify-center rounded-full font-mono text-[9px] font-bold"
          style={{
            backgroundColor: "rgba(0,229,255,0.10)",
            color: "var(--color-brand)",
          }}
        >
          {phaseNum}
        </span>
        <span className="text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-secondary)">
          {label}
        </span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[9px] text-(--color-text-dim)">
        {metrics.map((m, i) => (
          <span key={i}>
            {m.label}:{" "}
            <span
              style={{
                color:
                  m.good === undefined
                    ? "var(--color-text-secondary)"
                    : m.good
                      ? "var(--color-accent-success)"
                      : "var(--color-accent-danger)",
              }}
            >
              {m.value}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

export function FinalSnapshotView() {
  const results = useCommitteeMonitorStore((s) => s.results);
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const phase = useCommitteeMonitorStore((s) => s.phase);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const phaseTimings = useCommitteeMonitorStore((s) => s.phaseTimings);

  const cache = phaseCache[6] as FinalCache | null;
  const ph1 = phaseCache[1] as Phase1Cache | null;
  const ph2 = phaseCache[2] as Phase2Cache | null;
  const ph4 = phaseCache[4] as Phase4Cache | null;
  const ph5 = phaseCache[5] as Phase5Cache | null;

  const isComplete = phase === "completed";
  const isRunning = phaseNumber >= 1 && !isComplete;

  if (isRunning && !cache) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-16">
        <div className="text-[11px] text-(--color-text-muted)">
          Snapshot results will be available when the full cycle completes.
        </div>
      </div>
    );
  }

  // Build phase summary rows from aggregated cache data
  const phaseSummaries = [];
  if (ph1) {
    phaseSummaries.push({
      phaseNum: 1,
      label: "Feature Sweep",
      metrics: [
        { label: "Locked", value: `${ph1.lockedFeaturesCount}` },
        { label: "Pruned", value: `${ph1.prunedFeaturesCount}` },
        { label: "Top", value: ph1.topImportanceFeature || "\u2014" },
      ],
    });
  }
  if (ph2) {
    const nSuccess = Object.values(ph2.hpoStatus).filter((s) => s === "success").length;
    const nTotal = Object.keys(ph2.hpoStatus).length;
    phaseSummaries.push({
      phaseNum: 2,
      label: "HPO Tuning",
      metrics: [
        { label: "Success", value: `${nSuccess}/${nTotal}`, good: nSuccess > 0 },
        { label: "Params", value: `${ph2.hpoModelParamsCount || 0}` },
      ],
    });
  }
  if (ph4) {
    phaseSummaries.push({
      phaseNum: 4,
      label: "Validation",
      metrics: [
        { label: "Fold CV", value: (ph4.foldConsistencyCv ?? 0).toFixed(4), good: (ph4.foldConsistencyCv ?? 1) < 0.5 },
        { label: "PBO", value: (ph4.pbo ?? 0).toFixed(3), good: (ph4.pbo ?? 1) < 0.2 },
        { label: "DSR", value: (ph4.dsr ?? 0).toFixed(3), good: (ph4.dsr ?? 0) > 0.5 },
        { label: "Trust", value: ph4.trustScore ? `${(ph4.trustScore.trust_score * 100).toFixed(0)}%` : "\u2014" },
      ],
    });
  }
  if (ph5) {
    phaseSummaries.push({
      phaseNum: 5,
      label: "Factory Optimization",
      metrics: [
        { label: "Best Sharpe", value: (ph5.bestSharpe ?? 0).toFixed(4), good: (ph5.bestSharpe ?? 0) >= 0.5 },
        { label: "Accepted", value: `${ph5.acceptedCount}/${ph5.totalIterations}` },
        { label: "Stop", value: (ph5.stopReason || "completed").replace(/_/g, " ").slice(0, 40) },
      ],
    });
  }

  return (
    <div className="flex flex-col gap-5 px-2 py-4 sm:px-4">
      <div>
        <h4
          className="text-[10px] font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-accent-success)" }}
        >
          {isRunning ? "Running \u2014 Snapshot Pending" : "Snapshot & Deployment"}
        </h4>
      </div>

      {/* Phase Summary Timeline */}
      {phaseSummaries.length > 0 && (
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
            Phase Summary
          </div>
          <div className="flex flex-col gap-2">
            {phaseSummaries.map((ps) => (
              <PhaseSummaryRow
                key={ps.phaseNum}
                phaseNum={ps.phaseNum}
                label={ps.label}
                metrics={ps.metrics}
              />
            ))}
          </div>
        </div>
      )}

      {/* Final WFO + Deployment cards */}
      {cache?.finalFullWfo && (
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
            Final 10-Year Walk-Forward
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricCard
              label="Fold Sharpe CV"
              value={(cache.finalFoldCv ?? 0).toFixed(4)}
              good={(cache.finalFoldCv ?? 1) < 0.5}
              sub={cache.finalFoldPass ? "PASS" : "FAIL"}
            />
            {cache.finalSeedSharpe > 0 && (
              <MetricCard
                label="5-Seed Avg Sharpe"
                value={(cache.finalSeedSharpe ?? 0).toFixed(4)}
                good={cache.finalSeedPass}
                sub={cache.finalSeedPass ? "All seeds positive" : "Some seeds failed"}
              />
            )}
          </div>
        </div>
      )}

      {/* Aggregate stats from results */}
      {results && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <MetricCard
            label="Total Time"
            value={`${((results.total_time_s ?? 0) / 60).toFixed(1)}m`}
            sub={`${(results.total_time_s ?? 0).toFixed(0)}s`}
          />
          {results.factory_best_sharpe > 0 && (
            <MetricCard
              label="Factory Best Sharpe"
              value={(results.factory_best_sharpe ?? 0).toFixed(4)}
              good={(results.factory_best_sharpe ?? 0) >= 0.5}
            />
          )}
          {results.factory_accepted_count > 0 && (
            <MetricCard
              label="Accepted Moves"
              value={`${results.factory_accepted_count}/${results.factory_total_iterations}`}
            />
          )}
          {results.factory_stop_reason && (
            <MetricCard
              label="Stop Reason"
              value={results.factory_stop_reason.replace(/_/g, " ")}
            />
          )}
        </div>
      )}

      {/* Per-phase timing */}
      {Object.keys(phaseTimings).length > 0 && (
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
            Phase Timing
          </div>
          <div className="flex flex-wrap gap-2">
            {[1, 2, 3, 4, 5].map((pn) => {
              const elapsed = phaseTimings[`phase${pn}`];
              if (!elapsed) return null;
              const label = ["Sweep", "HPO", "Assembly", "Validate", "Factory"][pn - 1];
              const mins = Math.floor(elapsed / 60);
              const secs = Math.round(elapsed % 60);
              return (
                <div
                  key={pn}
                  className="flex items-center gap-1.5 rounded-full border border-(--color-glass-border) bg-white/[0.02] px-2.5 py-1"
                >
                  <span className="text-[8px] font-semibold uppercase tracking-[0.06em] text-(--color-text-dim)">
                    {label}
                  </span>
                  <span className="font-mono text-[10px] font-semibold text-(--color-text-secondary)">
                    {mins}m{secs}s
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Deployment artifact */}
      {cache?.snapshotDir && (
        <div
          className="flex items-center gap-3 rounded-[2px] border p-3"
          style={{
            borderColor: "rgba(0,229,255,0.2)",
            backgroundColor: "rgba(0,229,255,0.04)",
          }}
        >
          <Box size={20} style={{ color: "var(--color-brand)" }} />
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-brand)">
              Deployment Artifact Ready
            </div>
            <div className="mt-0.5 text-[10px] font-mono text-(--color-text-dim)">
              Committee snapshot saved to {cache.snapshotDir}
            </div>
          </div>
        </div>
      )}

      {!isComplete && !isRunning && (
        <div className="flex flex-col items-center justify-center gap-4 py-16">
          <Database size={28} className="text-(--color-text-dim)" />
          <div className="text-[11px] text-(--color-text-muted)">
            No results available yet. Start a full cycle to populate this view.
          </div>
        </div>
      )}
    </div>
  );
}
