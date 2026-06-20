import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { Phase4Cache } from "@/stores/useCommitteeMonitorStore";
import type { TrustScoreResult } from "@/api/schemas";
import { Check, X, AlertTriangle, Shield, Target } from "lucide-react";

function GaugeRing({
  value,
  max,
  label,
  goodThreshold,
  invert,
}: {
  value: number;
  max: number;
  label: string;
  goodThreshold?: number;
  invert?: boolean;
}) {
  const pct = Math.min(((value ?? 0) / max) * 100, 100);
  const r = 28;
  const c = 2 * Math.PI * r;
  const dashLen = (pct / 100) * c;
  const isGood =
    goodThreshold !== undefined
      ? invert
        ? (value ?? 0) <= goodThreshold
        : (value ?? 0) >= goodThreshold
      : undefined;

  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg width={64} height={64} viewBox="0 0 64 64">
        <circle
          cx={32}
          cy={32}
          r={r}
          fill="none"
          stroke="var(--color-text-dim)"
          strokeWidth={3}
          opacity={0.15}
        />
        <circle
          cx={32}
          cy={32}
          r={r}
          fill="none"
          stroke={
            isGood ? "var(--color-accent-success)" : "var(--color-accent-warning)"
          }
          strokeWidth={3}
          strokeLinecap="round"
          strokeDasharray={`${dashLen} ${c}`}
          transform="rotate(-90 32 32)"
          style={{ transition: "stroke-dasharray 500ms ease" }}
        />
        <text
          x={32}
          y={30}
          textAnchor="middle"
          dominantBaseline="central"
          className="font-mono text-[13px] font-bold"
          fill="var(--color-text-primary)"
        >
          {(value ?? 0).toFixed(3)}
        </text>
        <text
          x={32}
          y={44}
          textAnchor="middle"
          dominantBaseline="central"
          className="text-[7px] uppercase tracking-[0.04em]"
          fill="var(--color-text-dim)"
        >
          {label}
        </text>
      </svg>
    </div>
  );
}

function TrustBanner({ trust }: { trust: TrustScoreResult }) {
  const actionColors: Record<string, string> = {
    deploy: "var(--color-accent-success)",
    proceed: "var(--color-accent-success)",
    flag: "var(--color-accent-warning)",
    reject: "var(--color-accent-danger)",
  };
  const actionIcons: Record<string, typeof Shield> = {
    deploy: Shield,
    proceed: Shield,
    flag: AlertTriangle,
    reject: X,
  };
  const Icon = actionIcons[trust.action] || Shield;
  const borderColor = actionColors[trust.action] || "var(--color-glass-border)";

  return (
    <div
      className="flex items-start gap-3 rounded-[2px] border p-3"
      style={{
        borderColor,
        backgroundColor: `${borderColor}0a`,
      }}
    >
      <Icon size={18} style={{ color: borderColor }} className="shrink-0" />
      <div>
        <div
          className="text-[10px] font-semibold uppercase tracking-[0.04em]"
          style={{ color: borderColor }}
        >
          Trust Score: {(trust.trust_score * 100).toFixed(1)}%
          {" \u2014 "}
          {trust.action.toUpperCase()}
        </div>
        <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-4">
          {Object.entries(trust.sub_scores).map(([k, v]) => (
            <div key={k} className="font-mono text-[9px]">
              <span className="text-(--color-text-dim)">{k.replace(/_/g, " ").replace("contribution", "")}</span>
              <span className="ml-1 text-(--color-text-secondary)">{(v * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface RegimeCoverageEntry {
  sharpe?: number;
  trades?: number;
  folds_active?: number;
  covered?: boolean;
}

function RegimeScorecard({
  regime,
  entry,
}: {
  regime: string;
  entry: RegimeCoverageEntry;
}) {
  const sharpe = entry.sharpe ?? 0;
  const trades = entry.trades ?? 0;
  const covered = entry.covered ?? false;
  const foldsActive = entry.folds_active ?? 0;

  return (
    <div
      className="rounded-[2px] border p-2.5"
      style={{
        borderColor: covered ? "rgba(16,185,129,0.2)" : "rgba(244,63,94,0.15)",
        backgroundColor: covered ? "rgba(16,185,129,0.03)" : "rgba(244,63,94,0.02)",
      }}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] font-semibold text-(--color-text-primary)">
          {regime.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
        </span>
        <span
          className="text-[8px] font-semibold uppercase tracking-[0.04em]"
          style={{ color: covered ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}
        >
          {covered ? "Covered" : "Missing"}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-1">
        <div>
          <div className="text-[7px] uppercase tracking-[0.04em] text-(--color-text-dim)">Sharpe</div>
          <div
            className="font-mono text-[10px] font-semibold"
            style={{ color: sharpe >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}
          >
            {sharpe >= 0 ? "+" : ""}{sharpe.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-[7px] uppercase tracking-[0.04em] text-(--color-text-dim)">Trades</div>
          <div className="font-mono text-[10px] font-semibold text-(--color-text-secondary)">{trades}</div>
        </div>
        <div>
          <div className="text-[7px] uppercase tracking-[0.04em] text-(--color-text-dim)">Folds</div>
          <div className="font-mono text-[10px] font-semibold text-(--color-text-secondary)">{foldsActive}</div>
        </div>
      </div>
    </div>
  );
}

export function ValidationView() {
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const survivingModels = useCommitteeMonitorStore((s) => s.survivingModels);
  // Live data from status endpoint
  const liveCv = useCommitteeMonitorStore((s) => s.liveFoldConsistencyCv);
  const livePbo = useCommitteeMonitorStore((s) => s.livePbo);
  const liveDsr = useCommitteeMonitorStore((s) => s.liveDsr);
  const liveTrustScore = useCommitteeMonitorStore((s) => s.liveTrustScore);
  const liveRegimeCoverage = useCommitteeMonitorStore((s) => s.liveRegimeCoverage);
  const liveSeedSharpes = useCommitteeMonitorStore((s) => s.liveSeedSharpes);
  const liveSeedAvg = useCommitteeMonitorStore((s) => s.liveSeedAvgSharpe);
  const liveSeedPass = useCommitteeMonitorStore((s) => s.liveSeedPass);
  const liveWfoFoldSharpes = useCommitteeMonitorStore((s) => s.liveWfoFoldSharpes);
  const liveWfoFoldProgress = useCommitteeMonitorStore((s) => s.liveWfoFoldProgress);
  const liveWfoRunningAvg = useCommitteeMonitorStore((s) => s.liveWfoRunningAvgSharpe);

  const cache = phaseCache[4] as Phase4Cache | null;
  // Merge: live data takes priority, cache as fallback
  const foldCv = liveCv ?? cache?.foldConsistencyCv;
  const pbo = livePbo ?? cache?.pbo;
  const dsr = liveDsr ?? cache?.dsr;
  const trustScore = liveTrustScore ?? cache?.trustScore;
  const regimeCoverage = liveRegimeCoverage ?? cache?.regimeCoverage;
  const seedSharpes = liveSeedSharpes.length > 0 ? liveSeedSharpes : cache?.seedSharpes ?? [];
  const seedAvg = liveSeedAvg ?? (seedSharpes.length > 0 ? seedSharpes.reduce((a, b) => a + b, 0) / seedSharpes.length : null);
  const seedPass = liveSeedPass ?? cache?.seedPass;
  const hasData = cache !== null || liveCv !== null || livePbo !== null || liveDsr !== null;

  // Parse regime coverage entries
  const regimeEntries: [string, RegimeCoverageEntry][] = [];
  if (regimeCoverage && typeof regimeCoverage === "object") {
    for (const [regime, val] of Object.entries(regimeCoverage)) {
      if (val && typeof val === "object") {
        regimeEntries.push([regime, val as RegimeCoverageEntry]);
      }
    }
  }
  // Sort: covered first, then by Sharpe desc
  regimeEntries.sort(([, a], [, b]) => {
    if (a.covered !== b.covered) return (a.covered ? 0 : 1) - (b.covered ? 0 : 1);
    return (b.sharpe ?? 0) - (a.sharpe ?? 0);
  });

  return (
    <div className="flex flex-col gap-5 px-2 py-4 sm:px-4">
      <div>
        <h4 className="text-[10px] font-semibold uppercase tracking-[0.08em] text-(--color-text-secondary)">
          Validation Health Dashboard
        </h4>
      </div>

      {liveWfoFoldProgress && liveWfoFoldSharpes.length > 0 && !hasData && (
        <div className="rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-3">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
            WFO in Progress — {liveWfoFoldProgress} folds
          </div>
          <div className="flex items-end gap-1 h-[60px] pb-4">
            {liveWfoFoldSharpes.map((s, i) => {
              const h = Math.max(4, Math.abs(s) * 40);
              return (
                <div key={i} className="flex flex-1 flex-col items-center gap-0.5">
                  <span className="font-mono text-[8px]" style={{ color: s >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
                    {s >= 0 ? "+" : ""}{s.toFixed(2)}
                  </span>
                  <div
                    className="w-full rounded-t-sm"
                    style={{
                      height: h,
                      backgroundColor: s >= 0.3 ? "var(--color-accent-success)" : s >= 0 ? "var(--color-accent-warning)" : "var(--color-accent-danger)",
                      opacity: 0.7,
                    }}
                  />
                  <span className="font-mono text-[7px] text-(--color-text-dim)">F{i + 1}</span>
                </div>
              );
            })}
            {Array.from({ length: Math.max(0, 8 - liveWfoFoldSharpes.length) }, (_, i) => (
              <div key={`empty-${i}`} className="flex flex-1 flex-col items-center gap-0.5 opacity-25">
                <div className="w-full h-[8px] rounded-t-sm bg-(--color-glass-border)" />
              </div>
            ))}
          </div>
          {liveWfoRunningAvg != null && (
            <div className="text-center font-mono text-[10px] text-(--color-text-secondary)">
              Running avg Sharpe:{" "}
              <span style={{ color: liveWfoRunningAvg >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
                {liveWfoRunningAvg >= 0 ? "+" : ""}{liveWfoRunningAvg.toFixed(3)}
              </span>
            </div>
          )}
        </div>
      )}

      {hasData ? (
        <>
          {/* Gauges row */}
          <div className="flex flex-wrap justify-center gap-6 sm:justify-start">
              <GaugeRing
                value={foldCv ?? 0}
                max={1}
                label="Fold CV"
              goodThreshold={0.5}
              invert
            />
            <GaugeRing
              value={pbo ?? 0}
              max={1}
              label="PBO"
              goodThreshold={0.2}
              invert
            />
            <GaugeRing
              value={dsr ?? 0}
              max={1}
              label="DSR"
              goodThreshold={0.5}
            />
            <GaugeRing
              value={seedAvg ?? (seedSharpes[0] ?? 0)}
              max={3}
              label="3-Seed Sharpe"
              goodThreshold={0.5}
            />
          </div>

          {/* Trust score */}
          {trustScore && <TrustBanner trust={trustScore} />}

          {/* Per-regime scorecards */}
          {regimeEntries.length > 0 && (
            <div>
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
                Per-Regime Coverage ({regimeEntries.filter(([, e]) => e.covered).length}/{regimeEntries.length})
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                {regimeEntries.map(([regime, entry]) => (
                  <RegimeScorecard key={regime} regime={regime} entry={entry} />
                ))}
              </div>
            </div>
          )}

          {/* Seed robustness checklist */}
          <div>
            <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
              3-Seed Robustness
            </div>
            <div className="flex flex-wrap gap-3">
              {[42, 101, 202].map((seed, i) => {
                const sharpe = seedSharpes[i] ?? 0;
                const pass = sharpe > 0;
                return (
                  <div
                    key={seed}
                    className="flex items-center gap-1.5 rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] px-2.5 py-1.5"
                  >
                    <span className="font-mono text-[9px] text-(--color-text-dim)">
                      Seed {seed}
                    </span>
                    <span
                      className="font-mono text-[10px] font-semibold"
                      style={{
                        color: pass ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                      }}
                    >
                      {sharpe.toFixed(4)}
                    </span>
                    {pass ? (
                      <Check size={12} className="text-(--color-accent-success)" />
                    ) : (
                      <X size={12} className="text-(--color-accent-danger)" />
                    )}
                  </div>
                );
              })}
            </div>
            <div className="mt-1 text-[9px] text-(--color-text-dim)">
              {seedPass ? (
                <span className="text-(--color-accent-success)">All seeds passed</span>
              ) : (
                <span className="text-(--color-accent-danger)">Some seeds failed</span>
              )}
            </div>
          </div>

          {/* Surviving models */}
          {survivingModels.length > 0 && (
            <div>
              <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
                Surviving Models ({survivingModels.length})
              </div>
              <div className="flex flex-wrap gap-1">
                {survivingModels.map((m) => (
                  <span
                    key={m}
                    className="rounded-[2px] px-1.5 py-0.5 font-mono text-[9px]"
                    style={{
                      backgroundColor: "rgba(0,229,255,0.06)",
                      color: "var(--color-brand)",
                      border: "1px solid rgba(0,229,255,0.12)",
                    }}
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <Target size={28} className="text-(--color-text-dim)" />
          <div className="text-[11px] text-(--color-text-muted)">
            {phaseNumber >= 4
              ? "Validation results will be available when the full cycle completes."
              : "Validation has not started yet."}
          </div>
        </div>
      )}
    </div>
  );
}
