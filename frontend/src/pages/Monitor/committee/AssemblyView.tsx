import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { Phase3Cache } from "@/stores/useCommitteeMonitorStore";
import { AlertTriangle } from "lucide-react";

const REGIME_DISPLAY: Record<string, string> = {
  trend_up: "Trend Up",
  trend_down: "Trend Down",
  sideways: "Sideways",
  volatile: "Volatile",
  quiet: "Quiet",
  reversal: "Reversal",
  breakout: "Breakout",
};

const ALL_REGIMES = ["trend_up", "trend_down", "sideways", "volatile", "quiet", "reversal", "breakout"];

function RegimeHeatmapCell({
  weight,
  color,
}: {
  weight: number;
  color: string;
}) {
  const pct = Math.round(weight * 100);
  return (
    <div
      className="flex items-center justify-center rounded-[2px] border font-mono text-[9px] font-semibold transition-all"
      style={{
        backgroundColor: `${color}${Math.round(weight * 40).toString(16).padStart(2, "0")}`,
        borderColor: `${color}${Math.round(weight * 60 + 20).toString(16).padStart(2, "0")}`,
        color: weight > 0.3 ? "var(--color-text-inverse)" : "var(--color-text-dim)",
        minWidth: 36,
        height: 28,
      }}
    >
      {pct > 0 ? `${pct}%` : ""}
    </div>
  );
}

function ModelDiversityPanel({ usage, allModels }: { usage: Record<string, number>; allModels: string[] }) {
  const entries = Object.entries(usage).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) return null;
  const maxUsage = entries[0]?.[1] ?? 1;
  const nRegimes = Math.max(...Object.values(usage), 1);
  return (
    <div>
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
        Model Diversity
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {entries.map(([model, count]) => {
          const pct = nRegimes > 0 ? (count / nRegimes) * 100 : 0;
          const isDominant = pct > 60;
          return (
            <div key={model} className="flex items-center gap-1.5">
              <span className="font-mono text-[9px] text-(--color-text-secondary)">{model.toUpperCase()}</span>
              <div className="h-1 flex-1 overflow-hidden rounded-full" style={{ backgroundColor: "rgba(0,229,255,0.06)" }}>
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: isDominant ? "var(--color-accent-warning)" : "var(--color-brand)",
                    opacity: 0.6,
                  }}
                />
              </div>
              <span className="w-[22px] text-right font-mono text-[8px] text-(--color-text-dim)">
                {count}/{nRegimes}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AssemblyView() {
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const liveConfig = useCommitteeMonitorStore((s) => s.liveCommitteeConfig);

  const cache = phaseCache[3] as Phase3Cache | null;
  // Use live config when available (during execution), fall back to results cache
  const config = (liveConfig?.regimes ? liveConfig : cache?.committeeConfig) as Record<string, { models: string[]; weights: number[] }> | null | undefined;

  const regimes = config?.regimes
    ? Object.entries(config.regimes).sort(
        ([, a], [, b]) => (b.weights?.[0] ?? 0) - (a.weights?.[0] ?? 0),
      )
    : [];

  // Collect all models across regimes for the heatmap
  const allModels = new Set<string>();
  const regimeWeightMap = new Map<string, Map<string, number>>();
  for (const [regime, assignment] of regimes) {
    regimeWeightMap.set(regime, new Map());
    for (let i = 0; i < assignment.models.length; i++) {
      const model = assignment.models[i];
      allModels.add(model);
      regimeWeightMap.get(regime)!.set(model, assignment.weights[i] ?? 0);
    }
  }
  const modelList = [...allModels].sort();

  // Compute model usage across regimes
  const modelUsage: Record<string, number> = {};
  for (const model of modelList) {
    let count = 0;
    for (const [, weights] of regimeWeightMap) {
      if (weights.has(model)) count++;
    }
    modelUsage[model] = count;
  }

  // Find uncovered regimes
  const uncoveredRegimes = regimes.length > 0
    ? ALL_REGIMES.filter((r) => !regimeWeightMap.has(r))
    : [];

  return (
    <div className="flex flex-col gap-5 px-2 py-4 sm:px-4">
      <div>
        <h4 className="text-[10px] font-semibold uppercase tracking-[0.08em] text-(--color-text-secondary)">
          Per-Regime Committee Assembly
        </h4>
      </div>

      {regimes.length > 0 ? (
        <>
          {/* Regime x Model heatmap */}
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--color-glass-border)" }}>
                  <th className="py-1.5 pr-2 text-left text-[9px] font-semibold uppercase tracking-[0.06em] text-(--color-text-dim)">
                    Regime
                  </th>
                  {modelList.map((m) => (
                    <th key={m} className="px-1 py-1.5 text-center font-mono text-[8px] uppercase tracking-[0.04em] text-(--color-text-dim)">
                      {m.slice(0, 6)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {regimes.map(([regime, assignment]) => (
                  <tr key={regime} className="border-b" style={{ borderColor: "rgba(51,65,85,0.2)" }}>
                    <td className="py-1.5 pr-2 font-semibold text-[10px] text-(--color-text-primary) whitespace-nowrap">
                      {REGIME_DISPLAY[regime] || regime.replace(/_/g, " ")}
                    </td>
                    {modelList.map((m) => {
                      const w = regimeWeightMap.get(regime)?.get(m) ?? 0;
                      return (
                        <td key={m} className="px-1 py-1 text-center">
                          <RegimeHeatmapCell weight={w} color="var(--color-brand)" />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Weights breakdown */}
          <div>
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
              Weight Breakdown
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {regimes.map(([regime, assignment]) => (
                <div
                  key={regime}
                  className="rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] p-2.5"
                >
                  <div className="mb-1.5 text-[10px] font-semibold text-(--color-text-secondary)">
                    {REGIME_DISPLAY[regime] || regime.replace(/_/g, " ")}
                  </div>
                  {assignment.models.map((m, i) => {
                    const w = assignment.weights[i] ?? 0;
                    return (
                      <div key={m} className="flex items-center gap-2 text-[9px]">
                        <span className="w-[60px] truncate font-mono text-(--color-text-dim)">{m}</span>
                        <div className="h-1 flex-1 overflow-hidden rounded-full" style={{ backgroundColor: "rgba(0,229,255,0.06)" }}>
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.round(w * 100)}%`,
                              backgroundColor: "var(--color-brand)",
                              opacity: 0.7,
                            }}
                          />
                        </div>
                        <span className="w-[28px] text-right font-mono text-(--color-text-dim)">
                          {Math.round(w * 100)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          {/* Model Diversity */}
          <ModelDiversityPanel usage={modelUsage} allModels={modelList} />

          {/* Coverage warnings */}
          {uncoveredRegimes.length > 0 && (
            <div
              className="flex items-start gap-2 rounded-[2px] border p-2.5"
              style={{
                borderColor: "rgba(245,158,11,0.3)",
                backgroundColor: "rgba(245,158,11,0.05)",
                color: "var(--color-accent-warning)",
              }}
            >
              <AlertTriangle size={14} className="shrink-0 mt-0.5" />
              <div className="text-[9px]">
                No models assigned to: {uncoveredRegimes.map((r) => REGIME_DISPLAY[r] || r).join(", ")}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <div className="text-[11px] text-(--color-text-muted)">
            {phaseNumber >= 3
              ? "Committee assembly results will be available when the full cycle completes."
              : "Committee assembly has not started yet."}
          </div>
        </div>
      )}

      {/* Fallback */}
      {config?.fallback && regimes.length === 0 && (
        <div className="mt-2">
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-accent-warning)">
            Fallback (All Regimes)
          </div>
          <div className="flex flex-wrap gap-1">
            {config.fallback.models.map((m) => (
              <span
                key={m}
                className="rounded-[2px] px-1.5 py-0.5 font-mono text-[9px]"
                style={{
                  backgroundColor: "rgba(245,158,11,0.08)",
                  color: "var(--color-accent-warning)",
                  border: "1px solid rgba(245,158,11,0.15)",
                }}
              >
                {m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
