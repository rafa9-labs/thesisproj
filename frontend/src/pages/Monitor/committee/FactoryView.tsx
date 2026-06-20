import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { Phase5Cache } from "@/stores/useCommitteeMonitorStore";
import type { FactoryIterationRecord } from "@/api/schemas";
import { Check, X, TrendingUp, Loader2, Zap } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface ChartPoint {
  iteration: number;
  sharpe: number;
}

function RegimeDeltaGrid({ history }: { history: FactoryIterationRecord[] }) {
  // Collect all regimes across all iterations
  const regimeSet = new Set<string>();
  for (const r of history) {
    if (r.per_regime_delta && typeof r.per_regime_delta === "object") {
      for (const reg of Object.keys(r.per_regime_delta)) {
        regimeSet.add(reg);
      }
    }
  }
  const regimes = [...regimeSet].sort();
  // Get last N iterations (max 12 visible rows)
  const recent = history.slice(-12);

  if (regimes.length === 0 || recent.length === 0) return null;

  const maxAbs = Math.max(
    0.01,
    ...recent.flatMap((r) => {
      if (!r.per_regime_delta) return [0];
      return Object.values(r.per_regime_delta).map((v) => Math.abs(Number(v) || 0));
    }),
  );

  return (
    <div>
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
        Per-Regime Sharpe Delta (Recent Iterations)
      </div>
      <div className="overflow-x-auto rounded-[2px] border border-(--color-glass-border)">
        <table className="w-full border-collapse text-left">
          <thead
            className="sticky top-0 z-10 border-b text-[9px] font-semibold uppercase tracking-[0.06em]"
            style={{
              backgroundColor: "var(--color-surface)",
              borderColor: "var(--color-glass-border)",
            }}
          >
            <tr>
              <th className="py-1.5 pl-2 pr-1 text-(--color-text-dim)">#</th>
              {regimes.map((r) => (
                <th key={r} className="px-1 py-1.5 text-center text-(--color-text-dim)">
                  {r.slice(0, 8)}
                </th>
              ))}
              <th className="py-1.5 pl-1 pr-2 text-center text-(--color-text-dim)">Acc</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((row) => (
              <tr
                key={row.iteration}
                className="border-b text-[10px]"
                style={{
                  borderColor: "rgba(51,65,85,0.2)",
                  borderLeft: `2px solid ${
                    row.accepted ? "var(--color-accent-success)" : "transparent"
                  }`,
                }}
              >
                <td className="py-1 pl-2 pr-1 font-mono text-(--color-text-dim)">
                  {row.iteration}
                </td>
                {regimes.map((reg) => {
                  const delta = Number(row.per_regime_delta?.[reg] ?? 0);
                  const intensity = Math.min(Math.abs(delta) / maxAbs, 1);
                  return (
                    <td
                      key={reg}
                      className="px-1 py-1 text-center font-mono text-[8px] font-semibold"
                      style={{
                        backgroundColor:
                          delta > 0
                            ? `rgba(16,185,129,${(intensity * 0.25).toFixed(2)})`
                            : delta < 0
                              ? `rgba(244,63,94,${(intensity * 0.2).toFixed(2)})`
                              : "transparent",
                        color:
                          delta > 0
                            ? "var(--color-accent-success)"
                            : delta < 0
                              ? "var(--color-accent-danger)"
                              : "var(--color-text-dim)",
                      }}
                    >
                      {delta > 0 ? "+" : ""}
                      {delta.toFixed(2)}
                    </td>
                  );
                })}
                <td className="py-1 pl-1 pr-2 text-center">
                  {row.accepted ? (
                    <Check size={12} className="inline text-(--color-accent-success)" />
                  ) : (
                    <X size={12} className="inline text-(--color-text-dim)" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function FactoryView() {
  const sharpeTrajectory = useCommitteeMonitorStore((s) => s.sharpeTrajectory);
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const bestSharpeSoFar = useCommitteeMonitorStore((s) => s.bestSharpeSoFar);
  const iteration = useCommitteeMonitorStore((s) => s.iteration);
  const totalIterations = useCommitteeMonitorStore((s) => s.totalIterations);
  const currentAction = useCommitteeMonitorStore((s) => s.currentAction);
  // Live factory fields
  const liveAcceptedCount = useCommitteeMonitorStore((s) => s.liveFactoryAcceptedCount);
  const liveLastDelta = useCommitteeMonitorStore((s) => s.liveFactoryLastDelta);
  const liveLastAction = useCommitteeMonitorStore((s) => s.liveFactoryLastAction);
  const liveLastRegime = useCommitteeMonitorStore((s) => s.liveFactoryLastRegime);
  const liveLastAccepted = useCommitteeMonitorStore((s) => s.liveFactoryLastAccepted);

  const cache = phaseCache[5] as Phase5Cache | null;
  const isRunning = phaseNumber === 5;
  const isDone = phaseNumber > 5 || cache !== null;

  const history = cache?.factoryHistory ?? [];

  // Build chart data from trajectory (live) or history (completed)
  const chartData: ChartPoint[] = history.length > 0
    ? history.map((r) => ({ iteration: r.iteration, sharpe: r.after_sharpe }))
    : sharpeTrajectory;

  const baseline = chartData.length > 0 ? chartData[0].sharpe : 0;

  const acceptedCount = liveAcceptedCount > 0 ? liveAcceptedCount : (cache?.acceptedCount ?? history.filter((r) => r.accepted).length);

  // Convergence analysis
  const recentImprovements = chartData.length > 10
    ? chartData.slice(-10).map((p, i, arr) => i > 0 ? p.sharpe - arr[i - 1].sharpe : 0).slice(1)
    : [];
  const avgRecentDelta = recentImprovements.length > 0
    ? recentImprovements.reduce((a, b) => a + b, 0) / recentImprovements.length
    : undefined;
  const recentDeltaSum = recentImprovements.length > 0
    ? recentImprovements.reduce((a, b) => a + Math.abs(b), 0)
    : 0;

  return (
    <div className="flex flex-col gap-5 px-2 py-4 sm:px-4">
      {/* Live status header */}
      {isRunning && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <div className="flex items-center gap-2">
            <Loader2 size={14} className="animate-spin text-(--color-brand)" />
            <span className="font-mono text-[10px] text-(--color-text-secondary)">
              Optimization running \u2014 iteration {iteration}{totalIterations > 0 ? `/${totalIterations}` : ""}
            </span>
          </div>
          {currentAction && (
            <span className="font-mono text-[10px] text-(--color-text-dim)">
              {currentAction}
            </span>
          )}
        </div>
      )}

      {/* KPI pills */}
      <div className="flex flex-wrap gap-2">
        <KpiPill
          label="Iteration"
          value={totalIterations > 0 ? `${iteration}/${totalIterations}` : `${iteration}`}
        />
        <KpiPill
          label="Best Sharpe"
          value={bestSharpeSoFar.toFixed(4)}
          dim={bestSharpeSoFar === 0}
          good={bestSharpeSoFar >= 0.5}
        />
        <KpiPill
          label="Accepted"
          value={`${acceptedCount}`}
          good={acceptedCount > 0}
        />
        <KpiPill
          label="Rejected"
          value={`${(history.length || Math.max(0, iteration || 0)) - acceptedCount}`}
          dim
        />
        {/* Convergence indicator */}
        {(avgRecentDelta !== undefined && (chartData.length > 10 || isDone)) && (
          <KpiPill
            label={avgRecentDelta < 0.005 ? "Converged" : "Improving"}
            value={
              avgRecentDelta >= 0.005
                ? `\u0394${avgRecentDelta.toFixed(4)}/iter`
                : "\u0394<0.005"
            }
            good={avgRecentDelta < 0.01}
            dim={chartData.length <= 10}
          />
        )}
        {cache?.stopReason && (
          <KpiPill label="Stopped" value={cache.stopReason.replace(/_/g, " ")} dim />
        )}
      </div>

      {/* Sharpe trajectory chart */}
      <div>
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
          Sharpe Trajectory
        </div>
        <div className="rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] p-3">
          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="var(--color-text-dim)"
                  opacity={0.1}
                />
                <XAxis
                  dataKey="iteration"
                  tick={{ fontSize: 9, fontFamily: "JetBrains Mono", fill: "var(--color-text-dim)" }}
                  axisLine={{ stroke: "var(--color-glass-border)" }}
                  tickLine={false}
                  label={{ value: "Iteration", position: "insideBottom", offset: -4, fontSize: 9, fill: "var(--color-text-dim)" }}
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 9, fontFamily: "JetBrains Mono", fill: "var(--color-text-dim)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => v.toFixed(2)}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--color-surface)",
                    border: "1px solid var(--color-glass-border)",
                    borderRadius: 2,
                    fontSize: 10,
                    fontFamily: "JetBrains Mono",
                  }}
                  labelFormatter={(l: number) => `Iteration ${l}`}
                  formatter={(v: number) => [v.toFixed(4), "Sharpe"]}
                />
                <ReferenceLine
                  y={baseline}
                  stroke="var(--color-text-dim)"
                  strokeDasharray="4 4"
                  opacity={0.4}
                />
                <Line
                  type="monotone"
                  dataKey="sharpe"
                  stroke="var(--color-brand)"
                  strokeWidth={1.5}
                  dot={{ r: 2, fill: "var(--color-brand)", strokeWidth: 0 }}
                  activeDot={{ r: 4, fill: "var(--color-brand)", strokeWidth: 0 }}
                  animationDuration={300}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center py-12 text-[10px] text-(--color-text-dim) font-mono">
              {isRunning ? "Collecting data..." : "No factory iteration data available"}
            </div>
          )}
        </div>
      </div>

      {/* Regime delta heatmap */}
      {history.length > 0 && (
        <RegimeDeltaGrid history={history} />
      )}

      {/* Action ledger */}
      {history.length > 0 && (
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
            Action Ledger
          </div>
          <div className="max-h-[320px] overflow-y-auto rounded-[2px] border border-(--color-glass-border)">
            <table className="w-full border-collapse text-left">
              <thead
                className="sticky top-0 z-10 border-b text-[9px] font-semibold uppercase tracking-[0.06em]"
                style={{
                  backgroundColor: "var(--color-surface)",
                  borderColor: "var(--color-glass-border)",
                }}
              >
                <tr>
                  <th className="py-1.5 pl-3 pr-2 text-(--color-text-dim)">#</th>
                  <th className="py-1.5 pr-2 text-(--color-text-dim)">Action</th>
                  <th className="py-1.5 pr-2 text-(--color-text-dim)">Regime</th>
                  <th className="py-1.5 pr-2 text-(--color-text-dim)">Change</th>
                  <th className="py-1.5 pr-2 text-right text-(--color-text-dim)">Delta</th>
                  <th className="py-1.5 pr-3 text-center text-(--color-text-dim)">\u2713</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row) => (
                  <LedgerRow key={row.iteration} row={row} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!isRunning && !isDone && (
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <TrendingUp size={28} className="text-(--color-text-dim)" />
          <div className="text-[11px] text-(--color-text-muted)">
            Factory optimization has not started yet.
          </div>
        </div>
      )}
    </div>
  );
}

function LedgerRow({ row }: { row: FactoryIterationRecord }) {
  const delta = row.after_sharpe - row.before_sharpe;
  const a = row.action;

  return (
    <tr
      className="border-b text-[10px] transition-colors duration-150"
      style={{
        borderColor: "rgba(51,65,85,0.2)",
        borderLeft: `2px solid ${
          row.accepted ? "var(--color-accent-success)" : "transparent"
        }`,
      }}
    >
      <td className="py-1.5 pl-3 pr-2 font-mono text-(--color-text-dim)">
        {row.iteration}
      </td>
      <td className="py-1.5 pr-2">
        <span
          className="rounded-[2px] px-1 py-0.5 text-[8px] font-semibold uppercase tracking-[0.04em]"
          style={{
            backgroundColor:
              a.type === "swap_model"
                ? "rgba(0,229,255,0.08)"
                : a.type === "add_model"
                  ? "rgba(16,185,129,0.08)"
                  : "rgba(244,63,94,0.06)",
            color:
              a.type === "swap_model"
                ? "var(--color-brand)"
                : a.type === "add_model"
                  ? "var(--color-accent-success)"
                  : "var(--color-accent-danger)",
          }}
        >
          {a.type.replace(/_/g, " ")}
        </span>
      </td>
      <td className="py-1.5 pr-2 text-(--color-text-muted)">
        {a.regime.replace(/_/g, " ")}
      </td>
      <td className="py-1.5 pr-2 font-mono text-[9px] text-(--color-text-dim)">
        {a.model_remove && (
          <span className="text-(--color-accent-danger)">{a.model_remove}</span>
        )}
        {a.model_remove && a.model_add && " \u2192 "}
        {a.model_add && (
          <span className="text-(--color-accent-success)">{a.model_add}</span>
        )}
      </td>
      <td
        className="py-1.5 pr-2 text-right font-mono text-[10px] font-semibold"
        style={{
          color: delta >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
        }}
      >
        {delta >= 0 ? "+" : ""}
        {delta.toFixed(4)}
      </td>
      <td className="py-1.5 pr-3 text-center">
        {row.accepted ? (
          <Check size={12} className="inline text-(--color-accent-success)" />
        ) : (
          <X size={12} className="inline text-(--color-text-dim)" />
        )}
      </td>
    </tr>
  );
}

function KpiPill({
  label,
  value,
  good,
  dim,
}: {
  label: string;
  value: string;
  good?: boolean;
  dim?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5 rounded-full border border-(--color-glass-border) bg-white/[0.02] px-2.5 py-1">
      <span className="text-[8px] font-semibold uppercase tracking-[0.06em] text-(--color-text-dim)">
        {label}
      </span>
      <span
        className="font-mono text-[10px] font-semibold"
        style={{
          color: dim
            ? "var(--color-text-dim)"
            : good
              ? "var(--color-accent-success)"
              : "var(--color-text-secondary)",
        }}
      >
        {value}
      </span>
    </div>
  );
}
