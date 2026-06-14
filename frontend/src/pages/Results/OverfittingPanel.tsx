import { useMemo } from "react";
import type { OverfittingReportData, WalkForwardPeriod } from "@/api/schemas";

interface Props {
  overfitting: OverfittingReportData | null;
  walkforwardPeriods: WalkForwardPeriod[] | null;
}

const RISK_COLORS: Record<string, string> = {
  green: "var(--color-accent-success)",
  yellow: "var(--color-accent-warning)",
  red: "var(--color-accent-danger)",
};

export function OverfittingPanel({ overfitting, walkforwardPeriods }: Props) {
  const gapPeriods = useMemo(() => {
    if (!walkforwardPeriods || walkforwardPeriods.length === 0) return [];
    return walkforwardPeriods.map((p, i) => ({
      label: (p.period_start ?? "").slice(0, 7) || `P${i + 1}`,
      gap: p.sharpe_gap_pct,
      trainSharpe: p.train_sharpe,
      testSharpe: p.test_sharpe,
    }));
  }, [walkforwardPeriods]);

  if (!overfitting) return null;

  const hasGaps = gapPeriods.some((g) => g.gap != null);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold tracking-[0.1em] text-(--color-text-muted) uppercase">
          Score
        </span>
        <div className="flex items-center gap-2">
          <div
            className="h-2 w-2 rounded-full"
            style={{
              backgroundColor: RISK_COLORS[overfitting.risk_color] ?? "var(--color-text-muted)",
            }}
          />
          <span
            className="font-mono text-sm font-bold"
            style={{
              color: RISK_COLORS[overfitting.risk_color] ?? "var(--color-text-muted)",
            }}
          >
            {overfitting.overfit_score.toFixed(0)}
          </span>
        </div>
      </div>

      {overfitting.dsr_min_sharpe != null && (
        <div className="mb-4 rounded border border-[rgba(99,102,241,0.15)] bg-[rgba(99,102,241,0.06)] px-2 py-1 font-mono text-[10px] text-(--color-text-secondary)">
          Significance threshold: Sharpe &ge; {overfitting.dsr_min_sharpe.toFixed(2)} required for
          95% confidence (adjusted for {overfitting.n_periods} OOS periods and{" "}
          {overfitting.n_signal_periods > 0 ? "HPO testing" : "multiple tests"}).
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        {/* Left: CI bars */}
        <div className="flex flex-col gap-3">
          {overfitting.sharpe_ci?.mean != null && (
            <CiBarRow label="Sharpe" ci={overfitting.sharpe_ci} format="number" />
          )}
          {overfitting.return_ci?.mean != null && (
            <CiBarRow label="Return" ci={overfitting.return_ci} format="pct" />
          )}
          {overfitting.maxdd_ci?.mean != null && (
            <CiBarRow label="Max DD" ci={overfitting.maxdd_ci} format="pct" />
          )}
          {overfitting.cv_sharpe_std != null && (
            <div className="mt-1 flex items-center gap-2">
              <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
                CV fold &sigma;
              </span>
              <span className="font-mono text-xs text-(--color-text-primary)">
                {overfitting.cv_sharpe_std.toFixed(3)}
              </span>
            </div>
          )}
          {overfitting.interaction_effects && overfitting.interaction_effects.length > 0 && (
            <div className="mt-3">
              <span className="mb-1.5 block text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
                Parameter Interactions
              </span>
              {overfitting.interaction_effects.slice(0, 5).map((ie, i) => (
                <div key={i} className="mb-1 flex items-center gap-2">
                  <span className="font-mono text-[10px] text-(--color-brand)">{ie.param}</span>
                  <div className="h-2 flex-1 rounded-full bg-(--color-elevated)">
                    <div
                      className="h-full rounded-full opacity-70"
                      style={{
                        width: `${Math.min(ie.interaction_pct, 100)}%`,
                        backgroundColor:
                          ie.interaction_pct > 20
                            ? "var(--color-accent-warning)"
                            : "var(--color-accent)",
                      }}
                    />
                  </div>
                  <span className="min-w-[30px] text-right font-mono text-[9px] text-(--color-text-muted)">
                    {ie.interaction_pct}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Gap bars */}
        <div>
          {hasGaps ? (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
                Per-Period Gap
              </span>
              <div className="flex h-[80px] items-end gap-0.5 pb-3.5">
                {gapPeriods.map((g, i) => {
                  const absGap = Math.abs(g.gap ?? 0);
                  const pct = Math.min(absGap / 60, 1);
                  const color =
                    (g.gap ?? 0) > 40
                      ? "var(--color-accent-danger)"
                      : (g.gap ?? 0) > 15
                        ? "var(--color-accent-warning)"
                        : "var(--color-accent-success)";
                  return (
                    <div
                      key={i}
                      className="flex min-w-[8px] flex-1 flex-col items-center justify-end gap-0.5"
                      title={`${g.label}: train=${g.trainSharpe?.toFixed(2) ?? "?"} test=${g.testSharpe?.toFixed(2) ?? "?"} gap=${g.gap?.toFixed(0) ?? "?"}%`}
                    >
                      <div
                        className="w-full rounded-t-sm opacity-80 transition-all"
                        style={{
                          height: `${Math.max(pct * 60, 4)}px`,
                          backgroundColor: color,
                        }}
                      />
                      <span className="font-mono text-[8px] text-(--color-text-muted)">
                        {g.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center">
              <span className="text-[10px] text-(--color-text-muted)">No period gap data</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CiBarRow({
  label,
  ci,
  format,
}: {
  label: string;
  ci: { low?: number | null; high?: number | null; mean?: number | null };
  format: "number" | "pct";
}) {
  const low = ci.low ?? 0;
  const high = ci.high ?? 0;
  const mean = ci.mean ?? 0;
  const range = Math.max(Math.abs(high - low), 0.01);
  const absMax = Math.max(Math.abs(low), Math.abs(high), 0.01);

  const fmt = (v: number) => (format === "pct" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2));

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
          {label}
        </span>
        <span className="font-mono text-[10px] text-(--color-text-primary)">{fmt(mean)}</span>
      </div>
      <div className="relative h-[4px] w-full rounded-[2px] bg-(--color-elevated)">
        <div
          className="absolute top-0 h-full rounded-sm opacity-70"
          style={{
            left: `${((low - -absMax) / (2 * absMax)) * 100}%`,
            width: `${Math.max((range / (2 * absMax)) * 100, 4)}%`,
            backgroundColor:
              RISK_COLORS[
                ci.mean != null
                  ? ci.mean >= 0.5
                    ? "green"
                    : ci.mean >= 0
                      ? "yellow"
                      : "red"
                  : "green"
              ],
          }}
        />
        <div
          className="absolute top-0 h-full rounded-full bg-(--color-text-primary)"
          style={{ left: `${((mean - -absMax) / (2 * absMax)) * 100 - 2.5}%`, width: 5 }}
        />
      </div>
    </div>
  );
}
