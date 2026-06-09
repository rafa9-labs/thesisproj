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
        <span
          className="text-[10px] font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          Score
        </span>
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: RISK_COLORS[overfitting.risk_color] ?? "var(--color-text-muted)" }}
          />
          <span
            className="text-sm font-bold"
            style={{
              fontFamily: "JetBrains Mono, monospace",
              color: RISK_COLORS[overfitting.risk_color] ?? "var(--color-text-muted)",
            }}
          >
            {overfitting.overfit_score.toFixed(0)}
          </span>
        </div>
      </div>

      {overfitting.dsr_min_sharpe != null && (
        <div
          className="mb-4 rounded px-2 py-1 text-[10px]"
          style={{
            backgroundColor: "rgba(99,102,241,0.06)",
            border: "1px solid rgba(99,102,241,0.15)",
            color: "var(--color-text-secondary)",
            fontFamily: "var(--font-mono)",
          }}
        >
          Significance threshold: Sharpe &ge; {overfitting.dsr_min_sharpe.toFixed(2)} required
          for 95% confidence (adjusted for {overfitting.n_periods} OOS periods
          and {overfitting.n_signal_periods > 0 ? "HPO testing" : "multiple tests"}).
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
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
                CV fold &sigma;
              </span>
              <span className="text-xs font-mono" style={{ color: "var(--color-text-primary)" }}>
                {overfitting.cv_sharpe_std.toFixed(3)}
              </span>
            </div>
          )}
          {overfitting.interaction_effects && overfitting.interaction_effects.length > 0 && (
            <div className="mt-3">
              <span className="text-[10px] uppercase tracking-[0.06em] block mb-1.5" style={{ color: "var(--color-text-muted)" }}>
                Parameter Interactions
              </span>
              {overfitting.interaction_effects.slice(0, 5).map((ie, i) => (
                <div key={i} className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-mono" style={{ color: "var(--color-brand)" }}>
                    {ie.param}
                  </span>
                  <div className="flex-1 h-2 rounded-full" style={{ backgroundColor: "var(--color-elevated)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(ie.interaction_pct, 100)}%`,
                        backgroundColor: ie.interaction_pct > 20 ? "var(--color-accent-warning)" : "var(--color-accent)",
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <span className="text-[9px] font-mono" style={{ color: "var(--color-text-muted)", minWidth: 30, textAlign: "right" }}>
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
              <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
                Per-Period Gap
              </span>
              <div className="flex items-end gap-0.5 pb-3.5" style={{ height: 80 }}>
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
                      className="flex-1 flex flex-col items-center justify-end gap-0.5"
                      style={{ minWidth: 8 }}
                      title={`${g.label}: train=${g.trainSharpe?.toFixed(2) ?? "?"} test=${g.testSharpe?.toFixed(2) ?? "?"} gap=${g.gap?.toFixed(0) ?? "?"}%`}
                    >
                      <div
                        className="w-full rounded-t-sm transition-all"
                        style={{
                          height: `${Math.max(pct * 60, 4)}px`,
                          backgroundColor: color,
                          opacity: 0.8,
                        }}
                      />
                      <span className="text-[8px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                        {g.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full">
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                No period gap data
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CiBarRow({ label, ci, format }: { label: string; ci: { low?: number | null; high?: number | null; mean?: number | null }; format: "number" | "pct" }) {
  const low = ci.low ?? 0;
  const high = ci.high ?? 0;
  const mean = ci.mean ?? 0;
  const range = Math.max(Math.abs(high - low), 0.01);
  const absMax = Math.max(Math.abs(low), Math.abs(high), 0.01);

  const fmt = (v: number) => (format === "pct" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2));

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
          {label}
        </span>
        <span className="text-[10px] font-mono" style={{ color: "var(--color-text-primary)" }}>
          {fmt(mean)}
        </span>
      </div>
      <div className="relative w-full" style={{ height: 4, backgroundColor: "var(--color-elevated)", borderRadius: 2 }}>
        <div
          className="absolute top-0 h-full rounded-sm"
          style={{
            left: `${((low - (-absMax)) / (2 * absMax)) * 100}%`,
            width: `${Math.max((range / (2 * absMax)) * 100, 4)}%`,
            backgroundColor: RISK_COLORS[ci.mean != null ? (ci.mean >= 0.5 ? "green" : ci.mean >= 0 ? "yellow" : "red") : "green"],
            opacity: 0.7,
          }}
        />
        <div
          className="absolute top-0 h-full rounded-full"
          style={{
            left: `${((mean - (-absMax)) / (2 * absMax)) * 100 - 2.5}%`,
            width: 5,
            backgroundColor: "var(--color-text-primary)",
          }}
        />
      </div>
    </div>
  );
}

