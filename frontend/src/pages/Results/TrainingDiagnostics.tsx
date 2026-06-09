import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { BarChart3, Grid3X3, TrendingUp, Layers, Activity } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import type { TrainingDiagnostics as TrainingDiagnosticsType } from "@/api/schemas";

interface Props {
  data: TrainingDiagnosticsType | null;
  modelName: string;
}

// Muted institutional slate-blue for bars
const BAR_BASE = "#2962FF";
const BAR_TOP2 = "rgba(41,98,255,0.75)";
const BAR_REST = "rgba(41,98,255,0.40)";

const METHOD_COLORS: Record<string, { bg: string; fg: string }> = {
  shap:         { bg: "rgba(34,197,94,0.08)",   fg: "#22c55e" },
  gain:         { bg: "rgba(34,197,94,0.08)",   fg: "#22c55e" },
  gradient:     { bg: "rgba(6,182,212,0.08)",   fg: "#06b6d4" },
  coefficients: { bg: "rgba(41,98,255,0.10)",   fg: "#4f83ff" },
  permutation:  { bg: "rgba(120,123,134,0.08)", fg: "#787b86" },
  mdi:          { bg: "rgba(245,158,11,0.08)",  fg: "#f59e0b" },
  submodel:     { bg: "rgba(168,85,247,0.08)",  fg: "#a855f7" },
  none:         { bg: "rgba(239,68,68,0.05)",   fg: "var(--color-event-high)" },
  unknown:      { bg: "var(--color-surface)",                fg: "#787b86" },
};

const FAMILY_COLORS: Record<string, string> = {
  trend: "#22c55e", momentum: "#f97316", volatility: "#6366f1",
  composite: "#a855f7", regime: "#ec4899", returns: "#14b8a6",
  news: "#eab308", time: "#787b86", rolling: "#3b82f6", other: "#6b7280",
};

function ImportanceChart({ features }: { features: { feature: string; importance: number }[] }) {
  const sorted = useMemo(() => [...features].sort((a, b) => b.importance - a.importance).slice(0, 15), [features]);
  if (sorted.length === 0) return null;

  return (
    // Thin compact bars — h-2 equivalent height per bar
    <div style={{ height: Math.max(sorted.length * 22, 100) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ left: 4, right: 8, top: 2, bottom: 2 }}
          barCategoryGap="35%"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "JetBrains Mono" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={88}
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "Inter" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-surface)",
              border: "1px solid #2A2E39",
              borderRadius: 6,
              fontSize: 11,
              fontFamily: "JetBrains Mono",
            }}
            formatter={(v: number) => [v.toFixed(4), "Importance"]}
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
          />
          <Bar dataKey="importance" radius={[0, 2, 2, 0]} maxBarSize={8}>
            {sorted.map((_, i) => (
              <Cell
                key={i}
                fill={i === 0 ? BAR_BASE : i < 3 ? BAR_TOP2 : BAR_REST}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Tight, borderless 3×3 heatmap for confusion matrix */
function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const rowTotals = matrix.map((row) => Math.max(row.reduce((s, v) => s + v, 0), 1));
  return (
    <div className="inline-flex flex-col gap-0.5">
      {/* Column headers */}
      <div className="flex gap-0.5 ml-8">
        {labels.map((l) => (
          <div
            key={l}
            className="flex items-center justify-center text-center"
            style={{
              width: 40,
              height: 18,
              fontSize: 9,
              fontFamily: "Inter, sans-serif",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: "var(--color-text-muted)",
            }}
          >
            {l}
          </div>
        ))}
      </div>
      {/* Rows */}
      {matrix.map((row, ri) => (
        <div key={ri} className="flex items-center gap-0.5">
          {/* Row label */}
          <div
            className="flex items-center justify-end"
            style={{
              width: 30,
              fontSize: 9,
              fontFamily: "Inter, sans-serif",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: "var(--color-text-muted)",
              paddingRight: 4,
            }}
          >
            {labels[ri]}
          </div>
          {/* Cells */}
          {row.map((val, ci) => {
            const intensity = val / rowTotals[ri];
            const isDiag = ri === ci;
            const bg = isDiag
              ? `rgba(41,98,255,${0.12 + 0.55 * intensity})`
              : val > 0
                ? `rgba(242,54,69,${0.06 + 0.22 * intensity})`
                : "var(--color-app)";
            const tc = isDiag
              ? `rgba(79,131,255,${0.5 + 0.5 * intensity})`
              : val > 0
                ? "var(--color-accent-danger)"
                : "var(--color-text-dim)";
            return (
              <div
                key={ci}
                className="flex items-center justify-center font-semibold"
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 3,
                  backgroundColor: bg,
                  color: tc,
                  fontSize: 12,
                  fontFamily: "JetBrains Mono, monospace",
                  transition: "background-color 0.15s",
                }}
                title={`Predicted ${labels[ci]}, Actual ${labels[ri]}: ${val}`}
              >
                {val}
              </div>
            );
          })}
        </div>
      ))}
      {/* Axis labels */}
      <div
        className="flex items-center gap-2 mt-1"
        style={{ fontSize: 9, color: "var(--color-text-dim)", fontFamily: "Inter, sans-serif" }}
      >
        <span className="ml-8 uppercase tracking-wider">Predicted</span>
        <span style={{ color: "var(--color-glass-border)" }}>|</span>
        <span className="uppercase tracking-wider">Row = Actual</span>
      </div>
    </div>
  );
}

function PredictionHistogram({ bins }: { bins: { bin_start: number; bin_end: number; bin_center: number; count: number }[] }) {
  if (!bins || bins.length === 0) return null;
  const total = bins.reduce((s, b) => s + b.count, 0);
  const above07 = bins.filter((b) => b.bin_center >= 0.7).reduce((s, b) => s + b.count, 0);
  const above08 = bins.filter((b) => b.bin_center >= 0.8).reduce((s, b) => s + b.count, 0);
  const pct07 = total > 0 ? ((above07 / total) * 100).toFixed(0) : "0";
  const pct08 = total > 0 ? ((above08 / total) * 100).toFixed(0) : "0";

  return (
    <div>
      <ChartCard title="" subtitle="" height={130}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 2, right: 6, left: -8, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
            <XAxis
              dataKey="bin_center"
              tickFormatter={(v: number) => v.toFixed(2)}
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "JetBrains Mono" }}
            />
            <YAxis tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "JetBrains Mono" }} hide />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-surface)",
                border: "1px solid #2A2E39",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "JetBrains Mono",
              }}
              formatter={(v: number) => [v, "Predictions"]}
              labelFormatter={(v: number) => `Conf: ${v.toFixed(2)}–${(v + 0.033).toFixed(2)}`}
            />
            <Bar dataKey="count" radius={[2, 2, 0, 0]} maxBarSize={20}>
              {bins.map((b) => (
                <Cell
                  key={b.bin_center}
                  fill={b.bin_center >= 0.8 ? "var(--color-accent-success)" : b.bin_center >= 0.65 ? BAR_TOP2 : BAR_REST}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <div className="flex items-center gap-4 mt-1">
        <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "JetBrains Mono" }}>
          {pct07}% above 0.7
        </span>
        <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "JetBrains Mono" }}>
          {pct08}% above 0.8
        </span>
      </div>
    </div>
  );
}

function ConfidenceBandTable({ bands }: { bands: { band_min: number; band_max: number; count: number; accuracy: number; mean_return: number }[] }) {
  if (!bands || bands.length === 0) return null;
  return (
    <table className="w-full" style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10 }}>
      <thead>
        <tr style={{ color: "var(--color-text-muted)" }}>
          <th className="text-left py-1 pr-2" style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Conf.</th>
          <th className="text-right py-1 pr-2" style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Trades</th>
          <th className="text-right py-1 pr-2" style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Acc.</th>
          <th className="text-right py-1" style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Ret.</th>
        </tr>
      </thead>
      <tbody>
        {bands.map((b, i) => {
          const ac = b.accuracy >= 0.55 ? "var(--color-accent-success)" : b.accuracy >= 0.45 ? "var(--color-accent-warning)" : "var(--color-accent-danger)";
          const rc = b.mean_return > 0 ? "var(--color-accent-success)" : b.mean_return < 0 ? "var(--color-accent-danger)" : "var(--color-text-muted)";
          return (
            <tr key={i} style={{ borderTop: "1px solid #2A2E39" }}>
              <td className="py-0.5 pr-2" style={{ color: "var(--color-text-primary)" }}>
                {(b.band_min * 100).toFixed(0)}&ndash;{(b.band_max * 100).toFixed(0)}%
              </td>
              <td className="text-right py-0.5 pr-2" style={{ color: "var(--color-text-muted)" }}>{b.count}</td>
              <td className="text-right py-0.5 pr-2" style={{ color: ac }}>{(b.accuracy * 100).toFixed(1)}%</td>
              <td className="text-right py-0.5" style={{ color: rc }}>
                {b.mean_return >= 0 ? "+" : ""}{(b.mean_return * 100).toFixed(3)}%
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function TrainingDiagnosticsPanel({ data, modelName }: Props) {
  const [vifDismissed, setVifDismissed] = useState(false);

  if (!data) {
    return (
      <div
        className="rounded-sm border p-4"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-surface)" }}
      >
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 size={14} style={{ color: "var(--color-text-muted)" }} />
          <span className="text-[11px] font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
            Training Diagnostics
          </span>
        </div>
        <p className="text-[11px]" style={{ color: "var(--color-text-dim)" }}>
          No diagnostics available for {modelName}.
        </p>
      </div>
    );
  }

  const hasFeatures = data.feature_importance && data.feature_importance.length > 0;
  const hasHist = data.prediction_histogram && data.prediction_histogram.length > 0;
  const hasCm = data.confusion_matrix && data.confusion_matrix.matrix && data.confusion_matrix.matrix.length > 0;
  const hasBands = data.confidence_bands && data.confidence_bands.length > 0;
  const hasVif = !vifDismissed && data.vif_warnings && data.vif_warnings.length > 0;
  const method = data.importance_method;

  return (
    <div className="flex flex-col gap-3">
      {/* ── VIF warning — dismissible panel ──────────────────────── */}
      {hasVif && (
        <div
          className="flex items-start justify-between rounded-md px-3 py-2"
          style={{ backgroundColor: "rgba(245,158,11,0.07)", border: "1px solid rgba(245,158,11,0.2)" }}
        >
          <div className="flex items-start gap-2 flex-1 min-w-0">
            <span style={{ color: "var(--color-accent-warning)", fontSize: 13, lineHeight: 1 }}>&#9888;</span>
            <div className="flex flex-col gap-1 min-w-0">
              <span
                className="text-[10px] font-semibold uppercase tracking-wider"
                style={{ color: "var(--color-accent-warning)" }}
              >
                Collinear Features (VIF &gt; 10)
              </span>
              <div className="flex flex-wrap gap-1">
                {data.vif_warnings!.map((w) => (
                  <span
                    key={w.feature}
                    className="rounded px-1.5 py-0.5 text-[9px]"
                    style={{
                      backgroundColor: "rgba(245,158,11,0.12)",
                      color: "var(--color-accent-warning)",
                      fontFamily: "JetBrains Mono, monospace",
                    }}
                    title={`VIF = ${w.vif}`}
                  >
                    {w.feature} {w.vif}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <button
            onClick={() => setVifDismissed(true)}
            className="ml-2 flex-shrink-0"
            style={{ color: "var(--color-text-muted)", cursor: "pointer", background: "none", border: "none", fontSize: 12 }}
          >
            &#10005;
          </button>
        </div>
      )}

      {/* ── Feature Importance ───────────────────────────────────── */}
      {hasFeatures && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <Layers size={11} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Feature Importance (Top 15)
            </span>
            {method && (
              <span
                className="rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider"
                style={{
                  backgroundColor: METHOD_COLORS[method]?.bg ?? "var(--color-surface)",
                  color: METHOD_COLORS[method]?.fg ?? "var(--color-text-muted)",
                  border: `1px solid ${METHOD_COLORS[method]?.fg ?? "var(--color-glass-border)"}`,
                }}
              >
                {method}
              </span>
            )}
          </div>
          <ImportanceChart features={data.feature_importance!} />
          {/* Feature family strip */}
          {data.feature_families && Object.keys(data.feature_families).length > 0 && (
            <div className="mt-2">
              <div className="flex items-center gap-0.5 h-1.5">
                {Object.entries(data.feature_families)
                  .sort((a, b) => b[1] - a[1])
                  .map(([fam, cnt]) => (
                    <div
                      key={fam}
                      className="flex-1 rounded-sm h-full"
                      style={{ backgroundColor: FAMILY_COLORS[fam] ?? FAMILY_COLORS.other, opacity: 0.75, minWidth: 3 }}
                      title={`${fam}: ${cnt}`}
                    />
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Prediction Confidence Histogram ──────────────────────── */}
      {hasHist && (
        <div>
          <div className="flex items-center gap-1.5 mb-1">
            <Activity size={11} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Prediction Confidence
            </span>
          </div>
          <PredictionHistogram bins={data.prediction_histogram!} />
        </div>
      )}

      {/* ── Confusion Matrix ──────────────────────────────────────── */}
      {hasCm && (
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <Grid3X3 size={11} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Confusion Matrix
            </span>
          </div>
          <ConfusionMatrix
            matrix={data.confusion_matrix!.matrix!}
            labels={data.confusion_matrix!.labels}
          />
        </div>
      )}

      {/* ── Confidence vs Accuracy ────────────────────────────────── */}
      {hasBands && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <TrendingUp size={11} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Confidence vs Accuracy
            </span>
          </div>
          <ConfidenceBandTable bands={data.confidence_bands!} />
        </div>
      )}

      {!hasFeatures && !hasHist && !hasCm && !hasBands && (
        <p className="text-[11px]" style={{ color: "var(--color-text-dim)" }}>
          No diagnostics available for {modelName}. Deep learning models do not produce feature importance.
        </p>
      )}
    </div>
  );
}
