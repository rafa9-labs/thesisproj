import { useMemo } from "react";
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
import { TrendingUp, Activity, Grid3X3, Layers } from "lucide-react";
import type { TrainingDiagnostics as TrainingDiagnosticsType } from "@/api/schemas";

interface SectionProps {
  data: TrainingDiagnosticsType | null;
  modelName: string;
}

// Muted institutional slate-blue for bars
const BAR_TOP2 = "rgba(41,98,255,0.75)";
const BAR_REST = "rgba(41,98,255,0.40)";

const METHOD_COLORS: Record<string, { bg: string; fg: string }> = {
  shap: { bg: "rgba(34,197,94,0.08)", fg: "#22c55e" },
  gain: { bg: "rgba(34,197,94,0.08)", fg: "#22c55e" },
  gradient: { bg: "rgba(6,182,212,0.08)", fg: "#06b6d4" },
  coefficients: { bg: "rgba(41,98,255,0.10)", fg: "#4f83ff" },
  permutation: { bg: "rgba(120,123,134,0.08)", fg: "#787b86" },
  mdi: { bg: "rgba(245,158,11,0.08)", fg: "#f59e0b" },
  submodel: { bg: "rgba(168,85,247,0.08)", fg: "#a855f7" },
  none: { bg: "rgba(239,68,68,0.05)", fg: "var(--color-event-high)" },
  unknown: { bg: "var(--color-surface)", fg: "#787b86" },
};

const FAMILY_COLORS: Record<string, string> = {
  trend: "#22c55e",
  momentum: "#f97316",
  volatility: "#6366f1",
  composite: "#a855f7",
  regime: "#ec4899",
  returns: "#14b8a6",
  news: "#eab308",
  time: "#787b86",
  rolling: "#3b82f6",
  other: "#6b7280",
};

function ConfusionMatrix({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const rowTotals = matrix.map((row) => Math.max(row.reduce((s, v) => s + v, 0), 1));
  return (
    <div className="flex flex-col gap-0.5">
      <div className="ml-8 flex gap-0.5">
        {labels.map((l) => (
          <div
            key={l}
            className="flex h-[18px] w-[48px] items-center justify-center text-center font-sans text-[9px] font-bold tracking-[0.06em] text-(--color-text-muted) uppercase"
          >
            {l}
          </div>
        ))}
      </div>
      {matrix.map((row, ri) => (
        <div key={ri} className="flex items-center gap-0.5">
          <div className="flex w-[30px] items-center justify-end pr-1 font-sans text-[9px] font-bold tracking-[0.06em] text-(--color-text-muted) uppercase">
            {labels[ri]}
          </div>
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
                className="flex h-[48px] w-[48px] items-center justify-center rounded-[3px] font-mono text-xs font-semibold"
                style={{ backgroundColor: bg, color: tc, transition: "background-color 0.15s" }}
                title={`Predicted ${labels[ci]}, Actual ${labels[ri]}: ${val}`}
              >
                {val}
              </div>
            );
          })}
        </div>
      ))}
      <div className="mt-1 flex items-center gap-2 font-sans text-[9px] text-(--color-text-dim)">
        <span className="ml-8 tracking-wider uppercase">Predicted</span>
        <span className="text-(--color-glass-border)">|</span>
        <span className="tracking-wider uppercase">Row = Actual</span>
      </div>
    </div>
  );
}

function PredictionHistogram({
  bins,
}: {
  bins: { bin_start: number; bin_end: number; bin_center: number; count: number }[];
}) {
  if (!bins || bins.length === 0) return null;

  return (
    <div className="w-full flex-1">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={bins} margin={{ top: 2, right: 6, left: -8, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
          <XAxis
            dataKey="bin_center"
            tickFormatter={(v: number) => v.toFixed(2)}
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
          />
          <YAxis hide />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-elevated)",
              borderRadius: 6,
              fontSize: 11,
              fontFamily: "var(--font-mono)",
            }}
            formatter={(v) => [v, "Predictions"]}
            labelFormatter={(v) => { const n = typeof v === "number" ? v : 0; return `Conf: ${n.toFixed(2)}–${(n + 0.033).toFixed(2)}`; }}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]} maxBarSize={22}>
            {bins.map((b) => (
              <Cell
                key={b.bin_center}
                fill={
                  b.bin_center >= 0.8
                    ? "var(--color-accent-success)"
                    : b.bin_center >= 0.65
                      ? BAR_TOP2
                      : BAR_REST
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ConfidenceBandTable({
  bands,
}: {
  bands: {
    band_min: number;
    band_max: number;
    count: number;
    accuracy: number;
    mean_return: number;
  }[];
}) {
  if (!bands || bands.length === 0) return null;
  return (
    <table className="w-full font-mono text-[10px]">
      <thead>
        <tr className="text-(--color-text-muted)">
          <th className="py-1 pr-2 text-left text-[9px] font-semibold tracking-[0.06em] uppercase">
            Conf.
          </th>
          <th className="py-1 pr-2 text-right text-[9px] font-semibold tracking-[0.06em] uppercase">
            Trades
          </th>
          <th className="py-1 pr-2 text-right text-[9px] font-semibold tracking-[0.06em] uppercase">
            Acc.
          </th>
          <th className="py-1 text-right text-[9px] font-semibold tracking-[0.06em] uppercase">
            Ret.
          </th>
        </tr>
      </thead>
      <tbody>
        {bands.map((b, i) => {
          const ac =
            b.accuracy >= 0.55
              ? "var(--color-accent-success)"
              : b.accuracy >= 0.45
                ? "var(--color-accent-warning)"
                : "var(--color-accent-danger)";
          const rc =
            b.mean_return > 0
              ? "var(--color-accent-success)"
              : b.mean_return < 0
                ? "var(--color-accent-danger)"
                : "var(--color-text-muted)";
          return (
            <tr key={i} className="border-t border-(--color-elevated)">
              <td className="py-0.5 pr-2 text-(--color-text-primary)">
                {(b.band_min * 100).toFixed(0)}&ndash;{(b.band_max * 100).toFixed(0)}%
              </td>
              <td className="py-0.5 pr-2 text-right text-(--color-text-muted)">{b.count}</td>
              <td className="py-0.5 pr-2 text-right" style={{ color: ac }}>
                {(b.accuracy * 100).toFixed(1)}%
              </td>
              <td className="py-0.5 text-right" style={{ color: rc }}>
                {b.mean_return >= 0 ? "+" : ""}
                {(b.mean_return * 100).toFixed(3)}%
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Section 3: Feature importance high-density list + inline collinear (VIF) warnings. */
export function FeatureImportanceSection({ data, modelName }: SectionProps) {
  const features = useMemo(
    () =>
      data
        ? [...(data.feature_importance ?? [])]
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 20)
        : [],
    [data],
  );
  const vifMap = useMemo(
    () => new Map((data?.vif_warnings ?? []).map((w) => [w.feature, w.vif])),
    [data],
  );

  if (!data) {
    return (
      <p className="text-[11px] text-(--color-text-dim)">
        No training diagnostics for {modelName}.
      </p>
    );
  }

  const maxImp = features.length > 0 ? features[0].importance : 1;
  const method = data.importance_method;
  const hasFeatures = features.length > 0;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Layers size={11} className="text-(--color-text-muted)" />
        <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
          Feature Importance (Top {features.length})
        </span>
        {method && (
          <span
            className="rounded px-1.5 py-0.5 text-[9px] font-semibold tracking-wider uppercase"
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

      {hasFeatures ? (
        <div className="flex flex-col">
          {features.map((f) => {
            const hasVif = vifMap.has(f.feature);
            const widthPct = Math.max((f.importance / maxImp) * 100, 1);
            return (
              <div
                key={f.feature}
                className="flex items-center gap-2 border-b border-[rgba(255,255,255,0.03)] py-1 last:border-b-0"
              >
                <div className="flex min-w-0 w-[140px] items-center gap-1">
                  <span className="truncate font-mono text-[11px] text-(--color-text-primary)">
                    {f.feature}
                  </span>
                  {hasVif && (
                    <span
                      className="shrink-0 cursor-help text-[12px] leading-none text-(--color-accent-warning)"
                      title={`VIF = ${vifMap.get(f.feature)}`}
                    >
                      &#9888;
                    </span>
                  )}
                </div>
                <div className="h-[5px] flex-1 rounded-full bg-(--color-elevated)">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${widthPct}%`, backgroundColor: "#2962FF" }}
                  />
                </div>
                <span className="w-14 shrink-0 text-right font-mono text-[10px] text-(--color-text-secondary) tabular-nums">
                  {f.importance.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-[11px] text-(--color-text-dim)">
          No feature importance for {modelName}. Deep models do not produce feature importance.
        </p>
      )}

      {data.feature_families && Object.keys(data.feature_families).length > 0 && (
        <div className="mt-1">
          <div className="flex h-1.5 items-center gap-0.5">
            {Object.entries(data.feature_families)
              .sort((a, b) => b[1] - a[1])
              .map(([fam, cnt]) => (
                <div
                  key={fam}
                  className="h-full min-w-[3px] flex-1 rounded-sm opacity-75"
                  style={{ backgroundColor: FAMILY_COLORS[fam] ?? FAMILY_COLORS.other }}
                  title={`${fam}: ${cnt}`}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Section 4: Prediction Confidence Histogram | Confusion Matrix | Confidence vs Accuracy. */
export function PredictionConfidenceSection({ data }: Omit<SectionProps, "modelName">) {
  if (!data) {
    return (
      <p className="text-[11px] text-(--color-text-dim)">No prediction diagnostics available</p>
    );
  }

  const hasHist = data.prediction_histogram && data.prediction_histogram.length > 0;
  const hasCm =
    data.confusion_matrix && data.confusion_matrix.matrix && data.confusion_matrix.matrix.length > 0;
  const hasBands = data.confidence_bands && data.confidence_bands.length > 0;

  const bins = data.prediction_histogram ?? [];
  const total = bins.reduce((s, b) => s + b.count, 0);
  const above07 = bins.filter((b) => b.bin_center >= 0.7).reduce((s, b) => s + b.count, 0);
  const above08 = bins.filter((b) => b.bin_center >= 0.8).reduce((s, b) => s + b.count, 0);
  const pct07 = total > 0 ? ((above07 / total) * 100).toFixed(0) : "0";
  const pct08 = total > 0 ? ((above08 / total) * 100).toFixed(0) : "0";

  if (!hasHist && !hasCm && !hasBands) {
    return (
      <p className="text-[11px] text-(--color-text-dim)">
        No prediction confidence data captured for this run
      </p>
    );
  }

  return (
    <div className="flex flex-col divide-y divide-(--color-glass-border) md:flex-row md:divide-x md:divide-y-0">
      {/* Prediction Confidence Histogram */}
      <div className="flex flex-1 flex-col gap-1 px-2 py-1">
        <div className="flex items-center justify-between self-stretch">
          <div className="flex items-center gap-1.5">
            <Activity size={11} className="text-(--color-text-muted)" />
            <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
              Prediction Confidence
            </span>
          </div>
          {hasHist && (
            <span className="font-mono text-[9px] text-(--color-text-muted)">
              {pct07}% above 0.7 · {pct08}% above 0.8
            </span>
          )}
        </div>
        <div className="flex w-full flex-1 min-w-0">
          {hasHist ? (
            <PredictionHistogram bins={data.prediction_histogram!} />
          ) : (
            <span className="text-[11px] text-(--color-text-dim)">No histogram</span>
          )}
        </div>
      </div>

      {/* Confusion Matrix — inline visual heatmap */}
      <div className="flex flex-1 flex-col gap-1 px-2 py-1">
        <div className="flex items-center gap-1.5 self-start">
          <Grid3X3 size={11} className="text-(--color-text-muted)" />
          <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
            Confusion Matrix
          </span>
        </div>
        <div className="flex w-full flex-1 items-center justify-center min-w-0">
          {hasCm ? (
            <ConfusionMatrix
              matrix={data.confusion_matrix!.matrix!}
              labels={data.confusion_matrix!.labels}
            />
          ) : (
            <span className="text-[11px] text-(--color-text-dim)">No matrix</span>
          )}
        </div>
      </div>

      {/* Confidence vs Accuracy */}
      <div className="flex flex-1 flex-col gap-1 px-2 py-1">
        <div className="flex items-center gap-1.5 self-start">
          <TrendingUp size={11} className="text-(--color-text-muted)" />
          <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
            Confidence vs Accuracy
          </span>
        </div>
        <div className="flex w-full flex-1 min-w-0">
          {hasBands ? (
            <ConfidenceBandTable bands={data.confidence_bands!} />
          ) : (
            <span className="text-[11px] text-(--color-text-dim)">No band data</span>
          )}
        </div>
      </div>
    </div>
  );
}
