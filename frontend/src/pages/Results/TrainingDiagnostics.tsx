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
import { BarChart3, Grid3X3, TrendingUp, Layers, Activity } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { formatMetric, formatPercent } from "@/lib/formatters";
import type { TrainingDiagnostics as TrainingDiagnosticsType } from "@/api/schemas";

interface Props {
  data: TrainingDiagnosticsType | null;
  modelName: string;
}

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

const METHOD_COLORS: Record<string, { bg: string; fg: string }> = {
  shap: { bg: "rgba(34,197,94,0.08)", fg: "#22c55e" },
  gain: { bg: "rgba(34,197,94,0.08)", fg: "#22c55e" },
  gradient: { bg: "rgba(6,182,212,0.08)", fg: "#06b6d4" },
  coefficients: { bg: "rgba(59,130,246,0.08)", fg: "#3b82f6" },
  permutation: { bg: "rgba(120,123,134,0.08)", fg: "#787b86" },
  mdi: { bg: "rgba(245,158,11,0.08)", fg: "#f59e0b" },
  submodel: { bg: "rgba(168,85,247,0.08)", fg: "#a855f7" },
  none: { bg: "rgba(239,68,68,0.05)", fg: "#ef4444" },
  unknown: { bg: "var(--color-elevated)", fg: "var(--color-text-muted)" },
};

const METHOD_LABELS: Record<string, string> = {
  shap: "TreeSHAP Shapley values — gold standard for tree models",
  gain: "XGBoost gain-based importance — biased toward high-cardinality splits",
  gradient: "Mean |gradient| via TF GradientTape — model-agnostic for deep networks",
  coefficients: "Standardized coefficient magnitude — check VIF for collinearity",
  permutation: "Permutation importance (3 repeats, 500 samples) — model-agnostic fallback",
  mdi: "Mean decrease impurity (Gini) — biased toward continuous features with many splits. Prefer SHAP.",
  submodel: "Delegated to sub-model in ensemble",
  none: "No importance available for this model type",
  unknown: "",
};

function FeatureFamilyStrip({ families }: { families: Record<string, number> }) {
  const entries = Object.entries(families).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, c]) => s + c, 0);

  return (
    <div className="mt-2">
      <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
        Feature Families
      </span>
      <div className="flex items-center gap-0.5 mt-1">
        {entries.map(([family, count]) => (
          <div
            key={family}
            className="flex-1 rounded-sm"
            style={{
              height: 8,
              minWidth: 4,
              backgroundColor: FAMILY_COLORS[family] ?? FAMILY_COLORS.other,
              opacity: 0.8,
              cursor: "default",
            }}
            title={`${family}: ${count}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 mt-1.5">
        {entries.map(([family, count]) => (
          <div key={family} className="flex items-center gap-1">
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{ backgroundColor: FAMILY_COLORS[family] ?? FAMILY_COLORS.other }}
            />
            <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
              {family} {count}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImportanceChart({ features }: { features: { feature: string; importance: number }[] }) {
  const sorted = useMemo(() => [...features].sort((a, b) => b.importance - a.importance).slice(0, 15), [features]);
  if (sorted.length === 0) return null;
  const maxImp = Math.max(...sorted.map((f) => f.importance), 0.001);
  return (
    <div style={{ height: Math.max(sorted.length * 28, 120) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={sorted} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" horizontal={false} />
          <XAxis type="number" tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }} />
          <YAxis
            type="category"
            dataKey="feature"
            width={90}
            tick={{ fill: "var(--color-text-secondary)", fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-elevated)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 6,
              fontSize: 11,
              fontFamily: "var(--font-mono)",
            }}
            formatter={(v: number) => [v.toFixed(4), "Importance"]}
          />
          <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
            {sorted.map((_, i) => (
              <Cell
                key={i}
                fill={i < 3 ? "var(--color-brand)" : i < 6 ? "rgba(0,229,255,0.5)" : "rgba(0,229,255,0.2)"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const rowMax = matrix.map((row) => Math.max(...row, 1));
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${labels.length + 1}, 48px)` }}>
        <div />
        {labels.map((l) => (
          <div key={l} className="text-[9px] uppercase tracking-wider text-center font-semibold" style={{ color: "var(--color-text-muted)" }}>
            {l}
          </div>
        ))}
        {matrix.map((row, i) => (
          <div key={`row-${i}`} className="contents">
            <div className="text-[9px] uppercase tracking-wider font-semibold flex items-center justify-end pr-1" style={{ color: "var(--color-text-muted)" }}>
              {labels[i]}
            </div>
            {row.map((val, j) => (
              <div
                key={`${i}-${j}`}
                className="rounded text-[11px] font-semibold flex items-center justify-center"
                style={{
                  fontFamily: "var(--font-mono)",
                  backgroundColor: i === j
                    ? `rgba(0,229,255,${0.15 + 0.55 * (val / rowMax[i])})`
                    : val > 0 ? "rgba(239,68,68,0.08)" : "var(--color-elevated)",
                  color: i === j ? "var(--color-brand)" : val > 0 ? "var(--color-text-secondary)" : "var(--color-text-muted)",
                  minWidth: 48,
                  minHeight: 32,
                }}
              >
                {val}
              </div>
            ))}
          </div>
        ))}
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
      <ChartCard title="" subtitle="" height={160}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 4, right: 8, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
            <XAxis
              dataKey="bin_center"
              tickFormatter={(v: number) => v.toFixed(2)}
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
            />
            <YAxis tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }} hide />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
              formatter={(v: number) => [v, "Predictions"]}
              labelFormatter={(v: number) => `Confidence: ${v.toFixed(2)}–${(v + 0.033).toFixed(2)}`}
            />
            <Bar dataKey="count" radius={[2, 2, 0, 0]} maxBarSize={24}>
              {bins.map((b) => (
                <Cell
                  key={b.bin_center}
                  fill={
                    b.bin_center >= 0.8
                      ? "var(--color-accent-success)"
                      : b.bin_center >= 0.65
                        ? "var(--color-brand)"
                        : "rgba(245,158,11,0.6)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <div className="flex items-center gap-4 mt-1.5">
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {pct07}% above 0.7 confidence
        </span>
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {pct08}% above 0.8 confidence
        </span>
      </div>
    </div>
  );
}

function ConfidenceBandTable({ bands }: { bands: { band_min: number; band_max: number; count: number; accuracy: number; mean_return: number }[] }) {
  if (!bands || bands.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[10px]" style={{ fontFamily: "var(--font-mono)" }}>
        <thead>
          <tr style={{ color: "var(--color-text-muted)" }}>
            <th className="text-left py-1 pr-2 uppercase tracking-wider">Confidence</th>
            <th className="text-right py-1 pr-2 uppercase tracking-wider">Trades</th>
            <th className="text-right py-1 pr-2 uppercase tracking-wider">Accuracy</th>
            <th className="text-right py-1 uppercase tracking-wider">Avg Return</th>
          </tr>
        </thead>
        <tbody>
          {bands.map((b, i) => {
            const accColor = b.accuracy >= 0.55 ? "var(--color-accent-success)" : b.accuracy >= 0.45 ? "var(--color-accent-warning)" : "var(--color-accent-danger)";
            const retColor = b.mean_return > 0 ? "var(--color-accent-success)" : b.mean_return < 0 ? "var(--color-accent-danger)" : "var(--color-text-muted)";
            return (
              <tr key={i} className="border-t" style={{ borderColor: "var(--color-glass-border)" }}>
                <td className="py-1 pr-2" style={{ color: "var(--color-text-primary)" }}>
                  {(b.band_min * 100).toFixed(0)}–{(b.band_max * 100).toFixed(0)}%
                </td>
                <td className="text-right py-1 pr-2" style={{ color: "var(--color-text-secondary)" }}>{b.count}</td>
                <td className="text-right py-1 pr-2" style={{ color: accColor }}>{(b.accuracy * 100).toFixed(1)}%</td>
                <td className="text-right py-1" style={{ color: retColor }}>{b.mean_return >= 0 ? "+" : ""}{(b.mean_return * 100).toFixed(3)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function TrainingDiagnosticsPanel({ data, modelName }: Props) {
  if (!data) {
    return (
      <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 size={16} style={{ color: "var(--color-text-muted)" }} />
          <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
            Training Diagnostics
          </h3>
        </div>
        <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          No diagnostics data available for {modelName}. Run a backtest to generate this report.
        </p>
      </div>
    );
  }

  const hasFeatures = data.feature_importance && data.feature_importance.length > 0;
  const hasHist = data.prediction_histogram && data.prediction_histogram.length > 0;
  const hasCm = data.confusion_matrix && data.confusion_matrix.matrix && data.confusion_matrix.matrix.length > 0;
  const hasBands = data.confidence_bands && data.confidence_bands.length > 0;

  return (
    <div className="rounded-xl border p-5" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
      <div className="flex items-center gap-2 mb-5">
        <BarChart3 size={16} style={{ color: "var(--color-brand)" }} />
        <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
          Training Diagnostics
        </h3>
        <span className="text-[10px] ml-auto" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {modelName}
        </span>
      </div>

      {hasFeatures && (
        <div className="mb-5">
          <div className="flex items-center gap-1.5 mb-2">
            <Layers size={12} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Feature Importance (Top 15)
            </span>
            {data.importance_method && (
              <span
                className="ml-1 rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider"
                style={{
                  backgroundColor: METHOD_COLORS[data.importance_method]?.bg ?? "var(--color-elevated)",
                  color: METHOD_COLORS[data.importance_method]?.fg ?? "var(--color-text-muted)",
                  border: `1px solid ${METHOD_COLORS[data.importance_method]?.fg ?? "var(--color-border)"}`,
                }}
                title={METHOD_LABELS[data.importance_method] ?? ""}
              >
                {data.importance_method}
              </span>
            )}
          </div>
          <div className="rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
            <ImportanceChart features={data.feature_importance!} />
          </div>
          {data.feature_families && Object.keys(data.feature_families).length > 0 && (
            <FeatureFamilyStrip families={data.feature_families} />
          )}
          {data.vif_warnings && data.vif_warnings.length > 0 && (
            <div className="mt-2 rounded p-2" style={{ backgroundColor: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.2)" }}>
              <div className="flex items-center gap-1.5 mb-1">
                <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-accent-warning)" }}>
                  VIF Warning
                </span>
                <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                  Collinear features (VIF &gt; 10)
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {data.vif_warnings.map((w) => (
                  <span
                    key={w.feature}
                    className="rounded px-1.5 py-0.5 text-[9px]"
                    style={{
                      backgroundColor: "rgba(245,158,11,0.15)",
                      color: "var(--color-accent-warning)",
                      fontFamily: "var(--font-mono)",
                    }}
                    title={`VIF = ${w.vif} — coefficient magnitudes may be unreliable for this feature`}
                  >
                    {w.feature} VIF={w.vif}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {hasHist && (
        <div className="mb-5">
          <div className="flex items-center gap-1.5 mb-2">
            <Activity size={12} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Prediction Confidence Distribution
            </span>
          </div>
          <div className="rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
            <PredictionHistogram bins={data.prediction_histogram!} />
          </div>
        </div>
      )}

      {hasCm && (
        <div className="mb-5">
          <div className="flex items-center gap-1.5 mb-2">
            <Grid3X3 size={12} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Confusion Matrix (Aggregated)
            </span>
          </div>
          <div className="rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
            <ConfusionMatrix matrix={data.confusion_matrix!.matrix!} labels={data.confusion_matrix!.labels} />
          </div>
        </div>
      )}

      {hasBands && (
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <TrendingUp size={12} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Confidence vs Accuracy
            </span>
          </div>
          <div className="rounded-lg p-3" style={{ backgroundColor: "var(--color-elevated)" }}>
            <ConfidenceBandTable bands={data.confidence_bands!} />
          </div>
        </div>
      )}

      {!hasFeatures && !hasHist && !hasCm && !hasBands && (
        <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          No diagnostics data available for {modelName}. Deep learning models (CNN, LSTM, Transformer) do not produce feature importance.
        </p>
      )}
    </div>
  );
}