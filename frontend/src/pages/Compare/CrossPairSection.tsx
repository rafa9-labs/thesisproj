import { useState } from "react";
import { GitCompare } from "lucide-react";
import { useHeatmap } from "@/api/queries";
import apiClient from "@/api/client";
import { CrossPairOverlayChart } from "@/components/charts/CrossPairOverlayChart";
import { EmptyState } from "@/components/shared/EmptyState";
import type { EquityPoint } from "@/api/schemas";

interface PairCurve {
  pair: string;
  data: EquityPoint[];
}

interface CrossPairCurvesResponse {
  model: string;
  curves: { model: string; pair: string; equity_curve: EquityPoint[] | null }[];
}

export function CrossPairSection() {
  const { data: heatmapData } = useHeatmap();
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedPairs, setSelectedPairs] = useState<string[]>([]);
  const [curves, setCurves] = useState<PairCurve[]>([]);
  const [loading, setLoading] = useState(false);

  const models = heatmapData?.models ?? [];
  const pairs = heatmapData?.pairs ?? [];
  const effectiveModel = selectedModel || models[0] || "";

  const fetchCurves = async () => {
    if (!effectiveModel || selectedPairs.length < 2) return;
    setLoading(true);
    try {
      const { data } = await apiClient.get<CrossPairCurvesResponse>("/backtest/cross-pair-curves", {
        params: { model: effectiveModel, pairs: selectedPairs.join(",") },
      });
      const loaded: PairCurve[] = (data.curves ?? [])
        .filter((c) => c.equity_curve && c.equity_curve.length > 0)
        .map((c) => ({ pair: c.pair, data: c.equity_curve! }));
      setCurves(loaded);
    } catch {
      setCurves([]);
    } finally {
      setLoading(false);
    }
  };

  const togglePair = (pair: string) => {
    setSelectedPairs((prev) =>
      prev.includes(pair) ? prev.filter((p) => p !== pair) : [...prev, pair],
    );
  };

  if (models.length === 0 || pairs.length < 2) {
    return (
      <EmptyState
        icon={<GitCompare size={48} />}
        title="Cross-pair comparison unavailable"
        description="Run backtests on at least 2 currency pairs to enable cross-pair equity overlay."
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h3
        className="text-xs font-semibold uppercase tracking-[0.08em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Cross-Pair Comparison
      </h3>

      <div className="flex flex-wrap items-start gap-4">
        <div className="flex flex-col gap-1">
          <label
            className="text-[10px] font-semibold uppercase tracking-wide"
            style={{ color: "var(--color-text-muted)" }}
          >
            Model
          </label>
          <select
            value={effectiveModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="rounded-md border px-2 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-surface)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-mono)",
              minWidth: 160,
            }}
          >
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label
            className="text-[10px] font-semibold uppercase tracking-wide"
            style={{ color: "var(--color-text-muted)" }}
          >
            Pairs (select 2+)
          </label>
          <div className="flex flex-wrap gap-1">
            {pairs.map((p) => (
              <button
                key={p}
                onClick={() => togglePair(p)}
                className="rounded-md border px-2 py-1 text-[10px] transition-colors"
                style={{
                  borderColor: selectedPairs.includes(p) ? "var(--color-accent)" : "var(--color-border)",
                  backgroundColor: selectedPairs.includes(p) ? "rgba(41,98,255,0.1)" : "var(--color-surface)",
                  color: selectedPairs.includes(p) ? "var(--color-accent)" : "var(--color-text-secondary)",
                  cursor: "pointer",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-end">
          <button
            onClick={fetchCurves}
            disabled={selectedPairs.length < 2 || loading}
            className="rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors"
            style={{
              borderColor: "var(--color-accent)",
              backgroundColor: selectedPairs.length < 2 || loading ? "var(--color-surface)" : "var(--color-accent)",
              color: selectedPairs.length < 2 || loading ? "var(--color-text-muted)" : "#fff",
              cursor: selectedPairs.length < 2 || loading ? "not-allowed" : "pointer",
              fontFamily: "var(--font-mono)",
            }}
          >
            {loading ? "Loading…" : "Compare"}
          </button>
        </div>
      </div>

      {(curves.length > 0 || loading) && (
        <CrossPairOverlayChart model={effectiveModel} curves={curves} />
      )}
    </div>
  );
}