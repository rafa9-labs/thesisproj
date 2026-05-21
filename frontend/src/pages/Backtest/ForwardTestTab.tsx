import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Play, TriangleAlert } from "lucide-react";
import apiClient from "@/api/client";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";

interface DeployedModel {
  id: string;
  model_type: string;
  best_sharpe: number | null;
  best_return: number | null;
  created_at: string;
  status: string;
  tags: string[];
  missing_on_disk?: boolean;
}

const SIZING_OPTIONS = [
  { value: "fixed", label: "Fixed (1 lot)" },
  { value: "fixed_fractional", label: "Fractional (% of equity)" },
  { value: "kelly", label: "Kelly Criterion" },
  { value: "atr", label: "ATR Volatility" },
  { value: "vol_target", label: "Vol Target" },
];

export function ForwardTestTab() {
  const navigate = useNavigate();
  const pair = useBacktestStore((s) => s.pair);
  const timeframe = useBacktestStore((s) => s.timeframe);
  const startDate = useBacktestStore((s) => s.startDate);
  const endDate = useBacktestStore((s) => s.endDate);
  const setField = useBacktestStore((s) => s.setField);
  const startJob = useJobStore((s) => s.startJob);

  const [selectedModel, setSelectedModel] = useState<string>("");
  const [posSizing, setPosSizing] = useState("fixed");
  const [useCosts, setUseCosts] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: models, isLoading: loadingModels } = useQuery<DeployedModel[]>({
    queryKey: ["deployed-models"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: DeployedModel[] }>("/models/deployed");
      return data.models;
    },
    refetchOnMount: true,
  });

  const activeModels = (models ?? []).filter(
    (m) => !m.missing_on_disk
  );

  const handleRun = useCallback(async () => {
    if (!selectedModel) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const { data } = await apiClient.post<{ job_id: string; status: string }>(
        `/models/${selectedModel}/forward-test`,
        {
          model_id: selectedModel,
          pair,
          timeframe,
          start_date: startDate || undefined,
          end_date: endDate || undefined,
          position_sizing: posSizing,
          trading_costs: useCosts,
        }
      );
      startJob(data.job_id, pair, [selectedModel]);
      navigate("/monitor");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Failed to start forward test");
    } finally {
      setIsSubmitting(false);
    }
  }, [selectedModel, pair, timeframe, startDate, endDate, posSizing, useCosts, navigate, startJob]);

  const selectedName = activeModels.find((m) => m.id === selectedModel);

  return (
    <div className="flex flex-col gap-5 pt-1">
      <div>
        <h3 className="text-sm font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-primary)" }}>
          Forward Test
        </h3>
        <p className="text-[11px] mt-1" style={{ color: "var(--color-text-muted)" }}>
          Test a saved model on any date range without retraining. The model predicts every bar — costs and sizing are applied, equity is tracked.
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-lg border px-3 py-2" style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(239,68,68,0.05)" }}>
          <TriangleAlert size={14} style={{ color: "var(--color-accent-danger)" }} />
          <span className="text-xs" style={{ color: "var(--color-accent-danger)" }}>{error}</span>
        </div>
      )}

      {/* Model selection */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
          Model
        </span>
        {loadingModels ? (
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading models...</span>
        ) : activeModels.length === 0 ? (
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            No deployed models. Run a backtest first to save a model.
          </span>
        ) : (
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="rounded-md border px-3 py-2 text-xs transition focus:outline-none"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          >
            <option value="">Select a saved model...</option>
            {activeModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.model_type} — SR: {m.best_sharpe != null ? (m.best_sharpe >= 0 ? "+" : "") + m.best_sharpe.toFixed(2) : "—"}
                {m.status === "active" ? " (ACTIVE)" : ""}
                {m.tags.length > 0 ? ` [${m.tags.join(",")}]` : ""}
              </option>
            ))}
          </select>
        )}
      </div>

      {selectedName && (
        <div className="flex flex-wrap items-center gap-2 text-[10px]" style={{ color: "var(--color-text-muted)" }}>
          <span>Type: <span style={{ color: "var(--color-text-secondary)" }}>{selectedName.model_type}</span></span>
          <span>|</span>
          <span>SR: <span style={{ color: "var(--color-accent-success)", fontFamily: "var(--font-mono)" }}>{selectedName.best_sharpe != null ? (selectedName.best_sharpe >= 0 ? "+" : "") + selectedName.best_sharpe.toFixed(2) : "—"}</span></span>
          <span>|</span>
          <span>Return: <span style={{ fontFamily: "var(--font-mono)" }}>{selectedName.best_return != null ? (selectedName.best_return >= 0 ? "+" : "") + selectedName.best_return.toFixed(1) + "%" : "—"}</span></span>
          <span>|</span>
          <span>{selectedName.created_at?.slice(0, 10)}</span>
        </div>
      )}

      {/* Pair / Timeframe */}
      <div className="flex items-center gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Pair</span>
          <input
            value={pair}
            onChange={(e) => setField("pair", e.target.value)}
            className="rounded-md border px-3 py-2 text-xs transition focus:outline-none w-[120px]"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Timeframe</span>
          {(["M30", "H1", "H4"] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setField("timeframe", tf)}
              className="rounded-md border px-3 py-1 text-[10px] font-medium uppercase transition-all"
              style={{
                borderColor: timeframe === tf ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: timeframe === tf ? "var(--color-brand-glow)" : "var(--color-glass)",
                color: timeframe === tf ? "var(--color-brand)" : "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Date range */}
      <div className="flex items-center gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Start Date</span>
          <input
            type="date"
            value={startDate ?? ""}
            onChange={(e) => setField("startDate", e.target.value)}
            className="rounded-md border px-3 py-2 text-xs transition focus:outline-none"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>End Date</span>
          <input
            type="date"
            value={endDate ?? ""}
            onChange={(e) => setField("endDate", e.target.value)}
            className="rounded-md border px-3 py-2 text-xs transition focus:outline-none"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          />
        </div>
      </div>

      {/* Position sizing */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Position Sizing</span>
        <div className="flex flex-wrap gap-2">
          {SIZING_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setPosSizing(opt.value)}
              className="rounded-md border px-3 py-1.5 text-[10px] font-medium transition-all"
              style={{
                borderColor: posSizing === opt.value ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: posSizing === opt.value ? "var(--color-brand-glow)" : "var(--color-glass)",
                color: posSizing === opt.value ? "var(--color-brand)" : "var(--color-text-muted)",
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Trading costs */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={useCosts}
          onChange={(e) => setUseCosts(e.target.checked)}
          className="rounded"
        />
        <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
          Apply trading costs (spread + slippage)
        </span>
      </label>

      {/* Run button */}
      <button
        onClick={handleRun}
        disabled={!selectedModel || isSubmitting}
        className="flex items-center gap-2 rounded-md px-5 py-2.5 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
        style={{
          backgroundColor: "var(--color-accent-success)",
          color: "var(--color-text-inverse)",
          alignSelf: "flex-start",
        }}
      >
        <Play size={14} />
        {isSubmitting ? "Submitting..." : "Run Forward Test"}
      </button>
    </div>
  );
}
