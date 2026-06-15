import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Play, TriangleAlert, ChevronDown } from "lucide-react";
import apiClient from "@/api/client";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";

/* ─────────────────────── shared style tokens ─────────────────────── */

/* ─────────────────────── sizing options ─────────────────────── */
const SIZING_OPTIONS = [
  { value: "fixed", label: "Fixed Lot", hint: "Constant 1-lot size every trade" },
  { value: "fixed_fractional", label: "Fractional %", hint: "Risk a fixed % of equity" },
  { value: "kelly", label: "Kelly Criterion", hint: "Optimal growth fraction" },
  { value: "atr", label: "ATR Volatility", hint: "Size adjusted to current volatility" },
  { value: "vol_target", label: "Vol Target", hint: "Target a fixed realised-vol level" },
];

const TIMEFRAMES = ["M30", "H1", "H4"] as const;

/* ─────────────────────── sub-components ─────────────────────── */

function FieldGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col">
      <div className="mb-1.5 text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
        {label}
      </div>
      {children}
    </div>
  );
}

interface ModelBadgeProps {
  type: string;
}
function ModelTypeBadge({ type }: ModelBadgeProps) {
  const palette: Record<string, { bg: string; color: string }> = {
    logistic: { bg: "rgba(0,229,255,0.10)", color: "var(--color-brand)" },
    xgboost: { bg: "rgba(91,192,235,0.12)", color: "var(--color-accent-classical)" },
    random_forest: { bg: "rgba(72,213,151,0.10)", color: "var(--color-accent-success)" },
    decision_tree: { bg: "rgba(72,213,151,0.08)", color: "var(--color-accent-success)" },
    lightgbm: { bg: "rgba(72,213,151,0.12)", color: "var(--color-accent-success)" },
    catboost: { bg: "rgba(72,213,151,0.10)", color: "var(--color-accent-success)" },
    lstm: { bg: "rgba(180,120,255,0.10)", color: "var(--color-accent-deep)" },
    cnn: { bg: "rgba(255,165,0,0.10)", color: "var(--color-accent-warning)" },
    transformer: { bg: "rgba(180,120,255,0.12)", color: "var(--color-accent-deep)" },
    gru: { bg: "rgba(50,200,255,0.10)", color: "var(--color-brand)" },
    gru_lstm: { bg: "rgba(50,200,255,0.12)", color: "var(--color-brand)" },
    svm: { bg: "rgba(255,82,82,0.10)", color: "var(--color-accent-danger)" },
    dqn: { bg: "rgba(255,82,82,0.12)", color: "var(--color-accent-danger)" },
    ensemble: { bg: "rgba(72,213,151,0.10)", color: "var(--color-accent-success)" },
  };
  const key = Object.keys(palette).find((k) => type?.toLowerCase().includes(k)) ?? "logistic";
  const { bg, color } = palette[key];
  return (
    <span
      className="rounded px-2 py-0.5 font-mono text-[10px] font-semibold tracking-[0.06em] uppercase"
      style={{ backgroundColor: bg }}
    >
      {type}
    </span>
  );
}

function StatChip({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="flex min-w-[80px] flex-col gap-0.5 rounded-sm bg-(--color-elevated) px-3 py-2">
      <span className="text-[9px] tracking-[0.08em] text-(--color-text-muted) uppercase">
        {label}
      </span>
      <span
        className="font-mono text-xs font-semibold"
        style={{
          color:
            positive === true
              ? "var(--color-accent-success)"
              : positive === false
                ? "var(--color-accent-danger)"
                : "var(--color-text-primary)",
        }}
      >
        {value}
      </span>
    </div>
  );
}

/* ─────────────────────── main component ─────────────────────── */
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
  const [modelOpen, setModelOpen] = useState(false);

  const { data: models, isLoading: loadingModels } = useQuery<
    Array<{
      id: string;
      model_type: string;
      best_sharpe: number | null;
      best_return: number | null;
      created_at: string;
      status: string;
      tags: string[];
      missing_on_disk?: boolean;
    }>
  >({
    queryKey: ["deployed-models"],
    queryFn: async () => {
      const { data } = await apiClient.get<{
        models: Array<{
          id: string;
          model_type: string;
          best_sharpe: number | null;
          best_return: number | null;
          created_at: string;
          status: string;
          tags: string[];
          missing_on_disk?: boolean;
        }>;
      }>("/models/deployed");
      return data.models;
    },
    refetchOnMount: true,
  });

  const activeModels = (models ?? []).filter((m) => !m.missing_on_disk);
  const selected = activeModels.find((m) => m.id === selectedModel);

  const handleSelect = useCallback((id: string) => {
    setSelectedModel(id);
    setModelOpen(false);
  }, []);

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
        },
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

  const fmt = (n: number | null, suffix = "", places = 2) =>
    n != null ? `${n >= 0 ? "+" : ""}${n.toFixed(places)}${suffix}` : "—";

  return (
    <Panel>
      <PanelHeader title="Forward Test" subtitle="Validate a saved model on out-of-sample data." />

      {/* ── error banner ── */}
      {error && (
        <div className="flex items-center gap-2 rounded-sm border border-(--color-accent-danger) bg-[rgba(239,68,68,0.06)] px-4 py-2.5">
          <TriangleAlert size={14} className="shrink-0 text-(--color-accent-danger)" />
          <span className="text-xs text-(--color-accent-danger)">{error}</span>
        </div>
      )}

      {/* ── section 1: model selection ── */}
      <Section title="Model" description="Choose a previously saved model to forward test.">
        {loadingModels ? (
          <div className="flex items-center gap-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-9 flex-1 animate-pulse rounded-sm bg-(--color-elevated)" />
            ))}
          </div>
        ) : activeModels.length === 0 ? (
          <div className="flex items-center gap-3 rounded-sm border border-dashed border-(--color-glass-border) bg-(--color-elevated) px-4 py-3">
            <TriangleAlert size={14} className="text-(--color-accent-warning)" />
            <span className="text-xs text-(--color-text-muted)">
              No deployed models found. Run a backtest first to save a model.
            </span>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {/* custom dropdown trigger */}
            <div style={{ position: "relative" }}>
              <button
                onClick={() => setModelOpen((o) => !o)}
                className="flex w-full cursor-pointer items-center justify-between rounded-sm border bg-(--color-elevated) px-3 py-2.5 transition-all duration-150"
                style={{
                  borderColor: modelOpen ? "var(--color-brand)" : "var(--color-glass-border)",
                  boxShadow: modelOpen ? "0 0 0 1px var(--color-brand)" : "none",
                }}
              >
                <span
                  className="font-mono text-xs"
                  style={{
                    color: selected ? "var(--color-text-primary)" : "var(--color-text-muted)",
                  }}
                >
                  {selected ? (
                    <span className="flex items-center gap-2">
                      <ModelTypeBadge type={selected.model_type} />
                      <span>{selected.id.slice(0, 8)}…</span>
                    </span>
                  ) : (
                    "Select a saved model..."
                  )}
                </span>
                <ChevronDown
                  size={14}
                  className="text-(--color-text-muted)"
                  style={{
                    transform: modelOpen ? "rotate(180deg)" : "rotate(0deg)",
                    transition: "transform 0.15s",
                  }}
                />
              </button>

              {modelOpen && (
                <div className="absolute top-[calc(100%+4px)] right-0 left-0 z-50 max-h-[280px] overflow-y-auto rounded-[10px] border border-(--color-glass-border) bg-(--color-glass) shadow-[0_8px_32px_rgba(0,0,0,0.4)] backdrop-blur-[16px]">
                  {activeModels.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => handleSelect(m.id)}
                      className="flex w-full cursor-pointer items-center gap-3 border-b border-(--color-glass-border) px-4 py-3 text-left transition-colors duration-100"
                      style={{
                        backgroundColor:
                          m.id === selectedModel ? "var(--color-brand-glow)" : "transparent",
                      }}
                    >
                      <ModelTypeBadge type={m.model_type} />
                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <span className="font-mono text-[11px] text-(--color-text-primary)">
                          {m.id}
                        </span>
                        <span className="text-[10px] text-(--color-text-muted)">
                          {m.created_at?.slice(0, 10)}
                          {m.tags.length > 0 && ` · ${m.tags.join(", ")}`}
                        </span>
                      </div>
                      <div className="flex shrink-0 items-center gap-3">
                        <span
                          className="font-mono text-[11px]"
                          style={{
                            color:
                              m.best_sharpe != null && m.best_sharpe >= 0
                                ? "var(--color-accent-success)"
                                : "var(--color-accent-danger)",
                          }}
                        >
                          SR {fmt(m.best_sharpe)}
                        </span>
                        <span className="font-mono text-[11px] text-(--color-text-secondary)">
                          {fmt(m.best_return, "%", 1)}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* selected model stats */}
            {selected && (
              <div className="flex flex-wrap gap-2 pt-1">
                <StatChip
                  label="Sharpe"
                  value={fmt(selected.best_sharpe)}
                  positive={selected.best_sharpe != null && selected.best_sharpe >= 0}
                />
                <StatChip
                  label="Return"
                  value={fmt(selected.best_return, "%", 1)}
                  positive={selected.best_return != null && selected.best_return >= 0}
                />
                <StatChip label="Status" value={selected.status?.toUpperCase() ?? "—"} />
                <StatChip label="Saved" value={selected.created_at?.slice(0, 10) ?? "—"} />
              </div>
            )}
          </div>
        )}
      </Section>

      {/* ── section 2: market ── */}
      <Section
        title="Market"
        accent="var(--color-accent-classical)"
        description="Instrument and timeframe for the test window."
      >
        <div className="grid max-w-[480px] grid-cols-2 gap-x-8 gap-y-4">
          {/* pair */}
          <FieldGroup label="Pair">
            <input
              value={pair}
              onChange={(e) => setField("pair", e.target.value.toUpperCase())}
              className="w-full rounded-lg border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 font-mono text-xs text-(--color-text-primary) transition-colors duration-150 outline-none focus:border-(--color-brand)"
              placeholder="EURUSD"
            />
          </FieldGroup>

          {/* timeframe */}
          <FieldGroup label="Timeframe">
            <div className="flex gap-1.5">
              {TIMEFRAMES.map((tf) => {
                const active = timeframe === tf;
                return (
                  <button
                    key={tf}
                    onClick={() => setField("timeframe", tf)}
                    className="flex-1 cursor-pointer rounded-[6px] py-[6px] font-mono text-[11px]"
                    style={{
                      border: `1px solid ${active ? "var(--color-brand)" : "var(--color-glass-border)"}`,
                      backgroundColor: active ? "var(--color-brand-glow)" : "var(--color-elevated)",
                      color: active ? "var(--color-brand)" : "var(--color-text-muted)",
                      fontWeight: active ? 600 : 400,
                      transition: "all 0.12s",
                      boxShadow: active ? "0 0 8px rgba(0,229,255,0.15)" : "none",
                    }}
                  >
                    {tf}
                  </button>
                );
              })}
            </div>
          </FieldGroup>
        </div>
      </Section>

      {/* ── section 3: date range ── */}
      <Section
        title="Date Range"
        accent="var(--color-accent-deep)"
        description="Leave blank to use the full available range."
      >
        <div className="grid max-w-[480px] grid-cols-2 gap-x-8">
          <FieldGroup label="Start Date">
            <input
              type="date"
              value={startDate ?? ""}
              onChange={(e) => setField("startDate", e.target.value)}
              className="w-full rounded-lg border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 font-mono text-xs text-(--color-text-primary) transition-colors duration-150 outline-none focus:border-(--color-brand)"
              style={{ colorScheme: "dark" }}
            />
          </FieldGroup>
          <FieldGroup label="End Date">
            <input
              type="date"
              value={endDate ?? ""}
              onChange={(e) => setField("endDate", e.target.value)}
              className="w-full rounded-lg border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 font-mono text-xs text-(--color-text-primary) transition-colors duration-150 outline-none focus:border-(--color-brand)"
              style={{ colorScheme: "dark" }}
            />
          </FieldGroup>
        </div>

        <p className="mt-2.5 text-[10px] text-(--color-text-muted)">
          Leave blank to use the full available data range for the selected pair.
        </p>
      </Section>

      {/* ── section 4: position sizing ── */}
      <Section
        title="Position Sizing"
        accent="var(--color-accent-warning)"
        description="How trade size is determined during the test."
      >
        <div className="flex flex-wrap gap-2">
          {SIZING_OPTIONS.map((opt) => {
            const active = posSizing === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setPosSizing(opt.value)}
                className="flex min-w-[130px] cursor-pointer flex-col gap-0.5 rounded-sm border px-3 py-2 text-left transition-all duration-150"
                style={{
                  borderColor: active ? "var(--color-brand)" : "var(--color-glass-border)",
                  backgroundColor: active ? "var(--color-brand-glow)" : "var(--color-elevated)",
                  boxShadow: active ? "0 0 8px rgba(0,229,255,0.12)" : "none",
                }}
              >
                <span
                  className="text-[11px] font-semibold tracking-[0.02em]"
                  style={{
                    color: active ? "var(--color-brand)" : "var(--color-text-secondary)",
                  }}
                >
                  {opt.label}
                </span>
                <span className="text-[9px] leading-[1.4] text-(--color-text-muted)">
                  {opt.hint}
                </span>
              </button>
            );
          })}
        </div>
      </Section>

      {/* ── section 5: execution ── */}
      <Section
        title="Execution"
        accent="var(--color-accent-success)"
        description="Cost-simulation settings for realistic results."
      >
        <button
          onClick={() => setUseCosts((v) => !v)}
          className="flex w-fit cursor-pointer items-center gap-3 rounded-sm border px-4 py-3 text-left transition-all duration-150"
          style={{
            borderColor: useCosts ? "var(--color-brand)" : "var(--color-glass-border)",
            backgroundColor: useCosts ? "var(--color-brand-glow)" : "var(--color-elevated)",
          }}
        >
          {/* toggle pill */}
          <div
            className="relative h-[18px] w-8 shrink-0 rounded-[9px]"
            style={{
              backgroundColor: useCosts ? "var(--color-brand)" : "var(--color-glass-border)",
              transition: "background-color 0.15s",
            }}
          >
            <div
              className="absolute top-[3px] h-3 w-3 rounded-full"
              style={{
                left: useCosts ? 17 : 3,
                backgroundColor: useCosts ? "var(--color-text-inverse)" : "var(--color-text-muted)",
                transition: "left 0.15s",
              }}
            />
          </div>
          <div className="flex flex-col gap-0.5">
            <span
              className="text-xs font-medium"
              style={{
                color: useCosts ? "var(--color-text-primary)" : "var(--color-text-muted)",
              }}
            >
              Apply Trading Costs
            </span>
            <span className="text-[10px] text-(--color-text-muted)">
              Spread + slippage simulation — recommended for realistic results
            </span>
          </div>
        </button>
      </Section>

      {/* ── run button ── */}
      <div className="flex items-center gap-4 pt-1">
        <button
          onClick={handleRun}
          disabled={!selectedModel || isSubmitting}
          className="flex items-center gap-2.5 rounded-sm bg-(--color-brand) px-6 py-2.5 text-[11px] font-bold tracking-[0.08em] text-(--color-text-inverse) uppercase transition-all duration-150 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          style={{
            boxShadow: !selectedModel || isSubmitting ? "none" : "0 0 16px rgba(0,229,255,0.3)",
          }}
        >
          <Play size={14} />
          {isSubmitting ? "Submitting..." : "Run Forward Test"}
        </button>

        {!selectedModel && (
          <span className="text-[11px] text-(--color-text-muted)">Select a model to continue</span>
        )}
      </div>
    </Panel>
  );
}
