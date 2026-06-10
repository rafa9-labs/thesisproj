import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  Play,
  TriangleAlert,
  ChevronDown,
} from "lucide-react";
import apiClient from "@/api/client";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useJobStore } from "@/stores/useJobStore";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";

/* ─────────────────────── shared style tokens ─────────────────────── */
const fieldLabel: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 500,
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  color: "var(--color-text-muted)",
  marginBottom: 6,
};

const inputBase: React.CSSProperties = {
  borderRadius: 8,
  border: "1px solid var(--color-glass-border)",
  backgroundColor: "var(--color-elevated)",
  color: "var(--color-text-primary)",
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  padding: "7px 12px",
  outline: "none",
  transition: "border-color 0.15s",
};

/* ─────────────────────── sizing options ─────────────────────── */
const SIZING_OPTIONS = [
  { value: "fixed",            label: "Fixed Lot",          hint: "Constant 1-lot size every trade" },
  { value: "fixed_fractional", label: "Fractional %",       hint: "Risk a fixed % of equity" },
  { value: "kelly",            label: "Kelly Criterion",    hint: "Optimal growth fraction" },
  { value: "atr",              label: "ATR Volatility",     hint: "Size adjusted to current volatility" },
  { value: "vol_target",       label: "Vol Target",         hint: "Target a fixed realised-vol level" },
];

const TIMEFRAMES = ["M30", "H1", "H4"] as const;

/* ─────────────────────── sub-components ─────────────────────── */

function FieldGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col">
      <div style={fieldLabel}>{label}</div>
      {children}
    </div>
  );
}

interface ModelBadgeProps {
  type: string;
}
function ModelTypeBadge({ type }: ModelBadgeProps) {
  const palette: Record<string, { bg: string; color: string }> = {
    logistic:       { bg: "rgba(0,229,255,0.10)",   color: "var(--color-brand)" },
    xgboost:        { bg: "rgba(91,192,235,0.12)",  color: "var(--color-accent-classical)" },
    random_forest:  { bg: "rgba(72,213,151,0.10)",  color: "var(--color-accent-success)" },
    decision_tree:  { bg: "rgba(72,213,151,0.08)",  color: "var(--color-accent-success)" },
    lightgbm:       { bg: "rgba(72,213,151,0.12)",  color: "var(--color-accent-success)" },
    catboost:       { bg: "rgba(72,213,151,0.10)",  color: "var(--color-accent-success)" },
    lstm:           { bg: "rgba(180,120,255,0.10)", color: "var(--color-accent-deep)" },
    cnn:            { bg: "rgba(255,165,0,0.10)",   color: "var(--color-accent-warning)" },
    transformer:    { bg: "rgba(180,120,255,0.12)", color: "var(--color-accent-deep)" },
    gru:            { bg: "rgba(50,200,255,0.10)",  color: "var(--color-brand)" },
    gru_lstm:       { bg: "rgba(50,200,255,0.12)",  color: "var(--color-brand)" },
    svm:            { bg: "rgba(255,82,82,0.10)",   color: "var(--color-accent-danger)" },
    dqn:            { bg: "rgba(255,82,82,0.12)",   color: "var(--color-accent-danger)" },
    ensemble:       { bg: "rgba(72,213,151,0.10)",  color: "var(--color-accent-success)" },
  };
  const key = Object.keys(palette).find((k) => type?.toLowerCase().includes(k)) ?? "logistic";
  const { bg, color } = palette[key];
  return (
    <span
      style={{
        backgroundColor: bg,
        color,
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: "0.06em",
        padding: "2px 8px",
        borderRadius: 4,
        textTransform: "uppercase",
      }}
    >
      {type}
    </span>
  );
}

function StatChip({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  return (
    <div
      className="flex flex-col gap-0.5 rounded-sm px-3 py-2"
      style={{ backgroundColor: "var(--color-elevated)", minWidth: 80 }}
    >
      <span style={{ fontSize: 9, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {label}
      </span>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          fontWeight: 600,
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
  const pair       = useBacktestStore((s) => s.pair);
  const timeframe  = useBacktestStore((s) => s.timeframe);
  const startDate  = useBacktestStore((s) => s.startDate);
  const endDate    = useBacktestStore((s) => s.endDate);
  const setField   = useBacktestStore((s) => s.setField);
  const startJob   = useJobStore((s) => s.startJob);

  const [selectedModel, setSelectedModel] = useState<string>("");
  const [posSizing,     setPosSizing]     = useState("fixed");
  const [useCosts,      setUseCosts]      = useState(true);
  const [isSubmitting,  setIsSubmitting]  = useState(false);
  const [error,         setError]         = useState<string | null>(null);
  const [modelOpen,     setModelOpen]     = useState(false);

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
          start_date:       startDate || undefined,
          end_date:         endDate   || undefined,
          position_sizing:  posSizing,
          trading_costs:    useCosts,
        }
      );
      startJob(data.job_id, pair, [selectedModel]);
      navigate("/monitor");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data
        ?.detail;
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
        <div
          className="flex items-center gap-2 rounded-sm border px-4 py-2.5"
          style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(239,68,68,0.06)" }}
        >
          <TriangleAlert size={14} style={{ color: "var(--color-accent-danger)", flexShrink: 0 }} />
          <span style={{ fontSize: 12, color: "var(--color-accent-danger)" }}>{error}</span>
        </div>
      )}

      {/* ── section 1: model selection ── */}
      <Section title="Model" description="Choose a previously saved model to forward test.">

        {loadingModels ? (
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-9 rounded-sm animate-pulse flex-1" style={{ backgroundColor: "var(--color-elevated)" }} />
            ))}
          </div>
        ) : activeModels.length === 0 ? (
          <div
            className="flex items-center gap-3 rounded-sm px-4 py-3"
            style={{ backgroundColor: "var(--color-elevated)", border: "1px dashed var(--color-glass-border)" }}
          >
            <TriangleAlert size={14} style={{ color: "var(--color-accent-warning)" }} />
            <span style={{ fontSize: 12, color: "var(--color-text-muted)" }}>
              No deployed models found. Run a backtest first to save a model.
            </span>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {/* custom dropdown trigger */}
            <div style={{ position: "relative" }}>
              <button
                onClick={() => setModelOpen((o) => !o)}
                className="flex w-full items-center justify-between rounded-sm border px-3 py-2.5 transition-all duration-150"
                style={{
                  backgroundColor: "var(--color-elevated)",
                  borderColor: modelOpen ? "var(--color-brand)" : "var(--color-glass-border)",
                  boxShadow: modelOpen ? "0 0 0 1px var(--color-brand)" : "none",
                  cursor: "pointer",
                }}
              >
                <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: selected ? "var(--color-text-primary)" : "var(--color-text-muted)" }}>
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
                  style={{
                    color: "var(--color-text-muted)",
                    transform: modelOpen ? "rotate(180deg)" : "rotate(0deg)",
                    transition: "transform 0.15s",
                  }}
                />
              </button>

              {modelOpen && (
                <div
                  style={{
                    position: "absolute",
                    top: "calc(100% + 4px)",
                    left: 0,
                    right: 0,
                    zIndex: 50,
                    backgroundColor: "var(--color-glass)",
                    border: "1px solid var(--color-glass-border)",
                    borderRadius: 10,
                    backdropFilter: "blur(16px)",
                    maxHeight: 280,
                    overflowY: "auto",
                    boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                  }}
                >
                  {activeModels.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => handleSelect(m.id)}
                      className="flex w-full items-center gap-3 px-4 py-3 transition-colors duration-100"
                      style={{
                        borderBottom: "1px solid var(--color-glass-border)",
                        backgroundColor: m.id === selectedModel ? "var(--color-brand-glow)" : "transparent",
                        cursor: "pointer",
                        textAlign: "left",
                      }}
                    >
                      <ModelTypeBadge type={m.model_type} />
                      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                          {m.id}
                        </span>
                        <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                          {m.created_at?.slice(0, 10)}
                          {m.tags.length > 0 && ` · ${m.tags.join(", ")}`}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: m.best_sharpe != null && m.best_sharpe >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
                          SR {fmt(m.best_sharpe)}
                        </span>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--color-text-secondary)" }}>
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
                <StatChip label="Sharpe" value={fmt(selected.best_sharpe)} positive={selected.best_sharpe != null && selected.best_sharpe >= 0} />
                <StatChip label="Return" value={fmt(selected.best_return, "%", 1)} positive={selected.best_return != null && selected.best_return >= 0} />
                <StatChip label="Status" value={selected.status?.toUpperCase() ?? "—"} />
                <StatChip label="Saved" value={selected.created_at?.slice(0, 10) ?? "—"} />
              </div>
            )}
          </div>
        )}
      </Section>

      {/* ── section 2: market ── */}
      <Section title="Market" accent="var(--color-accent-classical)" description="Instrument and timeframe for the test window.">

        <div className="grid grid-cols-2 gap-x-8 gap-y-4" style={{ maxWidth: 480 }}>
          {/* pair */}
          <FieldGroup label="Pair">
            <input
              value={pair}
              onChange={(e) => setField("pair", e.target.value.toUpperCase())}
              style={{ ...inputBase, width: "100%" }}
              placeholder="EURUSD"
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--color-brand)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "var(--color-glass-border)")}
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
                    style={{
                      flex: 1,
                      padding: "6px 0",
                      borderRadius: 6,
                      border: `1px solid ${active ? "var(--color-brand)" : "var(--color-glass-border)"}`,
                      backgroundColor: active ? "var(--color-brand-glow)" : "var(--color-elevated)",
                      color: active ? "var(--color-brand)" : "var(--color-text-muted)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      fontWeight: active ? 600 : 400,
                      cursor: "pointer",
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
      <Section title="Date Range" accent="var(--color-accent-deep)" description="Leave blank to use the full available range.">

        <div className="grid grid-cols-2 gap-x-8" style={{ maxWidth: 480 }}>
          <FieldGroup label="Start Date">
            <input
              type="date"
              value={startDate ?? ""}
              onChange={(e) => setField("startDate", e.target.value)}
              style={{ ...inputBase, width: "100%", colorScheme: "dark" }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--color-brand)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "var(--color-glass-border)")}
            />
          </FieldGroup>
          <FieldGroup label="End Date">
            <input
              type="date"
              value={endDate ?? ""}
              onChange={(e) => setField("endDate", e.target.value)}
              style={{ ...inputBase, width: "100%", colorScheme: "dark" }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--color-brand)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "var(--color-glass-border)")}
            />
          </FieldGroup>
        </div>

        <p style={{ fontSize: 10, color: "var(--color-text-muted)", marginTop: 10 }}>
          Leave blank to use the full available data range for the selected pair.
        </p>
      </Section>

      {/* ── section 4: position sizing ── */}
      <Section title="Position Sizing" accent="var(--color-accent-warning)" description="How trade size is determined during the test.">

        <div className="flex flex-wrap gap-2">
          {SIZING_OPTIONS.map((opt) => {
            const active = posSizing === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setPosSizing(opt.value)}
                className="flex flex-col gap-0.5 rounded-sm border px-3 py-2 text-left transition-all duration-150"
                style={{
                  borderColor: active ? "var(--color-brand)" : "var(--color-glass-border)",
                  backgroundColor: active ? "var(--color-brand-glow)" : "var(--color-elevated)",
                  boxShadow: active ? "0 0 8px rgba(0,229,255,0.12)" : "none",
                  cursor: "pointer",
                  minWidth: 130,
                }}
              >
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: active ? "var(--color-brand)" : "var(--color-text-secondary)",
                    letterSpacing: "0.02em",
                  }}
                >
                  {opt.label}
                </span>
                <span style={{ fontSize: 9, color: "var(--color-text-muted)", lineHeight: 1.4 }}>
                  {opt.hint}
                </span>
              </button>
            );
          })}
        </div>
      </Section>

      {/* ── section 5: execution ── */}
      <Section title="Execution" accent="var(--color-accent-success)" description="Cost-simulation settings for realistic results.">

        <button
          onClick={() => setUseCosts((v) => !v)}
          className="flex items-center gap-3 rounded-sm border px-4 py-3 transition-all duration-150"
          style={{
            borderColor: useCosts ? "var(--color-brand)" : "var(--color-glass-border)",
            backgroundColor: useCosts ? "var(--color-brand-glow)" : "var(--color-elevated)",
            cursor: "pointer",
            textAlign: "left",
            width: "fit-content",
          }}
        >
          {/* toggle pill */}
          <div
            style={{
              width: 32,
              height: 18,
              borderRadius: 9,
              backgroundColor: useCosts ? "var(--color-brand)" : "var(--color-glass-border)",
              position: "relative",
              transition: "background-color 0.15s",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 3,
                left: useCosts ? 17 : 3,
                width: 12,
                height: 12,
                borderRadius: "50%",
                backgroundColor: useCosts ? "var(--color-text-inverse)" : "var(--color-text-muted)",
                transition: "left 0.15s",
              }}
            />
          </div>
          <div className="flex flex-col gap-0.5">
            <span style={{ fontSize: 12, fontWeight: 500, color: useCosts ? "var(--color-text-primary)" : "var(--color-text-muted)" }}>
              Apply Trading Costs
            </span>
            <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
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
          className="flex items-center gap-2.5 rounded-sm px-6 py-2.5 transition-all duration-150 hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            backgroundColor: "var(--color-brand)",
            color: "var(--color-text-inverse)",
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            boxShadow: !selectedModel || isSubmitting ? "none" : "0 0 16px rgba(0,229,255,0.3)",
          }}
        >
          <Play size={14} />
          {isSubmitting ? "Submitting..." : "Run Forward Test"}
        </button>

        {!selectedModel && (
          <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
            Select a model to continue
          </span>
        )}
      </div>
    </Panel>
  );
}
