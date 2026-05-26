import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Info, CheckCircle2, AlertCircle, Clock } from "lucide-react";
import apiClient from "@/api/client";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useRuntimeEstimate } from "@/api/queries";

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
  { value: "fixed_fractional", label: "Fractional (% equity)" },
  { value: "kelly", label: "Kelly Criterion" },
  { value: "atr", label: "ATR Volatility" },
  { value: "vol_target", label: "Vol Target" },
];

/* ── tiny shared primitives ─────────────────────────────────────── */

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 mb-3">
      <span
        style={{
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: "0.12em",
          color: "#4B5563",
          textTransform: "uppercase",
          fontFamily: "var(--font-mono)",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
      <div style={{ flex: 1, height: 1, backgroundColor: "#2A2E39" }} />
    </div>
  );
}

function Tip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex" style={{ lineHeight: 0 }}>
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        style={{ color: "#374151", lineHeight: 0, background: "none", border: "none", padding: 0, cursor: "default" }}
        tabIndex={-1}
      >
        <Info size={11} strokeWidth={1.5} />
      </button>
      {open && (
        <span
          className="absolute z-50 rounded"
          style={{
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            width: 220,
            padding: "6px 9px",
            backgroundColor: "#1E222D",
            border: "1px solid #2A2E39",
            color: "#9CA3AF",
            fontSize: 10,
            lineHeight: 1.5,
            fontFamily: "inherit",
            whiteSpace: "normal",
            pointerEvents: "none",
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}

function RowLabel({ label, tip }: { label: string; tip?: string }) {
  return (
    <div className="flex items-center gap-1 flex-shrink-0" style={{ minWidth: 110 }}>
      <span style={{ fontSize: 10, color: "#6B7280", letterSpacing: "0.06em", textTransform: "uppercase", fontFamily: "inherit" }}>
        {label}
      </span>
      {tip && <Tip text={tip} />}
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="rounded-full flex-shrink-0 transition-colors"
      style={{
        width: 28,
        height: 14,
        backgroundColor: checked ? "#1D4ED833" : "#1F2937",
        border: `1px solid ${checked ? "#3B82F6" : "#374151"}`,
        position: "relative",
        cursor: "pointer",
      }}
    >
      <span
        className="absolute rounded-full transition-all"
        style={{
          width: 8,
          height: 8,
          top: 2,
          left: checked ? 15 : 3,
          backgroundColor: checked ? "#3B82F6" : "#4B5563",
        }}
      />
    </button>
  );
}

function CompactSelect({
  value,
  onChange,
  options,
  width = 180,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  width?: number;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        height: 26,
        width,
        paddingLeft: 8,
        paddingRight: 24,
        fontSize: 11,
        fontFamily: "var(--font-mono)",
        color: "#E5E7EB",
        backgroundColor: "#131722",
        border: "1px solid #2A2E39",
        borderRadius: 3,
        outline: "none",
        cursor: "pointer",
        appearance: "none",
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath d='M2 3.5l3 3 3-3' stroke='%236B7280' stroke-width='1.2' fill='none' stroke-linecap='round'/%3E%3C/svg%3E")`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 7px center",
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

function DateInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input
      type="date"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        height: 26,
        padding: "0 8px",
        fontSize: 11,
        fontFamily: "var(--font-mono)",
        color: "#E5E7EB",
        backgroundColor: "#131722",
        border: "1px solid #2A2E39",
        borderRadius: 3,
        outline: "none",
        width: 130,
        colorScheme: "dark",
      }}
    />
  );
}

/* ── deployment summary grid ────────────────────────────────────── */

function DeploymentSummary() {
  const pair = useBacktestStore((s) => s.pair);
  const timeframe = useBacktestStore((s) => s.timeframe);
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const nTrials = useBacktestStore((s) => s.nTrials);
  const repeats = useBacktestStore((s) => s.repeats);
  const trainMonths = useBacktestStore((s) => s.trainMonths);
  const testMonths = useBacktestStore((s) => s.testMonths);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);
  const startDate = useBacktestStore((s) => s.startDate);
  const endDate = useBacktestStore((s) => s.endDate);

  const { data: runtimeData } = useRuntimeEstimate(
    selectedModels as string[],
    testMonths as number,
    hpoIntensity,
  );

  const fmtRuntime = (lo: number, hi: number) => {
    const fmt = (m: number) => {
      if (m < 1) return `${Math.round(m * 60)}s`;
      if (m < 60) return `~${Math.round(m)}m`;
      const h = Math.floor(m / 60);
      const min = Math.round(m % 60);
      return min > 0 ? `~${h}h ${min}m` : `~${h}h`;
    };
    return `${fmt(lo)} – ${fmt(hi)}`;
  };

  const walkFwdLabel = `${nTrials} tri × ${repeats} run`;
  const dateRange = startDate && endDate ? `${startDate} → ${endDate}` : `${trainMonths}mo train / ${testMonths}mo test`;

  const checks = [
    { label: "Asset configured", ok: !!(pair && timeframe), detail: pair ? `${pair} · ${timeframe}` : "Not set" },
    { label: "Models selected", ok: selectedModels.length > 0, detail: selectedModels.length > 0 ? `${selectedModels.length} selected` : "None" },
    { label: "Date range set", ok: !!(startDate && endDate), detail: dateRange },
  ];

  const items: { label: string; value: string; highlight?: boolean }[] = [
    { label: "ASSET", value: pair && timeframe ? `${pair} · ${timeframe}` : "—" },
    { label: "MODELS", value: selectedModels.length > 0 ? `${selectedModels.length} selected` : "None" },
    { label: "HPO COMPLEXITY", value: walkFwdLabel },
    {
      label: "EST. COMPUTE",
      value: runtimeData
        ? fmtRuntime(runtimeData.estimated_minutes_low, runtimeData.estimated_minutes_high)
        : "—",
      highlight: true,
    },
    { label: "OOS WINDOW", value: dateRange },
    { label: "WALK-FWD", value: `${testMonths}mo steps` },
    { label: "TRAIN WINDOW", value: `${trainMonths}mo` },
    { label: "INTENSITY", value: hpoIntensity.toUpperCase() },
  ];

  return (
    <div
      className="rounded"
      style={{
        backgroundColor: "#181C25",
        border: "1px solid #2A2E39",
        padding: "14px 16px",
      }}
    >
      <SectionHeader label="Deployment Summary" />

      {/* Pre-flight checks */}
      <div className="flex items-center gap-5 mb-4">
        {checks.map((c) => (
          <div key={c.label} className="flex items-center gap-1.5">
            {c.ok ? (
              <CheckCircle2 size={11} style={{ color: "#22C55E", flexShrink: 0 }} />
            ) : (
              <AlertCircle size={11} style={{ color: "#EF4444", flexShrink: 0 }} />
            )}
            <span style={{ fontSize: 10, color: c.ok ? "#6B7280" : "#EF4444", fontFamily: "var(--font-mono)" }}>
              {c.detail}
            </span>
          </div>
        ))}
      </div>

      {/* 4-column summary grid */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "1px",
          backgroundColor: "#2A2E39",
          border: "1px solid #2A2E39",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        {items.map((item) => (
          <div
            key={item.label}
            style={{ backgroundColor: "#131722", padding: "9px 12px" }}
          >
            <div style={{ fontSize: 9, color: "#4B5563", letterSpacing: "0.1em", textTransform: "uppercase", fontFamily: "var(--font-mono)", marginBottom: 4 }}>
              {item.label}
            </div>
            <div
              style={{
                fontSize: 12,
                fontFamily: "var(--font-mono)",
                color: item.highlight ? "#F59E0B" : "#E5E7EB",
                fontWeight: item.highlight ? 600 : 400,
              }}
            >
              {item.highlight && item.value !== "—" && (
                <Clock size={10} style={{ display: "inline", marginRight: 4, verticalAlign: "middle", color: "#F59E0B" }} />
              )}
              {item.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── main component ─────────────────────────────────────────────── */

export function ForwardTestTab() {
  const startDate = useBacktestStore((s) => s.startDate);
  const endDate = useBacktestStore((s) => s.endDate);
  const setField = useBacktestStore((s) => s.setField);

  const [selectedModel, setSelectedModel] = useState<string>("all");
  const [posSizing, setPosSizing] = useState("fixed");
  const [useCosts, setUseCosts] = useState(true);

  const { data: models, isLoading: loadingModels } = useQuery<DeployedModel[]>({
    queryKey: ["deployed-models"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: DeployedModel[] }>("/models/deployed");
      return data.models;
    },
    refetchOnMount: true,
  });

  const activeModels = (models ?? []).filter((m) => !m.missing_on_disk);

  const modelOptions = [
    { value: "all", label: "All Selected Models" },
    ...activeModels.map((m) => ({
      value: m.id,
      label: `${m.model_type}${m.best_sharpe != null ? ` (SR ${m.best_sharpe >= 0 ? "+" : ""}${m.best_sharpe.toFixed(2)})` : ""}`,
    })),
  ];

  const panelStyle: React.CSSProperties = {
    backgroundColor: "#1E222D",
    border: "1px solid #2A2E39",
    borderRadius: 3,
    padding: "12px 14px",
  };

  const rowStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 20,
    flexWrap: "wrap" as const,
  };

  return (
    <div className="flex flex-col gap-4 pt-1">
      {/* Outer card */}
      <div
        style={{
          backgroundColor: "#131722",
          border: "1px solid #2A2E39",
          borderRadius: 3,
          padding: "14px 16px",
        }}
      >
        <div
          style={{
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: "0.14em",
            color: "#4B5563",
            textTransform: "uppercase",
            fontFamily: "var(--font-mono)",
            marginBottom: 14,
          }}
        >
          Forward Test
        </div>

        {/* Section 1: Out-of-Sample Window */}
        <div style={{ ...panelStyle, marginBottom: 10 }}>
          <SectionHeader label="Out-of-Sample Window" />
          <div style={rowStyle}>
            {/* Target model */}
            <div className="flex items-center gap-2">
              <RowLabel label="Target Model" tip="Which trained model(s) to evaluate in the OOS window. 'All Selected' runs each model independently." />
              {loadingModels ? (
                <span style={{ fontSize: 11, color: "#6B7280", fontFamily: "var(--font-mono)" }}>Loading...</span>
              ) : (
                <CompactSelect value={selectedModel} onChange={setSelectedModel} options={modelOptions} width={200} />
              )}
            </div>

            {/* OOS Start Date */}
            <div className="flex items-center gap-2">
              <RowLabel label="OOS Start" tip="First bar of the out-of-sample period. Must be after the training window ends." />
              <DateInput value={startDate ?? ""} onChange={(v) => setField("startDate", v)} />
            </div>

            {/* OOS End Date */}
            <div className="flex items-center gap-2">
              <RowLabel label="OOS End" tip="Last bar of the out-of-sample period." />
              <DateInput value={endDate ?? ""} onChange={(v) => setField("endDate", v)} />
            </div>
          </div>
        </div>

        {/* Section 2: Execution Overrides */}
        <div style={panelStyle}>
          <SectionHeader label="Execution Overrides" />
          <div style={rowStyle}>
            {/* Position Sizing */}
            <div className="flex items-center gap-2">
              <RowLabel label="Position Sizing" tip="How trade size is calculated. Overrides the Execution tab setting for this forward test only." />
              <CompactSelect value={posSizing} onChange={setPosSizing} options={SIZING_OPTIONS} width={180} />
            </div>

            {/* Trading Costs toggle */}
            <div className="flex items-center gap-2">
              <RowLabel label="Trading Costs" tip="Apply spread and slippage to each trade. Disabling shows a gross P&L upper bound." />
              <Toggle checked={useCosts} onChange={setUseCosts} />
              <span style={{ fontSize: 10, color: useCosts ? "#60A5FA" : "#4B5563", fontFamily: "var(--font-mono)" }}>
                {useCosts ? "Apply Spread + Slippage" : "Gross only"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: Deployment Summary */}
      <DeploymentSummary />
    </div>
  );
}
