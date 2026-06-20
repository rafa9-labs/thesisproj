import { useState, useMemo } from "react";
import {
  CheckCircle, Clock, SkipForward, XCircle,
} from "lucide-react";
import {
  ResponsiveContainer, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ComposedChart,
} from "recharts";
import { useFullCycleResults, useSaveCommittee } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { formatStopReason } from "@/lib/constants";
import apiClient from "@/api/client";
import { CommitteeMetricsGrid } from "./CommitteeMetricsGrid";
import { CommitteeRegimeChart } from "./CommitteeRegimeChart";
import { CommitteeFoldChart } from "./CommitteeFoldChart";
import { CommitteeVoteAgreementChart } from "./CommitteeVoteAgreementChart";
import { ModelAgreementMatrix } from "./ModelAgreementMatrix";
import { ModelContributionChart } from "./ModelContributionChart";
import type { FactoryIterationRecord } from "@/api/schemas";

/* ════════════════════════════════════════════════════════════════════
   Helpers
   ════════════════════════════════════════════════════════════════════ */

const CHART_COLORS = {
  equity: "#00e5ff",
  drawdown: "#f23645",
  buyHold: "rgba(255,255,255,0.35)",
  grid: "rgba(255,255,255,0.06)",
  green: "#089981",
  red: "#f23645",
  amber: "#f2b436",
};

function BentoCard({
  title, right, children, className,
}: {
  title?: string; right?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <div className={`rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3 ${className ?? ""}`}>
      {(title || right) && (
        <div className="mb-2 flex items-center justify-between">
          {title && (
            <h3 className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">{title}</h3>
          )}
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

function PhaseHeader({
  number, label, seconds, status, summary,
}: {
  number: number; label: string; seconds?: number; status: "completed" | "skipped" | "failed" | "running";
  summary?: string;
}) {
  const icon =
    status === "completed" ? <CheckCircle size={13} className="text-(--color-accent-success)" /> :
    status === "skipped" ? <SkipForward size={13} className="text-(--color-text-dim)" /> :
    status === "failed" ? <XCircle size={13} className="text-(--color-accent-danger)" /> :
    <Clock size={13} className="text-(--color-accent-warning)" />;

  const bg =
    status === "completed" ? "rgba(8,153,129,0.08)" :
    status === "skipped" ? "var(--color-elevated)" :
    status === "failed" ? "rgba(242,54,69,0.08)" :
    "rgba(242,180,54,0.08)";

  const border =
    status === "completed" ? "rgba(8,153,129,0.2)" :
    status === "skipped" ? "var(--color-glass-border)" :
    status === "failed" ? "rgba(242,54,69,0.2)" :
    "rgba(242,180,54,0.2)";

  return (
    <div
      className="flex flex-wrap items-center gap-3 rounded-sm border px-3 py-2"
      style={{ backgroundColor: bg, borderColor: border }}
    >
      {icon}
      <span className="text-[11px] font-semibold tracking-[0.06em] text-(--color-text-primary) uppercase">
        Phase {number}: {label}
      </span>
      {seconds != null && (
        <span className="font-mono text-[10px] text-(--color-text-muted) tabular-nums">
          {seconds < 60 ? `${seconds.toFixed(0)}s` : `${(seconds / 60).toFixed(1)}m`}
        </span>
      )}
      {summary && (
        <span className="text-[10px] text-(--color-text-dim)">— {summary}</span>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Sub-components
   ════════════════════════════════════════════════════════════════════ */

function EquityDrawdownCharts({
  equity, drawdown, buyHold, monthly,
}: {
  equity?: Array<{ bar_index: number; value: number }>;
  drawdown?: Array<{ bar_index: number; value: number }>;
  buyHold?: Array<{ bar_index: number; value: number }>;
  monthly?: Array<{ month: number; return_pct: number }>;
}) {
  const [chartView, setChartView] = useState<"equity" | "drawdown" | "monthly">("equity");

  const eqData = useMemo(() => {
    if (!equity || equity.length === 0) return [];
    return equity.map((p) => ({ idx: p.bar_index, equity: +(p.value * 100 - 100).toFixed(2), dd: 0 }));
  }, [equity]);

  const ddData = useMemo(() => {
    if (!drawdown || drawdown.length === 0) return [];
    return drawdown.map((p) => ({ idx: p.bar_index, dd: +(p.value * 100).toFixed(2) }));
  }, [drawdown]);

  const bhData = useMemo(() => {
    if (!buyHold || buyHold.length === 0) return [];
    return buyHold.map((p) => ({ idx: p.bar_index, bh: +(p.value * 100 - 100).toFixed(2) }));
  }, [buyHold]);

  const combinedData = useMemo(() => {
    const map = new Map<number, { idx: number; equity?: number; bh?: number }>();
    for (const p of eqData) map.set(p.idx, { idx: p.idx, equity: p.equity });
    for (const p of bhData) {
      const e = map.get(p.idx);
      if (e) e.bh = p.bh;
      else map.set(p.idx, { idx: p.idx, bh: p.bh });
    }
    return [...map.values()].sort((a, b) => a.idx - b.idx);
  }, [eqData, bhData]);

  const finalEquity = equity && equity.length > 0 ? equity[equity.length - 1].value : 1;
  const maxDD = drawdown && drawdown.length > 0
    ? Math.min(...drawdown.map((d) => d.value)) * 100
    : 0;
  const finalBH = buyHold && buyHold.length > 0 ? buyHold[buyHold.length - 1].value : 1;

  return (
    <BentoCard
      title={
        chartView === "equity" ? "Equity Curve" :
        chartView === "drawdown" ? "Drawdown" :
        "Monthly Returns"
      }
      right={
        <div className="flex items-center gap-1">
          {(["equity", "drawdown", "monthly"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setChartView(v)}
              className="rounded px-2 py-0.5 text-[9px] font-medium uppercase transition"
              style={{
                backgroundColor: chartView === v ? "var(--color-brand-glow)" : "transparent",
                color: chartView === v ? "var(--color-brand)" : "var(--color-text-muted)",
              }}
            >
              {v === "equity" ? "Equity" : v === "drawdown" ? "DD" : "Monthly"}
            </button>
          ))}
        </div>
      }
    >
      {chartView === "equity" && (
        <div>
          <div className="mb-2 flex items-center gap-4 font-mono text-[10px]">
            <span>
              Final: <span style={{ color: CHART_COLORS.green }}>{(finalEquity * 100 - 100).toFixed(2)}%</span>
            </span>
            <span>
              B&H: <span style={{ color: finalBH >= finalEquity ? CHART_COLORS.amber : CHART_COLORS.green }}>
                {(finalBH * 100 - 100).toFixed(2)}%
              </span>
            </span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={combinedData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="idx" hide />
              <YAxis tick={{ fontSize: 9, fill: "var(--color-text-muted)" }} width={40} />
              <Tooltip
                contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, fontSize: 10 }}
                formatter={(v: number, name: string) => [`${v.toFixed(2)}%`, name === "equity" ? "Strategy" : "Buy & Hold"]}
              />
              <Line type="monotone" dataKey="equity" stroke={CHART_COLORS.equity} dot={false} strokeWidth={1.5} connectNulls />
              <Line type="monotone" dataKey="bh" stroke={CHART_COLORS.buyHold} dot={false} strokeWidth={1} strokeDasharray="4 3" connectNulls />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {chartView === "drawdown" && (
        <div>
          <div className="mb-2 font-mono text-[10px]">
            Max DD: <span style={{ color: CHART_COLORS.red }}>{maxDD.toFixed(2)}%</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={ddData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="idx" hide />
              <YAxis tick={{ fontSize: 9, fill: "var(--color-text-muted)" }} width={40} domain={["auto", 0]} />
              <Tooltip
                contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, fontSize: 10 }}
                formatter={(v: number) => [`${v.toFixed(2)}%`, "Drawdown"]}
              />
              <defs>
                <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CHART_COLORS.drawdown} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={CHART_COLORS.drawdown} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="dd" stroke={CHART_COLORS.drawdown} fill="url(#ddFill)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {chartView === "monthly" && monthly && monthly.length > 0 && (
        <MonthlyReturnsHeatmap monthly={monthly} />
      )}
    </BentoCard>
  );
}

function MonthlyReturnsHeatmap({ monthly }: { monthly: Array<{ month: number; return_pct: number }> }) {
  const months = monthly.slice(-24);
  if (months.length === 0) return <span className="text-[10px] text-(--color-text-muted)">No monthly data</span>;

  const maxAbs = Math.max(...months.map((m) => Math.abs(m.return_pct)), 1);

  return (
    <div className="flex flex-wrap gap-[3px]">
      {months.map((m) => {
        const pct = m.return_pct;
        const intensity = Math.min(Math.abs(pct) / maxAbs, 1);
        const isPositive = pct >= 0;
        const r = isPositive ? 8 : 242;
        const g = isPositive ? 153 : 54;
        const b = isPositive ? 129 : 69;
        return (
          <div
            key={m.month}
            className="flex flex-col items-center rounded-[2px] px-[5px] py-[3px]"
            style={{ backgroundColor: `rgba(${r},${g},${b},${0.08 + intensity * 0.22})` }}
            title={`M${m.month}: ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`}
          >
            <span className="font-mono text-[9px] tabular-nums" style={{ color: isPositive ? CHART_COLORS.green : CHART_COLORS.red }}>
              {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function RegimeMatrixHeatmap({ profileMatrix }: { profileMatrix?: Record<string, unknown> }) {
  if (!profileMatrix) return null;

  const regimes = (profileMatrix.regimes as string[]) ?? [];
  const models = (profileMatrix.models as string[]) ?? [];
  const sharpeMatrix = (profileMatrix.sharpe as number[][]) ?? [];
  const tradesMatrix = (profileMatrix.trades as number[][]) ?? [];

  if (regimes.length === 0 || models.length === 0) return null;

  const flatSharpes = sharpeMatrix.flat().filter((s) => isFinite(s));
  const maxSharpe = Math.max(...flatSharpes, 1);
  const minSharpe = Math.min(...flatSharpes, -1);
  const range = maxSharpe - minSharpe || 1;

  return (
    <BentoCard title="Regime × Model Performance" right={
      <span className="text-[9px] text-(--color-text-muted)">Sharpe ratio</span>
    }>
      <div className="overflow-auto">
        <table className="border-collapse text-[10px]">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-(--color-text-dim)" />
              {models.map((m) => (
                <th key={m} className="px-2 py-1 text-center font-mono text-[9px] text-(--color-text-muted)">
                  {m.slice(0, 8)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regimes.map((regime, ri) => (
              <tr key={regime}>
                <td className="px-2 py-1 text-left font-medium text-(--color-text-secondary) whitespace-nowrap">
                  {regime.replace(/_/g, " ")}
                </td>
                {models.map((_, mi) => {
                  const sr = sharpeMatrix[ri]?.[mi];
                  const tr = tradesMatrix[ri]?.[mi];
                  if (sr == null || !isFinite(sr)) {
                    return <td key={mi} className="px-2 py-1 text-center text-(--color-text-dim)">—</td>;
                  }
                  const t = (sr - minSharpe) / range;
                  const r = sr >= 1 ? Math.round(8 + t * 20) : sr >= 0 ? Math.round(242 - t * 200) : 242;
                  const g = sr >= 1 ? Math.round(153 + t * 60) : sr >= 0 ? Math.round(54 + t * 80) : Math.round(54 - (sr + 1) * 30);
                  const b = sr >= 1 ? Math.round(129 + t * 60) : sr >= 0 ? Math.round(69 + t * 30) : 69;
                  return (
                    <td
                      key={mi}
                      className="px-2 py-1 text-center font-mono tabular-nums"
                      style={{ backgroundColor: `rgba(${r},${g},${b},0.25)`, color: sr >= 0.5 ? CHART_COLORS.green : sr >= 0 ? CHART_COLORS.amber : CHART_COLORS.red }}
                      title={`${models[mi]} @ ${regime.replace(/_/g, " ")}: Sharpe ${sr.toFixed(2)}, ${tr?.toFixed(0) ?? "?"} trades`}
                    >
                      {sr.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BentoCard>
  );
}

function FeatureSweepDetail({
  lockedFeatures, lockedCount, prunedCount, topFeature, survivors,
}: {
  lockedFeatures?: string[];
  lockedCount: number;
  prunedCount: number;
  topFeature: string;
  survivors: string[];
}) {
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <BentoCard title="Feature Selection">
        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="rounded-sm border border-(--color-glass-border) bg-(--color-elevated) px-3 py-2 text-center">
            <div className="font-mono text-lg font-semibold text-(--color-accent-success)">{lockedCount}</div>
            <div className="text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Locked</div>
          </div>
          <div className="rounded-sm border border-(--color-glass-border) bg-(--color-elevated) px-3 py-2 text-center">
            <div className="font-mono text-lg font-semibold text-(--color-accent-danger)">{prunedCount}</div>
            <div className="text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Pruned</div>
          </div>
          <div className="rounded-sm border border-(--color-glass-border) bg-(--color-elevated) px-3 py-2 text-center">
            <div className="font-mono text-[11px] font-semibold text-(--color-brand) truncate">{topFeature || "N/A"}</div>
            <div className="text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Top Feature</div>
          </div>
        </div>
        {lockedFeatures && lockedFeatures.length > 0 && (
          <div className="flex flex-wrap gap-[3px]">
            {lockedFeatures.map((f) => (
              <span key={f} className="rounded-[2px] border border-(--color-glass-border) bg-(--color-elevated) px-[6px] py-[2px] font-mono text-[9px] text-(--color-text-dim)">
                {f}
              </span>
            ))}
          </div>
        )}
      </BentoCard>

      {survivors.length > 0 && (
        <BentoCard title={`Model Survivors — ${survivors.length} models`}>
          <div className="flex flex-wrap gap-[4px]">
            {survivors.map((m) => (
              <span key={m} className="rounded-[3px] border border-[rgba(8,153,129,0.25)] bg-[rgba(8,153,129,0.12)] px-[9px] py-[3px] font-mono text-[10px] text-[#089981]">
                {m}
              </span>
            ))}
          </div>
        </BentoCard>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Main Component
   ════════════════════════════════════════════════════════════════════ */

interface Props { jobId: string; onRunAgain: () => void; }

const TH_CLASSES = "px-2 py-1.5 text-left font-medium tracking-[0.06em] uppercase text-[10px]";
const TH_STYLE: React.CSSProperties = { color: "var(--color-text-muted)" };
const TD_CLASSES = "px-2 py-1.5 text-[11px]";
const TD_STYLE: React.CSSProperties = { color: "var(--color-text-secondary)" };
const TD_MONO_CLASSES = "px-2 py-1.5 text-[11px] font-mono";
const TD_MONO_STYLE: React.CSSProperties = { color: "var(--color-text-secondary)" };

export function FullCycleResults({ jobId, onRunAgain }: Props) {
  const { data: results } = useFullCycleResults(jobId);
  const store = useFullCycleStore();
  const saveMutation = useSaveCommittee();
  const [saveSuccess, setSaveSuccess] = useState(false);

  if (!results) {
    return (
      <div className="p-[28px] text-center text-[11px] text-(--color-text-muted)">Loading results...</div>
    );
  }

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `full_cycle_${jobId ?? "results"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDeploy = () => {
    apiClient
      .post("/trading/live/committee/start", {
        pair: store.deployedPair,
        timeframe: store.deployedTimeframe,
        initial_equity: 10000.0,
        confidence_threshold: 0.55,
        mode: store.executionMode,
        full_cycle_job_id: jobId,
      })
      .then((r: { data: { session_id: string; pair: string; timeframe: string; models: string[] } }) => {
        store.setDeployedSession(r.data.session_id, r.data.pair || store.deployedPair, r.data.timeframe || store.deployedTimeframe);
        store.setDeployedJobId(jobId);
      })
      .catch((err: { response?: { data?: { detail?: string } }; message?: string }) => {
        console.error("Deploy failed:", err?.response?.data?.detail ?? err?.message);
      });
  };

  const handleSaveCommittee = () => {
    if (!results?.factory_best_config) return;
    const name = `Committee ${new Date().toISOString().slice(0, 10)} ${jobId.slice(0, 6)}`;
    saveMutation.mutate(
      { name, full_cycle_job_id: jobId, pair: store.deployedPair, timeframe: store.deployedTimeframe, config_json: results.factory_best_config, trust_score: results.trust_score?.trust_score ?? null, avg_sharpe: results.factory_best_sharpe ?? null },
      { onSuccess: () => setSaveSuccess(true) },
    );
  };

  const rb = results.racecar_backtest;
  const hasBacktest = rb && Object.keys(rb).length > 0;
  const hasRegimeData = !!(rb?.per_regime_summary && Object.keys(rb.per_regime_summary).length > 0);
  const foldResults = rb?.folds_detail ?? [];
  const hasFolds = Array.isArray(foldResults) && foldResults.length > 0;
  const isFailed = results.status === "validation_failed";
  const hasFactory = (results.factory_total_iterations ?? 0) > 0;
  const hasProfileMatrix = !!(results.racecar_profile_matrix && Object.keys(results.racecar_profile_matrix).length > 0);

  const timings = results.phase_timings ?? {};
  const phaseSeconds = (startKey: string, endKey: string) => {
    const s = timings[startKey];
    const e = timings[endKey];
    if (s != null && e != null) return e - s;
    return undefined;
  };

  const phase1Status: "completed" | "skipped" = results.locked_features_count > 0 ? "completed" : "skipped";
  const phase2Status: "completed" | "skipped" = (results.hpo_status && Object.keys(results.hpo_status).length > 0) ? "completed" : "skipped";
  const phase3Status: "completed" | "skipped" = results.racecar_committee_config ? "completed" : "skipped";
  const phase4Status: "completed" | "skipped" | "failed" = isFailed ? "failed" : hasBacktest ? "completed" : "skipped";
  const phase5Status: "completed" | "skipped" = hasFactory ? "completed" : "skipped";

  return (
    <div className="flex flex-col gap-3 pb-4">
      {/* ════════════════ HEADER ════════════════ */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">Committee Results</h2>
          <span className="font-mono text-[11px] text-(--color-text-muted)">{Number(results.total_time_s).toFixed(0)}s</span>
        </div>
        {results.trust_score && (
          <div className="flex items-center gap-2">
            <div
              className="flex h-[36px] w-[36px] items-center justify-center rounded-full"
              style={{
                background: results.trust_score.trust_score >= 0.8 ? "rgba(8,153,129,0.15)" :
                  results.trust_score.trust_score >= 0.6 ? "rgba(242,180,54,0.15)" :
                  results.trust_score.trust_score >= 0.4 ? "rgba(242,145,54,0.15)" :
                  "rgba(242,54,69,0.15)",
                border: `2px solid ${results.trust_score.trust_score >= 0.8 ? "#089981" :
                  results.trust_score.trust_score >= 0.6 ? "#F2B436" :
                  results.trust_score.trust_score >= 0.4 ? "#F29136" : "#F23645"}`,
              }}
            >
              <span className="font-mono text-[13px] font-bold" style={{
                color: results.trust_score.trust_score >= 0.8 ? "#089981" :
                  results.trust_score.trust_score >= 0.6 ? "#F2B436" :
                  results.trust_score.trust_score >= 0.4 ? "#F29136" : "#F23645",
              }}>
                {Number(results.trust_score.trust_score * 100).toFixed(0)}
              </span>
            </div>
            <span className="rounded-[3px] px-[10px] py-[2px] text-[11px] font-semibold tracking-[0.08em] uppercase" style={{
              background: results.trust_score.action === "deploy" ? "rgba(8,153,129,0.15)" :
                results.trust_score.action === "proceed" ? "rgba(242,180,54,0.15)" :
                "rgba(242,54,69,0.15)",
              color: results.trust_score.action === "deploy" ? "#089981" :
                results.trust_score.action === "proceed" ? "#F2B436" : "#F23645",
            }}>
              {results.trust_score.action}
            </span>
          </div>
        )}
      </div>

      {/* ════════════════ KPI TICKER ════════════════ */}
      <CommitteeMetricsGrid results={results} />

      {/* Validation failed warning */}
      {isFailed && (
        <div className="rounded-sm border border-[rgba(242,180,54,0.2)] bg-[rgba(242,180,54,0.06)] p-3 text-[10px] leading-[1.5] text-(--color-text-secondary)">
          <span className="font-semibold text-[#F2B436]">Validation halted pipeline.</span> Phase 5 was skipped. Review diagnostics and adjust parameters.
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
         PHASE 1: Feature Sweep
         ════════════════════════════════════════════════════════════ */}
      <PhaseHeader
        number={1}
        label="Feature Sweep"
        seconds={phaseSeconds("phase1_start", "phase1_end")}
        status={phase1Status}
        summary={phase1Status === "completed" ? `${results.locked_features_count} features locked, ${results.pruned_features_count} pruned` : "Skipped"}
      />

      {phase1Status === "completed" && (
        <FeatureSweepDetail
          lockedFeatures={results.locked_features_list}
          lockedCount={results.locked_features_count}
          prunedCount={results.pruned_features_count}
          topFeature={results.top_importance_feature}
          survivors={results.phase0_survivors}
        />
      )}

      {/* ════════════════════════════════════════════════════════════
         PHASE 2: HPO Tuning
         ════════════════════════════════════════════════════════════ */}
      <PhaseHeader
        number={2}
        label="HPO Tuning"
        seconds={phaseSeconds("phase2_start", "phase2_end")}
        status={phase2Status}
        summary={phase2Status === "completed" ? `${Object.keys(results.hpo_status ?? {}).length} models tuned` : "Skipped"}
      />

      {phase2Status === "completed" && (
        <div className="flex flex-col gap-3">
          {results.hpo_status && Object.keys(results.hpo_status).length > 0 && (
            <BentoCard title={`HPO Status — ${Object.keys(results.hpo_status).length} models`}>
              <div className="flex flex-wrap gap-[4px]">
                {Object.entries(results.hpo_status).map(([model, statusVal]) => {
                  const sc = statusVal === "success" ? "#089981" :
                    statusVal === "timed_out" ? "#F2B436" :
                    statusVal === "crashed" || statusVal === "no_folds" ? "#F23645" : "var(--color-text-dim)";
                  const sbg = statusVal === "success" ? "rgba(8,153,129,0.12)" :
                    statusVal === "timed_out" ? "rgba(242,180,54,0.12)" :
                    statusVal === "crashed" || statusVal === "no_folds" ? "rgba(242,54,69,0.12)" : "var(--color-elevated)";
                  return (
                    <span key={model} className="flex items-center gap-[5px] rounded-[3px] px-[8px] py-[3px] font-mono text-[10px]"
                      style={{ background: sbg, border: `1px solid ${sc}33`, color: sc }}>
                      <span className="font-medium">{model}</span>
                      <span className="text-[9px] opacity-70">{statusVal.replace(/_/g, " ")}</span>
                    </span>
                  );
                })}
              </div>
            </BentoCard>
          )}

          {hasProfileMatrix && <RegimeMatrixHeatmap profileMatrix={results.racecar_profile_matrix} />}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
         PHASE 3: Committee Assembly
         ════════════════════════════════════════════════════════════ */}
      <PhaseHeader
        number={3}
        label="Committee Assembly"
        seconds={phaseSeconds("phase3_start", "phase3_end")}
        status={phase3Status}
        summary={phase3Status === "completed" ? "Regime-based committee built" : "Skipped"}
      />

      {phase3Status === "completed" && results.racecar_committee_config && (
        <BentoCard title="Assembled Committee Config">
          <div className="flex flex-col gap-[6px]">
            {Object.entries(
              ((results.racecar_committee_config as Record<string, unknown>).regimes as Record<string, Record<string, unknown>>) ?? {},
            ).map(([regime, a]) => (
              <div key={regime} className="flex items-center gap-[14px] text-[11px]">
                <span className="w-20 sm:w-[130px] shrink-0 font-medium tracking-[0.06em] text-(--color-brand) uppercase">
                  {regime.replace(/_/g, " ")}
                </span>
                <div className="flex flex-wrap gap-[4px]">
                  {(a.models as string[])?.map((m: string, i: number) => {
                    const w = Number((a.weights as number[])?.[i] ?? 0);
                    return (
                      <span key={`${m}-${i}`} className="rounded-[3px] border border-(--color-glass-border) bg-(--color-elevated) px-[8px] py-[3px] font-mono text-[10px] text-(--color-text-secondary)">
                        {m} {(w * 100).toFixed(0)}%
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </BentoCard>
      )}

      {/* ════════════════════════════════════════════════════════════
         PHASE 4: Validation (WFO Backtest)
         ════════════════════════════════════════════════════════════ */}
      <PhaseHeader
        number={4}
        label="Validation"
        seconds={phaseSeconds("phase4_start", "phase4_end")}
        status={phase4Status}
        summary={isFailed ? "Validation failed — pipeline halted" :
          phase4Status === "completed" ? `${rb?.folds ?? 0} folds, avg Sharpe ${rb?.avg_sharpe?.toFixed(2) ?? "?"}` : "Skipped"}
      />

      {phase4Status !== "skipped" && (
        <div className="flex flex-col gap-3">
          {/* Equity + Drawdown charts */}
          <EquityDrawdownCharts
            equity={results.racecar_backtest?.equity_curve}
            drawdown={results.drawdown_curve}
            buyHold={results.buy_hold_curve}
            monthly={results.monthly_returns}
          />

          {/* Regime + Fold Consistency */}
          <div className="grid grid-cols-1 gap-3 items-start lg:grid-cols-2">
            {hasRegimeData && (
              <BentoCard title="Regime-Stratified Performance">
                <CommitteeRegimeChart racecarBacktest={rb} />
              </BentoCard>
            )}
            {hasFolds && (
              <BentoCard title="Fold Consistency">
                <CommitteeFoldChart
                  foldResults={foldResults}
                  foldCv={results.final_fold_consistency_cv || results.phase3_fold_consistency_cv}
                  foldCvPass={results.final_fold_consistency_pass}
                  avgSharpe={rb?.avg_sharpe}
                />
              </BentoCard>
            )}
          </div>

          {/* Vote Agreement + Model Contribution */}
          {(rb?.diagnostics?.vote_agreement || rb?.diagnostics?.model_contributions) && (
            <div className="grid grid-cols-1 gap-3 items-start lg:grid-cols-2">
              {rb?.diagnostics?.vote_agreement && (
                <BentoCard title="Committee Vote Agreement">
                  <CommitteeVoteAgreementChart voteAgreement={rb.diagnostics.vote_agreement} />
                </BentoCard>
              )}
              {rb?.diagnostics?.model_contributions && (
                <BentoCard title="Model Contribution">
                  <ModelContributionChart contributions={rb.diagnostics.model_contributions} />
                </BentoCard>
              )}
            </div>
          )}

          {/* Model Agreement Matrix */}
          {rb?.diagnostics?.model_agreement && (
            <BentoCard title="Model Agreement">
              <ModelAgreementMatrix modelAgreement={rb.diagnostics.model_agreement} />
            </BentoCard>
          )}

          {/* Trade Log */}
          {rb?.trades && rb.trades.length > 0 && (
            <BentoCard
              title={`Trade Log — ${rb.trades.length} trades`}
              right={
                <span className="font-mono text-[9px] text-(--color-text-muted)">
                  {rb.trades.filter((t: { return_pct: number }) => t.return_pct > 0).length}/{rb.trades.length} positive
                </span>
              }
            >
              <div className="max-h-[300px] overflow-auto">
                <table className="min-w-[500px] w-full border-collapse text-[10px]">
                  <thead>
                    <tr>
                      <th className="border-b border-(--color-glass-border) px-2 py-1 text-left font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Entry</th>
                      <th className="border-b border-(--color-glass-border) px-2 py-1 text-left font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Exit</th>
                      <th className="border-b border-(--color-glass-border) px-2 py-1 text-center font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Dir</th>
                      <th className="border-b border-(--color-glass-border) px-2 py-1 text-right font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Return</th>
                      <th className="hidden sm:table-cell border-b border-(--color-glass-border) px-2 py-1 text-right font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Bars</th>
                      <th className="hidden sm:table-cell border-b border-(--color-glass-border) px-2 py-1 text-left font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">Regime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rb.trades.slice(-20).reverse().map((t: { entry_time: string; exit_time: string; direction: string; return_pct: number; duration_bars: number; regime: string }, i: number) => (
                      <tr key={i} className="border-b border-(--color-glass-border)">
                        <td className="px-2 py-1 font-mono text-(--color-text-secondary)">{t.entry_time?.slice(0, 19) || "—"}</td>
                        <td className="px-2 py-1 font-mono text-(--color-text-secondary)">{t.exit_time?.slice(0, 19) || "—"}</td>
                        <td className="px-2 py-1 text-center font-bold" style={{ color: t.direction === "BUY" ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>{t.direction}</td>
                        <td className="px-2 py-1 text-right font-mono tabular-nums" style={{ color: t.return_pct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>{t.return_pct >= 0 ? "+" : ""}{(t.return_pct * 100).toFixed(3)}%</td>
                        <td className="hidden sm:table-cell px-2 py-1 text-right font-mono text-(--color-text-muted)">{t.duration_bars}</td>
                        <td className="hidden sm:table-cell px-2 py-1 font-mono text-[9px] text-(--color-text-dim)">{t.regime.replace(/_/g, " ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </BentoCard>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
         PHASE 5: Factory Optimization
         ════════════════════════════════════════════════════════════ */}
      <PhaseHeader
        number={5}
        label="Factory Optimization"
        seconds={phaseSeconds("phase5_start", "phase5_end")}
        status={phase5Status}
        summary={hasFactory
          ? `${results.factory_accepted_count}/${results.factory_total_iterations} accepted, best Sharpe ${results.factory_best_sharpe.toFixed(2)}`
          : "Skipped"}
      />

      {hasFactory && (
        <div className="flex flex-col gap-3">
          <BentoCard
            title="Factory Optimization"
            right={
              <div className="flex items-center gap-3 font-mono text-[10px] text-(--color-text-muted)">
                <span>Best Sharpe: <span className="text-(--color-accent-success)">{results.factory_best_sharpe.toFixed(4)}</span></span>
                <span>{results.factory_total_iterations} iters</span>
                <span>{results.factory_accepted_count} accepted</span>
              </div>
            }
          >
            {(results.factory_history?.length ?? 0) > 0 && (
              <div className="max-h-[300px] overflow-auto">
                <table className="min-w-[500px] w-full border-collapse text-[10px]">
                  <thead>
                    <tr>
                      <th className={`border-b border-(--color-glass-border) ${TH_CLASSES}`} style={TH_STYLE}>#</th>
                      <th className={`border-b border-(--color-glass-border) ${TH_CLASSES}`} style={TH_STYLE}>Action</th>
                      <th className={`border-b border-(--color-glass-border) ${TH_CLASSES}`} style={TH_STYLE}>Regime</th>
                      <th className={`border-b border-(--color-glass-border) ${TH_CLASSES}`} style={TH_STYLE}>Model Change</th>
                      <th className={`border-b border-(--color-glass-border) ${TH_CLASSES} text-right`} style={TH_STYLE}>Sharpe Δ</th>
                      <th className={`border-b border-(--color-glass-border) ${TH_CLASSES}`} style={TH_STYLE}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {(results.factory_history ?? []).map((row: FactoryIterationRecord, i: number) => {
                      const delta = row.after_sharpe - row.before_sharpe;
                      const a = row.action;
                      return (
                        <tr key={i} className="border-b border-(--color-glass-border)">
                          <td className={TD_MONO_CLASSES} style={TD_MONO_STYLE}>{row.iteration}</td>
                          <td style={{ color: "var(--color-text-secondary)", padding: "0.375rem 0.5rem", fontSize: 11 }}>{a.type.toUpperCase()}</td>
                          <td className={TD_CLASSES} style={TD_STYLE}>{a.regime.replace(/_/g, " ")}</td>
                          <td style={{ color: "var(--color-text-dim)", padding: "0.375rem 0.5rem", fontSize: 11 }}>
                            {[a.model_add, a.model_remove].filter(Boolean).join(" / ") || "—"}
                          </td>
                          <td className={`${TD_MONO_CLASSES} text-right`} style={{ color: delta >= 0 ? "#089981" : "#F23645" }}>
                            {delta >= 0 ? "+" : ""}{Number(delta).toFixed(4)}
                          </td>
                          <td className={TD_CLASSES + " text-center text-xs"} style={TD_STYLE}>
                            {row.accepted ? <span className="text-[#089981]">✓</span> : <span className="text-[#F23645]">✗</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {results.factory_stop_reason && (
              <div className="mt-[8px] rounded-[4px] border border-[rgba(0,229,255,0.15)] bg-[rgba(0,229,255,0.05)] p-[8px] text-[11px] tracking-[0.04em] text-(--color-brand)">
                Stopped — {formatStopReason(results.factory_stop_reason)}
              </div>
            )}
          </BentoCard>

          {results.factory_best_config && (
            <BentoCard title="Optimized Committee Config">
              <div className="flex flex-col gap-[6px]">
                {Object.entries(
                  ((results.factory_best_config as Record<string, unknown>).regimes as Record<string, Record<string, unknown>>) ?? {},
                ).map(([regime, a]) => (
                  <div key={regime} className="flex items-center gap-[14px] text-[11px]">
                    <span className="w-20 sm:w-[130px] shrink-0 font-medium tracking-[0.06em] text-(--color-brand) uppercase">
                      {regime.replace(/_/g, " ")}
                    </span>
                    <div className="flex flex-wrap gap-[4px]">
                      {(a.models as string[])?.map((m: string, i: number) => {
                        const w = Number((a.weights as number[])?.[i] ?? 0);
                        return (
                          <span key={`${m}-${i}`} className="rounded-[3px] border border-(--color-glass-border) bg-(--color-elevated) px-[8px] py-[3px] font-mono text-[10px] text-(--color-text-secondary)">
                            {m} {(w * 100).toFixed(0)}%
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </BentoCard>
          )}
        </div>
      )}

      {/* ════════════════ ACTIONS ════════════════ */}
      <div className="mt-1 flex flex-wrap gap-[12px]">
        <button onClick={onRunAgain} className="cursor-pointer rounded-sm border border-(--color-glass-border) bg-(--color-elevated) px-5 py-2 text-[10px] font-medium tracking-[0.06em] text-(--color-text-secondary) uppercase">Run Again</button>
        <button onClick={handleDownload} className="cursor-pointer rounded-sm border border-(--color-glass-border) bg-(--color-elevated) px-5 py-2 text-[10px] font-medium tracking-[0.06em] text-(--color-text-secondary) uppercase">Download JSON</button>
        {results.factory_best_config && !isFailed && (
          <>
            <button onClick={handleSaveCommittee} disabled={saveMutation.isPending || saveSuccess}
              className="cursor-pointer rounded-sm border border-[rgba(0,229,255,0.25)] bg-[rgba(0,229,255,0.08)] px-5 py-2 text-[10px] font-semibold tracking-[0.06em] text-(--color-brand) uppercase">
              {saveSuccess ? "Saved" : saveMutation.isPending ? "Saving..." : "Save Committee"}
            </button>
            <button onClick={handleDeploy} className="cursor-pointer rounded-sm border-none bg-(--color-accent-success) px-5 py-2 text-[10px] font-semibold tracking-[0.06em] uppercase text-white">
              Deploy to Trading
            </button>
          </>
        )}
      </div>
    </div>
  );
}
