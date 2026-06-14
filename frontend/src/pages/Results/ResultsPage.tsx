import { useParams, useNavigate } from "react-router-dom";
import { useState, useRef, useMemo } from "react";
import {
  BarChart3,
  ArrowLeft,
  Play,
  Save,
  GitCompare,
  ChevronDown,
  ChevronUp,
  X,
} from "lucide-react";
import { useJobResults, useTradeChartData, useSaveModelFromJob } from "@/api/queries";
import { EmptyState } from "@/components/shared/EmptyState";
import { ExportBar } from "@/components/shared/ExportBar";
import { StatusDot } from "@/components/shared/StatusDot";
import { MetricsGrid } from "./MetricsGrid";
import { EquitySection } from "./EquitySection";
import { TradeLogTable } from "./TradeLogTable";
import { MonthlySection } from "./MonthlySection";
import { HpoDiagnostics } from "./HpoDiagnostics";
import { OverfittingPanel } from "./OverfittingPanel";
import { ValidationScorecard } from "./ValidationScorecard";
import { WalkForwardPanel } from "./WalkForwardPanel";
import { ParameterExplorer } from "./ParameterExplorer";
import { LLMAdvisor } from "./LLMAdvisor";
import { TrainingDiagnosticsPanel } from "./TrainingDiagnostics";
import { ConfigViewer } from "./ConfigViewer";
import { LeaderboardTable } from "../Compare/LeaderboardTable";
import { EquityOverlayChart } from "../Compare/EquityOverlayChart";
import { SignificanceMatrix } from "../Compare/SignificanceMatrix";
import { CrossPairSection } from "../Compare/CrossPairSection";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { TradeDistributionChart } from "@/components/charts/TradeDistributionChart";
import { RollingMetricsChart } from "@/components/charts/RollingMetricsChart";
import { CumulativePnlChart } from "@/components/charts/CumulativePnlChart";
import { ParameterSensitivityChart } from "@/components/charts/ParameterSensitivityChart";
import { normalizeEquityCurve } from "@/lib/chartUtils";
import { BacktestChart } from "./BacktestChart";
import { BacktestPlayback } from "./BacktestPlayback";
import type { TradeRecord } from "@/api/schemas";
import type { EquityCurveChartHandle } from "@/components/charts/EquityCurveChart";

function Skeleton() {
  return (
    <div className="flex animate-pulse flex-col gap-3">
      <div className="h-5 w-48 rounded bg-(--color-elevated)" />
      <div className="h-12 rounded bg-(--color-surface)" />
      <div className="h-[320px] rounded bg-(--color-surface)" />
      <div className="grid grid-cols-3 gap-3">
        {Array.from({ length: 3 }, (_, i) => (
          <div key={i} className="h-[220px] rounded bg-(--color-surface)" />
        ))}
      </div>
    </div>
  );
}

/** Thin collapsible accordion wrapper */
function Accordion({
  label,
  open,
  onToggle,
  children,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-sm border border-(--color-glass-border)">
      <button
        onClick={onToggle}
        className="flex w-full cursor-pointer items-center justify-between bg-(--color-surface) px-4 py-2 text-left font-sans text-[11px] font-semibold tracking-[0.1em] text-(--color-text-muted) uppercase transition-all hover:opacity-80"
        style={{ border: "none" }}
      >
        {label}
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>
      {open && <div className="bg-(--color-app)">{children}</div>}
    </div>
  );
}

export function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [activeModelIdx, setActiveModelIdx] = useState(0);
  const [selectedTrade, setSelectedTrade] = useState<TradeRecord | null>(null);
  const [showPlayback, setShowPlayback] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<"results" | "compare">("results");

  // Progressive disclosure accordion states
  const [showWalkForward, setShowWalkForward] = useState(true);
  const [showMonthly, setShowMonthly] = useState(true);
  const [showDiagnostics, setShowDiagnostics] = useState(true);
  const [showTradeLog, setShowTradeLog] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: results, isLoading, isError } = useJobResults(jobId ?? null);
  const saveModelMutation = useSaveModelFromJob();
  const equityChartRef = useRef<EquityCurveChartHandle>(null);

  const activeMetric = results?.metrics?.length
    ? results.metrics[Math.min(activeModelIdx, results.metrics.length - 1)]
    : null;
  const canSaveModel =
    activeMetric?.snapshot_path &&
    activeMetric.total_trades != null &&
    activeMetric.total_trades > 0 &&
    activeMetric.sharpe != null &&
    isFinite(activeMetric.sharpe);
  const metrics = results?.metrics ?? [];
  const modelCurves = useMemo(() => {
    if (!metrics.length) return [];
    return metrics
      .filter((m) => m.equity_curve && m.equity_curve.length > 0)
      .map((m) => ({
        model: m.model,
        data: m.equity_curve!,
      }));
  }, [metrics]);
  const { data: tradeChartData } = useTradeChartData(jobId ?? "", activeMetric?.model ?? "");

  const handleExportPng = () => {
    equityChartRef.current?.takeScreenshot();
  };

  if (!jobId) {
    return (
      <div className="flex flex-col gap-5">
        <EmptyState
          icon={<BarChart3 size={48} strokeWidth={1} />}
          title="No results to display"
          description="Run a backtest first, then navigate here to see equity curves, metrics, trade logs, and HPO diagnostics."
          actionLabel="Run Backtest"
          onAction={() => navigate("/backtest")}
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1 text-xs text-(--color-text-muted) transition-all duration-200 hover:border-[var(--color-border-active)]"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Back
          </button>
        </div>
        <Skeleton />
      </div>
    );
  }

  if (isError || !results) {
    return (
      <div className="flex flex-col gap-5">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1 text-xs text-(--color-text-muted) transition-all duration-200 hover:border-[var(--color-border-active)]"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Back
          </button>
        </div>
        <EmptyState
          icon={<BarChart3 size={48} strokeWidth={1} />}
          title="Failed to load results"
          description="The job may not exist or hasn't completed yet."
          actionLabel="Run Backtest"
          onAction={() => navigate("/backtest")}
        />
      </div>
    );
  }

  const handleExportCsv = () => {
    if (!activeMetric?.trades?.length) return;
    const trades = activeMetric.trades;
    const header = Object.keys(trades[0]).join(",");
    const rows = trades.map((t: Record<string, unknown>) =>
      Object.values(t)
        .map((v) => String(v ?? ""))
        .join(","),
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades_${results.job_id}_${activeMetric?.model ?? "unknown"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJson = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `results_${results.job_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full animate-fade-in flex-col gap-5">
      {/* ── Top ribbon ─────────────────────────────────────────────── */}
      <div className="flex min-h-[36px] items-center justify-between">
        <div className="flex items-center gap-3 leading-none">
          <button
            onClick={() => navigate("/backtest")}
            className="flex cursor-pointer items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1.5 text-[11px] leading-none text-(--color-text-muted) transition-all duration-200 hover:border-[var(--color-brand)]"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Back
          </button>
          <h2 className="text-[11px] font-semibold leading-none tracking-[0.12em] text-(--color-text-muted) uppercase">
            Results
          </h2>
          <span className="font-mono text-[11px] leading-none text-(--color-text-secondary)">
            {results.pair}
          </span>
          <StatusDot color="var(--color-brand)" />
        </div>
        <div className="flex items-center gap-1.5 leading-none">
          {canSaveModel && (
            <>
              {saveMsg ? (
                <button
                  onClick={() => navigate("/models")}
                  className="flex cursor-pointer items-center gap-1.5 rounded-md border border-(--color-accent-success) bg-[rgba(8,153,129,0.1)] px-2.5 py-1 text-[11px] font-semibold text-(--color-accent-success) uppercase transition-all hover:brightness-110"
                >
                  <Save size={11} /> View in Models
                </button>
              ) : (
                <button
                  onClick={() => {
                    setSaveMsg(null);
                    saveModelMutation.mutate(
                      { jobId: jobId!, modelName: activeMetric.model },
                      {
                        onSuccess: () => setSaveMsg("Saved"),
                        onError: (e: unknown) => setSaveMsg(`Error: ${(e as Error).message}`),
                      },
                    );
                  }}
                  disabled={saveModelMutation.isPending}
                  className="flex items-center gap-1.5 rounded-md border border-(--color-accent-warning) bg-[rgba(245,158,11,0.1)] px-2.5 py-1 text-[11px] font-semibold text-(--color-accent-warning) uppercase transition-all hover:brightness-110"
                  style={{
                    cursor: saveModelMutation.isPending ? "not-allowed" : "pointer",
                    opacity: saveModelMutation.isPending ? 0.6 : 1,
                  }}
                >
                  <Save size={11} /> {saveModelMutation.isPending ? "Saving..." : "Save Model"}
                </button>
              )}
            </>
          )}
          <ExportBar
            onExportCsv={handleExportCsv}
            onExportPng={handleExportPng}
            onExportJson={handleExportJson}
          />
          {jobId && activeMetric?.model && (
            <button
              onClick={() => setShowPlayback(true)}
              className="flex items-center gap-1.5 rounded-md border border-(--color-accent-success) bg-[rgba(8,153,129,0.1)] px-2.5 py-1 text-[11px] font-semibold text-(--color-accent-success) uppercase transition-all"
            >
              <Play size={11} /> Replay
            </button>
          )}
        </div>
      </div>

      {/* ── No-trades warning ───────────────────────────────────────── */}
      {activeMetric && (activeMetric.total_trades ?? 0) === 0 && (
        <div className="rounded-md border border-(--color-accent-warning) bg-[rgba(245,158,11,0.05)] px-3 py-2">
          <p className="text-[11px] text-(--color-accent-warning)">
            This backtest produced no trades. All walk-forward months were flat. Try increasing HPO
            trials, tightening bounds, or selecting a different model.
          </p>
        </div>
      )}

      {/* ── KPI ticker tape ─────────────────────────────────────────── */}
      {activeMetric && (
        <MetricsGrid
          metrics={activeMetric}
          modelName={activeMetric.model}
          warnings={
            activeMetric.diagnostics?.vif_warnings?.map((w) => `${w.feature} VIF=${w.vif}`) ?? []
          }
        />
      )}

      {/* ── Model selector (multi-model runs) ───────────────────────── */}
      {metrics.length > 1 && (
        <div className="flex flex-wrap items-center gap-1">
          {metrics.map((m, i) => (
            <button
              key={m.model}
              onClick={() => setActiveModelIdx(i)}
              className="cursor-pointer rounded-md border px-3 py-1 text-[11px] font-medium tracking-[0.06em] uppercase transition-all duration-200"
              style={{
                borderColor:
                  i === activeModelIdx ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: i === activeModelIdx ? "rgba(0,229,255,0.07)" : "transparent",
                color: i === activeModelIdx ? "var(--color-brand)" : "var(--color-text-muted)",
              }}
            >
              {m.model} — Sharpe {m.sharpe?.toFixed(2) ?? "—"}
            </button>
          ))}
        </div>
      )}

      {/* ── Results / Compare tab bar (multi-model) ─────────────────── */}
      {metrics.length > 1 && (
        <div className="flex items-center gap-1 border-b border-(--color-surface)">
          <button
            onClick={() => setTab("results")}
            className="relative cursor-pointer bg-transparent px-4 py-1.5 text-[11px] font-semibold tracking-[0.08em] uppercase transition-all duration-200"
            role="tab"
            aria-selected={tab === "results"}
            style={{
              color: tab === "results" ? "var(--color-brand)" : "var(--color-text-muted)",
              border: "none",
              borderBottom:
                tab === "results" ? "2px solid var(--color-brand)" : "2px solid transparent",
            }}
          >
            <BarChart3 size={11} className="mr-1.5 inline" />
            Results
          </button>
          <button
            onClick={() => setTab("compare")}
            className="relative cursor-pointer bg-transparent px-4 py-1.5 text-[11px] font-semibold tracking-[0.08em] uppercase transition-all duration-200"
            role="tab"
            aria-selected={tab === "compare"}
            style={{
              color: tab === "compare" ? "var(--color-brand)" : "var(--color-text-muted)",
              border: "none",
              borderBottom:
                tab === "compare" ? "2px solid var(--color-brand)" : "2px solid transparent",
            }}
          >
            <GitCompare size={11} className="mr-1.5 inline" />
            Compare
          </button>
        </div>
      )}

      {tab === "results" && (
        <div className="flex flex-col gap-3 pb-4">
          {/* ── Validation Scorecard ─────────────────────────────── */}
          {(activeMetric?.summary_text || activeMetric?.overfitting || activeMetric?.walkforward_periods) && (
            <ValidationScorecard
              overfitting={activeMetric.overfitting ?? null}
              walkforwardPeriods={activeMetric.walkforward_periods ?? null}
            />
          )}

          {/* ── HERO: main chart area ───────────────────────────────── */}
          <div className="overflow-hidden rounded-sm border border-(--color-glass-border)">
            {/* Trade visualization */}
            {jobId && activeMetric?.model && (
              <div className="border-b border-(--color-glass-border)">
                <BacktestChart jobId={jobId} model={activeMetric.model} />
              </div>
            )}
            {/* Equity + Drawdown stacked below */}
            <div className="bg-(--color-app) px-3 pt-2 pb-3">
              <EquitySection
                ref={equityChartRef}
                equityCurve={normalizeEquityCurve(activeMetric?.equity_curve ?? null)}
                buyHoldCurve={normalizeEquityCurve(activeMetric?.buy_hold_curve ?? null)}
                drawdownCurve={normalizeEquityCurve(activeMetric?.drawdown_curve ?? null)}
              />
            </div>
          </div>

          {/* ── LLM advisor ─────────────────────────────────────────── */}
          <LLMAdvisor jobId={jobId ?? null} modelName={activeMetric?.model ?? null} />

          {/* ── 3-column diagnostics grid ───────────────────────────── */}
          <div className="grid grid-cols-3 gap-3">
            {/* Col 1: Walk-Forward */}
            <Accordion
              label={`Walk-Forward ${activeMetric?.walkforward_periods ? `— ${activeMetric.walkforward_periods.length} periods` : ""}`}
              open={showWalkForward}
              onToggle={() => setShowWalkForward((v) => !v)}
            >
              <div className="p-2">
                <WalkForwardPanel
                  periods={activeMetric?.walkforward_periods ?? null}
                  modelName={activeMetric?.model ?? ""}
                />
              </div>
            </Accordion>

            {/* Col 2: Monthly Returns */}
            <Accordion
              label="Monthly Returns"
              open={showMonthly}
              onToggle={() => setShowMonthly((v) => !v)}
            >
              <div className="p-2">
                <MonthlySection monthlyResults={activeMetric?.monthly_results ?? null} />
              </div>
            </Accordion>

            {/* Col 3: Training Diagnostics */}
            <Accordion
              label={`Training Diagnostics — ${activeMetric?.model ?? ""}`}
              open={showDiagnostics}
              onToggle={() => setShowDiagnostics((v) => !v)}
            >
              <div className="p-2">
                <TrainingDiagnosticsPanel
                  data={activeMetric?.diagnostics ?? null}
                  modelName={activeMetric?.model ?? ""}
                />
              </div>
            </Accordion>
          </div>

          {/* ── Additional charts (collapsible) ─────────────────────── */}
          <Accordion
            label="Advanced Analytics"
            open={showAdvanced}
            onToggle={() => setShowAdvanced((v) => !v)}
          >
            <div className="flex flex-col gap-8 p-6">
              <DrawdownChart
                drawdownCurve={normalizeEquityCurve(activeMetric?.drawdown_curve ?? null)}
              />
              <CumulativePnlChart
                trades={activeMetric?.trades ? (activeMetric.trades as TradeRecord[]) : null}
              />
              <RollingMetricsChart
                equityCurve={normalizeEquityCurve(activeMetric?.equity_curve ?? null)}
              />
              <TradeDistributionChart
                trades={activeMetric?.trades ? (activeMetric.trades as TradeRecord[]) : null}
              />
              <HpoDiagnostics
                paramImportance={activeMetric?.hpo_param_importance ?? null}
                trials={activeMetric?.hpo_trials ?? null}
              />
              {activeMetric?.overfitting && (
                <OverfittingPanel
                  overfitting={activeMetric.overfitting}
                  walkforwardPeriods={activeMetric.walkforward_periods ?? null}
                />
              )}
              <ParameterSensitivityChart trials={activeMetric?.hpo_trials ?? null} />
              {activeMetric && <ParameterExplorer metrics={activeMetric} />}
              <ConfigViewer config={results.config ?? null} />
            </div>
          </Accordion>

          {/* ── Trade log ───────────────────────────────────────────── */}
          <Accordion
            label={`Trade Log — ${activeMetric?.total_trades ?? 0} trades`}
            open={showTradeLog}
            onToggle={() => setShowTradeLog((v) => !v)}
          >
            <div className="p-2">
              <TradeLogTable
                trades={
                  activeMetric?.trades
                    ? (activeMetric.trades as import("@/api/schemas").TradeRecord[])
                    : null
                }
                onTradeSelect={setSelectedTrade}
              />
            </div>
          </Accordion>
        </div>
      )}

      {tab === "compare" && (
        <div className="flex flex-col gap-3 pb-4">
          <LeaderboardTable metrics={metrics} sortMetric="sharpe" />
          <EquityOverlayChart curves={modelCurves} />
          <SignificanceMatrix models={metrics.map((m) => m.model)} pValues={null} />
          {metrics.length > 0 && metrics[0].hpo_trials && (
            <ParameterSensitivityChart trials={metrics[0].hpo_trials} />
          )}
          <CrossPairSection />
        </div>
      )}

      {/* ── Selected trade tooltip ───────────────────────────────────── */}
      {selectedTrade && (
        <div className="fixed right-6 bottom-12 z-50 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3 font-mono text-xs backdrop-blur-[12px]">
          <div className="flex items-center justify-between gap-4">
            <span className="text-(--color-text-muted)">Trade #{selectedTrade.trade_id}</span>
            <span
              style={{
                color:
                  selectedTrade.direction === "BUY"
                    ? "var(--color-accent-success)"
                    : "var(--color-accent-danger)",
              }}
            >
              {selectedTrade.direction}
            </span>
            <span
              style={{
                color:
                  (selectedTrade.return_pct ?? 0) >= 0
                    ? "var(--color-accent-success)"
                    : "var(--color-accent-danger)",
              }}
            >
              {(selectedTrade.return_pct ?? 0) >= 0 ? "+" : ""}
              {(selectedTrade.return_pct ?? 0).toFixed(2)}%
            </span>
            <button
              onClick={() => setSelectedTrade(null)}
              className="cursor-pointer bg-none text-(--color-text-muted)"
              style={{ border: "none" }}
              aria-label="Close trade tooltip"
            >
              <X size={12} />
            </button>
          </div>
        </div>
      )}

      {/* ── Playback modal ───────────────────────────────────────────── */}
      {showPlayback && jobId && activeMetric?.model && tradeChartData && (
        <BacktestPlayback
          candles={tradeChartData.candles ?? []}
          trades={tradeChartData.trades ?? []}
          equityCurve={tradeChartData.equity_curve ?? []}
          monthlyResults={activeMetric.monthly_results ?? null}
          hpoTrials={activeMetric.hpo_trials ?? null}
          pair={results?.pair ?? ""}
          model={activeMetric.model}
          timeframe={results?.config?.timeframe ?? "M30"}
          onClose={() => setShowPlayback(false)}
        />
      )}
    </div>
  );
}
