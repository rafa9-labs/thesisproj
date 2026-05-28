import { useParams, useNavigate } from "react-router-dom";
import { useState, useRef, useMemo } from "react";
import { BarChart3, ArrowLeft, Play, Save, GitCompare, ChevronDown, ChevronUp, X } from "lucide-react";
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
import { BacktestSummary } from "./BacktestSummary";
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
    <div className="flex flex-col gap-3 animate-pulse">
      <div className="h-5 w-48 rounded" style={{ backgroundColor: "var(--color-elevated)" }} />
      <div className="h-12 rounded" style={{ backgroundColor: "var(--color-surface)" }} />
      <div className="h-[320px] rounded" style={{ backgroundColor: "var(--color-surface)" }} />
      <div className="grid grid-cols-3 gap-3">
        {Array.from({ length: 3 }, (_, i) => (
          <div key={i} className="h-[220px] rounded" style={{ backgroundColor: "var(--color-surface)" }} />
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
    <div style={{ border: "1px solid #2A2E39", borderRadius: 8, overflow: "hidden" }}>
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-2 text-left transition-all hover:opacity-80"
        style={{
          backgroundColor: "#1E222D",
          color: "#787B86",
          fontFamily: "Inter, sans-serif",
          fontSize: 11,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.1em",
          cursor: "pointer",
          border: "none",
        }}
      >
        {label}
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>
      {open && (
        <div style={{ backgroundColor: "#131722" }}>
          {children}
        </div>
      )}
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
  const [showSummary, setShowSummary] = useState(false);
  const [showOverfitting, setShowOverfitting] = useState(false);
  const [showWalkForward, setShowWalkForward] = useState(false);
  const [showMonthly, setShowMonthly] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(true);
  const [showTradeLog, setShowTradeLog] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: results, isLoading, isError } = useJobResults(jobId ?? null);
  const saveModelMutation = useSaveModelFromJob();
  const equityChartRef = useRef<EquityCurveChartHandle>(null);

  const activeMetric = results?.metrics?.length ? results.metrics[Math.min(activeModelIdx, results.metrics.length - 1)] : null;
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
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "transparent",
              color: "var(--color-text-muted)",
            }}
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
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "transparent",
              color: "var(--color-text-muted)",
            }}
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
      Object.values(t).map((v) => String(v ?? "")).join(",")
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
    <div className="flex flex-col gap-3 overflow-y-auto animate-fade-in" style={{ height: "100%" }}>

      {/* ── Top ribbon ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between" style={{ minHeight: 32 }}>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "#2A2E39",
              backgroundColor: "transparent",
              color: "#787B86",
              cursor: "pointer",
            }}
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Back
          </button>
          <h2
            className="text-[11px] font-semibold uppercase tracking-[0.12em]"
            style={{ color: "#787B86" }}
          >
            Results
          </h2>
          <span
            className="text-[11px]"
            style={{ color: "#4A5568", fontFamily: "JetBrains Mono, monospace" }}
          >
            {results.pair}
          </span>
          <StatusDot color="var(--color-brand)" />
        </div>
        <div className="flex items-center gap-1.5">
          {activeMetric?.snapshot_path && (
            <>
              {saveMsg ? (
                <button
                  onClick={() => navigate("/models")}
                  className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-semibold uppercase transition-all hover:brightness-110"
                  style={{ borderColor: "#089981", backgroundColor: "rgba(8,153,129,0.1)", color: "#089981", cursor: "pointer" }}
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
                      }
                    );
                  }}
                  disabled={saveModelMutation.isPending}
                  className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-semibold uppercase transition-all hover:brightness-110"
                  style={{ borderColor: "#F59E0B", backgroundColor: "rgba(245,158,11,0.1)", color: "#F59E0B", cursor: saveModelMutation.isPending ? "not-allowed" : "pointer", opacity: saveModelMutation.isPending ? 0.6 : 1 }}
                >
                  <Save size={11} /> {saveModelMutation.isPending ? "Saving..." : "Save Model"}
                </button>
              )}
            </>
          )}
          <ExportBar onExportCsv={handleExportCsv} onExportPng={handleExportPng} onExportJson={handleExportJson} />
          {jobId && activeMetric?.model && (
            <button
              onClick={() => setShowPlayback(true)}
              className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-semibold uppercase transition-all"
              style={{
                borderColor: "#089981",
                backgroundColor: "rgba(8,153,129,0.1)",
                color: "#089981",
              }}
            >
              <Play size={11} /> Replay
            </button>
          )}
        </div>
      </div>

      {/* ── No-trades warning ───────────────────────────────────────── */}
      {activeMetric && (activeMetric.total_trades ?? 0) === 0 && (
        <div
          className="rounded-md border px-3 py-2"
          style={{ borderColor: "#F59E0B", backgroundColor: "rgba(245,158,11,0.05)" }}
        >
          <p className="text-[11px]" style={{ color: "#F59E0B" }}>
            This backtest produced no trades. All walk-forward months were flat. Try increasing HPO trials, tightening bounds, or selecting a different model.
          </p>
        </div>
      )}

      {/* ── KPI ticker tape ─────────────────────────────────────────── */}
      {activeMetric && (
        <MetricsGrid
          metrics={activeMetric}
          modelName={activeMetric.model}
          warnings={activeMetric.diagnostics?.vif_warnings?.map((w) => `${w.feature} VIF=${w.vif}`) ?? []}
          overfittingScore={activeMetric.overfitting?.cv_sharpe_std ?? null}
          overfittingColor={activeMetric.overfitting?.risk_color ?? null}
          onShowSummary={() => setShowSummary((v) => !v)}
          onShowOverfitting={() => setShowOverfitting((v) => !v)}
        />
      )}

      {/* ── Model selector (multi-model runs) ───────────────────────── */}
      {metrics.length > 1 && (
        <div className="flex items-center gap-1 flex-wrap">
          {metrics.map((m, i) => (
            <button
              key={m.model}
              onClick={() => setActiveModelIdx(i)}
              className="rounded-md border px-3 py-1 text-[11px] font-medium uppercase tracking-[0.06em] transition-all duration-200"
              style={{
                borderColor: i === activeModelIdx ? "var(--color-brand)" : "#2A2E39",
                backgroundColor: i === activeModelIdx ? "rgba(0,229,255,0.07)" : "transparent",
                color: i === activeModelIdx ? "var(--color-brand)" : "#787B86",
                cursor: "pointer",
              }}
            >
              {m.model} — Sharpe {m.sharpe?.toFixed(2) ?? "—"}
            </button>
          ))}
        </div>
      )}

      {/* ── Results / Compare tab bar (multi-model) ─────────────────── */}
      {metrics.length > 1 && (
        <div className="flex items-center gap-1 border-b" style={{ borderColor: "#1E222D" }}>
          <button
            onClick={() => setTab("results")}
            className="relative px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all duration-200"
            style={{
              color: tab === "results" ? "var(--color-brand)" : "#787B86",
              borderBottom: tab === "results" ? "2px solid var(--color-brand)" : "2px solid transparent",
              cursor: "pointer",
              background: "transparent",
              border: "none",
              borderBottomWidth: 2,
              borderBottomStyle: "solid",
              borderBottomColor: tab === "results" ? "var(--color-brand)" : "transparent",
            }}
          >
            <BarChart3 size={11} className="inline mr-1.5" />
            Results
          </button>
          <button
            onClick={() => setTab("compare")}
            className="relative px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all duration-200"
            style={{
              color: tab === "compare" ? "var(--color-brand)" : "#787B86",
              cursor: "pointer",
              background: "transparent",
              border: "none",
              borderBottomWidth: 2,
              borderBottomStyle: "solid",
              borderBottomColor: tab === "compare" ? "var(--color-brand)" : "transparent",
            }}
          >
            <GitCompare size={11} className="inline mr-1.5" />
            Compare
          </button>
        </div>
      )}

      {tab === "results" && (
        <div className="flex flex-col gap-3 pb-4">

          {/* Progressive disclosure: Summary + Overfitting as accordions */}
          {(activeMetric?.summary_text || activeMetric?.overfitting) && (
            <div className="flex flex-col gap-2">
              {activeMetric?.summary_text && (
                <Accordion
                  label="Backtest Summary"
                  open={showSummary}
                  onToggle={() => setShowSummary((v) => !v)}
                >
                  <div className="px-4 py-3">
                    <BacktestSummary text={activeMetric.summary_text} />
                  </div>
                </Accordion>
              )}
              {activeMetric?.overfitting && (
                <Accordion
                  label={`Overfitting Assessment — Score ${activeMetric.overfitting.overfit_score.toFixed(0)}`}
                  open={showOverfitting}
                  onToggle={() => setShowOverfitting((v) => !v)}
                >
                  <div className="p-3">
                    <OverfittingPanel
                      overfitting={activeMetric.overfitting ?? null}
                      walkforwardPeriods={activeMetric.walkforward_periods ?? null}
                    />
                  </div>
                </Accordion>
              )}
            </div>
          )}

          {/* ── HERO: main chart area ───────────────────────────────── */}
          <div
            className="rounded-lg overflow-hidden"
            style={{ border: "1px solid #2A2E39" }}
          >
            {/* Trade visualization */}
            {jobId && activeMetric?.model && (
              <div style={{ borderBottom: "1px solid #2A2E39" }}>
                <BacktestChart jobId={jobId} model={activeMetric.model} />
              </div>
            )}
            {/* Equity + Drawdown stacked below */}
            <div className="px-3 pt-2 pb-3" style={{ backgroundColor: "#131722" }}>
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
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
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
            <div className="flex flex-col gap-3 p-3">
              <DrawdownChart drawdownCurve={normalizeEquityCurve(activeMetric?.drawdown_curve ?? null)} />
              <CumulativePnlChart trades={activeMetric?.trades ? (activeMetric.trades as TradeRecord[]) : null} />
              <RollingMetricsChart equityCurve={normalizeEquityCurve(activeMetric?.equity_curve ?? null)} />
              <TradeDistributionChart trades={activeMetric?.trades ? (activeMetric.trades as TradeRecord[]) : null} />
              <HpoDiagnostics
                paramImportance={activeMetric?.hpo_param_importance ?? null}
                trials={activeMetric?.hpo_trials ?? null}
              />
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
                trades={activeMetric?.trades ? (activeMetric.trades as import("@/api/schemas").TradeRecord[]) : null}
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
          <SignificanceMatrix
            models={metrics.map((m) => m.model)}
            pValues={null}
          />
          {metrics.length > 0 && metrics[0].hpo_trials && (
            <ParameterSensitivityChart trials={metrics[0].hpo_trials} />
          )}
          <CrossPairSection />
        </div>
      )}

      {/* ── Selected trade tooltip ───────────────────────────────────── */}
      {selectedTrade && (
        <div
          className="fixed bottom-12 right-6 rounded-lg border p-3 text-xs"
          style={{
            backgroundColor: "#1E222D",
            borderColor: "#2A2E39",
            fontFamily: "JetBrains Mono, monospace",
            zIndex: 50,
            backdropFilter: "blur(12px)",
          }}
        >
          <div className="flex items-center justify-between gap-4">
            <span style={{ color: "#787B86" }}>Trade #{selectedTrade.trade_id}</span>
            <span style={{ color: selectedTrade.direction === "BUY" ? "#089981" : "#F23645" }}>
              {selectedTrade.direction}
            </span>
            <span style={{ color: (selectedTrade.return_pct ?? 0) >= 0 ? "#089981" : "#F23645" }}>
              {(selectedTrade.return_pct ?? 0) >= 0 ? "+" : ""}
              {(selectedTrade.return_pct ?? 0).toFixed(2)}%
            </span>
            <button
              onClick={() => setSelectedTrade(null)}
              style={{ color: "#787B86", cursor: "pointer", background: "none", border: "none" }}
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
