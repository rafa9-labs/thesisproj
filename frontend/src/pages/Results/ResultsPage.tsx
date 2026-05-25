import { useParams, useNavigate } from "react-router-dom";
import { useState, useRef, useMemo } from "react";
import { BarChart3, ArrowLeft, Play, Save, GitCompare } from "lucide-react";
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
    <div className="flex flex-col gap-6 animate-pulse">
      <div className="h-6 w-48 rounded" style={{ backgroundColor: "var(--color-elevated)" }} />
      <div className="grid grid-cols-4 gap-3">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className="h-24 rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
        ))}
      </div>
      <div className="h-[420px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
      <div className="h-[300px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
      <div className="h-[360px] rounded-lg" style={{ backgroundColor: "var(--color-surface)" }} />
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

  const { data: results, isLoading, isError } = useJobResults(jobId ?? null);
  const saveModelMutation = useSaveModelFromJob();
  const equityChartRef = useRef<EquityCurveChartHandle>(null);

  const activeMetric = results?.metrics?.length ? results.metrics[Math.min(activeModelIdx, results.metrics.length - 1)] : null;
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

  const metrics = results?.metrics ?? [];

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
    <div className="flex flex-col gap-6 overflow-y-auto" style={{ height: "100%" }}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "transparent",
              color: "var(--color-text-muted)",
              cursor: "pointer",
            }}
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Back
          </button>
          <h2
            className="text-base font-medium uppercase tracking-[0.1em]"
            style={{ color: "var(--color-text-secondary)" }}
          >
            Results
          </h2>
          <span
            className="text-xs font-light"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
          >
            {results.pair}
          </span>
          <StatusDot color="var(--color-brand)" />
        </div>
        <div className="flex items-center gap-2">
          {activeMetric?.snapshot_path && (
            <>
              {saveMsg ? (
                <button
                  onClick={() => navigate("/models")}
                  className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-semibold uppercase transition-all duration-200 hover:brightness-110"
                  style={{ borderColor: "var(--color-accent-success)", backgroundColor: "rgba(34,197,94,0.1)", color: "var(--color-accent-success)", cursor: "pointer" }}
                >
                  <Save size={12} /> View in Models
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
                  className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-semibold uppercase transition-all duration-200 hover:brightness-110"
                  style={{ borderColor: "var(--color-accent-warning)", backgroundColor: "rgba(245,158,11,0.1)", color: "var(--color-accent-warning)", cursor: saveModelMutation.isPending ? "not-allowed" : "pointer", opacity: saveModelMutation.isPending ? 0.6 : 1 }}
                >
                  <Save size={12} /> {saveModelMutation.isPending ? "Saving..." : "Save Model"}
                </button>
              )}
            </>
          )}
          <ExportBar onExportCsv={handleExportCsv} onExportPng={handleExportPng} onExportJson={handleExportJson} />
          {jobId && activeMetric?.model && (
            <button
              onClick={() => setShowPlayback(true)}
              className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-semibold uppercase transition-all duration-200"
              style={{
                borderColor: "var(--color-accent-success)",
                backgroundColor: "rgba(34,197,94,0.1)",
                color: "var(--color-accent-success)",
              }}
              title="Replay backtest bar-by-bar"
            >
              <Play size={12} /> Replay
            </button>
          )}
        </div>
      </div>

      {activeMetric && (activeMetric.total_trades ?? 0) === 0 && (
        <div
          className="rounded-xl border p-4"
          style={{ borderColor: "var(--color-accent-warning)", backgroundColor: "rgba(245,158,11,0.05)" }}
        >
          <p className="text-xs" style={{ color: "var(--color-accent-warning)" }}>
            This backtest produced no trades. All walk-forward months were flat — the model did not find enough signal to enter any positions. Try increasing HPO trials, tightening the search bounds, or selecting a different model.
          </p>
        </div>
      )}

      {activeMetric && (
        <MetricsGrid metrics={activeMetric} modelName={activeMetric.model} monthlyResults={activeMetric.monthly_results} />
      )}

      {activeMetric && (
        <BacktestSummary text={activeMetric.summary_text ?? null} />
      )}

      {activeMetric && (
        <OverfittingPanel
          overfitting={activeMetric.overfitting ?? null}
          walkforwardPeriods={activeMetric.walkforward_periods ?? null}
        />
      )}

      {activeMetric && (
        <WalkForwardPanel periods={activeMetric.walkforward_periods ?? null} modelName={activeMetric.model} />
      )}

      {activeMetric && (
        <TrainingDiagnosticsPanel data={activeMetric.diagnostics ?? null} modelName={activeMetric.model} />
      )}

      {activeMetric && (
        <ParameterExplorer metrics={activeMetric} />
      )}

      {metrics.length > 1 && (
        <div className="flex items-center gap-1 border-b" style={{ borderColor: "var(--color-border-subtle)" }}>
          <button
            onClick={() => setTab("results")}
            className="relative px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all duration-200"
            style={{
              color: tab === "results" ? "var(--color-brand)" : "var(--color-text-muted)",
              borderBottom: tab === "results" ? "2px solid var(--color-brand)" : "2px solid transparent",
              cursor: "pointer",
              background: "transparent",
            }}
          >
            <BarChart3 size={12} className="inline mr-1.5" />
            Results
          </button>
          <button
            onClick={() => setTab("compare")}
            className="relative px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all duration-200"
            style={{
              color: tab === "compare" ? "var(--color-brand)" : "var(--color-text-muted)",
              borderBottom: tab === "compare" ? "2px solid var(--color-brand)" : "2px solid transparent",
              cursor: "pointer",
              background: "transparent",
            }}
          >
            <GitCompare size={12} className="inline mr-1.5" />
            Compare
          </button>
        </div>
      )}

      {tab === "results" && (
        <>
          <LLMAdvisor jobId={jobId ?? null} modelName={activeMetric?.model ?? null} />

          {metrics.length > 1 && (
            <div className="flex flex-wrap gap-2">
              {metrics.map((m, i) => (
                <button
                  key={m.model}
                  onClick={() => setActiveModelIdx(i)}
                  className="rounded-md border px-3.5 py-1.5 text-[11px] font-medium uppercase tracking-[0.08em] transition-all duration-200"
                  style={{
                    borderColor: i === activeModelIdx ? "var(--color-brand)" : "var(--color-glass-border)",
                    backgroundColor: i === activeModelIdx ? "var(--color-brand-glow)" : "var(--color-glass)",
                    color: i === activeModelIdx ? "var(--color-brand)" : "var(--color-text-muted)",
                    cursor: "pointer",
                    boxShadow: i === activeModelIdx ? "0 0 12px rgba(0,229,255,0.1)" : "none",
                    backdropFilter: "blur(8px)",
                  }}
                >
                  {m.model} — Sharpe {m.sharpe?.toFixed(2) ?? "—"}
                </button>
              ))}
            </div>
          )}

          {jobId && activeMetric?.model && (
            <BacktestChart jobId={jobId} model={activeMetric.model} />
          )}

          <EquitySection
            ref={equityChartRef}
            equityCurve={normalizeEquityCurve(activeMetric?.equity_curve ?? null)}
            buyHoldCurve={normalizeEquityCurve(activeMetric?.buy_hold_curve ?? null)}
            drawdownCurve={normalizeEquityCurve(activeMetric?.drawdown_curve ?? null)}
          />

          <DrawdownChart drawdownCurve={normalizeEquityCurve(activeMetric?.drawdown_curve ?? null)} />

          <CumulativePnlChart trades={activeMetric?.trades ? (activeMetric.trades as TradeRecord[]) : null} />

          <MonthlySection monthlyResults={activeMetric?.monthly_results ?? null} />

          <RollingMetricsChart equityCurve={normalizeEquityCurve(activeMetric?.equity_curve ?? null)} />

          <TradeDistributionChart trades={activeMetric?.trades ? (activeMetric.trades as TradeRecord[]) : null} />

          <HpoDiagnostics
            paramImportance={activeMetric?.hpo_param_importance ?? null}
            trials={activeMetric?.hpo_trials ?? null}
          />

          <ParameterSensitivityChart trials={activeMetric?.hpo_trials ?? null} />

          <TradeLogTable
            trades={activeMetric?.trades ? (activeMetric.trades as import("@/api/schemas").TradeRecord[]) : null}
            onTradeSelect={setSelectedTrade}
          />

          <ConfigViewer config={results.config ?? null} />
        </>
      )}

      {tab === "compare" && (
        <>
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
        </>
      )}

      {selectedTrade && (
        <div
          className="fixed bottom-12 right-6 rounded-lg border p-3 text-xs"
          style={{
            backgroundColor: "var(--color-glass)",
            borderColor: "var(--color-glass-border)",
            fontFamily: "var(--font-mono)",
            zIndex: 50,
            backdropFilter: "blur(12px)",
          }}
        >
          <div className="flex items-center justify-between gap-4">
            <span style={{ color: "var(--color-text-secondary)" }}>
              Trade #{selectedTrade.trade_id}
            </span>
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
            <span style={{ color: (selectedTrade.return_pct ?? 0) >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
              {(selectedTrade.return_pct ?? 0) >= 0 ? "+" : ""}
              {(selectedTrade.return_pct ?? 0).toFixed(2)}%
            </span>
            <button
              onClick={() => setSelectedTrade(null)}
              className="text-xs"
              style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
            >
              ✕
            </button>
          </div>
        </div>
      )}

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
