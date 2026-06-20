import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { useState, useRef, useMemo } from "react";
import {
  BarChart3,
  ArrowLeft,
  Play,
  Save,
  GitCompare,
  X,
  Download,
  Upload,
} from "lucide-react";
import { useJobResults, useTradeChartData, useSaveModelFromJob } from "@/api/queries";
import { EmptyState } from "@/components/shared/EmptyState";
import { ExportBar } from "@/components/shared/ExportBar";
import { StatusDot } from "@/components/shared/StatusDot";
import { MonthlyReturnsChart } from "@/components/charts/MonthlyReturnsChart";
import { MetricsGrid } from "./MetricsGrid";
import { UnifiedAnalytics, type UnifiedAnalyticsHandle } from "./UnifiedAnalytics";
import { TradeLogTable } from "./TradeLogTable";
import { HpoDiagnostics, BestStudyCard } from "./HpoDiagnostics";
import { OverfittingPanel } from "./OverfittingPanel";
import { ValidationScorecard } from "./ValidationScorecard";
import { WalkForwardPanel } from "./WalkForwardPanel";
import { ParameterExplorer } from "./ParameterExplorer";
import { LLMAdvisor } from "./LLMAdvisor";
import { FeatureImportanceSection, PredictionConfidenceSection } from "./TrainingDiagnostics";
import { LeaderboardTable } from "../Compare/LeaderboardTable";
import { EquityOverlayChart } from "../Compare/EquityOverlayChart";
import { SignificanceMatrix } from "../Compare/SignificanceMatrix";
import { CrossPairSection } from "../Compare/CrossPairSection";
import { ParameterSensitivityChart } from "@/components/charts/ParameterSensitivityChart";
import { normalizeEquityCurve } from "@/lib/chartUtils";
import { BacktestPlayback } from "./BacktestPlayback";
import { CommitteeResultsPage } from "./CommitteeResultsPage";
import type { TradeRecord } from "@/api/schemas";

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

/** Bento Box card — subtle border, minimal padding, optional title bar. */
function BentoCard({
  title,
  right,
  children,
  className,
}: {
  title?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-3 ${className ?? ""}`}>
      {(title || right) && (
        <div className="mb-2 flex items-center justify-between">
          {title && (
            <h3 className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
              {title}
            </h3>
          )}
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Branch: committee jobs render their own page
  if (searchParams.get("type") === "committee") {
    return <CommitteeResultsPage />;
  }

  const [activeModelIdx, setActiveModelIdx] = useState(0);
  const [selectedTrade, setSelectedTrade] = useState<TradeRecord | null>(null);
  const [showPlayback, setShowPlayback] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<"results" | "compare">("results");
  const [showTradeLog, setShowTradeLog] = useState(false);
  const [diagnosticsModelIdx, setDiagnosticsModelIdx] = useState(0);

  const { data: results, isLoading, isError } = useJobResults(jobId ?? null);
  const saveModelMutation = useSaveModelFromJob();
  const analyticsRef = useRef<UnifiedAnalyticsHandle>(null);

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

  const analyticsModels = useMemo(
    () =>
      metrics.map((m) => ({
        model: m.model,
        equityCurve: normalizeEquityCurve(m.equity_curve),
        drawdownCurve: normalizeEquityCurve(m.drawdown_curve),
        trades: m.trades ? (m.trades as TradeRecord[]) : null,
      })),
    [metrics],
  );

  const modelCurves = useMemo(() => {
    if (!metrics.length) return [];
    return metrics
      .filter((m) => m.equity_curve && m.equity_curve.length > 0)
      .map((m) => ({ model: m.model, data: m.equity_curve! }));
  }, [metrics]);

  const { data: tradeChartData } = useTradeChartData(jobId ?? "", activeMetric?.model ?? "");

  const handleExportPng = () => {
    analyticsRef.current?.takeScreenshot();
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
            onClick={() => navigate("/results")}
            className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1 text-xs text-(--color-text-muted) transition-all duration-200 hover:border-[var(--color-border-active)]"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Results
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
            onClick={() => navigate("/results")}
            className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1 text-xs text-(--color-text-muted) transition-all duration-200 hover:border-[var(--color-border-active)]"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Results
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
    const rows = trades.map((t) =>
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

  const handleExportConfig = () => {
    if (!results.config || Object.keys(results.config).length === 0) return;
    const blob = new Blob([JSON.stringify(results.config, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kodaquant-config_${results.job_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoadConfig = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const loaded = JSON.parse(evt.target?.result as string);
          const configOverrides = loaded?.config ?? loaded;
          const backtestUrl = `/backtest?config=${encodeURIComponent(JSON.stringify(configOverrides))}`;
          window.open(backtestUrl, "_blank");
        } catch {
          alert("Invalid JSON config file.");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const diagnosticsMetric = metrics.length
    ? metrics[Math.min(diagnosticsModelIdx, metrics.length - 1)]
    : null;

  const diagMonthly = diagnosticsMetric?.monthly_results ?? [];
  const diagPeriods = diagnosticsMetric?.walkforward_periods;
  const hasDiagHpo = !!diagnosticsMetric?.hpo_trials && diagnosticsMetric.hpo_trials.length > 0;
  const diagTradeCount = diagnosticsMetric?.trades?.length ?? 0;
  const hasWalkForward = (diagPeriods?.length ?? 0) > 1;
  const hasMonthly = diagMonthly.length > 1;

  const monthlyStart = diagMonthly.length > 0 ? (diagMonthly[0]?.month ?? "").slice(0, 7) : "";
  const monthlyEnd = diagMonthly.length > 0 ? (diagMonthly[diagMonthly.length - 1]?.month ?? "").slice(0, 7) : "";
  const monthlyPositive = diagMonthly.filter((m) => (m.return_pct ?? 0) > 0).length;

  return (
    <div className="flex h-full animate-fade-in flex-col gap-4">
      {/* ── Top ribbon ─────────────────────────────────────────────── */}
      <div className="flex min-h-[36px] flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3 leading-none">
          <button
            onClick={() => navigate("/results")}
            className="flex cursor-pointer items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1.5 text-[11px] leading-none text-(--color-text-muted) transition-all duration-200 hover:border-[var(--color-brand)]"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Results
          </button>
          <span className="font-mono text-[11px] leading-none text-(--color-text-secondary)">
            {results.pair}
          </span>
          <StatusDot color="var(--color-brand)" />
        </div>
        <div className="flex flex-wrap items-center gap-1.5 leading-none">
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
          {jobId && activeMetric?.model && (
            <button
              onClick={() => setShowPlayback(true)}
              className="flex cursor-pointer items-center gap-1.5 rounded-md border border-(--color-accent-success) bg-[rgba(8,153,129,0.1)] px-2.5 py-1 text-[11px] font-semibold text-(--color-accent-success) uppercase transition-all hover:brightness-110"
            >
              <Play size={11} /> Replay
            </button>
          )}
          {/* Config export / load — grouped with export buttons */}
          <button
            onClick={handleExportConfig}
            className="flex cursor-pointer items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-(--color-surface) px-2.5 py-1 text-[11px] font-semibold text-(--color-text-secondary) uppercase transition-all duration-150 hover:border-[var(--color-brand)]"
            title="Export this run's config as JSON"
          >
            <Download size={11} /> Config
          </button>
          <button
            onClick={handleLoadConfig}
            className="flex cursor-pointer items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-(--color-surface) px-2.5 py-1 text-[11px] font-semibold text-(--color-text-secondary) uppercase transition-all duration-150 hover:border-[var(--color-brand)]"
            title="Load a config JSON into a new backtest"
          >
            <Upload size={11} /> Load
          </button>
          <ExportBar
            onExportCsv={handleExportCsv}
            onExportPng={handleExportPng}
            onExportJson={handleExportJson}
          />
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

      {/* ── Validation badges (pill row) ────────────────────────────── */}
      {(activeMetric?.overfitting || activeMetric?.walkforward_periods) && (
        <BentoCard>
          <ValidationScorecard
            overfitting={activeMetric?.overfitting ?? null}
            walkforwardPeriods={activeMetric?.walkforward_periods ?? null}
          />
        </BentoCard>
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
          {/* ── Unified Analytics (central chart container) ────────── */}
          <UnifiedAnalytics
            ref={analyticsRef}
            models={analyticsModels}
            buyHoldCurve={normalizeEquityCurve(activeMetric?.buy_hold_curve ?? null)}
          />

          {/* ── Diagnostics Model Selector ──────────────────────────── */}
          {metrics.length > 1 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                Diagnostics:
              </span>
              {metrics.map((m, i) => (
                <button
                  key={m.model}
                  onClick={() => setDiagnosticsModelIdx(i)}
                  className="cursor-pointer rounded-md border px-2.5 py-0.5 text-[11px] font-medium tracking-[0.04em] uppercase transition-all duration-200"
                  style={{
                    borderColor:
                      i === diagnosticsModelIdx
                        ? "var(--color-brand)"
                        : "var(--color-glass-border)",
                    backgroundColor:
                      i === diagnosticsModelIdx
                        ? "rgba(0,229,255,0.07)"
                        : "transparent",
                    color:
                      i === diagnosticsModelIdx
                        ? "var(--color-brand)"
                        : "var(--color-text-muted)",
                  }}
                >
                  {m.model}
                </button>
              ))}
            </div>
          )}

          {/* ── Walk-Forward — full width, only if >1 period ──────── */}
          {hasWalkForward && (
            <BentoCard
              title={`Walk-Forward — ${diagPeriods!.length} periods`}
            >
              <WalkForwardPanel
                periods={diagPeriods ?? null}
              />
            </BentoCard>
          )}

          {/* ── Monthly Returns — full width, only if >1 month ─────── */}
          {hasMonthly && (
            <BentoCard
              title="Monthly Returns"
              right={
                <span className="font-mono text-[9px] text-(--color-text-muted)">
                  {monthlyStart} → {monthlyEnd} · {monthlyPositive}/{diagMonthly.length} positive
                </span>
              }
            >
              <MonthlyReturnsChart monthlyResults={diagMonthly} />
            </BentoCard>
          )}

          {/* ── Feature Importance + HPO — 2-column grid ───────────── */}
          <div className="grid grid-cols-1 gap-3 items-start lg:grid-cols-2">
            <BentoCard
              title={`Feature Importance — ${diagnosticsMetric?.model ?? ""}`}
            >
              <FeatureImportanceSection
                data={diagnosticsMetric?.diagnostics ?? null}
                modelName={diagnosticsMetric?.model ?? ""}
              />
            </BentoCard>

            {hasDiagHpo ? (
              <BentoCard
                title="HPO Diagnostics & Sensitivity"
                right={
                  <span className="font-mono text-[10px] text-(--color-text-muted)">
                    {diagnosticsMetric?.hpo_trials?.length ?? 0} trials
                  </span>
                }
              >
                <div className="flex flex-col gap-4">
                  <ParameterExplorer metrics={diagnosticsMetric} />
                  <HpoDiagnostics
                    paramImportance={diagnosticsMetric?.hpo_param_importance ?? null}
                    trials={diagnosticsMetric?.hpo_trials ?? null}
                  />
                  <BestStudyCard bestStudy={diagnosticsMetric?.best_study ?? null} />
                  <ParameterSensitivityChart trials={diagnosticsMetric?.hpo_trials ?? null} />
                  {diagnosticsMetric?.overfitting && (
                    <OverfittingPanel
                      overfitting={diagnosticsMetric.overfitting}
                      walkforwardPeriods={diagnosticsMetric.walkforward_periods ?? null}
                    />
                  )}
                </div>
              </BentoCard>
            ) : (
              <BentoCard title="HPO Diagnostics & Sensitivity">
                <p className="text-[11px] text-(--color-text-dim)">
                  No HPO trials for {diagnosticsMetric?.model ?? "this model"}.
                </p>
              </BentoCard>
            )}
          </div>

          {/* ── Prediction & Confidence — full width ──────────────── */}
          <BentoCard>
            <PredictionConfidenceSection data={diagnosticsMetric?.diagnostics ?? null} />
          </BentoCard>

          {/* ── Trade Log — self-contained, full width ──────────── */}
          <TradeLogTable
            trades={
              diagnosticsMetric?.trades
                ? (diagnosticsMetric.trades as import("@/api/schemas").TradeRecord[])
                : null
            }
            onTradeSelect={setSelectedTrade}
            title={`Trade Log — ${diagTradeCount} executed · ${diagnosticsMetric?.total_trades ?? 0} signals`}
            open={showTradeLog}
            onToggle={() => setShowTradeLog((v) => !v)}
          />

          {/* ── LLM advisor (below trade log) ────────────────────────── */}
          <LLMAdvisor jobId={jobId ?? null} modelName={diagnosticsMetric?.model ?? null} />
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
          timeframe={(results?.config?.timeframe as string) ?? "M30"}
          onClose={() => setShowPlayback(false)}
        />
      )}
    </div>
  );
}
