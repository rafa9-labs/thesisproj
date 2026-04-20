import { useParams, useNavigate } from "react-router-dom";
import { useState, useRef } from "react";
import { BarChart3, ArrowLeft } from "lucide-react";
import { useJobResults } from "@/api/queries";
import { EmptyState } from "@/components/shared/EmptyState";
import { ExportBar } from "@/components/shared/ExportBar";
import { StatusDot } from "@/components/shared/StatusDot";
import { MetricsGrid } from "./MetricsGrid";
import { EquitySection } from "./EquitySection";
import { TradeLogTable } from "./TradeLogTable";
import { MonthlySection } from "./MonthlySection";
import { HpoDiagnostics } from "./HpoDiagnostics";
import { ConfigViewer } from "./ConfigViewer";
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

  const { data: results, isLoading, isError } = useJobResults(jobId ?? null);
  const equityChartRef = useRef<EquityCurveChartHandle>(null);

  const handleExportPng = () => {
    equityChartRef.current?.takeScreenshot();
  };

  if (!jobId) {
    return (
      <div className="flex flex-col gap-6">
        <h2
          className="text-base font-semibold uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Results
        </h2>
        <EmptyState
          icon={<BarChart3 size={48} />}
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
      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-surface)",
              color: "var(--color-text-secondary)",
            }}
          >
            <ArrowLeft size={12} /> Back
          </button>
          <h2
            className="text-base font-semibold uppercase tracking-[0.1em]"
            style={{ color: "var(--color-text-secondary)" }}
          >
            Results
          </h2>
        </div>
        <Skeleton />
      </div>
    );
  }

  if (isError || !results) {
    return (
      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate("/backtest")}
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-surface)",
              color: "var(--color-text-secondary)",
            }}
          >
            <ArrowLeft size={12} /> Back
          </button>
        </div>
        <EmptyState
          icon={<BarChart3 size={48} />}
          title="Failed to load results"
          description="The job may not exist or hasn't completed yet."
          actionLabel="Run Backtest"
          onAction={() => navigate("/backtest")}
        />
      </div>
    );
  }

  const metrics = results.metrics ?? [];
  const activeMetric = metrics.length > 0 ? metrics[Math.min(activeModelIdx, metrics.length - 1)] : null;

  const handleExportCsv = () => {
    if (!results.trades) return;
    const header = "trade_id,entry_date,exit_date,direction,entry_price,exit_price,pips,return_pct,duration_bars,barrier_hit";
    const rows = results.trades.map(
      (t) =>
        `${t.trade_id},${t.entry_date},${t.exit_date},${t.direction},${t.entry_price},${t.exit_price},${t.pips},${t.return_pct},${t.duration_bars},${t.barrier_hit ?? ""}`,
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades_${results.job_id}.csv`;
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
            className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-colors"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-surface)",
              color: "var(--color-text-secondary)",
              cursor: "pointer",
            }}
          >
            <ArrowLeft size={12} /> Back
          </button>
          <h2
            className="text-base font-semibold uppercase tracking-[0.1em]"
            style={{ color: "var(--color-text-secondary)" }}
          >
            Results
          </h2>
          <span
            className="text-xs"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
          >
            {results.pair}
          </span>
          <StatusDot color="var(--color-accent-success)" />
        </div>
        <ExportBar onExportCsv={handleExportCsv} onExportPng={handleExportPng} onExportJson={handleExportJson} />
      </div>

      {activeMetric && (
        <MetricsGrid metrics={activeMetric} modelName={activeMetric.model} />
      )}

      {metrics.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {metrics.map((m, i) => (
            <button
              key={m.model}
              onClick={() => setActiveModelIdx(i)}
              className="rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors"
              style={{
                borderColor: i === activeModelIdx ? "var(--color-accent)" : "var(--color-border)",
                backgroundColor: i === activeModelIdx ? "rgba(41,98,255,0.1)" : "var(--color-surface)",
                color: i === activeModelIdx ? "var(--color-accent)" : "var(--color-text-secondary)",
                cursor: "pointer",
              }}
            >
              {m.model} — Sharpe {m.sharpe?.toFixed(2) ?? "—"}
            </button>
          ))}
        </div>
      )}

      <EquitySection
        ref={equityChartRef}
        equityCurve={results.equity_curve ?? null}
        buyHoldCurve={results.buy_hold_curve ?? null}
        drawdownCurve={results.drawdown_curve ?? null}
      />

      <MonthlySection monthlyResults={results.monthly_results ?? null} />

      <HpoDiagnostics
        paramImportance={results.hpo_param_importance ?? null}
        trials={results.hpo_trials ?? null}
      />

      <TradeLogTable
        trades={results.trades ?? null}
        onTradeSelect={setSelectedTrade}
      />

      <ConfigViewer config={results.config ?? null} />

      {selectedTrade && (
        <div
          className="fixed bottom-12 right-6 rounded-lg border p-3 text-xs"
          style={{
            backgroundColor: "var(--color-elevated)",
            borderColor: "var(--color-border)",
            fontFamily: "var(--font-mono)",
            zIndex: 50,
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
            <span style={{ color: selectedTrade.return_pct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
              {selectedTrade.return_pct >= 0 ? "+" : ""}
              {selectedTrade.return_pct.toFixed(2)}%
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
    </div>
  );
}
