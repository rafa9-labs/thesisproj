import { useFullCycleResults } from "@/api/queries";
import { useSaveCommittee } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { formatStopReason } from "@/lib/constants";
import apiClient from "@/api/client";
import type { FactoryIterationRecord } from "@/api/schemas";
import { useState } from "react";

interface Props {
  jobId: string;
  onRunAgain: () => void;
}

function SectionHeader({ label }: { label: string }) {
  return (
    <span className="mb-[8px] block text-[11px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
      {label}
    </span>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-elevated) px-[16px] py-[14px] text-center">
      <div className="mb-[6px] font-mono text-[22px] font-semibold">{value}</div>
      <div className="text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
        {label}
      </div>
    </div>
  );
}

const TH_CLASSES = "px-2 py-1.5 text-left font-medium tracking-[0.06em] uppercase text-[10px]";
const TH_STYLE: React.CSSProperties = { color: "var(--color-text-muted)" };

const TD_CLASSES = "px-2 py-1.5 text-[11px]";
const TD_STYLE: React.CSSProperties = { color: "var(--color-text-secondary)" };
const TD_MONO_CLASSES = "px-2 py-1.5 text-[11px] font-mono";
const TD_MONO_STYLE: React.CSSProperties = { color: "var(--color-text-secondary)" };

export function FullCycleResults({ jobId, onRunAgain }: Props) {
  const { data: results } = useFullCycleResults(jobId);
  const { data: status } = { data: null };
  const store = useFullCycleStore();
  const saveMutation = useSaveCommittee();
  const [saveSuccess, setSaveSuccess] = useState(false);

  if (!results) {
    return (
      <div className="p-[28px] text-center text-[11px] text-(--color-text-muted)">
        Loading results...
      </div>
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
      .then(
        (r: {
          data: { session_id: string; pair: string; timeframe: string; models: string[] };
        }) => {
          store.setDeployedSession(
            r.data.session_id,
            r.data.pair || store.deployedPair,
            r.data.timeframe || store.deployedTimeframe,
          );
          store.setDeployedJobId(jobId);
        },
      )
      .catch((err: { response?: { data?: { detail?: string } }; message?: string }) => {
        console.error("Deploy failed:", err?.response?.data?.detail ?? err?.message);
      });
  };

  const handleSaveCommittee = () => {
    if (!results?.factory_best_config) return;
    const name = `Committee ${new Date().toISOString().slice(0, 10)} ${jobId.slice(0, 6)}`;
    saveMutation.mutate(
      {
        name,
        full_cycle_job_id: jobId,
        pair: store.deployedPair,
        timeframe: store.deployedTimeframe,
        config_json: results.factory_best_config,
        trust_score: results.trust_score?.trust_score ?? null,
        avg_sharpe: results.factory_best_sharpe ?? null,
      },
      {
        onSuccess: () => setSaveSuccess(true),
      },
    );
  };

  return (
    <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[28px]">
      <h3 className="mb-[20px] text-[14px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
        Pipeline Results
        {results.status === "validation_failed" && (
          <span className="ml-[10px] text-[11px] font-medium text-[#F2B436]">
            VALIDATION FAILED
          </span>
        )}
        <span className="ml-[10px] font-mono text-[11px] font-normal text-(--color-text-muted)">
          {Number(results.total_time_s).toFixed(0)}s
        </span>
      </h3>

      {/* Phase 1 Feature Sweep */}
      {results.locked_features_count !== undefined && results.locked_features_count > 0 && (
        <>
          <SectionHeader label="Phase 1: Feature Sweep" />
          <div className="mb-[20px] grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-[10px]">
            <MetricCard
              label="Features Locked"
              value={String(results.locked_features_count)}
              color="#089981"
            />
            <MetricCard
              label="Features Pruned"
              value={String(results.pruned_features_count ?? 0)}
              color="#F23645"
            />
            <MetricCard
              label="Top Feature"
              value={results.top_importance_feature || "N/A"}
              color="var(--color-text-secondary)"
            />
          </div>
        </>
      )}

      {/* Phase 3 HPO Status */}
      {results.hpo_status && Object.keys(results.hpo_status).length > 0 && (
        <>
          <SectionHeader label="Phase 3: HPO Status" />
          <div className="mb-[20px] flex flex-wrap gap-[6px]">
            {Object.entries(results.hpo_status).map(([model, status]) => {
              const statusColor =
                status === "success"
                  ? "#089981"
                  : status === "timed_out"
                    ? "#F2B436"
                    : status === "crashed"
                      ? "#F23645"
                      : status === "no_folds"
                        ? "#EF5350"
                        : "var(--color-text-dim)";
              const statusBg =
                status === "success"
                  ? "rgba(8,153,129,0.12)"
                  : status === "timed_out"
                    ? "rgba(242,180,54,0.12)"
                    : status === "crashed"
                      ? "rgba(242,54,69,0.12)"
                      : status === "no_folds"
                        ? "rgba(239,83,80,0.10)"
                        : "var(--color-elevated)";
              return (
                <span
                  key={model}
                  className="flex items-center gap-[6px] rounded-[3px] px-[10px] py-[4px] font-mono text-[10px]"
                  style={{
                    background: statusBg,
                    border: `1px solid ${statusColor}33`,
                    color: statusColor,
                  }}
                >
                  <span className="font-medium">{model}</span>
                  <span className="text-[9px] opacity-70">{status.replace(/_/g, " ")}</span>
                </span>
              );
            })}
          </div>
        </>
      )}

      {/* Alternative: show survivors list when hpo_status not available */}
      {(!results.hpo_status || Object.keys(results.hpo_status).length === 0) &&
        (results.phase0_survivors?.length ?? 0) > 0 && (
          <>
            <SectionHeader label="Phase 3: Survivors" />
            <div className="mb-[20px] flex flex-wrap gap-[4px]">
              {(results.phase0_survivors ?? []).map((m: string) => (
                <span
                  key={m}
                  className="rounded-[3px] border border-[rgba(8,153,129,0.25)] bg-[rgba(8,153,129,0.12)] px-[9px] py-[3px] font-mono text-[10px] text-[#089981]"
                >
                  {m}
                </span>
              ))}
            </div>
          </>
        )}

      {/* Phase 4: Trust Score + Validation */}
      {results.trust_score && (
        <>
          <SectionHeader label="Phase 4: Validation — Trust Score" />
          <div className="mb-[20px]">
            {/* Trust Score Gauge */}
            <div className="mb-[16px] flex items-center gap-[16px] rounded-[6px] border border-(--color-glass-border) bg-(--color-elevated) p-[16px]">
              <div
                className="flex h-[64px] w-[64px] shrink-0 items-center justify-center rounded-full"
                style={{
                  background:
                    results.trust_score.trust_score >= 0.8
                      ? "rgba(8,153,129,0.15)"
                      : results.trust_score.trust_score >= 0.6
                        ? "rgba(242,180,54,0.15)"
                        : results.trust_score.trust_score >= 0.4
                          ? "rgba(242,145,54,0.15)"
                          : "rgba(242,54,69,0.15)",
                  border: `3px solid ${
                    results.trust_score.trust_score >= 0.8
                      ? "#089981"
                      : results.trust_score.trust_score >= 0.6
                        ? "#F2B436"
                        : results.trust_score.trust_score >= 0.4
                          ? "#F29136"
                          : "#F23645"
                  }`,
                }}
              >
                <span
                  className="font-mono text-[22px] font-bold"
                  style={{
                    color:
                      results.trust_score.trust_score >= 0.8
                        ? "#089981"
                        : results.trust_score.trust_score >= 0.6
                          ? "#F2B436"
                          : results.trust_score.trust_score >= 0.4
                            ? "#F29136"
                            : "#F23645",
                  }}
                >
                  {Number(results.trust_score.trust_score * 100).toFixed(0)}
                </span>
              </div>
              <div>
                <div className="mb-[2px] font-mono text-[14px] font-semibold text-(--color-text-primary)">
                  Trust Score: {results.trust_score.trust_score.toFixed(3)}
                </div>
                <span
                  className="rounded-[3px] px-[10px] py-[2px] text-[11px] font-semibold tracking-[0.08em] uppercase"
                  style={{
                    background:
                      results.trust_score.action === "deploy"
                        ? "rgba(8,153,129,0.15)"
                        : results.trust_score.action === "proceed"
                          ? "rgba(242,180,54,0.15)"
                          : results.trust_score.action === "flag"
                            ? "rgba(242,145,54,0.15)"
                            : "rgba(242,54,69,0.15)",
                    color:
                      results.trust_score.action === "deploy"
                        ? "#089981"
                        : results.trust_score.action === "proceed"
                          ? "#F2B436"
                          : results.trust_score.action === "flag"
                            ? "#F29136"
                            : "#F23645",
                  }}
                >
                  {results.trust_score.action.toUpperCase()}
                </span>
              </div>
            </div>

            {/* Sub-score metrics */}
            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-[10px]">
              <MetricCard
                label="PBO"
                value={Number(results.pbo ?? 1).toFixed(3)}
                color={
                  Number(results.pbo ?? 1) < 0.2
                    ? "#089981"
                    : Number(results.pbo ?? 1) < 0.5
                      ? "#F2B436"
                      : "#F23645"
                }
              />
              <MetricCard
                label="DSR"
                value={Number(results.dsr ?? 0).toFixed(3)}
                color={
                  Number(results.dsr ?? 0) > 0.9
                    ? "#089981"
                    : Number(results.dsr ?? 0) > 0.5
                      ? "#F2B436"
                      : "#F23645"
                }
              />
              <MetricCard
                label="Fold CV"
                value={Number(results.phase3_fold_consistency_cv ?? 0).toFixed(3)}
                color={
                  Number(results.phase3_fold_consistency_cv ?? 0) < 1.0 ? "#089981" : "#F2B436"
                }
              />
              <MetricCard
                label="Seed Sharpe"
                value={Number(results.phase3_seed_robustness_sharpe ?? 0).toFixed(3)}
                color={
                  Number(results.phase3_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"
                }
              />
            </div>
          </div>
        </>
      )}

      {/* Fallback: show old-style Phase 5 gates if trust_score not available */}
      {!results.trust_score && (
        <>
          <SectionHeader label="Phase 4: Validation" />
          <div className="mb-[20px] grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-[10px]">
            <MetricCard
              label="Fold CV"
              value={Number(results.phase3_fold_consistency_cv ?? 0).toFixed(3)}
              color={Number(results.phase3_fold_consistency_cv ?? 0) < 1.0 ? "#089981" : "#F2B436"}
            />
            <MetricCard
              label="Seed Sharpe"
              value={Number(results.phase3_seed_robustness_sharpe ?? 0).toFixed(3)}
              color={
                Number(results.phase3_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"
              }
            />
          </div>
        </>
      )}

      {results.status === "validation_failed" && (
        <div className="mb-[20px] rounded-[4px] border border-[rgba(242,180,54,0.2)] bg-[rgba(242,180,54,0.06)] p-[14px] text-[10px] leading-[1.5] text-(--color-text-secondary)">
          <span className="font-semibold text-[#F2B436]">Validation halted pipeline.</span> Phase 5
          was skipped. Review the diagnostics above and adjust parameters. Common fixes: increase
          train_months, try different model types, or adjust HPO trial budgets.
        </div>
      )}

      {/* Team Backtest */}
      {results.racecar_backtest && <SectionHeader label="Team Backtest" />}
      {results.racecar_backtest && (
        <div className="mb-[24px] grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-[10px]">
          <MetricCard
            label="Avg Sharpe"
            value={Number(
              (results.racecar_backtest as Record<string, unknown>)?.avg_sharpe ?? 0,
            ).toFixed(3)}
            color={
              Number((results.racecar_backtest as Record<string, unknown>)?.avg_sharpe ?? 0) >= 0
                ? "#089981"
                : "#F23645"
            }
          />
          <MetricCard
            label="Avg Trades"
            value={String((results.racecar_backtest as Record<string, unknown>)?.avg_trades ?? 0)}
            color="var(--color-text-secondary)"
          />
          <MetricCard
            label="Folds"
            value={String((results.racecar_backtest as Record<string, unknown>)?.total_folds ?? 0)}
            color="var(--color-text-secondary)"
          />
          <MetricCard
            label="Models in Config"
            value={String(
              ((results.racecar_backtest as Record<string, unknown>)?.models as unknown[])
                ?.length ?? 0,
            )}
            color="var(--color-text-secondary)"
          />
        </div>
      )}

      {/* Final validation (10-year WFO) */}
      {results.final_fold_consistency_cv !== undefined && results.final_fold_consistency_cv > 0 && (
        <>
          <SectionHeader label="Final Validation (10-year WFO + 5-seed)" />
          <div className="mb-[20px] grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-[10px]">
            <MetricCard
              label="Fold CV"
              value={Number(results.final_fold_consistency_cv ?? 0).toFixed(3)}
              color={results.final_fold_consistency_pass ? "#089981" : "#F23645"}
            />
            <MetricCard
              label="Regime Coverage"
              value={results.final_regime_coverage ? "PASS" : "FAIL"}
              color={results.final_regime_coverage ? "#089981" : "#F23645"}
            />
            <MetricCard
              label="Seed Robustness"
              value={results.final_seed_robustness_pass ? "PASS" : "FAIL"}
              color={results.final_seed_robustness_pass ? "#089981" : "#F23645"}
            />
            <MetricCard
              label="Seed Sharpe"
              value={Number(results.final_seed_robustness_sharpe ?? 0).toFixed(3)}
              color={Number(results.final_seed_robustness_sharpe ?? 0) >= 0 ? "#089981" : "#F23645"}
            />
          </div>
        </>
      )}

      {/* Factory results */}
      {(results.factory_total_iterations ?? 0) > 0 && (
        <>
          <SectionHeader label="Factory Optimization" />
          <div className="mb-[24px] grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-[10px]">
            <MetricCard
              label="Best Sharpe"
              value={Number(results.factory_best_sharpe).toFixed(4)}
              color="#089981"
            />
            <MetricCard
              label="Iterations"
              value={String(results.factory_total_iterations)}
              color="var(--color-text-secondary)"
            />
            <MetricCard
              label="Accepted"
              value={String(results.factory_accepted_count)}
              color="var(--color-text-secondary)"
            />
          </div>
        </>
      )}

      {/* Best config after Factory */}
      {results.factory_best_config && (
        <div className="mb-[20px]">
          <span className="text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
            Optimized Committee Config
          </span>
          <div className="mt-[10px] flex flex-col gap-[6px]">
            {Object.entries(
              ((results.factory_best_config as Record<string, unknown>).regimes as Record<
                string,
                Record<string, unknown>
              >) ?? {},
            ).map(([regime, a]) => (
              <div key={regime} className="flex items-center gap-[14px] text-[11px]">
                <span className="w-[130px] shrink-0 font-medium tracking-[0.06em] text-(--color-brand) uppercase">
                  {regime.replace(/_/g, " ")}
                </span>
                <div className="flex flex-wrap gap-[4px]">
                  {(a.models as string[])?.map((m: string, i: number) => (
                    <span
                      key={`${m}-${i}`}
                      className="rounded-[3px] border border-(--color-glass-border) bg-(--color-elevated) px-[9px] py-[3px] font-mono text-[10px] text-(--color-text-secondary)"
                    >
                      {m} {(Number((a.weights as number[])?.[i] ?? 0) * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Factory iteration history */}
      {(results.factory_history?.length ?? 0) > 0 && (
        <div className="mb-[16px]">
          <span className="text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
            Optimization Log
          </span>
          <div className="mt-[8px] max-h-[300px] overflow-auto">
            <table className="w-full border-collapse text-[10px]">
              <thead>
                <tr>
                  <th
                    className={"border-b border-(--color-glass-border) " + TH_CLASSES}
                    style={TH_STYLE}
                  >
                    #
                  </th>
                  <th
                    className={"border-b border-(--color-glass-border) " + TH_CLASSES}
                    style={TH_STYLE}
                  >
                    Action
                  </th>
                  <th
                    className={"border-b border-(--color-glass-border) " + TH_CLASSES}
                    style={TH_STYLE}
                  >
                    Regime
                  </th>
                  <th
                    className={"border-b border-(--color-glass-border) " + TH_CLASSES}
                    style={TH_STYLE}
                  >
                    Model Change
                  </th>
                  <th
                    className={
                      "border-b border-(--color-glass-border) " + TH_CLASSES + " text-right"
                    }
                    style={TH_STYLE}
                  >
                    Sharpe Δ
                  </th>
                  <th
                    className={"border-b border-(--color-glass-border) " + TH_CLASSES}
                    style={TH_STYLE}
                  ></th>
                </tr>
              </thead>
              <tbody>
                {(results.factory_history ?? []).map((row: FactoryIterationRecord, i: number) => (
                  <tr key={i} className="border-b border-(--color-glass-border)">
                    <td className={TD_MONO_CLASSES} style={TD_MONO_STYLE}>
                      {row.iteration}
                    </td>
                    <td className="text-(--color-text-secondary)">
                      {row.action_type.toUpperCase()}
                    </td>
                    <td className={TD_CLASSES} style={TD_STYLE}>
                      {row.regime.replace(/_/g, " ")}
                    </td>
                    <td className="text-(--color-text-dim)">
                      {[row.model_add, row.model_remove].filter(Boolean).join(" / ")}
                    </td>
                    <td
                      className={TD_MONO_CLASSES + " text-right"}
                      style={{ color: row.delta_sharpe >= 0 ? "#089981" : "#F23645" }}
                    >
                      {row.delta_sharpe >= 0 ? "+" : ""}
                      {Number(row.delta_sharpe).toFixed(4)}
                    </td>
                    <td className={TD_CLASSES + " text-center text-xs"} style={TD_STYLE}>
                      {row.accepted ? (
                        <span className="text-[#089981]">✓</span>
                      ) : (
                        <span className="text-[#F23645]">✗</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Stop reason */}
      {results.factory_stop_reason && (
        <div className="mt-[8px] rounded-[4px] border border-[rgba(0,229,255,0.15)] bg-[rgba(0,229,255,0.05)] p-[10px] text-[11px] tracking-[0.04em] text-(--color-brand)">
          Stopped — {formatStopReason(results.factory_stop_reason)}
        </div>
      )}

      <div className="mt-[24px] flex gap-[12px]">
        <button
          onClick={onRunAgain}
          className="cursor-pointer rounded-[4px] border border-(--color-glass-border) bg-(--color-elevated) px-[20px] py-[8px] text-[10px] font-medium tracking-[0.06em] text-(--color-text-secondary) uppercase"
        >
          Run Again
        </button>
        <button
          onClick={handleDownload}
          className="cursor-pointer rounded-[4px] border border-(--color-glass-border) bg-(--color-elevated) px-[20px] py-[8px] text-[10px] font-medium tracking-[0.06em] text-(--color-text-secondary) uppercase"
        >
          Download JSON
        </button>
        {results.factory_best_config && results.status !== "validation_failed" && (
          <>
            <button
              onClick={handleSaveCommittee}
              disabled={saveMutation.isPending || saveSuccess}
              className="cursor-pointer rounded-[4px] border border-[rgba(0,229,255,0.25)] bg-[rgba(0,229,255,0.08)] px-[20px] py-[8px] text-[10px] font-semibold tracking-[0.06em] text-(--color-brand) uppercase"
            >
              {saveSuccess ? "Saved" : saveMutation.isPending ? "Saving..." : "Save Committee"}
            </button>
            <button
              onClick={handleDeploy}
              className="cursor-pointer rounded-[4px] border-none bg-(--color-accent-success) px-[20px] py-[8px] text-[10px] font-semibold tracking-[0.06em] text-(--color-text-inverse) uppercase"
            >
              Deploy to Trading
            </button>
          </>
        )}
      </div>
    </div>
  );
}
