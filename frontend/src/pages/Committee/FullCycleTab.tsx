import { useState, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useStartFullCycle, useFullCycleStatus } from "@/api/queries";
import apiClient from "@/api/client";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { FullCycleProgress } from "./FullCycleProgress";
import { FullCycleResults } from "./FullCycleResults";
import type { FullCycleRequest } from "@/api/schemas";

/**
 * Lightweight Full Cycle runner — progress + results only.
 * Config UI has been extracted to the Committee wizard tabs.
 */
export function FullCycleTab() {
  const store = useFullCycleStore();
  const startMutation = useStartFullCycle();
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (store.selectedHistoryJobId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setJobId(store.selectedHistoryJobId);
      store.setSelectedHistoryJobId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.selectedHistoryJobId]);

  const { data: status } = useFullCycleStatus(jobId);
  const terminalPhases = new Set([
    "completed",
    "failed",
    "validation_failed",
    "cancelled",
    "orphaned",
  ]);
  const isRunning = status && !terminalPhases.has(status.phase);
  const isDone = status && terminalPhases.has(status.phase);

  const cancelMutation = useMutation({
    mutationFn: async () => {
      await apiClient.post(`/committee/full-cycle/${jobId}/cancel`);
    },
    onSuccess: () => {
      queryClient.cancelQueries({ queryKey: ["full-cycle", "status", jobId] });
      if (jobId)
        queryClient.setQueryData(["full-cycle", "status", jobId], {
          phase: "cancelled",
          phase_number: 0,
          phase_progress: "",
          iteration: 0,
          total_iterations: 0,
          current_action: "Cancelled by user",
          best_sharpe_so_far: 0,
          started_at: "",
          error: "Cancelled by user",
          surviving_models: [],
          locked_features_count: 0,
          job_id: jobId,
        });
    },
  });

  const handleDeploy = async () => {
    const payload: FullCycleRequest = {
      models: store.selectedModels,
      pair: store.pair,
      timeframe: store.timeframe,
      sweep_n_estimators: store.sweepNEstimators,
      sweep_max_depth: store.sweepMaxDepth,
      skip_feature_sweep: store.skipFeatureSweep,
      use_boruta_shap: store.useBorutaShap,
      boruta_percentile: store.borutaPercentile,
      boruta_max_iter: store.borutaMaxIter,
      enable_phase3: store.enablePhase3,
      enable_phase4: store.enablePhase4,
      enable_phase5: store.enablePhase5,
      enable_phase6: store.enablePhase6,
      debug_mode: store.debugMode,
      committee_top_k: store.committeeTopK,
      committee_weight_method: store.committeeWeightMethod,
      committee_min_sharpe: store.committeeMinSharpe,
      train_months: store.trainMonths,
      test_months: store.testMonths,
      hpo_sampler: store.hpoSampler,
      cv_blocks: store.cvBlocks,
      cv_val_frac: 0.05,
      plateau_patience: 15,
      proposer: store.proposer,
      llm_backend: store.llmBackend,
      ucb_c: store.ucb1ExplorationC,
      max_iterations: store.maxIterations,
      patience: store.factoryPatience,
      stopping_tolerance: store.stoppingTolerance,
    };
    if (Object.keys(store.hpoTrials).length > 0)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (payload as any).hpo_trials = store.hpoTrials;
    if (Object.keys(store.hpoStartupTrials).length > 0)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (payload as any).hpo_startup_trials = store.hpoStartupTrials;
    try {
      const result = await startMutation.mutateAsync(payload);
      setErrorMsg(null);
      setJobId(result.job_id);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail ?? err?.message ?? "Failed to start full cycle");
    }
  };

  const hasModels = store.selectedModels.length > 0;

  return (
    <div className="flex flex-1 flex-col">
      {!isRunning && !isDone && (
        <div className="mt-6 flex flex-col gap-4">
          <div className="rounded border border-(--color-glass-border) bg-(--color-surface) p-5 text-center">
            <p className="text-[11px] text-(--color-text-secondary)">
              Config via the Committee wizard tabs. Deploy to start.
            </p>
          </div>

          {hasModels && (
            <div className="flex justify-center">
              <button
                onClick={handleDeploy}
                disabled={startMutation.isPending}
                className="rounded bg-(--color-accent-success) px-8 py-3 text-xs font-semibold tracking-[0.08em] text-(--color-text-inverse) uppercase transition hover:brightness-110"
              >
                {startMutation.isPending ? "Starting..." : "Run Pipeline"}
              </button>
            </div>
          )}

          {errorMsg && (
            <div className="rounded border border-[rgba(255,77,77,0.25)] bg-[rgba(255,77,77,0.08)] px-4 py-2 font-mono text-[11px] text-(--color-accent-danger)">
              {errorMsg}
            </div>
          )}
        </div>
      )}

      {isRunning && (
        <div className="mt-6 flex flex-col gap-4">
          <FullCycleProgress jobId={jobId!} onCancel={() => cancelMutation.mutate()} />
        </div>
      )}

      {isDone && (
        <div className="mt-6 flex flex-col gap-4">
          <FullCycleProgress jobId={jobId!} onCancel={() => {}} />
          <FullCycleResults jobId={jobId!} onRunAgain={() => setJobId(null)} />
        </div>
      )}
    </div>
  );
}
