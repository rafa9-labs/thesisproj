import { useState, useEffect, useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useStartFullCycle, useFullCycleStatus } from "@/api/queries";
import apiClient from "@/api/client";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { TabBar } from "@/components/shared/TabBar";
import { CommitteeValidationBar } from "./CommitteeValidationBar";
import { AssetTimeTab } from "./AssetTimeTab";
import { ModelsTab } from "./ModelsTab";
import { PipelineTab } from "./PipelineTab";
import { FullCycleProgress } from "./FullCycleProgress";
import type { FullCycleRequest } from "@/api/schemas";

const CONFIG_TABS = [
  { key: "asset", label: "Assets & Time" },
  { key: "models", label: "Models" },
  { key: "pipeline", label: "Pipeline" },
];

const MONITOR_TAB = { key: "monitor", label: "Monitor" };

export function CommitteePage() {
  const store = useFullCycleStore();
  const startMutation = useStartFullCycle();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const urlJobId = searchParams.get("jobId") || null;
  const [activeTab, setActiveTab] = useState(urlJobId ? "monitor" : "asset");
  const [jobId, setJobId] = useState<string | null>(urlJobId);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Clean URL param on mount (external navigation from running bar)
  useEffect(() => {
    if (urlJobId) {
      const next = new URLSearchParams(searchParams);
      next.delete("jobId");
      setSearchParams(next, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      start_date: store.startDate || undefined,
      end_date: store.endDate || undefined,
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
      setActiveTab("monitor");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail ?? err?.message ?? "Failed to start full cycle");
    }
  };

  const hasModels = store.selectedModels.length > 0;
  const canDeploy = hasModels && !startMutation.isPending;

  const showMonitor = isRunning || isDone;
  const tabs = useMemo(
    () => (showMonitor ? [...CONFIG_TABS, MONITOR_TAB] : CONFIG_TABS),
    [showMonitor],
  );

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="pt-1">
        <TabBar tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      <div className="h-6 shrink-0" />

      <div className="flex-1 overflow-y-auto pb-6">
        {/* Config tabs — always visible */}
        {activeTab === "asset" && <AssetTimeTab />}
        {activeTab === "models" && <ModelsTab />}
        {activeTab === "pipeline" && <PipelineTab />}

        {/* Monitor tab — shows progress when running, progress + results when done */}
        {activeTab === "monitor" && (
          <div className="flex flex-col gap-4">
            {isRunning && (
              <FullCycleProgress jobId={jobId!} onCancel={() => cancelMutation.mutate()} />
            )}
            {isDone && (
              <>
                <FullCycleProgress jobId={jobId!} onCancel={() => {}} />
                <div className="flex flex-col items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-6 text-center">
                  <span className="font-mono text-sm text-(--color-text-primary)">
                    Committee run complete
                  </span>
                  <span className="text-[11px] text-(--color-text-muted)">
                    Full diagnostics, equity curves, and trade log are available in the results viewer.
                  </span>
                  <button
                    onClick={() => navigate(`/results/${jobId}?type=committee`)}
                    className="cursor-pointer rounded-sm border-none bg-(--color-brand) px-6 py-2 text-[11px] font-semibold tracking-[0.06em] text-(--color-text-inverse) uppercase"
                  >
                    View Full Results
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {errorMsg && (
          <div className="mt-6 rounded border border-[rgba(255,77,77,0.25)] bg-[rgba(255,77,77,0.08)] px-4 py-2 font-mono text-[11px] text-(--color-accent-danger)">
            {errorMsg}
          </div>
        )}
      </div>

      {/* Validation bar — hidden while running */}
      {!isRunning && (
        <CommitteeValidationBar
          canDeploy={canDeploy}
          isSubmitting={startMutation.isPending}
          onDeploy={handleDeploy}
        />
      )}
    </div>
  );
}
