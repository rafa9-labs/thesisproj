import { create } from "zustand";
import type {
  FullCycleStatusResponse,
  FullCycleResultsResponse,
  FactoryIterationRecord,
  CommitteeConfigSchema,
  LogEntry,
  TrustScoreResult,
} from "@/api/schemas";

const MAX_LOG_ENTRIES = 1000;

export type PhaseStatus = "pending" | "active" | "complete" | "failed" | "skipped";

export interface Phase1Cache {
  lockedFeaturesCount: number;
  prunedFeaturesCount: number;
  topImportanceFeature: string;
  survivors: string[];
  pruned: string[];
}

export interface Phase2Cache {
  hpoStatus: Record<string, string>;
  survivingModels: string[];
  hpoModelParamsCount: number;
  hpoScores: Record<string, number | null>;
}

export interface Phase3Cache {
  committeeConfig: CommitteeConfigSchema | null;
}

export interface Phase4Cache {
  foldConsistencyCv: number;
  pbo: number;
  dsr: number;
  trustScore: TrustScoreResult | null;
  seedSharpes: number[];
  seedPass: boolean;
  regimeCoverage: Record<string, unknown> | null;
}

export interface Phase5Cache {
  bestSharpe: number;
  totalIterations: number;
  acceptedCount: number;
  factoryHistory: FactoryIterationRecord[];
  stopReason: string;
}

export interface FinalCache {
  finalFullWfo: Record<string, unknown> | null;
  finalFoldCv: number;
  finalFoldPass: boolean;
  finalRegimeCoverage: Record<string, unknown> | null;
  finalSeedSharpe: number;
  finalSeedPass: boolean;
  snapshotDir: string | null;
}

export interface PhaseCache {
  1: Phase1Cache | null;
  2: Phase2Cache | null;
  3: Phase3Cache | null;
  4: Phase4Cache | null;
  5: Phase5Cache | null;
  6: FinalCache | null;
}

export interface CommitteeMonitorState {
  // Connection
  selectedJobId: string | null;

  // Live status (updated every 2s from status.json polling)
  phase: string;
  phaseNumber: number;
  phaseProgress: string;
  iteration: number;
  totalIterations: number;
  currentAction: string;
  bestSharpeSoFar: number;
  error: string | null;
  startedAt: string;
  survivingModels: string[];
  lockedFeaturesCount: number;

  // View state
  viewPhase: number;
  mode: "live" | "review";

  // Full results (from results.json — only available at job completion)
  results: FullCycleResultsResponse | null;

  // Accumulated live trajectory (built from status polling)
  sharpeTrajectory: { iteration: number; sharpe: number }[];

  // HPO scores (from status polling — per-model best Sharpe)
  hpoScores: Record<string, number | null>;

  // Phase timing (seconds elapsed per phase, from backend)
  phaseTimings: Record<string, number>;

  // Live phase data (enriched by status endpoint during execution)
  liveFeatureNamesLocked: string[];
  liveFeatureNamesPruned: string[];
  livePrunedCount: number;
  liveHpoStatus: Record<string, string>;
  liveCommitteeConfig: Record<string, unknown> | null;
  liveFoldConsistencyCv: number | null;
  livePbo: number | null;
  liveDsr: number | null;
  liveTrustScore: TrustScoreResult | null;
  liveRegimeCoverage: Record<string, { sharpe: number; trades: number; folds_active: number; covered: boolean }> | null;
  liveSeedSharpes: number[];
  liveSeedAvgSharpe: number | null;
  liveSeedPass: boolean | null;
  liveFactoryAcceptedCount: number;
  liveFactoryLastDelta: number | null;
  liveFactoryLastAction: string;
  liveFactoryLastRegime: string;
  liveFactoryLastAccepted: boolean | null;
  // Phase 4 WFO live progress
  liveWfoFoldProgress: string;
  liveWfoFoldSharpes: number[];
  liveWfoFoldTrades: number[];
  liveWfoRunningAvgSharpe: number | null;

  // Phase caches (populated from results.json when available)
  phaseCache: PhaseCache;

  // Log buffer
  logs: LogEntry[];
  logsNextIndex: number;
  logsAutoScroll: boolean;

  // Actions
  selectJob: (jobId: string) => void;
  reset: () => void;
  updateFromStatus: (status: FullCycleStatusResponse) => void;
  ingestResults: (results: FullCycleResultsResponse) => void;
  appendLogs: (entries: LogEntry[], nextIndex: number) => void;
  setViewPhase: (phase: number) => void;
  setLogsAutoScroll: (enabled: boolean) => void;
}

const INITIAL_PHASE_CACHE: PhaseCache = {
  1: null,
  2: null,
  3: null,
  4: null,
  5: null,
  6: null,
};

function extractPhaseCache(results: FullCycleResultsResponse): PhaseCache {
  return {
    1: {
      lockedFeaturesCount: results.locked_features_count,
      prunedFeaturesCount: results.pruned_features_count,
      topImportanceFeature: results.top_importance_feature,
      survivors: results.phase0_survivors ?? [],
      pruned: results.phase0_pruned ?? [],
    },
    2: {
      hpoStatus: results.hpo_status ?? {},
      survivingModels: results.phase0_survivors ?? [],
      hpoModelParamsCount: results.hpo_model_params_count ?? 0,
      hpoScores: {},
    },
    3: {
      committeeConfig:
        (results.racecar_committee_config as CommitteeConfigSchema) ?? null,
    },
    4: {
      foldConsistencyCv: results.phase3_fold_consistency_cv,
      pbo: results.pbo,
      dsr: results.dsr,
      trustScore: results.trust_score ?? null,
      seedSharpes: [
        results.phase3_seed_robustness_sharpe,
      ],
      seedPass: results.phase3_seed_robustness_pass,
      regimeCoverage: results.phase3_regime_coverage ?? null,
    },
    5: {
      bestSharpe: results.factory_best_sharpe,
      totalIterations: results.factory_total_iterations,
      acceptedCount: results.factory_accepted_count,
      factoryHistory: results.factory_history ?? [],
      stopReason: results.factory_stop_reason,
    },
    6: {
      finalFullWfo: results.final_full_wfo ?? null,
      finalFoldCv: results.final_fold_consistency_cv,
      finalFoldPass: results.final_fold_consistency_pass,
      finalRegimeCoverage: results.final_regime_coverage ?? null,
      finalSeedSharpe: results.final_seed_robustness_sharpe,
      finalSeedPass: results.final_seed_robustness_pass,
      snapshotDir: results.snapshot_dir ?? null,
    },
  };
}

export const useCommitteeMonitorStore = create<CommitteeMonitorState>()(
  (set, get) => ({
    selectedJobId: null,
    phase: "",
    phaseNumber: 0,
    phaseProgress: "",
    iteration: 0,
    totalIterations: 0,
    currentAction: "",
    bestSharpeSoFar: 0,
    error: null,
    startedAt: "",
    survivingModels: [],
    lockedFeaturesCount: 0,
    viewPhase: 1,
    mode: "live",
    results: null,
    sharpeTrajectory: [],
    phaseCache: { ...INITIAL_PHASE_CACHE },
    hpoScores: {},
    phaseTimings: {},
    // Live phase data defaults
    liveFeatureNamesLocked: [],
    liveFeatureNamesPruned: [],
    livePrunedCount: 0,
    liveHpoStatus: {},
    liveCommitteeConfig: null,
    liveFoldConsistencyCv: null,
    livePbo: null,
    liveDsr: null,
    liveTrustScore: null,
    liveRegimeCoverage: null,
    liveSeedSharpes: [],
    liveSeedAvgSharpe: null,
    liveSeedPass: null,
    liveFactoryAcceptedCount: 0,
    liveFactoryLastDelta: null,
    liveFactoryLastAction: "",
    liveFactoryLastRegime: "",
    liveFactoryLastAccepted: null,
    liveWfoFoldProgress: "",
    liveWfoFoldSharpes: [],
    liveWfoFoldTrades: [],
    liveWfoRunningAvgSharpe: null,
    logs: [],
    logsNextIndex: 0,
    logsAutoScroll: true,

    selectJob: (jobId) => {
      const state = get();
      if (state.selectedJobId === jobId) return;
      // Full reset for new job
      set({
        selectedJobId: jobId,
        phase: "",
        phaseNumber: 0,
        phaseProgress: "",
        iteration: 0,
        totalIterations: 0,
        currentAction: "",
        bestSharpeSoFar: 0,
        error: null,
        startedAt: "",
        survivingModels: [],
        lockedFeaturesCount: 0,
        viewPhase: 1,
        mode: "live",
        results: null,
        sharpeTrajectory: [],
        hpoScores: {},
        phaseTimings: {},
        phaseCache: { ...INITIAL_PHASE_CACHE },
        logs: [],
        logsNextIndex: 0,
        logsAutoScroll: true,
      });
    },

    reset: () => {
      set({
        selectedJobId: null,
        phase: "",
        phaseNumber: 0,
        phaseProgress: "",
        iteration: 0,
        totalIterations: 0,
        currentAction: "",
        bestSharpeSoFar: 0,
        error: null,
        startedAt: "",
        survivingModels: [],
        lockedFeaturesCount: 0,
        viewPhase: 1,
        mode: "live",
        results: null,
        sharpeTrajectory: [],
        hpoScores: {},
        phaseTimings: {},
        phaseCache: { ...INITIAL_PHASE_CACHE },
        logs: [],
        logsNextIndex: 0,
        logsAutoScroll: true,
      });
    },

    updateFromStatus: (status) => {
      const state = get();
      const newPhaseNumber = status.phase_number ?? 0;

      // Accumulate Sharpe trajectory for Phase 5 chart
      let nextTrajectory = state.sharpeTrajectory;
      if (newPhaseNumber >= 5 && status.best_sharpe_so_far !== state.bestSharpeSoFar) {
        const existing = nextTrajectory.find(
          (p) => p.iteration === status.iteration,
        );
        if (!existing && status.iteration > 0) {
          nextTrajectory = [
            ...nextTrajectory,
            { iteration: status.iteration, sharpe: status.best_sharpe_so_far },
          ];
        }
      }

      const isTerminal =
        status.phase === "completed" ||
        status.phase === "failed" ||
        status.phase === "validation_failed" ||
        status.phase === "cancelled";

      set({
        phase: status.phase ?? "",
        phaseNumber: newPhaseNumber,
        phaseProgress: status.phase_progress ?? "",
        iteration: status.iteration ?? 0,
        totalIterations: status.total_iterations ?? 0,
        currentAction: status.current_action ?? "",
        bestSharpeSoFar: status.best_sharpe_so_far ?? 0,
        error: status.error || null,
        startedAt: status.started_at ?? "",
        survivingModels: status.surviving_models ?? [],
        lockedFeaturesCount: status.locked_features_count ?? 0,
        hpoScores: status.hpo_model_scores ?? state.hpoScores,
        phaseTimings: status.phase_timings ?? state.phaseTimings,
        // Live phase data (ingested incrementally from status endpoint)
        liveFeatureNamesLocked: status.feature_names_locked ?? state.liveFeatureNamesLocked,
        liveFeatureNamesPruned: status.feature_names_pruned ?? state.liveFeatureNamesPruned,
        livePrunedCount: status.pruned_count ?? state.livePrunedCount,
        liveHpoStatus: status.hpo_status ?? state.liveHpoStatus,
        liveCommitteeConfig: status.committee_config ?? state.liveCommitteeConfig,
        liveFoldConsistencyCv: status.fold_consistency_cv ?? state.liveFoldConsistencyCv,
        livePbo: status.phase4_pbo ?? state.livePbo,
        liveDsr: status.phase4_dsr ?? state.liveDsr,
        liveTrustScore: status.trust_score ?? state.liveTrustScore,
        liveRegimeCoverage: status.regime_coverage ?? state.liveRegimeCoverage,
        liveSeedSharpes: status.seed_sharpes ?? state.liveSeedSharpes,
        liveSeedAvgSharpe: status.seed_avg_sharpe ?? state.liveSeedAvgSharpe,
        liveSeedPass: status.seed_pass ?? state.liveSeedPass,
        liveFactoryAcceptedCount: status.factory_accepted_count ?? state.liveFactoryAcceptedCount,
        liveFactoryLastDelta: status.factory_last_delta ?? state.liveFactoryLastDelta,
        liveFactoryLastAction: status.factory_last_action ?? "",
        liveFactoryLastRegime: status.factory_last_regime ?? "",
        liveFactoryLastAccepted: status.factory_last_accepted ?? null,
        liveWfoFoldProgress: status.wfo_fold_progress ?? state.liveWfoFoldProgress,
        liveWfoFoldSharpes: status.wfo_fold_sharpes ?? state.liveWfoFoldSharpes,
        liveWfoFoldTrades: status.wfo_fold_trades ?? state.liveWfoFoldTrades,
        liveWfoRunningAvgSharpe: status.wfo_running_avg_sharpe ?? state.liveWfoRunningAvgSharpe,
        // Follow backend phase for live tracking
        viewPhase: isTerminal
          ? 6
          : state.mode === "live"
            ? Math.max(1, newPhaseNumber || 1)
            : state.viewPhase,
        mode: isTerminal ? "review" : state.mode,
        sharpeTrajectory: nextTrajectory,
      });
    },

    ingestResults: (results) => {
      const phaseCache = extractPhaseCache(results);
      const isTerminal =
        results.status === "completed" ||
        results.status === "validation_failed" ||
        results.status === "cancelled";

      set({
        results,
        phaseCache,
        // Populate sharpeTrajectory from factory_history if available
        sharpeTrajectory:
          results.factory_history && results.factory_history.length > 0
            ? results.factory_history.map((r) => ({
                iteration: r.iteration,
                sharpe: r.after_sharpe,
              }))
            : get().sharpeTrajectory,
        mode: isTerminal ? "review" : get().mode,
        viewPhase: isTerminal ? 6 : get().viewPhase,
        survivingModels:
          results.phase0_survivors?.length
            ? results.phase0_survivors
            : get().survivingModels,
        lockedFeaturesCount:
          results.locked_features_count ?? get().lockedFeaturesCount,
      });
    },

    appendLogs: (entries, nextIndex) => {
      if (!entries.length) return;
      const state = get();
      const existingIndices = new Set(state.logs.map((l) => l.index));
      const newEntries = entries.filter((e) => !existingIndices.has(e.index));
      if (!newEntries.length) {
        set({ logsNextIndex: nextIndex });
        return;
      }
      const combined = [...state.logs, ...newEntries];
      // Trim to max
      const trimmed = combined.slice(-MAX_LOG_ENTRIES);
      set({
        logs: trimmed,
        logsNextIndex: nextIndex,
      });
    },

    setViewPhase: (phase) => {
      const state = get();
      const isTerminal =
        state.phase === "completed" ||
        state.phase === "failed" ||
        state.phase === "validation_failed" ||
        state.phase === "cancelled";

      // User can only click forward if phase has cache or is terminal
      // Can click back freely (past phases have already completed)
      // Can click current phase (follows live)
      const clampedPhase = Math.max(1, Math.min(6, phase));
      const isForward = clampedPhase > state.phaseNumber;
      const hasCache = state.phaseCache[clampedPhase as keyof PhaseCache] !== null;

      if (isForward && !isTerminal && !hasCache) {
        // Block — no data yet for this future phase
        return;
      }

      const newMode =
        clampedPhase === state.phaseNumber && !isTerminal ? "live" : "review";

      set({
        viewPhase: clampedPhase,
        mode: newMode,
      });
    },

    setLogsAutoScroll: (enabled) => {
      set({ logsAutoScroll: enabled });
    },
  }),
);
