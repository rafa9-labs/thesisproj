import { create } from "zustand";
import { FC_PRESETS } from "@/lib/constants";

export const CORE_MODELS = [
  "logistic",
  "svm",
  "random_forest",
  "xgboost",
  "lightgbm",
  "catboost",
  "lstm",
  "ensemble_adaptive_regime",
];

export const ALL_MODELS = [
  "logistic",
  "svm",
  "random_forest",
  "decision_tree",
  "xgboost",
  "lightgbm",
  "catboost",
  "cnn",
  "lstm",
  "transformer",
  "gru",
  "gru_lstm",
  "meta_ensemble",
  "stacking_ensemble",
  "ensemble_adaptive_regime",
];

interface FullCycleState {
  pair: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  selectedModels: string[];
  trainMonths: number;
  testMonths: number;
  hpoSampler: string;
  cvBlocks: number;
  profilingInitTrials: number;
  profilingOptTrials: number;
  sweepNEstimators: number;
  sweepMaxDepth: number;
  skipFeatureSweep: boolean;
  useBorutaShap: boolean;
  borutaPercentile: number;
  borutaMaxIter: number;
  enablePhase3: boolean;
  enablePhase4: boolean;
  enablePhase5: boolean;
  enablePhase6: boolean;
  debugMode: boolean;
  proposer: string;
  llmBackend: string;
  ucb1ExplorationC: number;
  maxIterations: number;
  factoryPatience: number;
  stoppingTolerance: number;
  activePreset: string | null;
  hpoTrials: Record<string, number>;
  hpoStartupTrials: Record<string, number>;
  committeeTopK: number;
  committeeWeightMethod: string;
  committeeMinSharpe: number;
  deployedSessionId: string | null;
  deployedPair: string;
  deployedTimeframe: string;
  deployedJobId: string | null;
  executionMode: string;
  selectedHistoryJobId: string | null;
}

interface FullCycleActions {
  setPair: (v: string) => void;
  setTimeframe: (v: string) => void;
  setStartDate: (v: string) => void;
  setEndDate: (v: string) => void;
  setDateRange: (start: string, end: string) => void;
  toggleModel: (m: string) => void;
  selectModel: (m: string) => void;
  setSelectedModels: (v: string[]) => void;
  setTrainMonths: (v: number) => void;
  setTestMonths: (v: number) => void;
  setHpoSampler: (v: string) => void;
  setCvBlocks: (v: number) => void;
  setProfilingInitTrials: (v: number) => void;
  setProfilingOptTrials: (v: number) => void;
  setSweepNEstimators: (v: number) => void;
  setSweepMaxDepth: (v: number) => void;
  setSkipFeatureSweep: (v: boolean) => void;
  setUseBorutaShap: (v: boolean) => void;
  setBorutaPercentile: (v: number) => void;
  setBorutaMaxIter: (v: number) => void;
  setEnablePhase3: (v: boolean) => void;
  setEnablePhase4: (v: boolean) => void;
  setEnablePhase5: (v: boolean) => void;
  setEnablePhase6: (v: boolean) => void;
  setDebugMode: (v: boolean) => void;
  setProposer: (v: string) => void;
  setLlmBackend: (v: string) => void;
  setUcb1ExplorationC: (v: number) => void;
  setMaxIterations: (v: number) => void;
  setFactoryPatience: (v: number) => void;
  setStoppingTolerance: (v: number) => void;
  setHpoTrial: (model: string, value: number) => void;
  setHpoStartupTrial: (model: string, value: number) => void;
  setCommitteeTopK: (v: number) => void;
  setCommitteeWeightMethod: (v: string) => void;
  setCommitteeMinSharpe: (v: number) => void;
  setDeployedSession: (sessionId: string, pair: string, timeframe: string) => void;
  setDeployedJobId: (jobId: string | null) => void;
  clearDeployedSession: () => void;
  setExecutionMode: (v: string) => void;
  setSelectedHistoryJobId: (id: string | null) => void;
  applyPreset: (key: string) => void;
}

export const useFullCycleStore = create<FullCycleState & FullCycleActions>()((set) => ({
  pair: "EURUSD",
  timeframe: "H1",
  startDate: "",
  endDate: "",
  selectedModels: [...CORE_MODELS],
  trainMonths: 36,
  testMonths: 1,
  hpoSampler: "tpe",
  cvBlocks: 3,
  profilingInitTrials: 2,
  profilingOptTrials: 3,
  sweepNEstimators: 100,
  sweepMaxDepth: 5,
  skipFeatureSweep: false,
  useBorutaShap: true,
  borutaPercentile: 90,
  borutaMaxIter: 20,
  enablePhase3: true,
  enablePhase4: true,
  enablePhase5: true,
  enablePhase6: true,
  debugMode: false,
  proposer: "llm",
  llmBackend: "deepseek",
  ucb1ExplorationC: 2.0,
  maxIterations: 20,
  factoryPatience: 5,
  stoppingTolerance: 0.02,
  activePreset: null,
  hpoTrials: {},
  hpoStartupTrials: {},
  committeeTopK: 3,
  committeeWeightMethod: "sharpe_proportional",
  committeeMinSharpe: 0.0,
  deployedSessionId: null,
  deployedPair: "EURUSD",
  deployedTimeframe: "H1",
  deployedJobId: null,
  executionMode: "paper",
  selectedHistoryJobId: null,

  setPair: (v) => set({ pair: v }),
  setTimeframe: (v) => set({ timeframe: v }),
  setStartDate: (v) => set({ startDate: v }),
  setEndDate: (v) => set({ endDate: v }),
  setDateRange: (start, end) => set({ startDate: start, endDate: end }),

  toggleModel: (m) =>
    set((s) => ({
      selectedModels: s.selectedModels.includes(m)
        ? s.selectedModels.filter((x) => x !== m)
        : [...s.selectedModels, m],
      activePreset: null,
    })),

  selectModel: (m) => set({ selectedModels: [m], activePreset: null }),

  setSelectedModels: (v) => set({ selectedModels: v }),

  setTrainMonths: (v) => set({ trainMonths: v }),
  setTestMonths: (v) => set({ testMonths: v }),
  setHpoSampler: (v) => set({ hpoSampler: v }),
  setCvBlocks: (v) => set({ cvBlocks: v }),
  setProfilingInitTrials: (v) => set({ profilingInitTrials: v }),
  setProfilingOptTrials: (v) => set({ profilingOptTrials: v }),
  setSweepNEstimators: (v) => set({ sweepNEstimators: v }),
  setSweepMaxDepth: (v) => set({ sweepMaxDepth: v }),
  setSkipFeatureSweep: (v) => set({ skipFeatureSweep: v }),
  setUseBorutaShap: (v) => set({ useBorutaShap: v }),
  setBorutaPercentile: (v) => set({ borutaPercentile: v }),
  setBorutaMaxIter: (v) => set({ borutaMaxIter: v }),
  setEnablePhase3: (v) => set({ enablePhase3: v }),
  setEnablePhase4: (v) => set({ enablePhase4: v }),
  setEnablePhase5: (v) => set({ enablePhase5: v }),
  setEnablePhase6: (v) => set({ enablePhase6: v }),
  setDebugMode: (v) => set({ debugMode: v }),
  setProposer: (v) => set({ proposer: v }),
  setLlmBackend: (v) => set({ llmBackend: v }),
  setUcb1ExplorationC: (v) => set({ ucb1ExplorationC: v }),
  setMaxIterations: (v) => set({ maxIterations: v }),
  setFactoryPatience: (v) => set({ factoryPatience: v }),
  setStoppingTolerance: (v) => set({ stoppingTolerance: v }),

  setHpoTrial: (model, value) => set((s) => ({ hpoTrials: { ...s.hpoTrials, [model]: value } })),
  setHpoStartupTrial: (model, value) =>
    set((s) => ({ hpoStartupTrials: { ...s.hpoStartupTrials, [model]: value } })),

  setCommitteeTopK: (v) => set({ committeeTopK: v }),
  setCommitteeWeightMethod: (v) => set({ committeeWeightMethod: v }),
  setCommitteeMinSharpe: (v) => set({ committeeMinSharpe: v }),

  setDeployedSession: (sessionId, pair, timeframe) =>
    set({ deployedSessionId: sessionId, deployedPair: pair, deployedTimeframe: timeframe }),
  setDeployedJobId: (jobId) => set({ deployedJobId: jobId }),
  clearDeployedSession: () => set({ deployedSessionId: null, deployedJobId: null }),
  setExecutionMode: (v) => set({ executionMode: v }),
  setSelectedHistoryJobId: (id) => set({ selectedHistoryJobId: id }),

  applyPreset: (key) => {
    const preset = FC_PRESETS[key];
    if (!preset) return;
    const cfg = preset.config as Record<string, unknown>;
    set({
      activePreset: key,
      ...(cfg.selectedModels !== undefined && { selectedModels: cfg.selectedModels as string[] }),
    });
  },
}));
