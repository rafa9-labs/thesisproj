import { create } from "zustand";
import { useQueryClient } from "@tanstack/react-query";
import type { Metrics, WsEvent, HpoTrialRow, OosPeriodResult } from "@/api/schemas";

interface ModelProgress {
  phase: "pending" | "hpo" | "simulation" | "complete" | "failed";
  hpoTrial: number;
  hpoTotalTrials: number;
  simMonth: number;
  simTotalMonths: number;
}

type ActiveTab = "hpo-and-results" | "trade";

interface JobState {
  jobId: string;
  pair: string;
  models: string[];
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  progressText: string;
  completedModels: string[];
  currentModel: string | null;
  metrics: Map<string, Partial<Metrics>>;
  error: string | null;
  createdAt: Date;
  completedAt: Date | null;
  totalWork: number;
  completedWork: number;
  modelPhases: Map<string, ModelProgress>;
  hpoTrials: HpoTrialRow[];
  bestTrial: HpoTrialRow | null;
  oosPeriods: OosPeriodResult[];
  oosEquity: { time: number; model: number; bh: number }[];
  activeTab: ActiveTab;
}

interface JobStore {
  activeJobs: Map<string, JobState>;
  selectedJobId: string | null;

  startJob: (jobId: string, pair: string, models: string[]) => void;
  handleWsEvent: (event: WsEvent) => void;
  selectJob: (jobId: string | null) => void;
  removeJob: (jobId: string) => void;
  getJob: (jobId: string) => JobState | undefined;
  setActiveTab: (jobId: string, tab: ActiveTab) => void;
}

export const useJobStore = create<JobStore>()((set, get) => ({
  activeJobs: new Map(),
  selectedJobId: null,

  startJob: (jobId, pair, models) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.set(jobId, {
        jobId,
        pair,
        models,
        status: "pending",
        progress: 0,
        progressText: "Starting...",
        completedModels: [],
        currentModel: null,
        metrics: new Map(),
        error: null,
        createdAt: new Date(),
        completedAt: null,
        totalWork: 0,
        completedWork: 0,
        modelPhases: new Map(),
        hpoTrials: [],
        bestTrial: null,
        oosPeriods: [],
        oosEquity: [],
        activeTab: "hpo-and-results",
      });
      return { activeJobs: next, selectedJobId: jobId };
    }),

  handleWsEvent: (event) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      const jobId = (event as { job_id?: string }).job_id;
      if (!jobId) return state;

      const job = next.get(jobId);
      if (!job) return state;

      const updated = { ...job };

      if (event.event === "job_started") {
        updated.status = "running";
        updated.progressText = "Backtest running...";
        updated.totalWork = event.total_work ?? 0;
        updated.activeTab = "hpo-and-results";
      }

      if (event.event === "model_training") {
        if (event.status === "starting") {
          updated.currentModel = event.model;
          updated.progressText = `Training ${event.model}...`;
          updated.activeTab = "hpo-and-results";
        }
        if (event.status === "complete") {
          updated.completedModels = [...updated.completedModels, event.model];
          const nextMetrics = new Map(updated.metrics);
          nextMetrics.set(event.model, event.metrics ?? {});
          updated.metrics = nextMetrics;
          const phaseMap = new Map(updated.modelPhases);
          const mp = phaseMap.get(event.model) || { phase: "complete" as const, hpoTrial: 0, hpoTotalTrials: 0, simMonth: 0, simTotalMonths: 0 };
          phaseMap.set(event.model, { ...mp, phase: "complete" });
          updated.modelPhases = phaseMap;
          const totalModels = updated.models?.length || 1;
          updated.progressText = `${event.model} complete (${updated.completedModels.length}/${totalModels})`;
        }
      }

      if (event.event === "model_phase") {
        const m = event.model;
        const phaseMap = new Map(updated.modelPhases);
        const mp = phaseMap.get(m) || { phase: "pending" as const, hpoTrial: 0, hpoTotalTrials: 0, simMonth: 0, simTotalMonths: 0 };
        phaseMap.set(m, { ...mp, phase: event.phase as "hpo" | "simulation" });
        updated.modelPhases = phaseMap;
        updated.currentModel = m;
        if (event.phase === "hpo") {
          updated.progressText = `HPO: tuning ${m}...`;
        } else if (event.phase === "simulation") {
          updated.progressText = `Simulating ${m}...`;
        }
        if (event.total_work !== undefined) {
          updated.totalWork = event.total_work;
        }
      }

      if (event.event === "hpo_progress") {
        const m = event.model;
        const trial = event.trial ?? 0;
        const totalTrials = event.total_trials ?? event.n_trials ?? 0;
        const phaseMap = new Map(updated.modelPhases);
        const mp = phaseMap.get(m) || { phase: "hpo" as const, hpoTrial: 0, hpoTotalTrials: 0, simMonth: 0, simTotalMonths: 0 };
        phaseMap.set(m, { ...mp, phase: "hpo", hpoTrial: trial, hpoTotalTrials: totalTrials });
        updated.modelPhases = phaseMap;
        updated.completedWork = event.completed_work ?? updated.completedWork;
        updated.totalWork = event.total_work ?? updated.totalWork;
        updated.progress = event.progress_pct ?? updated.progress;
        updated.progressText = `HPO: trial ${trial}/${totalTrials} (${m})`;
      }

      if (event.event === "hpo_trial_result") {
        const row: HpoTrialRow = {
          trial_number: event.trial_number,
          score: event.score,
          params: event.params ?? {},
          best_score_so_far: event.best_score_so_far,
          trial_state: event.trial_state ?? "COMPLETE",
        };
        updated.hpoTrials = [...updated.hpoTrials, row];
        if (row.score != null && (updated.bestTrial == null || row.score > (updated.bestTrial.score ?? -Infinity))) {
          updated.bestTrial = row;
        }
      }

      if (event.event === "month_progress") {
        const m = event.model;
        const month = event.month ?? event.period ?? 0;
        const totalMonths = event.total_months ?? event.total_periods ?? 0;
        const phaseMap = new Map(updated.modelPhases);
        const mp = phaseMap.get(m) || { phase: "simulation" as const, hpoTrial: 0, hpoTotalTrials: 0, simMonth: 0, simTotalMonths: 0 };
        phaseMap.set(m, { ...mp, phase: "simulation", simMonth: month, simTotalMonths: totalMonths });
        updated.modelPhases = phaseMap;
        updated.completedWork = event.completed_work ?? updated.completedWork;
        updated.totalWork = event.total_work ?? updated.totalWork;
        updated.progress = event.progress_pct ?? updated.progress;
        updated.progressText = `${m}: month ${month}/${totalMonths}`;
      }

      if (event.event === "oos_result") {
        const periodResult: OosPeriodResult = {
          period: event.period,
          total_periods: event.total_periods,
          equity: event.equity,
          equity_bh: event.equity_bh,
          sharpe: event.sharpe,
          return_pct: event.return_pct,
          trades: event.trades,
          drawdown: event.drawdown,
          win_rate: event.win_rate,
          precision: event.precision,
          f1: event.f1,
          flat: event.flat,
        };
        updated.oosPeriods = [...updated.oosPeriods, periodResult];
        if (event.equity != null && event.equity_bh != null) {
          updated.oosEquity = [
            ...updated.oosEquity,
            { time: event.period, model: event.equity, bh: event.equity_bh },
          ];
        }
      }

      if (event.event === "job_complete") {
        updated.status = "completed";
        updated.progress = 100;
        updated.progressText = "Complete";
        updated.completedAt = new Date();
        try {
          const qc = useQueryClient();
          qc.invalidateQueries({ queryKey: ["jobs"] });
          qc.invalidateQueries({ queryKey: ["job-results", jobId] });
        } catch (_e) { void _e }
      }

      if (event.event === "job_failed") {
        updated.status = "failed";
        updated.error = event.error;
        updated.progressText = `Failed: ${event.error}`;
        try {
          const qc = useQueryClient();
          qc.invalidateQueries({ queryKey: ["jobs"] });
          qc.invalidateQueries({ queryKey: ["job", jobId] });
        } catch (_e) { void _e }
      }

      next.set(jobId, updated);
      return { activeJobs: next };
    }),

  selectJob: (jobId) => set({ selectedJobId: jobId }),

  removeJob: (jobId) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.delete(jobId);
      return {
        activeJobs: next,
        selectedJobId: state.selectedJobId === jobId ? null : state.selectedJobId,
      };
    }),

  getJob: (jobId) => get().activeJobs.get(jobId),

  setActiveTab: (jobId, tab) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      const job = next.get(jobId);
      if (job) {
        next.set(jobId, { ...job, activeTab: tab });
      }
      return { activeJobs: next };
    }),
}));