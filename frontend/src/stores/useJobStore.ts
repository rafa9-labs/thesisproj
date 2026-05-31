import { create } from "zustand";
import type { Metrics, WsEvent, HpoTrialRow, OosPeriodResult } from "@/api/schemas";

const _processedEventIds = new Map<string, Set<number>>();

interface ModelProgress {
  phase: "pending" | "hpo" | "simulation" | "complete" | "failed";
  hpoTrial: number;
  hpoTotalTrials: number;
  simMonth: number;
  simTotalMonths: number;
}

export interface CycleState {
  model: string;
  cycleNumber: number;
  phase: "pending" | "hpo" | "simulation" | "complete";
  hpoTrials: HpoTrialRow[];
  bestTrial: HpoTrialRow | null;
  hpoTrialCurrent: number;
  hpoTrialTotal: number;
  testMonths: OosPeriodResult[];
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
  oosEquity: { period: number; modelName: string; equity: number | null; bh: number | null }[];
  periodTotals: number;
  activeTab: ActiveTab;
  cycles: CycleState[];
}

interface JobStore {
  activeJobs: Map<string, JobState>;
  selectedJobId: string | null;
  completedJobIds: Set<string>;

  startJob: (jobId: string, pair: string, models: string[]) => void;
  ensureJob: (jobId: string, pair: string, models: string[]) => void;
  handleWsEvent: (event: WsEvent) => void;
  selectJob: (jobId: string | null) => void;
  removeJob: (jobId: string) => void;
  getJob: (jobId: string) => JobState | undefined;
  setActiveTab: (jobId: string, tab: ActiveTab) => void;
  markCompleted: (jobId: string) => void;
  clearCompletedJobs: () => void;
}

export const useJobStore = create<JobStore>()((set, get) => ({
  activeJobs: new Map(),
  selectedJobId: null,
  completedJobIds: new Set(),

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
        periodTotals: 0,
        activeTab: "hpo-and-results",
        cycles: [],
      });
      return { activeJobs: next, selectedJobId: jobId };
    }),

  ensureJob: (jobId, pair, models) =>
    set((state) => {
      if (state.activeJobs.has(jobId)) return state;
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
        periodTotals: 0,
        activeTab: "hpo-and-results",
        cycles: [],
      });
      return { activeJobs: next };
    }),

  handleWsEvent: (event) =>
    set((state) => {
      if (import.meta.env.DEV) {
        const e = event as { event?: string; model?: string; job_id?: string; status?: string; trial_number?: number; phase?: string };
        console.log("[Store] handleWsEvent:", e.event, e.model ?? "", e.status ?? e.phase ?? e.trial_number ?? "");
      }
      const next = new Map(state.activeJobs);
      const jobId = (event as { job_id?: string }).job_id;
      if (!jobId) return state;

      const idx = (event as { _idx?: number })._idx;
      if (idx !== undefined) {
        const seen = _processedEventIds.get(jobId);
        if (seen && seen.has(idx)) return state;
        if (!seen) {
          _processedEventIds.set(jobId, new Set([idx]));
        } else {
          seen.add(idx);
        }
      }

      const job = next.get(jobId);
      if (!job) return state;

      const updated = { ...job };

      if (event.event === "job_started") {
        updated.status = "running";
        updated.progressText = "Backtest running...";
        updated.totalWork = event.total_work ?? 0;
        updated.activeTab = "hpo-and-results";
      }

      if (event.event === "cycle_started") {
        const cycle: CycleState = {
          model: event.model,
          cycleNumber: event.cycle_number ?? (updated.cycles.length + 1),
          phase: "hpo",
          hpoTrials: [],
          bestTrial: null,
          hpoTrialCurrent: 0,
          hpoTrialTotal: 0,
          testMonths: [],
        };
        updated.cycles = [...updated.cycles, cycle];
        updated.currentModel = event.model;
        updated.progressText = `Cycle ${cycle.cycleNumber}: ${event.model} started`;
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
          updated.cycles = updated.cycles.map((c) => {
            if (c.model === event.model) {
              return { ...c, phase: "complete" as const };
            }
            return c;
          });
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
        updated.cycles = updated.cycles.map((c) => {
          if (c.model === m) {
            return { ...c, hpoTrialTotal: totalTrials, hpoTrialCurrent: trial };
          }
          return c;
        });
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
        const modelName = (event as { model?: string }).model;
        if (modelName) {
          updated.cycles = updated.cycles.map((c) => {
            if (c.model === modelName) {
              const nextBest = row.score != null && (c.bestTrial == null || row.score > (c.bestTrial.score ?? -Infinity))
                ? row : c.bestTrial;
              return {
                ...c,
                phase: "hpo" as const,
                hpoTrials: [...c.hpoTrials, row],
                bestTrial: nextBest,
                hpoTrialCurrent: row.trial_number,
                hpoTrialTotal: Math.max(c.hpoTrialTotal, row.trial_number),
              };
            }
            return c;
          });
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

      if (event.event === "simulation_started") {
        updated.periodTotals = event.n_periods ?? updated.periodTotals;
        const entries: typeof updated.oosEquity = [];
        for (let p = 1; p <= (event.n_periods ?? 0); p++) {
          const bhEntry = (event.bh_curve as Array<{ period: number; bh: number }> | undefined)?.find((b) => b.period === p);
          entries.push({
            period: p,
            modelName: event.model,
            equity: null,
            bh: bhEntry?.bh ?? null,
          });
        }
        if (entries.length > 0) {
          const existing = new Set(updated.oosEquity.map((e) => `${e.period}:${e.modelName}`));
          const deduped = entries.filter((e) => !existing.has(`${e.period}:${e.modelName}`));
          updated.oosEquity = [...updated.oosEquity, ...deduped];
        }
        const simModel = (event as { model?: string }).model;
        if (simModel) {
          updated.cycles = updated.cycles.map((c) => {
            if (c.model === simModel) {
              return { ...c, phase: "simulation" as const };
            }
            return c;
          });
        }
      }

      if (event.event === "oos_result") {
        const periodResult: OosPeriodResult = {
          period: event.period,
          total_periods: event.total_periods,
          model: event.model,
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
          train_sharpe: event.train_sharpe,
          sharpe_gap_pct: event.sharpe_gap_pct,
          signals_raw: event.signals_raw,
          signals_passed_gate: event.signals_passed_gate,
        };
        updated.oosPeriods = [...updated.oosPeriods, periodResult];
        if (event.equity != null && event.equity_bh != null) {
          const existingIdx = updated.oosEquity.findIndex(
            (e) => e.period === event.period && e.modelName === event.model
          );
          const entry = { period: event.period, modelName: event.model, equity: event.equity, bh: event.equity_bh };
          if (existingIdx >= 0) {
            updated.oosEquity = [
              ...updated.oosEquity.slice(0, existingIdx),
              entry,
              ...updated.oosEquity.slice(existingIdx + 1),
            ];
          } else {
            updated.oosEquity = [...updated.oosEquity, entry];
          }
        }
        const oosModel = (event as { model?: string }).model;
        if (oosModel) {
          updated.cycles = updated.cycles.map((c) => {
            if (c.model === oosModel) {
              return {
                ...c,
                phase: "simulation" as const,
                testMonths: [...c.testMonths, periodResult],
              };
            }
            return c;
          });
        }
      }

      if (event.event === "job_complete") {
        updated.status = "completed";
        updated.progress = 100;
        updated.progressText = "Complete";
        updated.completedAt = new Date();
        const nextCompleted = new Set(state.completedJobIds);
        nextCompleted.add(jobId);
        next.set(jobId, updated);
        return { activeJobs: next, completedJobIds: nextCompleted };
      }

      if (event.event === "job_failed") {
        updated.status = "failed";
        updated.error = event.error;
        updated.progressText = `Failed: ${event.error}`;
        const nextCompleted = new Set(state.completedJobIds);
        nextCompleted.add(jobId);
        next.set(jobId, updated);
        return { activeJobs: next, completedJobIds: nextCompleted };
      }

      next.set(jobId, updated);
      return { activeJobs: next };
    }),

  selectJob: (jobId) => set({ selectedJobId: jobId }),

  removeJob: (jobId) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.delete(jobId);
      const nextCompleted = new Set(state.completedJobIds);
      nextCompleted.delete(jobId);
      _processedEventIds.delete(jobId);
      return {
        activeJobs: next,
        completedJobIds: nextCompleted,
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

  markCompleted: (jobId) =>
    set((state) => {
      if (state.completedJobIds.has(jobId)) return state;
      const nextCompleted = new Set(state.completedJobIds);
      nextCompleted.add(jobId);
      return { completedJobIds: nextCompleted };
    }),

  clearCompletedJobs: () =>
    set((state) => {
      const next = new Map(state.activeJobs);
      const removed: string[] = [];
      for (const [id, job] of next) {
        if (job.status === "completed" || job.status === "failed") {
          removed.push(id);
        }
      }
      for (const id of removed) next.delete(id);
      return { activeJobs: next, completedJobIds: new Set() };
    }),
}));