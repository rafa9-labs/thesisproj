import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
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
  status: "pending" | "running" | "completed" | "failed" | "queued";
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
  elapsedSec: number;
  etaSec: number | null;
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
  unreadCompletedCount: number;
  completedJobs: Array<{ jobId: string; models: string[]; pair: string; status: string; completedAt: Date }>;

  startJob: (jobId: string, pair: string, models: string[]) => void;
  ensureJob: (jobId: string, pair: string, models: string[]) => void;
  handleWsEvent: (event: WsEvent) => void;
  selectJob: (jobId: string | null) => void;
  removeJob: (jobId: string) => void;
  getJob: (jobId: string) => JobState | undefined;
  setActiveTab: (jobId: string, tab: ActiveTab) => void;
  markCompleted: (jobId: string) => void;
  clearCompletedJobs: () => void;
  clearUnreadCount: () => void;
}

const MAX_JOBS = 5;

function _evictOldest(next: Map<string, JobState>, completedJobIds: Set<string>): string | null {
  let oldestCompletedId = "";
  let oldestCompletedDate = new Date("2099-12-31");
  let oldestAnyId = "";
  let oldestAnyDate = new Date();
  for (const [id, j] of next) {
    if (j.createdAt < oldestAnyDate) {
      oldestAnyDate = j.createdAt;
      oldestAnyId = id;
    }
    if ((j.status === "completed" || j.status === "failed") && j.createdAt < oldestCompletedDate) {
      oldestCompletedDate = j.createdAt;
      oldestCompletedId = id;
    }
  }
  const evictId = oldestCompletedId || oldestAnyId;
  if (evictId) {
    next.delete(evictId);
    completedJobIds.delete(evictId);
    _processedEventIds.delete(evictId);
  }
  return evictId;
}

function _toMap<K, V>(value: unknown): Map<K, V> {
  if (value instanceof Map) return new Map(value) as Map<K, V>;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return new Map(Object.entries(value)) as Map<K, V>;
  }
  return new Map<K, V>();
}

function _toSet<T>(value: unknown): Set<T> {
  if (value instanceof Set) return new Set(value);
  if (Array.isArray(value)) return new Set(value);
  return new Set<T>();
}

const _serialize = (state: unknown): string => {
  return JSON.stringify(state, (_key, value) => {
    if (value instanceof Map) return { __zt: "Map", v: Array.from(value.entries()) };
    if (value instanceof Set) return { __zt: "Set", v: Array.from(value) };
    if (value instanceof Date) return { __zt: "Date", v: value.toISOString() };
    return value;
  });
};

const _deserialize = (str: string): unknown => {
  const parsed = JSON.parse(str, (_key, value) => {
    if (value && typeof value === "object" && value.__zt) {
      if (value.__zt === "Map") return new Map(value.v);
      if (value.__zt === "Set") return new Set(value.v);
      if (value.__zt === "Date") return new Date(value.v);
    }
    return value;
  });
  if (parsed?.state?.activeJobs && !(parsed.state.activeJobs instanceof Map)) {
    const raw = parsed.state.activeJobs;
    if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
      parsed.state.activeJobs = new Map(Object.entries(raw));
    } else {
      parsed.state.activeJobs = new Map();
    }
  }
  if (parsed?.state?.completedJobIds && !(parsed.state.completedJobIds instanceof Set)) {
    const raw = parsed.state.completedJobIds;
    if (Array.isArray(raw)) {
      parsed.state.completedJobIds = new Set(raw);
    } else if (typeof raw === "object" && raw !== null && Symbol.iterator in raw) {
      parsed.state.completedJobIds = new Set(raw as Iterable<unknown>);
    } else {
      parsed.state.completedJobIds = new Set();
    }
  }
  return parsed;
};

export const useJobStore = create<JobStore>()(
  persist(
    (set, get) => ({
      activeJobs: new Map(),
      selectedJobId: null,
      completedJobIds: new Set(),
      unreadCompletedCount: 0,
      completedJobs: [],

      startJob: (jobId, pair, models) =>
        set((state) => {
          const next = _toMap(state.activeJobs);
          const nextCompleted = _toSet(state.completedJobIds);

          // FIFO: if at capacity, evict oldest completed, then oldest any
          if (next.size >= MAX_JOBS) {
            _evictOldest(next, nextCompleted);
          }

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
            elapsedSec: 0,
            etaSec: null,
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
          const next = _toMap(state.activeJobs);
          const existing = next.get(jobId);
          if (existing) {
            if (existing.status === "stale") {
              next.set(jobId, {
                ...existing,
                status: "running",
                pair: pair || existing.pair,
                models: models.length > 0 ? models : existing.models,
                progressText: existing.progressText || "Reconnecting...",
              });
              return { activeJobs: next };
            }
            const needsUpdate = existing.pair !== pair || existing.models.join(",") !== models.join(",");
            if (needsUpdate) {
              next.set(jobId, { ...existing, pair, models });
              return { activeJobs: next };
            }
            return state;
          }

          // FIFO: if at capacity, evict oldest completed, then oldest any
          if (next.size >= MAX_JOBS) {
            const nextCompleted = _toSet(state.completedJobIds);
            _evictOldest(next, nextCompleted);
            state.completedJobIds = nextCompleted;
          }

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
            elapsedSec: 0,
            etaSec: null,
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
        const e = event as {
          event?: string;
          model?: string;
          job_id?: string;
          status?: string;
          trial_number?: number;
          phase?: string;
        };
        console.log(
          "[Store] handleWsEvent:",
          e.event,
          e.model ?? "",
          e.status ?? e.phase ?? e.trial_number ?? "",
        );
      }
      const next = _toMap(state.activeJobs);
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
      if (!job) {
        const eventModel = (event as { model?: string }).model;
        const newJob: JobState = {
          jobId,
          pair: "",
          models: eventModel ? [eventModel] : [],
          status: "running",
          progress: 0,
          progressText: "Backtest running...",
          completedModels: [],
          currentModel: eventModel ?? null,
          metrics: new Map(),
          error: null,
          createdAt: new Date(),
          completedAt: null,
          totalWork: (event as { total_work?: number }).total_work ?? 0,
          completedWork: 0,
          modelPhases: new Map(),
          hpoTrials: [],
          bestTrial: null,
          oosPeriods: [],
          oosEquity: [],
          periodTotals: 0,
          activeTab: "hpo-and-results",
          cycles: eventModel ? [{
            model: eventModel,
            cycleNumber: 1,
            phase: "hpo" as const,
            hpoTrials: [],
            bestTrial: null,
            hpoTrialCurrent: 0,
            hpoTrialTotal: 0,
            testMonths: [],
          }] : [],
        };
        next.set(jobId, newJob);
        if (event.event === "cycle_started" || event.event === "job_started" ||
            event.event === "hpo_progress") {
          return { activeJobs: next };
        }
      }

      const entry = next.get(jobId);
      if (!entry) {
        return { activeJobs: next };
      }

      const updated = { ...entry };

      if (event.event === "job_started") {
        updated.status = "running";
        updated.progressText = "Backtest running...";
        updated.totalWork = event.total_work ?? 0;
        updated.activeTab = "hpo-and-results";
      }

      if (event.event === "cycle_started") {
        const existingIdx = updated.cycles.findIndex((c) => c.model === event.model);
        const cycle: CycleState = {
          model: event.model,
          cycleNumber: event.cycle_number ?? updated.cycles.length + 1,
          phase: "hpo",
          hpoTrials: [],
          bestTrial: null,
          hpoTrialCurrent: 0,
          hpoTrialTotal: 0,
          testMonths: [],
        };
        if (existingIdx >= 0) {
          const existing = updated.cycles[existingIdx];
          updated.cycles = updated.cycles.map((c, i) => i === existingIdx
            ? { ...existing, ...cycle, hpoTrials: existing.hpoTrials, bestTrial: existing.bestTrial }
            : c);
        } else {
          updated.cycles = [...updated.cycles, cycle];
        }
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
          const nextMetrics = _toMap<string, Partial<Metrics>>(updated.metrics);
          nextMetrics.set(event.model, event.metrics ?? {});
          updated.metrics = nextMetrics;
          const phaseMap = _toMap<string, ModelProgress>(updated.modelPhases);
          const mp = phaseMap.get(event.model) || {
            phase: "complete" as const,
            hpoTrial: 0,
            hpoTotalTrials: 0,
            simMonth: 0,
            simTotalMonths: 0,
          };
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
        const phaseMap = _toMap<string, ModelProgress>(updated.modelPhases);
        const mp = phaseMap.get(m) || {
          phase: "pending" as const,
          hpoTrial: 0,
          hpoTotalTrials: 0,
          simMonth: 0,
          simTotalMonths: 0,
        };
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
        const phaseMap = _toMap<string, ModelProgress>(updated.modelPhases);
        const mp = phaseMap.get(m) || {
          phase: "hpo" as const,
          hpoTrial: 0,
          hpoTotalTrials: 0,
          simMonth: 0,
          simTotalMonths: 0,
        };
        phaseMap.set(m, { ...mp, phase: "hpo", hpoTrial: trial, hpoTotalTrials: totalTrials });
        updated.modelPhases = phaseMap;
        updated.completedWork = event.completed_work ?? updated.completedWork;
        updated.totalWork = event.total_work ?? updated.totalWork;
        updated.progress = event.progress_pct ?? updated.progress;
        updated.elapsedSec = event.elapsed_seconds ?? updated.elapsedSec;
        updated.etaSec = event.eta_seconds ?? updated.etaSec;
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
        if (
          row.score != null &&
          (updated.bestTrial == null || row.score > (updated.bestTrial.score ?? -Infinity))
        ) {
          updated.bestTrial = row;
        }
        const modelName = (event as { model?: string }).model;
        if (modelName) {
          updated.cycles = updated.cycles.map((c) => {
            if (c.model === modelName) {
              const nextBest =
                row.score != null &&
                (c.bestTrial == null || row.score > (c.bestTrial.score ?? -Infinity))
                  ? row
                  : c.bestTrial;
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
          const hasMatchingCycle = updated.cycles.some((c) => c.model === modelName);
          if (!hasMatchingCycle) {
            updated.cycles = [...updated.cycles, {
              model: modelName,
              cycleNumber: updated.cycles.length + 1,
              phase: "hpo" as const,
              hpoTrials: [row],
              bestTrial: row.score != null ? row : null,
              hpoTrialCurrent: row.trial_number,
              hpoTrialTotal: row.trial_number,
              testMonths: [],
            }];
          }
        }
      }

      if (event.event === "month_progress") {
        const m = event.model;
        const month = event.month ?? event.period ?? 0;
        const totalMonths = event.total_months ?? event.total_periods ?? 0;
        const phaseMap = _toMap<string, ModelProgress>(updated.modelPhases);
        const mp = phaseMap.get(m) || {
          phase: "simulation" as const,
          hpoTrial: 0,
          hpoTotalTrials: 0,
          simMonth: 0,
          simTotalMonths: 0,
        };
        phaseMap.set(m, {
          ...mp,
          phase: "simulation",
          simMonth: month,
          simTotalMonths: totalMonths,
        });
        updated.modelPhases = phaseMap;
        updated.completedWork = event.completed_work ?? updated.completedWork;
        updated.totalWork = event.total_work ?? updated.totalWork;
        updated.progress = event.progress_pct ?? updated.progress;
        updated.elapsedSec = event.elapsed_seconds ?? updated.elapsedSec;
        updated.etaSec = event.eta_seconds ?? updated.etaSec;
        updated.progressText = `${m}: month ${month}/${totalMonths}`;
      }

      if (event.event === "simulation_started") {
        updated.periodTotals = event.n_periods ?? updated.periodTotals;
        const entries: typeof updated.oosEquity = [];
        for (let p = 1; p <= (event.n_periods ?? 0); p++) {
          const bhEntry = (
            event.bh_curve as Array<{ period: number; bh: number }> | undefined
          )?.find((b) => b.period === p);
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
          sharpe_ann: event.sharpe_ann,
          wins: event.wins,
          return_pct: event.return_pct,
          trades: event.trades,
          drawdown: event.drawdown,
          win_rate: event.win_rate,
          precision: event.precision,
          f1: event.f1,
          flat: event.flat,
          train_sharpe: event.train_sharpe,
          train_sharpe_is_objective: event.train_sharpe_is_objective,
          sharpe_gap_pct: event.sharpe_gap_pct,
          signals_raw: event.signals_raw,
          signals_passed_gate: event.signals_passed_gate,
          directional_accuracy: event.directional_accuracy,
          active_rate: event.active_rate,
          signal_coverage: event.signal_coverage,
          profit_per_hit: event.profit_per_hit,
          outperformance: event.outperformance,
        };
        updated.oosPeriods = [...updated.oosPeriods, periodResult];
        if (event.equity != null && event.equity_bh != null) {
          const existingIdx = updated.oosEquity.findIndex(
            (e) => e.period === event.period && e.modelName === event.model,
          );
          const entry = {
            period: event.period,
            modelName: event.model,
            equity: event.equity,
            bh: event.equity_bh,
          };
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
        const nextCompleted = _toSet(state.completedJobIds);
        nextCompleted.add(jobId);
        const entry = {
          jobId,
          models: updated.models,
          pair: updated.pair,
          status: "completed",
          completedAt: updated.completedAt,
        };
        const nextList = [...(state.completedJobs || []), entry].slice(-10);
        next.set(jobId, updated);
        return {
          activeJobs: next,
          completedJobIds: nextCompleted,
          completedJobs: nextList,
          unreadCompletedCount: (state.unreadCompletedCount || 0) + 1,
        };
      }

      if (event.event === "job_failed") {
        updated.status = "failed";
        updated.error = event.error;
        updated.progressText = `Failed: ${event.error}`;
        updated.completedAt = new Date();
        const nextCompleted = _toSet(state.completedJobIds);
        nextCompleted.add(jobId);
        const entry = {
          jobId,
          models: updated.models,
          pair: updated.pair,
          status: "failed",
          completedAt: updated.completedAt,
        };
        const nextList = [...(state.completedJobs || []), entry].slice(-10);
        next.set(jobId, updated);
        return {
          activeJobs: next,
          completedJobIds: nextCompleted,
          completedJobs: nextList,
          unreadCompletedCount: (state.unreadCompletedCount || 0) + 1,
        };
      }

      next.set(jobId, updated);
      return { activeJobs: next };
    }),

  selectJob: (jobId) => set({ selectedJobId: jobId }),

  removeJob: (jobId) =>
    set((state) => {
      const next = _toMap(state.activeJobs);
      next.delete(jobId);
      const nextCompleted = _toSet(state.completedJobIds);
      nextCompleted.delete(jobId);
      _processedEventIds.delete(jobId);

      let nextSelectedId = state.selectedJobId;
      if (state.selectedJobId === jobId) {
        // Fallback: pick the most recent remaining job
        let mostRecent: { id: string; date: Date } | null = null;
        for (const [id, j] of next) {
          if (!mostRecent || j.createdAt > mostRecent.date) {
            mostRecent = { id, date: j.createdAt };
          }
        }
        nextSelectedId = mostRecent?.id ?? null;
      }

      return {
        activeJobs: next,
        completedJobIds: nextCompleted,
        selectedJobId: nextSelectedId,
      };
    }),

  getJob: (jobId) => _toMap<string, JobState>(get().activeJobs).get(jobId),

  setActiveTab: (jobId, tab) =>
    set((state) => {
      const next = _toMap(state.activeJobs);
      const job = next.get(jobId);
      if (job) {
        next.set(jobId, { ...job, activeTab: tab });
      }
      return { activeJobs: next };
    }),

  markCompleted: (jobId) =>
    set((state) => {
      if (state.completedJobIds.has(jobId)) return state;
      const nextCompleted = _toSet(state.completedJobIds);
      nextCompleted.add(jobId);
      return { completedJobIds: nextCompleted };
    }),

  clearCompletedJobs: () =>
    set((state) => {
      const next = _toMap(state.activeJobs);
      const removed: string[] = [];
      for (const [id, job] of next) {
        if (job.status === "completed" || job.status === "failed") {
          removed.push(id);
        }
      }
      for (const id of removed) next.delete(id);
      return { activeJobs: next, completedJobIds: new Set() };
    }),

  clearUnreadCount: () => set({ unreadCompletedCount: 0 }),
}),
{
  name: "kodaquant-monitor-jobs",
  storage: createJSONStorage(() => localStorage),
  serialize: _serialize,
  deserialize: _deserialize,
  partialize: (state) => ({
    activeJobs: state.activeJobs,
    completedJobIds: state.completedJobIds,
    selectedJobId: state.selectedJobId ?? null,
    completedJobs: state.completedJobs,
    unreadCompletedCount: state.unreadCompletedCount,
  }),
  onRehydrateStorage: () => (state) => {
    if (state && state.activeJobs instanceof Map) {
      state.activeJobs.forEach((job) => {
        if (job.status === "pending" || job.status === "running" || job.status === "queued") {
          job.status = "stale";
        }
      });
      state.selectedJobId = null;
    }
  },
},
),
);
